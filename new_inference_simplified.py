"""Pipeline opsional: frame JPEG -> YOLO/ByteTrack -> Face ID -> Lark.

Pipeline ini terpisah dari main.py. Jalankan setelah memasang ultralytics dan
torch serta menyediakan weights/faces/employees.npz.
"""
import csv
import glob
import os
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import torch
from ultralytics import YOLO

from inference.face_identity import FaceIdentifier
from inference.lark_notifier import send_face_match_notification

INPUT_PATH = "frames"
OUTPUT_PATH = "output"
PROJECT_DIR = Path(__file__).resolve().parent
FRAME_PER_SECOND = float(os.getenv("FRAME_PER_SECOND", "1"))
GRACE_PERIOD_FRAMES = 30
FACE_ID_RETRY_EVERY_FRAMES = 15
FACE_ID_MAX_ATTEMPTS = 8
FACE_MIN_SCORE = 0.30
FACE_MIN_MARGIN = 0.10
FACE_MIN_FACE_PX = 17
VIDEO_TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M:%S"


def compute_frame_timestamp(base_dt, frame_count, fps):
    if base_dt is None:
        return None
    return (base_dt + timedelta(seconds=(frame_count - 1) / fps)).strftime(VIDEO_TIMESTAMP_FORMAT)


class InferenceSession:
    def __init__(self, output_dir, face_identifier, start_dt=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.detection_log_path = os.path.join(output_dir, f"detection_log_{stamp}.csv")
        self.csv_file = None
        self.writer = None
        self.video_base_dt = start_dt
        self.face_identifier = face_identifier
        self.track_last_seen = {}
        self.track_employee = {}
        self.track_face_attempts = {}
        self.track_last_face_try = {}
        self.notified_tracks = set()
        self.frame_count = 0
        self.results_log = []

    def _resolve_employee_id(self, track_id, crop):
        cached = self.track_employee.get(track_id)
        if cached is not None:
            return cached if cached != "UNKNOWN" else None
        attempts = self.track_face_attempts.get(track_id, 0)
        if attempts >= FACE_ID_MAX_ATTEMPTS:
            self.track_employee[track_id] = "UNKNOWN"
            return None
        last = self.track_last_face_try.get(track_id, -FACE_ID_RETRY_EVERY_FRAMES)
        if self.frame_count - last < FACE_ID_RETRY_EVERY_FRAMES:
            return None
        self.track_last_face_try[track_id] = self.frame_count
        self.track_face_attempts[track_id] = attempts + 1
        match = self.face_identifier.identify(crop)
        if match is None:
            if self.track_face_attempts[track_id] >= FACE_ID_MAX_ATTEMPTS:
                self.track_employee[track_id] = "UNKNOWN"
            return None
        result = {"employee_id": match.employee_id, "employee_name": match.employee_name, "score": match.score}
        self.track_employee[track_id] = result
        return result

    def process_frame(self, frame_path):
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Gagal membaca frame: {frame_path}")
        if self.csv_file is None:
            self.csv_file = open(self.detection_log_path, "w", newline="", encoding="utf-8")
            fields = ["frame", "frame_file", "track_id", "x1", "y1", "x2", "y2", "frame_timestamp",
                      "employee_id", "employee_name", "employee_score"]
            self.writer = csv.DictWriter(self.csv_file, fieldnames=fields)
            self.writer.writeheader()
        self.frame_count += 1
        for tid in list(self.track_last_seen):
            if self.frame_count - self.track_last_seen[tid] > GRACE_PERIOD_FRAMES:
                for state in (self.track_last_seen, self.track_employee, self.track_face_attempts, self.track_last_face_try):
                    state.pop(tid, None)
                self.notified_tracks.discard(tid)
        model = getattr(self, "_yolo", None)
        if model is None:
            model = self._yolo = YOLO("yolov8l-worldv2.pt")
            model.set_classes(["person"])
        result = model.track(frame, conf=0.35, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        if result.boxes.id is None:
            self.csv_file.flush()
            return {"frame_count": self.frame_count, "detections": 0}
        for box, track_id in zip(result.boxes.xyxy.cpu().numpy().astype(int), result.boxes.id.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = map(int, box)
            self.track_last_seen[int(track_id)] = self.frame_count
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            employee = self._resolve_employee_id(int(track_id), crop)
            entry = {"frame": self.frame_count, "frame_file": os.path.basename(frame_path), "track_id": int(track_id),
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "frame_timestamp": compute_frame_timestamp(self.video_base_dt, self.frame_count, FRAME_PER_SECOND) or "-",
                     "employee_id": employee["employee_id"] if employee else "-", "employee_name": employee["employee_name"] if employee else "-",
                     "employee_score": f"{employee['score']:.3f}" if employee else "-"}
            self.writer.writerow(entry)
            self.results_log.append(entry)
            if employee and int(track_id) not in self.notified_tracks:
                self.notified_tracks.add(int(track_id))
                send_face_match_notification(employee["employee_id"], employee["employee_name"], employee["score"],
                                             int(track_id), self.frame_count, entry["frame_timestamp"])
        self.csv_file.flush()
        return {"frame_count": self.frame_count, "detections": len(result.boxes)}

    def finalize(self):
        if self.csv_file:
            self.csv_file.close()
        return {"detection_log_path": self.detection_log_path, "Total_detections": len(self.results_log),
                "Output_dir": self.output_dir}


def run_inference(frame_folder=INPUT_PATH, output_dir=OUTPUT_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    store_path = PROJECT_DIR / "weights" / "faces" / "employees.npz"
    identifier = FaceIdentifier(str(store_path), FACE_MIN_SCORE, FACE_MIN_MARGIN, FACE_MIN_FACE_PX, device)
    frame_dir = Path(frame_folder)
    if not frame_dir.is_absolute():
        frame_dir = PROJECT_DIR / frame_dir
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_absolute():
        output_dir_path = PROJECT_DIR / output_dir_path
    frame_paths = sorted(
        path for extension in ("*.jpg", "*.jpeg", "*.png")
        for path in frame_dir.glob(extension)
    )
    if not frame_paths:
        raise FileNotFoundError(
            f"Tidak ada frame gambar di {frame_dir}. "
            "Letakkan file .jpg/.jpeg/.png di folder frames atau gunakan "
            "run_inference(frame_folder='folder_anda')."
        )
    session = InferenceSession(str(output_dir_path), identifier)
    for frame_path in frame_paths:
        session.process_frame(str(frame_path))
    return session.finalize()


if __name__ == "__main__":
    print(run_inference())
