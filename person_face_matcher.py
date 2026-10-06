"""Deteksi manusia dengan YOLO lalu cocokkan wajahnya dengan Face Library lokal.

Modul ini hanya melakukan analisis gambar. Pengiriman notifikasi tetap menjadi
tanggung jawab pipeline utama sehingga tidak menimbulkan pesan duplikat.
"""
import argparse
import io
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import load_settings
from face_matcher import FaceMatcher

logger = logging.getLogger(__name__)


class PersonFaceMatcher:
    def __init__(
        self,
        reference_dir,
        tolerance=0.5,
        min_confidence=50.0,
        model_path="yolov8n.pt",
        person_confidence=0.35,
        cache_path=None,
    ):
        self.person_confidence = person_confidence
        self.detector = YOLO(model_path)
        self.face_matcher = FaceMatcher(
            reference_dir=reference_dir,
            tolerance=tolerance,
            min_confidence=min_confidence,
            cache_path=cache_path,
        )

    @staticmethod
    def _encode_crop(image, bbox):
        x1, y1, x2, y2 = bbox
        height, width = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        success, encoded = cv2.imencode(".jpg", image[y1:y2, x1:x2])
        if not success:
            raise ValueError("Gagal mengubah crop manusia menjadi JPEG.")
        return encoded.tobytes()

    def analyze(self, image_bytes):
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Input bukan gambar yang valid.")

        detection = self.detector.predict(
            source=image,
            conf=self.person_confidence,
            classes=[0],
            verbose=False,
        )[0]
        people = []
        if detection.boxes is None:
            return people

        for index, (box, confidence) in enumerate(
            zip(detection.boxes.xyxy.cpu().numpy(), detection.boxes.conf.cpu().numpy()),
            start=1,
        ):
            bbox = tuple(int(round(value)) for value in box)
            crop_bytes = self._encode_crop(image, bbox)
            if crop_bytes is None:
                continue
            face_matches = self.face_matcher.match_faces(crop_bytes)
            if face_matches:
                for face_index, match in enumerate(face_matches, start=1):
                    people.append({
                        "person_index": index,
                        "face_index": face_index,
                        "person_confidence": round(float(confidence), 3),
                        "bbox": bbox,
                        "name": match["name"],
                        "face_confidence": match["confidence"],
                        "reference": match["reference"],
                        "contact": self.face_matcher.contact_for(match["name"]),
                        "crop": match["crop"],
                        "reference_image": self.face_matcher.reference_image(match["name"]),
                    })
            else:
                people.append({
                    "person_index": index,
                    "face_index": None,
                    "person_confidence": round(float(confidence), 3),
                    "bbox": bbox,
                    "name": None,
                    "face_confidence": 0.0,
                    "reference": None,
                    "contact": "",
                    "crop": crop_bytes,
                    "reference_image": None,
                })
        return people

    @staticmethod
    def annotate(image_bytes, results):
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Input bukan gambar yang valid.")
        drawn = set()
        for result in results:
            bbox = result["bbox"]
            if bbox not in drawn:
                x1, y1, x2, y2 = bbox
                color = (0, 200, 0) if result["name"] else (0, 165, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = result["name"] or "Tidak dikenali"
                cv2.putText(image, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                drawn.add(bbox)
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Gagal membuat gambar hasil anotasi.")
        return encoded.tobytes()


def _json_results(results):
    return [
        {
            key: value
            for key, value in result.items()
            if key not in {"crop", "reference_image"}
        }
        for result in results
    ]


def main():
    parser = argparse.ArgumentParser(description="Deteksi manusia dan pencocokan wajah Face Library.")
    parser.add_argument("--image", required=True, help="Path gambar input.")
    parser.add_argument("--output", help="Path gambar hasil anotasi.")
    parser.add_argument("--json-output", help="Path hasil JSON.")
    parser.add_argument("--model", default="yolov8n.pt", help="Model YOLO person detector.")
    parser.add_argument("--person-confidence", type=float, default=0.35)
    args = parser.parse_args()

    settings = load_settings()
    matcher = PersonFaceMatcher(
        reference_dir=settings.reference_dir,
        tolerance=settings.face_match_tolerance,
        min_confidence=settings.face_min_confidence,
        model_path=args.model,
        person_confidence=args.person_confidence,
    )
    image_path = Path(args.image).resolve()
    image_bytes = image_path.read_bytes()
    results = matcher.analyze(image_bytes)

    print(json.dumps(_json_results(results), ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).resolve().write_bytes(matcher.annotate(image_bytes, results))
    if args.json_output:
        Path(args.json_output).resolve().write_text(
            json.dumps(_json_results(results), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"Manusia terdeteksi: {len({item['person_index'] for item in results})}")
    print(f"Wajah dianalisis: {len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
