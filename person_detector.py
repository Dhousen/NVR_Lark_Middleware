"""Tahap deteksi manusia sebelum FaceMatcher existing."""
import cv2
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.35):
        self.confidence = confidence
        self.model = YOLO(model_path)
        self.last_annotated_capture = None

    def match_faces(self, image_bytes, face_matcher):
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("data snapshot bukan gambar yang valid")
        result = self.model.predict(
            source=image,
            conf=self.confidence,
            classes=[0],
            verbose=False,
        )[0]
        if result.boxes is None:
            self.last_annotated_capture = image_bytes
            return []

        matches = []
        annotated = image.copy()
        height, width = image.shape[:2]
        confidences = result.boxes.conf.cpu().numpy()
        for box, person_confidence in zip(
            result.boxes.xyxy.cpu().numpy(), confidences
        ):
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(
                annotated,
                "Unknown",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )
            success, encoded = cv2.imencode(".jpg", image[y1:y2, x1:x2])
            if not success:
                continue
            crop_bytes = encoded.tobytes()
            face_matches = face_matcher.match_faces(crop_bytes)
            if face_matches:
                matches.extend(face_matches)
            else:
                # YOLO tetap memberi hasil object detection ketika wajah
                # terlalu kecil, tertutup, atau tidak menghadap kamera.
                matches.append(
                    {
                        "name": None,
                        "confidence": 0.0,
                        "crop": crop_bytes,
                        "reference": None,
                        "person_confidence": round(float(person_confidence), 3),
                    }
                )
        success, annotated_bytes = cv2.imencode(".jpg", annotated)
        self.last_annotated_capture = (
            annotated_bytes.tobytes() if success else image_bytes
        )
        return matches
