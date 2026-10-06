"""Identifikasi wajah ArcFace dari crop orang dan store NumPy."""
import logging
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceMatch:
    employee_id: str
    employee_name: str
    score: float
    margin: float


class FaceIdentifier:
    def __init__(self, store_path, min_score=0.30, min_margin=0.10, min_face_px=17, device="cpu"):
        self.min_score = min_score
        self.min_margin = min_margin
        self.min_face_px = min_face_px
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
        self.employee_ids, self.employee_names, self.embeddings = self._load_store(store_path)

    @staticmethod
    def _load_store(store_path):
        if not os.path.isfile(store_path):
            raise FileNotFoundError(f"Face store tidak ditemukan: {store_path}")
        data = np.load(store_path, allow_pickle=True)
        if "embeddings" in data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        elif "vectors" in data:
            embeddings = np.asarray(data["vectors"], dtype=np.float32)
        else:
            raise ValueError("employees.npz harus memiliki key 'embeddings' atau 'vectors'.")
        ids = [str(value) for value in data["employee_ids"].tolist()]
        names_key = "employee_names" in data
        names = [str(value) for value in data["employee_names"].tolist()] if names_key else []
        if not names_key:
            metadata_path = os.path.splitext(store_path)[0] + ".json"
            if not os.path.isfile(metadata_path):
                raise FileNotFoundError(
                    f"Metadata nama tidak ditemukan: {metadata_path}. "
                    "Jalankan scripts/enroll_faces.py terlebih dahulu."
                )
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            names_by_id = {str(item["employee_id"]): str(item["employee_name"]) for item in metadata}
            names = [names_by_id.get(employee_id, employee_id) for employee_id in ids]
        if len(embeddings) != len(ids) or len(ids) != len(names) or not len(ids):
            raise ValueError("Data face store employee dan embedding tidak seimbang.")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return ids, names, embeddings / np.maximum(norms, 1e-12)

    def identify(self, crop):
        if crop is None or crop.size == 0:
            return None
        faces = self.app.get(crop)
        faces = [face for face in faces if min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]) >= self.min_face_px]
        if not faces:
            return None
        face = max(faces, key=lambda item: item.det_score)
        embedding = np.asarray(face.embedding, dtype=np.float32)
        embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
        scores = self.embeddings @ embedding
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        second = float(scores[order[1]]) if len(order) > 1 else 0.0
        score = float(scores[best])
        margin = score - second
        if score < self.min_score or margin < self.min_margin:
            return None
        return FaceMatch(self.employee_ids[best], self.employee_names[best], score, margin)
