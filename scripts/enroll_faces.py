"""Membangun database embedding ArcFace dari foto per folder karyawan."""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EMPLOYEES_DIR = PROJECT_DIR / "employees"
DEFAULT_REFERENCE_DIR = PROJECT_DIR / "data" / "reference_photos"
STORE_DIR = PROJECT_DIR / "weights" / "faces"
VECTORS_PATH = STORE_DIR / "employees.npz"
METADATA_PATH = STORE_DIR / "employees.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def slugify(name):
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "unknown"


def main():
    parser = argparse.ArgumentParser(description="Enroll foto wajah karyawan.")
    parser.add_argument("--employees-dir", default=str(DEFAULT_EMPLOYEES_DIR))
    parser.add_argument("--min-face-px", type=int, default=32)
    args = parser.parse_args()
    employees_dir = Path(args.employees_dir).resolve()
    if not employees_dir.is_dir():
        if args.employees_dir == str(DEFAULT_EMPLOYEES_DIR) and DEFAULT_REFERENCE_DIR.is_dir():
            employees_dir = DEFAULT_REFERENCE_DIR
            print(
                f"[INFO] Folder employees tidak ditemukan; "
                f"menggunakan foto referensi: {employees_dir}"
            )
        else:
            print(f"[ERROR] Folder tidak ditemukan: {employees_dir}")
            print("Buat contoh: employees\\Budi Santoso\\foto1.jpg")
            return 1

    folders = sorted(path for path in employees_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not folders:
        print(f"[ERROR] Tidak ada subfolder karyawan di {employees_dir}")
        return 1

    providers = ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=-1, det_size=(640, 640))
    vectors, employee_ids, metadata = [], [], []

    for folder in folders:
        accepted = 0
        rejected = 0
        employee_id = slugify(folder.name)
        for photo in sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
            image = cv2.imread(str(photo))
            faces = app.get(image) if image is not None else []
            if len(faces) != 1:
                rejected += 1
                print(f"[SKIP] {photo.name}: ditemukan {len(faces)} wajah (harus tepat 1)")
                continue
            face = faces[0]
            x1, y1, x2, y2 = face.bbox
            if min(x2 - x1, y2 - y1) < args.min_face_px:
                rejected += 1
                print(f"[SKIP] {photo.name}: wajah terlalu kecil")
                continue
            vectors.append(np.asarray(face.normed_embedding, dtype=np.float32))
            employee_ids.append(employee_id)
            accepted += 1
        if accepted:
            metadata.append({
                "employee_id": employee_id,
                "employee_name": folder.name,
                "num_photos": accepted,
                "num_rejected": rejected,
                "enrolled_at": datetime.now(timezone.utc).isoformat(),
                "active": True,
            })
            print(f"[OK] {folder.name}: {accepted} foto diterima")
        else:
            print(f"[SKIP] {folder.name}: tidak ada foto valid")

    if not vectors:
        print("[ERROR] Tidak ada foto valid; store tidak ditulis.")
        return 1
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        VECTORS_PATH,
        vectors=np.stack(vectors).astype(np.float32),
        employee_ids=np.asarray(employee_ids, dtype="<U128"),
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Store: {VECTORS_PATH}")
    print(f"[DONE] Metadata: {METADATA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
