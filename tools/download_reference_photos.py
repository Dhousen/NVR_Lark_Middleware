"""
Utilitas sekali-jalan: mengunduh semua foto wajah terdaftar dari Face
Library NVR (FDLib) dan mengelompokkannya ke
data/reference_photos/<Nama>/photo_001.jpg, supaya beberapa entri dengan
nama sama tidak saling menimpa.

Jalankan sekali di awal setup, dan jalankan lagi setiap kali ada orang
yang ditambah/dihapus dari Face Library di NVR. Setelah itu hapus
data/encodings.pkl (kalau ada) atau cukup restart middleware supaya
encoding dibangun ulang dari foto-foto terbaru.

Cara pakai:
    python tools/download_reference_photos.py
"""
import os
import json
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_settings  # noqa: E402
from nvr_client import NVRClient  # noqa: E402


def migrate_flat_photos(reference_dir: str) -> None:
    """Pindahkan foto format lama <Nama>.jpg ke folder <Nama>/."""
    for filename in os.listdir(reference_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        source = os.path.join(reference_dir, filename)
        if not os.path.isfile(source):
            continue
        safe_name, extension = os.path.splitext(filename)
        person_dir = os.path.join(reference_dir, safe_name)
        os.makedirs(person_dir, exist_ok=True)
        target = os.path.join(person_dir, "photo_001" + extension.lower())
        if os.path.exists(target):
            continue
        shutil.move(source, target)
        print(f"  ! Migrasi foto lama: {source} -> {target}")


def canonical_first_name(name: str) -> str:
    normalized = re.sub(r"[_-]+", " ", name).strip()
    first = normalized.split()[0] if normalized else "Unknown"
    return first[:1].upper() + first[1:]


def main() -> None:
    settings = load_settings()
    if not settings.nvr_fdid:
        print(
            "NVR_FDID belum diisi di .env - cari nilainya lewat "
            "GET /ISAPI/Intelligent/FDLib lalu isi NVR_FDID sebelum "
            "menjalankan script ini."
        )
        sys.exit(1)

    nvr = NVRClient(settings.nvr_base_url, settings.nvr_username, settings.nvr_password)

    print(f"Mengambil daftar wajah dari Face Library (FDID: {settings.nvr_fdid})...\n")
    try:
        people = nvr.search_face_library(settings.nvr_fdid)
    except Exception as e:  # noqa: BLE001
        print(f"Gagal mengambil daftar Face Library: {e}")
        sys.exit(1)

    if not people:
        print("Tidak ada orang yang terdaftar di Face Library ini.")
        return

    os.makedirs(settings.reference_dir, exist_ok=True)
    migrate_flat_photos(settings.reference_dir)
    count = 0
    contacts = {}
    name_counts = {}
    for person in people:
        name = person.get("name", "").strip()
        pic_url = person.get("pic_url", "").strip()
        if not name or not pic_url:
            print(f"  ! Melewati entri tanpa nama/foto: {person}")
            continue

        try:
            image_bytes = nvr.download_picture(pic_url)
        except Exception as e:  # noqa: BLE001
            print(f"  ! Gagal unduh foto untuk {name}: {e}")
            continue

        safe_name = canonical_first_name(name)
        person_dir = os.path.join(settings.reference_dir, safe_name)
        os.makedirs(person_dir, exist_ok=True)
        name_counts[safe_name] = name_counts.get(safe_name, 0) + 1
        out_path = os.path.join(person_dir, f"photo_{name_counts[safe_name]:03d}.jpg")
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        print(f"  \u2713 Tersimpan: {out_path}")
        contacts[safe_name] = person.get("contact", "").strip()
        count += 1

    contacts_path = os.path.join(settings.reference_dir, "contacts.json")
    with open(contacts_path, "w", encoding="utf-8") as f:
        json.dump(contacts, f, ensure_ascii=False, indent=2)
    print(f"Metadata contact tersimpan: {contacts_path}")

    print(f"\nSelesai. {count} foto referensi tersimpan di {settings.reference_dir}")
    encodings_cache = os.path.join(
        os.path.dirname(settings.reference_dir.rstrip("/\\")) or ".", "encodings.pkl"
    )
    if os.path.exists(encodings_cache):
        os.remove(encodings_cache)
        print("Cache encoding lama dihapus, akan dibangun ulang saat middleware dijalankan.")
    print("Sekarang jalankan `python main.py` untuk mulai memantau.")


if __name__ == "__main__":
    main()
