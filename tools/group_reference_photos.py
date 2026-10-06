"""Gabungkan folder foto referensi berdasarkan nama depan."""
import json
import re
import shutil
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_DIR / "data" / "reference_photos"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def canonical_first_name(name):
    normalized = re.sub(r"[_-]+", " ", name).strip()
    first = normalized.split()[0] if normalized else "Unknown"
    return first[:1].upper() + first[1:]


def next_photo_path(folder, suffix):
    index = 1
    while True:
        path = folder / f"photo_{index:03d}{suffix.lower()}"
        if not path.exists():
            return path
        index += 1


def main():
    if not REFERENCE_DIR.is_dir():
        print(f"[ERROR] Folder tidak ditemukan: {REFERENCE_DIR}")
        return 1
    contacts_path = REFERENCE_DIR / "contacts.json"
    source_contacts = {}
    if contacts_path.exists():
        try:
            source_contacts = json.loads(contacts_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[ERROR] contacts.json tidak valid: {exc}")
            return 1

    grouped_contacts = {}
    folders = sorted(path for path in REFERENCE_DIR.iterdir() if path.is_dir())
    for source_folder in folders:
        target_name = canonical_first_name(source_folder.name)
        target_folder = REFERENCE_DIR / target_name
        target_folder.mkdir(exist_ok=True)
        moved = 0
        if source_folder.resolve() == target_folder.resolve():
            contact = str(source_contacts.get(source_folder.name, "") or "").strip()
            if contact:
                grouped_contacts[source_folder.name] = contact
            print(f"[OK] {source_folder.name}: sudah canonical")
            continue
        for photo in sorted(source_folder.rglob("*")):
            if not photo.is_file() or photo.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            target = next_photo_path(target_folder, photo.suffix)
            if photo.resolve() != target.resolve():
                shutil.move(str(photo), str(target))
                moved += 1
        contact = str(source_contacts.get(source_folder.name, "") or "").strip()
        if contact and target_name not in grouped_contacts:
            grouped_contacts[target_name] = contact
        if not any(source_folder.iterdir()):
            source_folder.rmdir()
        print(f"[OK] {source_folder.name} -> {target_name}: {moved} foto")

    for key, value in source_contacts.items():
        target_name = canonical_first_name(key)
        if value and target_name not in grouped_contacts:
            grouped_contacts[target_name] = value
    contacts_path.write_text(
        json.dumps(grouped_contacts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] Foto telah dikelompokkan di {REFERENCE_DIR}")
    print(f"[DONE] Metadata contact diperbarui: {contacts_path}")
    print("[INFO] Restart middleware agar cache encoding dibangun ulang.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
