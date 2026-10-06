"""
Script tes manual: ambil snapshot sekarang dari satu channel dan kirim ke Lark.

Cara pakai:
    python tools/test_snapshot_now.py --track 101
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_settings  # noqa: E402
from face_matcher import FaceMatcher  # noqa: E402
from lark_client import LarkClient  # noqa: E402
from nvr_client import NVRClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=int, required=True, help="trackID, contoh: 101 untuk D1")
    args = parser.parse_args()

    settings = load_settings()
    nvr = NVRClient(settings.nvr_base_url, settings.nvr_username, settings.nvr_password)
    matcher = (
        FaceMatcher(
            settings.reference_dir,
            settings.face_match_tolerance,
            settings.face_min_confidence,
        )
        if settings.enable_face_matching
        else None
    )
    person_detector = (
        _create_person_detector(settings)
        if matcher and settings.enable_person_detection
        else None
    )
    lark = LarkClient(settings.lark_webhook_url, settings.lark_app_id, settings.lark_app_secret)
    label = f"D{args.track // 100}"

    print(f"Mengambil snapshot dari channel {label} (trackID {args.track})...")
    try:
        image_bytes = nvr.take_snapshot(args.track)
    except Exception as e:  # noqa: BLE001
        print(f"Gagal ambil snapshot: {e}")
        sys.exit(1)

    print(f"Snapshot berhasil diambil ({len(image_bytes)} bytes).")
    debug_path = "data/last_test_snapshot.jpg"
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    with open(debug_path, "wb") as f:
        f.write(image_bytes)
    print(f"Salinan disimpan ke: {debug_path}")

    matches = []
    recognized = []
    if matcher:
        matches = (
            person_detector.match_faces(image_bytes, matcher)
            if person_detector
            else matcher.match_faces(image_bytes)
        )
        recognized = [match for match in matches if match["name"]]
        print(f"Jumlah wajah terdeteksi: {len(matches)}")
        for index, match in enumerate(matches, start=1):
            face_label = match["name"] or "tidak dikenali"
            score = f" ({match['confidence']}%)" if match["name"] else ""
            print(f"  Wajah {index}: {face_label}{score}")
        if recognized:
            print(
                f"Hasil face matching: {recognized[0]['name']} "
                f"({recognized[0]['confidence']}%)"
            )
        else:
            print("Hasil face matching: tidak ada wajah yang dikenali (0.0%)")
    else:
        print("Face matching dinonaktifkan (ENABLE_FACE_MATCHING=false).")

    print("Mengirim ke Lark...")
    unknown_matches = [match for match in matches if not match["name"]]
    if unknown_matches:
        unknown_capture = (
            person_detector.last_annotated_capture
            if person_detector and person_detector.last_annotated_capture
            else image_bytes
        )
        for index, match in enumerate(unknown_matches, start=1):
            sent = lark.send_capture(
                unknown_capture,
                None,
                0.0,
                "",
                label,
                datetime.now().isoformat(),
                extra_images=[match["crop"]] if match.get("crop") else None,
                extra_text=f"👁️ Object person: Unknown #{index} dari {len(matches)}",
            )
            if not sent:
                print("GAGAL terkirim. Cek logs/middleware.log.")
                sys.exit(1)
        print(f"BERHASIL terkirim ke Lark: {len(unknown_matches)} object Unknown.")
    if not matches:
        print("Tidak ada object person atau wajah yang terdeteksi; tidak mengirim ke Lark.")
        return
    sent_all = True
    for match in recognized:
        name = match["name"]
        sent = lark.send_capture(
            image_bytes,
            name,
            match["confidence"],
            matcher.contact_for(name),
            label,
            datetime.now().isoformat(),
            extra_images=[
                image
                for image in (match["crop"], matcher.reference_image(name))
                if image
            ],
            extra_text=f"👁️ Wajah dikenali: {name}",
        )
        sent_all = sent_all and sent
    if sent_all:
        print(f"BERHASIL terkirim ke Lark: {len(recognized)} pesan.")
    else:
        print("Sebagian pesan gagal terkirim. Cek logs/middleware.log.")
        sys.exit(1)


def _create_person_detector(settings):
    try:
        from person_detector import PersonDetector
    except ImportError as exc:
        raise RuntimeError(
            "Deteksi person diaktifkan, tetapi ultralytics belum terpasang. "
            "Install requirements-inference.txt atau set ENABLE_PERSON_DETECTION=false."
        ) from exc
    return PersonDetector(settings.person_detector_model, settings.person_detector_confidence)


if __name__ == "__main__":
    main()
