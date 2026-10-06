"""
Script diagnosa: menjalankan query ContentMgmt/search ke NVR dan
menampilkan XML mentah hasilnya, supaya kamu bisa:
  1. Memastikan metadataDescriptor mana yang benar-benar menghasilkan
     capture (bukan 'NO MATCHES').
  2. Melihat persis nama field yang dipakai untuk item hasil, untuk
     menyesuaikan _parse_search_result() di nvr_client.py kalau perlu.

WAJIB dijalankan dan divalidasi sebelum mengandalkan main.py di
produksi, karena skema respons untuk kasus "ada hasil" belum
terverifikasi saat middleware ini pertama kali dibuat.

Cara pakai:
    python tools/test_capture_search.py --track 103 --minutes 60
    python tools/test_capture_search.py --track 103 --minutes 1440 \\
        --descriptor recordType.meta.hikvision.com/manualSnapShot
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_settings  # noqa: E402
from nvr_client import NVRClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", type=int, required=True, help="trackID, contoh: 103 untuk D1")
    parser.add_argument("--minutes", type=int, default=60, help="Cari mundur berapa menit dari sekarang")
    parser.add_argument(
        "--descriptor",
        default=None,
        help="Override NVR_PICTURE_DESCRIPTOR dari .env untuk tes ini saja",
    )
    args = parser.parse_args()

    settings = load_settings()
    nvr = NVRClient(settings.nvr_base_url, settings.nvr_username, settings.nvr_password)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=args.minutes)
    descriptor = args.descriptor or settings.nvr_picture_descriptor

    print(f"Mencari trackID={args.track} descriptor={descriptor}")
    print(f"Rentang waktu: {start_time.isoformat()} .. {end_time.isoformat()}\n")
    try:
        result = nvr.search_pictures(
            track_id=args.track,
            start_time=start_time,
            end_time=end_time,
            metadata_descriptor=descriptor,
            max_results=settings.max_results_per_poll,
        )
        print("=== Result ===")
        print(result)
    except Exception as e:  # noqa: BLE001
        print(f"Request gagal: {e}")
        sys.exit(1)

    print("=== RESPONSE XML MENTAH ===")
    print(result["raw_xml"])
    print("\n=== RINGKASAN HASIL PARSING ===")
    print(f"totalMatches (hasil parsing): {result['total_matches']}")
    print(f"Jumlah item terparsing: {len(result['items'])}")
    for i, item in enumerate(result["items"], start=1):
        print(f"\n--- item {i} ---")
        for k, v in item.items():
            print(f"  {k}: {v}")

    if result["total_matches"] > 0 and not result["items"]:
        print(
            "\n[PERHATIAN] totalMatches > 0 tapi tidak ada item yang berhasil di-parse.\n"
            "Ini artinya struktur XML hasil berbeda dari yang diasumsikan "
            "_parse_search_result() di nvr_client.py. Salin XML mentah di atas dan "
            "sesuaikan candidate_paths / nama field pada fungsi tersebut."
        )


if __name__ == "__main__":
    main()
