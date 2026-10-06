"""
Loader konfigurasi untuk middleware NVR -> Lark.

Semua konfigurasi diambil dari environment variable (lihat .env.example).
Kalau ada nilai wajib yang belum diisi, program akan berhenti dengan
pesan error yang jelas alih-alih crash dengan traceback membingungkan.
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

# Paksa baca file .env yang PERSIS ada di folder yang sama dengan config.py
# ini (bukan hasil pencarian otomatis ke folder induk), supaya tidak
# tertukar kalau ada file .env lain di folder di atasnya (misal akibat
# struktur folder ZIP yang ter-extract bertingkat).
_ENV_PATH = Path(__file__).resolve().parent / ".env"
_PROJECT_DIR = _ENV_PATH.parent
load_dotenv(dotenv_path=_ENV_PATH, override=True)


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"[CONFIG ERROR] Environment variable '{name}' wajib diisi tapi kosong.")
        print("Salin .env.example menjadi .env lalu isi nilainya sebelum menjalankan program.")
        sys.exit(1)
    return value


def _optional(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _parse_track_ids(raw: str) -> List[int]:
    try:
        return [int(t.strip()) for t in raw.split(",") if t.strip()]
    except ValueError:
        print("[CONFIG ERROR] NVR_TRACK_IDS harus berupa angka dipisah koma, contoh: 103,203,303")
        sys.exit(1)


@dataclass(frozen=True)
class Settings:
    # --- NVR ---
    nvr_base_url: str
    nvr_username: str
    nvr_password: str
    nvr_track_ids: List[int] = field(default_factory=list)
    nvr_picture_descriptor: str = "recordType.meta.hikvision.com/allPic"
    nvr_fdid: str = ""

    # --- Lark ---
    lark_webhook_url: str = ""
    lark_app_id: str = ""
    lark_app_secret: str = ""

    # --- Perilaku ---
    per_person_cooldown_seconds: int = 0
    unknown_person_cooldown_seconds: int = 900
    attendance_poll_interval_seconds: int = 60
    attendance_entry_start: str = "07:00"
    attendance_entry_end: str = "10:00"
    attendance_exit_start: str = "15:00"
    attendance_exit_end: str = "23:00"
    schedule_late_tolerance_seconds: int = 120
    face_match_tolerance: float = 0.677
    face_min_confidence: float = 30
    enable_face_matching: bool = True
    enable_person_detection: bool = True
    person_detector_model: str = "yolov8n.pt"
    person_detector_confidence: float = 0.35
    snapshot_times: Tuple[str, ...] = (
        "10:00:00", "10:00:10", "10:00:20",
        "12:00:00", "12:00:10", "12:00:20",
        "15:00:00", "15:00:10", "15:00:20",
        "17:00:00", "17:00:10", "17:00:20",
    )

    # --- Lokasi file ---
    state_file: str = "data/state.json"
    reference_dir: str = "data/reference_photos"
    log_dir: str = "logs"
    log_level: str = "INFO"


def load_settings() -> Settings:
    snapshot_times = tuple(
        value.strip()
        for value in _optional(
            "SNAPSHOT_TIMES",
            "10:00:00,10:00:10,10:00:20,12:00:00,12:00:10,12:00:20,"
            "15:00:00,15:00:10,15:00:20,17:00:00,17:00:10,17:00:20",
        ).split(",")
        if value.strip()
    )
    for value in snapshot_times:
        try:
            hour, minute, second = (int(part) for part in value.split(":"))
            if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
                raise ValueError
        except ValueError:
            print(
                "[CONFIG ERROR] SNAPSHOT_TIMES harus berformat HH:MM:SS dipisah koma, "
                "contoh: 10:00:00,10:00:10"
            )
            sys.exit(1)

    def project_path(raw: str) -> str:
        path = Path(raw)
        return str(path if path.is_absolute() else _PROJECT_DIR / path)

    return Settings(
        nvr_base_url=_require("NVR_BASE_URL").rstrip("/"),
        nvr_username=_require("NVR_USERNAME"),
        nvr_password=_require("NVR_PASSWORD"),
        nvr_track_ids=_parse_track_ids(_require("NVR_TRACK_IDS")),
        nvr_picture_descriptor=_optional(
            "NVR_PICTURE_DESCRIPTOR", "recordType.meta.hikvision.com/allPic"
        ),
        nvr_fdid=_optional("NVR_FDID", ""),
        lark_webhook_url=_require("LARK_WEBHOOK_URL"),
        lark_app_id=_optional("LARK_APP_ID", ""),
        lark_app_secret=_optional("LARK_APP_SECRET", ""),
        per_person_cooldown_seconds=int(_optional("PER_PERSON_COOLDOWN_SECONDS", "0")),
        unknown_person_cooldown_seconds=int(
            _optional("UNKNOWN_PERSON_COOLDOWN_SECONDS", "900")
        ),
        attendance_poll_interval_seconds=int(
            _optional("ATTENDANCE_POLL_INTERVAL_SECONDS", "60")
        ),
        attendance_entry_start=_optional("ATTENDANCE_ENTRY_START", "06:00"),
        attendance_entry_end=_optional("ATTENDANCE_ENTRY_END", "10:00"),
        attendance_exit_start=_optional("ATTENDANCE_EXIT_START", "15:00"),
        attendance_exit_end=_optional("ATTENDANCE_EXIT_END", "23:00"),
        schedule_late_tolerance_seconds=int(
            _optional("SCHEDULE_LATE_TOLERANCE_SECONDS", "120")
        ),
        face_match_tolerance=float(_optional("FACE_MATCH_TOLERANCE", "0.677")),
        face_min_confidence=float(_optional("FACE_MIN_CONFIDENCE", "33.3")),
        enable_face_matching=_optional("ENABLE_FACE_MATCHING", "true").lower() == "true",
        enable_person_detection=_optional("ENABLE_PERSON_DETECTION", "true").lower() == "true",
        person_detector_model=_optional("PERSON_DETECTOR_MODEL", "yolov8n.pt"),
        person_detector_confidence=float(_optional("PERSON_DETECTOR_CONFIDENCE", "0.35")),
        snapshot_times=snapshot_times,
        state_file=project_path(_optional("STATE_FILE", "data/state.json")),
        reference_dir=project_path(_optional("REFERENCE_DIR", "data/reference_photos")),
        log_dir=project_path(_optional("LOG_DIR", "logs")),
        log_level=_optional("LOG_LEVEL", "INFO"),
    )