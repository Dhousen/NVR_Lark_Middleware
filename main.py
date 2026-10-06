"""
Loop utama middleware: mengambil snapshot pada jam yang dikonfigurasi,
mencocokkan wajahnya dengan foto referensi, lalu mengirim notifikasi ke
Lark.

Cara pakai:
    python main.py

Berhenti dengan Ctrl+C (akan berhenti dengan rapi, menyelesaikan siklus
yang sedang berjalan dulu sebelum benar-benar keluar).
"""
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, time as dt_time

from config import load_settings
from face_matcher import FaceMatcher
from lark_client import LarkClient
from nvr_client import NVRClient
from state_store import StateStore
from utils.logger import setup_logging

logger = logging.getLogger("main")

_shutdown_requested = False


def _handle_shutdown(signum, frame):  # noqa: ARG001
    global _shutdown_requested
    logger.info("Sinyal berhenti diterima, menyelesaikan siklus saat ini lalu keluar...")
    _shutdown_requested = True


def channel_label_for_track(track_id: int) -> str:
    """Contoh: trackID 103 -> 'D1', trackID 203 -> 'D2'."""
    return f"D{track_id // 100}"


def run() -> None:
    settings = load_settings()
    setup_logging(settings.log_dir, settings.log_level)

    logger.info("Memulai middleware NVR -> Lark")
    logger.info("Mode absensi: masuk %s-%s, pulang %s-%s",
                settings.attendance_entry_start, settings.attendance_entry_end,
                settings.attendance_exit_start, settings.attendance_exit_end)

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
    state = StateStore(settings.state_file)

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    while not _shutdown_requested:
        now = datetime.now()
        session = _attendance_session(now, settings)
        if session:
            for track_id in settings.nvr_track_ids:
                try:
                    _snapshot_channel(
                        nvr, matcher, person_detector, lark, state, settings,
                        track_id, now, session,
                    )
                except Exception:
                    logger.error("Error channel %s pada sesi %s.", track_id, session, exc_info=True)
            _sleep_interruptible(settings.attendance_poll_interval_seconds)
        else:
            _sleep_interruptible(5)

    logger.info("Middleware berhenti dengan rapi.")


def _sleep_interruptible(seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end and not _shutdown_requested:
        time.sleep(min(1, max(0.0, end - time.time())))


def _create_person_detector(settings):
    try:
        from person_detector import PersonDetector
    except ImportError as exc:
        raise RuntimeError(
            "Deteksi person diaktifkan, tetapi ultralytics belum terpasang. "
            "Install requirements-inference.txt atau set ENABLE_PERSON_DETECTION=false."
        ) from exc
    return PersonDetector(settings.person_detector_model, settings.person_detector_confidence)


def _parse_hhmm(value):
    hour, minute = (int(part) for part in value.split(":"))
    return dt_time(hour, minute)


def _attendance_session(now, settings):
    current = now.time()
    if _parse_hhmm(settings.attendance_entry_start) <= current < _parse_hhmm(settings.attendance_entry_end):
        return "entry"
    if _parse_hhmm(settings.attendance_exit_start) <= current < _parse_hhmm(settings.attendance_exit_end):
        return "exit"
    return None


def _snapshot_channel(nvr, matcher, person_detector, lark, state, settings, track_id, captured_at, session) -> None:
    label = channel_label_for_track(track_id)
    try:
        image_bytes = nvr.take_snapshot(track_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("Snapshot gagal untuk channel %s (trackID %s): %s", label, track_id, e)
        return

    if matcher:
        matches = (
            person_detector.match_faces(image_bytes, matcher)
            if person_detector
            else matcher.match_faces(image_bytes)
        )
        recognized = [match for match in matches if match["name"]]
    else:
        matches = []
        recognized = []

    unknown_matches = [match for match in matches if not match["name"]]
    if unknown_matches:
        unknown_capture = (
            person_detector.last_annotated_capture
            if person_detector and person_detector.last_annotated_capture
            else image_bytes
        )
        for index, match in enumerate(unknown_matches, start=1):
            unknown_key = f"unknown:{captured_at:%Y-%m-%d}:{session}:{track_id}:{index}"
            if not state.claim_notification(
                unknown_key, settings.unknown_person_cooldown_seconds
            ):
                continue
            sent = lark.send_capture(
                unknown_capture,
                None,
                0.0,
                "",
                label,
                captured_at.isoformat(),
                extra_images=[match["crop"]] if match["crop"] else None,
                extra_text=f"👁️ Object person: Unknown #{index} dari {len(matches)} ({session})",
            )
            if sent:
                logger.info("Snapshot %s: wajah tidak dikenali #%d terkirim.", label, index)
            else:
                state.release_notification(unknown_key)
    if not matcher:
        lark.send_capture(
            image_bytes,
            None,
            0.0,
            "",
            label,
            captured_at.isoformat(),
            extra_text="⚠️ Deteksi wajah dinonaktifkan",
        )
        return
    if not recognized:
        return

    for match in recognized:
        name = match["name"]
        confidence = match["confidence"]
        person_key = name
        if not state.claim_attendance(captured_at.strftime("%Y-%m-%d"), session, person_key):
            logger.info("Absensi %s %s sudah tercatat pada sesi %s.", label, person_key, session)
            continue

        nik = matcher.contact_for(name) if matcher else ""
        notification_images = [
            image
            for image in (match["crop"], matcher.reference_image(name))
            if image
        ]
        sent = lark.send_capture(
            image_bytes,
            name,
            confidence,
            nik,
            label,
            captured_at.isoformat(),
            extra_images=notification_images,
            extra_text=f"👁️ Wajah dikenali: {name}",
        )
        if sent:
            logger.info(
                "Snapshot %s terkirim: %s (%.1f%%).", label, name, confidence
            )
        else:
            state.release_attendance(
                captured_at.strftime("%Y-%m-%d"), session, person_key
            )


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        logging.getLogger("main").critical("Error fatal, middleware berhenti: %s", e, exc_info=True)
        sys.exit(1)
