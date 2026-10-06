"""Notifier teks minimal untuk pipeline inference berbasis track."""
import logging
import requests

from config import load_settings

logger = logging.getLogger(__name__)


def send_face_match_notification(
    employee_id, employee_name, score, person_id, frame_count, frame_timestamp
):
    settings = load_settings()
    text = (
        f"👤 {employee_name}\n"
        f"🆔 Employee ID: {employee_id}\n"
        f"✅ Score: {score:.3f}\n"
        f"🎥 Track ID: {person_id}\n"
        f"🎞️ Frame: {frame_count}\n"
        f"🕒 Waktu: {frame_timestamp}"
    )
    response = requests.post(
        settings.lark_webhook_url,
        json={"msg_type": "text", "content": {"text": text}},
        timeout=15,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or result.get("code") != 0:
        raise RuntimeError(f"Webhook Lark menolak notifikasi: {result}")
