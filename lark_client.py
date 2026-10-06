"""
Client untuk Lark: upload gambar dan kirim pesan lewat Custom Bot Webhook.
"""
import logging
import time
import io

import requests
from PIL import Image, ImageDraw, ImageOps

logger = logging.getLogger(__name__)

TOKEN_URL = "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal"
IMAGE_UPLOAD_URL = "https://open.larksuite.com/open-apis/im/v1/images"


class LarkClient:
    def __init__(self, webhook_url: str, app_id: str = "", app_secret: str = "", timeout: int = 15):
        self.webhook_url = webhook_url
        self.app_id = app_id
        self.app_secret = app_secret
        self.timeout = timeout
        self._token = None
        self._token_expires_at = 0.0

    def _get_token(self):
        if not self.app_id or not self.app_secret:
            return None
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token

        try:
            resp = requests.post(
                TOKEN_URL,
                json={"app_id": self.app_id, "app_secret": self.app_secret},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                logger.error("Permintaan token Lark gagal: %s", data)
                return None
            self._token = data["tenant_access_token"]
            self._token_expires_at = time.time() + data.get("expire", 7200)
            return self._token
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            logger.error("Tidak bisa menghubungi endpoint token Lark: %s", e)
            return None

    def _upload_image(self, image_bytes: bytes):
        token = self._get_token()
        if not token:
            return None
        try:
            resp = requests.post(
                IMAGE_UPLOAD_URL,
                headers={"Authorization": "Bearer " + token},
                data={"image_type": "message"},
                files={"image": ("capture.jpg", image_bytes, "image/jpeg")},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict) or data.get("code") != 0:
                logger.error("Upload gambar ke Lark gagal: %s", data)
                return None
            image_key = data.get("data", {}).get("image_key")
            if not isinstance(image_key, str) or not image_key:
                logger.error("Respons upload Lark tidak memiliki data.image_key: %s", data)
                return None
            return image_key
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            detail = ""
            if e.response is not None:
                detail = f" | Respons Lark: {e.response.text.strip()}"
            logger.error("Tidak bisa upload gambar ke Lark: %s%s", e, detail)
            return None

    def send_capture(
        self,
        image_bytes: bytes,
        name,
        confidence: float,
        nik: str,
        channel_label: str,
        timestamp_str: str,
        extra_images=None,
        extra_text: str = "",
        max_retries: int = 3,
    ) -> bool:
        image_keys = []
        if self.app_id and self.app_secret:
            gallery = self._compose_gallery([image_bytes] + list(extra_images or []))
            image_key = self._upload_image(gallery)
            if image_key:
                image_keys.append(image_key)
            if not image_keys:
                logger.warning("Foto tidak ikut dikirim; melanjutkan dengan notifikasi teks.")

        who = name if name else "Wajah tidak dikenali"
        conf_text = f" ({confidence}%)" if name else ""
        nik_text = f"\n🪪 NIK: {nik}" if nik else ""
        text = f"👤 {who}{conf_text}{nik_text}"
        if extra_text:
            text += f"\n{extra_text}"
        text += f"\n📍 Kamera: {channel_label}\n🕒 Waktu: {timestamp_str}"

        if image_keys:
            content = [[{"tag": "text", "text": text}]]
            content.extend([[{"tag": "img", "image_key": key}] for key in image_keys])
            payload = {
                "msg_type": "post",
                "content": {
                    "post": {
                        "en_us": {
                            "title": "Face Detected",
                            "content": content,
                        }
                    }
                },
            }
        else:
            payload = {"msg_type": "text", "content": {"text": text}}

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(self.webhook_url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                result = resp.json()
                if isinstance(result, dict) and result.get("code") == 0:
                    logger.info("Notifikasi Lark terkirim untuk: %s", who)
                    return True
                logger.warning("Webhook Lark merespon error: %s", result)
            except (requests.RequestException, ValueError, TypeError) as e:
                logger.warning(
                    "Percobaan kirim ke Lark %d/%d gagal: %s", attempt, max_retries, e
                )
            time.sleep(min(2 ** attempt, 10))

        logger.error(
            "Menyerah mengirim notifikasi untuk %s setelah %d percobaan.",
            who,
            max_retries,
        )
        return False

    @staticmethod
    def _compose_gallery(images):
        """Susun capture dan detail wajah dalam panel dengan ukuran seimbang."""
        decoded = []
        for image_bytes in images:
            try:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    decoded.append(image.convert("RGB"))
            except (OSError, ValueError):
                continue

        if not decoded:
            return images[0]

        canvas_width = 1200
        margin = 20
        gap = 12
        label_height = 34
        border = 4
        background = (245, 245, 245)
        panel_background = (25, 25, 27)

        # The first image is always the full camera capture. Keep its aspect
        # ratio, but place it in a fixed-width top panel.
        capture_width = canvas_width - margin * 2
        capture_height = 560
        detail_width = (capture_width - gap) // 2
        detail_height = 340
        detail_rows = max(1, (len(decoded) - 1 + 1) // 2)
        total_height = (
            margin
            + label_height
            + capture_height
            + gap
            + detail_rows * (label_height + detail_height)
            + max(0, detail_rows - 1) * gap
            + margin
        )
        canvas = Image.new("RGB", (canvas_width, total_height), background)
        draw = ImageDraw.Draw(canvas)

        def paste_panel(image, x, y, width, height, label):
            draw.rectangle(
                (x, y, x + width, y + label_height + height),
                fill=panel_background,
                outline=(255, 255, 255),
                width=border,
            )
            draw.text((x + 12, y + 8), label, fill="black")
            fitted = ImageOps.contain(
                image, (width - border * 2, height - border * 2)
            )
            image_x = x + (width - fitted.width) // 2
            image_y = y + label_height + (height - fitted.height) // 2
            canvas.paste(fitted, (image_x, image_y))

        paste_panel(
            decoded[0],
            margin,
            margin,
            capture_width,
            capture_height,
            "CAPTURE PENUH",
        )

        details = decoded[1:]
        y = margin + label_height + capture_height + gap
        for index in range(0, len(details), 2):
            row = details[index:index + 2]
            for column, image in enumerate(row):
                label = (
                    f"WAJAH {index // 2 + 1}"
                    if index + column == 0
                    else f"FOTO REFERENSI {index // 2 + 1}"
                )
                paste_panel(
                    image,
                    margin + column * (detail_width + gap),
                    y,
                    detail_width,
                    detail_height,
                    label,
                )
            y += label_height + detail_height + gap

        output = io.BytesIO()
        canvas.save(output, format="JPEG", quality=90, optimize=True)
        return output.getvalue()
