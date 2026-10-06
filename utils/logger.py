"""
Setup logging terpusat: tulis ke console DAN ke file (dengan rotasi
otomatis supaya file log tidak membengkak tanpa batas).
"""
import logging
import logging.handlers
import os


def setup_logging(log_dir: str, level: str = "INFO") -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "middleware.log")

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    # Hindari duplikasi handler kalau setup_logging dipanggil lebih dari sekali
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # requests/urllib3 log-nya berisik di level DEBUG, redam supaya log kita bersih
    logging.getLogger("urllib3").setLevel(logging.WARNING)
