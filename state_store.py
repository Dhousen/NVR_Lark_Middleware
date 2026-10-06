"""Persistent cooldown state with a cross-process atomic notification claim."""
import json
import logging
import msvcrt
import os
import tempfile
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class StateStore:
    def __init__(self, path: str, last_notified_ttl_seconds: int = 7 * 24 * 3600):
        self.path = path
        self.ttl = last_notified_ttl_seconds
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.lock_path = path + ".lock"

    @contextmanager
    def _exclusive_lock(self):
        with open(self.lock_path, "a+b") as lock_file:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "last_notified": data.get("last_notified", {}),
                "attendance": data.get("attendance", {}),
            }
        except FileNotFoundError:
            return {"last_notified": {}, "attendance": {}}
        except (json.JSONDecodeError, OSError, AttributeError) as e:
            logger.warning("File state tidak bisa dibaca, mulai dari kosong: %s", e)
            return {"last_notified": {}, "attendance": {}}

    def _save(self, data):
        directory = os.path.dirname(self.path) or "."
        fd, temporary = tempfile.mkstemp(dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def claim_notification(self, person_key: str, cooldown_seconds: int) -> bool:
        """Atomically check cooldown, record the claim, and prune old keys."""
        now = time.time()
        with self._exclusive_lock():
            data = self._load()
            last_notified = data["last_notified"]
            cutoff = now - self.ttl
            last_notified = {
                key: value for key, value in last_notified.items()
                if isinstance(value, (int, float)) and value >= cutoff
            }
            last = last_notified.get(person_key, 0)
            if now - last < cooldown_seconds:
                data["last_notified"] = last_notified
                self._save(data)
                return False
            last_notified[person_key] = now
            data["last_notified"] = last_notified
            self._save(data)
            return True

    def claim_attendance(self, date_key: str, session: str, person_key: str) -> bool:
        """Klaim absensi sekali per orang, tanggal, dan sesi secara atomic."""
        with self._exclusive_lock():
            data = self._load()
            attendance = data.setdefault("attendance", {})
            day = attendance.setdefault(date_key, {})
            people = day.setdefault(session, {})
            if person_key in people:
                return False
            people[person_key] = time.time()
            for old_date in list(attendance):
                if old_date != date_key:
                    attendance.pop(old_date, None)
            self._save(data)
            return True

    def release_notification(self, person_key: str) -> None:
        with self._exclusive_lock():
            data = self._load()
            data["last_notified"].pop(person_key, None)
            self._save(data)

    def release_attendance(self, date_key: str, session: str, person_key: str) -> None:
        with self._exclusive_lock():
            data = self._load()
            day = data.get("attendance", {}).get(date_key, {})
            day.get(session, {}).pop(person_key, None)
            self._save(data)
