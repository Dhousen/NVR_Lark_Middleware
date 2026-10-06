"""Pencocokan wajah lokal menggunakan InsightFace/ArcFace."""
import io
import json
import logging
import os
import pickle
import hashlib
import tempfile

from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
except Exception as exc:  # dependency may fail during native/runtime loading
    cv2 = np = FaceAnalysis = None
    _INSIGHTFACE_IMPORT_ERROR = exc


class FaceMatcher:
    CACHE_VERSION = "insightface-arcface-buffalo_l-v1"
    def __init__(
        self,
        reference_dir: str,
        tolerance: float = 0.5,
        min_confidence: float = 50.0,
        cache_path: str = None,
    ):
        self.reference_dir = reference_dir
        self.tolerance = tolerance
        self.min_confidence = min_confidence
        self.cache_path = cache_path or os.path.join(
            os.path.dirname(reference_dir.rstrip("/\\")) or ".", "encodings.pkl"
        )
        if FaceAnalysis is None:
            raise RuntimeError(
                "InsightFace/ArcFace belum tersedia. Install dependencies dengan "
                "`pip install -r requirements.txt` (atau set ENABLE_FACE_MATCHING=false)."
            ) from _INSIGHTFACE_IMPORT_ERROR
        self.known_encodings = []
        self.known_names = []
        self.reference_paths = {}
        self.contacts = {}
        contacts_path = os.path.join(self.reference_dir, "contacts.json")
        if os.path.isfile(contacts_path):
            try:
                with open(contacts_path, "r", encoding="utf-8") as f:
                    self.contacts = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Gagal memuat metadata contact: %s", e)

        try:
            self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            # Larger detection canvas improves recall for faces occupying few
            # pixels in wide-angle room snapshots.
            self.app.prepare(ctx_id=0, det_size=(1024, 1024))
        except Exception as exc:
            raise RuntimeError(
                "InsightFace gagal diinisialisasi. Pastikan model buffalo_l dapat "
                "diunduh dan ONNX Runtime terpasang."
            ) from exc
        self._load_or_build_encodings()

    # ------------------------------------------------------------------
    def _load_or_build_encodings(self) -> None:
        fingerprint = self._reference_fingerprint()
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cache = pickle.load(f)
                if (
                    cache.get("backend") != self.CACHE_VERSION
                    or cache.get("reference_fingerprint") != fingerprint
                ):
                    raise ValueError("cache lama")
                self.known_encodings = cache["encodings"]
                self.known_names = cache["names"]
                self._index_reference_paths()
                logger.info("Memuat %d encoding wajah dari cache.", len(self.known_names))
                return
            except (
                OSError,
                pickle.PickleError,
                KeyError,
                ValueError,
                TypeError,
                AttributeError,
            ) as e:
                logger.warning("Gagal memuat cache encoding, membangun ulang: %s", e)

        self.build_encodings()

    def build_encodings(self) -> None:
        """
        Hitung ulang encoding wajah dari setiap file gambar di reference_dir.
        Nama file (tanpa ekstensi) dipakai sebagai nama orang yang ditampilkan.
        """
        encodings, names = [], []
        if not os.path.isdir(self.reference_dir):
            logger.warning("Folder foto referensi tidak ditemukan: %s", self.reference_dir)
            self._save_cache(encodings, names)
            return

        image_paths = []
        for root, _, filenames in os.walk(self.reference_dir):
            image_paths.extend(
                os.path.join(root, filename)
                for filename in sorted(filenames)
                if filename.lower().endswith((".jpg", ".jpeg", ".png"))
            )

        for path in sorted(image_paths):
            try:
                image = cv2.imread(path)
                faces = self.app.get(image) if image is not None else []
                if len(faces) != 1:
                    logger.warning(
                        "Foto referensi %s harus memiliki tepat satu wajah; ditemukan %d.",
                        path,
                        len(faces),
                    )
                    continue
                encodings.append(np.asarray(faces[0].embedding, dtype=np.float32))
                relative_dir = os.path.relpath(os.path.dirname(path), self.reference_dir)
                display_name = (
                    os.path.basename(relative_dir)
                    if relative_dir != "."
                    else os.path.splitext(os.path.basename(path))[0]
                )
                names.append(display_name)
                self.reference_paths.setdefault(display_name, path)
            except Exception as e:  # noqa: BLE001 - satu foto rusak jangan sampai hentikan proses
                logger.warning("Gagal memproses foto referensi %s: %s", path, e)

        self.known_encodings = encodings
        self.known_names = names
        self._save_cache(encodings, names)
        logger.info("Berhasil membangun %d encoding wajah dari foto referensi.", len(names))

    def _index_reference_paths(self) -> None:
        self.reference_paths = {}
        for root, _, filenames in os.walk(self.reference_dir):
            relative_dir = os.path.relpath(root, self.reference_dir)
            for filename in sorted(filenames):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                name = (
                    os.path.basename(relative_dir)
                    if relative_dir != "."
                    else os.path.splitext(filename)[0]
                )
                self.reference_paths.setdefault(name, os.path.join(root, filename))

    def _save_cache(self, encodings, names) -> None:
        tmp_path = None
        try:
            directory = os.path.dirname(self.cache_path) or "."
            os.makedirs(directory, exist_ok=True)
            tmp_fd, tmp_path = tempfile.mkstemp(dir=directory)
            with os.fdopen(tmp_fd, "wb") as f:
                pickle.dump(
                    {
                        "backend": self.CACHE_VERSION,
                        "reference_fingerprint": self._reference_fingerprint(),
                        "encodings": encodings,
                        "names": names,
                    },
                    f,
                )
            os.replace(tmp_path, self.cache_path)
        except OSError as e:
            logger.warning("Gagal menyimpan cache encoding: %s", e)
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _reference_fingerprint(self) -> str:
        digest = hashlib.sha256()
        if not os.path.isdir(self.reference_dir):
            return digest.hexdigest()
        image_paths = []
        for root, _, filenames in os.walk(self.reference_dir):
            image_paths.extend(
                os.path.join(root, filename)
                for filename in filenames
                if filename.lower().endswith((".jpg", ".jpeg", ".png"))
            )
        for path in sorted(image_paths):
            try:
                stat = os.stat(path)
                digest.update(os.path.relpath(path, self.reference_dir).encode("utf-8"))
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
            except OSError:
                continue
        return digest.hexdigest()

    # ------------------------------------------------------------------
    def match(self, image_bytes: bytes):
        """
        Return (nama, confidence_persen) untuk kecocokan terbaik,
        atau (None, 0.0) kalau tidak ada wajah/kecocokan/matcher nonaktif.
        """
        name, confidence, _ = self.match_and_crop(image_bytes)
        return name, confidence

    def match_and_crop(self, image_bytes: bytes):
        """
        Cocokkan semua wajah pada gambar dan kembalikan crop wajah terbaik.

        Crop diperbesar secara digital agar gambar yang dikirim ke Lark lebih
        mudah diperiksa. Pembesaran tidak menambah detail kamera yang hilang.
        """
        if not self.known_encodings:
            return None, 0.0, image_bytes

        try:
            image = self._decode_image(image_bytes)
            faces = self.app.get(image)
            locations = [self._bbox_to_location(face.bbox, image.shape) for face in faces]
        except Exception as e:  # noqa: BLE001
            logger.warning("Gagal memproses gambar capture untuk matching: %s", e)
            return None, 0.0, image_bytes

        if not faces:
            return None, 0.0, image_bytes

        best_face_index = 0
        best_person_index = 0
        best_distance = float("inf")
        for face_index, face in enumerate(faces):
            distances = self._distances(face.embedding)
            person_index = int(np.argmin(distances))
            distance = float(distances[person_index])
            if distance < best_distance:
                best_face_index = face_index
                best_person_index = person_index
                best_distance = distance

        cropped_bytes = self._crop_face(image_bytes, locations[best_face_index])

        confidence = max(0.0, (1 - best_distance)) * 100
        if best_distance <= self.tolerance and confidence >= self.min_confidence:
            matched_name = self.known_names[best_person_index]
            return (
                matched_name,
                round(confidence, 1),
                self._comparison_image(
                    cropped_bytes,
                    self.reference_paths.get(matched_name),
                    matched_name,
                    confidence,
                ),
            )

        return None, 0.0, cropped_bytes

    def match_faces(self, image_bytes: bytes):
        """Kembalikan hasil untuk setiap wajah, tanpa mengubah gambar capture."""
        try:
            image = self._decode_image(image_bytes)
            faces = self.app.get(image)
            locations = [self._bbox_to_location(face.bbox, image.shape) for face in faces]
        except Exception as e:  # noqa: BLE001
            logger.warning("Gagal memproses wajah pada capture: %s", e)
            return []

        results = []
        for face, location in zip(faces, locations):
            name = None
            confidence = 0.0
            if self.known_encodings:
                distances = self._distances(face.embedding)
                person_index = int(np.argmin(distances))
                distance = float(distances[person_index])
                match_confidence = max(0.0, (1 - distance)) * 100
                if distance <= self.tolerance and match_confidence >= self.min_confidence:
                    name = self.known_names[person_index]
                    confidence = round(match_confidence, 1)
            results.append({
                "name": name,
                "confidence": confidence,
                "crop": self._crop_face(image_bytes, location),
                "reference": self.reference_paths.get(name) if name else None,
            })
        return results

    @staticmethod
    def _comparison_image(
        live_bytes: bytes, reference_path: str, name: str, confidence: float
    ) -> bytes:
        """Buat panel live-vs-reference seperti tampilan face recognition NVR."""
        with Image.open(io.BytesIO(live_bytes)).convert("RGB") as live:
            if reference_path and os.path.isfile(reference_path):
                with Image.open(reference_path).convert("RGB") as reference:
                    target_height = max(live.height, reference.height)
                    panel_width = max(live.width, reference.width)
                    panel = Image.new("RGB", (panel_width * 2, target_height + 72), "white")
                    panel.paste(live.resize((panel_width, target_height)), (0, 0))
                    panel.paste(reference.resize((panel_width, target_height)), (panel_width, 0))
            else:
                panel = Image.new("RGB", (live.width, live.height + 72), "white")
                panel.paste(live, (0, 0))

            from PIL import ImageDraw

            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, panel.height - 72, panel.width, panel.height), fill=(35, 35, 35))
            draw.text((12, panel.height - 60), f"Live        Reference    {confidence:.1f}%", fill="white")
            draw.text((12, panel.height - 34), name, fill="white")
            output = io.BytesIO()
            panel.save(output, format="JPEG", quality=92)
            return output.getvalue()

    @staticmethod
    def _crop_face(image_bytes: bytes, location) -> bytes:
        top, right, bottom, left = location
        with Image.open(io.BytesIO(image_bytes)) as source:
            width, height = source.size
            margin_x = max(24, (right - left) // 2)
            margin_y = max(24, (bottom - top) // 2)
            left = max(0, left - margin_x)
            top = max(0, top - margin_y)
            right = min(width, right + margin_x)
            bottom = min(height, bottom + margin_y)
            crop = source.crop((left, top, right, bottom))
            crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            crop.convert("RGB").save(output, format="JPEG", quality=92)
            return output.getvalue()

    def contact_for(self, name: str):
        if not name:
            return ""
        if name in self.contacts:
            return self.contacts[name]
        normalized = name.casefold()
        for contact_name, contact in self.contacts.items():
            if str(contact_name).casefold() == normalized:
                return contact
        return ""

    @staticmethod
    def _decode_image(image_bytes: bytes):
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("data snapshot bukan gambar yang valid")
        return image

    def _distances(self, embedding):
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm:
            vector = vector / norm
        known = np.asarray(self.known_encodings, dtype=np.float32)
        known_norms = np.linalg.norm(known, axis=1, keepdims=True)
        known = known / np.maximum(known_norms, 1e-12)
        return 1.0 - np.dot(known, vector)

    @staticmethod
    def _bbox_to_location(bbox, shape):
        height, width = shape[:2]
        left, top, right, bottom = [int(round(value)) for value in bbox]
        return (
            max(0, top), min(width, right), min(height, bottom), max(0, left)
        )

    def reference_image(self, name: str):
        path = self.reference_paths.get(name)
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError as e:
            logger.warning("Gagal membaca foto referensi %s: %s", name, e)
            return None
