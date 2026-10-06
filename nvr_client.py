"""
Client untuk berkomunikasi dengan NVR Hikvision lewat ISAPI.

Menangani:
  - Pencarian gambar capture (ContentMgmt/search)
  - Download gambar (baik dari URL penuh maupun path relatif ISAPI)
  - Daftar & pencarian Face Library (FDLib)

CATATAN PENTING soal search_pictures():
  Selama pengembangan awal, kita berhasil mengonfirmasi endpoint
  POST /ISAPI/ContentMgmt/search benar-benar ada dan bisa diakses
  (response valid, status 200), TAPI kita belum sempat menguji kasus
  di mana hasilnya benar-benar ADA capture yang cocok (totalMatches > 0).
  Jadi bagian _parse_search_result() di bawah ini ditulis SEDEFENSIF
  mungkin (mencoba beberapa kemungkinan nama tag), tapi WAJIB divalidasi
  memakai tools/test_capture_search.py begitu ada capture nyata untuk
  dites, dan disesuaikan kalau nama field-nya ternyata berbeda.
"""
import logging
import uuid
from datetime import datetime, timezone

import requests
import xml.etree.ElementTree as ET
from requests.auth import HTTPDigestAuth

logger = logging.getLogger(__name__)

NS = {"hik": "http://www.hikvision.com/ver20/XMLSchema"}


class NVRClient:
    def __init__(self, base_url: str, username: str, password: str, timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.auth = HTTPDigestAuth(username, password)
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Helper HTTP dasar
    # ------------------------------------------------------------------
    def _post_xml(self, path: str, xml_body: str) -> str:
        url = f"{self.base_url}{path}"
        resp = requests.post(
            url,
            data=xml_body.encode("utf-8"),
            auth=self.auth,
            headers={"Content-Type": "application/xml"},
            timeout=self.timeout,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            detail = resp.text.strip()
            if detail:
                raise requests.HTTPError(f"{exc} | NVR response: {detail}", response=resp) from exc
            raise
        return resp.text

    def _get(self, path: str, params: dict = None) -> requests.Response:
        url = f"{self.base_url}{path}"
        resp = requests.get(url, auth=self.auth, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------
    # Pencarian gambar capture
    # ------------------------------------------------------------------
    def search_pictures(
        self,
        track_id: int,
        start_time: datetime,
        end_time: datetime,
        metadata_descriptor: str,
        max_results: int = 30,
        position: int = 0,
    ) -> dict:
        """
        Cari gambar capture pada satu trackID (channel) di antara start_time
        dan end_time (keduanya harus timezone-aware datetime).

        Return dict: {"raw_xml": str, "total_matches": int, "items": [dict, ...]}
        """
        search_id = str(uuid.uuid4())
        start_str = start_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        xml_body = f"""<?xml version="1.0" encoding="utf-8"?>
<CMSearchDescription>
<searchID>{search_id}</searchID>
<trackIDList><trackID>{track_id}</trackID></trackIDList>
<timeSpanList><timeSpan>
<startTime>{start_str}</startTime>
<endTime>{end_str}</endTime>
</timeSpan></timeSpanList>
<maxResults>{max_results}</maxResults>
<searchResultPostion>{position}</searchResultPostion>
<metadataList><metadataDescriptor>{metadata_descriptor}</metadataDescriptor></metadataList>
</CMSearchDescription>"""

        raw = self._post_xml("/ISAPI/ContentMgmt/search", xml_body)
        return self._parse_search_result(raw)

    @staticmethod
    def _parse_search_result(raw_xml: str) -> dict:
        result = {"raw_xml": raw_xml, "total_matches": 0, "items": []}
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as e:
            logger.error("Gagal parse XML hasil pencarian: %s | XML: %s", e, raw_xml[:500])
            return result

        total_text = root.findtext("hik:totalMatches", default="0", namespaces=NS) \
            or root.findtext("hik:numOfMatches", default="0", namespaces=NS)
        try:
            result["total_matches"] = int(total_text)
        except (TypeError, ValueError):
            result["total_matches"] = 0

        # Coba beberapa kemungkinan struktur berbeda untuk daftar item hasil,
        # karena skema pasti untuk kasus "ada hasil" belum terverifikasi.
        candidate_paths = [
            ".//hik:matchList/hik:searchMatchItem",
            ".//hik:searchMatchItem",
            ".//hik:MatchList/hik:MatchElement",
            ".//hik:MatchElement",
        ]
        raw_items = []
        for xpath in candidate_paths:
            found = root.findall(xpath, NS)
            if found:
                raw_items = found
                break

        for item in raw_items:
            entry = {}
            for child in item.iter():
                tag = child.tag.split("}")[-1]
                if child.text and child.text.strip():
                    entry[tag] = child.text.strip()

            # Coba temukan URI/URL gambar di beberapa lokasi umum yang berbeda
            picture_uri = (
                entry.get("playbackURI")
                or entry.get("picUrl")
                or entry.get("picURL")
                or entry.get("fileList")
            )
            entry["_picture_uri"] = picture_uri
            result["items"].append(entry)

        logger.debug(
            "Parsed search result: total_matches=%s, items_found=%s",
            result["total_matches"], len(result["items"]),
        )
        return result

    # ------------------------------------------------------------------
    # Download gambar
    # ------------------------------------------------------------------
    def download_picture(self, picture_url_or_path: str) -> bytes:
        """
        Download gambar. Menerima baik URL penuh (contoh: picURL dari FDLib)
        maupun path relatif ISAPI (contoh: dari hasil ContentMgmt/search).
        """
        if picture_url_or_path.startswith("http"):
            resp = requests.get(picture_url_or_path, auth=self.auth, timeout=self.timeout)
        else:
            resp = self._get(picture_url_or_path)
        resp.raise_for_status()
        return resp.content

    def take_snapshot(self, track_id: int) -> bytes:
        """Ambil JPEG snapshot terbaru dari channel NVR."""
        resp = self._get(f"/ISAPI/Streaming/channels/{track_id}/picture")
        content_type = resp.headers.get("Content-Type", "").lower()
        if not content_type.startswith("image/"):
            raise ValueError(
                f"NVR mengembalikan Content-Type bukan gambar untuk track {track_id}: {content_type}"
            )
        return resp.content

    # ------------------------------------------------------------------
    # Face Library (FDLib)
    # ------------------------------------------------------------------
    def list_face_libraries(self) -> dict:
        """GET /ISAPI/Intelligent/FDLib?format=json - daftar Face Library di NVR."""
        resp = self._get("/ISAPI/Intelligent/FDLib", params={"format": "json"})
        return resp.json()

    def search_face_library(self, fdid: str, max_results: int = 200) -> list:
        """
        POST /ISAPI/Intelligent/FDLib/FDSearch - daftar orang & foto referensi
        yang terdaftar di satu Face Library (FDID).

        Return: list of dict {"name", "pid", "pic_url", "contact"}
        """
        search_id = f"fdsearch-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        xml_body = f"""<?xml version="1.0" encoding="UTF-8"?>
<FDSearchDescription version="1.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
<searchID>{search_id}</searchID>
<searchResultPosition>0</searchResultPosition>
<maxResults>{max_results}</maxResults>
<FDID>{fdid}</FDID>
</FDSearchDescription>"""

        raw = self._post_xml("/ISAPI/Intelligent/FDLib/FDSearch", xml_body)
        root = ET.fromstring(raw)

        people = []
        for el in root.findall(".//hik:MatchElement", NS):
            people.append(
                {
                    "name": el.findtext("hik:name", default="", namespaces=NS).strip(),
                    "pid": el.findtext("hik:PID", default="", namespaces=NS).strip(),
                    "pic_url": el.findtext("hik:picURL", default="", namespaces=NS).strip(),
                    "contact": (
                        el.findtext("hik:contact", default="", namespaces=NS).strip()
                        or el.findtext("hik:phoneNumber", default="", namespaces=NS).strip()
                    ),
                }
            )
        return people
