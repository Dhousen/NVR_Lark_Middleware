# NVR Hikvision → Lark: Middleware Notifikasi Wajah

Middleware ini memantau capture wajah dari NVR Hikvision, mencocokkannya
dengan foto referensi (dari Face Library NVR), lalu mengirim notifikasi
(foto + nama + waktu) ke grup Lark lewat Custom Bot Webhook.

## Perubahan Terbaru

Perubahan berikut sudah diterapkan pada session ini:

- Pencarian `POST /ISAPI/ContentMgmt/search` diperbaiki agar memakai
  `searchID` UUID dan format XML Hikvision yang sesuai.
- Error HTTP dari NVR sekarang menampilkan body XML respons, sehingga alasan
  penolakan lebih mudah didiagnosis.
- Ditemukan bahwa `trackID` yang tersedia pada NVR adalah `101`, `201`, dan
  `301`; konfigurasi tidak lagi memakai `103`, `203`, dan `303`.
- Hasil `ContentMgmt/search` terbukti berupa segmen video RTSP, bukan foto
  snapshot, sehingga tidak digunakan sebagai sumber face recognition.
- Ditambahkan pengambilan JPEG langsung melalui
  `/ISAPI/Streaming/channels/{trackID}/picture`.
- Face recognition sekarang memproses snapshot langsung dari NVR, bukan URL
  RTSP hasil pencarian video.
- Snapshot diambil tiga kali pada setiap jadwal: `:00`, `:10`, dan `:20`
  berdasarkan waktu lokal komputer/server.
- InsightFace/ArcFace memproses foto referensi lokal (enam foto ditemukan)
  pada folder `data/reference_photos/`.
- Metadata `contact` dari Face Library sekarang disimpan sebagai metadata
  lokal dan ditampilkan sebagai NIK pada alert untuk wajah yang berhasil
  dikenali. Jika `contact` tidak dikirim firmware, middleware menggunakan
  field `phoneNumber` sebagai fallback.

Konfigurasi absensi dapat diubah melalui:

```env
NVR_TRACK_IDS=101,201,301
ATTENDANCE_ENTRY_START=06:00
ATTENDANCE_ENTRY_END=10:00
ATTENDANCE_EXIT_START=15:00
ATTENDANCE_EXIT_END=23:00
ATTENDANCE_POLL_INTERVAL_SECONDS=60
```

## Arsitektur

```

## Pipeline frame/video opsional

Selain pipeline snapshot NVR, tersedia [`new_inference_simplified.py`](./new_inference_simplified.py)
untuk memproses kumpulan JPEG dari folder. Pipeline ini memakai YOLO World
untuk deteksi `person`, ByteTrack untuk `track_id`, lalu ArcFace untuk
identifikasi. Pipeline ini **tidak menggantikan** `main.py`, scheduler,
cooldown atomic, kolase, contact/NIK, atau notifikasi multi-wajah yang sudah
ada.

Pasang dependency tambahannya hanya bila fitur ini diperlukan:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```

Siapkan store `weights/faces/employees.npz` dengan array:
`embeddings`, `employee_ids`, dan `employee_names`, lalu letakkan frame `.jpg`
di folder `frames` dan jalankan:

```powershell
.\venv\Scripts\python.exe new_inference_simplified.py
```

Store dapat dibuat dari foto dengan struktur folder berikut:

```text
employees\
  Ferrdy\
    foto1.jpg
    foto2.jpg
  Novriani\
    foto1.jpg
```

Jalankan enrollment:

```powershell
.\venv\Scripts\python.exe scripts\enroll_faces.py
```

Setiap foto harus memiliki tepat satu wajah. Hasilnya disimpan ke
`weights\faces\employees.npz` dan `employees.json`. Loader juga kompatibel
dengan format lama yang memakai key `vectors` dan metadata JSON.
Jika folder `employees` belum dibuat, script otomatis menggunakan
`data\reference_photos` sebagai sumber foto.

Hasil CSV ditulis ke folder `output`. Notifikasi pipeline ini dikirim satu kali
ketika sebuah `track_id` pertama kali berhasil dikenali. Notifikasi snapshot
utama tetap menggunakan `main.py` dan tetap mengirim per wajah sesuai alur
existing.

## Deteksi manusia lalu pencocokan Face Library

Untuk menguji gambar secara manual dengan alur `person detection -> face
matching`, gunakan [`person_face_matcher.py`](./person_face_matcher.py).
YOLO hanya mendeteksi objek kelas `person`; pencocokan wajah tetap memakai
`FaceMatcher` dan folder `data\reference_photos` yang sama dengan `main.py`.

```powershell
.\venv\Scripts\python.exe person_face_matcher.py `
  --image data\last_test_snapshot.jpg `
  --output output\person_face_result.jpg `
  --json-output output\person_face_result.json
```

Pada eksekusi pertama, `yolov8n.pt` dapat diunduh otomatis oleh Ultralytics.
Wajah yang tidak dikenali tetap dicatat dengan `name: null`; script ini tidak
mengirim notifikasi Lark.
[NVR Hikvision] --(scheduled JPEG snapshot)-----------> [main.py]
                                                              |
                                                    [face_matcher.py]
                                                  (cocokkan dgn foto referensi)
                                                              |
                                                       [lark_client.py]
                                                    (upload foto + kirim pesan)
                                                              |
                                                        [Grup Lark]
```

Berbeda dari rencana awal (memakai fitur "Face Comparison" internal
NVR), middleware ini **melakukan pencocokan wajah sendiri** di sisi
middleware, memakai foto referensi yang diunduh dari Face Library NVR.
Alasannya: hasil comparison internal NVR (nama + similarity) ternyata
hanya bisa dilihat di monitor lokal yang terhubung langsung ke NVR,
tidak diekspos lewat web/API dengan cara yang bisa diakses eksternal.

## Catatan Validasi NVR

Endpoint snapshot yang digunakan dan sudah divalidasi:

```bash
GET /ISAPI/Streaming/channels/101/picture
```

NVR mengembalikan `HTTP 200`, `Content-Type: image/jpeg`, dan data JPEG.
Endpoint pencarian video tetap tersedia untuk diagnosis, tetapi tidak dipakai
oleh alur face recognition.

Untuk menguji pencarian video secara terpisah:

```bash
python tools/test_capture_search.py --track 101 --minutes 1440 --descriptor //recordType.meta.std-cgi.com
```

Perintah tersebut dapat menghasilkan `totalMatches`, tetapi hasilnya adalah
`playbackURI` RTSP dan bukan foto.

## Instalasi

### 1. Python

Butuh Python 3.9 atau lebih baru.

### 2. Buat virtual environment (disarankan)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### Catatan khusus Windows: InsightFace + ArcFace

Matching memakai `insightface` (model ArcFace `buffalo_l`) dan
`onnxruntime`, bukan `face_recognition`/`dlib`. Jalankan:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Pada inisialisasi pertama, InsightFace mengunduh model `buffalo_l` ke cache
lokal (perlu akses internet). Runtime menggunakan CPU secara default; GPU
memerlukan paket ONNX Runtime GPU dan konfigurasi provider yang sesuai.
Jika InsightFace atau model tidak tersedia, middleware berhenti dengan pesan
error yang jelas (tidak melakukan fallback diam-diam). Untuk menjalankan tanpa
matching, set `ENABLE_FACE_MATCHING=false`.

### 4. Konfigurasi

```bash
copy .env.example .env
```

Buka `.env`, isi minimal:
- `NVR_BASE_URL`, `NVR_USERNAME`, `NVR_PASSWORD`
- `NVR_TRACK_IDS` (channel mana saja yang mau dipantau)
- `ATTENDANCE_ENTRY_START` / `ATTENDANCE_ENTRY_END` untuk sesi masuk
- `ATTENDANCE_EXIT_START` / `ATTENDANCE_EXIT_END` untuk sesi pulang
- `LARK_WEBHOOK_URL` (dari Custom Bot di grup Lark)
- `NVR_FDID` (ID Face Library, didapat dari `GET /ISAPI/Intelligent/FDLib`)

`LARK_APP_ID` dan `LARK_APP_SECRET` opsional — isi kalau mau foto ikut
terkirim (butuh Internal App Lark dengan permission `im:resource`).
Kalau dikosongkan, notifikasi tetap terkirim tapi hanya teks.

Deteksi manusia sebelum face matching aktif secara default. Konfigurasinya:

```env
ENABLE_PERSON_DETECTION=true
PERSON_DETECTOR_MODEL=yolov8n.pt
PERSON_DETECTOR_CONFIDENCE=0.35
```

Alur snapshot sekarang adalah NVR -> YOLO kelas `person` -> crop bounding box
-> `FaceMatcher.match_faces()` -> notifikasi Lark. Untuk kembali ke alur
InsightFace langsung pada seluruh snapshot, gunakan
`ENABLE_PERSON_DETECTION=false`. Dependency YOLO tersedia di
`requirements-inference.txt`.

Jika YOLO menemukan manusia tetapi InsightFace tidak menemukan wajah (wajah
terlalu kecil, tertutup, atau membelakangi kamera), bounding box manusia tetap
dikirim sebagai notifikasi anonim dengan label `Wajah tidak dikenali`.

### 5. Unduh foto referensi dan metadata NIK

```bash
set PYTHONIOENCODING=utf-8
python tools/download_reference_photos.py
```

Ini akan mengisi folder `data/reference_photos/` dengan subfolder per nama
orang yang terdaftar di Face Library NVR. Setiap foto diberi nomor berurutan,
sehingga beberapa entri dengan nama sama tidak saling menimpa:
`data/reference_photos/Nama Orang/photo_001.jpg`. Script juga menghasilkan
`data/reference_photos/contacts.json`.

Nama folder referensi dinormalisasi berdasarkan nama depan. Contoh `Ferrdy`,
`Ferrdy Syach Putera`, `Ferrdy_syachp`, dan `Ferrdy_Syach_Putera` akan masuk
ke folder `data/reference_photos/Ferrdy/`. Untuk menggabungkan folder lama
yang sudah terlanjur dibuat, jalankan:

```powershell
.\venv\Scripts\python.exe tools\group_reference_photos.py
```

Nilai NIK diambil dari field Face Library berikut:

1. `contact`
2. `phoneNumber` sebagai fallback untuk firmware yang tidak mengirim `contact`

Setelah mengubah data orang atau contact di Hikvision, jalankan ulang perintah
di atas lalu restart middleware:

```bash
python main.py
```

Format alert untuk wajah yang dikenali:

```text
Wajah: Nama Orang
NIK: nilai-contact
Kamera: D1
Waktu: ...
```

Untuk wajah yang tidak dikenali, NIK tidak ditampilkan karena tidak ada
identitas yang dapat dipetakan secara aman.

### 5a. Digital zoom pada wajah

Snapshot dari kamera yang mencakup ruangan luas dapat berisi wajah yang sangat
kecil. Middleware sekarang mendeteksi semua wajah, membandingkan setiap wajah
dengan seluruh foto referensi, lalu mengirim crop digital dengan pembesaran 3x
untuk wajah dengan kecocokan terbaik. Jika tidak ada yang melewati tolerance,
crop wajah terbaik tetap dikirim dengan label `Wajah tidak dikenali`.

Digital zoom hanya memperbesar piksel yang sudah ada; sistem tidak dapat
menciptakan detail baru. Karena itu wajah yang terlalu kecil, buram, tertutup,
atau menghadap jauh dari kamera tetap mungkin tidak terdeteksi.

Jika wajah berhasil dikenali, gambar Lark sekarang berupa panel perbandingan:
crop wajah dari kamera di sisi kiri, foto referensi FDLib di sisi kanan, nama,
dan skor kecocokan. Ini meniru tampilan comparison Hikvision. Jika wajah tidak
berhasil dikenali, tidak ada foto referensi yang aman untuk ditampilkan sehingga
yang dikirim adalah gambar capture penuh tanpa zoom.

Jika satu capture berisi beberapa wajah yang berhasil dikenali, middleware
mengirim satu pesan Lark untuk setiap wajah. Setiap pesan berisi capture penuh,
crop/zoom wajah tersebut, foto referensinya, nama, NIK, dan skor kecocokan.
Wajah yang tidak dikenali tidak membuat pesan identitas.

Kolase menggunakan layout seimbang: capture penuh berada pada panel besar di
bagian atas, sedangkan crop wajah dan foto referensi berada pada dua panel
berukuran sama di bawahnya dengan margin dan garis pemisah yang konsisten.

Deteksi InsightFace menggunakan canvas `1024x1024` agar wajah kecil pada
snapshot wide-angle lebih mudah ditemukan. Semua wajah yang terdeteksi sekarang
ikut dimasukkan ke kolase: wajah yang dikenali dipasangkan dengan foto
referensi, sedangkan wajah yang belum dikenali tetap ditampilkan sebagai crop
tanpa foto referensi. Jumlah wajah terdeteksi dicantumkan pada teks alert.

Cache encoding otomatis dibangun ulang jika foto referensi ditambah, dihapus,
atau berubah. Cache juga memiliki versi backend sehingga perubahan model tidak
menggunakan encoding lama. Scheduler memakai toleransi keterlambatan
`SCHEDULE_LATE_TOLERANCE_SECONDS` dan setiap channel diproses terpisah agar
error satu channel tidak menghentikan channel lain.

Foto referensi yang memiliki nol atau lebih dari satu wajah dilewati agar tidak
menghasilkan identitas yang ambigu. Path `STATE_FILE`, `REFERENCE_DIR`, dan
`LOG_DIR` relatif selalu dihitung dari folder project, bukan dari current
working directory. State cooldown memakai file lock Windows dan klaim atomic
agar dua instance tidak mengirim notifikasi yang sama; entri lama dibersihkan
otomatis.

NIK/contact hanya ditambahkan jika wajah berhasil dikenali. Jika hasil
matching masih `Wajah tidak dikenali`, tidak ada nama yang aman untuk dipetakan
ke `contacts.json`. Setelah mengubah data Face Library, jalankan ulang:

```powershell
.\venv\Scripts\python.exe tools\download_reference_photos.py
```

Periksa `data/reference_photos/contacts.json` dan pastikan kunci contact sama
dengan nama file foto referensi. Nilai `phoneNumber` dari firmware Hikvision
juga digunakan sebagai fallback; pastikan field tersebut benar-benar berisi
NIK, bukan jabatan atau nomor telepon.

### 6. (Opsional) Diagnosa pencarian video

```bash
python tools/test_capture_search.py --track 101 --minutes 1440 --descriptor //recordType.meta.std-cgi.com
```

### 7. Tes snapshot sekarang

Untuk menguji snapshot, face recognition, NIK, dan pengiriman Lark tanpa
menunggu jadwal:

```powershell
.\venv\Scripts\python.exe tools\test_snapshot_now.py --track 101
```

Gunakan `--track 201` atau `--track 301` untuk kamera lain. Script juga
menyimpan snapshot terakhir di `data/last_test_snapshot.jpg`.

### 8. Jalankan middleware

```bash
python main.py
```

Middleware memeriksa jadwal secara berkala, tetapi hanya mengambil snapshot
pada jam yang dikonfigurasi. Log akan tampil di layar DAN tersimpan di
`logs/middleware.log`
(otomatis berotasi kalau ukurannya membesar).

Berhenti dengan `Ctrl+C` — middleware akan berhenti dengan rapi.

## Menjalankan sebagai Service (Berjalan Terus di Background)

### Windows (pakai Task Scheduler)

1. Buka Task Scheduler → Create Task.
2. Trigger: "At startup" atau sesuai kebutuhan.
3. Action: "Start a program"
   - Program: path ke `python.exe` di dalam folder `venv\Scripts\`
   - Arguments: path lengkap ke `main.py`
   - Start in: folder project ini
4. Centang "Run whether user is logged on or not" kalau perlu jalan
   walau tidak ada yang login.

### Alternatif: NSSM (Non-Sucking Service Manager)

```bash
nssm install NVRLarkMiddleware "C:\path\ke\venv\Scripts\python.exe" "C:\path\ke\main.py"
nssm start NVRLarkMiddleware
```

## Struktur File

```
.
├── .env.example                # Contoh konfigurasi, salin jadi .env
├── requirements.txt
├── config.py                   # Loader konfigurasi dari .env
├── nvr_client.py                # Komunikasi ISAPI ke NVR
├── face_matcher.py              # Pencocokan wajah lokal
├── lark_client.py               # Upload gambar & kirim pesan ke Lark
├── state_store.py               # Dedupe & cooldown, persisten ke disk
├── main.py                      # Loop utama
├── utils/
│   └── logger.py                # Setup logging + rotasi file
├── tools/
│   ├── download_reference_photos.py   # Unduh foto dari Face Library
│   └── test_capture_search.py         # Diagnosa endpoint pencarian
└── data/
    ├── reference_photos/        # Foto referensi (diisi oleh tools di atas)
    │   └── contacts.json        # Pemetaan nama Face Library ke contact/NIK
    ├── encodings.pkl            # Cache encoding wajah (auto-generated)
    └── state.json               # State dedupe & cooldown (auto-generated)
```

## Troubleshooting

**"Environment variable 'X' wajib diisi tapi kosong"**
→ Cek file `.env` (bukan `.env.example`), pastikan semua field wajib
sudah diisi.

**Snapshot tidak diambil pada jadwal**
→ Pastikan proses `main.py` tetap berjalan, jam komputer/server benar, dan
Jendela absensi menggunakan waktu lokal komputer/server. Polling berlangsung
setiap `ATTENDANCE_POLL_INTERVAL_SECONDS` detik; identitas yang sudah tercatat
di sesi masuk tidak dikirim ulang sampai sesi pulang dimulai.

**Snapshot gagal diambil**
→ Pastikan `NVR_TRACK_IDS` berisi track yang tersedia (`101,201,301`) dan
akun NVR memiliki izin melihat gambar/snapshot.

**Semua capture berlabel "Wajah tidak dikenali" padahal orangnya
terdaftar**
→ Jalankan ulang `python tools/download_reference_photos.py` untuk
memastikan foto referensi terbaru, hapus `data/encodings.pkl`, lalu
jalankan `main.py` lagi. Jangan menaikkan `FACE_MATCH_TOLERANCE` tanpa
menetapkan batas confidence karena dapat menyebabkan salah identifikasi.
Gunakan `FACE_MIN_CONFIDENCE=50` sebagai batas aman awal. Nilai kecocokan
35,8% seperti pada contoh tidak akan dianggap sebagai identitas yang valid.
sedikit (misal dari 0.5 ke 0.6) kalau foto referensi kurang jelas.

**Notifikasi ke Lark tidak muncul foto, cuma teks**
→ Pastikan `LARK_APP_ID` dan `LARK_APP_SECRET` sudah diisi dan App
tersebut sudah punya permission `im:resource` di Lark Developer Console.

**NIK tidak muncul pada alert**
→ Pastikan nilai Contact sudah disimpan pada entri Face Library yang benar,
lalu jalankan ulang `tools/download_reference_photos.py`. Periksa file
`data/reference_photos/contacts.json`. Jika file berisi jabatan atau data
lain, berarti firmware/API mengembalikan `phoneNumber` tersebut sebagai
fallback, bukan NIK.

**Error koneksi ke NVR (timeout/connection refused)**
→ Pastikan laptop/server yang menjalankan middleware ini berada di
jaringan yang sama dengan NVR (`ping <NVR_BASE_URL>` dulu), dan
`NVR_USERNAME`/`NVR_PASSWORD` benar.

## Keamanan

- Jangan commit file `.env` ke Git/repo publik — berisi kredensial NVR
  dan Lark.
- Disarankan buat user API khusus di NVR (bukan `admin`) dengan
  permission minimal (Remote Notify + akses gambar), lalu pakai user
  itu di `NVR_USERNAME`/`NVR_PASSWORD`.
