# AutoBiz Data

AutoBiz Data adalah aplikasi desktop lokal untuk membantu pemilik dan tim
bisnis mengimpor data, memeriksa kualitasnya, melihat ringkasan KPI, mencoba
analisis, dan mengekspor laporan. Aplikasi tidak mengirim file dataset ke
layanan AI atau menyimpan isi dataset file ke database proyek.

## Menjalankan aplikasi

Pastikan Python 3.10 atau lebih baru tersedia. Dari folder proyek:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Di Windows, aktifkan virtual environment dengan
`.venv\Scripts\activate` lalu jalankan `python main.py`.

## Alur kerja

1. Buka **Data & Proyek**, buat proyek baru bila diperlukan, lalu impor CSV/Excel
   atau ambil JSON dari API.
2. Tinjau jumlah baris/kolom, nilai kosong, dan pratinjau. Pembersihan dasar
   membuang baris yang seluruh nilainya kosong dan baris duplikat; file sumber
   tidak ditimpa.
3. Buka **Analisis Bisnis** untuk menjalankan ringkasan KPI, EDA, perbandingan
   regresi/klasifikasi, clustering, forecasting, deteksi anomali, atau analisis
   korelasi.
4. Baca hasil dan batasannya, lalu simpan laporan atau ekspor dataset.
5. Simpan proyek untuk merekam metadata/lokasi dataset dan riwayat analisis.

Ringkasan KPI mencoba mengenali kolom dari nama dalam Bahasa Indonesia maupun
Inggris. Pemetaan ini heuristik: cocokkan satuan dan makna kolom sebelum
menggunakan angkanya sebagai KPI bisnis.

## Proyek dan database

SQLite menyimpan nama proyek, lokasi file dataset, sumber data, serta riwayat
analisis. Isi file CSV/Excel tetap berada pada lokasi asal dan tidak dimasukkan
ke database. Lokasi database:

- macOS: `~/Library/Application Support/AutoBizData/autobiz.sqlite3`
- Windows: `%APPDATA%/AutoBizData/autobiz.sqlite3`
- Linux: `$XDG_DATA_HOME/AutoBizData/autobiz.sqlite3` atau
  `~/.local/share/AutoBizData/autobiz.sqlite3`

Jika file sumber dipindah atau dihapus, buka proyek dan tautkan ulang dataset.
Perubahan hasil pembersihan hanya berada di sesi aktif sampai diekspor sebagai
file baru. Impor dari API publik dibuat sebagai snapshot CSV lokal di folder
`api_imports` pada direktori data aplikasi agar bisa dibuka kembali.

## Integrasi API gratis

### Mengambil data ke AutoBiz

Di **Data & Proyek**:

- **Ambil JSON** menerima URL API HTTPS yang tidak memerlukan autentikasi.
  Respons harus berupa daftar objek JSON atau objek yang memuat `data`,
  `results`, `records`, `items`, atau `holidays`.
- **Impor kalender libur** mengambil kalender publik Nager.Date berdasarkan
  kode negara dan tahun; contoh bawaan Indonesia (`ID`). Koneksi internet
  diperlukan. Integrasi memakai Community API v4; endpoint publik dan aturan
  penggunaannya dapat berubah.

Untuk menjaga respons aplikasi, permintaan API dibatasi 15 detik dan ukuran
respons 10 MB. HTTP tanpa enkripsi hanya diizinkan untuk localhost. URL,
termasuk parameter yang mungkin berisi token, tidak disimpan dalam proyek.
Versi awal ini belum mendukung OAuth atau penyimpanan kredensial API.

### Menyediakan data dari AutoBiz

Di **Pengaturan**, jalankan API lokal. API hanya bind ke `127.0.0.1` dan
menyediakan endpoint baca-saja:

- `GET /health`
- `GET /api/v1/projects`
- `GET /api/v1/projects/{project_id}`
- `GET /api/v1/projects/{project_id}/analyses`

API tidak menyediakan isi file dataset dan tidak menerima operasi tulis.
Endpoint tidak memiliki autentikasi karena hanya ditujukan untuk komputer yang
sama; proses lokal lain tetap dapat mengakses metadata dan hasil analisis.
Hentikan API setelah selesai dan jangan meneruskan port ini ke jaringan atau
internet.

## Fitur analisis dan batasan

- EDA: ringkasan ukuran, tipe kolom, nilai kosong, duplikat, dan statistik
  deskriptif.
- Regresi/klasifikasi: perbandingan model baseline pada pembagian train/test.
- Clustering, forecasting baseline berbasis fitur lag, dan deteksi anomali.
- Korelasi untuk kolom numerik.
- Laporan teks berisi profil dataset, ringkasan KPI, analisis terakhir, dan
  riwayat terbaru.

Model dan KPI adalah bantuan eksplorasi, bukan nasihat keuangan, audit, atau
jaminan prediksi. Validasi hasil terhadap definisi dan konteks bisnis Anda.
AI Advisor tetap berjalan lokal dan bukan model AI eksternal.

## Struktur proyek

```text
main.py                 Launcher aplikasi
autobiz/
  window.py             Antarmuka desktop dan alur pengguna
  storage.py            Metadata proyek dan riwayat SQLite
  integrations.py       Impor JSON dan kalender hari libur
  api_server.py          REST API baca-saja localhost
  business.py            Ringkasan KPI bisnis berbasis nama kolom
tests/                   Pengujian unit dan API lokal
assets/                  Logo aplikasi
```

## Pengujian

Jalankan pengujian tanpa dependensi test tambahan:

```bash
python -m unittest discover -s tests -v
```
