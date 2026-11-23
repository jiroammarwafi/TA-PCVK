# Naskah Video Progres Tugas Akhir

**Replikasi Kar et al. (2021) — SWT + PHOG + LDA + LS-SVM**

Dokumen kerja untuk perekaman video progres. Total durasi ±12 menit, dibagi menjadi enam segmen dengan pembagian perekaman terpisah agar mudah diulang bila ada kesalahan.

---

## Bagian 0 — Struktur Video (total ±12 menit)

| Urutan | Segmen | Durasi | Yang tampil |
|:--:|---|---:|---|
| 1 | Pembuka + peta pembagian divisi | 1 menit | 1 orang (bisa kamu) |
| 2 | A1 — Data & Praproses | 2,5 menit | Zuhdi |
| 3 | A2 — Feature Engineering | 2 menit | Jiro |
| 4 | A3 — Model & Optimasi | 3 menit | Sandy |
| 5 | A4 — Web & Integrasi | 1,5 menit | Naila |
| 6 | Penutup: status & rencana | 1,5 menit | kamu |

**Aturan main video:** setiap orang wajib membuka file asli di layar, bukan cuma bicara. Angka yang disebut harus berasal dari file yang sedang dibuka.

## Bagian 1 — Peta Pembagian (slide pembuka)

Tampilkan tabel ini di awal:

| Kode | Divisi | Anggota | Bobot | Status |
|:--:|---|:--:|:--:|---|
| A1 | Data & Praproses | Zuhdi | 25% | ✅ selesai (+1 branch belum di-PR) |
| A2 | Feature Engineering | Jiro | 25% | ✅ selesai, ⚠️ fitur noise belum |
| A3 | Model & Optimasi | Sandy | 30% | ⚠️ grid search selesai, Jaya belum |
| A4 | Web & Integrasi | Naila | 20% | ❌ belum mulai |

**Kalimat pembuka yang bisa kamu baca:**

> "Video ini menjelaskan progres tugas akhir replikasi jurnal Kar et al. 2021 tentang pengenalan ekspresi wajah dengan SWT + PHOG + LDA + LS-SVM. Repo dibagi empat divisi: A1 data dan praproses, A2 ekstraksi fitur, A3 model dan optimasi, A4 web. Kami laporkan apa yang sudah selesai, apa yang belum, dan siapa mengerjakan apa."

<div class="segment"></div>

## Bagian 2 — Naskah per Orang

### 🟦 ZUHDI — A1 Data & Praproses (2,5 menit)

**File yang dibuka (urut):**

1. `konteks.md`
2. `research/a1-data-preprocessing/noise.py`
3. `research/a1-data-preprocessing/evaluate_subject_independent.py`
4. `reports/a1-subject-independent/subject_independent_summary.csv`
5. `research/a1-data-preprocessing/diagnose_identity.py`
6. `reports/a1-subject-independent/identity_summary.csv`
7. `docs/bab3-dataset-dan-praproses.md`

**Naskah:**

**Buka `konteks.md`** — "Ini ringkasan satu halaman proyek. Pipeline-nya: citra mentah dipraproses di A1, diekstrak fiturnya di A2 jadi 680 dimensi, lalu diklasifikasi di A3, dan ditampilkan di A4. Dataset yang dipakai CK+ 981 citra 118 subjek, dan JAFFE 213 citra 10 subjek, keduanya 7 kelas."

**Buka `noise.py`, scroll ke fungsi `add_gaussian_noise`** — "Di sini saya buat uji ketahanan noise Gaussian sesuai jurnal bagian 5.5. Noise ditambahkan di ruang intensitas 0 sampai 1, lalu di-clip dan dikembalikan ke uint8. Variansinya 0,01 sampai 0,10. Artefaknya 8 file: 2 dataset kali 4 variansi, ada di folder `data/a1-noise/`. Yang penting: hasilnya deterministik karena seed-nya 42, jadi bisa direproduksi."

**Buka `evaluate_subject_independent.py`, scroll ke fungsi `make_folds`** — "Ini temuan utama saya. Saya bandingkan dua protokol pembagian data. Protokol pertama split acak, protokol kedua split per subjek pakai StratifiedGroupKFold, jadi seluruh citra satu orang hanya ada di train atau hanya di test. Kodenya juga punya assert yang memastikan tidak ada subjek bocor."

**Buka `subject_independent_summary.csv`** — "Hasilnya di sini. CK+ split acak 96,64 persen, tapi begitu subjek dipisah jatuh ke 48,21 persen. JAFFE 80,31 turun ke 38,22. Selisihnya 48 dan 42 poin persen. Artinya hampir separuh akurasi itu bukan mengenali ekspresi, tapi mengenali orangnya."

**Buka `diagnose_identity.py`, scroll ke docstring bagian atas** — "Karena itu saya bikin diagnosa lanjutan untuk membuktikan penyebabnya. Ada tiga pengukuran: pertama, melatih klasifier menebak SIAPA subjeknya dari fitur 680 dimensi. Kedua, klasifikasi emosi di fitur mentah tanpa LDA. Ketiga, klasifikasi emosi setelah LDA, dengan LDA di-fit per fold supaya tidak bocor."

**Buka `identity_summary.csv`** — "Ini hasilnya. Menebak subjek berhasil 98,58 persen di CK+ padahal peluangnya cuma 1,03 persen. Di JAFFE 99,53 persen padahal peluangnya 10 persen. Jadi fitur kita hampir sepenuhnya menyimpan identitas orang. Lalu yang menarik: tanpa LDA, emosi per subjek bisa 89,90 persen, tapi setelah LDA turun jadi 53,19. Jadi LDA justru membuang informasi ekspresi 36,71 poin persen."

**Buka `docs/bab3-dataset-dan-praproses.md`, scroll ke bagian 3.6** — "Semua ini sudah saya tulis jadi draft Bab 3, lengkap dengan tabel distribusi kelas dan deviasi terhadap jurnal. Bagian 3.6 khusus membahas validitas protokol."

**Penutup** — "Status A1: selesai. Yang tersisa cuma push branch dan buka pull request. Test suite sekarang 50 item, 49 lulus, 1 modul di-skip karena DLL PyWavelets diblokir kebijakan Windows."

**Angka yang WAJIB disebut:** 981 / 118, 213 / 10, 96,64 → 48,21, 80,31 → 38,22, 98,58 / 99,53, 89,90 → 53,19.

<div class="segment"></div>

### 🟩 JIRO — A2 Feature Engineering (2 menit)

**File yang dibuka:**

1. `research/a2-feature-engineering/features.py`
2. `reports/a2-features/phog_levels_visualization.png`
3. `research/pipeline/02-a2-feature-extraction.ipynb`
4. `data/a2-features/ck_features.npz` (jalankan potongan Python untuk lihat shape)

**Naskah:**

**Buka `features.py`, scroll ke fungsi `swt_haar_level1`** — "A2 tugasnya mengubah citra 128 kali 128 jadi vektor fitur. Langkah pertama Stationary Wavelet Transform Haar level 1. Bedanya dengan wavelet biasa, SWT tidak melakukan downsampling, jadi sifatnya shift-invariant. Hasilnya empat subband: A1 aproksimasi, H1 horizontal, V1 vertikal, D1 diagonal. Yang dipakai cuma A1 karena memuat struktur bentuk wajah frekuensi rendah."

**Scroll ke fungsi `phog`** — "Langkah kedua PHOG, Pyramid of Histograms of Oriented Gradients. Gradien dihitung pakai Sobel 3 kali 3, lalu dibagi ke piramida spasial level 0 sampai 3. Jumlah selnya 1 tambah 4 tambah 16 tambah 64 sama dengan 85 sel. Tiap sel punya 8 bin orientasi 0 sampai 360 derajat. Jadi totalnya 85 kali 8 sama dengan 680 dimensi per citra. Ini persis rumus persamaan 2 di jurnal."

**Buka `phog_levels_visualization.png`** — "Ini visualisasi piramida PHOG-nya. Bisa dilihat level 0 satu sel, level 1 empat sel, makin ke level 3 makin detail. Normalisasinya L2 per level, bukan global, supaya tiap resolusi punya kontribusi seimbang."

**Jalankan di terminal:**

```
python -c "import numpy as np; d=np.load('data/a2-features/ck_features.npz'); print(d['F'].shape, d['F'].dtype)"
```

"Hasilnya 981 kali 680 float32 untuk CK+, dan 213 kali 680 untuk JAFFE. Tidak ada NaN atau Inf, sudah saya test."

**Buka notebook `02-a2-feature-extraction.ipynb`** — "Notebook ini menjelaskan teorinya dan memanggil modul, tidak menyalin logika. Jadi kalau ada perbaikan di `features.py`, semua notebook otomatis ikut benar."

**Penutup** — "Satu hal yang saya tekankan: modul ini punya fallback SWT murni numpy. Jadi kalau PyWavelets diblokir kebijakan Windows, ekstraksi fitur tetap jalan dengan hasil identik sampai 1e-12. Yang belum selesai dari saya: ekstraksi fitur dari 8 artefak citra ber-noise. Itu saya targetkan selesai dulu karena menahan evaluasi noise di A3."

**Angka yang WAJIB disebut:** 128×128 → 4 subband, 85 sel, 8 bin, 680 dimensi, 981×680, 213×680.

<div class="segment"></div>

### 🟨 SANDY — A3 Model & Optimasi (3 menit)

**File yang dibuka:**

1. `research/a3-model-optimization/lssvm.py`
2. `research/a3-model-optimization/baseline.py`
3. `reports/a3-models/baseline_summary.csv`
4. `reports/a3-models/lssvm_multiclass_summary.csv`
5. `reports/a3-models/figures/lssvm_toy_xor_iris.png`
6. `config.yaml`, scroll ke bagian `grid_search`
7. `reports/a3-models/figures/lssvm_confusion_matrix_ck.png`

**Naskah:**

**Buka `lssvm.py`, scroll ke kelas `BinaryLSSVM` / fungsi `solve` KKT** — "A3 tugasnya membangun classifier. LS-SVM bedanya dengan SVM biasa: kalau SVM pakai pertidaksamaan margin, LS-SVM mengubahnya jadi persamaan, sehingga solusinya jadi sistem persamaan linear KKT, bukan quadratic programming. Matriksnya dipecahkan pakai `scipy.linalg.solve`, dan di unit test saya verifikasi residualnya di bawah 1e-5."

**Scroll ke `OneVsOneLSSVMClassifier`** — "Karena ada 7 kelas, saya pakai strategi One-vs-One. Jumlah classifier binernya 7 kali 6 dibagi 2 sama dengan 21. Prediksi akhirnya majority voting, kalau seri pakai confidence margin sebagai penentu."

**Buka `lssvm_toy_xor_iris.png`** — "Sebelum dipakai ke data wajah, saya uji dulu di data mainan. XOR itu kasus non-linear, akurasinya 98 persen. Iris Setosa vs Versicolor 100 persen, identik dengan SVC scikit-learn. Ini bukti implementasi LS-SVM saya benar."

**Buka `baseline_summary.csv`** — "Baseline-nya LDA plus SVM-RBF dari scikit-learn, 5-fold stratified cross-validation. CK+ 97,66 persen, JAFFE 81,27 persen."

**Buka `lssvm_multiclass_summary.csv`** — "Kalau pakai LS-SVM OvO dengan grid search, CK+ 96,64 persen, JAFFE 79,36 persen. Sedikit di bawah baseline, tapi ini implementasi sendiri, bukan library."

**Buka `config.yaml`, scroll ke `grid_search`** — "Grid search-nya gamma 1 sampai 500, sigma 1 sampai 5, total 25 kombinasi. Dan di sini ada temuan: di CK+ semua fold memilih gamma 1 dan sigma 5 dengan akurasi CV internal 100 persen. Artinya ruang grid-nya terlalu sempit atau CV internalnya saturasi. Ini justru jadi alasan kuat kenapa optimasi metaheuristik seperti Jaya perlu dikerjakan."

**Buka `lssvm_confusion_matrix_ck.png`** — "Ini confusion matrix-nya. Bisa dilihat kelas contempt paling sering salah, karena datanya cuma 54 citra, paling sedikit."

**Penutup — bagian paling penting, harus jujur** — "Yang belum selesai dari A3 ada lima. Pertama dan paling utama: algoritma Jaya belum ada. Judul TA kita Jaya-optimised LS-SVM, tapi optimasinya masih grid search. Kedua, kurva konvergensi optimizer, itu ikut Jaya. Ketiga, pembanding PSO dan GA. Keempat, evaluasi akurasi pada data ber-noise, ini nunggu fitur noise dari A2. Kelima, ekspor model untuk A4. Prioritas saya: Jaya dulu."

**Angka yang WAJIB disebut:** 21 classifier, 97,66 / 81,27, 96,64 / 79,36, 25 kombinasi, 98% XOR, 100% Iris.

> ⚠️ **Jangan bilang "Jaya sudah selesai".** Itu bagian yang paling mungkin ditanya penguji.

<div class="segment"></div>

### 🟥 NAILA — A4 Web & Integrasi (1,5 menit)

**Kondisi sebenarnya:** `app/` hanya berisi `README.md`. Belum ada satu baris kode pun. Lihat Bagian 3 untuk cara menanganinya.

**Kalau pakai opsi "jujur + rencana" (Opsi A), naskahnya:**

**Buka `app/README.md`** — "A4 tugasnya menyajikan sistem dan hasilnya lewat web. Folder `app/` sengaja saya pisahkan dari `research/` supaya kode penelitian tidak tercampur dengan kode aplikasi. Rencananya: `main.py` sebagai entry point, `templates/` untuk HTML, `static/` untuk CSS dan JS vanilla tanpa CDN supaya bisa jalan offline, dan `inference.py` untuk memuat model dan melakukan prediksi."

**Scroll ke tabel "Bahan yang dibutuhkan dari A1–A3"** — "Dari sisi bahan, fitur sudah tersedia dari A2, metrik dan hasil sudah tersedia dari A3. Yang masih kurang satu: parameter model LDA dan LS-SVM belum diekspor oleh A3. Jadi A4 memang menunggu artefak model."

**Penutup** — "Status A4: belum ada implementasi. Rencana saya mulai pekan ini dengan API skeleton dan UI upload memakai model dummy dulu, supaya kontrak datanya selesai lebih awal dan tidak menunggu model final."

**Kalau pakai opsi "demo skeleton" (Opsi B), tambahkan:** buka `http://localhost:8000`, tunjukkan halaman upload, upload satu gambar, tunjukkan respons JSON dummy.

## Bagian 3 — Masalah Naila (ini yang perlu kamu putuskan)

Naila belum mengerjakan apa pun, dan `app/` kosong. Ada tiga cara menanganinya:

### Opsi A — Jujur: tampilkan rencana saja

Naila tampil dengan `app/README.md` dan menjelaskan rencana + kontrak data.

- ✅ Cepat, tidak ada risiko
- ❌ Di video progres, divisi 20% terlihat kosong. Penguji/PA pasti tanya "kenapa belum jalan"

### Opsi B — Sprint 1–2 hari, bikin skeleton minimal (saya rekomendasikan ini)

Naila bikin tiga hal saja sebelum rekam:

1. `app/main.py` — FastAPI dengan `/health` dan `/predict` yang mengembalikan dummy JSON
2. `app/templates/index.html` + `app/static/` — halaman upload dan preview
3. `app/inference.py` — loader yang nanti membaca `model_v1.npz`

- ✅ Saat rekam, dia bisa demo langsung di browser. Ini mengubah "belum mulai" jadi "sudah jalan dengan model dummy"
- ✅ Tidak perlu menunggu Sandy, karena memang dummy
- ✅ Sesuai rencana yang sudah ditulis di `app/README.md` sendiri
- ❌ Butuh 1–2 hari kerja Naila

### Opsi C — Tukar tugas

A4 dipindah ke kamu (A1 sudah selesai, jadi kamu punya waktu), Naila ambil dokumentasi dan penyusunan laporan.

- ✅ A4 lebih terjamin selesai
- ❌ Perlu kesepakatan ulang pembagian bobot, dan kalau ini tugas kelompok resmi biasanya tidak boleh diubah sepihak

**Rekomendasi:** Opsi B, dengan Opsi A sebagai cadangan. Kalau sampai waktu rekam Naila tetap belum mengerjakan apa pun, jangan dikarang — pakai naskah Opsi A dan di slide status tulis apa adanya "belum mulai, target pekan ini". Video progres yang jujur lebih aman daripada yang terlihat lengkap tapi bohong.

**Satu hal yang perlu kamu sampaikan ke Naila:** dia tidak diblokir total. Satu-satunya yang menunggu Sandy adalah model asli. API skeleton, UI, dan kontrak JSON bisa dikerjakan sekarang.

## Bagian 4 — Slide Status Jujur (untuk penutup video)

### Sudah selesai

| Divisi | Yang selesai | Bukti file |
|:--:|---|---|
| A1 | Praproses + artefak `.npz` | `data/a1-preprocessing/` |
| A1 | Noise Gaussian 8 artefak | `data/a1-noise/` |
| A1 | Split acak vs per-subjek | `reports/a1-subject-independent/subject_independent_summary.csv` |
| A1 | Diagnosa identitas subjek | `reports/a1-subject-independent/identity_summary.csv` |
| A1 | Draft Bab 3 | `docs/bab3-dataset-dan-praproses.md` |
| A2 | SWT + PHOG 680-D | `data/a2-features/*.npz` |
| A2 | Visualisasi PHOG | `reports/a2-features/phog_levels_visualization.png` |
| A3 | Baseline LDA+SVM-RBF | `reports/a3-models/baseline_summary.csv` |
| A3 | LS-SVM OvO 21 classifier | `reports/a3-models/lssvm_multiclass_summary.csv` |
| A3 | Uji data mainan XOR/Iris | `reports/a3-models/figures/lssvm_toy_xor_iris.png` |
| Semua | 50 test, 49 lulus | `tests/` |

### Belum selesai (urut prioritas)

| # | Item | Divisi | Blokir siapa |
|:--:|---|:--:|---|
| 1 | Algoritma Jaya | A3 | — (inti judul TA) |
| 2 | Fitur dari citra ber-noise | A2 | A3 evaluasi noise |
| 3 | Evaluasi akurasi data ber-noise | A3 | Bab 4 |
| 4 | Kurva konvergensi optimizer | A3 | Bab 4 |
| 5 | Ekspor model untuk A4 | A3 | A4 model nyata |
| 6 | Pembanding PSO & GA | A3 | Bab 4 |
| 7 | Web A4 (belum mulai) | A4 | — |
| 8 | Bab 2, 4, 5, 6 | semua | laporan |

**Kalimat penutup yang bisa kamu baca:**

> "Kesimpulannya, A1 dan A2 selesai, A3 selesai untuk tahap baseline dan LS-SVM multiclass, A4 belum mulai. Prioritas berikutnya: algoritma Jaya karena itu inti judul, lalu fitur noise supaya evaluasi ketahanan bisa jalan, lalu web. Satu branch A1 masih perlu di-push dan di-PR."

## Bagian 5 — Checklist Sebelum Rekam

- ☐ Semua orang sudah buka repo di VS Code dengan font editor diperbesar
- ☐ Tutup notifikasi Windows (mode fokus)
- ☐ Panel terminal siap: `git status`, `python -c "..."` untuk demo shape
- ☐ Rekam per segmen, jangan sekali jalan — gampang mengulang kalau salah
- ☐ Sebelum rekam, jalankan `python -m pytest -q` sekali supaya kalau ditanya "test-nya lulus?" kamu sudah tahu jawabannya (49 passed, 1 skipped)
- ☐ Putuskan dulu Opsi A/B/C untuk Naila sebelum hari rekaman

## Catatan terpisah: soal error SECRET_KEY

Itu bukan dari repo TA-PCVK. Saya sudah lacak, sumbernya di proyek lain:

```
D:\Semester5\PBL-MOBILE\App\Web\compose.yaml
  → SECRET_KEY: ${SECRET_KEY:?Set SECRET_KEY in .env}
```

**Penyebabnya:** `compose.yaml` mewajibkan variabel `SECRET_KEY`, tapi file `.env` tidak ada (hanya `.env.example`). Di proyek itu sudah ada `.github/workflows/web.yml` yang seharusnya membuat `.env` otomatis — jadi kemungkinan workflow-nya tidak jalan atau `.env` tidak terbaca.

**Perbaikan cepat** (di folder `PBL-MOBILE\App\Web`):

```powershell
Copy-Item .env.example .env
# lalu isi SECRET_KEY dan POSTGRES_PASSWORD dengan nilai acak berbeda
```
