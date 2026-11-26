# Ringkasan Hasil

Semua angka dihasilkan oleh kode di repo ini dan dapat direproduksi dengan
`.\run.ps1 test` + `.\run.ps1 report`. Dataset kita **berbeda dari jurnal**
(CK+ 981 citra/7 kelas vs 450/8 kelas), sehingga angka jurnal hanya dipakai
sebagai kolom pembanding — bukan target.

---

## 1. Baseline: LDA + SVM-RBF

Sumber: `reports/a3-models/baseline_summary.csv`
Perintah: `.\run.ps1 baseline`

| Dataset | Sampel | Fitur | Akurasi (%) | Std (%) | Macro Sen (%) | Macro Spe (%) |
|---|---|---|---|---|---|---|
| CK+ | 981 | 680 | **97,66** | 1,00 | 97,49 | 99,59 |
| JAFFE | 213 | 680 | **81,27** | 8,37 | 81,15 | 96,87 |

## 2. LS-SVM One-vs-One + grid search

Sumber: `reports/a3-models/lssvm_multiclass_summary.csv`
Perintah: `.\run.ps1 lssvm`

| Dataset | Model | Akurasi (%) | Std (%) | Macro Sen (%) | Macro Spe (%) |
|---|---|---|---|---|---|
| CK+ | LDA + OvO LS-SVM | **96,64** | 0,94 | 97,02 | 99,44 |
| JAFFE | LDA + OvO LS-SVM | **79,36** | 7,52 | 79,28 | 96,56 |

Parameter terbaik per fold (CK+): seluruh fold memilih γ=1,0 dan σ=5,0 dengan
akurasi CV internal 100% — indikasi ruang grid terlalu sempit dan/atau CV
internal saturasi. Ini salah satu alasan optimasi metaheuristik (Jaya) masih
perlu dikerjakan di A3.

## 3. Split acak vs subject-independent

Sumber: `reports/a1-subject-independent/subject_independent_summary.csv`
Perintah: `.\run.ps1 split`

| Dataset | Split | Akurasi (%) | Std (%) | Macro Sen (%) | Macro Spe (%) |
|---|---|---|---|---|---|
| CK+ | acak | **96,64** | 0,94 | 97,02 | 99,44 |
| CK+ | **per-subjek** | **48,21** | 4,23 | 45,71 | 91,44 |
| JAFFE | acak | **80,31** | 8,23 | 80,20 | 96,71 |
| JAFFE | **per-subjek** | **38,22** | 10,54 | 38,59 | 89,76 |

**Selisih: CK+ 48,43 poin persen · JAFFE 42,09 poin persen.**

### Interpretasi

1. Angka CK+ split acak (96,64%) **identik** dengan hasil A3 Minggu 13 —
   mengonfirmasi implementasi konsisten dengan pipeline A3.
2. Namun hampir **separuh** akurasi tersebut hilang ketika subjek benar-benar
   dipisah. Artinya model sebagian besar belajar **identitas orang**, bukan
   ekspresi.
3. Penyebab struktural: CK+ memuat 327 sekuens × **tepat 3 frame berurutan**
   per subjek. Split acak menempatkan frame se-subjek di train *dan* test.
4. JAFFE juga terdampak karena setiap subjek menyumbang ketujuh ekspresinya.

**Rekomendasi pelaporan:** tampilkan kedua protokol. Split acak untuk
perbandingan dengan jurnal; **subject-independent** untuk klaim generalisasi.

## 4. Uji ketahanan noise

Artefak: `data/a1-noise/{ck,jaffe}_noise_v{0.01,0.02,0.05,0.10}.npz` (8 file)
Perintah: `.\run.ps1 noise`

Artefak tidak di-commit (`.gitignore`: `data/a1-noise/*.npz`, ~70 MB) karena
dapat diregenerasi deterministik dengan `seed=42`.

### 4.1 Status per tahap

| Tahap | Penanggung jawab | Status | Bukti |
|---|---|---|---|
| A1 — bangun artefak citra ber-noise | A1 | ✅ selesai | 8 file `.npz`, kunci `X, y, subjects, class_names, variance, seed, source` |
| A1 — verifikasi statistik noise | A1 | ✅ selesai | tabel 4.2 di bawah |
| A2 — ekstraksi fitur 680-D dari citra ber-noise | A2 | ✅ selesai | 8 file `.npz` di `data/a2-features/*_features_noise_v*.npz` |
| A3 — evaluasi akurasi pada data ber-noise | A3 | ✅ selesai | `reports/a3-models/table14_noise_robustness.csv` |

### 4.2 Verifikasi statistik

σ empiris dihitung dari selisih `X_noise − X_bersih` pada ruang intensitas
`[0,1]`.

| Variansi | σ teori | σ empiris CK+ (n=981) | Max \|diff\| CK+ | σ empiris JAFFE (n=213) | Max \|diff\| JAFFE |
|---|---|---|---|---|---|
| 0,01 | 0,1000 | 0,0936 | 0,529 | 0,0948 | 0,518 |
| 0,02 | 0,1414 | 0,1293 | 0,753 | 0,1306 | 0,718 |
| 0,05 | 0,2236 | 0,1941 | 1,000 | 0,1954 | 0,980 |
| 0,10 | 0,3162 | 0,2566 | 1,000 | 0,2578 | 1,000 |

σ empiris lebih kecil dari teori karena dua sebab: (1) clipping ke `[0,1]`
membuang ekor distribusi, dan (2) kuantisasi ke 256 level. Efeknya makin
terasa pada variansi besar (0,10), di mana deviasi mencapai ~19%.

`variance=0` menghasilkan salinan **identik** — dipakai sebagai kontrol di
`tests/test_noise.py`.

### 4.3 Angka jurnal sebagai rujukan (Tabel 14)

| Kondisi | CK+ Sen / Spe / Acc | JAFFE Sen / Spe / Acc |
|---|---|---|
| Tanpa noise | 94,08 / 99,51 / 98,38 | 95,23 / 99,61 / 98,99 |
| Dengan noise | 95,95 / 96,86 / **97,17** | 91,50 / 98,59 / **97,59** |

> Pada jurnal, noise **CK+ = 97,17%** dan **JAFFE = 97,59%**. Keduanya berbeda.

### 4.4 Hasil Replikasi Tabel 14 (Evaluasi Riil Fitur Ber-noise A2)

Sumber: `reports/a3-models/table14_noise_robustness.csv` dan `.json`
Perintah: `python research/a3-model-optimization/evaluate_noise.py`

| Variansi Noise | Akurasi CK+ (%) | Std CK+ (%) | Akurasi JAFFE (%) | Std JAFFE (%) |
|---|---|---|---|---|
| 0.00 (Bersih) | **95,62** | 1,60 | **79,86** | 7,79 |
| 0.01 | 19,47 | 3,56 | 30,17 | 14,12 |
| 0.02 | 16,21 | 5,89 | 24,93 | 9,35 |
| 0.05 | 14,47 | 6,44 | 19,76 | 6,55 |
| 0.10 | 14,47 | 6,63 | 17,84 | 4,31 |

> **Catatan Analisis Kritis:** Pada ekstraksi fitur riil (SWT Haar + PHOG) dari citra yang diberi noise Gaussian secara langsung tanpa denosing filter, akurasi klasifikasi mengalami penurunan drastis karena gradien tepi PHOG terdistorsi oleh bintik-bintik derau frekuensi tinggi. Hal ini merupakan temuan empiris penting bahwa model tanpa preprocessing denoising sangat rentan terhadap derau Gaussian riil.

## 5. Diagnosa identitas subjek

Sumber: `reports/a1-subject-independent/identity_{ck,jaffe}.json`
Perintah: `.\run.ps1 identity`

Mengukur **penyebab** anjloknya akurasi pada §3: seberapa besar vektor fitur
680-D menyimpan identitas orang, bukan ekspresi.

### 5.1 Hasil

| Pengukuran | CK+ (118 subjek) | JAFFE (10 subjek) |
|---|---|---|
| Menebak **subjek** dari fitur (split acak) | **98,58%** | **99,53%** |
| Peluang acak menebak subjek | 1,03% | 10,00% |
| Emosi — fitur mentah, tanpa LDA (per subjek) | **89,90%** (logreg) | **57,68%** (logreg) |
| Emosi — LDA nested, 6 komponen (per subjek) | 53,19% | 36,00% |
| **Penalti LDA** | **36,71 poin** | **21,69 poin** |

Pembanding klasifier pada fitur mentah (protokol per subjek):

| Klasifier | CK+ | JAFFE |
|---|---|---|
| LogisticRegression | **89,90%** | **57,68%** |
| SVC RBF (C=10) | 81,18% | 46,57% |
| Random Forest | dilewati — DLL diblokir | dilewati — DLL diblokir |

### 5.2 Interpretasi

1. **Fitur hampir sepenuhnya memuat identitas.** Menebak *siapa* orangnya
   berhasil 98,58% (CK+) dan 99,53% (JAFFE), padahal peluangnya hanya
   1,03% dan 10,00%. Fitur hybrid SWT+PHOG memang deskriptor bentuk wajah —
   bentuk wajah per orang jauh lebih menonjol daripada ekspresinya.
2. **LDA justru merugikan pada protokol jujur.** Tanpa LDA, akurasi emosi
   per subjek mencapai 89,90% (CK+), sedangkan dengan LDA nested hanya
   53,19%. Artinya LDA membuang **36,71 poin persen** informasi ekspresi.
   Untuk JAFFE penaltinya 21,69 poin.
3. **Ini menjelaskan angka §3.** Pipeline A3 memakai LDA → 48,21% (CK+).
   Tanpa LDA, batas atasnya 89,90%. Jadi kejatuhan 96,64% → 48,21% bukan
   semata kebocoran identitas; sebagian disebabkan LDA yang di-fit per fold
   mempertahankan arah diskriminan identitas, bukan ekspresi.
4. **Rekomendasi untuk A3:** laporkan protokol subject-independent **dengan
   dan tanpa LDA**. Klaim generalisasi memakai angka tanpa LDA; angka
   dengan LDA tetap ditampilkan sebagai pembanding setia terhadap jurnal.

> Catatan metodologis: tugas "tebak subjek" sengaja memakai split **acak**,
> bukan split per subjek. Dengan split per subjek, subjek di data uji tidak
> pernah ada di data latih sehingga akurasinya selalu ~0% — itu mengukur
> generalisasi ke orang baru, bukan kebocoran yang ingin dibuktikan.

## 6. Status Pengerjaan Seluruh Divisi

| Item | Divisi | Status | Keterangan / Artefak |
|---|---|---|---|
| Praproses citra & normalisasi (A1) | A1 | ✅ Selesai | `data/a1-preprocessing/*.npz` |
| Artefak citra ber-noise Gaussian (A1) | A1 | ✅ Selesai | 8 file di `data/a1-noise/` |
| Ekstraksi fitur SWT + PHOG 680-D (A2) | A2 | ✅ Selesai | `data/a2-features/*.npz` |
| Ekstraksi fitur dari citra ber-noise (A2) | A2 | ✅ Selesai | 8 file di `data/a2-features/*_noise_v*.npz` |
| Algoritma Jaya (A3) | A3 | ✅ Selesai | `research/a3-model-optimization/optimizers.py` |
| Pembanding PSO & GA (A3) | A3 | ✅ Selesai | `table3_optimizer_comparison.csv` |
| Kurva konvergensi optimizer (A3) | A3 | ✅ Selesai | `figures/convergence_ck.png`, `convergence_jaffe.png` |
| Perbandingan Lin/Poly LS-SVM, RF, BPNN (A3) | A3 | ✅ Selesai | `table11_classifier_comparison.csv` |
| Waktu latih/uji (A3) | A3 | ✅ Selesai | `table10_execution_time.csv` |
| Evaluasi akurasi data ber-noise (A3) | A3 | ✅ Selesai | `table14_noise_robustness.csv` |
| Ekspor model terlatih .npz (A3) | A3 | ✅ Selesai | `reports/a3-models/model.npz`, `app/model.npz` |
| Web Deployment / Antarmuka Pengguna (A4) | A4 | ⏳ On Progress | Siap menggunakan `app/model.npz` & `app/inference.py` |
