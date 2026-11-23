# Arsitektur Sistem

## 1. Gambaran umum

Sistem ini adalah **replikasi penelitian** (bukan aplikasi produksi) untuk
pengenalan ekspresi wajah. Alurnya linier: citra → praproses → fitur →
reduksi dimensi → klasifikasi → evaluasi.

```
┌─────────────────────────────────────────────────────────────────────┐
│  A1 · Data & Praproses                                              │
│                                                                     │
│  dataset/ck+/*.png (981)      dataset/jaffe/*.png (213)             │
│            │                          │                             │
│            └──────────┬───────────────┘                             │
│                       ▼                                             │
│        grayscale → resize 128×128 → histogram equalization          │
│                       │                                             │
│                       ▼                                             │
│        data/a1-preprocessing/{ck,jaffe}_preprocessed.npz            │
│        kunci: X, y, subjects, class_names                           │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────────┐
│  A2 · Feature Engineering                                           │
│                                                                     │
│        SWT Haar level-1  →  subband A1 (128×128)                    │
│                       ▼                                             │
│        PHOG L=3, K=8, 0–360°, magnitudo Sobel 3×3                   │
│                       ▼                                             │
│        F = K · Σ(l=0..3) 4^l = 8 · 85 = 680 dimensi                  │
│                       │                                             │
│                       ▼                                             │
│        data/a2-features/{ck,jaffe}_features.npz                     │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────────┐
│  A3 · Model & Optimasi                                              │
│                                                                     │
│        LDA (fit di fold latih) → C−1 komponen (6)                   │
│                       ▼                                             │
│        StandardScaler                                               │
│                       ▼                                             │
│        ┌──────────────┴──────────────┐                              │
│        ▼                             ▼                              │
│   SVM-RBF (baseline)          LS-SVM One-vs-One (21 model)          │
│   scikit-learn                + grid search (γ, σ)                  │
│        │                             │                              │
│        └──────────────┬──────────────┘                              │
│                       ▼                                             │
│        reports/a3-models/*.json + confusion matrix                  │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────────┐
│  A1 · Uji Ketahanan & Validasi Protokol                             │
│                                                                     │
│  noise Gaussian var 0.01–0.1   →   data/a1-noise/*.npz              │
│  split acak vs subject-independent → reports/a1-subject-independent │
│  diagnosa identitas subjek     →   reports/.../identity_*.json       │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Pemisahan tanggung jawab

| Lapisan | Folder | Isi |
|---|---|---|
| Modul (logika) | `research/a{1,2,3}-*/` | fungsi & kelas `.py`, dapat diimpor |
| Orkestrasi (narasi) | `research/pipeline/` | notebook: penjelasan + pemanggilan modul |
| Artefak (data) | `data/a1-*/`, `data/a2-*/` | `.npz`, tidak di-commit bila besar |
| Hasil (laporan) | `reports/a1-*/`, `reports/a2-*/`, `reports/a3-*/` | JSON, CSV, PNG |
| Verifikasi | `tests/` | pytest, 50 item (49 lulus, 1 modul di-skip) |
| Aplikasi | `app/` | disiapkan untuk A4 |

**Prinsip:** notebook **tidak menyalin logika**. Semua perhitungan ada di modul
`.py`, sehingga perbaikan pada modul otomatis berlaku di semua notebook dan
dapat diuji dengan pytest.

## 3. Format artefak

Semua artefak memakai `.npz` (bukan pickle) agar portabel dan aman.

| Artefak | Kunci | Shape |
|---|---|---|
| `*_preprocessed.npz` | `X`, `y`, `subjects`, `class_names` | CK+ (981,128,128) uint8 · JAFFE (213,128,128) uint8 |
| `*_features.npz` | `F`, `y`, `subjects`, `class_names` | CK+ (981,680) float32 · JAFFE (213,680) float32 |
| `*_noise_v{v}.npz` | idem praproses + `variance`, `seed`, `source` | sama dengan praproses |

Kunci `subjects` inilah yang memungkinkan evaluasi subject-independent.

Hasil diagnosa identitas (`identity_{ck,jaffe}.json`) memakai struktur:

| Kunci | Isi |
|---|---|
| `subject_identification` | akurasi menebak subjek (split acak) + `chance` + `excess_over_chance` |
| `raw_features` | akurasi emosi pada fitur mentah per klasifier (protokol per subjek) |
| `lda_nested` | akurasi emosi setelah LDA di-fit per fold |
| `skipped_classifiers` | klasifier yang tidak dapat dimuat di lingkungan tersebut |
| `summary` | `lda_penalty_points`, `identity_dominates_expression`, dll |

## 4. Alur data antar divisi

```
A1 ──(data/a1-preprocessing/*.npz)──▶ A2
A1 ──(data/a1-noise/*.npz)──────────▶ A2 (opsional, untuk uji noise)
A2 ──(data/a2-features/*.npz)───────▶ A3
A3 ──(reports/a3-models/*.json)─────▶ A4 (web)
A1 ──(reports/a1-subject-independent/)──▶ laporan
```

Setiap serah-terima punya kunci yang terdokumentasi, sehingga divisi
berikutnya tidak perlu menjalankan ulang tahap sebelumnya.
