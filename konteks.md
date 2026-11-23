# Konteks Proyek TA-PCVK

Ringkasan satu halaman untuk anggota baru, dosen pembimbing, atau siapa pun yang
baru membuka repo ini. Detail lengkap ada di `README.md`.

---

## 1. Apa yang dikerjakan

Replikasi jurnal:

> Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021). *A hybrid feature
> descriptor with Jaya optimised least squares SVM for facial expression
> recognition.* IET Image Processing, 15(7), 1471–1483.
> DOI: [10.1049/ipr2.12118](https://doi.org/10.1049/ipr2.12118)

Tujuan: mengenali **7 kelas ekspresi wajah** dari citra wajah memakai
deskriptor hybrid **SWT Haar + PHOG**, reduksi dimensi **LDA**, dan classifier
**LS-SVM** yang hyperparameternya dioptimasi **algoritma Jaya** (JOLS-SVM).

## 2. Pipeline

```
Citra mentah
  → [A1] grayscale → resize 128×128 → histogram equalization   → data/a1-preprocessing/
  → [A2] SWT Haar level-1 (A1) → PHOG L=3 K=8                  → data/a2-features/  (680-D)
  → [A3] LDA (C−1 komponen) → LS-SVM OvO + tuning              → reports/a3-models/
  → [A1] uji noise Gaussian & evaluasi subject-independent      → reports/a1-subject-independent/
```

## 3. Dataset

| Dataset | Citra | Kelas | Subjek | Ukuran | Lokasi |
|---|---|---|---|---|---|
| CK+ | 981 (327 sekuens × 3 frame) | 7 (tanpa neutral) | 118 | 48×48 | `../dataset/ck+/` |
| JAFFE | 213 | 7 | 10 | 128×128 | `../dataset/jaffe/` |

Dataset **tidak** disertakan di repo (lisensi non-komersial). Atur lokasi lewat
env `FER_DATASET_DIR` bila berbeda dari default `../dataset`.

> **Catatan penting:** dataset kita **berbeda dari jurnal** (jurnal: CK+ 450
> citra/8 kelas). Karena itu angka akurasi **tidak boleh** diklaim setara
> jurnal; jurnal hanya dipakai sebagai kolom pembanding.

## 4. Struktur folder

| Folder | Isi |
|---|---|
| `research/a1-data-preprocessing/` | modul `.py` A1: noise, evaluasi split, diagnosa identitas |
| `research/a2-feature-engineering/` | modul `.py` A2: SWT + PHOG |
| `research/a3-model-optimization/` | modul `.py` A3: LS-SVM, baseline, evaluator |
| `research/pipeline/` | **semua notebook**, termasuk `00-pipeline-lengkap.ipynb` |
| `data/a1-preprocessing/` · `data/a1-noise/` · `data/a2-features/` | artefak biner |
| `reports/a1-…` · `a2-…` · `a3-…` | hasil per divisi |
| `docs/` | arsitektur, metodologi, ringkasan hasil |
| `tests/` | unit test seluruh divisi |
| `app/` | **kosong** — disiapkan untuk A4 (web) |

## 5. Pembagian divisi

| Kode | Cakupan | Status |
|---|---|---|
| **A1** | Data, praproses, noise Gaussian, evaluasi subject-independent, diagnosa identitas | ✅ selesai |
| **A2** | SWT Haar + PHOG 680-D | ✅ selesai |
| **A3** | LDA, baseline, LS-SVM OvO, grid search | ✅ Minggu 12–13; Minggu 14 (Jaya, PSO/GA) belum |
| **A4** | Web (HTML/CSS/JS) | ⏳ belum |

## 6. Hasil utama

**Baseline & LS-SVM (split acak):**

| Dataset | LDA + SVM-RBF | LDA + OvO LS-SVM (grid search) |
|---|---|---|
| CK+ | 97,66% | 96,64% |
| JAFFE | 81,27% | 79,36% |

**Split acak vs subject-independent (temuan A1):**

| Dataset | Split acak | Split per-subjek | Selisih |
|---|---|---|---|
| CK+ | 96,64% | **48,21%** | 48,43 poin |
| JAFFE | 80,31% | **38,22%** | 42,09 poin |

Interpretasi: CK+ memuat 3 frame berurutan per sekuens, sehingga split acak
menempatkan frame se-subjek di train *dan* test. Hampir separuh akurasi berasal
dari pengenalan identitas, bukan ekspresi.

**Diagnosa identitas (A1) — mengapa akurasinya jatuh:**

| Pengukuran | CK+ | JAFFE |
|---|---|---|
| Menebak subjek dari fitur 680-D (peluang: 1,03% / 10,00%) | **98,58%** | **99,53%** |
| Emosi per subjek — fitur mentah (tanpa LDA) | **89,90%** | **57,68%** |
| Emosi per subjek — LDA nested | 53,19% | 36,00% |
| Penalti LDA | 36,71 poin | 21,69 poin |

Kesimpulan: fitur hybrid hampir sepenuhnya memuat identitas, dan **LDA merugikan**
pada protokol jujur. Detail di `docs/hasil.md` §5.

## 7. Perintah penting

```powershell
.\run.ps1 test        # seluruh unit test
.\run.ps1 pipeline    # notebook pipeline lengkap
.\run.ps1 noise       # artefak noise Gaussian
.\run.ps1 split       # split acak vs per-subjek
.\run.ps1 identity    # diagnosa kebocoran identitas subjek
.\run.ps1 report      # tampilkan tabel hasil
.\run.ps1 check       # diagnosa DLL yang diblokir
```

## 8. Kendala lingkungan

Kebijakan *Application Control* (WDAC / Device Guard) pada Windows dapat
memblokir DLL biner Python secara dinamis. Yang pernah teramati: `pywt`,
`cv2`, `_imaging`, `_pydantic_core`, `_sparsetools`, `pandas._libs._ujson`,
`sklearn._gradient_boosting`.

Dampaknya:

| Modul diblokir | Yang tidak bisa dijalankan |
|---|---|
| `pywt` | `tests/test_features.py` (di-skip otomatis) |
| `cv2` | pembacaan citra di notebook A1 |
| `pandas` (`_ujson`) | **`import pandas` gagal total** — CSV ditulis dengan modul `csv` bawaan |
| `sklearn` (`_gradient_boosting`) | `RandomForestClassifier` — dilewati otomatis di diagnosa identitas |

**Penting:** modul `features.py` sudah punya **fallback SWT Haar murni numpy**
yang identik dengan PyWavelets (selisih < 1e-12). Jadi bila `pywt` diblokir,
notebook A2 dan ekstraksi fitur **tetap berjalan normal** — hanya test yang
menguji jalur pywt secara eksplisit yang di-skip.

Diagnosa cepat: `.\run.ps1 check`
