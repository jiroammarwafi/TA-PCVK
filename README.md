# TA-PCVK — Facial Expression Recognition (Hybrid Feature Descriptor + Jaya-LSSVM)

Replikasi jurnal tugas akhir:

> Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021). *A hybrid feature descriptor with Jaya optimised least squares SVM for facial expression recognition.* IET Image Processing, 15(7), 1471–1483. DOI: [10.1049/ipr2.12118](https://doi.org/10.1049/ipr2.12118) (Open Access, CC-BY)

Mengenali **7 kelas ekspresi wajah** dari dataset **CK+** dan **JAFFE** menggunakan deskriptor hybrid **SWT Haar + PHOG**, reduksi dimensi **LDA**, dan classifier **LSSVM** dengan parameter (γ, σ) yang dioptimasi **algoritma Jaya**, dievaluasi dengan **5-fold stratified cross-validation**.

## Pipeline & Pembagian Tugas

| Kode | Bagian | Cakupan | Status |
|------|--------|---------|--------|
| **A1** | Data & Praproses | Loading dataset (CK+ & JAFFE), EDA, grayscale → resize 128×128 → histogram equalization, artefak `.npz` | ✅ Selesai |
| **A2** | Feature Engineering | SWT Haar level-1 → PHOG pada subband A1 (LL), ekspor fitur 680-D | ✅ Selesai |
| **A3** | Model & Optimasi | Seleksi fitur LDA → classifier Jaya-LSSVM → evaluasi 5-fold stratified CV | 🟡 Minggu 12 Selesai (Minggu 13 On Progress) |
| **A4** | Web & Integrasi | Website penyajian sistem dan hasil | ⏳ Belum dimulai |

```
Citra wajah (CK+ / JAFFE)
   → [A1] Preprocessing       : grayscale, resize 128×128, histogram equalization
   → [A2] SWT Haar → PHOG(LL) : ekstraksi fitur hybrid spatial-frekuensi
   → [A3] LDA → Jaya-LSSVM    : reduksi dimensi + klasifikasi
   → [A4] Web                 : penyajian hasil
```

## Struktur Repository

```
.
├── README.md
├── config.yaml                # Konfigurasi parameter terpusat (DoD A3)
├── requirements.txt           # Dependensi Python
├── pytest.ini                 # Konfigurasi test runner pytest
├── data/                      # Artefak dataset praproses (A1) & matriks fitur (A2)
│   ├── ck_preprocessed.npz    # [A1] Citra praproses CK+ (981, 128, 128)
│   ├── jaffe_preprocessed.npz # [A1] Citra praproses JAFFE (213, 128, 128)
│   ├── ck_features.npz        # [A2] Matriks fitur CK+ (981, 680) float32
│   └── jaffe_features.npz     # [A2] Matriks fitur JAFFE (213, 680) float32
├── feature-engineering/       # [A2] Kode modular & eksperimen ekstraksi fitur
│   ├── __init__.py
│   ├── features.py            # Modul fungsional (swt_haar_level1, phog, extract_features)
│   └── A2-feature-extraction.ipynb # Notebook eksekusi, visualisasi, assert, statistik
├── reports/
│   ├── baseline_summary.csv   # [A3] Tabel ringkasan hasil baseline 5-fold CV
│   ├── baseline_ck_results.json
│   ├── baseline_jaffe_results.json
│   └── figures/
│       ├── phog_levels_visualization.png
│       ├── baseline_confusion_matrix_ck.png   # [A3] Confusion matrix baseline CK+
│       ├── baseline_confusion_matrix_jaffe.png# [A3] Confusion matrix baseline JAFFE
│       └── lssvm_toy_xor_iris.png             # [A3] Visualisasi batas keputusan LS-SVM
├── tests/
│   ├── test_features.py       # [A2] Unit test suite fitur
│   ├── test_baseline.py       # [A3] Unit test suite baseline LDA + SVM-RBF
│   └── test_lssvm.py          # [A3] Unit test suite LS-SVM (KKT, XOR, Iris)
└── model-experiment/
    ├── __init__.py
    ├── lssvm.py               # [A3] Implementasi modul LS-SVM biner (KKT solver)
    ├── baseline.py            # [A3] Pipeline baseline LDA + SVM-RBF & metrik Sen/Spe/Acc
    ├── evaluate_baseline.py   # [A3] Skrip eksekusi evaluasi baseline & ekspor laporan
    ├── visualize_toy_datasets.py # [A3] Visualisasi batas keputusan pada data mainan
    ├── A3-model-experiment.ipynb # [A3] Notebook eksperimen model Minggu 12
    └── LS-SVM.ipynb           # [A1] notebook praproses dataset
```

> Folder/kode tahap A2, A3, dan A4 akan ditambahkan oleh anggota masing-masing melalui feature branch sendiri.

## Dataset

Dataset **tidak disertakan** di repository (lisensi & ukuran file). Siapkan secara lokal di satu folder induk, lalu sesuaikan `DATASET_DIR` pada sel konfigurasi notebook:

| Dataset | Citra | Kelas | Subjek | Ukuran asli |
|---------|-------|-------|--------|-------------|
| CK+ | 981 | anger (135), contempt (54), disgust (177), fear (75), happy (207), sadness (84), surprise (249) | 118 | 48×48 |
| JAFFE | 213 | anger (30), disgust (29), fear (32), happy (31), neutral (30), sadness (31), surprise (30) | 10 | 128×128 |

Struktur folder yang diharapkan:

```
dataset/
├── ck+/
│   ├── anger/     ├── contempt/  ├── disgust/  ├── fear/
│   ├── happy/     ├── sadness/   └── surprise/          (berisi *.png)
└── jaffe/
    └── *.png      (format nama: 0000_KA.AN1.39.png → subjek KA, ekspresi AN)
```

Sitasi dataset:
- **CK+**: Lucey, P., Cohn, J.F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). *The Extended Cohn-Kanade Dataset (CK+): A complete dataset for action unit and emotion-specified expression.* IEEE CVPR Workshops, 94–101.
- **JAFFE**: Lyons, M.J., Akamatsu, S., Kamachi, M., & Gyoba, J. (1998). *Coding facial expressions with Gabor wavelets.* IEEE International Conference on Automatic Face & Gesture Recognition, 200–205.

## A1 — Data & Praproses

### Setup Environment (Windows)

```powershell
git clone https://github.com/jiroammarwafi/TA-PCVK.git
cd TA-PCVK

# Python 3.14 — pakai --copies bila kebijakan Device Guard/WDAC memblokir exe venv
python -m venv --copies .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Daftarkan kernel Jupyter (opsional; VS Code cukup memilih .venv)
python -m ipykernel install --user --name ta-pcvk --display-name "Python 3.14 (TA-PCVK)"
```

### Menjalankan Notebook A1

1. Buka `model-experiment/LS-SVM.ipynb` di VS Code / Jupyter.
2. Pilih kernel `.venv` (Python 3.14).
3. Sesuaikan `DATASET_DIR` (sel ke-2) dengan lokasi dataset di komputer Anda.
4. **Run All** → artefak praproses tertulis ulang ke `data/`.

### Artefak Praproses (`.npz`) — Handoff ke A2

A2/A3 **tidak perlu menjalankan ulang notebook A1**; cukup muat artefaknya:

```python
import numpy as np

data = np.load("data/ck_preprocessed.npz")
X = data["X"]                                # (981, 128, 128) uint8
y = data["y"]                                # (981,) int8
subjects = data["subjects"]                  # (981,) str — kode subjek
class_names = data["class_names"].tolist()   # ['anger', 'contempt', ...]
```

| Key | Shape | Dtype | Keterangan |
|-----|-------|-------|------------|
| `X` | CK+ `(981, 128, 128)`, JAFFE `(213, 128, 128)` | `uint8` | Citra hasil praproses (grayscale, 128×128, histogram equalized) |
| `y` | `(N,)` | `int8` | Indeks label kelas (urut terhadap `class_names`) |
| `subjects` | `(N,)` | `str` | Kode subjek (CK+: `S010`, JAFFE: `KA`) — untuk evaluasi *subject-independent* |
| `class_names` | `(7,)` | `str` | Nama kelas urut alfabet = urutan indeks label |

Pemetaan label:
- **CK+** → 0=anger, 1=contempt, 2=disgust, 3=fear, 4=happy, 5=sadness, 6=surprise
- **JAFFE** → 0=anger, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sadness, 6=surprise

### Ringkasan Praproses (mengikuti jurnal)

1. **Grayscale** — informasi ekspresi terletak pada geometri/intensitas wajah; SWT & PHOG bekerja pada citra satu kanal.
2. **Resize 128×128** dengan interpolasi adaptif (`INTER_AREA` untuk downscaling, `INTER_CUBIC` untuk upscaling CK+ 48×48) — syarat vektor fitur PHOG berukuran tetap.
3. **Histogram equalization** — normalisasi iluminasi/kontras agar fitur robust terhadap variasi pencahayaan.
4. **Tanpa split train/test di A1** — jurnal memakai *5-fold stratified cross-validation*; pembagian fold dilakukan di tahap eksperimen A3 (`sklearn.model_selection.StratifiedKFold`).


## A2 — Feature Engineering

Modul dan eksperimen fitur hybrid **Stationary Wavelet Transform (SWT Haar Level-1) + Pyramid of Histograms of Oriented Gradients (PHOG)** telah selesai diimplementasikan sesuai jurnal Kar et al. (2021).

### Cara Menggunakan Modul Modular (`features.py`)

A3 dan A4 dapat langsung mengimpor fungsi dari modul `features.py`:

```python
from features import swt_haar_level1, phog, extract_features

# 1. Dekomposisi wavelet pada 1 citra (128x128)
subbands = swt_haar_level1(img)  # dict: 'A1', 'H1', 'V1', 'D1' (semua 128x128 float64)

# 2. PHOG pada subband aproksimasi A1 (L=3, K=8 -> 680 dimensi)
vec = phog(subbands['A1'], L=3, K=8, norm='per_level')  # shape (680,) float32

# 3. Ekstraksi langsung untuk citra tunggal atau batch (N, 128, 128)
F = extract_features(X)  # shape (N, 680) float32
```

### Artefak Fitur — Handoff ke A3 (Model & Optimasi)

A3 **tidak perlu mengekstrak ulang fitur**; cukup muat artefak `.npz`:

```python
import numpy as np

ck_feat = np.load('data/ck_features.npz')
F_ck = ck_feat['F']             # (981, 680) float32
y_ck = ck_feat['y']             # (981,) int8
subjects_ck = ck_feat['subjects']
classes_ck = ck_feat['class_names'].tolist()

jaffe_feat = np.load('data/jaffe_features.npz')
F_jaffe = jaffe_feat['F']       # (213, 680) float32
y_jaffe = jaffe_feat['y']       # (213,) int8
```

### Parameter Resmi Sesuai Jurnal
- **Wavelet**: Haar level-1 (*undecimated* / tanpa downsampling → *shift-invariant*).
- **Subband**: Hanya $A_1$ (aproksimasi LL, kontur struktur wajah frekuensi rendah).
- **PHOG**:
  - $L = 3$ tingkat resolusi piramida spasial: $1 + 4 + 16 + 64 = 85$ sel spasial.
  - $K = 8$ bin orientasi dalam rentang $[0 - 360^\circ]$ (lebar tiap bin $45^\circ$).
  - Pembobotan bin menggunakan magnitudo gradien Sobel $3\times 3$.
  - Normalisasi: $L_2$ per level piramida lalu dikonkatenasi.
  - Dimensi vektor total: $85 \times 8 = 680$ fitur per citra.

### Menjalankan Unit Test
```powershell
pytest -v
```

## A3 — Model & Optimasi (Minggu 12)

Implementasi tahap **Minggu 12** berfokus pada setup konfigurasi terpusat, evaluasi **Baseline Model (LDA + SVM-RBF)** menggunakan scikit-learn pada fitur hybrid 680-D, derivasi dan implementasi prototipe **Least Squares SVM (LS-SVM)** biner, serta pengujian paritas pada dataset mainan (**XOR** dan **Iris**).

### 1. Konfigurasi Terpusat (`config.yaml`)
Seluruh parameter sistem (path dataset, cross-validation, hyperparameter LDA, SVM, LS-SVM, serta random seed) disentralisasi di `config.yaml` untuk memenuhi kriteria Definition of Done (tidak ada *hardcoding* parameter).

### 2. Hasil Baseline Model (LDA + SVM-RBF, 5-Fold Stratified CV)
Evaluasi baseline dilakukan secara ketat dengan 5-fold stratified cross-validation pada fold training saja untuk mencegah *data leakage*:

| Dataset | Sampel | Fitur | Mean Accuracy (%) | Std Acc (%) | Macro Sensitivity (%) | Macro Specificity (%) |
|---------|--------|-------|-------------------|-------------|-----------------------|-----------------------|
| **CK+** | 981 | 680 | **97.66%** | ±1.00% | 97.49% | 99.59% |
| **JAFFE** | 213 | 680 | **81.27%** | ±8.37% | 81.15% | 96.87% |

Artefak hasil baseline:
- Ringkasan tabel: `reports/baseline_summary.csv`
- Detail metrik & confusion matrix: `reports/baseline_ck_results.json` dan `reports/baseline_jaffe_results.json`
- Plot confusion matrix: `reports/figures/baseline_confusion_matrix_ck.png` dan `reports/figures/baseline_confusion_matrix_jaffe.png`

### 3. Prototipe LS-SVM Biner (`model-experiment/lssvm.py`)
LS-SVM (Suykens & Vandewalle, 1999) mengonversi persoalan quadratic programming (QP) SVM standar menjadi sistem persamaan linear Karush-Kuhn-Tucker (KKT):

$$\begin{bmatrix} 0 & y^T \\ y & \Omega + \left(\gamma^{-1} + \epsilon\right) I_N \end{bmatrix} \begin{bmatrix} b \\ \alpha \end{bmatrix} = \begin{bmatrix} 0 \\ \mathbf{1}_N \end{bmatrix}$$

- **Kernel RBF**: $K(x, z) = \exp\left(-\frac{\|x - z\|^2}{2\sigma^2}\right)$
- Solusi matriks diselesaikan secara eksak menggunakan `scipy.linalg.solve(assume_a='sym')` dengan verifikasi residual aljabar linear $\|A \cdot \text{sol} - B\| < 10^{-5}$.

### 4. Verifikasi pada Dataset Mainan (Toy Datasets)
- **Problem Non-Linear XOR**: LS-SVM biner dengan kernel RBF mencapai akurasi **98.0%** pada clean XOR dan **>95%** pada noisy XOR, membuktikan kemampuan memisahkan ruang non-linear.
- **Dataset Iris (Setosa vs Versicolor)**: LS-SVM biner mencapai akurasi **100%**, identik dengan performa `sklearn.svm.SVC(kernel='rbf')` (paritas sempurna).
- Visualisasi batas keputusan: `reports/figures/lssvm_toy_xor_iris.png`.

### Menjalankan Eksperimen & Test Suite
```powershell
# Jalankan seluruh unit test suite (16 test lulus 100%)
pytest -v

# Jalankan evaluasi baseline & ekspor laporan
python model-experiment/evaluate_baseline.py

# Buat visualisasi batas keputusan pada data mainan
python model-experiment/visualize_toy_datasets.py
```

## Konvensi Kontribusi (Tim)

- Satu **feature branch** per bagian: `feat/a1-…`, `feat/a2-…`, `feat/a3-…`, `feat/a4-…`
- Pesan commit mengikuti **Conventional Commits**: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`
- Selesai mengerjakan → buka **Pull Request** ke `main` dan minta review anggota lain
- Jangan commit `.venv/`, dataset mentah, atau file besar di luar `data/*.npz` (lihat `.gitignore`)
