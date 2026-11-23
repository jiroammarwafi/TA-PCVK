# TA-PCVK — Facial Expression Recognition (Hybrid Feature Descriptor + Jaya-LSSVM)

Replikasi jurnal tugas akhir:

> Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021). *A hybrid feature descriptor with Jaya optimised least squares SVM for facial expression recognition.* IET Image Processing, 15(7), 1471–1483. DOI: [10.1049/ipr2.12118](https://doi.org/10.1049/ipr2.12118) (Open Access, CC-BY)

Mengenali **7 kelas ekspresi wajah** dari dataset **CK+** dan **JAFFE** menggunakan deskriptor hybrid **SWT Haar + PHOG**, reduksi dimensi **LDA**, dan classifier **LSSVM** dengan parameter (γ, σ) yang dioptimasi **algoritma Jaya**, dievaluasi dengan **5-fold stratified cross-validation**.

## Pipeline & Pembagian Tugas

| Kode | Bagian | Cakupan | Status |
|------|--------|---------|--------|
| **A1** | Data & Praproses | Loading dataset (CK+ & JAFFE), EDA, grayscale → resize 128×128 → histogram equalization, artefak `.npz`, **noise Gaussian (Minggu 14)**, **evaluasi subject-independent (Minggu 14)** | ✅ Selesai |
| **A2** | Feature Engineering | SWT Haar level-1 → PHOG pada subband A1 (LL), ekspor fitur 680-D | ✅ Selesai |
| **A3** | Model & Optimasi | Seleksi fitur LDA → classifier Jaya-LSSVM → evaluasi 5-fold stratified CV | ✅ Minggu 12 & 13 Selesai (Minggu 14 On Progress) |
| **A4** | Web & Integrasi | Website penyajian sistem dan hasil | ⏳ Belum dimulai |

```
Citra wajah (CK+ / JAFFE)
   → [A1] Preprocessing       : grayscale, resize 128×128, histogram equalization
   → [A2] SWT Haar → PHOG(LL) : ekstraksi fitur hybrid spatial-frekuensi
   → [A3] LDA → Jaya-LSSVM    : reduksi dimensi + klasifikasi
   → [A4] Web                 : penyajian hasil
```

## Struktur Repository

Struktur dipisah **per divisi**: kode `.py` di `research/a{1,2,3}-*/`, seluruh
**notebook** di `research/pipeline/`, dan `app/` disiapkan untuk A4.

```
.
├── README.md                  # Dokumentasi utama
├── konteks.md                 # Ringkasan 1 halaman untuk onboarding
├── config.yaml                # Konfigurasi parameter terpusat (DoD A3)
├── requirements.txt           # Dependensi Python
├── pytest.ini                 # Konfigurasi test runner pytest
├── run.ps1                    # Runner: test / pipeline / noise / split / report / check
│
├── research/                  # SEMUA kode penelitian (A1–A3)
│   ├── a1-data-preprocessing/
│   │   ├── noise.py                        # [A1] Artefak citra ber-noise Gaussian
│   │   ├── evaluate_subject_independent.py # [A1] Split acak vs per-subjek
│   │   └── diagnose_identity.py            # [A1] Diagnosa kebocoran identitas subjek
│   ├── a2-feature-engineering/
│   │   ├── __init__.py
│   │   └── features.py                     # [A2] SWT Haar + PHOG (680-D)
│   ├── a3-model-optimization/
│   │   ├── __init__.py
│   │   ├── lssvm.py                        # [A3] LS-SVM biner + One-vs-One
│   │   ├── baseline.py                     # [A3] LDA + SVM-RBF & metrik
│   │   ├── evaluate_baseline.py            # [A3] Runner baseline
│   │   ├── evaluate_lssvm.py               # [A3] Runner LS-SVM OvO + grid search
│   │   └── visualize_toy_datasets.py       # [A3] Batas keputusan XOR/Iris
│   └── pipeline/                           # SEMUA notebook
│       ├── 00-pipeline-lengkap.ipynb        # A1 → A2 → A3 dalam satu alur
│       ├── 01-a1-preprocessing.ipynb        # [A1] praproses dataset
│       ├── 02-a2-feature-extraction.ipynb   # [A2] ekstraksi fitur
│       ├── 03-a3-baseline-dan-lssvm-biner.ipynb   # [A3] Minggu 12
│       ├── 04-a3-multiclass-lssvm-ovo.ipynb       # [A3] Minggu 13
│       └── 05-a1-noise-dan-subject-independent.ipynb  # [A1] Minggu 14
│
├── data/                      # Artefak biner, dipisah per divisi
│   ├── a1-preprocessing/      # [A1] ck/jaffe_preprocessed.npz (128×128 uint8)
│   ├── a1-noise/              # [A1] *_noise_v{0.01..0.10}.npz (gitignored)
│   └── a2-features/           # [A2] ck/jaffe_features.npz (680-D float32)
│
├── reports/                   # Hasil, dipisah per divisi
│   ├── a1-subject-independent/   # subject_independent_*, identity_* (json + csv)
│   ├── a2-features/              # phog_levels_visualization.png
│   └── a3-models/
│       ├── baseline_{summary.csv,ck_results.json,jaffe_results.json}
│       ├── lssvm_multiclass_{summary.csv,ck_results.json,jaffe_results.json}
│       └── figures/              # 5 confusion matrix & visualisasi toy
│
├── docs/                      # Dokumentasi
│   ├── arsitektur.md          # diagram alur & pembagian tanggung jawab
│   ├── metodologi.md          # praproses, fitur, LS-SVM, protokol evaluasi
│   ├── hasil.md               # ringkasan semua angka + interpretasi
│   ├── bab3-dataset-dan-praproses.md  # draft Bab 3 (dataset & praproses)
│   └── referensi/             # PDF jurnal
│
├── tests/                     # 50 item test (49 lulus, 1 modul di-skip)
│   ├── test_features.py          # [A2]
│   ├── test_baseline.py          # [A3]
│   ├── test_lssvm.py             # [A3]
│   ├── test_noise.py             # [A1]
│   ├── test_subject_independent.py  # [A1]
│   └── test_identity_diagnostic.py  # [A1]
│
├── scripts/
│   └── check_env.py           # diagnosa DLL yang diblokir Application Control
│
└── app/                       # [A4] disiapkan, masih kosong (lihat app/README.md)
```

> Notebook **tidak** menyalin logika: semuanya memanggil modul `.py` di
> `research/`, sehingga perbaikan modul otomatis berlaku dan dapat diuji pytest.

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

### Quickstart dengan `run.ps1`

Satu pintu untuk semua perintah umum:

```powershell
.\run.ps1 test        # seluruh unit test
.\run.ps1 pipeline    # buka notebook pipeline lengkap
.\run.ps1 noise       # bangun artefak citra ber-noise (A1)
.\run.ps1 split       # evaluasi split acak vs subject-independent (A1)
.\run.ps1 identity    # diagnosa kebocoran identitas subjek (A1)
.\run.ps1 baseline    # evaluasi baseline LDA + SVM-RBF (A3)
.\run.ps1 lssvm       # evaluasi LS-SVM OvO + grid search (A3)
.\run.ps1 report      # tampilkan semua tabel hasil
.\run.ps1 check       # diagnosa DLL yang diblokir Application Control
.\run.ps1 clean       # hapus sisa duplikat setelah restrukturisasi
```

### Notebook: di mana semuanya berada

Seluruh notebook ada di `research/pipeline/` dengan prefiks angka sesuai urutan
eksekusi:

| Notebook | Isi |
|---|---|
| `00-pipeline-lengkap.ipynb` | **A1 → A2 → A3 dalam satu alur** (flag `USE_CACHE`) |
| `01-a1-preprocessing.ipynb` | praproses dataset → `*_preprocessed.npz` |
| `02-a2-feature-extraction.ipynb` | SWT + PHOG → `*_features.npz` |
| `03-a3-baseline-dan-lssvm-biner.ipynb` | baseline LDA+SVM-RBF & LS-SVM biner |
| `04-a3-multiclass-lssvm-ovo.ipynb` | analisis hasil LS-SVM OvO |
| `05-a1-noise-dan-subject-independent.ipynb` | noise Gaussian & split per-subjek |

**`00-pipeline-lengkap.ipynb`** adalah pilihan tercepat untuk demo. Sel pertama
berisi:

```python
USE_CACHE = True    # True  → ~1 menit (pakai artefak .npz yang ada)
                    # False → ~40-70 menit (hitung ulang dari citra mentah)
```

### Menjalankan Notebook A1 (praproses)

1. Buka `research/pipeline/01-a1-preprocessing.ipynb` di VS Code / Jupyter.
2. Pilih kernel `.venv` (Python 3.14).
3. Dataset dicari otomatis di `../dataset`; override dengan env `FER_DATASET_DIR`.
4. **Run All** → artefak tertulis ke `data/a1-preprocessing/`.

### Artefak Praproses (`.npz`) — Handoff ke A2

A2/A3 **tidak perlu menjalankan ulang notebook A1**; cukup muat artefaknya:

```python
import numpy as np

data = np.load("data/a1-preprocessing/ck_preprocessed.npz")
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

### Catatan Deviasi terhadap Jurnal

Praproses di atas **tidak identik** dengan jurnal bagian 4.1. Perbedaan yang disengaja:

| Tahap | Jurnal | Implementasi ini | Alasan |
|---|---|---|---|
| Normalisasi kontras | Saturasi 1% piksel atas & bawah (*contrast stretching*) | Histogram equalization | Keduanya menormalkan iluminasi; equalization dipilih agar artefak A1 sudah stabil dan tidak bergantung pada persentil per citra |
| Deteksi wajah | Viola–Jones, lalu crop | Tidak ada (langsung resize) | Kedua dataset sudah berupa citra wajah ter-crop; `cv2.CascadeClassifier` juga dihapus pada OpenCV 5 |

Konsekuensinya, **angka akurasi kita tidak dapat dibandingkan langsung dengan jurnal** — dan memang tidak diniatkan demikian. Jurnal hanya dipakai sebagai rujukan metode.

### Catatan Sensitivitas Numerik Fitur

Nilai fitur 680-D **tidak sepenuhnya bebas dari versi library**:

| Komponen | Pengaruh |
|---|---|
| PyWavelets (SWT Haar) | perbedaan antarversi < 1e-12 → **tidak** berpengaruh |
| OpenCV (Sobel 3×3) | gradien berbeda ~1e-13 antarversi, tetapi **binning orientasi PHOG mengamplifikasi** menjadi ~2e-3 pada vektor fitur |

Akibatnya menghitung ulang fitur dengan versi OpenCV berbeda dapat menggeser
akurasi sekitar **0,2–1 poin persen**. Karena itu:

- `requirements.txt` mem-pin `opencv-python==5.0.0.93` agar hasil reprodusibel.
- Notebook `02-a2-feature-extraction` **tidak menimpa** artefak `.npz` secara
  default (`REGENERATE_FEATURES = False`). Artefak yang di-commit adalah acuan
  angka pada laporan.
- Bila artefak dihitung ulang, angka di `docs/hasil.md` dan README perlu
  diperbarui.

### Minggu 14 — Uji Ketahanan Noise Gaussian

Jurnal bagian 5.5 menguji ketahanan model dengan noise Gaussian bervariansi 0,01–0,1. Modul `research/a1-data-preprocessing/noise.py` menyiapkan artefak citranya:

```python
from noise import add_gaussian_noise, build_noise_artifacts

# noise ditambahkan pada ruang intensitas [0,1], lalu dikembalikan ke uint8
noisy = add_gaussian_noise(images, variance=0.05, seed=42)

# bangun seluruh artefak (CK+ & JAFFE × 4 variansi)
build_noise_artifacts(config)
```

```powershell
python research/a1-data-preprocessing/noise.py                       # semua dataset & variansi
python research/a1-data-preprocessing/noise.py --variances 0.01 0.1  # variansi tertentu
python research/a1-data-preprocessing/noise.py --datasets jaffe --force
```

Artefak yang dihasilkan: `data/{ck,jaffe}_noise_v{0.01,0.02,0.05,0.10}.npz`.

| Kunci | Shape | Keterangan |
|---|---|---|
| `X` | `(N, 128, 128)` `uint8` | Citra ber-noise (siap masuk `extract_features`) |
| `y`, `subjects`, `class_names` | sama dengan A1 | Metadata tidak berubah |
| `variance`, `seed` | skalar | Jejak parameter noise |

**Handoff ke A2:** cukup `np.load(...)["X"]` lalu `extract_features(X)` — tidak perlu mengubah apa pun di modul fitur.

Verifikasi statistik (CK+, 981 citra):

| Variansi diminta | σ teori | σ empiris | Catatan |
|---|---|---|---|
| 0,01 | 0,1000 | 0,0936 | sedikit turun karena clipping di 0/1 |
| 0,02 | 0,1414 | 0,1293 | |
| 0,05 | 0,2236 | 0,1941 | |
| 0,10 | 0,3162 | 0,2566 | clipping paling terasa |

`variance=0` menghasilkan salinan **identik** (dipakai sebagai kontrol di unit test).

**Angka jurnal (Tabel 14) sebagai rujukan** — bukan target replikasi:

| Kondisi | CK+ Sen / Spe / Acc | JAFFE Sen / Spe / Acc |
|---|---|---|
| Tanpa noise | 94,08 / 99,51 / 98,38 | 95,23 / 99,61 / 98,99 |
| Dengan noise (var 0,01–0,1) | 95,95 / 96,86 / **97,17** | 91,50 / 98,59 / **97,59** |

> Perhatikan: pada jurnal, angka noise **CK+ = 97,17%** dan **JAFFE = 97,59%**. Keduanya berbeda; jangan tertukar — sebagian rangkuman sering salah menulis 97,17% sebagai angka JAFFE.

> Status: artefak citra ber-noise **sudah dibuat**, tetapi evaluasi akurasi pada data ber-noise **belum dijalankan** (menunggu A2 mengekstrak ulang fitur, lalu A3 mengevaluasi). Lihat `docs/hasil.md` §4.

### Minggu 14 — Evaluasi Subject-Independent

CK+ memuat **3 frame berurutan per sekuens** dari subjek yang sama. Dengan `StratifiedKFold` biasa, frame se-subjek dapat masuk train *dan* test sekaligus sehingga akurasi terlihat lebih tinggi dari kemampuan generalisasi sebenarnya.

`research/a1-data-preprocessing/evaluate_subject_independent.py` menjalankan **dua protokol pada pipeline identik** (LDA → StandardScaler → OvO LS-SVM + grid search internal):

```powershell
python research/a1-data-preprocessing/evaluate_subject_independent.py
```

- `random`  → `StratifiedKFold` (protokol A3 saat ini)
- `subject` → `StratifiedGroupKFold(groups=subjects)` — seluruh citra satu subjek hanya di satu sisi

Skrip melakukan **assert** bahwa tidak ada subjek yang muncul di train dan test sekaligus, lalu melaporkan selisih akurasi antar kedua protokol.

Artefak: `reports/a1-subject-independent/subject_independent_{ck,jaffe}.json` dan
`subject_independent_summary.csv`.

**Hasil (pipeline LDA → StandardScaler → OvO LS-SVM, 5-fold):**

| Dataset | Split | Akurasi (%) | Std (%) | Macro Sen (%) | Macro Spe (%) |
|---|---|---|---|---|---|
| CK+ | random (acak) | **96,64** | 0,94 | 97,02 | 99,44 |
| CK+ | **subject** (per-subjek) | **48,21** | 4,23 | 45,71 | 91,44 |
| JAFFE | random (acak) | **80,31** | 8,23 | 80,20 | 96,71 |
| JAFFE | **subject** (per-subjek) | **38,22** | 10,54 | 38,59 | 89,76 |

Selisih: **CK+ 48,43 poin persen**, **JAFFE 42,09 poin persen**.

Temuan penting:

1. Angka CK+ `random` = **96,64%** identik dengan hasil A3 Minggu 13 — mengonfirmasi implementasi ini konsisten dengan pipeline A3.
2. Namun hampir **setengah** akurasi tersebut berasal dari kebocoran identitas: dengan subjek yang benar-benar terpisah, akurasi jatuh ke **48,21%**.
3. Penyebabnya: CK+ berisi 3 frame berurutan per sekuens (327 sekuens), sehingga pada split acak frame se-subjek muncul di train *dan* test. JAFFE juga menunjukkan pola serupa karena setiap subjek menyumbang 7 ekspresi.
4. Kesimpulan: untuk dataset dengan banyak citra per orang, **akurasi split acak tidak boleh dilaporkan sebagai kemampuan generalisasi**. Angka subject-independent inilah estimasi yang jujur.

### Minggu 14 — Diagnosa Kebocoran Identitas

`research/a1-data-preprocessing/diagnose_identity.py` mengukur **penyebab**
kejatuhan di atas: seberapa besar vektor fitur 680-D memuat identitas orang,
bukan ekspresi.

```powershell
python research/a1-data-preprocessing/diagnose_identity.py
# atau
.\run.ps1 identity
```

Tiga pengukuran:

| Kunci hasil | Protokol split | Yang diukur |
|---|---|---|
| `subject_identification` | **acak** | kemampuan menebak *siapa* orangnya dari fitur |
| `raw_features` | per subjek | akurasi emosi pada fitur mentah, tanpa LDA |
| `lda_nested` | per subjek, LDA per fold | akurasi emosi setelah reduksi LDA |

> Tugas "tebak subjek" sengaja memakai split **acak**: kebocoran pada protokol
> split acak memang terjadi karena frame orang yang sama muncul di train *dan*
> test. Dengan split per subjek, subjek uji tidak dikenal model sehingga
> akurasinya selalu ~0% dan tidak mengukur kebocoran.

**Hasil:**

| Pengukuran | CK+ | JAFFE |
|---|---|---|
| Menebak subjek (peluang 1,03% / 10,00%) | **98,58%** | **99,53%** |
| Emosi per subjek — fitur mentah (logreg) | **89,90%** | **57,68%** |
| Emosi per subjek — LDA nested (6 komponen) | 53,19% | 36,00% |
| Penalti LDA | **36,71 poin** | **21,69 poin** |

Kesimpulan: fitur hybrid hampir sepenuhnya memuat identitas, **dan LDA justru
membuang informasi ekspresi** pada protokol jujur (CK+ 89,90% → 53,19%).
Karena itu A3 disarankan melaporkan protokol subject-independent **dengan dan
tanpa LDA**.

Artefak: `reports/a1-subject-independent/identity_{ck,jaffe}.json` dan
`identity_summary.csv`.

> Catatan lingkungan: pada mesin dengan Application Control, `pandas` dapat
> gagal diimpor total (`_ujson` diblokir) sehingga CSV ditulis memakai modul
> `csv` bawaan. `RandomForestClassifier` juga dapat diblokir
> (`_gradient_boosting`); klasifier itu dilewati otomatis dan dicatat di
> `skipped_classifiers`.


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

ck_feat = np.load('data/a2-features/ck_features.npz')
F_ck = ck_feat['F']             # (981, 680) float32
y_ck = ck_feat['y']             # (981,) int8
subjects_ck = ck_feat['subjects']
classes_ck = ck_feat['class_names'].tolist()

jaffe_feat = np.load('data/a2-features/jaffe_features.npz')
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
.\run.ps1 test
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
- Plot confusion matrix: `reports/a3-models/figures/baseline_confusion_matrix_ck.png` dan `reports/a3-models/figures/baseline_confusion_matrix_jaffe.png`

### 3. Prototipe LS-SVM Biner (`research/a3-model-optimization/lssvm.py`)
LS-SVM (Suykens & Vandewalle, 1999) mengonversi persoalan quadratic programming (QP) SVM standar menjadi sistem persamaan linear Karush-Kuhn-Tucker (KKT):

$$\begin{bmatrix} 0 & y^T \\ y & \Omega + \left(\gamma^{-1} + \epsilon\right) I_N \end{bmatrix} \begin{bmatrix} b \\ \alpha \end{bmatrix} = \begin{bmatrix} 0 \\ \mathbf{1}_N \end{bmatrix}$$

- **Kernel RBF**: $K(x, z) = \exp\left(-\frac{\|x - z\|^2}{2\sigma^2}\right)$
- Solusi matriks diselesaikan secara eksak menggunakan `scipy.linalg.solve(assume_a='sym')` dengan verifikasi residual aljabar linear $\|A \cdot \text{sol} - B\| < 10^{-5}$.

### 4. Verifikasi pada Dataset Mainan (Toy Datasets)
- **Problem Non-Linear XOR**: LS-SVM biner dengan kernel RBF mencapai akurasi **98.0%** pada clean XOR dan **>95%** pada noisy XOR, membuktikan kemampuan memisahkan ruang non-linear.
- **Dataset Iris (Setosa vs Versicolor)**: LS-SVM biner mencapai akurasi **100%**, identik dengan performa `sklearn.svm.SVC(kernel='rbf')` (paritas sempurna).
- Visualisasi batas keputusan: `reports/a3-models/figures/lssvm_toy_xor_iris.png`.

### Menjalankan Eksperimen & Test Suite (Minggu 12)
```powershell
# Jalankan seluruh unit test suite
.\run.ps1 test

# Jalankan evaluasi baseline & ekspor laporan
python research/a3-model-optimization/evaluate_baseline.py

# Buat visualisasi batas keputusan pada data mainan
python research/a3-model-optimization/visualize_toy_datasets.py
```

## A3 — Model & Optimasi (Minggu 13)

Implementasi tahap **Minggu 13** berfokus pada:
1. **LS-SVM Multiclass (One-vs-One / OvO)**: Mengembangkan kelas `OneVsOneLSSVMClassifier` (`MulticlassLSSVM`) di `research/a3-model-optimization/lssvm.py`. Untuk 7 kelas ekspresi wajah, dibangun $7 \times 6 / 2 = 21$ classifier biner LS-SVM dengan strategi *majority voting* dan *confidence margin tie-breaking*.
2. **Evaluasi 5-Fold Stratified Cross-Validation dengan Grid Search Bersarang**:
   - Pipeline: **LDA ($680 \to 6$ komponen) $\to$ StandardScaler $\to$ OvO LS-SVM**.
   - Grid search internal 3-fold pada fold train untuk tuning hyperparameter $(\gamma, \sigma)$: $\gamma \in [1.0, 10.0, 50.0, 100.0, 500.0]$ dan $\sigma \in [1.0, 1.5, 2.0, 3.0, 5.0]$.
   - Mencegah *data leakage* dengan menerapkan fitting LDA dan StandardScaler hanya pada fold training.

### Hasil Baseline Deskriptif v0 (LS-SVM Multiclass, 5-Fold Stratified CV)

| Dataset | Sampel | Model | Mean Accuracy (%) | Std Acc (%) | Macro Sensitivity (%) | Macro Specificity (%) |
|---------|--------|-------|-------------------|-------------|-----------------------|-----------------------|
| **CK+** | 981 | LDA + OvO LS-SVM (Grid Search) | **96.64%** | ±0.94% | 97.02% | 99.44% |
| **JAFFE** | 213 | LDA + OvO LS-SVM (Grid Search) | **79.36%** | ±7.52% | 79.28% | 96.56% |

Artefak hasil Minggu 13:
- Ringkasan tabel: `reports/lssvm_multiclass_summary.csv`
- Detail parameter per-fold & confusion matrix: `reports/lssvm_multiclass_ck_results.json` dan `reports/lssvm_multiclass_jaffe_results.json`
- Plot confusion matrix: `reports/a3-models/figures/lssvm_confusion_matrix_ck.png` dan `reports/a3-models/figures/lssvm_confusion_matrix_jaffe.png`
- Notebook eksekusi & analisis per kelas: `research/pipeline/04-a3-multiclass-lssvm-ovo.ipynb`

### Menjalankan Eksperimen & Test Suite (Minggu 13)
```powershell
# Jalankan seluruh unit test suite
.\run.ps1 test

# Jalankan evaluasi 5-fold CV LS-SVM OvO Grid Search pada CK+ dan JAFFE
python research/a3-model-optimization/evaluate_lssvm.py
```

## Konvensi Kontribusi (Tim)

- Satu **feature branch** per bagian: `feat/a1-…`, `feat/a2-…`, `feat/a3-…`, `feat/a4-…`
- Pesan commit mengikuti **Conventional Commits**: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`
- Selesai mengerjakan → buka **Pull Request** ke `main` dan minta review anggota lain
- Jangan commit `.venv/`, dataset mentah, atau file besar di luar `data/*.npz` (lihat `.gitignore`)
