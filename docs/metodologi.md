# Metodologi

## 1. Praproses (A1)

| Langkah | Parameter | Catatan |
|---|---|---|
| Grayscale | konversi OpenCV BGR→GRAY | ekspresi terletak pada geometri/intensitas, bukan warna |
| Resize | 128×128, `INTER_AREA` (downscale), `INTER_CUBIC` (upscale CK+ 48→128) | syarat vektor PHOG berukuran tetap |
| Histogram equalization | `cv2.equalizeHist` | normalisasi iluminasi/kontras |

**Deviasi terhadap jurnal:** jurnal memakai *contrast stretching* 1% + deteksi
wajah Viola–Jones. Implementasi ini memakai histogram equalization dan tanpa
deteksi wajah (kedua dataset sudah ter-crop; `cv2.CascadeClassifier` juga
dihapus pada OpenCV 5). Konsekuensinya angka tidak dapat dibandingkan langsung
dengan jurnal.

## 2. Ekstraksi fitur (A2)

### 2.1 SWT Haar level-1

Stationary Wavelet Transform bersifat *shift-invariant* karena tidak ada
downsampling. Untuk filter Haar:

```
A1[i,j] = (x[i,j] + x[i,j+1] + x[i+1,j] + x[i+1,j+1]) / 2   (padding periodik)
```

Empat subband berukuran sama dengan citra: A1 (aproksimasi), H1, V1, D1.
Hanya **A1** yang dipakai, karena memuat struktur bentuk frekuensi rendah.

Implementasi memakai PyWavelets bila tersedia, dan otomatis beralih ke
**implementasi numpy murni** bila DLL PyWavelets diblokir kebijakan
Application Control. Keduanya identik sampai 1e-12, diverifikasi pada citra
acak 16/64/128 piksel.

### 2.2 PHOG

1. Gradien via Sobel 3×3 → magnitudo `M = √(Gx² + Gy²)`, arah `θ = atan2(Gy,Gx) mod 360°`
2. Piramida spasial level 0–3: `1 + 4 + 16 + 64 = 85` sel
3. Histogram 8 bin per sel, dibobot magnitudo, lebar bin 45°
4. Normalisasi L2 **per level**, lalu konkatenasi

Dimensi (persamaan 2 jurnal):

```
F = K · Σ(l=0..L) 4^l = 8 · 85 = 680
```

## 3. Reduksi dimensi (A3)

LDA memproyeksikan 680-D ke `C−1` komponen yang memaksimalkan rasio
scatter antar-kelas terhadap scatter dalam-kelas.

| Dataset | Kelas | Komponen LDA |
|---|---|---|
| CK+ | 7 | 6 |
| JAFFE | 7 | 6 |

**Catatan penting:** LDA di-fit **di dalam fold latih** (bukan pada seluruh
data). Ini mencegah kebocoran informasi dari data uji. Akibatnya angka kita
lebih rendah dari jurnal, yang tampaknya memakai LDA global.

## 4. Klasifikasi (A3)

### 4.1 LS-SVM biner

LS-SVM mengganti pertidaksamaan margin SVM dengan persamaan, sehingga solusi
berupa sistem linear:

```
[ 0     yᵀ                ] [ b ]   [ 0 ]
[ y     Ω + (1/γ + ε)·I   ] [ α ] = [ 1 ]
```

dengan `Ω_ij = y_i y_j K(x_i, x_j)` dan decision function
`f(x) = Σ α_i y_i K(x_i, x) + b`.

Kernel RBF: `K(x,z) = exp(−‖x−z‖² / (2σ²))`.

Diselesaikan dengan `scipy.linalg.solve(assume_a='sym')` dan diverifikasi
lewat residual `‖A·sol − B‖ < 1e-5` pada unit test.

### 4.2 Multiclass One-vs-One

Untuk 7 kelas dibangun `7·6/2 = 21` classifier biner. Prediksi ditentukan
**majority voting**, dengan *margin* sebagai penentu bila terjadi seri.

### 4.3 Tuning hyperparameter

Grid search internal 3-fold pada fold latih:

| Parameter | Nilai |
|---|---|
| γ (regularisasi) | 1, 10, 50, 100, 500 |
| σ (bandwidth RBF) | 1; 1,5; 2; 3; 5 |

Fitness = akurasi rata-rata CV internal. `StandardScaler` di-fit ulang pada
tiap split internal agar tidak bocor.

## 5. Protokol evaluasi

### 5.1 Split acak (protokol A3)

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

### 5.2 Split subject-independent (A1)

`StratifiedGroupKFold(groups=subjects)` — seluruh citra satu subjek hanya
berada di train **atau** test. Wajib untuk CK+ karena berisi 3 frame berurutan
per sekuens (327 sekuens × 3 = 981 citra).

Skrip melakukan **assert** bahwa tidak ada irisan subjek antara train dan test.

### 5.3 Uji noise (A1)

Noise Gaussian ditambahkan pada ruang intensitas `[0,1]`:

```
x_noisy = clip(x + N(0, variance), 0, 1) → uint8
```

Variansi mengikuti jurnal: 0,01 / 0,02 / 0,05 / 0,10 (σ = 0,10–0,316).
`variance=0` menghasilkan salinan identik (kontrol).

## 6. Metrik

Untuk setiap kelas `c` (one-vs-rest):

```
Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
Accuracy    = (TP + TN) / N
```

Dilaporkan sebagai rata-rata makro lintas kelas, plus confusion matrix agregat
dari seluruh fold.

## 7. Reproduksibilitas

| Aspek | Jaminan |
|---|---|
| Seed | `random_seed: 42` di `config.yaml` |
| Fold | `random_state=42`, stratified |
| Noise | `seed=42`, deterministik |
| Parameter | seluruhnya di `config.yaml`, tidak ada hardcode |
| Artefak | `.npz` tanpa pickle |
| Verifikasi | 41 unit test (`.\run.ps1 test`) |

## 8. Rujukan

Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021). *A hybrid feature
descriptor with Jaya optimised least squares SVM for facial expression
recognition.* IET Image Processing, 15(7), 1471–1483.

Suykens, J.A.K., & Vandewalle, J. (1999). *Least squares support vector machine
classifiers.* Neural Processing Letters, 9(3), 293–300.

Bosch, A., Zisserman, A., & Muñoz, X. (2007). *Representing shape with a spatial
pyramid kernel.* Proc. ACM CIVR, 401–408.
