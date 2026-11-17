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

Artefak: `data/a1-noise/{ck,jaffe}_noise_v{0.01,0.02,0.05,0.10}.npz`
Perintah: `.\run.ps1 noise`

Statistik verifikasi (CK+, 981 citra):

| Variansi | σ teori | σ empiris | Max \|diff\| |
|---|---|---|---|
| 0,01 | 0,1000 | 0,0936 | 0,529 |
| 0,02 | 0,1414 | 0,1293 | 0,753 |
| 0,05 | 0,2236 | 0,1941 | 1,000 |
| 0,10 | 0,3162 | 0,2566 | 1,000 |

σ empiris lebih kecil karena clipping ke [0,1] dan kuantisasi ke 256 level.

**Evaluasi akurasi pada data ber-noise belum dijalankan** — menunggu A2
mengekstrak ulang fitur dari artefak ini, lalu A3 mengevaluasi.

### Angka jurnal sebagai rujukan (Tabel 14)

| Kondisi | CK+ Sen / Spe / Acc | JAFFE Sen / Spe / Acc |
|---|---|---|
| Tanpa noise | 94,08 / 99,51 / 98,38 | 95,23 / 99,61 / 98,99 |
| Dengan noise | 95,95 / 96,86 / **97,17** | 91,50 / 98,59 / **97,59** |

> Pada jurnal, noise **CK+ = 97,17%** dan **JAFFE = 97,59%**. Keduanya berbeda.

## 5. Yang belum dikerjakan

| Item | Status |
|---|---|
| Algoritma Jaya (inti judul jurnal) | ❌ belum — masih grid search |
| Pembanding PSO & GA | ❌ belum |
| Kurva konvergensi optimizer | ❌ belum |
| Evaluasi akurasi pada data ber-noise | ⏳ menunggu A2 |
| Perbandingan Lin/Poly LS-SVM, RF, BPNN | ❌ belum |
| Waktu latih/uji | ❌ belum |
| Web (A4) | ⏳ belum |
