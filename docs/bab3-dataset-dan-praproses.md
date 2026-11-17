# BAB 3 — Dataset dan Praproses

> **Status:** draft A1. Semua angka pada bab ini dihasilkan langsung dari
> artefak `data/a1-preprocessing/*.npz` dan dapat diverifikasi ulang dengan
> `.\run.ps1 test` serta `.\run.ps1 split`.

---

## 3.1 Gambaran Umum

Penelitian ini memakai dua dataset publik pengenalan ekspresi wajah yang telah
menjadi acuan luas: **CK+** dan **JAFFE**. Keduanya dipilih karena mewakili dua
karakteristik berbeda:

| Aspek | CK+ | JAFFE |
|---|---|---|
| Jumlah citra | 981 | 213 |
| Jumlah subjek | 118 | 10 |
| Jumlah kelas | 7 | 7 |
| Resolusi berkas yang dipakai | 48 × 48 | 128 × 128 |
| Format berkas yang dipakai | PNG | PNG |
| Citra per subjek (median) | 9 | 21 |
| Rentang citra per subjek | 3 – 18 | 20 – 23 |

> Berkas JAFFE asli dari sumbernya berformat TIFF 256 × 256; versi yang dipakai
> di repositori ini sudah dikonversi menjadi PNG 128 × 128. CK+ sudah beresolusi
> 48 × 48 sejak awal.

Perbedaan paling penting bukan pada jumlah citra, melainkan pada **struktur
pengambilan data**. CK+ diambil dari sekuens video sehingga memuat beberapa
frame berurutan dari orang yang sama, sedangkan JAFFE memotret ekspresi
terisolasi. Perbedaan ini terbukti berpengaruh besar terhadap validitas
pengukuran (bagian 3.6).

## 3.2 Dataset CK+

*The Extended Cohn-Kanade* (CK+) merupakan perluasan dari dataset Cohn-Kanade.
Data diambil dari rekaman video subjek yang diminta menampilkan ekspresi
tertentu, lalu frame terakhir setiap sekuens dianotasi.

| Kelas | Jumlah citra | Proporsi |
|---|---|---|
| anger | 135 | 13,76% |
| contempt | 54 | 5,50% |
| disgust | 177 | 18,04% |
| fear | 75 | 7,65% |
| happy | 207 | 21,10% |
| sadness | 84 | 8,56% |
| surprise | 249 | 25,38% |
| **Total** | **981** | **100%** |

Dataset ini **tidak seimbang**: kelas `surprise` hampir lima kali lebih banyak
dibanding `contempt`. Karena itu seluruh evaluasi melaporkan **macro
sensitivity** dan **macro specificity**, bukan hanya akurasi, agar kelas minoritas
tetap terlihat.

Ke-981 citra berasal dari 327 sekuens, masing-masing menyumbang **3 frame
berurutan** dari subjek yang sama. Konsekuensinya, satu subjek dapat menyumbang
banyak citra yang hampir identik — inilah sumber masalah validasi yang dibahas
di 3.6.

Dataset CK+ tidak memiliki kelas `neutral` pada versi yang dipakai.

## 3.3 Dataset JAFFE

*Japanese Female Facial Expression* (JAFFE) memuat 213 citra dari 10 subjek
perempuan Jepang, masing-masing menampilkan tujuh ekspresi.

| Kelas | Jumlah citra | Proporsi |
|---|---|---|
| anger | 30 | 14,08% |
| disgust | 29 | 13,62% |
| fear | 32 | 15,02% |
| happy | 31 | 14,55% |
| neutral | 30 | 14,08% |
| sadness | 31 | 14,55% |
| surprise | 30 | 14,08% |
| **Total** | **213** | **100%** |

Berbeda dari CK+, JAFFE hampir **seimbang** (29–32 citra per kelas) dan setiap
subjek menyumbang 20–23 citra, yaitu hampir seluruh tujuh ekspresinya. Struktur
ini membuat JAFFE lebih rentan terhadap kebocoran identitas pada split acak,
meski jumlah subjeknya hanya sepuluh.

Kode subjek diambil dari nama berkas, misalnya `KA.AN1.39.tiff` berarti subjek
`KA` dengan ekspresi `AN` (anger).

## 3.4 Praproses

Seluruh citra melewati tiga langkah yang sama agar vektor fitur berukuran tetap
dan perbandingan antar-dataset adil.

### 3.4.1 Grayscale

Citra diubah menjadi satu kanal intensitas. Alasannya, informasi ekspresi
terletak pada geometri dan distribusi intensitas wajah, bukan warna; kedua
metode ekstraksi fitur yang dipakai (SWT dan PHOG) juga bekerja pada citra satu
kanal.

### 3.4.2 Resize ke 128 × 128

PHOG membutuhkan jumlah sel piramida yang tetap, sehingga ukuran citra harus
seragam. Interpolasi dipilih secara adaptif:

| Kondisi | Interpolasi | Alasan |
|---|---|---|
| Pengecilan (citra lebih besar dari 128 × 128) | `INTER_AREA` | merata-ratakan piksel, tidak menimbulkan aliasing |
| Pembesaran (CK+ 48 → 128) | `INTER_CUBIC` | mempertahankan kontur wajah lebih baik daripada `INTER_NEAREST` |

Pada dataset yang dipakai, hanya CK+ yang benar-benar mengalami perubahan ukuran
(48 → 128, pembesaran). JAFFE sudah 128 × 128 sehingga resize bersifat
identitas.

### 3.4.3 Histogram Equalization

Kontras dinormalisasi dengan `cv2.equalizeHist`. Tujuannya menekan pengaruh
pencahayaan yang berbeda antar-sesi pengambilan gambar, sehingga fitur lebih
bergantung pada bentuk ekspresi daripada kondisi lampu.

### 3.4.4 Tanpa Pembagian Train/Test di Tahap Ini

Praproses **tidak** membagi data menjadi train dan test. Pembagian dilakukan
saat evaluasi memakai *k-fold cross-validation* berstrata, karena pembagian
tetap akan mengubah komposisi kelas pada dataset yang tidak seimbang.

## 3.5 Artefak dan Serah-Terima Antar Tahap

Hasil praproses disimpan sebagai `.npz` (bukan pickle) agar portabel dan tidak
menjalankan kode saat dimuat.

| Kunci | Bentuk | Tipe | Keterangan |
|---|---|---|---|
| `X` | CK+ `(981, 128, 128)`, JAFFE `(213, 128, 128)` | `uint8` | citra hasil praproses |
| `y` | `(N,)` | `int8` | indeks kelas, urut terhadap `class_names` |
| `subjects` | `(N,)` | `str` | kode subjek (`S010` untuk CK+, `KA` untuk JAFFE) |
| `class_names` | `(7,)` | `str` | nama kelas urut abjad |

Kunci `subjects` disimpan secara sengaja: tanpa informasi ini, evaluasi
*subject-independent* pada 3.6 tidak mungkin dilakukan.

Pemetaan label:

- **CK+** → 0 anger, 1 contempt, 2 disgust, 3 fear, 4 happy, 5 sadness, 6 surprise
- **JAFFE** → 0 anger, 1 disgust, 2 fear, 3 happy, 4 neutral, 5 sadness, 6 surprise

## 3.6 Validasi Protokol: Mengapa Split Acak Menyesatkan

Bagian ini adalah temuan metodologis utama dari tahap data, dan menjadi dasar
pemilihan protokol evaluasi pada bab hasil.

### 3.6.1 Perbandingan Dua Protokol

Pipeline identik (LDA → StandardScaler → LS-SVM One-vs-One, 5-fold) dijalankan
pada dua protokol pembagian data:

| Dataset | Protokol | Akurasi | Std |
|---|---|---|---|
| CK+ | acak (`StratifiedKFold`) | **96,64%** | 0,94 |
| CK+ | per subjek (`StratifiedGroupKFold`) | **48,21%** | 4,23 |
| JAFFE | acak | **80,31%** | 8,23 |
| JAFFE | per subjek | **38,22%** | 10,54 |

Selisih: **48,43 poin persen** (CK+) dan **42,09 poin persen** (JAFFE).

Pada protokol acak, frame dari subjek yang sama dapat muncul di data latih dan
data uji sekaligus. Model kemudian dapat mengenali **orangnya**, bukan
ekspresinya, sehingga akurasi terlihat tinggi.

### 3.6.2 Bukti Kuantitatif: Fitur Memuat Identitas

Untuk memastikan penyebabnya, dilakukan diagnosa terpisah: sebuah klasifier
dilatih menebak **siapa subjek** dari vektor fitur 680 dimensi.

| Pengukuran | CK+ | JAFFE |
|---|---|---|
| Menebak subjek dari fitur (split acak) | **98,58%** | **99,53%** |
| Peluang acak menebak subjek | 1,03% | 10,00% |

Fitur hybrid SWT + PHOG hampir sepenuhnya memuat identitas orang. Hal ini wajar:
PHOG mendeskripsikan bentuk wajah, dan bentuk wajah per individu jauh lebih
menonjol daripada perubahan ekspresinya.

### 3.6.3 Implikasi

1. Akurasi split acak **tidak boleh** diklaim sebagai kemampuan generalisasi
   pada dataset yang memuat banyak citra per subjek.
2. Angka yang dilaporkan untuk klaim generalisasi adalah protokol
   **subject-independent**.
3. Angka split acak tetap ditampilkan, tetapi hanya sebagai pembanding terhadap
   angka jurnal yang memakai protokol serupa.

Rincian lengkap, termasuk efek reduksi LDA pada protokol jujur, disajikan pada
bab hasil.

## 3.7 Deviasi terhadap Jurnal Acuan

Praproses pada penelitian ini **tidak identik** dengan Kar et al. (2021) bagian
4.1. Perbedaan dilakukan secara sadar dan dicatat agar tidak menimbulkan klaim
yang berlebihan.

| Tahap | Jurnal | Penelitian ini | Alasan |
|---|---|---|---|
| Normalisasi kontras | saturasi 1% piksel atas dan bawah (*contrast stretching*) | histogram equalization | keduanya menormalkan iluminasi; equalization tidak bergantung pada persentil per citra sehingga artefak lebih stabil |
| Deteksi wajah | Viola–Jones lalu *crop* | tidak dilakukan, langsung resize | kedua dataset sudah berupa citra wajah ter-*crop*; `cv2.CascadeClassifier` juga dihapus pada OpenCV 5 |
| Jumlah citra CK+ | 450 citra, 8 kelas | 981 citra, 7 kelas | menyesuaikan versi dataset yang tersedia |

**Konsekuensi:** angka akurasi penelitian ini **tidak dapat dibandingkan
langsung** dengan angka jurnal. Jurnal dipakai sebagai rujukan metode, bukan
sebagai target numerik yang harus disamai.

## 3.8 Ringkasan Bab

1. Dua dataset dipakai: CK+ (981 citra, 118 subjek, tidak seimbang) dan JAFFE
   (213 citra, 10 subjek, hampir seimbang).
2. Praproses seragam: grayscale → resize 128 × 128 dengan interpolasi adaptif →
   histogram equalization.
3. Artefak `.npz` menyimpan `X`, `y`, `subjects`, dan `class_names`; kunci
   `subjects` memungkinkan evaluasi per subjek.
4. Ditemukan bahwa split acak menggelembungkan akurasi hingga ~48 poin persen,
   karena fitur memuat identitas subjek hingga 98–99%.
5. Karena itu protokol subject-independent dipakai untuk seluruh klaim
   generalisasi.

---

## Catatan Penulisan (bukan bagian bab)

Yang masih perlu dilengkapi saat penyusunan skripsi:

- [ ] Nomor gambar untuk contoh citra asli vs hasil praproses (belum dibuat).
- [ ] Nomor tabel final (Tabel 3.1, 3.2, …) setelah digabung dengan bab lain.
- [ ] Sitasi formal CK+ (Lucey et al., 2010) dan JAFFE (Lyons et al., 1998)
      pada 3.2 dan 3.3 — saat ini baru disebut naratif.
- [ ] Konfirmasi ke pembimbing apakah temuan 3.6 diletakkan di Bab 3 (sebagai
      validitas data) atau dipindah ke Bab 4 (sebagai hasil).
- [ ] Perbandingan dengan jurnal pada 3.7 perlu dicek ulang terhadap teks PDF
      bagian 4.1 agar kutipannya presisi.
