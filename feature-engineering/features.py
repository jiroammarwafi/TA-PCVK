"""
Modul Ekstraksi Fitur Hybrid: SWT Haar Level-1 + PHOG.

Replikasi Jurnal:
Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021).
"A hybrid feature descriptor with Jaya optimised least squares SVM for
facial expression recognition", IET Image Processing, 15(7), 1471–1483.
DOI: 10.1049/ipr2.12118

Modul ini mengimplementasikan:
1. swt_haar_level1(img) -> Dekomposisi Stationary Wavelet Transform Haar level-1
2. phog(img, L=3, K=8)  -> Pyramid of Histograms of Oriented Gradients pada subband A1
3. extract_features(X)  -> Ekstraksi fitur lengkap untuk citra tunggal atau batch (N, 680)
"""

from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
import pywt


def swt_haar_level1(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Dekomposisi Stationary Wavelet Transform (SWT) level-1 dengan wavelet Haar.

    Berbeda dengan DWT biasa yang melakukan desimasi (downsampling) sehingga
    kehilangan sifat translation-invariance, SWT tidak melakukan downsampling
    sehingga bersifat shift-invariant dan mempertahankan resolusi spasial penuh (128x128).

    Sesuai Jurnal Bagian §4.2.1 & §5.1 (hal. 1475, 1478):
    Menghasilkan 4 subband frekuensi berukuran 128x128 piksel:
    - A1: Approximation coefficients (LL - frekuensi rendah utama kontur wajah)
    - H1: Horizontal detail coefficients (LH)
    - V1: Vertical detail coefficients (HL)
    - D1: Diagonal detail coefficients (HH)

    Parameters
    ----------
    img : np.ndarray
        Citra wajah 2D grayscale (mis. 128x128 piksel), tipe data uint8 atau float.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary berisi empat subband: {"A1": cA, "H1": cH, "V1": cV, "D1": cD}.
        Semua subband bertipe float64 dan berdimensi sama dengan citra input.
    """
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)

    if img.ndim != 2:
        raise ValueError(
            f"Citra harus berupa array 2D (grayscale), tetapi diterima array berdimensi {img.shape} (ndim={img.ndim})."
        )

    # Konversi ke float64 jika belum (syarat kestabilan numerik PyWavelets)
    if img.dtype != np.float64:
        img_float = img.astype(np.float64)
    else:
        img_float = img

    # SWT memerlukan dimensi kelipatan 2^level (128x128 memenuhi syarat)
    h, w = img_float.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(
            f"Dimensi citra ({h}, {w}) harus genap untuk SWT level-1."
        )

    # pywt.swt2 dengan trim_approx=True menghasilkan [cA, (cH, cV, cD)] untuk level 1
    coeffs = pywt.swt2(img_float, wavelet="haar", level=1, trim_approx=True)

    cA = coeffs[0]
    cH, cV, cD = coeffs[1]

    return {
        "A1": cA,
        "H1": cH,
        "V1": cV,
        "D1": cD,
    }


def compute_gradients(
    image: np.ndarray,
    ksize: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Menghitung magnitudo dan sudut orientasi gradien citra menggunakan operator Sobel.

    Parameters
    ----------
    image : np.ndarray
        Citra 2D (subband A1).
    ksize : int, default=3
        Ukuran kernel Sobel (3x3).

    Returns
    -------
    magnitude : np.ndarray
        Peta magnitudo gradien sqrt(dx^2 + dy^2), dtype float64.
    angle_deg : np.ndarray
        Peta sudut orientasi dalam derajat rentang [0, 360), dtype float64.
    """
    img_f = np.asarray(image, dtype=np.float64)

    # Gradien horizontal (dx) dan vertikal (dy)
    dx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=ksize)

    # Magnitudo gradien
    magnitude = np.hypot(dx, dy)

    # Sudut orientasi [0, 360) derajat
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.rad2deg(angle_rad)
    angle_deg = (angle_deg + 360.0) % 360.0

    return magnitude, angle_deg


def phog(
    img: np.ndarray,
    L: int = 3,
    K: int = 8,
    angle_range: Tuple[float, float] = (0.0, 360.0),
    norm: str = "per_level",
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Ekstraksi Pyramid of Histograms of Orientation Gradients (PHOG).

    Sesuai Jurnal Kar et al. (2021) Bagian §4.2.2 & §5.1 (hal. 1475, 1478):
    - Dihitung HANYA pada subband aproksimasi A1 (LL).
    - L = 3 tingkat resolusi piramida spasial:
        Level 0: 2^0 x 2^0 = 1 sel   -> 1 x 8 = 8 bin
        Level 1: 2^1 x 2^1 = 4 sel   -> 4 x 8 = 32 bin
        Level 2: 2^2 x 2^2 = 16 sel  -> 16 x 8 = 128 bin
        Level 3: 2^3 x 2^3 = 64 sel  -> 64 x 8 = 512 bin
        Total sel = 1 + 4 + 16 + 64 = 85 sel.
    - K = 8 bin orientasi dalam rentang sudut [0 - 360°] (lebar 45° per interval).
    - Tiap sel diisi histogram orientasi terbobot magnitudo gradien.
    - Total panjang vektor fitur final = 85 x 8 = 680 dimensi.

    Catatan Skema Normalisasi:
    Jurnal menyatakan 'further normalised considering all the pyramid levels'.
    Implementasi default menggunakan normalisasi L2 per level piramida ('per_level')
    agar bobot level 0 (1 sel) tidak tertelan oleh level 3 (64 sel), kemudian
    seluruh level dikonkatenasi menjadi vektor 680-D.

    Parameters
    ----------
    img : np.ndarray
        Citra 2D (subband A1 dari SWT), ukuran 128x128.
    L : int, default=3
        Kedalaman piramida spasial (level 0 s.d. 3).
    K : int, default=8
        Jumlah bin orientasi gradien.
    angle_range : Tuple[float, float], default=(0.0, 360.0)
        Rentang orientasi gradien.
    norm : str, default='per_level'
        Metode normalisasi: 'per_level' (L2 per level), 'global' (L2 global), atau 'none'.
    eps : float, default=1e-7
        Epsilon pencegah pembagian nol saat normalisasi.

    Returns
    -------
    np.ndarray
        Vektor fitur 1D berukuran (680,) bertipe float32.
    """
    if img.ndim != 2:
        raise ValueError(
            f"Input PHOG harus berupa citra 2D, didapat shape {img.shape}"
        )

    h, w = img.shape
    magnitude, angle_deg = compute_gradients(img)

    min_ang, max_ang = angle_range
    ang_span = max_ang - min_ang
    bin_width = ang_span / K

    # Tentukan indeks bin orientasi tiap piksel (0 s.d. K-1)
    rel_angle = np.clip(angle_deg - min_ang, 0.0, ang_span - 1e-9)
    bin_idx = np.floor(rel_angle / bin_width).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, K - 1)

    # Hitung histogram pada grid terdalam (2^L x 2^L sel)
    # Untuk L=3 dan citra 128x128: 8x8 sel, tiap sel berukuran 16x16 piksel
    max_cells = 2 ** L
    cell_h = h / max_cells
    cell_w = w / max_cells

    finest_grid = np.zeros((max_cells, max_cells, K), dtype=np.float64)
    for i in range(max_cells):
        r0 = int(round(i * cell_h))
        r1 = int(round((i + 1) * cell_h))
        for j in range(max_cells):
            c0 = int(round(j * cell_w))
            c1 = int(round((j + 1) * cell_w))

            c_bins = bin_idx[r0:r1, c0:c1].ravel()
            c_mags = magnitude[r0:r1, c0:c1].ravel()
            finest_grid[i, j] = np.bincount(c_bins, weights=c_mags, minlength=K)

    # Bangun fitur piramida dari level 0 s.d. L
    level_features = []
    for l in range(L + 1):
        cells_l = 2 ** l
        step = max_cells // cells_l
        hists_l = []

        for i in range(cells_l):
            for j in range(cells_l):
                block = finest_grid[i * step : (i + 1) * step, j * step : (j + 1) * step]
                cell_hist = block.sum(axis=(0, 1))
                hists_l.append(cell_hist)

        v_l = np.concatenate(hists_l).astype(np.float32)

        if norm == "per_level":
            norm_val = np.linalg.norm(v_l)
            if norm_val > eps:
                v_l = v_l / norm_val

        level_features.append(v_l)

    # Gabungkan seluruh level piramida (L0 + L1 + L2 + L3)
    phog_vec = np.concatenate(level_features).astype(np.float32)

    if norm == "global":
        norm_val = np.linalg.norm(phog_vec)
        if norm_val > eps:
            phog_vec = phog_vec / norm_val

    return phog_vec


def extract_features(
    X: np.ndarray,
    L: int = 3,
    K: int = 8,
    norm: str = "per_level",
) -> np.ndarray:
    """
    Mengekstraksi fitur hybrid menyeluruh (SWT Haar Level-1 -> A1 -> PHOG).

    Sesuai Jurnal Persamaan (3)–(4):
        [Aj, Dj] = SWT(I)
        Fvec     = PHOG(Aj)

    Parameters
    ----------
    X : np.ndarray
        Array citra wajah 2D (128, 128) atau batch citra 3D (N, 128, 128).
    L : int, default=3
        Level piramida PHOG.
    K : int, default=8
        Jumlah bin orientasi PHOG.
    norm : str, default='per_level'
        Skema normalisasi PHOG.

    Returns
    -------
    np.ndarray
        Array 1D berukuran (680,) jika input citra tunggal,
        atau matriks 2D berukuran (N, 680) jika input batch (N, 128, 128).
        Tipe data float32, bebas NaN dan Inf.
    """
    if X.ndim == 2:
        # Citra tunggal
        subbands = swt_haar_level1(X)
        a1 = subbands["A1"]
        return phog(a1, L=L, K=K, norm=norm)

    elif X.ndim == 3:
        # Batch citra (N, H, W)
        n_samples = X.shape[0]
        expected_dim = K * sum(4 ** l for l in range(L + 1))  # 680
        features = np.empty((n_samples, expected_dim), dtype=np.float32)

        for i in range(n_samples):
            subbands = swt_haar_level1(X[i])
            a1 = subbands["A1"]
            features[i] = phog(a1, L=L, K=K, norm=norm)

        return features

    else:
        raise ValueError(
            f"Input X harus berdimensi 2 (H, W) atau 3 (N, H, W), diterima bentuk {X.shape} (ndim={X.ndim})."
        )
