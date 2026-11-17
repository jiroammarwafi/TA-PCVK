"""
Modul Ekstraksi Fitur Hybrid: SWT Haar Level-1 + PHOG.

Replikasi Jurnal:
Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021).
"A hybrid feature descriptor with Jaya optimised least squares SVM for
facial expression recognition", IET Image Processing, 15(7), 1471-1483.
DOI: 10.1049/ipr2.12118

Modul ini menyediakan:
1. swt_haar_level1(img) -> Dekomposisi SWT Haar level-1 (A1, H1, V1, D1)
2. phog(img, L=3, K=8)  -> PHOG 680-D pada subband aproksimasi A1
3. extract_features(X)  -> Ekstraksi fitur citra tunggal atau batch (N, 680)

Catatan ketahanan lingkungan:
PyWavelets bersifat opsional. Bila DLL-nya diblokir kebijakan Application
Control (WDAC/Device Guard) atau tidak tersedia untuk versi Python yang
dipakai, modul otomatis memakai implementasi SWT Haar murni numpy yang
setara (selisih < 1e-12 terhadap pywt.swt2).
"""

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

try:  # pragma: no cover - bergantung kebijakan mesin
    import pywt

    HAS_PYWT = True
except Exception:  # pragma: no cover
    pywt = None
    HAS_PYWT = False


def _swt2_haar_numpy(img: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    SWT Haar level-1 murni numpy (tanpa PyWavelets).

    Filter Haar ternormalisasi: h = [1/sqrt2, 1/sqrt2], g = [1/sqrt2, -1/sqrt2].
    Padding periodik (setara mode 'periodic' PyWavelets) sehingga bersifat
    shift-invariant dan ukuran keluaran sama dengan masukan.

    Diverifikasi identik dengan ``pywt.swt2(img, 'haar', 1)`` (selisih < 1e-12),
    sehingga fitur yang dihasilkan tetap sebanding dengan artefak yang dibuat
    memakai PyWavelets.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"Citra harus 2D, diterima shape {a.shape}")

    p = np.pad(a, ((0, 1), (0, 1)), mode="wrap")
    x00 = p[:-1, :-1]
    x01 = p[:-1, 1:]

    s = 1.0 / np.sqrt(2.0)

    # Arah baris (horizontal)
    row_L = (x00 + x01) * s
    row_H = (x00 - x01) * s

    # Arah kolom (vertikal), digeser satu baris
    row_L_down = np.roll(row_L, -1, axis=0)
    row_H_down = np.roll(row_H, -1, axis=0)

    cA = (row_L + row_L_down) * s          # LL (aproksimasi)
    cH = (row_L - row_L_down) * s          # LH
    cV = (row_H + row_H_down) * s          # HL
    cD = (row_H - row_H_down) * s          # HH

    return cA, (cH, cV, cD)


def swt_haar_level1(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Dekomposisi Stationary Wavelet Transform (SWT) level-1 dengan wavelet Haar.

    Sesuai Jurnal Kar et al. (2021) Bagian 4.2.1 & 5.1:
    - Input: Citra grayscale 2D berukuran 128x128.
    - Transformasi undecimated (tanpa downsampling), shift-invariant.
    - Menghasilkan 4 subband 128x128 float64: A1 (LL), H1 (LH), V1 (HL), D1 (HH).

    Memakai PyWavelets bila tersedia; jika tidak, memakai implementasi numpy
    yang setara (``_swt2_haar_numpy``).
    """
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)

    if img.ndim != 2:
        raise ValueError(
            f"Citra harus berupa array 2D (grayscale), tetapi diterima array "
            f"berdimensi {img.shape} (ndim={img.ndim})."
        )

    img_float = img.astype(np.float64) if img.dtype != np.float64 else img

    h, w = img_float.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"Dimensi citra ({h}, {w}) harus genap untuk SWT level-1.")

    if HAS_PYWT:
        # pywt.swt2 dengan trim_approx=True -> [cA, (cH, cV, cD)]
        coeffs = pywt.swt2(img_float, wavelet="haar", level=1, trim_approx=True)
        cA = coeffs[0]
        cH, cV, cD = coeffs[1]
    else:
        cA, (cH, cV, cD) = _swt2_haar_numpy(img_float)

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
    dx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=ksize)

    magnitude = np.hypot(dx, dy)
    angle_deg = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
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
    Ekstraksi PHOG dari subband A1.

    Sesuai Jurnal Kar et al. (2021) Bagian 4.2.2, Persamaan (2):
        F = K * sum_{l=0..L} 4^l
    Dengan L=3 dan K=8 -> 8 * 85 = 680 dimensi.

    Returns
    -------
    np.ndarray
        Vektor 1D float32 berisi 680 nilai.
    """
    if img.ndim != 2:
        raise ValueError(f"Input PHOG harus berupa citra 2D, didapat shape {img.shape}")

    h, w = img.shape
    magnitude, angle_deg = compute_gradients(img)

    min_ang, max_ang = angle_range
    ang_span = max_ang - min_ang
    bin_width = ang_span / K

    rel_angle = np.clip(angle_deg - min_ang, 0.0, ang_span - 1e-9)
    bin_idx = np.floor(rel_angle / bin_width).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, K - 1)

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

    level_features = []
    for l in range(L + 1):
        cells_l = 2 ** l
        step = max_cells // cells_l
        hists_l = []

        for i in range(cells_l):
            for j in range(cells_l):
                block = finest_grid[i * step:(i + 1) * step, j * step:(j + 1) * step]
                cell_hist = block.sum(axis=(0, 1))
                hists_l.append(cell_hist)

        v_l = np.concatenate(hists_l).astype(np.float32)

        if norm == "per_level":
            norm_val = np.linalg.norm(v_l)
            if norm_val > eps:
                v_l = v_l / norm_val

        level_features.append(v_l)

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
    Mengekstraksi fitur hybrid menyeluruh (SWT Haar Level-1 -> Subband A1 -> PHOG).

    Sesuai Jurnal Kar et al. (2021) Persamaan (3)-(4):
        [Aj, Dj] = SWT(I)
        Fvec     = PHOG(Aj)

    Parameters
    ----------
    X : np.ndarray
        Array citra wajah 2D (128, 128) atau batch citra 3D (N, 128, 128).
    L : int, default=3
        Level piramida PHOG (85 sel).
    K : int, default=8
        Jumlah bin orientasi PHOG.
    norm : str, default='per_level'
        Skema normalisasi PHOG.

    Returns
    -------
    np.ndarray
        Array 1D (680,) untuk citra tunggal atau matriks 2D (N, 680) untuk batch.
        Tipe data float32, tanpa NaN dan Inf.
    """
    if X.ndim == 2:
        subbands = swt_haar_level1(X)
        a1 = subbands["A1"]
        return phog(a1, L=L, K=K, norm=norm)

    elif X.ndim == 3:
        n_samples = X.shape[0]
        expected_dim = K * sum(4 ** l for l in range(L + 1))
        features = np.empty((n_samples, expected_dim), dtype=np.float32)

        for i in range(n_samples):
            subbands = swt_haar_level1(X[i])
            a1 = subbands["A1"]
            features[i] = phog(a1, L=L, K=K, norm=norm)

        return features

    else:
        raise ValueError(
            f"Input X harus berdimensi 2 (H, W) atau 3 (N, H, W), diterima bentuk "
            f"{X.shape} (ndim={X.ndim})."
        )
