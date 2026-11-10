"""
Modul Ekstraksi Fitur Hybrid: SWT Haar Level-1 + PHOG.

Replikasi Jurnal:
Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021).
"A hybrid feature descriptor with Jaya optimised least squares SVM for
facial expression recognition", IET Image Processing, 15(7), 1471–1483.
DOI: 10.1049/ipr2.12118

Modul ini menyediakan:
1. swt_haar_level1(img) -> Dekomposisi SWT Haar level-1 (A1, H1, V1, D1)
2. phog(img, L=3, K=8)  -> PHOG 680-D pada subband aproksimasi A1
3. extract_features(X)  -> Ekstraksi fitur citra tunggal atau batch (N, 680)
"""

from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
import pywt


def swt_haar_level1(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Dekomposisi Stationary Wavelet Transform (SWT) level-1 dengan wavelet Haar.
    Menghasilkan 4 subband 128x128 piksel: A1 (LL), H1 (LH), V1 (HL), D1 (HH).
    """
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)

    if img.ndim != 2:
        raise ValueError(
            f"Citra harus berupa array 2D (grayscale), tetapi diterima array berdimensi {img.shape} (ndim={img.ndim})."
        )

    img_float = img.astype(np.float64) if img.dtype != np.float64 else img

    h, w = img_float.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"Dimensi citra ({h}, {w}) harus genap untuk SWT level-1.")

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
    """Menghitung magnitudo dan orientasi sudut [0, 360) via Sobel."""
    img_f = np.asarray(image, dtype=np.float64)
    dx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.hypot(dx, dy)
    angle_rad = np.arctan2(dy, dx)
    angle_deg = (np.rad2deg(angle_rad) + 360.0) % 360.0
    return magnitude, angle_deg


def phog(
    img: np.ndarray,
    L: int = 3,
    K: int = 8,
    angle_range: Tuple[float, float] = (0.0, 360.0),
    norm: str = "per_level",
    eps: float = 1e-7,
) -> np.ndarray:
    """Ekstraksi PHOG dari subband A1 (85 sel x 8 bin = 680-D float32)."""
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
                block = finest_grid[i * step : (i + 1) * step, j * step : (j + 1) * step]
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
    """Ekstraksi fitur hybrid untuk citra tunggal (680,) atau batch (N, 680)."""
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
            f"Input X harus berdimensi 2 (H, W) atau 3 (N, H, W), diterima bentuk {X.shape} (ndim={X.ndim})."
        )
