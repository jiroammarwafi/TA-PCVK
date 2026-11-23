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

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import argparse
import os
import sys
import time

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


# ============================================================================
#  Orkestrasi Ekstraksi Fitur Citra Ber-Noise (A2 Handoff ke A3)
# ============================================================================

DEFAULT_VARIANCES: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.1)
DATASET_KEYS: Tuple[str, ...] = ("ck", "jaffe")


def _config_root(config: Dict[str, Any]) -> Path:
    raw = config.get("_config_path")
    if raw:
        return Path(str(raw)).resolve().parent
    return Path.cwd()


def _resolve_path(config: Dict[str, Any], value: Union[str, Path]) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return _config_root(config) / p


def noise_feature_path(
    config: Dict[str, Any],
    dataset_key: str,
    variance: float,
) -> Path:
    """Path artefak fitur citra ber-noise, mis. data/a2-features/ck_features_noise_v0.01.npz."""
    raw_dir = config.get("paths", {}).get("a2_features_dir", "data/a2-features")
    features_dir = _resolve_path(config, raw_dir)
    return features_dir / f"{dataset_key}_features_noise_v{float(variance):.2f}.npz"


def noise_input_image_path(
    config: Dict[str, Any],
    dataset_key: str,
    variance: float,
) -> Path:
    """Path artefak citra ber-noise dari A1, mis. data/a1-noise/ck_noise_v0.01.npz."""
    raw_dir = config.get("paths", {}).get("a1_noise_dir", "data/a1-noise")
    noise_dir = _resolve_path(config, raw_dir)
    return noise_dir / f"{dataset_key}_noise_v{float(variance):.2f}.npz"


def build_noise_features(
    config: Dict[str, Any],
    variances: Optional[Sequence[float]] = None,
    dataset_keys: Optional[Iterable[str]] = None,
    force: bool = False,
    verbose: bool = True,
) -> List[Path]:
    """
    Mengekstrak fitur 680-D (SWT Haar Level-1 + PHOG) dari seluruh artefak
    citra ber-noise yang dihasilkan A1 (data/a1-noise/), lalu mengekspornya ke
    data/a2-features/ untuk dikonsumsi oleh A3.

    Parameters
    ----------
    config : dict
        Konfigurasi proyek dari config.yaml.
    variances : sequence of float, optional
        Daftar variansi noise (default: 0.01, 0.02, 0.05, 0.1).
    dataset_keys : iterable of str, optional
        Kunci dataset ('ck', 'jaffe').
    force : bool, default=False
        Bila False, lewati artefak yang sudah ada di disk.
    verbose : bool, default=True
        Cetak informasi progres ke terminal.

    Returns
    -------
    written : list[Path]
        Daftar path artefak fitur yang ditulis atau sudah ada.
    """
    variances = tuple(DEFAULT_VARIANCES if variances is None else variances)
    keys = tuple(DATASET_KEYS if dataset_keys is None else dataset_keys)

    raw_out = config.get("paths", {}).get("a2_features_dir", "data/a2-features")
    out_dir = _resolve_path(config, raw_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    phog_cfg = config.get("feature_engineering", {}).get("phog", {})
    L = int(phog_cfg.get("pyramid_levels", 3))
    K = int(phog_cfg.get("num_bins", 8))
    norm = phog_cfg.get("norm", "per_level")

    written: List[Path] = []

    for key in keys:
        if verbose:
            print(f"[noise-features] Dataset: {key.upper()}")

        for var in variances:
            dst = noise_feature_path(config, key, var)
            if dst.exists() and not force:
                if verbose:
                    print(f"  [skip] {dst.name} sudah ada")
                written.append(dst)
                continue

            src = noise_input_image_path(config, key, var)
            if not src.exists():
                if verbose:
                    print(f"  [lewati] {src.name} belum ada di {src.parent}")
                continue

            with np.load(src, allow_pickle=True) as data:
                X = data["X"]
                y = data["y"]
                subjects = data["subjects"]
                class_names = data["class_names"]
                seed = int(data["seed"]) if "seed" in data else 42

            t0 = time.perf_counter()
            F = extract_features(X, L=L, K=K, norm=norm)
            dt = time.perf_counter() - t0

            assert F.shape == (len(X), 680), f"Dimensi tidak cocok: {F.shape}"
            assert np.isfinite(F).all(), f"Terdapat NaN/Inf pada {dst.name}"

            np.savez_compressed(
                dst,
                F=F,
                y=y,
                subjects=subjects,
                class_names=class_names,
                variance=float(var),
                seed=seed,
            )
            written.append(dst)
            if verbose:
                print(f"  [tulis] {dst.name} ({F.shape[0]} citra -> {F.shape[1]}-D dalam {dt:.2f}s)")

    return written


def main() -> None:
    """CLI runner untuk ekstraksi fitur A2."""
    parser = argparse.ArgumentParser(
        description="Ekstraksi fitur hybrid SWT + PHOG (A2) untuk citra normal dan ber-noise."
    )
    parser.add_argument("--config", default="config.yaml", help="Path config.yaml")
    parser.add_argument("--noise", action="store_true", help="Ekstrak fitur dari artefak citra ber-noise")
    parser.add_argument(
        "--variances",
        nargs="+",
        type=float,
        default=list(DEFAULT_VARIANCES),
        help="Variansi noise yang diproses",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_KEYS),
        default=list(DATASET_KEYS),
        help="Dataset yang diproses ('ck', 'jaffe')",
    )
    parser.add_argument("--force", action="store_true", help="Paksa ekstrak ulang file yang sudah ada")
    args = parser.parse_args()

    # Load config via baseline helper or PyYAML directly
    config_path = Path(args.config)
    if not config_path.exists() and (Path("..") / args.config).exists():
        config_path = Path("..") / args.config

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(config_path.resolve())

    if args.noise:
        print("=== A2: EKSTRAKSI FITUR CITRA BER-NOISE GAUSSIAN ===")
        out = build_noise_features(
            config,
            variances=args.variances,
            dataset_keys=args.datasets,
            force=args.force,
            verbose=True,
        )
        print(f"\nSelesai. {len(out)} artefak fitur noise tersedia di {config['paths']['a2_features_dir']}/")
    else:
        # Default extraction untuk dataset bersih (CK+ dan JAFFE)
        print("=== A2: EKSTRAKSI FITUR CITRA BERSIH ===")
        for key in args.datasets:
            pre_path = _resolve_path(config, config["paths"][f"{key}_preprocessed"])
            feat_path = _resolve_path(config, config["paths"][f"{key}_features"])
            if not pre_path.exists():
                print(f"[lewati] {key}: {pre_path} tidak ditemukan")
                continue
            with np.load(pre_path, allow_pickle=True) as data:
                X = data["X"]
                y = data["y"]
                subjects = data["subjects"]
                class_names = data["class_names"]
            t0 = time.perf_counter()
            F = extract_features(X)
            dt = time.perf_counter() - t0
            np.savez_compressed(
                feat_path,
                F=F,
                y=y,
                subjects=subjects,
                class_names=class_names,
            )
            print(f"[tulis] {feat_path.name}: {F.shape} dalam {dt:.2f}s")


if __name__ == "__main__":
    main()
