"""
Unit test suite untuk modul feature-engineering/features.py.

Verifikasi:
1. Output swt_haar_level1 memiliki keys A1, H1, V1, D1 berukuran (128, 128) float64.
2. Dimensi PHOG L=3, K=8 tepat 680 dimensi dan tipe data float32.
3. Fungsi extract_features bekerja untuk citra tunggal (680,) dan batch (N, 680).
4. Tidak ada NaN atau Inf (finite values).
5. Kinerja stabil pada sampel nyata CK+ dan JAFFE serta kondisi citra polos.
"""

from pathlib import Path
import numpy as np
import pytest

from features import (
    swt_haar_level1,
    compute_gradients,
    phog,
    extract_features,
)


def test_swt_haar_level1_keys_and_shapes():
    """Memverifikasi SWT level-1 menghasilkan 4 subband 128x128 tanpa downsampling."""
    dummy = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    subbands = swt_haar_level1(dummy)

    assert isinstance(subbands, dict)
    for k in ["A1", "H1", "V1", "D1"]:
        assert k in subbands, f"Key {k} harus ada pada output SWT"
        assert subbands[k].shape == (128, 128), f"Subband {k} harus berukuran (128, 128)"
        assert subbands[k].dtype == np.float64
        assert not np.isnan(subbands[k]).any()
        assert not np.isinf(subbands[k]).any()


def test_compute_gradients():
    """Memverifikasi perhitungan gradien Sobel menghasilkan sudut [0, 360)."""
    img = np.zeros((128, 128), dtype=np.float64)
    img[:, 64:] = 255.0  # Pola tepi step vertikal

    mag, ang = compute_gradients(img)
    assert mag.shape == (128, 128)
    assert ang.shape == (128, 128)
    assert (mag >= 0.0).all()
    assert (ang >= 0.0).all()
    assert (ang < 360.0).all()


def test_phog_dimension_and_dtype():
    """Memverifikasi dimensi PHOG = 85 sel * 8 bin = 680 dan dtype float32."""
    dummy = np.random.rand(128, 128).astype(np.float64)
    feat = phog(dummy, L=3, K=8, norm="per_level")

    assert feat.shape == (680,)
    assert feat.dtype == np.float32
    assert np.isfinite(feat).all()

    # Cek per-level L2 normalization
    # Level 0 (8), Level 1 (32), Level 2 (128), Level 3 (512)
    v0, v1, v2, v3 = feat[0:8], feat[8:40], feat[40:168], feat[168:680]
    np.testing.assert_almost_equal(np.linalg.norm(v0), 1.0, decimal=5)
    np.testing.assert_almost_equal(np.linalg.norm(v1), 1.0, decimal=5)
    np.testing.assert_almost_equal(np.linalg.norm(v2), 1.0, decimal=5)
    np.testing.assert_almost_equal(np.linalg.norm(v3), 1.0, decimal=5)


def test_extract_features_single_and_batch():
    """Memverifikasi fungsi extract_features untuk input citra tunggal dan batch."""
    dummy_single = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    feat_single = extract_features(dummy_single)
    assert feat_single.shape == (680,)
    assert feat_single.dtype == np.float32
    assert np.isfinite(feat_single).all()

    dummy_batch = np.random.randint(0, 256, (4, 128, 128), dtype=np.uint8)
    feat_batch = extract_features(dummy_batch)
    assert feat_batch.shape == (4, 680)
    assert feat_batch.dtype == np.float32
    assert np.isfinite(feat_batch).all()


def test_real_dataset_samples():
    """Memverifikasi ekstraksi fitur pada sampel nyata dataset CK+ dan JAFFE."""
    ck_path = Path("data/ck_preprocessed.npz")
    jaffe_path = Path("data/jaffe_preprocessed.npz")

    if ck_path.exists():
        d_ck = np.load(ck_path)
        f_ck = extract_features(d_ck["X"][0])
        assert f_ck.shape == (680,)
        assert f_ck.dtype == np.float32
        assert np.isfinite(f_ck).all()

    if jaffe_path.exists():
        d_jf = np.load(jaffe_path)
        f_jf = extract_features(d_jf["X"][0])
        assert f_jf.shape == (680,)
        assert f_jf.dtype == np.float32
        assert np.isfinite(f_jf).all()


def test_edge_cases_zero_gradients():
    """Memverifikasi kestabilan pada citra polos hitam dan putih (gradien nol)."""
    black = np.zeros((128, 128), dtype=np.uint8)
    white = np.full((128, 128), 255, dtype=np.uint8)

    f_black = extract_features(black)
    f_white = extract_features(white)

    assert f_black.shape == (680,)
    assert f_white.shape == (680,)
    assert np.isfinite(f_black).all()
    assert np.isfinite(f_white).all()
    np.testing.assert_array_equal(f_black, 0.0)
    np.testing.assert_array_equal(f_white, 0.0)


def test_invalid_input_shape():
    """Memverifikasi penanganan error saat input tidak valid."""
    with pytest.raises(ValueError):
        extract_features(np.zeros((128, 128, 3)))
    with pytest.raises(ValueError):
        extract_features(np.zeros((128,)))
