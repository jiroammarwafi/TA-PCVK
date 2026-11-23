"""
Unit test suite untuk modul research/a2-feature-engineering/features.py.

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

# Modul fitur membutuhkan PyWavelets. Pada mesin dengan kebijakan Application
# Control (WDAC/Device Guard), DLL pywt dapat diblokir sehingga modul tidak
# bisa dimuat. Dalam kondisi itu seluruh test di file ini dilewati, bukan
# menggagalkan koleksi pytest — karena penyebabnya lingkungan, bukan kode.
try:
    import pywt  # noqa: F401
except Exception as _exc:  # pragma: no cover - bergantung kebijakan mesin
    pytest.skip(
        f"PyWavelets tidak dapat dimuat ({type(_exc).__name__}); "
        "test fitur dilewati karena kebijakan Application Control",
        allow_module_level=True,
    )

from features import (  # noqa: E402
    swt_haar_level1,
    compute_gradients,
    phog,
    extract_features,
    noise_feature_path,
    build_noise_features,
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
    root = Path(__file__).resolve().parents[1]
    ck_path = root / "data" / "a1-preprocessing" / "ck_preprocessed.npz"
    jaffe_path = root / "data" / "a1-preprocessing" / "jaffe_preprocessed.npz"

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


def test_noise_feature_path_pattern(tmp_path):
    """Memverifikasi pembentukan path artefak fitur ber-noise."""
    config = {
        "paths": {"a2_features_dir": str(tmp_path)},
        "_config_path": str(tmp_path / "config.yaml"),
    }
    p = noise_feature_path(config, "ck", 0.05)
    assert p.name == "ck_features_noise_v0.05.npz"
    assert p.parent == tmp_path


def test_build_noise_features_mock(tmp_path):
    """Memverifikasi proses ekstraksi fitur citra ber-noise dari artefak A1."""
    noise_dir = tmp_path / "a1-noise"
    feat_dir = tmp_path / "a2-features"
    noise_dir.mkdir()
    feat_dir.mkdir()

    # Buat dummy artefak citra ber-noise A1
    X_dummy = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
    y_dummy = np.array([0, 1, 2])
    subj_dummy = np.array(["S01", "S02", "S03"])
    classes_dummy = np.array(["c0", "c1", "c2"])

    noise_file = noise_dir / "ck_noise_v0.01.npz"
    np.savez_compressed(
        noise_file,
        X=X_dummy,
        y=y_dummy,
        subjects=subj_dummy,
        class_names=classes_dummy,
        variance=0.01,
        seed=42,
    )

    config = {
        "paths": {
            "a1_noise_dir": str(noise_dir),
            "a2_features_dir": str(feat_dir),
        },
        "_config_path": str(tmp_path / "config.yaml"),
    }

    out = build_noise_features(
        config,
        variances=[0.01],
        dataset_keys=["ck"],
        force=True,
        verbose=False,
    )

    assert len(out) == 1
    dst = out[0]
    assert dst.exists()
    assert dst.name == "ck_features_noise_v0.01.npz"

    with np.load(dst) as data:
        assert data["F"].shape == (3, 680)
        assert data["F"].dtype == np.float32
        assert np.isfinite(data["F"]).all()
        assert (data["y"] == y_dummy).all()
        assert (data["subjects"] == subj_dummy).all()
        assert float(data["variance"]) == 0.01

