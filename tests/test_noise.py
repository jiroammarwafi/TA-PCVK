"""
Unit test suite untuk modul noise Gaussian (Divisi A1).

Memverifikasi:
1. variance=0 menghasilkan citra identik (kontrol).
2. Shape, dtype, dan rentang nilai tetap terjaga.
3. Noise reprodusibel untuk seed yang sama dan berbeda untuk seed berbeda.
4. Statistik empiris mendekati variansi yang diminta (dalam toleransi clipping).
5. Artefak .npz memuat kunci yang dibutuhkan A2 (X, y, subjects, class_names).
6. Validasi input (dtype salah, variance negatif, dimensi salah).
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "a1-data-preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "a3-model-optimization"))

from noise import (  # noqa: E402
    DEFAULT_VARIANCES,
    add_gaussian_noise,
    build_noise_artifacts,
    load_noise_artifact,
    noise_artifact_path,
    noise_statistics,
)


def _sample_images(n=6, size=32, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, size, size)) * 255).astype(np.uint8)


def test_variance_nol_menghasilkan_salinan_identik():
    img = _sample_images()
    out = add_gaussian_noise(img, 0.0)
    assert np.array_equal(out, img)
    assert out is not img, "harus mengembalikan salinan, bukan objek yang sama"


def test_shape_dtype_dan_rentang_terjaga():
    img = _sample_images()
    for variance in DEFAULT_VARIANCES:
        out = add_gaussian_noise(img, variance)
        assert out.shape == img.shape
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255


def test_reprodusibel_dengan_seed_sama():
    img = _sample_images()
    a = add_gaussian_noise(img, 0.05, seed=42)
    b = add_gaussian_noise(img, 0.05, seed=42)
    assert np.array_equal(a, b)


def test_seed_berbeda_menghasilkan_noise_berbeda():
    img = _sample_images()
    a = add_gaussian_noise(img, 0.05, seed=1)
    b = add_gaussian_noise(img, 0.05, seed=2)
    assert not np.array_equal(a, b)


def test_variansi_empiris_mendekati_permintaan():
    """Clipping + kuantisasi uint8 membuat variansi empiris sedikit lebih kecil."""
    img = _sample_images(n=8, size=64, seed=3)
    for variance in (0.01, 0.05):
        out = add_gaussian_noise(img, variance, seed=7)
        stats = noise_statistics(img, out)
        # toleransi lebar: clipping di 0/1 dan pembulatan ke 256 level
        assert 0.5 * variance <= stats["empirical_variance"] <= 1.6 * variance, (
            variance,
            stats,
        )
        assert abs(stats["mean_shift"]) < 0.02


def test_noise_lebih_besar_menghasilkan_deviasi_lebih_besar():
    img = _sample_images(n=8, size=64, seed=4)
    kecil = noise_statistics(img, add_gaussian_noise(img, 0.01, seed=9))
    besar = noise_statistics(img, add_gaussian_noise(img, 0.1, seed=9))
    assert besar["empirical_std"] > kecil["empirical_std"]
    assert besar["max_abs_diff"] > kecil["max_abs_diff"]


def test_citra_2d_dukung():
    img = _sample_images(n=1, size=32)[0]
    out = add_gaussian_noise(img, 0.02)
    assert out.shape == img.shape


@pytest.mark.parametrize(
    "payload, error",
    [
        (np.zeros((4, 8, 8), dtype=np.float32), TypeError),
        (np.zeros((4, 8, 8), dtype=np.uint8), None),  # valid
    ],
)
def test_validasi_dtype(payload, error):
    if error is None:
        assert add_gaussian_noise(payload, 0.01).dtype == np.uint8
    else:
        with pytest.raises(error):
            add_gaussian_noise(payload, 0.01)


def test_variance_negatif_ditolak():
    with pytest.raises(ValueError):
        add_gaussian_noise(_sample_images(), -0.01)


def test_dimensi_tidak_valid_ditolak():
    with pytest.raises(ValueError):
        add_gaussian_noise(np.zeros((2, 2, 2, 2), dtype=np.uint8), 0.01)


# --------------------------------------------------------------------------- #
# artefak
# --------------------------------------------------------------------------- #
def _tulis_preprocessed(path: Path, n=6, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    np.savez_compressed(
        path,
        X=(rng.random((n, 16, 16)) * 255).astype(np.uint8),
        y=np.array([i % n_classes for i in range(n)], dtype=np.int8),
        subjects=np.array([f"S{i // 2:02d}" for i in range(n)]),
        class_names=np.array([f"c{i}" for i in range(n_classes)]),
    )


def test_build_artefak_memuat_kunci_untuk_a2(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _tulis_preprocessed(data_dir / "ck_preprocessed.npz")

    config = {
        "project": {"random_seed": 42},
        "paths": {
            "data_dir": str(data_dir),
            "ck_preprocessed": str(data_dir / "ck_preprocessed.npz"),
        },
    }

    written = build_noise_artifacts(
        config, variances=(0.0, 0.05), dataset_keys=("ck",), verbose=False
    )
    assert len(written) == 2

    art = load_noise_artifact(written[1])
    for key in ("X", "y", "subjects", "class_names", "variance", "seed"):
        assert key in art, f"kunci '{key}' hilang"
    assert art["variance"] == pytest.approx(0.05)
    assert art["X"].shape == (6, 16, 16)
    assert art["X"].dtype == np.uint8

    # variance 0 -> identik dengan sumber
    bersih = load_noise_artifact(written[0])
    asli = np.load(data_dir / "ck_preprocessed.npz", allow_pickle=True)
    assert np.array_equal(bersih["X"], asli["X"])


def test_build_artefak_melewati_dataset_tidak_ada(tmp_path):
    config = {
        "project": {"random_seed": 1},
        "paths": {
            "data_dir": str(tmp_path),
            "ck_preprocessed": str(tmp_path / "tidak_ada.npz"),
        },
    }
    assert build_noise_artifacts(config, dataset_keys=("ck",), verbose=False) == []


def test_path_artefak_mengikuti_pola():
    config = {"paths": {"data_dir": "data"}}
    p = noise_artifact_path(config, "ck", 0.05)
    assert p.name == "ck_noise_v0.05.npz"
    assert str(p.parent).replace("\\", "/").endswith("data")
