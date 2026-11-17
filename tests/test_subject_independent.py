"""
Unit test suite untuk evaluasi subject-independent (Divisi A1).

Memverifikasi:
1. Split 'subject' tidak pernah menempatkan subjek yang sama di train dan test.
2. Split 'random' memang membocorkan subjek (pembanding/negatif kontrol).
3. Jumlah fold dan cakupan sampel benar.
4. Detektor kebocoran benar-benar gagal saat ada kebocoran.
5. Evaluasi lengkap menghasilkan struktur hasil yang dibutuhkan laporan.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "a1-data-preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "a3-model-optimization"))

from baseline import load_config  # noqa: E402
from evaluate_subject_independent import (  # noqa: E402
    SPLIT_RANDOM,
    SPLIT_SUBJECT,
    assert_no_subject_leakage,
    load_features,
    make_folds,
    run_cv,
)


def _dataset(n_subjects=10, per_subject=6, n_classes=3, seed=0):
    """Bangun data sintetis: tiap subjek punya beberapa citra, kelas bervariasi."""
    rng = np.random.default_rng(seed)
    subjects, labels = [], []
    for s in range(n_subjects):
        for j in range(per_subject):
            subjects.append(f"S{s:03d}")
            labels.append((s + j) % n_classes)
    y = np.array(labels, dtype=np.int8)
    subjects = np.array(subjects)
    # fitur: centroid kelas + noise per subjek (meniru identitas)
    centers = rng.normal(0, 3, (n_classes, 12))
    F = centers[y] + rng.normal(0, 0.5, (len(y), 12))
    return F.astype(np.float32), y, subjects, [f"c{i}" for i in range(n_classes)]


def test_split_subject_tanpa_kebocoran():
    _, y, subjects, _ = _dataset()
    folds = make_folds(y, subjects, n_splits=5, split=SPLIT_SUBJECT)
    assert len(folds) == 5
    assert_no_subject_leakage(folds, subjects)  # tidak boleh raise


def test_split_subject_menutup_semua_sampel():
    _, y, subjects, _ = _dataset()
    folds = make_folds(y, subjects, n_splits=5, split=SPLIT_SUBJECT)
    test_idx = np.concatenate([te for _, te in folds])
    assert len(test_idx) == len(y)
    assert sorted(test_idx.tolist()) == list(range(len(y)))


def test_split_random_membocorkan_subjek():
    """Kontrol negatif: split acak memang menempatkan subjek yang sama di dua sisi."""
    _, y, subjects, _ = _dataset()
    folds = make_folds(y, subjects, n_splits=5, split=SPLIT_RANDOM)
    ada_kebocoran = any(
        set(subjects[tr]) & set(subjects[te]) for tr, te in folds
    )
    assert ada_kebocoran, "split acak seharusnya membocorkan subjek"


def test_detektor_kebocoran_menangkap_pelanggaran():
    subjects = np.array(["A", "A", "B", "B"])
    folds = [(np.array([0, 2]), np.array([1, 3]))]  # subjek A ada di dua sisi
    with pytest.raises(AssertionError):
        assert_no_subject_leakage(folds, subjects)


def test_jumlah_fold_dibatasi_jumlah_subjek():
    _, y, subjects, _ = _dataset(n_subjects=3, per_subject=4)
    folds = make_folds(y, subjects, n_splits=5, split=SPLIT_SUBJECT)
    assert len(folds) == 3


def test_split_tidak_dikenal_ditolak():
    _, y, subjects, _ = _dataset()
    with pytest.raises(ValueError):
        make_folds(y, subjects, split="campuran")


def test_panjang_subjects_harus_cocok():
    _, y, subjects, _ = _dataset()
    with pytest.raises(ValueError):
        make_folds(y, subjects[:-1], split=SPLIT_SUBJECT)


def test_run_cv_menghasilkan_struktur_lengkap():
    F, y, subjects, class_names = _dataset(n_subjects=9, per_subject=6)
    features = {"F": F, "y": y, "subjects": subjects, "class_names": class_names}
    config = {
        "cross_validation": {"n_splits": 3, "random_state": 42},
        "lda": {"solver": "svd", "n_components": None},
        "grid_search": {
            "gamma_values": [1.0, 10.0],
            "sigma_values": [1.0],
            "internal_cv_splits": 2,
        },
    }

    res = run_cv(features, config, split=SPLIT_SUBJECT, verbose=False)
    assert res["split"] == SPLIT_SUBJECT
    assert res["n_splits"] == 3
    assert len(res["fold_accuracies"]) == 3
    assert len(res["fold_params"]) == 3
    assert 0.0 <= res["mean_accuracy"] <= 1.0
    assert res["macro_sensitivity"] >= 0.0
    assert len(res["confusion_matrix"]) == len(class_names)
    # tiap fold melaporkan jumlah subjek di kedua sisi
    for fs in res["fold_sizes"]:
        assert fs["n_subjects_train"] > 0 and fs["n_subjects_test"] > 0


def test_run_cv_subject_lebih_rendah_daripada_random_pada_data_bocor():
    """Karena fitur memuat identitas subjek, split acak harus terlihat lebih baik."""
    F, y, subjects, class_names = _dataset(n_subjects=10, per_subject=6, seed=5)
    features = {"F": F, "y": y, "subjects": subjects, "class_names": class_names}
    config = {
        "cross_validation": {"n_splits": 5, "random_state": 42},
        "lda": {"solver": "svd", "n_components": None},
        "grid_search": {
            "gamma_values": [1.0, 10.0],
            "sigma_values": [1.0, 5.0],
            "internal_cv_splits": 2,
        },
    }
    random_res = run_cv(features, config, split=SPLIT_RANDOM, verbose=False)
    subject_res = run_cv(features, config, split=SPLIT_SUBJECT, verbose=False)
    assert random_res["mean_accuracy"] >= subject_res["mean_accuracy"]


def test_load_features_menyertakan_subjects(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "x_features.npz"
    np.savez_compressed(
        p,
        F=rng.random((5, 8)).astype(np.float32),
        y=np.array([0, 1, 0, 1, 0], dtype=np.int8),
        subjects=np.array(["A", "A", "B", "B", "C"]),
        class_names=np.array(["c0", "c1"]),
    )
    out = load_features(p)
    assert out["F"].shape == (5, 8)
    assert out["class_names"] == ["c0", "c1"]
    assert out["subjects"].tolist() == ["A", "A", "B", "B", "C"]


def test_load_features_path_relatif_diresolusi_ke_root_config(tmp_path, monkeypatch):
    """
    Regresi: path relatif pada config harus diresolusi terhadap root repo
    (lokasi config.yaml), bukan terhadap CWD.

    Bug sebelumnya: menjalankan notebook dari research/pipeline/ membuat
    'data/a2-features/ck_features.npz' gagal ditemukan karena hanya dicoba
    terhadap CWD dan '..'.
    """
    import yaml

    # Struktur palsu: <root>/config.yaml, <root>/data/a2-features/x_features.npz
    root = tmp_path / "repo"
    feat_dir = root / "data" / "a2-features"
    feat_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    np.savez_compressed(
        feat_dir / "x_features.npz",
        F=rng.random((6, 4)).astype(np.float32),
        y=np.array([0, 1, 0, 1, 0, 1], dtype=np.int8),
        subjects=np.array(["A", "A", "B", "B", "C", "C"]),
        class_names=np.array(["c0", "c1"]),
    )
    cfg_path = root / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "project": {"random_seed": 42},
                "paths": {
                    "data_dir": "data",
                    "ck_features": "data/a2-features/x_features.npz",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(cfg_path)

    # Simulasikan CWD notebook: folder dalam yang bukan root repo.
    nested = root / "research" / "pipeline"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    rel = config["paths"]["ck_features"]
    assert not Path(rel).exists(), "prasyarat: path relatif gagal dari CWD notebook"

    out = load_features(rel, config=config)
    assert out["F"].shape == (6, 4)
    assert out["subjects"].tolist() == ["A", "A", "B", "B", "C", "C"]
