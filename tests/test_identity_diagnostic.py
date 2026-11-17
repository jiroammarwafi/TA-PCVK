"""
Unit test suite untuk diagnosa kebocoran identitas subjek (Divisi A1).

Memverifikasi:
1. Filter subjek minimum citra bekerja dan tidak membuang subjek yang valid.
2. Encoding label subjek deterministik dan konsisten (urut abjad).
3. Matematika peluang (chance) dan normalisasi di atas peluang benar.
4. Diagnosa lengkap menghasilkan struktur hasil yang dibutuhkan laporan.
5. Fitur yang didominasi identitas memang terdeteksi sebagai dominan.
6. CSV ringkasan terbangun dari seluruh JSON yang ada.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "a1-data-preprocessing"))
sys.path.insert(0, str(ROOT / "research" / "a3-model-optimization"))

from diagnose_identity import (  # noqa: E402
    chance_level,
    describe,
    encode_subject_labels,
    excess_over_chance,
    run_diagnostics,
    select_subjects_with_min_images,
    write_summary_csv,
)


# --------------------------------------------------------------------------- #
# helper
# --------------------------------------------------------------------------- #
def _synthetic(
    n_subjects: int = 6,
    per_subject: int = 6,
    n_classes: int = 3,
    dim: int = 12,
    seed: int = 0,
    identity_strength: float = 5.0,
    expression_strength: float = 0.5,
):
    """
    Data sintetis: vektor fitur = pusat subjek (identitas) + pusat kelas
    (ekspresi) + noise. Rasio kedua kekuatan mengatur mana yang dominan.
    """
    rng = np.random.default_rng(seed)
    subject_centers = rng.normal(0.0, identity_strength, size=(n_subjects, dim))
    class_centers = rng.normal(0.0, expression_strength, size=(n_classes, dim))

    rows, labels, subjects = [], [], []
    for s in range(n_subjects):
        for j in range(per_subject):
            c = (s + j) % n_classes
            rows.append(
                subject_centers[s] + class_centers[c] + rng.normal(0.0, 0.1, dim)
            )
            labels.append(c)
            subjects.append(f"S{s:03d}")

    return {
        "F": np.asarray(rows, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int8),
        "subjects": np.asarray(subjects),
        "class_names": [f"c{i}" for i in range(n_classes)],
    }


def _fast_config(n_splits: int = 3, min_images: int = 5) -> dict:
    """Config minimal dengan parameter kecil agar test cepat."""
    return {
        "cross_validation": {"n_splits": n_splits, "random_state": 42},
        "lda": {"solver": "svd", "n_components": None},
        "diagnostics": {
            "min_images_per_subject": min_images,
            "subject_cv_splits": 3,
            "classifiers": {
                "svc": {"C": 10.0, "gamma": "scale"},
                "logreg": {"max_iter": 1000, "C": 1.0},
                "random_forest": {"n_estimators": 10, "n_jobs": 1, "random_state": 0},
            },
        },
    }


# --------------------------------------------------------------------------- #
# helper murni
# --------------------------------------------------------------------------- #
def test_select_subjects_with_min_images_filters_rare_subjects():
    subjects = ["A"] * 5 + ["B"] * 2 + ["C"] * 6
    mask = select_subjects_with_min_images(subjects, min_images=5)

    assert mask.dtype == bool
    assert mask.sum() == 11  # A (5) + C (6), B (2) dibuang
    assert not mask[5] and not mask[6]  # dua citra milik B


def test_select_subjects_keeps_all_when_threshold_met():
    subjects = ["A"] * 3 + ["B"] * 3
    mask = select_subjects_with_min_images(subjects, min_images=3)
    assert mask.all()


def test_select_subjects_handles_empty_input():
    mask = select_subjects_with_min_images([], min_images=5)
    assert mask.shape == (0,)


def test_encode_subject_labels_is_sorted_and_deterministic():
    subjects = ["S002", "S001", "S002", "S003", "S001"]
    codes = encode_subject_labels(subjects)

    # urut abjad: S001 -> 0, S002 -> 1, S003 -> 2
    assert codes.tolist() == [1, 0, 1, 2, 0]
    assert np.array_equal(codes, encode_subject_labels(subjects))


def test_chance_level_matches_number_of_outcomes():
    assert chance_level(7) == pytest.approx(1 / 7)
    assert chance_level(1) == pytest.approx(1.0)
    assert chance_level(0) == pytest.approx(1.0)  # dijaga agar tidak bagi nol


def test_excess_over_chance_normalises_between_zero_and_one():
    assert excess_over_chance(0.1429, 1 / 7) == pytest.approx(0.0, abs=1e-3)
    assert excess_over_chance(1.0, 1 / 7) == pytest.approx(1.0)
    assert excess_over_chance(0.5, 1 / 7) == pytest.approx((0.5 - 1 / 7) / (1 - 1 / 7))


# --------------------------------------------------------------------------- #
# diagnosa lengkap
# --------------------------------------------------------------------------- #
def test_run_diagnostics_returns_expected_structure():
    features = _synthetic()
    result = run_diagnostics(features, _fast_config(), verbose=False)

    for key in ("subject_identification", "raw_features", "lda_nested", "summary"):
        assert key in result

    assert result["n_samples"] == 36
    assert result["n_subjects"] == 6
    assert result["n_classes"] == 3
    assert result["chance_expression"] == pytest.approx(1 / 3)

    # klasifier pembanding pada fitur mentah; Random Forest dapat dilewati
    # bila DLL sklearn diblokir kebijakan Application Control.
    assert {"svc_rbf", "logreg"} <= set(result["raw_features"])
    assert set(result["raw_features"]) <= {"svc_rbf", "logreg", "random_forest"}
    for res in result["raw_features"].values():
        assert 0.0 <= res["accuracy"] <= 1.0
        assert len(res["fold_accuracies"]) == 3

    assert result["lda_nested"]["n_components"] == 2  # n_classes - 1
    assert len(result["lda_nested"]["fold_accuracies"]) == 3


def test_run_diagnostics_detects_identity_dominant_features():
    """Fitur yang didominasi identitas harus terdeteksi sebagai dominan."""
    features = _synthetic(identity_strength=8.0, expression_strength=0.2)
    result = run_diagnostics(features, _fast_config(), verbose=False)

    # 6 subjek -> peluang 1/6; fitur sekuat ini harus jauh di atas peluang
    assert result["subject_identification"]["accuracy"] > 0.8
    assert result["subject_identification"]["chance"] == pytest.approx(1 / 6)
    assert result["summary"]["identity_dominates_expression"] is True


def test_run_diagnostics_detects_expression_when_identity_is_flat():
    """Kontrol negatif: identitas lemah, ekspresi kuat -> identitas tidak dominan."""
    features = _synthetic(identity_strength=0.2, expression_strength=6.0)
    result = run_diagnostics(features, _fast_config(), verbose=False)

    assert result["summary"]["identity_dominates_expression"] is False


def test_run_diagnostics_rejects_dataset_without_enough_subjects():
    features = _synthetic(n_subjects=1, per_subject=10)
    with pytest.raises(ValueError, match="Tidak cukup subjek"):
        run_diagnostics(features, _fast_config(), verbose=False)


def test_describe_produces_readable_lines():
    features = _synthetic()
    result = run_diagnostics(features, _fast_config(), verbose=False)

    lines = describe(result)
    assert len(lines) >= 3
    assert any("identitas subjek" in line for line in lines)
    assert any("LDA nested" in line for line in lines)


# --------------------------------------------------------------------------- #
# CSV ringkasan
# --------------------------------------------------------------------------- #
def test_write_summary_csv_builds_rows_from_json(tmp_path):
    features = _synthetic()
    result = run_diagnostics(features, _fast_config(), verbose=False)

    for key in ("ck", "jaffe"):
        (tmp_path / f"identity_{key}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    csv_path = write_summary_csv(tmp_path, prefix="identity")

    assert csv_path is not None and csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "Subject Identification (%)" in text
    assert "LDA Penalty (pts)" in text
    assert text.count("\n") == 3  # header + 2 dataset


def test_write_summary_csv_returns_none_without_json(tmp_path):
    assert write_summary_csv(tmp_path, prefix="identity") is None
