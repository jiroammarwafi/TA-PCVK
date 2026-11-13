"""
Unit test suite untuk baseline pipeline LDA + SVM-RBF.
Memverifikasi:
1. Fungsi perhitungan metrik (Accuracy, Macro Sensitivity, Specificity, Confusion Matrix).
2. Eksekusi evaluasi 5-fold CV berjalan lancar pada dataset fitur CK+ dan JAFFE.
"""

from pathlib import Path
import numpy as np
import pytest
from baseline import compute_class_metrics, run_baseline_evaluation, load_config


def test_compute_class_metrics_perfect():
    """Memverifikasi metrik jika semua prediksi benar (100% akurasi)."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    metrics = compute_class_metrics(y_true, y_pred, num_classes=3)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_sensitivity"] == 1.0
    assert metrics["macro_specificity"] == 1.0
    assert metrics["per_class_sensitivity"] == [1.0, 1.0, 1.0]
    assert metrics["per_class_specificity"] == [1.0, 1.0, 1.0]


def test_compute_class_metrics_known_cm():
    """Memverifikasi perhitungan Sensitivity & Specificity pada confusion matrix biner yang diketahui."""
    # Kelas 0: TP=2, FN=1 -> Sen = 2/3
    # Kelas 1: TP=1, FP=1 -> FN=0? y_true=[0,0,0,1], y_pred=[0,0,1,1]
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    # Kelas 0: TP=2, FN=1, FP=0, TN=1 -> Sen=2/3, Spe=1/1=1.0
    # Kelas 1: TP=1, FN=0, FP=1, TN=2 -> Sen=1/1=1.0, Spe=2/3
    metrics = compute_class_metrics(y_true, y_pred, num_classes=2)
    assert metrics["accuracy"] == 0.75
    np.testing.assert_almost_equal(metrics["per_class_sensitivity"][0], 2.0 / 3.0)
    np.testing.assert_almost_equal(metrics["per_class_specificity"][0], 1.0)
    np.testing.assert_almost_equal(metrics["per_class_sensitivity"][1], 1.0)
    np.testing.assert_almost_equal(metrics["per_class_specificity"][1], 2.0 / 3.0)


def test_baseline_evaluation_jaffe():
    """
    Memverifikasi pipeline evaluasi baseline 5-fold CV berjalan dengan benar
    pada artefak fitur JAFFE (data/jaffe_features.npz).
    """
    jaffe_path = Path("data/jaffe_features.npz")
    if not jaffe_path.exists():
        pytest.skip("data/jaffe_features.npz belum tersedia")

    config = load_config()
    res = run_baseline_evaluation(jaffe_path, config=config)

    assert res["num_samples"] == 213
    assert res["num_features"] == 680
    assert len(res["fold_accuracies"]) == 5
    assert 0.0 <= res["mean_accuracy"] <= 1.0
    assert 0.0 <= res["macro_sensitivity"] <= 1.0
    assert 0.0 <= res["macro_specificity"] <= 1.0
    assert np.array(res["confusion_matrix"]).shape == (7, 7)
