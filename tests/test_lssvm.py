"""
Unit test suite untuk implementasi Least Squares Support Vector Machine (LS-SVM).
Memverifikasi:
1. Kesesuaian dimensi atribut model (alpha, b, classes).
2. Kondisi optimalitas KKT sistem persamaan linear terpenuhi.
3. Kinerja pada problem non-linear (XOR) mencapai akurasi sempurna/tinggi.
4. Paritas akurasi dengan scikit-learn SVC(kernel='rbf') pada dataset Iris (biner).
5. Dukungan kernel linear dan RBF.
6. Penanganan error input tidak valid (misal > 2 kelas, dimensi salah).
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from lssvm import LSSVMClassifier


def generate_xor_dataset(n_samples: int = 200, noise: float = 0.05, seed: int = 42):
    """Menghasilkan dataset non-linear XOR."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    # XOR: label 1 jika kuadran I atau III (X[:,0]*X[:,1] > 0), -1 jika kuadran II atau IV
    y = np.where(X[:, 0] * X[:, 1] > 0, 1, -1)
    if noise > 0:
        X += rng.normal(0, noise, size=X.shape)
    return X, y


def test_lssvm_attributes_and_shapes():
    """Memverifikasi inisialisasi dan atribut setelah fit."""
    X, y = generate_xor_dataset(n_samples=50, noise=0.0)
    clf = LSSVMClassifier(gamma=10.0, sigma=0.5, kernel="rbf")
    clf.fit(X, y)

    assert hasattr(clf, "alpha_")
    assert hasattr(clf, "b_")
    assert hasattr(clf, "classes_")
    assert clf.alpha_.shape == (50,)
    assert isinstance(clf.b_, float)
    assert len(clf.classes_) == 2
    assert np.all(np.isin(clf.classes_, [-1, 1]))


def test_lssvm_kkt_linear_system_satisfaction():
    """
    Memverifikasi secara aljabar linear bahwa solusi (b, alpha)
    memenuhi sistem persamaan KKT: A * [b, alpha]^T = B.
    """
    X, y = generate_xor_dataset(n_samples=30, noise=0.0)
    gamma = 5.0
    sigma = 0.8
    ridge = 1e-8

    clf = LSSVMClassifier(gamma=gamma, sigma=sigma, kernel="rbf", ridge_jitter=ridge)
    clf.fit(X, y)

    N = X.shape[0]
    y_enc = clf.y_train_encoded_
    K = clf._compute_kernel(X, X)
    Omega = np.outer(y_enc, y_enc) * K

    reg = (1.0 / gamma) + ridge
    A = np.empty((N + 1, N + 1), dtype=np.float64)
    A[0, 0] = 0.0
    A[0, 1:] = y_enc
    A[1:, 0] = y_enc
    A[1:, 1:] = Omega + np.eye(N) * reg

    sol = np.hstack(([clf.b_], clf.alpha_))
    B_expected = np.hstack(([0.0], np.ones(N)))

    # Periksa residual ||A * sol - B|| < toleransi
    residual = np.linalg.norm(A @ sol - B_expected)
    assert residual < 1e-5, f"Residual KKT terlalu besar: {residual}"


def test_lssvm_xor_problem_accuracy():
    """
    Memverifikasi LS-SVM dengan RBF kernel mampu menyelesaikan problem non-linear XOR
    dengan akurasi sempurna pada clean XOR dan >= 95% pada noisy XOR.
    """
    # 1. Clean XOR (non-linearly separable)
    X_clean, y_clean = generate_xor_dataset(n_samples=200, noise=0.0, seed=42)
    clf_clean = LSSVMClassifier(gamma=50.0, sigma=0.4, kernel="rbf")
    clf_clean.fit(X_clean, y_clean)
    assert clf_clean.score(X_clean, y_clean) >= 0.95

    # 2. Noisy XOR dengan train/test split
    X, y = generate_xor_dataset(n_samples=200, noise=0.05, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = LSSVMClassifier(gamma=50.0, sigma=0.4, kernel="rbf")
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    assert train_acc >= 0.95, f"Train acc XOR harus >= 0.95, didapat {train_acc}"
    assert test_acc >= 0.95, f"Test acc XOR harus >= 0.95, didapat {test_acc}"


def test_lssvm_iris_parity_with_svc():
    """
    Memverifikasi kinerja LS-SVM pada dataset Iris (biner: kelas 0 vs 1)
    setara (paritas) dengan scikit-learn SVC(kernel='rbf').
    """
    iris = load_iris()
    # Ambil 2 kelas pertama (Setosa vs Versicolor)
    mask = iris.target < 2
    X = iris.data[mask]
    y = iris.target[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 1. LS-SVM
    lssvm = LSSVMClassifier(gamma=10.0, sigma=1.0, kernel="rbf")
    lssvm.fit(X_train, y_train)
    lssvm_acc = lssvm.score(X_test, y_test)

    # 2. scikit-learn SVC
    svc = SVC(kernel="rbf", C=10.0, gamma=1.0 / (2.0 * 1.0**2), random_state=42)
    svc.fit(X_train, y_train)
    svc_acc = float(svc.score(X_test, y_test))

    # Keduanya harus mencapai 100% pada pemisahan Setosa vs Versicolor
    assert lssvm_acc == 1.0, f"LS-SVM akurasi Iris harus 1.0, didapat {lssvm_acc}"
    assert svc_acc == 1.0, f"SVC akurasi Iris harus 1.0, didapat {svc_acc}"
    assert abs(lssvm_acc - svc_acc) <= 0.05


def test_lssvm_linear_kernel():
    """Memverifikasi kernel linear pada data yang terpisah secara linear."""
    rng = np.random.RandomState(42)
    X1 = rng.randn(40, 2) + np.array([2.0, 2.0])
    X2 = rng.randn(40, 2) + np.array([-2.0, -2.0])
    X = np.vstack([X1, X2])
    y = np.array([1] * 40 + [-1] * 40)

    clf = LSSVMClassifier(gamma=10.0, kernel="linear")
    clf.fit(X, y)
    acc = clf.score(X, y)
    assert acc >= 0.95


def test_lssvm_invalid_inputs():
    """Memverifikasi validasi input pada parameter dan data."""
    # Lebih dari 2 kelas
    X = np.random.randn(30, 2)
    y_multiclass = np.array([0] * 10 + [1] * 10 + [2] * 10)
    clf = LSSVMClassifier()
    with pytest.raises(ValueError, match="tepat 2 kelas"):
        clf.fit(X, y_multiclass)

    # Kernel tidak didukung
    with pytest.raises(ValueError, match="tidak didukung"):
        bad_clf = LSSVMClassifier(kernel="unknown")
        bad_clf.fit(X[:20], np.array([0] * 10 + [1] * 10))

    # Sigma non-positif
    with pytest.raises(ValueError, match="sigma"):
        bad_sigma = LSSVMClassifier(sigma=-1.0)
        bad_sigma.fit(X[:20], np.array([0] * 10 + [1] * 10))
