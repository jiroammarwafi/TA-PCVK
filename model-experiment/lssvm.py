"""
Implementasi Least Squares Support Vector Machine (LS-SVM) untuk Klasifikasi.

Referensi:
- Suykens, J. A. K., & Vandewalle, J. (1999).
  Least Squares Support Vector Machine Classifiers. Neural Processing Letters, 9(3), 293-300.
- Kar, N. B., Nayak, D. R., Babu, K. S., & Zhang, Y.-D. (2021).
  A hybrid feature descriptor with Jaya optimised least squares SVM for facial expression recognition.
  IET Image Processing, 15(7), 1471-1483.
"""

from typing import Optional, Union, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve, LinAlgError, lstsq
from sklearn.base import BaseEstimator, ClassifierMixin


class LSSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    Least Squares Support Vector Machine (LS-SVM) Binary Classifier.

    Berbeda dengan standard SVM yang menyelesaikan Quadratic Programming (QP),
    LS-SVM memformulasikan fungsi objektif dengan constraint kesetaraan (equality constraints)
    dan fungsi penalti kuadratis (squared loss), menghasilkan sistem persamaan linear KKT:

        [ 0    y^T               ] [ b ]   [ 0 ]
        [ y    Omega + (1/γ + ε)I ] [ α ] = [ 1 ]

    di mana:
        Omega_ij = y_i * y_j * K(x_i, x_j)
        K(x, z)  = exp(-||x - z||^2 / (2 * sigma^2)) (RBF kernel)
        γ        = parameter regularisasi (gamma)
        σ        = bandwidth kernel RBF (sigma)

    Parameters
    ----------
    gamma : float, default=10.0
        Parameter regularisasi yang mengatur trade-off antara margin dan error.
    sigma : float, default=1.0
        Parameter bandwidth untuk kernel Radial Basis Function (RBF).
    kernel : str, default='rbf'
        Jenis fungsi kernel: 'rbf' atau 'linear'.
    ridge_jitter : float, default=1e-8
        Konstanta numerik kecil yang ditambahkan ke diagonal sistem KKT
        untuk memastikan kestabilan solusi matriks (mencegah singularitas).
    """

    def __init__(
        self,
        gamma: float = 10.0,
        sigma: float = 1.0,
        kernel: str = "rbf",
        ridge_jitter: float = 1e-8,
    ):
        self.gamma = gamma
        self.sigma = sigma
        self.kernel = kernel
        self.ridge_jitter = ridge_jitter

    def _compute_kernel(
        self, X1: np.ndarray, X2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Menghitung matriks kernel antara X1 dan X2."""
        if X2 is None:
            X2 = X1

        if self.kernel == "rbf":
            if self.sigma <= 0:
                raise ValueError(f"sigma kernel RBF harus positif, didapat {self.sigma}")
            sq_dists = cdist(X1, X2, metric="sqeuclidean")
            gamma_rbf = 1.0 / (2.0 * (self.sigma**2))
            return np.exp(-gamma_rbf * sq_dists)
        elif self.kernel == "linear":
            return np.dot(X1, X2.T)
        else:
            raise ValueError(
                f"Kernel '{self.kernel}' tidak didukung. Pilihan: 'rbf', 'linear'."
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSSVMClassifier":
        """
        Melatih model LS-SVM biner dengan menyelesaikan sistem persamaan linear KKT.

        Parameters
        ----------
        X : array-like berdimensi (N, d)
            Fitur data latih.
        y : array-like berdimensi (N,)
            Label kelas (tepat 2 kelas biner).

        Returns
        -------
        self : LSSVMClassifier
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X harus berdimensi 2 (N, d), didapat dimensi {X.ndim}")
        if y.ndim != 1:
            raise ValueError(f"y harus berdimensi 1 (N,), didapat dimensi {y.ndim}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Jumlah baris X ({X.shape[0]}) tidak cocok dengan panjang y ({y.shape[0]})"
            )

        # Validasi kelas biner
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"LSSVMClassifier biner membutuhkan tepat 2 kelas, ditemukan {len(classes)}: {classes}"
            )

        self.classes_ = classes
        # Mapping kelas ke {-1.0, +1.0}
        y_encoded = np.where(y == self.classes_[1], 1.0, -1.0).astype(np.float64)

        if self.gamma <= 0:
            raise ValueError(f"gamma regularisasi harus positif, didapat {self.gamma}")

        N = X.shape[0]
        self.X_train_ = X
        self.y_train_encoded_ = y_encoded

        # 1. Hitung matriks kernel K_train (N x N)
        K = self._compute_kernel(X, X)

        # 2. Bentuk matriks Omega = (y * y^T) * K
        Omega = np.outer(y_encoded, y_encoded) * K

        # 3. Bentuk matriks blok KKT:
        # A = [ 0       y^T               ]
        #     [ y   Omega + (1/γ + jitter)I ]
        reg_diag = (1.0 / self.gamma) + self.ridge_jitter
        A22 = Omega + np.eye(N, dtype=np.float64) * reg_diag

        A = np.empty((N + 1, N + 1), dtype=np.float64)
        A[0, 0] = 0.0
        A[0, 1:] = y_encoded
        A[1:, 0] = y_encoded
        A[1:, 1:] = A22

        # Vektor ruas kanan B = [0, 1, 1, ..., 1]^T
        B = np.empty(N + 1, dtype=np.float64)
        B[0] = 0.0
        B[1:] = 1.0

        # 4. Selesaikan sistem linear A * [b, α]^T = B
        try:
            # A simetris
            sol = solve(A, B, assume_a="sym")
        except LinAlgError:
            # Fallback ke least squares jika singular
            sol, _, _, _ = lstsq(A, B)

        self.b_ = float(sol[0])
        self.alpha_ = sol[1:].astype(np.float64)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Menghitung nilai fungsi keputusan laten f(x):
            f(x) = sum_{i=1}^N α_i y_i K(x_i, x) + b

        Parameters
        ----------
        X : array-like berdimensi (M, d)
            Sampel data uji.

        Returns
        -------
        decision : ndarray berdimensi (M,)
            Skor fungsi keputusan kontinu.
        """
        if not hasattr(self, "alpha_"):
            raise RuntimeError("Model belum dilatih. Panggil fit() terlebih dahulu.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X harus berdimensi 2 (M, d), didapat dimensi {X.ndim}")

        # Matriks kernel antara sampel uji dan sampel latih (M x N)
        K_test = self._compute_kernel(X, self.X_train_)

        # f(X) = K_test @ (alpha * y_train) + b
        weights = self.alpha_ * self.y_train_encoded_
        scores = np.dot(K_test, weights) + self.b_
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Memprediksi kelas untuk sampel X:
            y_pred = classes_[1] jika f(x) >= 0 else classes_[0]

        Parameters
        ----------
        X : array-like berdimensi (M, d)

        Returns
        -------
        y_pred : ndarray berdimensi (M,)
        """
        scores = self.decision_function(X)
        pred_idx = np.where(scores >= 0.0, 1, 0)
        return self.classes_[pred_idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Menghitung akurasi prediksi pada data (X, y)."""
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
