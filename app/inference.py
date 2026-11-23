"""
Engine Inferensi Mandiri (Standalone) Berbasis NumPy Murni (Divisi A3/A4) - TA-PCVK.

Modul ini membaca artefak parameter model terlatih (.npz) yang diekspor oleh A3,
dan menjalankan inferensi ekspresi wajah secara cepat (< 1 ms per citra)
tanpa memerlukan dependensi scikit-learn saat runtime aplikasi web.

Referensi:
Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021).
A hybrid feature descriptor with Jaya optimised least squares SVM for facial expression recognition.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class FERInferenceEngine:
    """
    Engine inferensi untuk klasifikasi ekspresi wajah menggunakan
    pipeline: 680-D Fitur -> LDA -> StandardScaler -> OvO LS-SVM (NumPy murni).

    Parameters
    ----------
    model_path : str atau Path
        Path ke berkas model.npz yang diekspor oleh A3.
    """

    def __init__(self, model_path: Union[str, Path]):
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Berkas model {model_path} tidak ditemukan.")

        self.model_data = np.load(p, allow_pickle=True)

        # 1. Parameter Proyeksi LDA & Scaler
        self.lda_scalings = self.model_data["lda_scalings"]  # (680, n_components)
        self.lda_xbar = self.model_data["lda_xbar"]          # (680,)
        self.scaler_mean = self.model_data["scaler_mean"]    # (n_components,)
        self.scaler_scale = self.model_data["scaler_scale"]  # (n_components,)

        self.class_names = [str(c) for c in self.model_data["class_names"]]
        self.num_classes = len(self.class_names)
        self.gamma = float(self.model_data["gamma"])
        self.sigma = float(self.model_data["sigma"])
        self.kernel = str(self.model_data.get("kernel", "rbf"))

        # 2. Rekonstruksi 21 Classifier Biner OvO
        self.pairs: List[Tuple[int, int]] = []
        self.binary_models: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for c1 in range(self.num_classes):
            for c2 in range(c1 + 1, self.num_classes):
                prefix = f"pair_{c1}_{c2}_"
                if f"{prefix}alpha" in self.model_data:
                    self.pairs.append((c1, c2))
                    self.binary_models[(c1, c2)] = {
                        "alpha": self.model_data[f"{prefix}alpha"],
                        "b": float(self.model_data[f"{prefix}b"]),
                        "X": self.model_data[f"{prefix}X"],
                        "y": self.model_data[f"{prefix}y"],
                    }

    def transform_features(self, X_680: np.ndarray) -> np.ndarray:
        """
        Mereduksi fitur 680-D ke ruang diskriminan LDA dan menormalisasi dengan StandardScaler.
        Formula LDA: (X - lda_xbar) @ lda_scalings
        Formula Scaler: (X_lda - scaler_mean) / scaler_scale
        """
        X = np.asarray(X_680, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        # Proyeksi LDA
        X_centered = X - self.lda_xbar
        X_lda = np.dot(X_centered, self.lda_scalings)

        # Standarisasi
        X_scaled = (X_lda - self.scaler_mean) / self.scaler_scale

        return X_scaled[0] if single else X_scaled

    def _decision_function_binary(
        self, X_scaled: np.ndarray, model: Dict[str, Any]
    ) -> np.ndarray:
        """Menghitung margin keputusan f(x) untuk 1 model biner menggunakan kernel RBF."""
        # Jarak kuadrat Euclidean antara X_scaled (M, d) dan support vectors X_train (N, d)
        X_train = model["X"]
        # sq_dists = ||x - z||^2 = ||x||^2 + ||z||^2 - 2 x @ z^T
        x_sq = np.sum(X_scaled**2, axis=1, keepdims=True)
        z_sq = np.sum(X_train**2, axis=1, keepdims=True).T
        sq_dists = np.maximum(0.0, x_sq + z_sq - 2.0 * np.dot(X_scaled, X_train.T))

        gamma_rbf = 1.0 / (2.0 * (self.sigma**2))
        K = np.exp(-gamma_rbf * sq_dists)

        weights = model["alpha"] * model["y"]
        return np.dot(K, weights) + model["b"]

    def predict(
        self, X_680: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Memprediksi ekspresi untuk satu atau batch sampel fitur 680-D.

        Returns
        -------
        pred_indices : np.ndarray (M,)
            Indeks kelas hasil prediksi.
        pred_labels : np.ndarray (M,)
            Nama string ekspresi hasil prediksi.
        probabilities : np.ndarray (M, num_classes)
            Estimasi probabilitas / skor ternormalisasi per kelas.
        """
        X_scaled = self.transform_features(X_680)
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)

        M = X_scaled.shape[0]
        K = self.num_classes
        votes = np.zeros((M, K), dtype=np.float64)
        scores = np.zeros((M, K), dtype=np.float64)

        for (c1, c2), model in self.binary_models.items():
            df = self._decision_function_binary(X_scaled, model)
            is_c2 = df >= 0.0

            votes[:, c2] += is_c2.astype(np.float64)
            votes[:, c1] += (~is_c2).astype(np.float64)

            scores[:, c2] += df
            scores[:, c1] -= df

        pred_indices = np.empty(M, dtype=int)
        for i in range(M):
            max_v = np.max(votes[i])
            candidates = np.where(votes[i] == max_v)[0]
            if len(candidates) == 1:
                pred_indices[i] = candidates[0]
            else:
                pred_indices[i] = candidates[np.argmax(scores[i, candidates])]

        pred_labels = np.array([self.class_names[idx] for idx in pred_indices])

        # Hitung probabilitas menggunakan softmax pada agregasi skor
        exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)

        return pred_indices, pred_labels, probs

    def predict_single(self, feat_680: np.ndarray) -> Dict[str, Any]:
        """
        Helper inferensi untuk 1 citra, menghasilkan dictionary praktis untuk backend API A4.
        """
        idx, label, probs = self.predict(feat_680)
        idx_val = int(idx[0])
        label_val = str(label[0])
        prob_dist = {
            self.class_names[c]: float(probs[0, c]) for c in range(self.num_classes)
        }

        return {
            "prediction_index": idx_val,
            "prediction_label": label_val,
            "confidence": float(probs[0, idx_val]),
            "probabilities": prob_dist,
        }
