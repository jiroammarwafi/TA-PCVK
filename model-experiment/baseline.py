"""
Eksperimen Baseline Model: LDA + SVM-RBF (scikit-learn).
Dievaluasi dengan 5-fold Stratified Cross-Validation pada CK+ dan JAFFE.

Referensi:
Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021).
A hybrid feature descriptor with Jaya optimised least squares SVM for facial expression recognition.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Union, Optional
import json
import yaml
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score


def load_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Memuat konfigurasi eksperimen dari file YAML."""
    path = Path(config_path)
    if not path.exists():
        # Coba cari di parent directory jika dipanggil dari subdirektori
        if (Path("..") / config_path).exists():
            path = Path("..") / config_path
        else:
            raise FileNotFoundError(f"File konfigurasi {config_path} tidak ditemukan.")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def compute_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, Any]:
    """
    Menghitung Sensitivity (Recall), Specificity, dan Accuracy secara per-kelas dan makro.

    Untuk setiap kelas c:
      TP = y_true == c & y_pred == c
      TN = y_true != c & y_pred != c
      FP = y_true != c & y_pred == c
      FN = y_true == c & y_pred != c

      Sensitivity (Recall) = TP / (TP + FN)
      Specificity         = TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total_samples = len(y_true)
    overall_acc = float(accuracy_score(y_true, y_pred))

    sensitivities = []
    specificities = []

    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(np.sum(cm[c, :]) - tp)
        fp = int(np.sum(cm[:, c]) - tp)
        tn = int(total_samples - (tp + fn + fp))

        sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        sensitivities.append(sen)
        specificities.append(spe)

    macro_sen = float(np.mean(sensitivities))
    macro_spe = float(np.mean(specificities))

    return {
        "accuracy": overall_acc,
        "macro_sensitivity": macro_sen,
        "macro_specificity": macro_spe,
        "per_class_sensitivity": sensitivities,
        "per_class_specificity": specificities,
        "confusion_matrix": cm.tolist(),
    }


def run_baseline_evaluation(
    features_npz_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Menjalankan evaluasi 5-fold Stratified CV untuk pipeline baseline LDA + SVM-RBF.

    Parameters
    ----------
    features_npz_path : str atau Path
        Path ke file .npz yang berisi 'F', 'y', 'class_names'.
    config : dict, optional
        Konfigurasi proyek dari config.yaml.

    Returns
    -------
    results : dict
        Metrik evaluasi baseline lengkap (akurasi per-fold, mean acc, std acc,
        macro sensitivity, macro specificity, confusion matrix agregat).
    """
    if config is None:
        config = load_config()

    features_path = Path(features_npz_path)
    if not features_path.exists():
        if (Path("..") / features_npz_path).exists():
            features_path = Path("..") / features_npz_path
        else:
            raise FileNotFoundError(f"File fitur {features_npz_path} tidak ditemukan.")

    data = np.load(features_path)
    X = data["F"]
    y = data["y"]
    class_names = data["class_names"].tolist()
    num_classes = len(class_names)

    # Parameter dari config
    cv_cfg = config.get("cross_validation", {})
    n_splits = cv_cfg.get("n_splits", 5)
    shuffle = cv_cfg.get("shuffle", True)
    random_state = cv_cfg.get("random_state", 42)

    lda_cfg = config.get("lda", {})
    lda_solver = lda_cfg.get("solver", "svd")
    lda_components = lda_cfg.get("n_components", None)

    svm_cfg = config.get("baseline_svm", {})
    svm_c = svm_cfg.get("C", 10.0)
    svm_gamma = svm_cfg.get("gamma", "scale")
    svm_kernel = svm_cfg.get("kernel", "rbf")

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )

    fold_accuracies: List[float] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Pipeline: LDA -> SVM-RBF (fit hanya pada fold train untuk mencegah kebocoran)
        pipeline = Pipeline(
            [
                (
                    "lda",
                    LinearDiscriminantAnalysis(
                        solver=lda_solver, n_components=lda_components
                    ),
                ),
                (
                    "svm",
                    SVC(
                        kernel=svm_kernel,
                        C=svm_c,
                        gamma=svm_gamma,
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)

        fold_acc = float(accuracy_score(y_val, y_val_pred))
        fold_accuracies.append(fold_acc)

        y_true_all.extend(y_val.tolist())
        y_pred_all.extend(y_val_pred.tolist())

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)

    metrics = compute_class_metrics(y_true_arr, y_pred_arr, num_classes)

    results = {
        "dataset_path": str(features_npz_path),
        "num_samples": int(len(y)),
        "num_features": int(X.shape[1]),
        "class_names": class_names,
        "n_splits": n_splits,
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "macro_sensitivity": metrics["macro_sensitivity"],
        "macro_specificity": metrics["macro_specificity"],
        "per_class_sensitivity": metrics["per_class_sensitivity"],
        "per_class_specificity": metrics["per_class_specificity"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    return results
