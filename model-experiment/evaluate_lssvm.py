"""
Evaluasi Minggu 13: LS-SVM Multiclass (One-vs-One) + StandardScaler + Grid Search (γ, σ).
Pipeline: LDA → StandardScaler → OvO LS-SVM, dievaluasi 5-fold Stratified CV pada CK+ dan JAFFE.

Catatan: StandardScaler diperlukan setelah LDA karena output LDA memiliki skala besar
(range -27 hingga +38 pada CK+), yang menyebabkan kernel RBF tidak efektif tanpa normalisasi.

Referensi:
Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021).
A hybrid feature descriptor with Jaya optimised least squares SVM for facial expression recognition.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from baseline import load_config, compute_class_metrics
from lssvm import OneVsOneLSSVMClassifier


def grid_search_lssvm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    gamma_values: List[float],
    sigma_values: List[float],
    n_splits: int = 3,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Mencari pasangan hyperparameter (gamma*, sigma*) terbaik menggunakan
    internal Stratified K-Fold pada fold training (sudah di-scale).

    Returns
    -------
    best_gamma, best_sigma, best_val_acc
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_acc = -1.0
    best_gamma = gamma_values[0]
    best_sigma = sigma_values[0]

    for gamma in gamma_values:
        for sigma in sigma_values:
            fold_accs = []
            for tr_idx, val_idx in skf.split(X_train, y_train):
                # StandardScaler hanya di-fit pada inner train split
                sc_inner = StandardScaler()
                X_inner_tr = sc_inner.fit_transform(X_train[tr_idx])
                X_inner_val = sc_inner.transform(X_train[val_idx])

                clf = OneVsOneLSSVMClassifier(gamma=gamma, sigma=sigma, kernel="rbf")
                clf.fit(X_inner_tr, y_train[tr_idx])
                acc = clf.score(X_inner_val, y_train[val_idx])
                fold_accs.append(acc)

            mean_acc = float(np.mean(fold_accs))
            # Tie-breaking: pilih akurasi tertinggi; jika seri, pilih sigma lebih besar (generalisasi lebih halus)
            if (mean_acc > best_acc) or (abs(mean_acc - best_acc) < 1e-6 and sigma > best_sigma):
                best_acc = mean_acc
                best_gamma = gamma
                best_sigma = sigma

    return best_gamma, best_sigma, best_acc


def run_lssvm_multiclass_evaluation(
    features_npz_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Menjalankan 5-fold Stratified CV untuk pipeline LDA + Multiclass LS-SVM (OvO)
    dengan Grid Search pada setiap fold.
    """
    path = Path(features_npz_path)
    if not path.exists():
        if (Path("..") / features_npz_path).exists():
            path = Path("..") / features_npz_path
        else:
            raise FileNotFoundError(f"File {features_npz_path} tidak ditemukan.")

    data = np.load(path)
    X = data["F"]
    y = data["y"]
    class_names = data["class_names"].tolist()
    num_classes = len(class_names)

    cv_cfg = config.get("cross_validation", {})
    n_splits = cv_cfg.get("n_splits", 5)
    shuffle = cv_cfg.get("shuffle", True)
    random_state = cv_cfg.get("random_state", 42)

    lda_cfg = config.get("lda", {})
    lda_solver = lda_cfg.get("solver", "svd")
    lda_components = lda_cfg.get("n_components", None)

    gs_cfg = config.get("grid_search", {})
    gamma_vals = gs_cfg.get("gamma_values", [0.1, 1.0, 10.0, 50.0, 100.0])
    sigma_vals = gs_cfg.get("sigma_values", [0.2, 0.5, 1.0, 2.0, 5.0])
    internal_splits = gs_cfg.get("internal_cv_splits", 3)

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )

    fold_accuracies: List[float] = []
    best_params_per_fold: List[Dict[str, float]] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr_raw, X_val_raw = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # 1. Reduksi dimensi LDA (fit HANYA pada fold training)
        lda = LinearDiscriminantAnalysis(
            solver=lda_solver, n_components=lda_components
        )
        X_tr = lda.fit_transform(X_tr_raw, y_tr)
        X_val = lda.transform(X_val_raw)

        # 2. Grid Search hyperparameter (gamma, sigma) pada fold training
        b_gamma, b_sigma, b_val_acc = grid_search_lssvm(
            X_tr,
            y_tr,
            gamma_values=gamma_vals,
            sigma_values=sigma_vals,
            n_splits=internal_splits,
            random_state=random_state,
        )
        best_params_per_fold.append(
            {"gamma": b_gamma, "sigma": b_sigma, "inner_cv_acc": b_val_acc}
        )

        # 3. Scaling pada fold train dan transformasi ke fold val (mencegah data leakage)
        sc_outer = StandardScaler()
        X_tr_s = sc_outer.fit_transform(X_tr)
        X_val_s = sc_outer.transform(X_val)

        # Latih model final pada seluruh data train fold dengan parameter terbaik
        clf = OneVsOneLSSVMClassifier(gamma=b_gamma, sigma=b_sigma, kernel="rbf")
        clf.fit(X_tr_s, y_tr)

        # 4. Evaluasi pada test/validation fold
        y_pred = clf.predict(X_val_s)
        acc = float(accuracy_score(y_val, y_pred))
        fold_accuracies.append(acc)

        y_true_all.extend(y_val.tolist())
        y_pred_all.extend(y_pred.tolist())

        print(
            f"  Fold {fold_idx+1}/{n_splits} -> Best params (γ={b_gamma}, σ={b_sigma}) -> Acc: {acc*100:.2f}%"
        )

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)

    metrics = compute_class_metrics(y_true_arr, y_pred_arr, num_classes)

    return {
        "dataset_path": str(features_npz_path),
        "num_samples": int(len(y)),
        "num_features_raw": int(X.shape[1]),
        "num_features_lda": int(X_tr.shape[1]),
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
        "best_params_per_fold": best_params_per_fold,
    }


def plot_confusion_matrix(cm, class_names, title, output_path):
    """Menyimpan gambar visualisasi confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = np.array(cm).max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i][j]
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontweight="bold",
            )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix disimpan di: {output_path}")


def main():
    config = load_config()
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # 1. Dataset CK+
    ck_path = Path(config["paths"]["ck_features"])
    if ck_path.exists():
        print(f"\n[INFO] Menjalankan 5-fold CV LS-SVM OvO Grid Search pada CK+...")
        res_ck = run_lssvm_multiclass_evaluation(str(ck_path), config=config)
        with open(
            reports_dir / "lssvm_multiclass_ck_results.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(res_ck, f, indent=2)

        plot_confusion_matrix(
            res_ck["confusion_matrix"],
            res_ck["class_names"],
            f"LS-SVM OvO on CK+ (Acc: {res_ck['mean_accuracy']*100:.2f}%)",
            figures_dir / "lssvm_confusion_matrix_ck.png",
        )

        summary_rows.append(
            {
                "Dataset": "CK+",
                "Samples": res_ck["num_samples"],
                "Model": "LDA + OvO LS-SVM (Grid Search)",
                "Mean Accuracy (%)": round(res_ck["mean_accuracy"] * 100, 2),
                "Std Accuracy (%)": round(res_ck["std_accuracy"] * 100, 2),
                "Macro Sensitivity (%)": round(res_ck["macro_sensitivity"] * 100, 2),
                "Macro Specificity (%)": round(res_ck["macro_specificity"] * 100, 2),
            }
        )

    # 2. Dataset JAFFE
    jaffe_path = Path(config["paths"]["jaffe_features"])
    if jaffe_path.exists():
        print(f"\n[INFO] Menjalankan 5-fold CV LS-SVM OvO Grid Search pada JAFFE...")
        res_jaffe = run_lssvm_multiclass_evaluation(str(jaffe_path), config=config)
        with open(
            reports_dir / "lssvm_multiclass_jaffe_results.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(res_jaffe, f, indent=2)

        plot_confusion_matrix(
            res_jaffe["confusion_matrix"],
            res_jaffe["class_names"],
            f"LS-SVM OvO on JAFFE (Acc: {res_jaffe['mean_accuracy']*100:.2f}%)",
            figures_dir / "lssvm_confusion_matrix_jaffe.png",
        )

        summary_rows.append(
            {
                "Dataset": "JAFFE",
                "Samples": res_jaffe["num_samples"],
                "Model": "LDA + OvO LS-SVM (Grid Search)",
                "Mean Accuracy (%)": round(res_jaffe["mean_accuracy"] * 100, 2),
                "Std Accuracy (%)": round(res_jaffe["std_accuracy"] * 100, 2),
                "Macro Sensitivity (%)": round(res_jaffe["macro_sensitivity"] * 100, 2),
                "Macro Specificity (%)": round(res_jaffe["macro_specificity"] * 100, 2),
            }
        )

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = reports_dir / "lssvm_multiclass_summary.csv"
        df.to_csv(csv_path, index=False)
        print("\n=== BASELINE DESKRIPTIF v0: LS-SVM MULTICLASS (Minggu 13) ===")
        print(df.to_string(index=False))
        print(f"\nTabel hasil awal disimpan di: {csv_path}")


if __name__ == "__main__":
    main()
