"""
Eksperimen Komprehensif Replikasi Jurnal (Divisi A3) - TA-PCVK.
Mengimplementasikan & mereplikasi:
1. Optimasi Jaya vs PSO vs GA (Tabel 3 & Tabel 10 paper)
2. Kurva konvergensi optimizer (Figure 5 paper)
3. Komparasi pengklasifikasi: JOLS-SVM, Lin LS-SVM, Poly LS-SVM, RF, BPNN (Tabel 11 paper)
4. Pengukuran waktu latih dan waktu uji (Tabel 10 paper)

Referensi:
Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021).
A hybrid feature descriptor with Jaya optimised least squares SVM for facial expression recognition.
IET Image Processing, 15(7), 1471-1483.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Tuple
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from baseline import load_config, compute_class_metrics
from lssvm import OneVsOneLSSVMClassifier
from optimizers import JayaOptimizer, PSOOptimizer, GAOptimizer


def evaluate_classifier_fold(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Melatih classifier dan mengukur waktu latih, waktu uji, akurasi, dan prediksi.
    """
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = clf.predict(X_val)
    test_time = (time.perf_counter() - t1) / len(y_val)  # waktu per sampel

    acc = float(accuracy_score(y_val, y_pred))
    return train_time, test_time, acc, y_pred


def run_metaheuristic_cv(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    n_splits: int = 5,
    max_iter: int = 60,
    pop_size: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Menjalankan 5-fold CV bersarang untuk Jaya, PSO, dan GA.
    Mencatat metrik klasifikasi, waktu latih/uji, dan kurva konvergensi rata-rata.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]  # Rentang log2 [2^-5, 2^5] sesuai Tabel 12 jurnal

    num_classes = len(np.unique(y))

    algorithms = ["Jaya", "PSO", "GA"]
    results = {
        alg: {
            "fold_accs": [],
            "train_times": [],
            "test_times": [],
            "y_trues": [],
            "y_preds": [],
            "history_curves": [],
        }
        for alg in algorithms
    }

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
        X_val_raw, y_val = X[val_idx], y[val_idx]

        # 1. Reduksi dimensi LDA (fit HANYA pada train fold)
        lda = LinearDiscriminantAnalysis(solver="svd")
        X_tr_lda = lda.fit_transform(X_tr_raw, y_tr)
        X_val_lda = lda.transform(X_val_raw)

        # 2. Standarisasi (fit HANYA pada train fold)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_lda)
        X_val = scaler.transform(X_val_lda)

        # 3. Inner CV untuk optimasi hyperparameter (C, sigma)
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state + fold_idx)

        def make_fitness_func(X_data, y_data):
            def fitness(params):
                C_val = float(2.0 ** params[0])
                sigma_val = float(2.0 ** params[1])
                cv_accs = []
                for in_tr, in_val in inner_skf.split(X_data, y_data):
                    clf = OneVsOneLSSVMClassifier(gamma=C_val, sigma=sigma_val, kernel="rbf")
                    clf.fit(X_data[in_tr], y_data[in_tr])
                    cv_accs.append(clf.score(X_data[in_val], y_data[in_val]))
                return float(np.mean(cv_accs))
            return fitness

        obj_func = make_fitness_func(X_tr, y_tr)

        # Jalankan Jaya, PSO, GA pada budget yang sama
        for alg_name in algorithms:
            if alg_name == "Jaya":
                opt = JayaOptimizer(bounds=bounds, pop_size=pop_size, max_iter=max_iter, random_state=random_state + fold_idx)
            elif alg_name == "PSO":
                opt = PSOOptimizer(bounds=bounds, pop_size=pop_size, max_iter=max_iter, c1=0.8, c2=0.2, w=0.8, random_state=random_state + fold_idx)
            elif alg_name == "GA":
                opt = GAOptimizer(bounds=bounds, pop_size=pop_size, max_iter=max_iter, crossover_rate=0.8, mutation_rate=0.2, random_state=random_state + fold_idx)

            res = opt.optimize(obj_func)
            results[alg_name]["history_curves"].append(res.history_fitness)

            # Latih model final pada train fold dengan parameter optimal
            best_C = float(2.0 ** res.best_params[0])
            best_sigma = float(2.0 ** res.best_params[1])

            clf = OneVsOneLSSVMClassifier(gamma=best_C, sigma=best_sigma, kernel="rbf")
            t_train, t_test, acc, y_pred = evaluate_classifier_fold(clf, X_tr, y_tr, X_val, y_val)

            results[alg_name]["fold_accs"].append(acc)
            results[alg_name]["train_times"].append(t_train)
            results[alg_name]["test_times"].append(t_test)
            results[alg_name]["y_trues"].extend(y_val.tolist())
            results[alg_name]["y_preds"].extend(y_pred.tolist())

            print(f"[{dataset_name}] Fold {fold_idx+1}/5 | {alg_name} (C={best_C:.2f}, σ={best_sigma:.2f}) -> Acc: {acc*100:.2f}% (train: {t_train:.2f}s)")

    # Hitung metrik agregat
    summary = {}
    for alg in algorithms:
        y_true_arr = np.array(results[alg]["y_trues"])
        y_pred_arr = np.array(results[alg]["y_preds"])
        class_metrics = compute_class_metrics(y_true_arr, y_pred_arr, num_classes)

        # Rata-rata kurva konvergensi sepanjang 5 fold
        mean_curve = np.mean(np.array(results[alg]["history_curves"]), axis=0).tolist()

        summary[alg] = {
            "mean_acc": float(np.mean(results[alg]["fold_accs"])),
            "std_acc": float(np.std(results[alg]["fold_accs"])),
            "macro_sensitivity": class_metrics["macro_sensitivity"],
            "macro_specificity": class_metrics["macro_specificity"],
            "confusion_matrix": class_metrics["confusion_matrix"],
            "avg_train_time": float(np.mean(results[alg]["train_times"])),
            "avg_test_time": float(np.mean(results[alg]["test_times"])),
            "convergence_curve": mean_curve,
        }

    return summary


def run_competitor_classifiers_cv(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Mengevaluasi pengklasifikasi pembanding (Tabel 11 jurnal):
    - Lin LS-SVM
    - Poly LS-SVM (Order 2)
    - Random Forest (100 trees, seed=1)
    - BPNN (MLPClassifier, hidden neurons 7 untuk CK+, 6 untuk JAFFE)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    num_classes = len(np.unique(y))
    hidden_dim = 7 if "ck" in dataset_name.lower() else 6

    models = {
        "Lin LS-SVM": lambda: OneVsOneLSSVMClassifier(gamma=10.0, kernel="linear"),
        "Poly LS-SVM": lambda: OneVsOneLSSVMClassifier(gamma=10.0, kernel="poly", degree=2, coef0=1.0),
        "RF": lambda: RandomForestClassifier(n_estimators=100, random_state=1),
        "BPNN": lambda: MLPClassifier(
            hidden_layer_sizes=(hidden_dim,),
            learning_rate_init=0.05,
            momentum=0.4,
            solver="sgd",
            max_iter=300,
            random_state=42,
        ),
    }

    results = {m: {"fold_accs": [], "train_times": [], "test_times": [], "y_trues": [], "y_preds": []} for m in models}

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
        X_val_raw, y_val = X[val_idx], y[val_idx]

        lda = LinearDiscriminantAnalysis(solver="svd")
        X_tr_lda = lda.fit_transform(X_tr_raw, y_tr)
        X_val_lda = lda.transform(X_val_raw)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_lda)
        X_val = scaler.transform(X_val_lda)

        for m_name, m_ctor in models.items():
            clf = m_ctor()
            t_tr, t_te, acc, y_pred = evaluate_classifier_fold(clf, X_tr, y_tr, X_val, y_val)

            results[m_name]["fold_accs"].append(acc)
            results[m_name]["train_times"].append(t_tr)
            results[m_name]["test_times"].append(t_te)
            results[m_name]["y_trues"].extend(y_val.tolist())
            results[m_name]["y_preds"].extend(y_pred.tolist())

    summary = {}
    for m_name in models:
        y_true_arr = np.array(results[m_name]["y_trues"])
        y_pred_arr = np.array(results[m_name]["y_preds"])
        class_metrics = compute_class_metrics(y_true_arr, y_pred_arr, num_classes)
        summary[m_name] = {
            "mean_acc": float(np.mean(results[m_name]["fold_accs"])),
            "std_acc": float(np.std(results[m_name]["fold_accs"])),
            "macro_sensitivity": class_metrics["macro_sensitivity"],
            "macro_specificity": class_metrics["macro_specificity"],
            "avg_train_time": float(np.mean(results[m_name]["train_times"])),
            "avg_test_time": float(np.mean(results[m_name]["test_times"])),
        }
    return summary


def plot_convergence_curves(
    opt_results: Dict[str, Any],
    dataset_name: str,
    output_path: Path,
):
    """
    Memplot kurva konvergensi (Figure 5 paper: fitness vs iterations untuk Jaya, PSO, GA).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Jaya": "dodgerblue", "PSO": "darkorange", "GA": "forestgreen"}
    markers = {"Jaya": "D", "PSO": "s", "GA": "^"}

    for alg, data in opt_results.items():
        curve = data["convergence_curve"]
        iters = np.arange(len(curve))
        # Plot garis dan mark setiap 5 iterasi
        mark_indices = np.arange(0, len(curve), max(1, len(curve) // 10))
        ax.plot(
            iters,
            curve,
            label=f"{alg} (Final Acc: {curve[-1]*100:.2f}%)",
            color=colors.get(alg, "blue"),
            linewidth=2.2,
            marker=markers.get(alg, "o"),
            markevery=list(mark_indices),
            markersize=6,
        )

    ax.set_title(f"Convergence Curves on {dataset_name} (Figure 5 Replikasi)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of iterations", fontsize=11)
    ax.set_ylabel("Classification accuracy (fitness)", fontsize=11)
    ax.set_ylim(0.85, 1.02)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Kurva konvergensi disimpan di: {output_path}")


def plot_confusion_matrix(cm, class_names, title, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Confusion matrix disimpan di: {output_path}")


def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parents[2]

    reports_dir = repo_root / config["paths"]["a3_reports_dir"]
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ck_feat_path = repo_root / config["paths"]["ck_features"]
    jaffe_feat_path = repo_root / config["paths"]["jaffe_features"]

    all_experiments = {}

    for d_name, feat_path in [("CK+", ck_feat_path), ("JAFFE", jaffe_feat_path)]:
        print(f"\n=======================================================")
        print(f"  MENJALANKAN EKSPERIMEN JURNAL PADA DATASET: {d_name}")
        print(f"=======================================================")
        data = np.load(feat_path)
        X = data["F"].astype(np.float64)
        y = data["y"]
        class_names = [str(c) for c in data["class_names"]]

        # 1. Evaluasi Jaya vs PSO vs GA
        print(f"\n[1/3] Menjalankan Evaluasi Metaheuristik: Jaya vs PSO vs GA...")
        opt_summary = run_metaheuristic_cv(
            X, y, dataset_name=d_name, n_splits=5, max_iter=50, pop_size=10, random_state=42
        )

        # 2. Plot Kurva Konvergensi (Figure 5)
        curve_path = figures_dir / f"convergence_{d_name.lower().replace('+', '')}.png"
        plot_convergence_curves(opt_summary, dataset_name=d_name, output_path=curve_path)

        # 3. Plot Confusion Matrix JOLS-SVM
        cm_path = figures_dir / f"jols_confusion_matrix_{d_name.lower().replace('+', '')}.png"
        plot_confusion_matrix(
            opt_summary["Jaya"]["confusion_matrix"],
            class_names,
            f"JOLS-SVM on {d_name} (Acc: {opt_summary['Jaya']['mean_acc']*100:.2f}%)",
            cm_path,
        )

        # 4. Evaluasi Pengklasifikasi Pembanding (Lin LS-SVM, Poly LS-SVM, RF, BPNN)
        print(f"\n[2/3] Menjalankan Pengklasifikasi Pembanding (Lin/Poly LS-SVM, RF, BPNN)...")
        comp_summary = run_competitor_classifiers_cv(X, y, dataset_name=d_name, n_splits=5, random_state=42)

        all_experiments[d_name] = {
            "optimizers": opt_summary,
            "competitors": comp_summary,
            "class_names": class_names,
        }

    # =========================================================
    # REKAPITULASI TABEL SESUAI STRUKTUR JURNAL
    # =========================================================

    # 1. Tabel 3: Performance evaluation of JO, PSO, and GA with LS-SVM
    table3_rows = []
    for alg in ["Jaya", "PSO", "GA"]:
        alg_label = "JOLS-SVM" if alg == "Jaya" else f"{alg}LS-SVM"
        ck_s = all_experiments["CK+"]["optimizers"][alg]
        jf_s = all_experiments["JAFFE"]["optimizers"][alg]
        table3_rows.append({
            "Method": alg_label,
            "CK+ Sen (%)": round(ck_s["macro_sensitivity"] * 100, 2),
            "CK+ Spe (%)": round(ck_s["macro_specificity"] * 100, 2),
            "CK+ Acc (%)": round(ck_s["mean_acc"] * 100, 2),
            "JAFFE Sen (%)": round(jf_s["macro_sensitivity"] * 100, 2),
            "JAFFE Spe (%)": round(jf_s["macro_specificity"] * 100, 2),
            "JAFFE Acc (%)": round(jf_s["mean_acc"] * 100, 2),
        })

    df_table3 = pd.DataFrame(table3_rows)
    df_table3.to_csv(reports_dir / "table3_optimizer_comparison.csv", index=False)
    print("\n\n=== TABEL 3 PAPER: EVALUASI PERFORMA JAYA, PSO, DAN GA DENGAN LS-SVM ===")
    print(df_table3.to_string(index=False))

    # 2. Tabel 10: Comparison of execution time (in seconds)
    table10_rows = []
    for alg in ["Jaya", "PSO", "GA"]:
        alg_label = "JOLS-SVM" if alg == "Jaya" else f"{alg}LS-SVM"
        ck_s = all_experiments["CK+"]["optimizers"][alg]
        jf_s = all_experiments["JAFFE"]["optimizers"][alg]
        table10_rows.append({
            "Method": alg_label,
            "CK+ Training time (s)": round(ck_s["avg_train_time"], 4),
            "CK+ Testing time (s)": round(ck_s["avg_test_time"], 6),
            "CK+ Acc (%)": round(ck_s["mean_acc"] * 100, 2),
            "JAFFE Training time (s)": round(jf_s["avg_train_time"], 4),
            "JAFFE Testing time (s)": round(jf_s["avg_test_time"], 6),
            "JAFFE Acc (%)": round(jf_s["mean_acc"] * 100, 2),
        })

    df_table10 = pd.DataFrame(table10_rows)
    df_table10.to_csv(reports_dir / "table10_execution_time.csv", index=False)
    print("\n\n=== TABEL 10 PAPER: PERBANDINGAN WAKTU EKSEKUSI (DETIK) ===")
    print(df_table10.to_string(index=False))

    # 3. Tabel 11: Performance evaluation of different classifiers using proposed hybrid features
    table11_rows = []
    classifier_list = [
        ("JOLS-SVM", ("optimizers", "Jaya")),
        ("Lin LS-SVM", ("competitors", "Lin LS-SVM")),
        ("Poly LS-SVM", ("competitors", "Poly LS-SVM")),
        ("RF", ("competitors", "RF")),
        ("BPNN", ("competitors", "BPNN")),
    ]

    for label, (group, key) in classifier_list:
        ck_c = all_experiments["CK+"][group][key]
        jf_c = all_experiments["JAFFE"][group][key]
        table11_rows.append({
            "Method": label,
            "CK+ Sen (%)": round(ck_c["macro_sensitivity"] * 100, 2),
            "CK+ Spe (%)": round(ck_c["macro_specificity"] * 100, 2),
            "CK+ Acc (%)": round(ck_c["mean_acc"] * 100, 2),
            "JAFFE Sen (%)": round(jf_c["macro_sensitivity"] * 100, 2),
            "JAFFE Spe (%)": round(jf_c["macro_specificity"] * 100, 2),
            "JAFFE Acc (%)": round(jf_c["mean_acc"] * 100, 2),
        })

    df_table11 = pd.DataFrame(table11_rows)
    df_table11.to_csv(reports_dir / "table11_classifier_comparison.csv", index=False)
    print("\n\n=== TABEL 11 PAPER: PERBANDINGAN BERBAGAI CLASSIFIER ===")
    print(df_table11.to_string(index=False))

    # Simpan seluruh detail ke JSON
    with open(reports_dir / "detailed_journal_results.json", "w", encoding="utf-8") as f:
        json.dump(all_experiments, f, indent=2)

    print(f"\n[SELESAI] Seluruh hasil replikasi tabel & gambar jurnal tersimpan di: {reports_dir}/")


if __name__ == "__main__":
    main()
