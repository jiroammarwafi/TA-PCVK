"""
Skrip Eksekusi Evaluasi Baseline Model (LDA + SVM-RBF) pada CK+ dan JAFFE.
Menghasilkan:
1. File JSON hasil lengkap di reports/
2. Tabel ringkasan CSV di reports/
3. Visualisasi Confusion Matrix di reports/figures/
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline import run_baseline_evaluation, load_config


def plot_confusion_matrix(cm, class_names, title, output_path):
    """Menyimpan plot confusion matrix ke file gambar."""
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

    # Nilai di dalam setiap sel
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
        print(f"\n[INFO] Menjalankan baseline 5-fold CV pada CK+ ({ck_path})...")
        res_ck = run_baseline_evaluation(ck_path, config=config)
        with open(reports_dir / "baseline_ck_results.json", "w", encoding="utf-8") as f:
            json.dump(res_ck, f, indent=2)

        plot_confusion_matrix(
            res_ck["confusion_matrix"],
            res_ck["class_names"],
            f"Baseline LDA+SVM-RBF on CK+ (Acc: {res_ck['mean_accuracy']*100:.2f}%)",
            figures_dir / "baseline_confusion_matrix_ck.png",
        )

        summary_rows.append(
            {
                "Dataset": "CK+",
                "Samples": res_ck["num_samples"],
                "Features": res_ck["num_features"],
                "Mean Accuracy (%)": round(res_ck["mean_accuracy"] * 100, 2),
                "Std Accuracy (%)": round(res_ck["std_accuracy"] * 100, 2),
                "Macro Sensitivity (%)": round(res_ck["macro_sensitivity"] * 100, 2),
                "Macro Specificity (%)": round(res_ck["macro_specificity"] * 100, 2),
            }
        )
    else:
        print(f"[WARN] File {ck_path} tidak ditemukan.")

    # 2. Dataset JAFFE
    jaffe_path = Path(config["paths"]["jaffe_features"])
    if jaffe_path.exists():
        print(f"\n[INFO] Menjalankan baseline 5-fold CV pada JAFFE ({jaffe_path})...")
        res_jaffe = run_baseline_evaluation(jaffe_path, config=config)
        with open(
            reports_dir / "baseline_jaffe_results.json", "w", encoding="utf-8"
        ) as f:
            json.dump(res_jaffe, f, indent=2)

        plot_confusion_matrix(
            res_jaffe["confusion_matrix"],
            res_jaffe["class_names"],
            f"Baseline LDA+SVM-RBF on JAFFE (Acc: {res_jaffe['mean_accuracy']*100:.2f}%)",
            figures_dir / "baseline_confusion_matrix_jaffe.png",
        )

        summary_rows.append(
            {
                "Dataset": "JAFFE",
                "Samples": res_jaffe["num_samples"],
                "Features": res_jaffe["num_features"],
                "Mean Accuracy (%)": round(res_jaffe["mean_accuracy"] * 100, 2),
                "Std Accuracy (%)": round(res_jaffe["std_accuracy"] * 100, 2),
                "Macro Sensitivity (%)": round(res_jaffe["macro_sensitivity"] * 100, 2),
                "Macro Specificity (%)": round(res_jaffe["macro_specificity"] * 100, 2),
            }
        )
    else:
        print(f"[WARN] File {jaffe_path} tidak ditemukan.")

    # Simpan ringkasan tabel CSV & cetak ke konsol
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        csv_path = reports_dir / "baseline_summary.csv"
        df_summary.to_csv(csv_path, index=False)
        print("\n=== RINGKASAN HASIL BASELINE (Minggu 12) ===")
        print(df_summary.to_string(index=False))
        print(f"\nTabel ringkasan disimpan di: {csv_path}")


if __name__ == "__main__":
    main()
