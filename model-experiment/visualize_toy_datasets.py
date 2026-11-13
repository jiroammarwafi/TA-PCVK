"""
Visualisasi Evaluasi LS-SVM Biner vs scikit-learn SVC pada Dataset Mainan (XOR & Iris).
Menghasilkan visualisasi batas keputusan (decision boundary) di reports/figures/lssvm_toy_xor_iris.png.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC

from lssvm import LSSVMClassifier


def plot_decision_boundary(clf, X, y, title, ax):
    """Membuat kontur batas keputusan model pada sumbu ax."""
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(grid)
    else:
        Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    # Kontur skor keputusan
    ax.contourf(xx, yy, Z, levels=20, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.contour(
        xx,
        yy,
        Z,
        levels=[0.0],
        linewidths=2,
        colors="black",
        linestyles="--",
    )

    # Plot titik sampel
    classes = np.unique(y)
    colors = ["blue", "red"]
    markers = ["o", "s"]
    for idx, c in enumerate(classes):
        mask = y == c
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=colors[idx],
            marker=markers[idx],
            label=f"Kelas {c}",
            edgecolors="k",
            s=40,
            alpha=0.85,
        )

    acc = clf.score(X, y) * 100.0
    ax.set_title(f"{title}\nAkurasi: {acc:.1f}%", fontsize=11, fontweight="bold")
    ax.set_xlabel("Fitur 1")
    ax.set_ylabel("Fitur 2")
    ax.legend(loc="lower right", fontsize=8)


def main():
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- EXPERIMENT 1: Problem Non-Linear XOR ---
    rng = np.random.RandomState(42)
    X_xor = rng.uniform(-1.0, 1.0, size=(200, 2))
    y_xor = np.where(X_xor[:, 0] * X_xor[:, 1] > 0, 1, -1)
    X_xor += rng.normal(0, 0.05, size=X_xor.shape)

    # LS-SVM pada XOR
    lssvm_xor = LSSVMClassifier(gamma=50.0, sigma=0.4, kernel="rbf")
    lssvm_xor.fit(X_xor, y_xor)
    plot_decision_boundary(
        lssvm_xor, X_xor, y_xor, "LS-SVM RBF (XOR)", axes[0, 0]
    )

    # scikit-learn SVC pada XOR
    gamma_svc = 1.0 / (2.0 * (0.4**2))
    svc_xor = SVC(kernel="rbf", C=50.0, gamma=gamma_svc)
    svc_xor.fit(X_xor, y_xor)
    plot_decision_boundary(
        svc_xor, X_xor, y_xor, "scikit-learn SVC RBF (XOR)", axes[0, 1]
    )

    # --- EXPERIMENT 2: Dataset Iris (Biner: Setosa vs Versicolor) ---
    iris = load_iris()
    mask = iris.target < 2
    # Gunakan 2 fitur untuk visualisasi 2D (Sepal Length vs Petal Length)
    X_iris = iris.data[mask][:, [0, 2]]
    y_iris = iris.target[mask]

    # LS-SVM pada Iris
    lssvm_iris = LSSVMClassifier(gamma=10.0, sigma=1.0, kernel="rbf")
    lssvm_iris.fit(X_iris, y_iris)
    plot_decision_boundary(
        lssvm_iris, X_iris, y_iris, "LS-SVM RBF (Iris Biner)", axes[1, 0]
    )

    # scikit-learn SVC pada Iris
    gamma_svc_iris = 1.0 / (2.0 * (1.0**2))
    svc_iris = SVC(kernel="rbf", C=10.0, gamma=gamma_svc_iris)
    svc_iris.fit(X_iris, y_iris)
    plot_decision_boundary(
        svc_iris, X_iris, y_iris, "scikit-learn SVC RBF (Iris Biner)", axes[1, 1]
    )

    fig.suptitle(
        "Verifikasi Prototipe LS-SVM Biner vs scikit-learn SVC (Minggu 12)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()

    out_file = figures_dir / "lssvm_toy_xor_iris.png"
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualisasi batas keputusan disimpan di: {out_file}")


if __name__ == "__main__":
    main()
