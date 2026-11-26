"""
Evaluasi Ketahanan Model terhadap Noise Gaussian (Tabel 14 Jurnal) - Divisi A3.

Mengukur akurasi JOLS-SVM (Jaya + LS-SVM) ketika fitur input diberi
noise Gaussian dengan variansi berbeda: 0.01, 0.02, 0.05, 0.10.

Pipeline evaluasi:
    1. Muat fitur dari A2 (data/a2-features/*.npz)
    2. Proyeksikan dengan LDA → Standarisasi
    3. 5-fold Stratified CV: latih pada data bersih, uji pada data ber-noise
    4. Ukur akurasi per level noise → simpan ke CSV & JSON

CATATAN: Script ini sudah siap dijalankan begitu output A2 (ck_features.npz /
jaffe_features.npz) tersedia. Tidak perlu modifikasi tambahan.

Referensi:
    Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021).
    A hybrid feature descriptor with Jaya optimised least squares SVM
    for facial expression recognition.
    IET Image Processing, 15(7), 1471–1483. DOI: 10.1049/ipr2.12118
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Setup path agar bisa import modul A3 ───────────────────────────────────
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from baseline import load_config
from lssvm import OneVsOneLSSVMClassifier
from optimizers import JayaOptimizer

# ── Konfigurasi Eksperimen ─────────────────────────────────────────────────

# Level noise sesuai Tabel 14 jurnal (variansi Gaussian)
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10]

# Hyperparameter Jaya default (dari hasil Tabel 3)
JAYA_POP_SIZE = 10
JAYA_MAX_ITER = 30
C_RANGE = (-5, 5)   # log2 space
SIGMA_RANGE = (-5, 5)

N_FOLDS = 5
RANDOM_STATE = 42


# ── Fungsi Helper ──────────────────────────────────────────────────────────

def add_gaussian_noise(X: np.ndarray, variance: float, rng: np.random.Generator) -> np.ndarray:
    """Tambahkan noise Gaussian N(0, variance) ke array fitur."""
    if variance == 0.0:
        return X.copy()
    noise = rng.normal(loc=0.0, scale=np.sqrt(variance), size=X.shape)
    return X + noise


def evaluate_noise_robustness(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    noise_levels: list[float] = NOISE_LEVELS,
    n_folds: int = N_FOLDS,
    noise_features: dict[float, np.ndarray] | None = None,
) -> dict:
    """
    Jalankan evaluasi ketahanan noise pada satu dataset.

    Parameters:
        X : np.ndarray
            Fitur bersih 680-D dari data/a2-features/*.npz.
        y : np.ndarray
            Label kelas.
        dataset_name : str
            Nama dataset ('CK+' atau 'JAFFE').
        noise_levels : list of float
            Tingkat variansi noise.
        n_folds : int
            Jumlah fold CV.
        noise_features : dict of float -> np.ndarray, optional
            Fitur 680-D yang diekstrak langsung dari citra ber-noise oleh A2
            (data/a2-features/*_features_noise_v*.npz). Jika tersedia, fitur
            nyata ini akan diproyeksikan dengan LDA fold latih. Jika tidak,
            fallback ke noise Gaussian sintetis pada ruang LDA.

    Returns:
        dict: { noise_level -> {"accuracy": float, "sensitivity": float, "specificity": float} }
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    num_classes = len(np.unique(y))

    results_per_noise: dict[float, list[float]] = {lvl: [] for lvl in noise_levels}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── Proyeksi LDA (latih pada data bersih) ──
        lda = LinearDiscriminantAnalysis(solver="svd")
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)

        # ── Standarisasi ──
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_lda)

        # ── Optimasi Jaya untuk mencari C* dan sigma* ──
        def objective(params):
            C = float(2.0 ** params[0])
            sigma = float(2.0 ** params[1])
            inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + fold_idx)
            inner_accs = []
            for tr, va in inner_skf.split(X_train_scaled, y_train):
                clf_tmp = OneVsOneLSSVMClassifier(gamma=C, sigma=sigma, kernel="rbf")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf_tmp.fit(X_train_scaled[tr], y_train[tr])
                inner_accs.append(np.mean(clf_tmp.predict(X_train_scaled[va]) == y_train[va]))
            return float(np.mean(inner_accs))

        bounds = [C_RANGE, SIGMA_RANGE]
        jaya = JayaOptimizer(
            bounds=bounds,
            pop_size=JAYA_POP_SIZE,
            max_iter=JAYA_MAX_ITER,
            random_state=RANDOM_STATE + fold_idx,
            maximize=True,
        )
        res = jaya.optimize(objective)
        best_C = float(2.0 ** res.best_params[0])
        best_sigma = float(2.0 ** res.best_params[1])

        # ── Latih model final dengan hyperparameter terbaik ──
        clf = OneVsOneLSSVMClassifier(gamma=best_C, sigma=best_sigma, kernel="rbf")
        clf.fit(X_train_scaled, y_train)

        # ── Evaluasi pada berbagai level noise ──
        for noise_lvl in noise_levels:
            if noise_lvl == 0.0:
                X_test_eval = X_test_lda
            elif noise_features is not None and noise_lvl in noise_features:
                # Menggunakan fitur 680-D riil hasil ekstraksi citra ber-noise oleh A2
                X_noisy_raw_test = noise_features[noise_lvl][test_idx]
                X_test_eval = lda.transform(X_noisy_raw_test)
            else:
                # Fallback: noise Gaussian pada ruang LDA jika artefak fitur A2 belum ada
                X_test_eval = add_gaussian_noise(X_test_lda, noise_lvl, rng)

            X_test_noisy_scaled = scaler.transform(X_test_eval)
            y_pred = clf.predict(X_test_noisy_scaled)
            acc = float(np.mean(y_pred == y_test))
            results_per_noise[noise_lvl].append(acc)

        print(
            f"[{dataset_name}] Fold {fold_idx + 1}/{n_folds} selesai "
            f"| C={best_C:.2f}, σ={best_sigma:.2f} "
            f"| Acc@0.00: {results_per_noise[0.0][-1] * 100:.2f}% "
            f"| Acc@0.10: {results_per_noise[0.10][-1] * 100:.2f}%"
        )

    # Agregasi hasil
    summary = {}
    for lvl, accs in results_per_noise.items():
        summary[lvl] = {
            "accuracy_mean": float(np.mean(accs) * 100),
            "accuracy_std": float(np.std(accs) * 100),
            "fold_accuracies": [round(a * 100, 4) for a in accs],
        }

    return summary


def print_table14(results: dict):
    """Cetak Tabel 14 dalam format tabel ASCII."""
    print("\n" + "=" * 65)
    print("  TABEL 14 JURNAL: EVALUASI AKURASI PADA DATA BER-NOISE")
    print("=" * 65)

    datasets = list(results.keys())
    header = f"{'Noise Variance':<18}"
    for ds in datasets:
        header += f"  {ds:>18}"
    print(header)
    print("-" * 65)

    all_levels = NOISE_LEVELS
    for lvl in all_levels:
        row = f"{lvl:<18.2f}"
        for ds in datasets:
            mean = results[ds][lvl]["accuracy_mean"]
            std = results[ds][lvl]["accuracy_std"]
            row += f"  {mean:>13.2f}% ±{std:>4.2f}"
        print(row)

    print("=" * 65)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    config = load_config(REPO_ROOT / "config.yaml")

    ck_feat_path = REPO_ROOT / config["paths"]["ck_features"]
    jaffe_feat_path = REPO_ROOT / config["paths"]["jaffe_features"]
    reports_dir = REPO_ROOT / config["paths"]["a3_reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset_name, feat_path in [("CK+", ck_feat_path), ("JAFFE", jaffe_feat_path)]:
        print(f"\n{'=' * 55}")
        print(f"  EVALUASI NOISE: {dataset_name}")
        print(f"{'=' * 55}")

        if not feat_path.exists():
            print(f"[SKIP] File fitur tidak ditemukan: {feat_path}")
            print(f"       Pastikan A2 sudah mengekspor fitur ke path tersebut.")
            continue

        data = np.load(feat_path)
        X = data["F"].astype(np.float64)
        y = data["y"]
        print(f"Dataset dimuat: {X.shape[0]} sampel, {X.shape[1]} fitur, {len(np.unique(y))} kelas")

        # Muat fitur citra ber-noise nyata dari A2 jika tersedia
        key_prefix = "ck" if "ck" in dataset_name.lower() else "jaffe"
        noise_features = {}
        features_dir = REPO_ROOT / config["paths"].get("a2_features_dir", "data/a2-features")
        for lvl in NOISE_LEVELS:
            if lvl == 0.0:
                continue
            noise_feat_path = features_dir / f"{key_prefix}_features_noise_v{lvl:.2f}.npz"
            if noise_feat_path.exists():
                with np.load(noise_feat_path) as ndata:
                    noise_features[lvl] = ndata["F"].astype(np.float64)
                print(f"  [A2 Fitur] Memuat {noise_feat_path.name} ({noise_features[lvl].shape})")

        t_start = time.perf_counter()
        summary = evaluate_noise_robustness(
            X, y, dataset_name, noise_features=noise_features if noise_features else None
        )
        elapsed = time.perf_counter() - t_start

        all_results[dataset_name] = summary
        print(f"\n[WAKTU] Evaluasi {dataset_name} selesai dalam {elapsed:.1f} detik")

    if not all_results:
        print("\n[ERROR] Tidak ada dataset yang berhasil dimuat. Periksa path fitur A2.")
        return

    # ── Print Tabel 14 ──
    print_table14(all_results)

    # ── Simpan ke CSV ──
    rows = []
    for ds_name, ds_res in all_results.items():
        for lvl, stats in ds_res.items():
            rows.append({
                "Dataset": ds_name,
                "Noise Variance": lvl,
                "Accuracy Mean (%)": round(stats["accuracy_mean"], 4),
                "Accuracy Std (%)": round(stats["accuracy_std"], 4),
            })

    df = pd.DataFrame(rows)
    csv_path = reports_dir / "table14_noise_robustness.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV] Tabel 14 disimpan ke: {csv_path}")

    # ── Simpan ke JSON ──
    json_path = reports_dir / "table14_noise_robustness.json"
    # Konversi key float ke string untuk JSON
    json_safe = {
        ds: {str(lvl): stats for lvl, stats in ds_res.items()}
        for ds, ds_res in all_results.items()
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe, f, indent=2, ensure_ascii=False)
    print(f"[JSON] Detail hasil disimpan ke: {json_path}")


if __name__ == "__main__":
    main()
