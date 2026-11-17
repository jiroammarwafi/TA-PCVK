"""
Evaluasi Subject-Independent (Divisi A1) - TA-PCVK.

Membandingkan dua protokol pembagian data pada dataset yang memiliki beberapa
citra per subjek (terutama CK+ yang berisi 3 frame berurutan per sekuens):

1. ``random``  : StratifiedKFold biasa (protokol yang dipakai A3 saat ini).
                 Frame dari subjek yang sama dapat muncul di train DAN test.
2. ``subject`` : StratifiedGroupKFold dengan ``groups=subjects``. Seluruh citra
                 milik satu subjek hanya berada di salah satu sisi (train atau
                 test), sehingga tidak ada kebocoran identitas.

Tujuan: mengukur seberapa besar protokol acak menggelembungkan akurasi ketika
dataset memuat banyak frame per orang. Pipeline yang dibandingkan identik:
LDA (fit di fold train) -> StandardScaler -> OvO LS-SVM + grid search internal.

Referensi:
Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021).
A hybrid feature descriptor with Jaya optimised least squares SVM for facial
expression recognition. IET Image Processing, 15(7), 1471-1483.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Modul A3 (baseline, lssvm) berada di research/a3-model-optimization/.
# Tambahkan ke sys.path agar modul ini tetap bisa dijalankan langsung
# (python research/a1-data-preprocessing/xxx.py), bukan hanya lewat pytest.
_A3_DIR = Path(__file__).resolve().parents[1] / "a3-model-optimization"
if str(_A3_DIR) not in sys.path:
    sys.path.insert(0, str(_A3_DIR))

from baseline import compute_class_metrics, load_config, resolve_path  # noqa: E402
from lssvm import OneVsOneLSSVMClassifier  # noqa: E402

SPLIT_RANDOM = "random"
SPLIT_SUBJECT = "subject"
VALID_SPLITS = (SPLIT_RANDOM, SPLIT_SUBJECT)


# --------------------------------------------------------------------------- #
# pemuatan data
# --------------------------------------------------------------------------- #
def _resolve_features_path(
    features_npz_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Cari berkas fitur, dengan urutan percobaan:

    1. path apa adanya (bila sudah absolut / cocok dengan CWD)
    2. relatif terhadap root repo (lokasi config.yaml) — jalur utama
    3. relatif terhadap parent CWD (perilaku lama, untuk kompatibilitas)

    Langkah 2 penting karena notebook berada di ``research/pipeline/`` sehingga
    path seperti ``data/a2-features/ck_features.npz`` tidak akan ditemukan bila
    diresolusi terhadap CWD.
    """
    path = Path(features_npz_path)

    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"File fitur {features_npz_path} tidak ditemukan.")

    if path.exists():
        return path

    if config is not None:
        resolved = resolve_path(config, path)
        if resolved.exists():
            return resolved

    parent = Path("..") / path
    if parent.exists():
        return parent

    raise FileNotFoundError(
        f"File fitur {features_npz_path} tidak ditemukan. "
        "Pastikan artefak A2 sudah dibuat (jalankan notebook 02) dan "
        "config.yaml berada di root repo."
    )


def load_features(
    features_npz_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Muat matriks fitur + label + subjek dari artefak A2."""
    data = np.load(_resolve_features_path(features_npz_path, config=config), allow_pickle=True)
    out = {
        "F": data["F"],
        "y": data["y"],
        "class_names": data["class_names"].tolist(),
    }
    out["subjects"] = (
        np.asarray(data["subjects"]).astype(str)
        if "subjects" in data.files
        else np.array([f"idx{i}" for i in range(len(data["y"]))])
    )
    return out


# --------------------------------------------------------------------------- #
# fold
# --------------------------------------------------------------------------- #
def make_folds(
    y: np.ndarray,
    subjects: np.ndarray,
    n_splits: int = 5,
    split: str = SPLIT_RANDOM,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Bangun daftar (train_idx, test_idx) sesuai protokol yang diminta.

    Untuk ``split='subject'``, jumlah fold dibatasi oleh jumlah subjek dan
    StratifiedGroupKFold menjaga proporsi kelas sedapat mungkin.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split harus salah satu dari {VALID_SPLITS}, diterima {split!r}")

    y = np.asarray(y)
    subjects = np.asarray(subjects).astype(str)
    if len(subjects) != len(y):
        raise ValueError(
            f"Panjang subjects ({len(subjects)}) != panjang label ({len(y)})"
        )

    if split == SPLIT_SUBJECT:
        n_groups = len(np.unique(subjects))
        k = max(2, min(int(n_splits), n_groups))
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=int(random_state))
        return list(cv.split(np.zeros(len(y)), y, groups=subjects))

    cv = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
    return list(cv.split(np.zeros(len(y)), y))


def assert_no_subject_leakage(
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    subjects: np.ndarray,
) -> None:
    """Pastikan tidak ada subjek yang muncul di train dan test sekaligus."""
    subjects = np.asarray(subjects).astype(str)
    for i, (tr, te) in enumerate(folds):
        overlap = set(subjects[tr]) & set(subjects[te])
        if overlap:
            raise AssertionError(
                f"Kebocoran subjek pada fold {i + 1}: {sorted(overlap)[:5]}"
            )


# --------------------------------------------------------------------------- #
# grid search internal
# --------------------------------------------------------------------------- #
def grid_search_lssvm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    gamma_values: Sequence[float],
    sigma_values: Sequence[float],
    n_splits: int = 3,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Cari (gamma, sigma) terbaik memakai CV internal pada fold latih.

    LDA diasumsikan sudah diterapkan sebelum fungsi ini dipanggil. StandardScaler
    di-fit ulang pada tiap split internal agar tidak bocor ke data validasi.
    """
    inner = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
    splits = list(inner.split(X_train, y_train))

    best = (-1.0, float(gamma_values[0]), float(sigma_values[0]))
    for gamma in gamma_values:
        for sigma in sigma_values:
            accs = []
            for tr, val in splits:
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_train[tr])
                Xval = scaler.transform(X_train[val])

                clf = OneVsOneLSSVMClassifier(gamma=float(gamma), sigma=float(sigma), kernel="rbf")
                clf.fit(Xtr, y_train[tr])
                accs.append(float(accuracy_score(y_train[val], clf.predict(Xval))))

            mean_acc = float(np.mean(accs))
            # tie-break: sigma lebih besar dianggap lebih halus (generalisasi)
            if mean_acc > best[0] + 1e-9 or (
                abs(mean_acc - best[0]) <= 1e-9 and float(sigma) > best[2]
            ):
                best = (mean_acc, float(gamma), float(sigma))

    return best[1], best[2], best[0]


# --------------------------------------------------------------------------- #
# evaluasi
# --------------------------------------------------------------------------- #
def run_cv(
    features: Dict[str, Any],
    config: Dict[str, Any],
    split: str = SPLIT_RANDOM,
    gamma_values: Optional[Sequence[float]] = None,
    sigma_values: Optional[Sequence[float]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Jalankan 5-fold CV untuk satu protokol split dan kembalikan metriknya."""
    F = features["F"]
    y = np.asarray(features["y"])
    subjects = features["subjects"]
    class_names = features["class_names"]
    num_classes = len(class_names)

    cv_cfg = config.get("cross_validation", {})
    n_splits = int(cv_cfg.get("n_splits", 5))
    random_state = int(cv_cfg.get("random_state", 42))

    lda_cfg = config.get("lda", {})
    gs_cfg = config.get("grid_search", {})
    gamma_values = list(gamma_values or gs_cfg.get("gamma_values", [1.0, 10.0, 50.0]))
    sigma_values = list(sigma_values or gs_cfg.get("sigma_values", [1.0, 2.0, 5.0]))
    internal_splits = int(gs_cfg.get("internal_cv_splits", 3))

    folds = make_folds(y, subjects, n_splits, split, random_state)
    if split == SPLIT_SUBJECT:
        assert_no_subject_leakage(folds, subjects)

    fold_accuracies: List[float] = []
    fold_params: List[Dict[str, Any]] = []
    fold_groups: List[Dict[str, int]] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for i, (tr, te) in enumerate(folds, start=1):
        lda = LinearDiscriminantAnalysis(
            solver=lda_cfg.get("solver", "svd"),
            n_components=lda_cfg.get("n_components", None),
        )
        Xtr_lda = lda.fit_transform(F[tr], y[tr])
        Xte_lda = lda.transform(F[te])

        best_gamma, best_sigma, best_inner = grid_search_lssvm(
            Xtr_lda, y[tr], gamma_values, sigma_values,
            n_splits=internal_splits, random_state=random_state,
        )

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr_lda)
        Xte = scaler.transform(Xte_lda)

        clf = OneVsOneLSSVMClassifier(gamma=best_gamma, sigma=best_sigma, kernel="rbf")
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)

        acc = float(accuracy_score(y[te], pred))
        fold_accuracies.append(acc)
        fold_params.append(
            {"gamma": best_gamma, "sigma": best_sigma, "inner_cv_acc": float(best_inner)}
        )
        fold_groups.append(
            {
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "n_subjects_train": int(len(np.unique(subjects[tr]))),
                "n_subjects_test": int(len(np.unique(subjects[te]))),
            }
        )
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(pred.tolist())

        if verbose:
            print(
                f"    fold {i}/{len(folds)}: train={len(tr)} test={len(te)} "
                f"(subjek {fold_groups[-1]['n_subjects_train']}/"
                f"{fold_groups[-1]['n_subjects_test']}) "
                f"gamma={best_gamma:g} sigma={best_sigma:g} acc={acc * 100:.2f}%"
            )

    metrics = compute_class_metrics(
        np.asarray(y_true_all), np.asarray(y_pred_all), num_classes
    )
    return {
        "split": split,
        "n_splits": len(folds),
        "n_samples": int(len(y)),
        "n_subjects": int(len(np.unique(subjects))),
        "n_classes": num_classes,
        "class_names": class_names,
        "fold_accuracies": fold_accuracies,
        "fold_params": fold_params,
        "fold_sizes": fold_groups,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "macro_sensitivity": metrics["macro_sensitivity"],
        "macro_specificity": metrics["macro_specificity"],
        "per_class_sensitivity": metrics["per_class_sensitivity"],
        "per_class_specificity": metrics["per_class_specificity"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def compare_splits(
    features_npz_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Jalankan kedua protokol pada satu dataset dan kembalikan perbandingannya."""
    config = config or load_config()
    features = load_features(features_npz_path, config=config)

    results: Dict[str, Any] = {}
    for split in VALID_SPLITS:
        if verbose:
            print(f"\n  [{split}] {Path(str(features_npz_path)).name}")
        results[split] = run_cv(features, config, split=split, verbose=verbose)

    random_acc = results[SPLIT_RANDOM]["mean_accuracy"]
    subject_acc = results[SPLIT_SUBJECT]["mean_accuracy"]
    results["gap"] = {
        "absolute": float(random_acc - subject_acc),
        "relative_percent": float((random_acc - subject_acc) / max(subject_acc, 1e-12) * 100.0),
    }
    return results


def write_summary_csv(
    reports_dir: Path,
    prefix: str = "subject_independent",
) -> Optional[Path]:
    """
    Bangun ulang CSV ringkasan dari SELURUH berkas JSON yang ada.

    Dibuat terpisah agar menjalankan satu dataset saja (mis. ``--datasets jaffe``)
    tidak menghapus baris dataset lain yang sudah pernah dievaluasi.

    Memakai modul ``csv`` bawaan Python, bukan pandas: pada mesin dengan
    kebijakan Application Control (WDAC/Device Guard) DLL ``pandas._libs._ujson``
    dapat diblokir sehingga ``import pandas`` gagal total.
    """
    import csv
    import json

    rows = []
    for path in sorted(Path(reports_dir).glob(f"{prefix}_*.json")):
        dataset_key = path.stem[len(prefix) + 1 :]
        with open(path, "r", encoding="utf-8") as fh:
            result = json.load(fh)
        for split in VALID_SPLITS:
            r = result.get(split)
            if not r:
                continue
            rows.append(
                {
                    "Dataset": dataset_key.upper(),
                    "Split": split,
                    "Samples": r["n_samples"],
                    "Subjects": r["n_subjects"],
                    "Mean Accuracy (%)": round(r["mean_accuracy"] * 100, 2),
                    "Std Accuracy (%)": round(r["std_accuracy"] * 100, 2),
                    "Macro Sensitivity (%)": round(r["macro_sensitivity"] * 100, 2),
                    "Macro Specificity (%)": round(r["macro_specificity"] * 100, 2),
                }
            )

    if not rows:
        return None

    csv_path = Path(reports_dir) / f"{prefix}_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def main() -> None:
    """CLI: bandingkan split acak vs subject-independent pada CK+ dan JAFFE."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Bandingkan split acak vs subject-independent (deteksi kebocoran)."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["ck", "jaffe"],
        choices=["ck", "jaffe"],
    )
    parser.add_argument("--out-prefix", default="subject_independent")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="hanya bangun ulang CSV dari JSON yang sudah ada (tanpa evaluasi)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    reports_dir = resolve_path(
        config, config["paths"].get("a1_reports_dir", config["paths"]["reports_dir"])
    )
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        csv_path = write_summary_csv(reports_dir, args.out_prefix)
        if csv_path:
            print(f"[simpan] {csv_path}")
        else:
            print("[info] belum ada JSON hasil untuk diringkas")
        return

    for key in args.datasets:
        features_path = config["paths"][f"{key}_features"]
        if not resolve_path(config, features_path).exists():
            print(f"[lewati] {key}: {features_path} tidak ditemukan")
            continue

        print(f"\n=== {key.upper()} ===")
        result = compare_splits(features_path, config=config, verbose=True)

        out_path = reports_dir / f"{args.out_prefix}_{key}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"  [simpan] {out_path}")
        print(
            f"  selisih akurasi (random - subject): "
            f"{result['gap']['absolute'] * 100:.2f} poin persen"
        )

    csv_path = write_summary_csv(reports_dir, args.out_prefix)
    if csv_path:
        print("\n=== RINGKASAN SPLIT ===")
        print(Path(csv_path).read_text(encoding="utf-8"))
        print(f"[simpan] {csv_path}")


if __name__ == "__main__":
    main()
