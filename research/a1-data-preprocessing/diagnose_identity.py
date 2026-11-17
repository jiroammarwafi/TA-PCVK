"""
Diagnosa Kebocoran Identitas Subjek (Divisi A1) - TA-PCVK.

Latar belakang: pada CK+ akurasi split acak mencapai 96,64%, tetapi jatuh ke
48,21% ketika subjek dipisah (lihat ``evaluate_subject_independent.py``).
Modul ini mengukur *penyebabnya* secara kuantitatif: seberapa besar vektor
fitur hybrid 680-D menyimpan **identitas orang**, bukan hanya ekspresi.

Tiga pengukuran, semuanya memakai protokol subject-independent
(``StratifiedGroupKFold`` dengan ``groups=subjects``) agar tidak ada subjek
yang muncul di train dan test sekaligus:

1. ``subject_identification``
   Klasifier dilatih menebak **siapa orangnya** dari fitur 680-D. Diukur dengan
   split **acak** (``StratifiedKFold``), bukan split per subjek: mekanisme
   kebocoran pada protokol split acak justru adalah frame orang yang sama
   muncul di train *dan* test. Jika akurasi menebak subjek jauh di atas
   peluang (1/n_subjek), fitur memang memuat identitas yang bisa dieksploitasi
   oleh klasifier emosi.

   Catatan metodologis: memakai split *per subjek* untuk tugas ini akan selalu
   menghasilkan ~0% karena subjek di data uji tidak pernah ada di data latih —
   itu mengukur hal yang berbeda (generalisasi ke orang baru), bukan kebocoran.

2. ``raw_features``
   Klasifikasi **emosi** langsung pada fitur mentah (tanpa LDA), memakai
   beberapa klasifier pembanding. Ini batas atas informasi ekspresi yang
   tersedia di fitur.

3. ``lda_nested``
   Klasifikasi emosi setelah LDA yang di-fit **di dalam fold latih** saja.
   Selisih terhadap (2) menunjukkan berapa banyak informasi ekspresi yang
   dibuang oleh LDA pada protokol jujur.

Keluaran: ``reports/a1-subject-independent/identity_{ck,jaffe}.json`` dan
``identity_summary.csv``.

Referensi:
Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021).
A hybrid feature descriptor with Jaya optimised least squares SVM for facial
expression recognition. IET Image Processing, 15(7), 1471-1483.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import argparse
import json
import sys
import time

import numpy as np

# Modul A3 (baseline) dan modul A1 lain berada di direktori berbeda.
# Tambahkan ke sys.path agar modul ini bisa dijalankan langsung
# (python research/a1-data-preprocessing/diagnose_identity.py), bukan hanya
# lewat pytest.
_HERE = Path(__file__).resolve().parent
_A3_DIR = _HERE.parent / "a3-model-optimization"
for _p in (str(_HERE), str(_A3_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baseline import load_config, resolve_path  # noqa: E402
from evaluate_subject_independent import (  # noqa: E402
    SPLIT_RANDOM,
    SPLIT_SUBJECT,
    assert_no_subject_leakage,
    load_features,
    make_folds,
)

DEFAULT_MIN_IMAGES_PER_SUBJECT = 5


# --------------------------------------------------------------------------- #
# helper murni (mudah diuji tanpa komputasi berat)
# --------------------------------------------------------------------------- #
def select_subjects_with_min_images(
    subjects: Sequence[str],
    min_images: int = DEFAULT_MIN_IMAGES_PER_SUBJECT,
) -> np.ndarray:
    """
    Boolean mask: True untuk citra milik subjek yang punya >= ``min_images``.

    Subjek dengan sangat sedikit citra tidak dapat di-CV secara adil saat
    menebak identitas, sehingga dikeluarkan dari pengukuran (bukan dari
    dataset aslinya).
    """
    subjects = np.asarray(subjects).astype(str)
    if len(subjects) == 0:
        return np.zeros(0, dtype=bool)

    uniq, counts = np.unique(subjects, return_counts=True)
    eligible = set(uniq[counts >= int(min_images)].tolist())
    return np.array([s in eligible for s in subjects], dtype=bool)


def encode_subject_labels(subjects: Sequence[str]) -> np.ndarray:
    """Ubah kode subjek menjadi indeks integer yang deterministik (urut abjad)."""
    subjects = np.asarray(subjects).astype(str)
    uniq = sorted(set(subjects.tolist()))
    index = {s: i for i, s in enumerate(uniq)}
    return np.array([index[s] for s in subjects], dtype=np.int64)


def chance_level(n_outcomes: int) -> float:
    """Akurasi tebakan acak untuk ``n_outcomes`` kelas yang seimbang."""
    return float(1.0 / max(1, int(n_outcomes)))


def excess_over_chance(accuracy: float, chance: float) -> float:
    """
    Akurasi yang dinormalisasi ke rentang 0-1.

    0 = setara tebakan acak, 1 = sempurna. Memungkinkan perbandingan adil
    antara tugas dengan jumlah kelas berbeda (7 emosi vs 118 subjek).
    """
    if chance >= 1.0:
        return 0.0
    return float((accuracy - chance) / (1.0 - chance))


# --------------------------------------------------------------------------- #
# evaluasi
# --------------------------------------------------------------------------- #
def _cv_accuracy(
    make_classifier,
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    scale: bool = True,
) -> Dict[str, Any]:
    """
    Jalankan CV manual pada daftar fold yang sudah ditentukan.

    ``StandardScaler`` di-fit ulang di dalam tiap fold agar tidak bocor.
    """
    from sklearn.preprocessing import StandardScaler

    accuracies: List[float] = []
    t0 = time.perf_counter()
    for tr, te in folds:
        if scale:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
        else:
            Xtr, Xte = X[tr], X[te]

        model = make_classifier()
        model.fit(Xtr, y[tr])
        accuracies.append(float((model.predict(Xte) == y[te]).mean()))

    return {
        "accuracy": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "fold_accuracies": [round(a, 4) for a in accuracies],
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


def run_diagnostics(
    features: Dict[str, Any],
    config: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Jalankan ketiga pengukuran untuk satu dataset.

    ``features`` berisi kunci ``F``, ``y``, ``subjects``, ``class_names``
    (format artefak A2).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    F = np.asarray(features["F"], dtype=np.float64)
    y = np.asarray(features["y"])
    subjects = np.asarray(features["subjects"]).astype(str)
    class_names = list(features["class_names"])

    diag_cfg = config.get("diagnostics", {})
    cv_cfg = config.get("cross_validation", {})
    n_splits = int(cv_cfg.get("n_splits", 5))
    random_state = int(cv_cfg.get("random_state", 42))
    min_images = int(diag_cfg.get("min_images_per_subject", DEFAULT_MIN_IMAGES_PER_SUBJECT))
    subject_splits = int(diag_cfg.get("subject_cv_splits", 3))

    clf_cfg = diag_cfg.get("classifiers", {})
    svc_cfg = clf_cfg.get("svc", {})
    logreg_cfg = clf_cfg.get("logreg", {})
    rf_cfg = clf_cfg.get("random_forest", {})

    result: Dict[str, Any] = {
        "n_samples": int(len(y)),
        "n_subjects": int(len(np.unique(subjects))),
        "n_classes": int(len(class_names)),
        "class_names": class_names,
        "chance_expression": chance_level(len(class_names)),
        "min_images_per_subject": min_images,
    }

    if verbose:
        print(
            f"  dataset: {len(y)} citra, {result['n_subjects']} subjek, "
            f"{len(class_names)} kelas"
        )

    # --- (1) menebak SUBJEK dari fitur ------------------------------------- #
    mask = select_subjects_with_min_images(subjects, min_images)
    subj_labels = encode_subject_labels(subjects[mask])
    n_subjects_eval = int(len(np.unique(subj_labels)))

    if n_subjects_eval < 2:
        raise ValueError(
            "Tidak cukup subjek untuk diagnosa identitas "
            f"(hanya {n_subjects_eval} subjek dengan >= {min_images} citra)."
        )

    # StratifiedKFold menuntut setiap kelas punya >= n_splits anggota; batasi
    # agar tetap aman bila pengguna menurunkan min_images_per_subject.
    min_class_count = int(np.bincount(subj_labels).min())
    n_subj_splits = max(2, min(subject_splits, min_class_count, n_subjects_eval))

    subj_folds = make_folds(
        subj_labels,
        subjects[mask],
        n_splits=n_subj_splits,
        split=SPLIT_RANDOM,
        random_state=random_state,
    )

    if verbose:
        print(
            f"  [1/3] menebak subjek dari fitur, split acak "
            f"({n_subjects_eval} subjek, {int(mask.sum())} citra) ..."
        )
    subject_ident = _cv_accuracy(
        lambda: SVC(C=float(svc_cfg.get("C", 10.0)),
                    gamma=svc_cfg.get("gamma", "scale")),
        F[mask],
        subj_labels,
        subj_folds,
    )
    subj_chance = chance_level(n_subjects_eval)
    subject_ident.update(
        {
            "split": SPLIT_RANDOM,
            "n_subjects_evaluated": n_subjects_eval,
            "n_samples_evaluated": int(mask.sum()),
            "chance": subj_chance,
            "excess_over_chance": excess_over_chance(subject_ident["accuracy"], subj_chance),
        }
    )
    result["subject_identification"] = subject_ident
    if verbose:
        print(
            f"        -> {subject_ident['accuracy'] * 100:.2f}% "
            f"(peluang {subj_chance * 100:.2f}%)"
        )

    # --- (2) emosi pada fitur MENTAH (tanpa LDA) --------------------------- #
    emotion_folds = make_folds(
        y, subjects, n_splits=n_splits, split=SPLIT_SUBJECT, random_state=random_state
    )
    assert_no_subject_leakage(emotion_folds, subjects)

    if verbose:
        print(f"  [2/3] emosi pada fitur mentah ({n_splits}-fold per subjek) ...")

    # Random Forest diimpor malas: DLL `_gradient_boosting` dapat diblokir
    # kebijakan Application Control, dan satu klasifier yang tidak tersedia
    # tidak boleh menggagalkan seluruh diagnosa.
    skipped: Dict[str, str] = {}
    candidates: List[Tuple[str, Any, bool]] = [
        (
            "svc_rbf",
            lambda: SVC(C=float(svc_cfg.get("C", 10.0)),
                        gamma=svc_cfg.get("gamma", "scale")),
            True,
        ),
        (
            "logreg",
            lambda: LogisticRegression(
                max_iter=int(logreg_cfg.get("max_iter", 3000)),
                C=float(logreg_cfg.get("C", 1.0)),
            ),
            True,
        ),
    ]
    try:
        from sklearn.ensemble import RandomForestClassifier

        candidates.append(
            (
                "random_forest",
                lambda: RandomForestClassifier(
                    n_estimators=int(rf_cfg.get("n_estimators", 300)),
                    n_jobs=int(rf_cfg.get("n_jobs", -1)),
                    random_state=int(rf_cfg.get("random_state", 0)),
                ),
                False,  # Random Forest tidak butuh standardisasi
            )
        )
    except Exception as exc:  # pragma: no cover - bergantung kebijakan mesin
        skipped["random_forest"] = f"{type(exc).__name__}: {exc}"

    result["raw_features"] = {}
    for name, make_classifier, scale in candidates:
        res = _cv_accuracy(make_classifier, F, y, emotion_folds, scale=scale)
        res["excess_over_chance"] = excess_over_chance(
            res["accuracy"], result["chance_expression"]
        )
        result["raw_features"][name] = res
        if verbose:
            print(f"        {name:14s} {res['accuracy'] * 100:6.2f}%  ({res['elapsed_s']}s)")

    if skipped:
        result["skipped_classifiers"] = skipped
        if verbose:
            for name, reason in skipped.items():
                print(f"        {name:14s} DILEWATI ({reason.split(':')[0]})")

    # --- (3) emosi melalui LDA nested -------------------------------------- #
    lda_cfg = config.get("lda", {})
    n_components = lda_cfg.get("n_components") or (len(class_names) - 1)

    if verbose:
        print(f"  [3/3] emosi melalui LDA nested ({n_components} komponen) ...")
    lda_accuracies: List[float] = []
    t0 = time.perf_counter()
    for tr, te in emotion_folds:
        lda = LinearDiscriminantAnalysis(
            solver=lda_cfg.get("solver", "svd"),
            n_components=n_components,
        )
        Xtr = lda.fit_transform(F[tr], y[tr])
        Xte = lda.transform(F[te])

        model = SVC(C=float(svc_cfg.get("C", 10.0)), gamma=svc_cfg.get("gamma", "scale"))
        model.fit(Xtr, y[tr])
        lda_accuracies.append(float((model.predict(Xte) == y[te]).mean()))

    result["lda_nested"] = {
        "accuracy": float(np.mean(lda_accuracies)),
        "std": float(np.std(lda_accuracies)),
        "fold_accuracies": [round(a, 4) for a in lda_accuracies],
        "n_components": int(n_components),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    result["lda_nested"]["excess_over_chance"] = excess_over_chance(
        result["lda_nested"]["accuracy"], result["chance_expression"]
    )
    if verbose:
        print(
            f"        SVC pada LDA nested {result['lda_nested']['accuracy'] * 100:6.2f}%  "
            f"({result['lda_nested']['elapsed_s']}s)"
        )

    # --- ringkasan --------------------------------------------------------- #
    if not result["raw_features"]:
        raise RuntimeError(
            "Tidak ada klasifier pembanding yang dapat dimuat di lingkungan ini."
        )
    best_raw_name = max(result["raw_features"], key=lambda k: result["raw_features"][k]["accuracy"])
    best_raw = result["raw_features"][best_raw_name]
    result["summary"] = {
        "subject_identification_accuracy": subject_ident["accuracy"],
        "subject_identification_excess": subject_ident["excess_over_chance"],
        "best_raw_classifier": best_raw_name,
        "best_raw_accuracy": best_raw["accuracy"],
        "best_raw_excess": best_raw["excess_over_chance"],
        "lda_nested_accuracy": result["lda_nested"]["accuracy"],
        "lda_penalty_points": float(best_raw["accuracy"] - result["lda_nested"]["accuracy"]),
        "identity_dominates_expression": bool(
            subject_ident["excess_over_chance"] > best_raw["excess_over_chance"]
        ),
    }
    return result


def describe(result: Dict[str, Any]) -> List[str]:
    """Susun kalimat interpretasi dari satu hasil diagnosa."""
    s = result["summary"]
    lines = [
        f"fitur memuat identitas subjek {s['subject_identification_accuracy'] * 100:.2f}% "
        f"(peluang {result['subject_identification']['chance'] * 100:.2f}%)",
        f"emosi pada fitur mentah terbaik {s['best_raw_accuracy'] * 100:.2f}% "
        f"({s['best_raw_classifier']})",
        f"emosi melalui LDA nested {s['lda_nested_accuracy'] * 100:.2f}%",
    ]
    if s["lda_penalty_points"] > 0.05:
        lines.append(
            f"LDA membuang {s['lda_penalty_points'] * 100:.2f} poin persen informasi ekspresi "
            "pada protokol subject-independent"
        )
    if s["identity_dominates_expression"]:
        lines.append(
            "sinyal identitas lebih kuat daripada sinyal ekspresi — akurasi split acak "
            "tidak boleh diklaim sebagai kemampuan generalisasi"
        )
    return lines


def write_summary_csv(reports_dir: Path, prefix: str = "identity") -> Optional[Path]:
    """
    Bangun ulang CSV ringkasan dari seluruh berkas JSON yang ada.

    Dipisah agar menjalankan satu dataset saja tidak menghapus baris dataset
    lain yang sudah pernah didiagnosa.

    Memakai modul ``csv`` bawaan Python, bukan pandas: pada mesin dengan
    kebijakan Application Control (WDAC/Device Guard) DLL ``pandas._libs._ujson``
    dapat diblokir sehingga ``import pandas`` gagal total.
    """
    import csv

    rows = []
    for path in sorted(Path(reports_dir).glob(f"{prefix}_*.json")):
        dataset_key = path.stem[len(prefix) + 1 :]
        with open(path, "r", encoding="utf-8") as fh:
            result = json.load(fh)
        s = result["summary"]
        rows.append(
            {
                "Dataset": dataset_key.upper(),
                "Samples": result["n_samples"],
                "Subjects": result["n_subjects"],
                "Chance Subject (%)": round(result["subject_identification"]["chance"] * 100, 2),
                "Subject Identification (%)": round(s["subject_identification_accuracy"] * 100, 2),
                "Chance Emotion (%)": round(result["chance_expression"] * 100, 2),
                "Best Raw Emotion (%)": round(s["best_raw_accuracy"] * 100, 2),
                "Best Raw Classifier": s["best_raw_classifier"],
                "LDA Nested (%)": round(s["lda_nested_accuracy"] * 100, 2),
                "LDA Penalty (pts)": round(s["lda_penalty_points"] * 100, 2),
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: diagnosa identitas untuk CK+ dan JAFFE."""
    parser = argparse.ArgumentParser(
        description="Ukur seberapa besar fitur 680-D memuat identitas subjek."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["ck", "jaffe"],
        choices=["ck", "jaffe"],
    )
    parser.add_argument("--out-prefix", default="identity")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="hanya bangun ulang CSV dari JSON yang sudah ada (tanpa evaluasi)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    reports_dir = resolve_path(
        config, config["paths"].get("a1_reports_dir", config["paths"]["reports_dir"])
    )
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        csv_path = write_summary_csv(reports_dir, args.out_prefix)
        print(f"[simpan] {csv_path}" if csv_path else "[info] belum ada JSON hasil")
        return

    for key in args.datasets:
        features_path = config["paths"][f"{key}_features"]
        if not resolve_path(config, features_path).exists():
            print(f"[lewati] {key}: {features_path} tidak ditemukan")
            continue

        print(f"\n=== {key.upper()} ===")
        features = load_features(features_path, config=config)
        result = run_diagnostics(features, config, verbose=True)

        out_path = reports_dir / f"{args.out_prefix}_{key}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"  [simpan] {out_path}")

        for line in describe(result):
            print(f"  - {line}")

    csv_path = write_summary_csv(reports_dir, args.out_prefix)
    if csv_path:
        print("\n=== RINGKASAN DIAGNOSA IDENTITAS ===")
        print(Path(csv_path).read_text(encoding="utf-8"))
        print(f"[simpan] {csv_path}")


if __name__ == "__main__":
    main()
