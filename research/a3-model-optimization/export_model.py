"""
Skrip Ekspor Parameter Model Terlatih ke Format .npz (Divisi A3) - TA-PCVK.

Mengekspor seluruh komponen parameter yang dibutuhkan oleh A4:
1. LDA scalings dan xbar
2. StandardScaler mean dan scale
3. Daftar class names
4. Seluruh support vectors, dual coefficients (alpha), dan bias (b) dari 21 classifier biner OvO
5. Menyediakan salinan langsung ke reports/a3-models/ dan app/

Referensi:
Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021).
A hybrid feature descriptor with Jaya optimised least squares SVM for facial expression recognition.
"""

from pathlib import Path
import shutil
from typing import Any, Dict, Optional
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from baseline import load_config
from lssvm import OneVsOneLSSVMClassifier
from inference import FERInferenceEngine


def train_and_export_model(
    features_path: Path,
    output_path: Path,
    gamma: float = 10.0,
    sigma: float = 2.0,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Melatih pipeline lengkap pada seluruh dataset dan menyimpan artefak ke .npz.
    """
    if config is None:
        config = load_config()

    data = np.load(features_path)
    X = data["F"].astype(np.float64)
    y = data["y"]
    class_names = [str(c) for c in data["class_names"]]

    # 1. Proyeksi LDA
    lda = LinearDiscriminantAnalysis(solver="svd")
    X_lda = lda.fit_transform(X, y)

    # 2. Standarisasi Fitur LDA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_lda)

    # 3. Classifier OvO LS-SVM
    clf = OneVsOneLSSVMClassifier(gamma=gamma, sigma=sigma, kernel="rbf")
    clf.fit(X_scaled, y)

    # 4. Kumpulkan seluruh array untuk ekspor .npz
    export_dict: Dict[str, Any] = {
        "lda_scalings": lda.scalings_.astype(np.float64),  # (680, 6)
        "lda_xbar": lda.xbar_.astype(np.float64),          # (680,)
        "scaler_mean": scaler.mean_.astype(np.float64),    # (6,)
        "scaler_scale": scaler.scale_.astype(np.float64),  # (6,)
        "class_names": np.array(class_names),
        "gamma": float(gamma),
        "sigma": float(sigma),
        "kernel": "rbf",
        "num_classes": len(class_names),
        "num_features_raw": X.shape[1],
        "num_components_lda": X_lda.shape[1],
    }

    # Simpan bobot tiap pasangan model biner
    for (c1, c2), bin_clf in clf.classifiers_.items():
        prefix = f"pair_{c1}_{c2}_"
        export_dict[f"{prefix}alpha"] = bin_clf.alpha_.astype(np.float64)
        export_dict[f"{prefix}b"] = float(bin_clf.b_)
        export_dict[f"{prefix}X"] = bin_clf.X_train_.astype(np.float64)
        export_dict[f"{prefix}y"] = bin_clf.y_train_encoded_.astype(np.float64)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **export_dict)
    print(f"[EXPORT] Model berhasil diekspor ke: {output_path} (ukuran: {output_path.stat().st_size / 1024:.1f} KB)")

    # 5. Verifikasi paritas inferensi engine terhadap model asli
    engine = FERInferenceEngine(output_path)
    engine_preds, _, _ = engine.predict(X)
    orig_preds = clf.predict(X_scaled)
    np.testing.assert_array_equal(
        engine_preds, orig_preds, err_msg="Paritas engine inferensi tidak cocok dengan model asli!"
    )
    acc = float(np.mean(engine_preds == y)) * 100.0
    print(f"[VERIFIKASI] Paritas engine inferensi 100% cocok! Akurasi pelatihan: {acc:.2f}%")

    return output_path


def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parents[2]

    ck_feat_path = repo_root / config["paths"]["ck_features"]
    jaffe_feat_path = repo_root / config["paths"]["jaffe_features"]
    reports_models_dir = repo_root / config["paths"]["a3_reports_dir"]
    app_dir = repo_root / "app"

    # Ekspor Model CK+
    ck_model_path = reports_models_dir / "model_ck.npz"
    train_and_export_model(
        ck_feat_path, ck_model_path, gamma=10.0, sigma=2.0, config=config
    )

    # Ekspor Model JAFFE
    jaffe_model_path = reports_models_dir / "model_jaffe.npz"
    train_and_export_model(
        jaffe_feat_path, jaffe_model_path, gamma=10.0, sigma=2.0, config=config
    )

    # Buat model default (CK+) di reports/a3-models/model.npz
    default_model_path = reports_models_dir / "model.npz"
    shutil.copyfile(ck_model_path, default_model_path)
    print(f"[DEFAULT] Model default disalin ke: {default_model_path}")

    # Salin ke folder app/ untuk kemudahan divisi A4
    app_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ck_model_path, app_dir / "model.npz")
    shutil.copyfile(
        Path(__file__).resolve().parent / "inference.py",
        app_dir / "inference.py",
    )
    print(f"[A4 HANDOFF] model.npz dan inference.py berhasil disalin ke: {app_dir}/")


if __name__ == "__main__":
    main()
