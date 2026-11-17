"""
Modul Uji Ketahanan Noise Gaussian (Divisi A1) - TA-PCVK.

Menambahkan noise Gaussian pada citra hasil praproses A1, lalu mengekspor
artefak baru agar A2 dapat mengekstrak ulang fitur dan A3 dapat mengevaluasi
ketahanan model terhadap noise.

Referensi:
Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021).
A hybrid feature descriptor with Jaya optimised least squares SVM for facial
expression recognition. IET Image Processing, 15(7), 1471-1483.
Bagian 5.5 menguji noise Gaussian dengan variansi antara 0.01 dan 0.1.

Catatan desain:
- Noise ditambahkan pada ruang intensitas ternormalisasi [0, 1], bukan pada
  nilai piksel mentah 0-255. Dengan demikian ``variance`` bermakna variansi
  intensitas relatif, dan standar deviasinya adalah ``sqrt(variance)``.
- Citra disimpan kembali sebagai uint8 agar A2 memakai jalur ekstraksi fitur
  yang sama persis dengan data bersih (tidak ada perbedaan tipe data).
- ``variance=0`` mengembalikan salinan identik, sehingga dapat dipakai sebagai
  kontrol/verifikasi.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import sys

import numpy as np

# Modul A3 (baseline) berada di research/a3-model-optimization/.
# Tambahkan ke sys.path agar CLI tetap bisa dijalankan langsung.
_A3_DIR = Path(__file__).resolve().parents[1] / "a3-model-optimization"
if str(_A3_DIR) not in sys.path:
    sys.path.insert(0, str(_A3_DIR))

# Variansi default mengikuti jurnal (bagian 5.5).
DEFAULT_VARIANCES: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.1)

# Pemetaan kunci dataset pada config.yaml -> nama berkas artefak A1.
DATASET_KEYS: Tuple[str, ...] = ("ck", "jaffe")


def add_gaussian_noise(
    images: np.ndarray,
    variance: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Menambahkan noise Gaussian N(0, variance) pada citra uint8.

    Parameters
    ----------
    images : np.ndarray
        Citra hasil praproses berdimensi (N, H, W) atau (H, W), dtype uint8,
        rentang 0-255.
    variance : float
        Variansi noise pada ruang intensitas [0, 1]. Nilai 0 berarti tanpa
        noise (mengembalikan salinan identik).
    seed : int, default 42
        Seed RNG agar hasil reprodusibel.

    Returns
    -------
    noisy : np.ndarray
        Citra ber-noise dengan shape, dtype, dan rentang yang sama.
    """
    arr = np.asarray(images)
    if arr.dtype != np.uint8:
        raise TypeError(f"Citra harus uint8, diterima {arr.dtype}")
    if arr.ndim not in (2, 3):
        raise ValueError(f"Citra harus 2D atau 3D (N,H,W), diterima shape {arr.shape}")
    if variance is None or float(variance) < 0:
        raise ValueError(f"variance harus >= 0, diterima {variance}")

    variance = float(variance)
    if variance == 0.0:
        return arr.copy()

    rng = np.random.default_rng(int(seed))
    base = arr.astype(np.float32) / 255.0
    sigma = float(np.sqrt(variance))
    noise = rng.normal(0.0, sigma, size=base.shape).astype(np.float32)

    noisy = base + noise
    np.clip(noisy, 0.0, 1.0, out=noisy)
    return np.rint(noisy * 255.0).astype(np.uint8)


def config_root(config: Dict[str, Any]) -> Path:
    """
    Direktori acuan untuk path relatif pada config.

    Bila config dimuat dari ``config.yaml`` di root repo, maka ``data`` dan
    ``reports`` relatif terhadap root tersebut. Ini membuat modul tetap benar
    meskipun dijalankan dari subdirektori (mis. ``model-experiment/``).
    """
    raw = config.get("_config_path")
    if raw:
        return Path(str(raw)).resolve().parent
    return Path.cwd()


def resolve_path(config: Dict[str, Any], value: Union[str, Path]) -> Path:
    """Ubah path config menjadi absolut, dengan fallback ke root config."""
    p = Path(str(value))
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return config_root(config) / p


def noise_artifact_path(
    config: Dict[str, Any],
    dataset_key: str,
    variance: float,
) -> Path:
    """Path artefak citra ber-noise, mis. data/a1-noise/ck_noise_v0.01.npz."""
    raw_dir = config["paths"].get("a1_noise_dir") or config["paths"]["data_dir"]
    data_dir = resolve_path(config, raw_dir)
    return data_dir / f"{dataset_key}_noise_v{float(variance):.2f}.npz"


def preprocessed_path(config: Dict[str, Any], dataset_key: str) -> Path:
    """Path artefak praproses A1 untuk satu dataset."""
    return resolve_path(config, config["paths"][f"{dataset_key}_preprocessed"])


def build_noise_artifacts(
    config: Dict[str, Any],
    variances: Optional[Sequence[float]] = None,
    dataset_keys: Optional[Iterable[str]] = None,
    force: bool = False,
    verbose: bool = True,
) -> List[Path]:
    """
    Membangun artefak citra ber-noise untuk seluruh dataset dan variansi.

    Setiap artefak berisi kunci yang sama dengan artefak praproses A1
    (X, y, subjects, class_names) ditambah metadata noise (variance, seed),
    sehingga A2 cukup memanggil ``extract_features(X)`` seperti biasa.

    Returns
    -------
    written : list[Path]
        Daftar berkas yang ditulis (atau sudah ada bila ``force=False``).
    """
    variances = tuple(DEFAULT_VARIANCES if variances is None else variances)
    keys = tuple(DATASET_KEYS if dataset_keys is None else dataset_keys)
    seed = int(config.get("project", {}).get("random_seed", 42))

    written: List[Path] = []
    for key in keys:
        src = preprocessed_path(config, key)
        if not src.exists():
            if verbose:
                print(f"[noise] lewati {key}: {src} tidak ditemukan")
            continue

        data = np.load(src, allow_pickle=True)
        images = data["X"]
        y = data["y"]
        subjects = data["subjects"]
        class_names = data["class_names"]

        if verbose:
            print(f"[noise] {key}: {images.shape[0]} citra, {len(class_names)} kelas")

        for variance in variances:
            dst = noise_artifact_path(config, key, variance)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and not force:
                if verbose:
                    print(f"  [skip] {dst.name} sudah ada")
                written.append(dst)
                continue

            noisy = add_gaussian_noise(images, variance, seed=seed)
            np.savez_compressed(
                dst,
                X=noisy,
                y=y,
                subjects=subjects,
                class_names=class_names,
                variance=np.float64(variance),
                seed=np.int64(seed),
                source=np.array(src.name),
            )
            if verbose:
                print(f"  [tulis] {dst.name}  (sigma={np.sqrt(variance):.4f})")
            written.append(dst)

    return written


def load_noise_artifact(path: Union[str, Path]) -> Dict[str, Any]:
    """Memuat artefak citra ber-noise dan mengembalikan dict biasa."""
    data = np.load(Path(path), allow_pickle=True)
    out: Dict[str, Any] = {k: data[k] for k in data.files}
    if "variance" in out:
        out["variance"] = float(np.asarray(out["variance"]).item())
    if "seed" in out:
        out["seed"] = int(np.asarray(out["seed"]).item())
    return out


def noise_statistics(
    clean: np.ndarray,
    noisy: np.ndarray,
) -> Dict[str, float]:
    """
    Statistik ringkas untuk memverifikasi noise yang ditambahkan.

    Dihitung pada ruang [0, 1] agar ``empirical_variance`` dapat dibandingkan
    langsung dengan nilai ``variance`` yang diminta (sebelum clipping).
    """
    a = np.asarray(clean, dtype=np.float64) / 255.0
    b = np.asarray(noisy, dtype=np.float64) / 255.0
    diff = (b - a).ravel()
    return {
        "mean_shift": float(np.mean(diff)),
        "empirical_variance": float(np.var(diff)),
        "empirical_std": float(np.std(diff)),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "saturated_fraction": float(np.mean((a.ravel() <= 0.0) | (a.ravel() >= 1.0))),
    }


def main() -> None:
    """CLI: bangun seluruh artefak noise dari config.yaml."""
    import argparse

    from baseline import load_config

    parser = argparse.ArgumentParser(
        description="Bangun artefak citra ber-noise Gaussian (uji ketahanan A1)."
    )
    parser.add_argument("--config", default="config.yaml", help="path config YAML")
    parser.add_argument(
        "--variances",
        nargs="*",
        type=float,
        default=None,
        help=f"daftar variansi (default: {DEFAULT_VARIANCES})",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        choices=list(DATASET_KEYS),
        help="batasi dataset tertentu",
    )
    parser.add_argument("--force", action="store_true", help="tulis ulang bila sudah ada")
    args = parser.parse_args()

    config = load_config(args.config)
    written = build_noise_artifacts(
        config,
        variances=args.variances,
        dataset_keys=args.datasets,
        force=args.force,
    )
    print(f"\nSelesai. {len(written)} artefak noise tersedia di {config['paths'].get('a1_noise_dir', config['paths']['data_dir'])}/")


if __name__ == "__main__":
    main()
