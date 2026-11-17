"""Diagnosa lingkungan: modul mana yang dapat dimuat di mesin ini.

Kebijakan Application Control (WDAC / Device Guard / Smart App Control) pada
Windows dapat memblokir DLL biner Python secara dinamis. Skrip ini melaporkan
status tiap dependensi dan menyimpulkan perintah mana yang bisa dijalankan.

Jalankan:
    .\\run.ps1 check
    # atau
    .venv\\Scripts\\python.exe scripts\\check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    ("numpy", "wajib - array & aljabar linear", True),
    ("scipy", "wajib - solver sistem linear KKT", True),
    ("sklearn", "wajib - LDA, StratifiedKFold, baseline SVM", True),
    ("pandas", "wajib - tabel hasil", True),
    ("yaml", "wajib - config.yaml", True),
    ("matplotlib", "grafik & confusion matrix", True),
    ("cv2", "pembacaan citra & Sobel (notebook A1/A2)", False),
    ("pywt", "SWT Haar (notebook A2, test_features)", False),
]


def check(name: str):
    try:
        mod = importlib.import_module(name)
        return True, getattr(mod, "__version__", "")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        return False, f"{type(exc).__name__}: {msg[:80]}"


def main() -> int:
    print("=" * 78)
    print("DIAGNOSA LINGKUNGAN TA-PCVK")
    print("=" * 78)
    print(f"Python : {sys.version.split()[0]} ({platform.machine()})")
    print(f"OS     : {platform.system()} {platform.release()}")
    print(f"Root   : {ROOT}")
    print()

    status = {}
    print(f"{'modul':<14} {'status':<8} keterangan")
    print("-" * 78)
    for name, note, required in MODULES:
        ok, info = check(name)
        status[name] = ok
        mark = "OK" if ok else "BLOKIR"
        detail = info if not ok else (f"v{info}" if info else "")
        tag = "" if required else "  (opsional)"
        print(f"{name:<14} {mark:<8} {detail:<26} {note}{tag}")
    print()

    blocked = [n for n, ok in status.items() if not ok]
    if blocked:
        print("!! Modul yang diblokir kebijakan Application Control:")
        for name in blocked:
            print(f"   - {name}")
        print()
        print("   Solusi: minta tim IT menambahkan allow rule untuk folder .venv,")
        print("   atau jalankan ulang setelah kebijakan diperbarui.")
        print()

    print("-" * 78)
    print("KEMAMPUAN YANG TERSEDIA")
    print("-" * 78)

    inti = all(status[m] for m in ("numpy", "scipy", "sklearn", "pandas", "yaml"))
    print(f"[{'x' if inti else ' '}] Modul inti (A1 & A3: noise, split, baseline, LS-SVM)")
    print(f"[{'x' if status['matplotlib'] else ' '}] Grafik & confusion matrix")
    print(f"[{'x' if status['pywt'] and status['cv2'] else ' '}] Notebook A2 & A1 (butuh pywt + cv2)")
    print(f"[{'x' if status['pywt'] else ' '}] Pipeline lengkap dengan USE_CACHE=False")
    print()

    if inti:
        print("Perintah yang bisa dijalankan sekarang:")
        print("  .\\run.ps1 test")
        print("  .\\run.ps1 noise")
        print("  .\\run.ps1 split")
        print("  .\\run.ps1 baseline")
        print("  .\\run.ps1 lssvm")
    print()

    artefak = [
        ("data/a1-preprocessing/ck_preprocessed.npz", "praproses CK+"),
        ("data/a1-preprocessing/jaffe_preprocessed.npz", "praproses JAFFE"),
        ("data/a2-features/ck_features.npz", "fitur CK+"),
        ("data/a2-features/jaffe_features.npz", "fitur JAFFE"),
    ]
    print("Artefak:")
    for rel, label in artefak:
        p = ROOT / rel
        print(f"  [{'ADA' if p.exists() else 'TIDAK ADA':9s}] {label:16s} {rel}")
    return 0 if inti else 1


if __name__ == "__main__":
    raise SystemExit(main())
