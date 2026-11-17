# ============================================================================
#  Runner TA-PCVK — satu pintu untuk semua perintah umum.
#
#  Pemakaian:
#    .\run.ps1 test        jalankan seluruh unit test
#    .\run.ps1 pipeline    buka notebook pipeline lengkap
#    .\run.ps1 noise       bangun artefak citra ber-noise (A1)
#    .\run.ps1 split       evaluasi split acak vs subject-independent (A1)
#    .\run.ps1 baseline    evaluasi baseline LDA + SVM-RBF (A3)
#    .\run.ps1 lssvm       evaluasi LS-SVM OvO + grid search (A3)
#    .\run.ps1 report      tampilkan semua tabel hasil
#    .\run.ps1 check       diagnosa lingkungan (DLL yang diblokir)
#    .\run.ps1 clean       hapus sisa duplikat setelah restrukturisasi
# ============================================================================
param(
    [Parameter(Position = 0)]
    [ValidateSet("test", "pipeline", "noise", "split", "baseline", "lssvm", "report", "check", "clean", "help")]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "[error] Python venv tidak ditemukan: $Py" -ForegroundColor Red
    Write-Host "        Buat dulu: uv venv --python 3.14 .venv && uv pip install -r requirements.txt"
    exit 1
}

# Windows console default (cp1252) tidak bisa mencetak karakter non-ASCII.
$env:PYTHONIOENCODING = "utf-8"

function Invoke-Py {
    param([string[]]$Arguments)
    & $Py @Arguments
    exit $LASTEXITCODE
}

switch ($Task) {
    "test" {
        Write-Host "[test] pytest" -ForegroundColor Cyan
        Invoke-Py @("-m", "pytest", "-v")
    }
    "pipeline" {
        $nb = "research\pipeline\00-pipeline-lengkap.ipynb"
        Write-Host "[pipeline] $nb" -ForegroundColor Cyan
        Write-Host "Buka di VS Code/Jupyter lalu Run All." -ForegroundColor Yellow
        Invoke-Py @("-m", "jupyter", "notebook", $nb)
    }
    "noise" {
        Write-Host "[noise] artefak citra ber-noise (A1)" -ForegroundColor Cyan
        Invoke-Py @("research\a1-data-preprocessing\noise.py")
    }
    "split" {
        Write-Host "[split] split acak vs subject-independent (A1)" -ForegroundColor Cyan
        Invoke-Py @("research\a1-data-preprocessing\evaluate_subject_independent.py")
    }
    "baseline" {
        Write-Host "[baseline] LDA + SVM-RBF (A3)" -ForegroundColor Cyan
        Invoke-Py @("research\a3-model-optimization\evaluate_baseline.py")
    }
    "lssvm" {
        Write-Host "[lssvm] LS-SVM OvO + grid search (A3)" -ForegroundColor Cyan
        Invoke-Py @("research\a3-model-optimization\evaluate_lssvm.py")
    }
    "report" {
        Write-Host "[report] tabel hasil" -ForegroundColor Cyan
        $files = @(
            "reports\a1-subject-independent\subject_independent_summary.csv",
            "reports\a3-models\baseline_summary.csv",
            "reports\a3-models\lssvm_multiclass_summary.csv"
        )
        foreach ($f in $files) {
            if (Test-Path $f) {
                Write-Host "`n--- $f ---" -ForegroundColor Green
                Get-Content $f
            }
        }
    }
    "clean" {
        Write-Host "[clean] hapus sisa duplikat setelah restrukturisasi" -ForegroundColor Cyan
        $targets = @(
            "data\ck_features.npz",
            "data\jaffe_features.npz",
            "model-experiment",
            "feature-engineering",
            "reports\figures"
        )
        $gagal = 0
        foreach ($t in $targets) {
            if (-not (Test-Path $t)) { continue }
            try {
                Remove-Item $t -Recurse -Force -ErrorAction Stop
                Write-Host "  [hapus] $t" -ForegroundColor Green
            }
            catch {
                Write-Host "  [terkunci] $t" -ForegroundColor Yellow
                Write-Host "            Tutup kernel Jupyter/VS Code lalu jalankan lagi." -ForegroundColor Yellow
                $gagal++
            }
        }
        if ($gagal -eq 0) { Write-Host "`nBersih." -ForegroundColor Green }
        exit 0
    }
    "check" {
        Write-Host "[check] diagnosa lingkungan" -ForegroundColor Cyan
        Invoke-Py @("scripts\check_env.py")
    }
    default {
        Get-Content $PSCommandPath | Select-Object -First 15
    }
}
