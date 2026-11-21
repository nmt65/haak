# Downloads WiMANS from Kaggle and trains the CSI baseline on 3 seeds.
# Prerequisites (done by you, once):
#   1. kaggle.com -> Settings -> API -> "Create New Token"; save kaggle.json in %USERPROFILE%\.kaggle\
#   2. Open https://www.kaggle.com/datasets/shuokanghuang/wimans and accept its terms.
# Run from the neurovision_benchmark folder:  powershell -ExecutionPolicy Bypass -File .\train_wimans.ps1
# Optional:  -Limit 500  for a quick first pass;  -WithYolo  once YOLO26n is approved (camera + fusion models).

param(
    [int]$Limit = 0,
    [switch]$WithYolo
)
$ErrorActionPreference = "Stop"
$python = "$env:LOCALAPPDATA\neurovision\venv\Scripts\python.exe"

if (-not (Test-Path "$env:USERPROFILE\.kaggle\kaggle.json")) {
    throw "Lipsește $env:USERPROFILE\.kaggle\kaggle.json. Creează tokenul Kaggle (vezi comentariul de sus)."
}

& $python -m pip install kaggle -q

# 1. Download (skipped if the archive is already there).
New-Item -ItemType Directory -Force data/downloads, data/raw/WiMANS | Out-Null
if (-not (Get-ChildItem data/downloads -Filter "wimans*.zip" -ErrorAction SilentlyContinue)) {
    & $python -m kaggle datasets download -d shuokanghuang/wimans -p data/downloads
}

# 2. Show the archive layout BEFORE extracting, so the folder structure can be verified.
$archive = Get-ChildItem data/downloads -Filter "wimans*.zip" | Select-Object -First 1
& $python -c "import zipfile,sys; names=zipfile.ZipFile(sys.argv[1]).namelist(); print(len(names),'files'); print('\n'.join(names[:15]))" $archive.FullName
if (-not (Test-Path data/raw/WiMANS/annotation.csv)) {
    Expand-Archive -Path $archive.FullName -DestinationPath data/raw/WiMANS -Force
}
# The archive may contain an extra top-level folder: find the real dataset root.
$annotation = Get-ChildItem data/raw/WiMANS -Recurse -Filter annotation.csv | Select-Object -First 1
if (-not $annotation) { throw "annotation.csv nu a fost găsit după dezarhivare. Verifică structura arhivei afișată mai sus." }
$root = $annotation.DirectoryName
Write-Host "Rădăcina WiMANS: $root"

# 3. Prepare + train.
$limitArgs = @(); if ($Limit -gt 0) { $limitArgs = @("--limit", $Limit) }
if ($WithYolo) {
    & $python -m pip install "ultralytics>=8.4" -q
    & $python prepare_wimans.py --root $root --with-yolo --weights yolo26n.pt --output data/processed/wimans_paired.npz @limitArgs
    & $python run_seeds.py --data data/processed/wimans_paired.npz --mode all --epochs 10 --seeds 42,43,44 --run-dir runs/wimans_three_seed
    & $python train.py --data data/processed/wimans_paired.npz --mode all --epochs 10 --seed 42 --run-dir runs/wimans_fusion
} else {
    & $python prepare_wimans.py --root $root --output data/processed/wimans_presence.npz @limitArgs
    & $python run_seeds.py --data data/processed/wimans_presence.npz --mode csi --epochs 10 --seeds 42,43,44 --run-dir runs/wimans_csi_three_seed
    & $python train.py --data data/processed/wimans_presence.npz --mode csi --epochs 10 --seed 42 --run-dir runs/wimans_csi
}
Write-Host "Gata. Repornește backend-ul ca să încarce modelele antrenate pe WiMANS."
