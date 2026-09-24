param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$SourcePath
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $SourcePath)) {
    throw "Source path does not exist: $SourcePath"
}

if (-not (Test-Path ".venv")) { py -3.12 -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python hth\preprocess.py `
  --input $SourcePath `
  --output build\preprocessed `
  --config config\preprocess.json `
  --derive `
  --contact-sheets `
  --overwrite
