$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv")) { py -3.12 -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python hth\preprocess.py --input data\source --output build\preprocessed --config config\preprocess.json --derive --contact-sheets --overwrite
