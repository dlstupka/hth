#!/usr/bin/env bash
set -euo pipefail

: "${HTH_VENV:?HTH_VENV is required}"
: "${HTH_BOOTSTRAP_PYTHON:?HTH_BOOTSTRAP_PYTHON is required}"

need_dhsegment="${HTH_NEED_DHSEGMENT:-false}"
need_kraken="${HTH_NEED_KRAKEN:-false}"

verify_pip() {
  python -m pip --version >/dev/null 2>&1
}

verify_base() {
  python - <<'PY'
from importlib import metadata
from pathlib import Path
from pip._vendor.packaging.requirements import Requirement

failures = []
for raw in Path("hth-pipeline/requirements.txt").read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or line.startswith(("-", "http:", "https:", "git+")):
        continue
    try:
        req = Requirement(line)
    except Exception as exc:
        failures.append(f"{line}: cannot verify ({exc})")
        continue
    if req.marker is not None and not req.marker.evaluate():
        continue
    try:
        installed = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        failures.append(f"{req.name}: not installed")
        continue
    if req.specifier and installed not in req.specifier:
        failures.append(f"{req.name}: installed {installed}, requires {req.specifier}")

if failures:
    print("Base dependency verification failed:")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("Base dependencies verified.")
PY
}

verify_dhsegment() {
  python - <<'PY'
from importlib import metadata
from pip._vendor.packaging.specifiers import SpecifierSet

try:
    version = metadata.version("tensorflow-cpu")
except metadata.PackageNotFoundError:
    print("dhSegment TensorFlow runtime not present.")
    raise SystemExit(1)

required = SpecifierSet(">=2.18,<2.21")
if version not in required:
    print(f"dhSegment TensorFlow runtime version mismatch: {version} not in {required}")
    raise SystemExit(1)

try:
    import tensorflow as tf
except Exception as exc:
    print(f"dhSegment TensorFlow runtime import failed: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

print(f"dhSegment TensorFlow runtime verified: {tf.__version__}")
PY
}

verify_kraken() {
  python - <<'PY'
from importlib import metadata

try:
    version = metadata.version("kraken")
except metadata.PackageNotFoundError:
    print("Kraken runtime not present.")
    raise SystemExit(1)

if version != "7.0.2":
    print(f"Kraken runtime version mismatch: expected 7.0.2, found {version}")
    raise SystemExit(1)

try:
    import torch
    import torchvision
    from kraken.tasks.segmentation import SegmentationTaskModel
except Exception as exc:
    print(f"Kraken runtime import failed: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

print(f"PyTorch verified    : {torch.__version__}")
print(f"Torchvision verified: {torchvision.__version__}")
PY
  if [[ "${RUNNER_OS:-}" == "Linux" ]]; then
    python - <<'PY'
import torch
import torchvision

if torch.cuda.is_available():
    raise SystemExit("HTH Kraken runtime unexpectedly reports CUDA available")
if "+cpu" not in torch.__version__:
    raise SystemExit(f"Expected CPU-only PyTorch wheel, got {torch.__version__}")
if "+cpu" not in torchvision.__version__:
    raise SystemExit(f"Expected CPU-only Torchvision wheel, got {torchvision.__version__}")
print("Kraken PyTorch/Torchvision backend verified: CPU-only")
PY
  fi
  python - <<'PY'
from kraken.tasks.segmentation import SegmentationTaskModel
print("Kraken SegmentationTaskModel import verified.")
PY
}

verify_complete_runtime() {
  verify_pip || return 1
  verify_base || return 1
  if [[ "$need_dhsegment" == "true" ]]; then
    verify_dhsegment || return 1
  fi
  if [[ "$need_kraken" == "true" ]]; then
    verify_kraken || return 1
  fi
  python -m pip check
}

install_complete_runtime() {
  # Do not upgrade pip merely because a newer release exists. The venv's pip is
  # sufficient unless it is actually absent/broken.
  if ! python -m pip --version >/dev/null 2>&1; then
    echo "pip is unavailable in the new venv; repairing with ensurepip."
    python -m ensurepip --upgrade
  fi

  python -m pip install --requirement hth-pipeline/requirements.txt

  if [[ "$need_dhsegment" == "true" ]]; then
    python -m pip install "tensorflow-cpu>=2.18,<2.21"
  fi

  if [[ "$need_kraken" == "true" ]]; then
    if [[ "${RUNNER_OS:-}" == "Linux" ]]; then
      echo "Installing matched CPU-only PyTorch/Torchvision pair for Kraken."
      python -m pip install \
        "torch==2.10.0" \
        "torchvision==0.25.0" \
        --index-url https://download.pytorch.org/whl/cpu
    fi
    python -m pip install "kraken==7.0.2"
  fi
}

if verify_complete_runtime; then
  echo "Managed runtime verified — using previous install; no install required."
  exit 0
fi

echo "Managed runtime is incomplete for this job; rebuilding once with the complete required dependency set."

backup="${HTH_VENV}.backup-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}"
rm -rf "$backup"
if [[ -d "$HTH_VENV" ]]; then
  mv "$HTH_VENV" "$backup"
fi

restore_previous_runtime() {
  rc="$?"
  if (( rc != 0 )); then
    echo "::error::Managed runtime rebuild failed; restoring previous reusable runtime."
    rm -rf "$HTH_VENV"
    if [[ -d "$backup" ]]; then
      mv "$backup" "$HTH_VENV"
    fi
  fi
  return "$rc"
}
trap restore_previous_runtime EXIT

"$HTH_BOOTSTRAP_PYTHON" -m venv "$HTH_VENV"
install_complete_runtime
verify_complete_runtime

rm -rf "$backup"
trap - EXIT
echo "Managed runtime rebuilt once and fully verified."
