#!/usr/bin/env bash
set -euo pipefail

: "${HTH_VENV:?HTH_VENV is required}"
: "${HTH_BOOTSTRAP_PYTHON:?HTH_BOOTSTRAP_PYTHON is required}"

need_dhsegment="${HTH_NEED_DHSEGMENT:-false}"
need_doc_ufcn="${HTH_NEED_DOC_UFCN:-false}"
need_mask_rcnn="${HTH_NEED_MASK_RCNN:-false}"
need_kraken="${HTH_NEED_KRAKEN:-false}"
need_orli="${HTH_NEED_ORLI:-false}"

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

verify_doc_ufcn() {
  python - <<'PYDOCUFCN'
try:
    import torch
    from doc_ufcn.main import DocUFCN
except Exception as exc:
    print(f"Doc-UFCN runtime import failed: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
print(f"Doc-UFCN runtime verified: HTH-pinned 0.2.0rc4 source; PyTorch {torch.__version__}")
PYDOCUFCN
}

verify_mask_rcnn() {
  python - <<'PYMASKRCNN'
from importlib import metadata

expected = {"iopath": "0.1.9", "black": "24.10.0"}
for package, wanted in expected.items():
    try:
        found = metadata.version(package)
    except metadata.PackageNotFoundError:
        print(f"Mask R-CNN runtime dependency missing: {package}=={wanted}")
        raise SystemExit(1)
    if found != wanted:
        print(f"Mask R-CNN runtime dependency mismatch: expected {package}=={wanted}, found {found}")
        raise SystemExit(1)

try:
    import torch
    import torchvision
    import detectron2
    from detectron2.engine import DefaultPredictor
except Exception as exc:
    print(f"Mask R-CNN runtime import failed: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
print(
    f"Mask R-CNN runtime verified: Detectron2; PyTorch {torch.__version__}; "
    f"Torchvision {torchvision.__version__}; iopath {expected['iopath']}; black {expected['black']}"
)
PYMASKRCNN
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

verify_orli() {
  python - <<'PY'
from importlib import metadata
try:
    version = metadata.version("orli")
except metadata.PackageNotFoundError:
    print("Orli runtime not present.")
    raise SystemExit(1)
if version != "0.0.2":
    print(f"Orli runtime version mismatch: expected 0.0.2, found {version}")
    raise SystemExit(1)
from orli.pred import segment
print(f"Orli runtime verified: {version}")
PY
}

verify_complete_runtime() {
  verify_pip || return 1
  verify_base || return 1
  if [[ "$need_dhsegment" == "true" ]]; then
    verify_dhsegment || return 1
  fi
  if [[ "$need_doc_ufcn" == "true" ]]; then
    verify_doc_ufcn || return 1
  fi
  if [[ "$need_mask_rcnn" == "true" ]]; then
    verify_mask_rcnn || return 1
  fi
  if [[ "$need_kraken" == "true" ]]; then
    verify_kraken || return 1
  fi
  if [[ "$need_orli" == "true" ]]; then
    verify_kraken || return 1
    verify_orli || return 1
  fi
  python -m pip check
}

ensure_pip() {
  if ! python -m pip --version >/dev/null 2>&1; then
    echo "pip is unavailable in the managed venv; repairing with ensurepip."
    python -m ensurepip --upgrade
  fi
}

install_base_runtime() {
  ensure_pip
  python -m pip install --requirement hth-pipeline/requirements.txt
}

install_dhsegment_layer() {
  echo "Installing missing dhSegment TensorFlow layer only."
  python -m pip install "tensorflow-cpu>=2.18,<2.21"
}

install_doc_ufcn_layer() {
  if [[ "${RUNNER_OS:-}" == "Linux" ]]; then
    echo "Installing CPU-only PyTorch layer for Doc-UFCN."
    python -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cpu
  else
    python -m pip install "torch==2.10.0"
  fi
  echo "Installing HTH-pinned Doc-UFCN 0.2.0rc4 source package without legacy distribution metadata/dependency downgrades."
  doc_ufcn_stage="$(mktemp -d)"
  python -m pip install --target "$doc_ufcn_stage" --no-deps --ignore-requires-python "doc-ufcn==0.2.0rc4"
  doc_ufcn_site="$(python - <<'PYDOCUFCNSITE'
import site
print(site.getsitepackages()[0])
PYDOCUFCNSITE
)"
  rm -rf "$doc_ufcn_site/doc_ufcn"
  cp -a "$doc_ufcn_stage/doc_ufcn" "$doc_ufcn_site/doc_ufcn"
  rm -rf "$doc_ufcn_stage"
}

install_mask_rcnn_layer() {
  if [[ "${RUNNER_OS:-}" == "Linux" ]]; then
    echo "Installing CPU-only PyTorch/Torchvision layer for Mask R-CNN."
    python -m pip install "torch==2.10.0" "torchvision==0.25.0" --index-url https://download.pytorch.org/whl/cpu
  else
    python -m pip install "torch==2.10.0" "torchvision==0.25.0"
  fi
  echo "Installing pinned Detectron2 source for HJDataset Mask R-CNN."
  FORCE_CUDA=0 CUDA_VISIBLE_DEVICES='' python -m pip install ninja
  FORCE_CUDA=0 CUDA_VISIBLE_DEVICES='' python -m pip install --no-build-isolation --no-deps \
    "git+https://github.com/facebookresearch/detectron2.git@57bdb21249d5418c130d54e2ebdc94dda7a4c01a"
  python -m pip install "black==24.10.0" "cloudpickle>=2" "fvcore>=0.1.5.post20221221" "iopath==0.1.9" "matplotlib>=3.8" "omegaconf>=2.1,<2.4" "hydra-core>=1.1" "pycocotools>=2.0.7" "tensorboard>=2.18" "termcolor>=2" "tabulate" "tqdm>=4.66" "yacs>=0.1.8"
}

install_kraken_layer() {
  if [[ "${RUNNER_OS:-}" == "Linux" ]]; then
    echo "Installing missing matched CPU-only PyTorch/Torchvision layer for Kraken."
    python -m pip install \
      "torch==2.10.0" \
      "torchvision==0.25.0" \
      --index-url https://download.pytorch.org/whl/cpu
  fi
  echo "Installing missing Kraken 7.0.2 layer only."
  python -m pip install "kraken==7.0.2"
}

install_orli_layer() {
  install_kraken_layer
  echo "Installing Orli 0.0.2 historical-document model plugin layer."
  python -m pip install "orli==0.0.2"
}

runtime_backup="${HTH_VENV}.backup-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}"

restore_runtime_backup() {
  local rc="$?"
  if (( rc != 0 )); then
    echo "::error::Managed runtime update failed; restoring previous reusable runtime."
    rm -rf "$HTH_VENV"
    if [[ -d "$runtime_backup" ]]; then
      mv "$runtime_backup" "$HTH_VENV"
    fi
  fi
  return "$rc"
}

begin_full_rebuild() {
  rm -rf "$runtime_backup"
  if [[ -d "$HTH_VENV" ]]; then
    mv "$HTH_VENV" "$runtime_backup"
  fi
  trap restore_runtime_backup EXIT
  "$HTH_BOOTSTRAP_PYTHON" -m venv "$HTH_VENV"
}

begin_incremental_update() {
  rm -rf "$runtime_backup"

  # Keep the canonical venv in place so its scripts and pyvenv metadata retain
  # their original absolute paths. Snapshot it locally for rollback instead of
  # rebuilding already-verified base packages. Reflinks make this effectively
  # copy-on-write when the filesystem supports them; ordinary local copy is the
  # safe fallback.
  echo "Snapshotting verified managed runtime before incremental augmentation."
  if cp --help 2>/dev/null | grep -q -- '--reflink'; then
    cp -a --reflink=auto "$HTH_VENV" "$runtime_backup"
  else
    cp -a "$HTH_VENV" "$runtime_backup"
  fi
  trap restore_runtime_backup EXIT
}

commit_runtime_update() {
  rm -rf "$runtime_backup"
  trap - EXIT
}

# Verify the base layer first. A bad base is the only condition that justifies
# recreating the environment and reinstalling requirements.txt.
if ! verify_pip || ! verify_base; then
  echo "Managed base runtime is invalid; rebuilding the base environment once."
  begin_full_rebuild
  install_base_runtime

  if [[ "$need_dhsegment" == "true" ]]; then
    install_dhsegment_layer
  fi
  if [[ "$need_doc_ufcn" == "true" ]]; then
    install_doc_ufcn_layer
  fi
  if [[ "$need_mask_rcnn" == "true" ]]; then
    install_mask_rcnn_layer
  fi
  if [[ "$need_orli" == "true" ]]; then
    install_orli_layer
  elif [[ "$need_kraken" == "true" ]]; then
    install_kraken_layer
  fi

  verify_complete_runtime
  commit_runtime_update
  echo "Managed runtime rebuilt from base and fully verified."
  exit 0
fi

# The base layer is already valid. Determine which optional layers, if any, are
# missing before touching the environment.
missing_dhsegment=false
missing_doc_ufcn=false
missing_mask_rcnn=false
missing_kraken=false
missing_orli=false

if [[ "$need_dhsegment" == "true" ]] && ! verify_dhsegment; then
  missing_dhsegment=true
fi

if [[ "$need_doc_ufcn" == "true" ]] && ! verify_doc_ufcn; then
  missing_doc_ufcn=true
fi

if [[ "$need_mask_rcnn" == "true" ]] && ! verify_mask_rcnn; then
  missing_mask_rcnn=true
fi

if [[ "$need_kraken" == "true" ]] && ! verify_kraken; then
  missing_kraken=true
fi

if [[ "$need_orli" == "true" ]] && ! verify_kraken; then
  missing_kraken=true
fi

if [[ "$need_orli" == "true" ]] && ! verify_orli; then
  missing_orli=true
fi

if [[ "$missing_dhsegment" == "false" && "$missing_doc_ufcn" == "false" && "$missing_mask_rcnn" == "false" && "$missing_kraken" == "false" && "$missing_orli" == "false" ]]; then
  python -m pip check
  echo "Managed runtime verified — using previous install; no install required."
  exit 0
fi

echo "Managed base runtime verified; augmenting only missing optional runtime layer(s)."
begin_incremental_update

if [[ "$missing_dhsegment" == "true" ]]; then
  install_dhsegment_layer
fi
if [[ "$missing_doc_ufcn" == "true" ]]; then
  install_doc_ufcn_layer
fi
if [[ "$missing_mask_rcnn" == "true" ]]; then
  install_mask_rcnn_layer
fi
if [[ "$missing_orli" == "true" ]]; then
  install_orli_layer
elif [[ "$missing_kraken" == "true" ]]; then
  install_kraken_layer
fi

verify_complete_runtime
commit_runtime_update
echo "Managed runtime augmentation completed and fully verified."
