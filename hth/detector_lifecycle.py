# detector lifecycle
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, re, shlex, shutil, tempfile, urllib.request, zipfile
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

PAGENET_REPOSITORY="https://github.com/ctensmeyer/pagenet"
PAGENET_LICENSE="BSD-3-Clause"
PAGENET_MODEL_ID="pagenet-ohio"
PAGENET_PROTOTXT_URL="https://raw.githubusercontent.com/ctensmeyer/pagenet/master/models/ohio_train_val.prototxt"
PAGENET_WEIGHTS_URL="https://raw.githubusercontent.com/ctensmeyer/pagenet/master/models/ohio_weights.caffemodel"

DHSEGMENT_REPOSITORY="https://github.com/dhlab-epfl/dhSegment"
DHSEGMENT_LICENSE="GPL-3.0"
DHSEGMENT_MODEL_ID="dhsegment-page-v0.2"
DHSEGMENT_MODEL_URL="https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/model.zip"

def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def _download(url,target):
    target=Path(target); target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as h: tmp=Path(h.name)
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response,out)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)

def _strip_named_layer(text,name):
    pattern=re.compile(r'layer\s*\{\s*name:\s*"'+re.escape(name)+r'".*?^\}',re.MULTILINE|re.DOTALL)
    return pattern.sub("",text)

def build_pagenet_deploy_prototxt(text):
    text=re.sub(r'\ninput:\s*"gt"\s*\ninput_dim:\s*1\s*\ninput_dim:\s*1\s*\ninput_dim:\s*256\s*\ninput_dim:\s*256\s*\n',"\n",text,count=1)
    for name in ("Silence","baselines_7_loss_0"): text=_strip_named_layer(text,name)
    if 'top: "out"' not in text: raise ValueError("PageNet prototxt missing output blob 'out'")
    if 'input: "gt"' in text or "SigmoidCrossEntropyLoss" in text: raise ValueError("training-only PageNet graph remains")
    return text.strip()+"\n"

def _validate_pagenet_network(net):
    """Resolve and execute PageNet's runtime output layer.

    Caffe names the output *blob* ``out``, but modern OpenCV DNN forward(name)
    expects a layer name.  Resolve the graph's actual unconnected output layer
    instead of hard-coding the historical blob name, then perform a dry-run so
    PREPARE means the model is executable in the current runtime.
    """
    names=[str(name) for name in net.getUnconnectedOutLayersNames()]
    if not names:
        raise RuntimeError("PageNet has no unconnected output layers")
    errors=[]
    blob=cv2.dnn.blobFromImage(
        np.zeros((256,256,3),dtype=np.uint8),
        scalefactor=0.0039, size=(256,256), mean=(127.0,127.0,127.0),
        swapRB=False, crop=False,
    )
    net.setInput(blob)
    for name in names:
        try:
            raw=net.forward(name)
            shape=tuple(getattr(raw,"shape",()))
            if raw is not None and len(shape) in (2,3,4):
                return name,shape
        except cv2.error as exc:
            errors.append(f"{name}: {exc}")
    detail="; ".join(errors) if errors else "no usable output tensor"
    raise RuntimeError(f"PageNet output-layer validation failed: {detail}")

def _write_env(path,values):
    """Write a shell-sourceable environment file for the current detector flow."""
    if path is None:
        return
    with Path(path).open("a",encoding="utf-8") as h:
        for k,v in values.items():
            h.write(f"{k}={shlex.quote(str(v))}\n")

def prepare_detector_legacy(detector,*,results_root,policy="reuse",github_env=None):
    detector=detector.strip().lower()
    if detector!="learned_page_mask":
        print(f"Detector lifecycle prepare: {detector} has no pre-exec hook")
        return {"detector":detector,"prepared":False}
    if policy not in {"reuse","refresh"}: raise ValueError(f"Unsupported lifecycle policy: {policy}")
    root=Path(results_root)/"models"/PAGENET_MODEL_ID
    train=root/"ohio_train_val.prototxt"; deploy=root/"ohio_deploy.prototxt"; weights=root/"ohio_weights.caffemodel"; provenance=root/"model-provenance.json"
    complete=deploy.is_file() and weights.is_file() and provenance.is_file()
    if policy=="refresh" or not complete:
        root.mkdir(parents=True,exist_ok=True)
        if policy=="refresh" or not train.is_file(): _download(PAGENET_PROTOTXT_URL,train)
        if policy=="refresh" or not weights.is_file(): _download(PAGENET_WEIGHTS_URL,weights)
        deploy.write_text(build_pagenet_deploy_prototxt(train.read_text(encoding="utf-8")),encoding="utf-8")
        payload={
            "schema_version":"1.0","model_id":PAGENET_MODEL_ID,"model_family":"PageNet",
            "training_domain":"Ohio Death Records","upstream_repository":PAGENET_REPOSITORY,
            "license":PAGENET_LICENSE,"prototxt_url":PAGENET_PROTOTXT_URL,"weights_url":PAGENET_WEIGHTS_URL,
            "prepared_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "train_prototxt_sha256":_sha256(train),"deploy_prototxt_sha256":_sha256(deploy),
            "weights_sha256":_sha256(weights),"inference_backend":"opencv-dnn-caffe",
            "input_contract":"BGR 256x256; 0.0039 * (pixel - 127)","output_blob":"out"}
        provenance.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    payload=json.loads(provenance.read_text(encoding="utf-8"))
    if payload["deploy_prototxt_sha256"]!=_sha256(deploy): raise RuntimeError("deploy prototxt SHA mismatch")
    if payload["weights_sha256"]!=_sha256(weights): raise RuntimeError("weights SHA mismatch")
    # A model is not ready merely because its files exist.  Validate the exact
    # runtime backend during PREPARE so incompatible OpenCV builds fail before
    # regression work is queued.  PageNet's released artifact is Caffe.
    try:
        net=cv2.dnn.readNet(str(weights),str(deploy),"Caffe")
        output_layer,output_shape=_validate_pagenet_network(net)
    except cv2.error as exc:
        raise RuntimeError(
            f"Learned Page-Mask cannot load PageNet with OpenCV {cv2.__version__}; "
            "the released model requires OpenCV DNN Caffe support. "
            "Install the repository's supported OpenCV dependency (opencv-python-headless<5)."
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(
            f"Learned Page-Mask PageNet validation failed with OpenCV {cv2.__version__}: {exc}"
        ) from exc
    env={
        "HTH_LEARNED_PAGE_MASK_PROTOTXT":deploy.resolve().as_posix(),
        "HTH_LEARNED_PAGE_MASK_WEIGHTS":weights.resolve().as_posix(),
        "HTH_LEARNED_PAGE_MASK_PROVENANCE":provenance.resolve().as_posix(),
        "HTH_LEARNED_PAGE_MASK_OUTPUT_LAYER":output_layer,
    }
    _write_env(github_env,env); os.environ.update(env)
    print(f"Learned Page-Mask ready: model={PAGENET_MODEL_ID} weights_sha256={payload['weights_sha256'][:12]} output_layer={output_layer} output_shape={output_shape}")
    return payload

def finalize_detector_legacy(detector,*,results_root):
    detector=detector.strip().lower()
    if detector!="learned_page_mask":
        print(f"Detector lifecycle finalize: {detector} has no post-exec hook")
        return {"detector":detector,"finalized":False}
    p=Path(results_root)/"models"/PAGENET_MODEL_ID/"model-provenance.json"
    if not p.is_file(): raise RuntimeError("Learned Page-Mask model provenance missing")
    payload=json.loads(p.read_text(encoding="utf-8"))
    print(f"Detector lifecycle finalize: {detector} weights_sha256={str(payload.get('weights_sha256') or '')[:12]}")
    return payload


def _safe_extract_zip(archive, target):
    archive=Path(archive); target=Path(target)
    target.mkdir(parents=True,exist_ok=True)
    root=target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            destination=(target/info.filename).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError(f"Refusing unsafe archive member: {info.filename}")
        zf.extractall(target)

def _find_saved_model(root):
    root=Path(root)
    candidates=sorted(
        (p.parent for p in root.rglob("saved_model.pb")),
        key=lambda p:(len(p.relative_to(root).parts),p.as_posix()),
    )
    if not candidates:
        raise RuntimeError(f"dhSegment release contains no TensorFlow SavedModel beneath {root}")
    return candidates[0]

def _prepare_dhsegment_page_mask_hook(*,results_root,policy,env_file):
    if policy not in {"reuse","refresh"}:
        raise ValueError(f"Unsupported lifecycle policy: {policy}")
    if importlib.util.find_spec("tensorflow") is None:
        raise RuntimeError(
            "dhsegment_page_mask requires TensorFlow; the regression/optimizer workflow "
            "must install the detector-specific runtime before PREPARE"
        )

    root=Path(results_root)/"models"/DHSEGMENT_MODEL_ID
    archive=root/"model.zip"
    extracted=root/"model"
    provenance=root/"model-provenance.json"

    complete=provenance.is_file() and extracted.is_dir()
    if policy=="refresh" or not complete:
        root.mkdir(parents=True,exist_ok=True)
        if policy=="refresh" or not archive.is_file():
            _download(DHSEGMENT_MODEL_URL,archive)
        if extracted.exists():
            shutil.rmtree(extracted)
        _safe_extract_zip(archive,extracted)
        model_dir=_find_saved_model(extracted)
        payload={
            "schema_version":"1.0",
            "model_id":DHSEGMENT_MODEL_ID,
            "model_family":"dhSegment",
            "variant":"v0.2 page extraction demo model",
            "upstream_repository":DHSEGMENT_REPOSITORY,
            "license":DHSEGMENT_LICENSE,
            "model_url":DHSEGMENT_MODEL_URL,
            "archive_sha256":_sha256(archive),
            "saved_model_relative_path":model_dir.relative_to(root).as_posix(),
            "prepared_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "inference_backend":"tensorflow-savedmodel",
            "serving_contract":"filename -> probs, original_shape",
        }
        provenance.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    payload=json.loads(provenance.read_text(encoding="utf-8"))
    if archive.is_file() and payload.get("archive_sha256") != _sha256(archive):
        raise RuntimeError("dhSegment model archive SHA mismatch")
    model_dir=root/str(payload["saved_model_relative_path"])
    if not (model_dir/"saved_model.pb").is_file():
        raise RuntimeError("dhSegment SavedModel is missing after preparation")

    env={
        "HTH_DHSEGMENT_PAGE_MODEL_DIR":model_dir.resolve().as_posix(),
        "HTH_DHSEGMENT_PAGE_PROVENANCE":provenance.resolve().as_posix(),
        "TF_CPP_MIN_LOG_LEVEL":"2",
        "CUDA_VISIBLE_DEVICES":"-1",
    }
    _write_env(env_file,env); os.environ.update(env)
    print(
        f"dhSegment Page-Mask ready: model={DHSEGMENT_MODEL_ID} "
        f"archive_sha256={str(payload.get('archive_sha256') or '')[:12]}"
    )
    return payload

def _finalize_dhsegment_page_mask_hook(*,results_root):
    provenance=Path(results_root)/"models"/DHSEGMENT_MODEL_ID/"model-provenance.json"
    if not provenance.is_file():
        raise RuntimeError("dhSegment Page-Mask model provenance missing")
    payload=json.loads(provenance.read_text(encoding="utf-8"))
    print(
        f"Detector lifecycle finalize: dhsegment_page_mask "
        f"archive_sha256={str(payload.get('archive_sha256') or '')[:12]}"
    )
    return payload

def _prepare_learned_page_mask_hook(*,results_root,policy,env_file):
    return prepare_detector_legacy(
        "learned_page_mask",
        results_root=results_root,
        policy=policy,
        github_env=env_file,
    )

def _finalize_learned_page_mask_hook(*,results_root):
    return finalize_detector_legacy("learned_page_mask",results_root=results_root)

_PREPARE_HOOKS={
    "dhsegment_page_mask":_prepare_dhsegment_page_mask_hook,
    "learned_page_mask":_prepare_learned_page_mask_hook,
}
_FINALIZE_HOOKS={
    "dhsegment_page_mask":_finalize_dhsegment_page_mask_hook,
    "learned_page_mask":_finalize_learned_page_mask_hook,
}

def _load_config(path):
    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    detector=str(payload.get("detector") or "").strip()
    if not detector:
        raise ValueError(f"Detector config has no detector id: {path}")
    lifecycle=payload.get("lifecycle") or {}
    if not isinstance(lifecycle,dict):
        raise ValueError(f"Detector lifecycle must be an object: {path}")
    return detector,lifecycle

def prepare_config(config_path,*,results_root,env_file=None,policy_override=None):
    detector,lifecycle=_load_config(config_path)
    hook_name=str(lifecycle.get("prepare") or "").strip()
    if not hook_name:
        print(f"Detector lifecycle prepare: {detector} has no configured hook")
        return {"detector":detector,"prepared":False}
    hook=_PREPARE_HOOKS.get(hook_name)
    if hook is None:
        raise ValueError(f"Unknown detector prepare hook {hook_name!r} for {detector}")
    policy=policy_override or str(lifecycle.get("model_policy") or "reuse")
    print(f"Detector lifecycle PREPARE detector={detector} hook={hook_name} policy={policy}")
    return hook(results_root=Path(results_root),policy=policy,env_file=env_file)

def finalize_config(config_path,*,results_root):
    detector,lifecycle=_load_config(config_path)
    hook_name=str(lifecycle.get("finalize") or lifecycle.get("post") or "").strip()
    if not hook_name:
        print(f"Detector lifecycle finalize: {detector} has no configured hook")
        return {"detector":detector,"finalized":False}
    hook=_FINALIZE_HOOKS.get(hook_name)
    if hook is None:
        raise ValueError(f"Unknown detector finalize hook {hook_name!r} for {detector}")
    print(f"Detector lifecycle FINALIZE detector={detector} hook={hook_name}")
    return hook(results_root=Path(results_root))

# Compatibility API retained for callers/tests that used the first learned-detector overlay.
def prepare_detector(detector,*,results_root,policy="reuse",github_env=None):
    detector=detector.strip().lower()
    hook=_PREPARE_HOOKS.get(detector)
    if hook is None:
        print(f"Detector lifecycle prepare: {detector} has no pre-exec hook")
        return {"detector":detector,"prepared":False}
    return hook(results_root=Path(results_root),policy=policy,env_file=github_env)

def finalize_detector(detector,*,results_root):
    detector=detector.strip().lower()
    hook=_FINALIZE_HOOKS.get(detector)
    if hook is None:
        print(f"Detector lifecycle finalize: {detector} has no post-exec hook")
        return {"detector":detector,"finalized":False}
    return hook(results_root=Path(results_root))

def main(argv=None):
    parser=argparse.ArgumentParser(description="Run detector lifecycle hooks.")
    sub=parser.add_subparsers(dest="command",required=True)

    prepare=sub.add_parser("prepare-config")
    prepare.add_argument("--config",type=Path,required=True)
    prepare.add_argument("--results-root",type=Path,required=True)
    prepare.add_argument("--env-file",type=Path)
    prepare.add_argument("--policy",choices=("reuse","refresh"))

    finalize=sub.add_parser("finalize-config")
    finalize.add_argument("--config",type=Path,required=True)
    finalize.add_argument("--results-root",type=Path,required=True)

    # Backward-compatible commands from the first overlay.
    legacy_prepare=sub.add_parser("prepare")
    legacy_prepare.add_argument("--detector",required=True)
    legacy_prepare.add_argument("--results-root",type=Path,required=True)
    legacy_prepare.add_argument("--policy",choices=("reuse","refresh"),default="reuse")
    legacy_prepare.add_argument("--github-env",type=Path)
    legacy_finalize=sub.add_parser("finalize")
    legacy_finalize.add_argument("--detector",required=True)
    legacy_finalize.add_argument("--results-root",type=Path,required=True)

    args=parser.parse_args(argv)
    if args.command=="prepare-config":
        prepare_config(args.config,results_root=args.results_root,env_file=args.env_file,policy_override=args.policy)
    elif args.command=="finalize-config":
        finalize_config(args.config,results_root=args.results_root)
    elif args.command=="prepare":
        prepare_detector(args.detector,results_root=args.results_root,policy=args.policy,github_env=args.github_env)
    else:
        finalize_detector(args.detector,results_root=args.results_root)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
