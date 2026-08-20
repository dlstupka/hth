# detector lifecycle
from __future__ import annotations
import argparse, hashlib, importlib.metadata, importlib.resources, importlib.util, json, os, re, shlex, shutil, tempfile, urllib.request, zipfile
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from hth.model_variants import ModelSource, resolve_model_variant

PAGENET_REPOSITORY="https://github.com/ctensmeyer/pagenet"
PAGENET_LICENSE="BSD-3-Clause"
PAGENET_MODEL_ID="pagenet-ohio"
PAGENET_PROTOTXT_URL="https://raw.githubusercontent.com/ctensmeyer/pagenet/master/models/ohio_train_val.prototxt"
PAGENET_WEIGHTS_URL="https://raw.githubusercontent.com/ctensmeyer/pagenet/master/models/ohio_weights.caffemodel"

DHSEGMENT_REPOSITORY="https://github.com/dhlab-epfl/dhSegment"
DHSEGMENT_LICENSE="GPL-3.0"
DHSEGMENT_MODEL_ID="dhsegment-page-v0.2"
DHSEGMENT_MODEL_URL="https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/model.zip"

KRAKEN_REPOSITORY="https://github.com/mittagessen/kraken"
KRAKEN_LICENSE="Apache-2.0"
KRAKEN_PACKAGE_VERSION="7.0.2"
KRAKEN_MODEL_ID="kraken-blla-default-7.0.2"

ORLI_REPOSITORY="https://pypi.org/project/orli/"
ORLI_LICENSE="Apache-2.0"
ORLI_PACKAGE_VERSION="0.0.2"
ORLI_MODEL_ID="orli-base-2026"
ORLI_MODEL_URL="https://zenodo.org/records/20558179/files/orli_base.safetensors?download=1"
ORLI_MODEL_DOI="10.5281/zenodo.20558179"

DOC_UFCN_REPOSITORY="https://github.com/johnlockejrr/doc-ufcn"
DOC_UFCN_LICENSE="BSD-3-Clause"
DOC_UFCN_PACKAGE_VERSION="0.2.0rc4"
DOC_UFCN_MODEL_ID="doc-ufcn-generic-page"
DOC_UFCN_MODEL_URL="https://huggingface.co/Teklia/doc-ufcn-generic-page/resolve/main/model.pth?download=true"
DOC_UFCN_PARAMETERS_URL="https://huggingface.co/Teklia/doc-ufcn-generic-page/resolve/main/parameters.yml?download=true"
DOC_UFCN_MODEL_REPOSITORY="https://huggingface.co/Teklia/doc-ufcn-generic-page"

MASK_RCNN_REPOSITORY="https://github.com/Layout-Parser/layout-parser"
MASK_RCNN_LICENSE="Apache-2.0"
MASK_RCNN_MODEL_ID="hjdataset-mask-rcnn-r50-fpn-3x"
MASK_RCNN_MODEL_URL="https://huggingface.co/layoutparser/detectron2/resolve/main/HJDataset/mask_rcnn_R_50_FPN_3x/model_final.pth?download=true"
MASK_RCNN_CONFIG_URL="https://huggingface.co/layoutparser/detectron2/resolve/main/HJDataset/mask_rcnn_R_50_FPN_3x/config.yml?download=true"
MASK_RCNN_MODEL_REPOSITORY="https://huggingface.co/layoutparser/detectron2/tree/main/HJDataset/mask_rcnn_R_50_FPN_3x"

EYNOLLAH_REPOSITORY="https://github.com/qurator-spk/eynollah"
EYNOLLAH_LICENSE="Apache-2.0"
EYNOLLAH_MODEL_ID="eynollah-page-extraction-2021-04-25"
EYNOLLAH_HF_REPOSITORY="https://huggingface.co/SBB/eynollah-page-extraction"
EYNOLLAH_HF_REFS=("main","fd3ea7df60462d97796520916326929e7e42c2fb")
EYNOLLAH_HF_LEGACY_REF="d2b86773d6a43eac8e18101ed1e5109565ea057e"
EYNOLLAH_SAVED_MODEL_SHA256="6a9639d6f77afec409d0fdb18f41ab3978ff1686eae10a0ce262ebfbd9f689a0"

DOCEXTRACTOR_REPOSITORY="https://github.com/monniert/docExtractor"
DOCEXTRACTOR_LICENSE="MIT"
DOCEXTRACTOR_MODEL_ID="docextractor-default-icfhr2020"
DOCEXTRACTOR_SOURCE_URL="https://codeload.github.com/monniert/docExtractor/zip/refs/heads/master"
DOCEXTRACTOR_MODEL_URL="https://imagine.enpc.fr/~monniert/docExtractor/resrc/models.zip"
DOCEXTRACTOR_GDRIVE_ID="13kHXW2vq30dJ10rGubDJBtrspZ_UyrkT"

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


def _download_from_sources(sources, target, *, artifact, variant):
    sources=tuple(sources or ())
    if not sources:
        raise RuntimeError(f"{variant} has no registered {artifact} download sources")
    failures=[]
    for attempt,source in enumerate(sources,1):
        if isinstance(source,ModelSource):
            site,url,reference=source.site,source.url,source.reference
        else:
            site,url,reference="unspecified",str(source),None
        ref=f" reference={reference}" if reference else ""
        print(f"Model download: variant={variant} artifact={artifact} attempt={attempt}/{len(sources)} site={site}{ref}")
        try:
            _download(url,target)
        except Exception as exc:
            detail=f"{type(exc).__name__}: {exc}"
            failures.append(f"{site}: {detail}")
            print(f"Model download failed: variant={variant} artifact={artifact} site={site} error={detail}")
            continue
        print(f"Model download succeeded: variant={variant} artifact={artifact} site={site}")
        return {"site":site,"url":url,"reference":reference,"attempt":attempt}
    raise RuntimeError(
        f"All {artifact} download sources failed for {variant}: " + "; ".join(failures)
    )

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
        # dhSegment v0.2 is CPU inference in HTH. Keep TensorFlow/absl legacy
        # loader chatter out of regression logs while preserving HTH diagnostics.
        "TF_CPP_MIN_LOG_LEVEL":"3",
        "ABSL_MIN_LOG_LEVEL":"3",
        "GLOG_minloglevel":"3",
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

def _prepare_kraken_page_mask_hook(*,results_root,policy,env_file):
    if policy not in {"reuse","refresh"}:
        raise ValueError(f"Unsupported lifecycle policy: {policy}")
    if importlib.util.find_spec("kraken") is None:
        raise RuntimeError(
            "kraken_page_mask requires Kraken; the regression/optimizer workflow "
            "must install the detector-specific runtime before PREPARE"
        )

    installed_version=importlib.metadata.version("kraken")
    if installed_version != KRAKEN_PACKAGE_VERSION:
        raise RuntimeError(
            f"kraken_page_mask requires Kraken {KRAKEN_PACKAGE_VERSION}; "
            f"found {installed_version}"
        )

    packaged_model=Path(str(importlib.resources.files("kraken").joinpath("blla.mlmodel")))
    if not packaged_model.is_file():
        raise RuntimeError(f"Kraken default BLLA model is missing: {packaged_model}")

    root=Path(results_root)/"models"/KRAKEN_MODEL_ID
    model=root/"blla.mlmodel"
    provenance=root/"model-provenance.json"
    complete=model.is_file() and provenance.is_file()

    if policy=="refresh" or not complete:
        root.mkdir(parents=True,exist_ok=True)
        shutil.copy2(packaged_model,model)
        payload={
            "schema_version":"1.0",
            "model_id":KRAKEN_MODEL_ID,
            "model_family":"Kraken BLLA",
            "variant":"bundled default baseline/region segmentation model",
            "kraken_version":installed_version,
            "upstream_repository":KRAKEN_REPOSITORY,
            "license":KRAKEN_LICENSE,
            "model_filename":"blla.mlmodel",
            "model_sha256":_sha256(model),
            "prepared_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "inference_backend":"kraken-7-task-api",
            "serving_contract":"PIL image -> Segmentation(regions, lines)",
            "device":"cpu",
        }
        provenance.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    payload=json.loads(provenance.read_text(encoding="utf-8"))
    if payload.get("model_sha256") != _sha256(model):
        raise RuntimeError("Kraken default BLLA model SHA mismatch")

    env={
        "HTH_KRAKEN_PAGE_MODEL":model.resolve().as_posix(),
        "HTH_KRAKEN_PAGE_PROVENANCE":provenance.resolve().as_posix(),
        "CUDA_VISIBLE_DEVICES":"-1",
    }
    _write_env(env_file,env); os.environ.update(env)
    print(
        f"Kraken Page-Mask ready: model={KRAKEN_MODEL_ID} "
        f"kraken={installed_version} model_sha256={str(payload.get('model_sha256') or '')[:12]}"
    )
    return payload


def _finalize_kraken_page_mask_hook(*,results_root):
    provenance=Path(results_root)/"models"/KRAKEN_MODEL_ID/"model-provenance.json"
    if not provenance.is_file():
        raise RuntimeError("Kraken Page-Mask model provenance missing")
    payload=json.loads(provenance.read_text(encoding="utf-8"))
    print(
        f"Detector lifecycle finalize: kraken_page_mask "
        f"model_sha256={str(payload.get('model_sha256') or '')[:12]}"
    )
    return payload



def _mask_rcnn_variant():
    requested=os.environ.get("HTH_MODEL_VARIANT", "default")
    variant=resolve_model_variant("mask_rcnn_page_mask", requested)
    if variant is None:
        raise RuntimeError("mask_rcnn_page_mask has no registered default model variant")
    return variant


def _prepare_mask_rcnn_page_mask_hook(*,results_root,policy,env_file):
    if policy not in {"reuse","refresh"}:
        raise ValueError(f"Unsupported lifecycle policy: {policy}")
    if importlib.util.find_spec("detectron2") is None:
        raise RuntimeError("mask_rcnn_page_mask requires Detectron2; the managed runtime must install the detector-specific runtime before PREPARE")
    variant=_mask_rcnn_variant()
    root=Path(results_root)/"models"/variant.model_id
    model=root/"model_final.pth"
    config=root/"config.yml"
    provenance=root/"model-provenance.json"
    complete=model.is_file() and config.is_file() and provenance.is_file()
    if policy=="refresh" or not complete:
        root.mkdir(parents=True,exist_ok=True)
        model_source=_download_from_sources(
            variant.model_sources,model,artifact="model",variant=variant.key
        )
        config_source=_download_from_sources(
            variant.config_sources,config,artifact="config",variant=variant.key
        )
        payload={
            "schema_version":"1.2","model_id":variant.model_id,"model_family":"Mask R-CNN",
            "model_variant":variant.key,"variant":variant.description,
            "upstream_repository":MASK_RCNN_REPOSITORY,"model_repository":variant.model_repository,
            "license":MASK_RCNN_LICENSE,
            "model_url":model_source["url"],"config_url":config_source["url"],
            "model_source_site":model_source["site"],"config_source_site":config_source["site"],
            "model_source_reference":model_source.get("reference"),"config_source_reference":config_source.get("reference"),
            "registered_model_sources":[{"site":x.site,"url":x.url,"reference":x.reference} for x in variant.model_sources],
            "registered_config_sources":[{"site":x.site,"url":x.url,"reference":x.reference} for x in variant.config_sources],
            "model_filename":model.name,"config_filename":config.name,"model_sha256":_sha256(model),"config_sha256":_sha256(config),
            "prepared_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "inference_backend":"detectron2-default-predictor","serving_contract":"BGR image -> Mask R-CNN instances",
            "training_domain":"HJDataset historical Japanese documents","device":"cpu",
        }
        provenance.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    payload=json.loads(provenance.read_text(encoding="utf-8"))
    if payload.get("model_sha256") != _sha256(model): raise RuntimeError("Mask R-CNN HJDataset model SHA mismatch")
    if payload.get("config_sha256") != _sha256(config): raise RuntimeError("Mask R-CNN HJDataset config SHA mismatch")
    recorded_variant=str(payload.get("model_variant") or variant.key)
    if recorded_variant != variant.key:
        raise RuntimeError(f"Mask R-CNN model variant provenance mismatch: expected {variant.key}, found {recorded_variant}")
    env={
        "HTH_MASK_RCNN_PAGE_MODEL":model.resolve().as_posix(),
        "HTH_MASK_RCNN_PAGE_CONFIG":config.resolve().as_posix(),
        "HTH_MASK_RCNN_PAGE_PROVENANCE":provenance.resolve().as_posix(),
        "HTH_ACTIVE_MODEL_DETECTOR":"mask_rcnn_page_mask",
        "HTH_ACTIVE_MODEL_VARIANT":variant.key,
        "HTH_ACTIVE_MODEL_ID":variant.model_id,
        "HTH_ACTIVE_MODEL_PROVENANCE":provenance.resolve().as_posix(),
        "HTH_MODEL_VARIANT_MASK_RCNN_PAGE_MASK":variant.key,
        "HTH_MODEL_ID_MASK_RCNN_PAGE_MASK":variant.model_id,
        "HTH_MODEL_PROVENANCE_MASK_RCNN_PAGE_MASK":provenance.resolve().as_posix(),
        "CUDA_VISIBLE_DEVICES":"-1",
    }
    _write_env(env_file,env); os.environ.update(env)
    print(f"Mask R-CNN Page-Mask ready: variant={variant.key} model={variant.model_id} model_sha256={str(payload.get('model_sha256') or '')[:12]}")
    return payload

def _finalize_mask_rcnn_page_mask_hook(*,results_root):
    provenance_raw=os.environ.get("HTH_ACTIVE_MODEL_PROVENANCE") or os.environ.get("HTH_MASK_RCNN_PAGE_PROVENANCE")
    if provenance_raw:
        provenance=Path(provenance_raw)
    else:
        variant=_mask_rcnn_variant()
        provenance=Path(results_root)/"models"/variant.model_id/"model-provenance.json"
    if not provenance.is_file(): raise RuntimeError("Mask R-CNN Page-Mask model provenance missing")
    payload=json.loads(provenance.read_text(encoding="utf-8"))
    print(f"Detector lifecycle finalize: mask_rcnn_page_mask variant={payload.get('model_variant','unknown')} model_sha256={str(payload.get('model_sha256') or '')[:12]}")
    return payload

def _prepare_doc_ufcn_page_mask_hook(*,results_root,policy,env_file):
    if policy not in {"reuse","refresh"}:
        raise ValueError(f"Unsupported lifecycle policy: {policy}")
    if importlib.util.find_spec("doc_ufcn") is None:
        raise RuntimeError("doc_ufcn_page_mask requires Doc-UFCN; the managed runtime must install the detector-specific runtime before PREPARE")
    installed_version=DOC_UFCN_PACKAGE_VERSION

    root=Path(results_root)/"models"/DOC_UFCN_MODEL_ID
    model=root/"model.pth"
    parameters=root/"parameters.yml"
    provenance=root/"model-provenance.json"
    complete=model.is_file() and parameters.is_file() and provenance.is_file()
    if policy=="refresh" or not complete:
        root.mkdir(parents=True,exist_ok=True)
        _download(DOC_UFCN_MODEL_URL,model)
        _download(DOC_UFCN_PARAMETERS_URL,parameters)
        payload={
            "schema_version":"1.0",
            "model_id":DOC_UFCN_MODEL_ID,
            "model_family":"Doc-UFCN",
            "variant":"Teklia generic historical page detection model",
            "doc_ufcn_version":installed_version,
            "upstream_repository":DOC_UFCN_REPOSITORY,
            "model_repository":DOC_UFCN_MODEL_REPOSITORY,
            "license":DOC_UFCN_LICENSE,
            "model_url":DOC_UFCN_MODEL_URL,
            "parameters_url":DOC_UFCN_PARAMETERS_URL,
            "model_filename":model.name,
            "model_sha256":_sha256(model),
            "parameters_sha256":_sha256(parameters),
            "model_release_version":"0.0.2",
            "classes":["background","page"],
            "input_size":768,
            "mean":[190,182,165],
            "std":[48,48,45],
            "upstream_min_cc":50,
            "prepared_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "inference_backend":"doc-ufcn-pytorch",
            "serving_contract":"RGB image -> class page polygons",
            "device":"cpu",
        }
        provenance.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    payload=json.loads(provenance.read_text(encoding="utf-8"))
    if payload.get("model_sha256") != _sha256(model):
        raise RuntimeError("Doc-UFCN generic page model SHA mismatch")
    if payload.get("parameters_sha256") != _sha256(parameters):
        raise RuntimeError("Doc-UFCN generic page parameters SHA mismatch")
    env={
        "HTH_DOC_UFCN_PAGE_MODEL":model.resolve().as_posix(),
        "HTH_DOC_UFCN_PAGE_PROVENANCE":provenance.resolve().as_posix(),
        "CUDA_VISIBLE_DEVICES":"-1",
    }
    _write_env(env_file,env); os.environ.update(env)
    print(f"Doc-UFCN Page-Mask ready: model={DOC_UFCN_MODEL_ID} doc-ufcn={installed_version} model_sha256={str(payload.get('model_sha256') or '')[:12]}")
    return payload

def _finalize_doc_ufcn_page_mask_hook(*,results_root):
    provenance=Path(results_root)/"models"/DOC_UFCN_MODEL_ID/"model-provenance.json"
    if not provenance.is_file():
        raise RuntimeError("Doc-UFCN Page-Mask model provenance missing")
    payload=json.loads(provenance.read_text(encoding="utf-8"))
    print(f"Detector lifecycle finalize: doc_ufcn_page_mask model_sha256={str(payload.get('model_sha256') or '')[:12]}")
    return payload

def _prepare_orli_page_mask_hook(*,results_root,policy,env_file):
    if policy not in {"reuse","refresh"}:
        raise ValueError(f"Unsupported lifecycle policy: {policy}")
    if importlib.util.find_spec("orli") is None:
        raise RuntimeError("orli_page_mask requires Orli; the managed runtime must install the detector-specific runtime before PREPARE")
    installed_version=importlib.metadata.version("orli")
    if installed_version != ORLI_PACKAGE_VERSION:
        raise RuntimeError(f"orli_page_mask requires Orli {ORLI_PACKAGE_VERSION}; found {installed_version}")

    root=Path(results_root)/"models"/ORLI_MODEL_ID
    model=root/"orli_base.safetensors"
    provenance=root/"model-provenance.json"
    complete=model.is_file() and provenance.is_file()
    if policy=="refresh" or not complete:
        root.mkdir(parents=True,exist_ok=True)
        _download(ORLI_MODEL_URL,model)
        payload={
            "schema_version":"1.0", "model_id":ORLI_MODEL_ID, "model_family":"Orli",
            "variant":"2026 high-resolution historical-document base model",
            "orli_version":installed_version, "upstream_repository":ORLI_REPOSITORY,
            "model_doi":ORLI_MODEL_DOI, "license":ORLI_LICENSE,
            "model_filename":model.name, "model_sha256":_sha256(model),
            "prepared_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "inference_backend":"orli.pred.segment", "serving_contract":"PIL image -> ordered baseline segmentation",
            "device":"cpu",
        }
        provenance.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    payload=json.loads(provenance.read_text(encoding="utf-8"))
    if payload.get("model_sha256") != _sha256(model):
        raise RuntimeError("Orli base model SHA mismatch")
    env={"HTH_ORLI_PAGE_MODEL":model.resolve().as_posix(), "HTH_ORLI_PAGE_PROVENANCE":provenance.resolve().as_posix(), "CUDA_VISIBLE_DEVICES":"-1"}
    _write_env(env_file,env); os.environ.update(env)
    print(f"Orli Page-Mask ready: model={ORLI_MODEL_ID} orli={installed_version} model_sha256={str(payload.get('model_sha256') or '')[:12]}")
    return payload

def _finalize_orli_page_mask_hook(*,results_root):
    provenance=Path(results_root)/"models"/ORLI_MODEL_ID/"model-provenance.json"
    if not provenance.is_file():
        raise RuntimeError("Orli Page-Mask model provenance missing")
    payload=json.loads(provenance.read_text(encoding="utf-8"))
    print(f"Detector lifecycle finalize: orli_page_mask model_sha256={str(payload.get('model_sha256') or '')[:12]}")
    return payload



def _prepare_pagenet_page_mask_hook(*,results_root,policy,env_file):
    payload=prepare_detector_legacy("learned_page_mask",results_root=results_root,policy=policy,github_env=env_file)
    print("PageNet Page-Mask ready: explicit detector identity reusing pagenet-ohio assets")
    return payload

def _finalize_pagenet_page_mask_hook(*,results_root):
    return finalize_detector_legacy("learned_page_mask",results_root=results_root)

def _eynollah_sources(relative):
    sources=[
        ModelSource(
            site="Hugging Face / SBB",
            url=f"https://huggingface.co/SBB/eynollah-page-extraction/resolve/{ref}/{relative}?download=true",
            reference=ref,
        )
        for ref in EYNOLLAH_HF_REFS
    ]
    sources.append(
        ModelSource(
            site="Hugging Face / SBB legacy layout",
            url=(
                "https://huggingface.co/SBB/eynollah-page-extraction/resolve/"
                f"{EYNOLLAH_HF_LEGACY_REF}/saved_model/2021-04-25/{relative}?download=true"
            ),
            reference=EYNOLLAH_HF_LEGACY_REF,
        )
    )
    return tuple(sources)

def _prepare_eynollah_page_mask_hook(*,results_root,policy,env_file):
    if policy not in {"reuse","refresh"}: raise ValueError(f"Unsupported lifecycle policy: {policy}")
    if importlib.util.find_spec("tensorflow") is None: raise RuntimeError("eynollah_page_mask requires the managed TensorFlow runtime")
    root=Path(results_root)/"models"/EYNOLLAH_MODEL_ID; model_dir=root/"saved_model"; provenance=root/"model-provenance.json"
    files=("saved_model.pb","keras_metadata.pb","variables/variables.index","variables/variables.data-00000-of-00001")
    complete=provenance.is_file() and all((model_dir/f).is_file() for f in files)
    used={}
    if policy=="refresh" or not complete:
        if model_dir.exists(): shutil.rmtree(model_dir)
        for rel in files:
            target=model_dir/rel
            used[rel]=_download_from_sources(_eynollah_sources(rel),target,artifact=rel,variant="eynollah_page_mask")
        saved_model_sha256=_sha256(model_dir/"saved_model.pb")
        if saved_model_sha256 != EYNOLLAH_SAVED_MODEL_SHA256:
            raise RuntimeError(
                f"Eynollah saved_model.pb SHA mismatch: expected {EYNOLLAH_SAVED_MODEL_SHA256}, found {saved_model_sha256}"
            )
        payload={"schema_version":"1.0","model_id":EYNOLLAH_MODEL_ID,"model_family":"Eynollah","variant":"page-extraction/2021-04-25","upstream_repository":EYNOLLAH_REPOSITORY,"model_repository":EYNOLLAH_HF_REPOSITORY,"license":EYNOLLAH_LICENSE,"expected_saved_model_sha256":EYNOLLAH_SAVED_MODEL_SHA256,"prepared_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"files":{rel:{"sha256":_sha256(model_dir/rel),"source":used[rel]} for rel in files},"inference_backend":"tensorflow-savedmodel-cpu"}
        provenance.parent.mkdir(parents=True,exist_ok=True); provenance.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    payload=json.loads(provenance.read_text(encoding="utf-8"))
    for rel,meta in payload.get("files",{}).items():
        if _sha256(model_dir/rel)!=meta.get("sha256"): raise RuntimeError(f"Eynollah model SHA mismatch: {rel}")
    if _sha256(model_dir/"saved_model.pb") != EYNOLLAH_SAVED_MODEL_SHA256:
        raise RuntimeError("Eynollah released saved_model.pb does not match the published model-card SHA-256")
    env={"HTH_EYNOLLAH_PAGE_MODEL_DIR":model_dir.resolve().as_posix(),"HTH_EYNOLLAH_PAGE_PROVENANCE":provenance.resolve().as_posix(),"CUDA_VISIBLE_DEVICES":"-1"}; _write_env(env_file,env); os.environ.update(env)
    print(f"Eynollah Page-Mask ready: model={EYNOLLAH_MODEL_ID} saved_model_sha256={payload['files']['saved_model.pb']['sha256'][:12]}")
    return payload

def _finalize_eynollah_page_mask_hook(*,results_root):
    p=Path(results_root)/"models"/EYNOLLAH_MODEL_ID/"model-provenance.json"
    if not p.is_file(): raise RuntimeError("Eynollah Page-Mask model provenance missing")
    return json.loads(p.read_text(encoding="utf-8"))

def _prepare_docextractor_page_mask_hook(*,results_root,policy,env_file):
    if policy not in {"reuse","refresh"}: raise ValueError(f"Unsupported lifecycle policy: {policy}")
    try: import torch  # noqa: F401
    except Exception as exc: raise RuntimeError("docextractor_page_mask requires the managed PyTorch runtime") from exc
    root=Path(results_root)/"models"/DOCEXTRACTOR_MODEL_ID; source_archive=root/"source.zip"; source_root=root/"source"; model_archive=root/"models.zip"; provenance=root/"model-provenance.json"
    model_path=next(iter(root.glob("models/default/model.pkl")),None)
    complete=provenance.is_file() and source_root.is_dir() and model_path is not None and model_path.is_file()
    if policy=="refresh" or not complete:
        root.mkdir(parents=True,exist_ok=True)
        _download(DOCEXTRACTOR_SOURCE_URL,source_archive)
        if source_root.exists(): shutil.rmtree(source_root)
        _safe_extract_zip(source_archive,source_root)
        try:
            _download(DOCEXTRACTOR_MODEL_URL,model_archive)
            model_source={"site":"ENPC / docExtractor","url":DOCEXTRACTOR_MODEL_URL}
        except Exception as first:
            print(f"Model download failed: variant=docextractor_page_mask artifact=models.zip site=ENPC / docExtractor error={type(first).__name__}: {first}")
            try:
                import gdown
                print(f"Model download: variant=docextractor_page_mask artifact=models.zip attempt=2/2 site=Google Drive / docExtractor reference={DOCEXTRACTOR_GDRIVE_ID}")
                result=gdown.download(id=DOCEXTRACTOR_GDRIVE_ID,output=str(model_archive),quiet=False)
                if not result or not model_archive.is_file(): raise RuntimeError("gdown did not produce models.zip")
                model_source={"site":"Google Drive / docExtractor","reference":DOCEXTRACTOR_GDRIVE_ID}
            except Exception as second:
                raise RuntimeError(f"All docExtractor model download sources failed: ENPC: {first}; Google Drive: {second}") from second
        for child in list(root.glob("models")):
            if child.is_dir(): shutil.rmtree(child)
        _safe_extract_zip(model_archive,root)
        candidates=sorted(root.rglob("model.pkl"))
        if not candidates: raise RuntimeError("docExtractor models.zip contains no model.pkl")
        model_path=candidates[0]
        extracted_dirs=sorted(p for p in source_root.iterdir() if p.is_dir())
        if not extracted_dirs: raise RuntimeError("docExtractor source archive is empty")
        repo_dir=extracted_dirs[0]
        payload={"schema_version":"1.0","model_id":DOCEXTRACTOR_MODEL_ID,"model_family":"docExtractor ResUNet","upstream_repository":DOCEXTRACTOR_REPOSITORY,"license":DOCEXTRACTOR_LICENSE,"source_archive_sha256":_sha256(source_archive),"model_archive_sha256":_sha256(model_archive),"model_sha256":_sha256(model_path),"model_relative_path":model_path.relative_to(root).as_posix(),"source_relative_path":repo_dir.relative_to(root).as_posix(),"model_source":model_source,"prepared_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"inference_backend":"pytorch-cpu"}
        provenance.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    payload=json.loads(provenance.read_text(encoding="utf-8")); model_path=root/payload["model_relative_path"]; repo_dir=root/payload["source_relative_path"]
    if _sha256(model_path)!=payload.get("model_sha256"): raise RuntimeError("docExtractor model SHA mismatch")
    env={"HTH_DOCEXTRACTOR_PAGE_MODEL":model_path.resolve().as_posix(),"HTH_DOCEXTRACTOR_PAGE_SOURCE":repo_dir.resolve().as_posix(),"HTH_DOCEXTRACTOR_PAGE_PROVENANCE":provenance.resolve().as_posix(),"CUDA_VISIBLE_DEVICES":"-1"}; _write_env(env_file,env); os.environ.update(env)
    print(f"docExtractor Page-Mask ready: model={DOCEXTRACTOR_MODEL_ID} model_sha256={payload['model_sha256'][:12]}")
    return payload

def _finalize_docextractor_page_mask_hook(*,results_root):
    p=Path(results_root)/"models"/DOCEXTRACTOR_MODEL_ID/"model-provenance.json"
    if not p.is_file(): raise RuntimeError("docExtractor Page-Mask model provenance missing")
    return json.loads(p.read_text(encoding="utf-8"))

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
    "doc_ufcn_page_mask":_prepare_doc_ufcn_page_mask_hook,
    "mask_rcnn_page_mask":_prepare_mask_rcnn_page_mask_hook,
    "kraken_page_mask":_prepare_kraken_page_mask_hook,
    "orli_page_mask":_prepare_orli_page_mask_hook,
    "learned_page_mask":_prepare_learned_page_mask_hook,
    "pagenet_page_mask":_prepare_pagenet_page_mask_hook,
    "eynollah_page_mask":_prepare_eynollah_page_mask_hook,
    "docextractor_page_mask":_prepare_docextractor_page_mask_hook,
}
_FINALIZE_HOOKS={
    "dhsegment_page_mask":_finalize_dhsegment_page_mask_hook,
    "doc_ufcn_page_mask":_finalize_doc_ufcn_page_mask_hook,
    "mask_rcnn_page_mask":_finalize_mask_rcnn_page_mask_hook,
    "kraken_page_mask":_finalize_kraken_page_mask_hook,
    "orli_page_mask":_finalize_orli_page_mask_hook,
    "learned_page_mask":_finalize_learned_page_mask_hook,
    "pagenet_page_mask":_finalize_pagenet_page_mask_hook,
    "eynollah_page_mask":_finalize_eynollah_page_mask_hook,
    "docextractor_page_mask":_finalize_docextractor_page_mask_hook,
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
    print(f"Detector lifecycle prepare detector={detector} hook={hook_name} policy={policy}")
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
    print(f"Detector lifecycle finalize detector={detector} hook={hook_name}")
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
