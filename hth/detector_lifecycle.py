# detector lifecycle
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, tempfile, urllib.request
from datetime import datetime, timezone
from pathlib import Path

PAGENET_REPOSITORY="https://github.com/ctensmeyer/pagenet"
PAGENET_LICENSE="BSD-3-Clause"
PAGENET_MODEL_ID="pagenet-ohio"
PAGENET_PROTOTXT_URL="https://raw.githubusercontent.com/ctensmeyer/pagenet/master/models/ohio_train_val.prototxt"
PAGENET_WEIGHTS_URL="https://raw.githubusercontent.com/ctensmeyer/pagenet/master/models/ohio_weights.caffemodel"

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

def _write_env(path,values):
    if path is None: return
    with Path(path).open("a",encoding="utf-8") as h:
        for k,v in values.items(): h.write(f"{k}={v}\n")

def prepare_detector(detector,*,results_root,policy="reuse",github_env=None):
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
    env={"HTH_LEARNED_PAGE_MASK_PROTOTXT":str(deploy),"HTH_LEARNED_PAGE_MASK_WEIGHTS":str(weights),"HTH_LEARNED_PAGE_MASK_PROVENANCE":str(provenance)}
    _write_env(github_env,env); os.environ.update(env)
    print(f"Learned Page-Mask ready: model={PAGENET_MODEL_ID} weights_sha256={payload['weights_sha256'][:12]}")
    return payload

def finalize_detector(detector,*,results_root):
    detector=detector.strip().lower()
    if detector!="learned_page_mask":
        print(f"Detector lifecycle finalize: {detector} has no post-exec hook")
        return {"detector":detector,"finalized":False}
    p=Path(results_root)/"models"/PAGENET_MODEL_ID/"model-provenance.json"
    if not p.is_file(): raise RuntimeError("Learned Page-Mask model provenance missing")
    payload=json.loads(p.read_text(encoding="utf-8"))
    print(f"Detector lifecycle finalize: {detector} weights_sha256={str(payload.get('weights_sha256') or '')[:12]}")
    return payload

def main(argv=None):
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    a=sub.add_parser("prepare"); a.add_argument("--detector",required=True); a.add_argument("--results-root",type=Path,required=True); a.add_argument("--policy",choices=("reuse","refresh"),default="reuse"); a.add_argument("--github-env",type=Path)
    b=sub.add_parser("finalize"); b.add_argument("--detector",required=True); b.add_argument("--results-root",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=="prepare": prepare_detector(args.detector,results_root=args.results_root,policy=args.policy,github_env=args.github_env)
    else: finalize_detector(args.detector,results_root=args.results_root)
    return 0
if __name__=="__main__": raise SystemExit(main())
