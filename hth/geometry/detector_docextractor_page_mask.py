from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import hashlib, json, os, sys, threading, time
import cv2, numpy as np
from .detector_evidence_mask_common import BASELINE_PARAMETERS, parameters as _common_parameters, candidate as _candidate, debug_images as _debug
METHOD='docextractor_page_mask'; MODEL_ENV='HTH_DOCEXTRACTOR_PAGE_MODEL'; SOURCE_ENV='HTH_DOCEXTRACTOR_PAGE_SOURCE'; PROVENANCE_ENV='HTH_DOCEXTRACTOR_PAGE_PROVENANCE'
_MODEL=None; _MODEL_KEY=None; _MODEL_LOCK=threading.Lock(); _INFERENCE_LOCK=threading.Lock(); _CACHE=OrderedDict(); _CACHE_LOCK=threading.Lock(); _CACHE_LIMIT=16

def _parameters(p): return _common_parameters(p,label='docExtractor Page-Mask')
def _asset(name):
    p=Path(os.environ.get(name,'').strip())
    if not p.exists(): raise RuntimeError(f'{METHOD} lifecycle did not set valid {name}')
    return p
def _provenance(): return json.loads(_asset(PROVENANCE_ENV).read_text())
def _load_model():
    global _MODEL,_MODEL_KEY
    model_path=_asset(MODEL_ENV).resolve(); source_root=_asset(SOURCE_ENV).resolve(); key=f'{model_path}|{source_root}'
    if _MODEL is not None and _MODEL_KEY==key: return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_KEY==key: return _MODEL
        src=source_root/'src'
        if not src.is_dir():
            matches=list(source_root.rglob('src/models'))
            if not matches: raise RuntimeError('docExtractor source tree has no src/models')
            src=matches[0].parent
        sys.path.insert(0,str(src)) if str(src) not in sys.path else None
        import torch
        from models import load_model_from_path
        original_load=torch.load
        def trusted_load(*a,**kw):
            kw.setdefault('weights_only',False); return original_load(*a,**kw)
        torch.load=trusted_load
        try: model,attrs=load_model_from_path(model_path,device=torch.device('cpu'),attributes_to_return=['train_resolution','restricted_labels','normalize'])
        finally: torch.load=original_load
        _MODEL=(torch,model,attrs); _MODEL_KEY=key
        print(f'docExtractor Page-Mask ready for inference: model={model_path.name} backend=pytorch device=cpu train_resolution={attrs[0]}',flush=True)
        return _MODEL

def _image_key(image):
    arr=np.ascontiguousarray(image); h=hashlib.blake2b(digest_size=16); h.update(str(arr.shape).encode()); h.update(memoryview(arr)); return h.hexdigest()
def _infer(image):
    key=_image_key(image)
    with _CACHE_LOCK:
        if key in _CACHE: _CACHE.move_to_end(key); return _CACHE[key]
    with _INFERENCE_LOCK:
        with _CACHE_LOCK:
            if key in _CACHE: _CACHE.move_to_end(key); return _CACHE[key]
        torch,model,attrs=_load_model(); train_resolution,restricted_labels,normalize=attrs
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        height,width=rgb.shape[:2]
        if isinstance(train_resolution,(int,float)):
            ratio=float(np.sqrt(float(train_resolution)/max(1.0,float(width*height))))
            out_width=max(1,int(round(ratio*width)))
            out_height=max(1,int(round(ratio*height)))
        else:
            target_width,target_height=(int(train_resolution[0]),int(train_resolution[1]))
            ratio=min(target_width/float(width),target_height/float(height))
            out_width=max(1,int(round(ratio*width)))
            out_height=max(1,int(round(ratio*height)))
        resized=cv2.resize(rgb,(out_width,out_height),interpolation=cv2.INTER_AREA)
        inp=np.asarray(resized,dtype=np.float32)/255.0
        if normalize:
            inp=(inp-inp.mean(axis=(0,1)))/(inp.std(axis=(0,1))+1e-7)
        tensor=torch.from_numpy(inp.transpose(2,0,1)).float().unsqueeze(0)
        with torch.no_grad():
            logits=model(tensor)[0]
            probs=torch.softmax(logits,dim=0)
            # Upstream reserves output channel 0 for background and maps each
            # restricted document label to channels 1..N.
            fg=1.0-probs[0]
        prob=fg.cpu().numpy().astype(np.float32); prob=np.array(prob,copy=True); prob.setflags(write=False)
        with _CACHE_LOCK:
            _CACHE[key]=prob; _CACHE.move_to_end(key)
            while len(_CACHE)>_CACHE_LIMIT: _CACHE.popitem(last=False)
        return prob

def precompute_golden_set_evidence(images,*,progress=None):
    keys=[]; total=len(images)
    for i,img in enumerate(images,1):
        key=_image_key(img); start=time.perf_counter(); progress and progress('start',i,total,key,0.0); _infer(img); progress and progress('finish',i,total,key,time.perf_counter()-start); keys.append(key)
    return tuple(keys)
def export_precomputed_golden_set_evidence(images,output_dir,*,progress=None):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); keys=precompute_golden_set_evidence(images,progress=progress); records=[]
    with _CACHE_LOCK:
        for key in keys:
            name=f'{key}.npy'; np.save(output_dir/name,np.asarray(_CACHE[key])); records.append({'image_key':key,'file':name})
    (output_dir/'manifest.json').write_text(json.dumps({'schema_version':'0.1','detector':METHOD,'representation':'docextractor-foreground-probability','records':records},sort_keys=True)); return output_dir/'manifest.json'
def load_precomputed_golden_set_evidence(output_dir,images):
    output_dir=Path(output_dir); payload=json.loads((output_dir/'manifest.json').read_text()); records={r['image_key']:r['file'] for r in payload['records']}; expected=tuple(_image_key(i) for i in images)
    if payload.get('detector')!=METHOD or any(k not in records for k in expected): raise ValueError('Shared docExtractor evidence mismatch')
    with _CACHE_LOCK:
        for key in expected:
            arr=np.load(output_dir/records[key]); arr.setflags(write=False); _CACHE[key]=arr
    return expected
def detect(*,image_bgr,mask,parameters=None):
    del mask; v=_parameters(parameters); prov=_provenance(); return _candidate(METHOD,image_bgr,_infer(image_bgr),v,{'model_id':prov.get('model_id'),'model_family':'docExtractor ResUNet','model_source':prov.get('upstream_repository'),'evidence':'docextractor_nonbackground_probability_envelope'})
def debug_images(*,image_bgr,mask,parameters=None,candidate_corners=None,verbose=False):
    del mask,verbose; return _debug(image_bgr,_infer(image_bgr),_parameters(parameters),candidate_corners,'docextractor-page')
