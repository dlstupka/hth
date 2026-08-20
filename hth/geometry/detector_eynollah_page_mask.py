from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import hashlib, json, os, threading, time
import cv2, numpy as np
from .detector_evidence_mask_common import BASELINE_PARAMETERS, parameters as _common_parameters, candidate as _candidate, debug_images as _debug
METHOD='eynollah_page_mask'; MODEL_DIR_ENV='HTH_EYNOLLAH_PAGE_MODEL_DIR'; PROVENANCE_ENV='HTH_EYNOLLAH_PAGE_PROVENANCE'
_MODEL=None; _MODEL_KEY=None; _MODEL_LOCK=threading.Lock(); _INFERENCE_LOCK=threading.Lock(); _CACHE=OrderedDict(); _CACHE_LOCK=threading.Lock(); _CACHE_LIMIT=16

def _parameters(p): return _common_parameters(p,label='Eynollah Page-Mask')
def _asset_dir():
    raw=os.environ.get(MODEL_DIR_ENV,'').strip(); p=Path(raw)
    if not raw or not (p/'saved_model.pb').is_file(): raise RuntimeError(f'{METHOD} lifecycle did not provide a valid SavedModel')
    return p
def _provenance():
    p=Path(os.environ.get(PROVENANCE_ENV,'').strip())
    if not p.is_file(): raise RuntimeError(f'{METHOD} provenance missing')
    return json.loads(p.read_text())

def _load_model():
    global _MODEL,_MODEL_KEY
    path=_asset_dir().resolve(); key=str(path)
    if _MODEL is not None and _MODEL_KEY==key: return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_KEY==key: return _MODEL
        os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1'); os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
        import tensorflow as tf
        loaded=tf.saved_model.load(str(path)); signatures=dict(getattr(loaded,'signatures',{}) or {})
        if not signatures: raise RuntimeError('Eynollah SavedModel exposes no signatures')
        fn=signatures.get('serving_default') or signatures[sorted(signatures)[0]]
        _MODEL=(tf,fn); _MODEL_KEY=key
        print(f'Eynollah Page-Mask ready for inference: model={path.name} backend=tensorflow-savedmodel device=cpu',flush=True)
        return _MODEL

def _image_key(image):
    arr=np.ascontiguousarray(image); h=hashlib.blake2b(digest_size=16); h.update(str(arr.shape).encode()); h.update(memoryview(arr)); return h.hexdigest()

def _probability_from_output(raw):
    if isinstance(raw,dict): raw=raw[sorted(raw)[0]]
    arr=np.asarray(raw,dtype=np.float32)
    while arr.ndim>3 and arr.shape[0]==1: arr=arr[0]
    if arr.ndim==3:
        if arr.shape[-1]>1:
            # If logits rather than probabilities, softmax them first.
            if float(arr.min())<0 or float(arr.max())>1:
                z=arr-arr.max(axis=-1,keepdims=True); e=np.exp(z); arr=e/np.maximum(e.sum(axis=-1,keepdims=True),1e-8)
            arr=arr[...,1]
        elif arr.shape[-1]==1: arr=arr[...,0]
        elif arr.shape[0]>1: arr=arr[1]
    if arr.ndim!=2: raise RuntimeError(f'Eynollah output has unexpected shape {tuple(arr.shape)}')
    if float(arr.max())>1 or float(arr.min())<0: arr=1/(1+np.exp(-arr))
    return np.clip(arr,0,1)

def _infer(image):
    key=_image_key(image)
    with _CACHE_LOCK:
        if key in _CACHE: _CACHE.move_to_end(key); return _CACHE[key]
    with _INFERENCE_LOCK:
        with _CACHE_LOCK:
            if key in _CACHE: _CACHE.move_to_end(key); return _CACHE[key]
        tf,fn=_load_model(); spec=next(iter(fn.structured_input_signature[1].values()))
        shape=spec.shape.as_list(); ih=shape[1] if len(shape)>2 and shape[1] else min(1024,max(256,image.shape[0])); iw=shape[2] if len(shape)>2 and shape[2] else min(1024,max(256,image.shape[1]))
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB); resized=cv2.resize(rgb,(int(iw),int(ih)),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        inp=tf.convert_to_tensor(resized[None,...],dtype=spec.dtype)
        name=next(iter(fn.structured_input_signature[1]))
        prob=_probability_from_output(fn(**{name:inp})); prob=np.array(prob,copy=True); prob.setflags(write=False)
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
    payload={'schema_version':'0.1','detector':METHOD,'representation':'eynollah-page-probability','records':records}; (output_dir/'manifest.json').write_text(json.dumps(payload,sort_keys=True))
    return output_dir/'manifest.json'
def load_precomputed_golden_set_evidence(output_dir,images):
    output_dir=Path(output_dir); payload=json.loads((output_dir/'manifest.json').read_text()); records={r['image_key']:r['file'] for r in payload['records']}; expected=tuple(_image_key(i) for i in images)
    if payload.get('detector')!=METHOD or any(k not in records for k in expected): raise ValueError('Shared Eynollah evidence mismatch')
    with _CACHE_LOCK:
        for key in expected:
            arr=np.load(output_dir/records[key]); arr.setflags(write=False); _CACHE[key]=arr
    return expected
def detect(*,image_bgr,mask,parameters=None):
    del mask; v=_parameters(parameters); prov=_provenance(); return _candidate(METHOD,image_bgr,_infer(image_bgr),v,{'model_id':prov.get('model_id'),'model_family':'Eynollah','model_source':prov.get('model_repository'),'evidence':'eynollah_page_extraction_probability'})
def debug_images(*,image_bgr,mask,parameters=None,candidate_corners=None,verbose=False):
    del mask,verbose; return _debug(image_bgr,_infer(image_bgr),_parameters(parameters),candidate_corners,'eynollah-page')
