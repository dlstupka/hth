#!/usr/bin/env python3
"""Run one approved detector calibration over a complete preprocessed collection."""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
from detector_lifecycle import prepare_detector, finalize_detector


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--image-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--detector',default='amsre_doc_ufcn_fusion'); p.add_argument('--catalog',type=Path,default=Path('config/document-detectors.json'))
    p.add_argument('--lifecycle-root',type=Path,required=True); p.add_argument('--overwrite',action='store_true')
    a=p.parse_args(); catalog=json.loads(a.catalog.read_text()); entry=(catalog.get('detectors') or {}).get(a.detector)
    if not entry: raise SystemExit(f"No approved document calibration for detector {a.detector!r}")
    # Gen3 depends on the managed Doc-UFCN lifecycle; other learned detectors can add their own dependency here.
    lifecycle='doc_ufcn_page_mask' if a.detector=='amsre_doc_ufcn_fusion' else a.detector
    prepare_detector(lifecycle,results_root=a.lifecycle_root,policy='reuse')
    try:
        with tempfile.TemporaryDirectory() as td:
            params=Path(td)/'parameters.json'; params.write_text(json.dumps(entry['parameters'],indent=2)+'\n')
            cmd=[sys.executable,str(Path(__file__).with_name('detect_geometry_candidates.py')),'--manifest',str(a.manifest),'--analysis',str(a.analysis),'--image-root',str(a.image_root),'--output',str(a.output),'--detector',a.detector,'--parameters-json',str(params),'--overwrite']
            subprocess.run(cmd,check=True)
        payload=json.loads(a.output.read_text())
        payload['document_detector']={'detector':a.detector,'display_name':entry.get('display_name'),'golden_set_id':entry.get('golden_set_id'),'parameter_set_id':entry.get('parameter_set_id'),'parameters':entry['parameters']}
        a.output.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    finally: finalize_detector(lifecycle,results_root=a.lifecycle_root)
    return 0
if __name__=='__main__': raise SystemExit(main())
