#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hth.parameter_liveness import audit_detector_directory

p=argparse.ArgumentParser(description="Audit HTH detector parameter-liveness metadata without guessing behavioral death.")
p.add_argument("--detector-dir", type=Path, default=Path("config/detectors"))
p.add_argument("--json", action="store_true")
args=p.parse_args()
report=audit_detector_directory(args.detector_dir)
if args.json:
    print(json.dumps(report, indent=2, sort_keys=True))
else:
    print(f"Detectors audited : {report['detector_count']}")
    print(f"Zombie-bearing    : {report['zombie_detector_count']}")
    print(f"Metadata errors   : {report['error_count']}")
    for item in report['detectors']:
        if item['zombie_parameters'] or item['errors']:
            print(f"{item['detector']}: zombies={','.join(item['zombie_parameters']) or 'none'} errors={'; '.join(item['errors']) or 'none'}")
raise SystemExit(1 if report['error_count'] else 0)
