from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hth.geometry import detector_dhsegment_page_mask, detector_kraken_page_mask
from hth.regression.runner import load_pages


EXPORTERS = {
    "kraken_page_mask": detector_kraken_page_mask.export_precomputed_golden_set_evidence,
    "dhsegment_page_mask": detector_dhsegment_page_mask.export_precomputed_golden_set_evidence,
}


def _progress(detector: str):
    def report(event: str, index: int, total: int, image_key: str, elapsed: float) -> None:
        if event == "start":
            print(
                f"[learned-evidence][{detector}] page {index}/{total} START "
                f"key={image_key[:12]}",
                flush=True,
            )
        else:
            print(
                f"[learned-evidence][{detector}] page {index}/{total} READY "
                f"key={image_key[:12]} elapsed={elapsed:.2f}s",
                flush=True,
            )
    return report


def prepare(
    *,
    detector: str,
    golden_set: Path,
    image_root: Path,
    maximum_dimension: int,
    output: Path,
) -> Path:
    exporter = EXPORTERS.get(detector)
    if exporter is None:
        raise ValueError(f"Detector does not support shared learned evidence: {detector}")

    started = time.perf_counter()
    print(f"[learned-evidence][{detector}] preparing shared Golden Set evidence", flush=True)
    pages = load_pages(golden_set, image_root, maximum_dimension)
    print(
        f"[learned-evidence][{detector}] loaded {len(pages)} Golden Set page(s); "
        "model inference will run once before pipeline fan-out",
        flush=True,
    )
    manifest = exporter(
        [page["image"] for page in pages],
        output,
        progress=_progress(detector),
    )
    elapsed = time.perf_counter() - started
    print(
        f"[learned-evidence][{detector}] SHARED EVIDENCE READY "
        f"pages={len(pages)} elapsed={elapsed:.2f}s path={manifest}",
        flush=True,
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--detector", choices=sorted(EXPORTERS), required=True)
    prep.add_argument("--golden-set", type=Path, required=True)
    prep.add_argument("--image-root", type=Path, required=True)
    prep.add_argument("--max-dimension", type=int, required=True)
    prep.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(
            detector=args.detector,
            golden_set=args.golden_set,
            image_root=args.image_root,
            maximum_dimension=args.max_dimension,
            output=args.output,
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
