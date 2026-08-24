from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from hth.geometry import detector_dhsegment_page_mask, detector_eynollah_page_mask, detector_docextractor_page_mask, detector_pagenet_page_mask, detector_doc_ufcn_page_mask, detector_mask_rcnn_page_mask, detector_kraken_page_mask, detector_orli_page_mask
from hth.regression.runner import load_pages


EXPORTERS = {
    "kraken_page_mask": detector_kraken_page_mask.export_precomputed_golden_set_evidence,
    "doc_ufcn_page_mask": detector_doc_ufcn_page_mask.export_precomputed_golden_set_evidence,
    "mask_rcnn_page_mask": detector_mask_rcnn_page_mask.export_precomputed_golden_set_evidence,
    "orli_page_mask": detector_orli_page_mask.export_precomputed_golden_set_evidence,
    "dhsegment_page_mask": detector_dhsegment_page_mask.export_precomputed_golden_set_evidence,
    "eynollah_page_mask": detector_eynollah_page_mask.export_precomputed_golden_set_evidence,
    "docextractor_page_mask": detector_docextractor_page_mask.export_precomputed_golden_set_evidence,
    "pagenet_page_mask": detector_pagenet_page_mask.export_precomputed_golden_set_evidence,
}

ORLI_EVIDENCE_INDEX = Path("indexes") / "orli-evidence-index.json"
ORLI_EVIDENCE_ROOT = Path("learned-evidence") / "orli_page_mask"
ORLI_PERSISTENCE_SCHEMA_VERSION = "1.0"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _orli_identity(*, golden_set: Path, maximum_dimension: int, images) -> dict:
    provenance_raw = os.environ.get("HTH_ORLI_PAGE_PROVENANCE", "")
    if not provenance_raw:
        raise RuntimeError("HTH_ORLI_PAGE_PROVENANCE is required for persistent Orli evidence")
    provenance_path = Path(provenance_raw)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    model_sha256 = str(provenance.get("model_sha256") or "")
    if not model_sha256:
        raise RuntimeError("Orli model provenance does not contain model_sha256")
    image_keys = [detector_orli_page_mask._image_key(image) for image in images]
    return {
        "schema_version": ORLI_PERSISTENCE_SCHEMA_VERSION,
        "detector": "orli_page_mask",
        "model_id": provenance.get("model_id"),
        "model_sha256": model_sha256,
        "orli_version": provenance.get("orli_version"),
        "inference_backend": provenance.get("inference_backend"),
        "serving_contract": provenance.get("serving_contract"),
        "golden_set_sha256": _sha256_file(golden_set),
        "maximum_dimension": int(maximum_dimension),
        "image_keys": image_keys,
        "evidence_representation": "immutable-json",
    }


def _orli_evidence_id(identity: dict) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _orli_artifact_path(results_root: Path, evidence_id: str) -> Path:
    return Path(results_root) / ORLI_EVIDENCE_ROOT / evidence_id / "manifest.json"


def _orli_index_path(results_root: Path) -> Path:
    return Path(results_root) / ORLI_EVIDENCE_INDEX


def _orli_manifest_matches(path: Path, *, evidence_id: str, identity: dict) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    persistence = payload.get("persistence") or {}
    return (
        payload.get("detector") == "orli_page_mask"
        and persistence.get("evidence_id") == evidence_id
        and persistence.get("identity") == identity
    )


def rebuild_orli_index(*, results_root: Path) -> Path:
    results_root = Path(results_root)
    entries = []
    evidence_root = results_root / ORLI_EVIDENCE_ROOT
    if evidence_root.is_dir():
        for manifest in sorted(evidence_root.glob("*/manifest.json")):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            persistence = payload.get("persistence") or {}
            identity = persistence.get("identity")
            evidence_id = str(persistence.get("evidence_id") or "")
            if not evidence_id or not isinstance(identity, dict):
                continue
            relative = manifest.relative_to(results_root).as_posix()
            entries.append({
                "evidence_id": evidence_id,
                "path": relative,
                "manifest_sha256": _sha256_file(manifest),
                "size_bytes": manifest.stat().st_size,
                "created_at_utc": persistence.get("created_at_utc"),
                "model_id": identity.get("model_id"),
                "model_sha256": identity.get("model_sha256"),
                "orli_version": identity.get("orli_version"),
                "golden_set_sha256": identity.get("golden_set_sha256"),
                "maximum_dimension": identity.get("maximum_dimension"),
                "page_count": len(identity.get("image_keys") or []),
                "image_keys": list(identity.get("image_keys") or []),
            })
    updated_at = max(
        (str(entry.get("created_at_utc") or "") for entry in entries),
        default="",
    ) or None
    payload = {
        "schema_version": ORLI_PERSISTENCE_SCHEMA_VERSION,
        "detector": "orli_page_mask",
        "updated_at_utc": updated_at,
        "entry_count": len(entries),
        "entries": entries,
    }
    target = _orli_index_path(results_root)
    _write_json_atomic(target, payload)
    return target


def _reuse_persistent_orli_evidence(*, output: Path, results_root: Path, identity: dict) -> Path | None:
    evidence_id = _orli_evidence_id(identity)
    artifact = _orli_artifact_path(results_root, evidence_id)
    if not _orli_manifest_matches(artifact, evidence_id=evidence_id, identity=identity):
        return None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "manifest.json"
    shutil.copy2(artifact, target)
    print(
        f"[learned-evidence][orli_page_mask] PERSISTENT CACHE HIT "
        f"evidence_id={evidence_id[:12]} pages={len(identity['image_keys'])} "
        f"bytes={artifact.stat().st_size} path={artifact}",
        flush=True,
    )
    return target


def _persist_orli_evidence(*, manifest: Path, results_root: Path, identity: dict) -> Path:
    evidence_id = _orli_evidence_id(identity)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    payload["persistence"] = {
        "schema_version": ORLI_PERSISTENCE_SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "identity": identity,
    }
    # Keep the process-local artifact identical to the authoritative persisted
    # artifact so workers exercise exactly what future builds will reuse.
    _write_json_atomic(Path(manifest), payload)

    artifact = _orli_artifact_path(results_root, evidence_id)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_name("manifest.json.tmp")
    shutil.copy2(manifest, temporary)
    os.replace(temporary, artifact)
    index = rebuild_orli_index(results_root=results_root)
    print(
        f"[learned-evidence][orli_page_mask] PERSISTED "
        f"evidence_id={evidence_id[:12]} pages={len(identity['image_keys'])} "
        f"bytes={artifact.stat().st_size} path={artifact} index={index}",
        flush=True,
    )
    return artifact


def prepare(
    *,
    detector: str,
    golden_set: Path,
    image_root: Path,
    maximum_dimension: int,
    output: Path,
    results_root: Path | None = None,
) -> Path:
    exporter = EXPORTERS.get(detector)
    if exporter is None:
        raise ValueError(f"Detector does not support shared learned evidence: {detector}")

    started = time.perf_counter()
    print(f"[learned-evidence][{detector}] preparing shared Golden Set evidence", flush=True)
    pages = load_pages(golden_set, image_root, maximum_dimension)
    images = [page["image"] for page in pages]

    orli_identity = None
    if detector == "orli_page_mask" and results_root is not None:
        orli_identity = _orli_identity(
            golden_set=golden_set,
            maximum_dimension=maximum_dimension,
            images=images,
        )
        reused = _reuse_persistent_orli_evidence(
            output=output,
            results_root=results_root,
            identity=orli_identity,
        )
        if reused is not None:
            elapsed = time.perf_counter() - started
            print(
                f"[learned-evidence][{detector}] SHARED EVIDENCE READY "
                f"pages={len(pages)} elapsed={elapsed:.2f}s path={reused} source=persisted",
                flush=True,
            )
            return reused

    print(
        f"[learned-evidence][{detector}] loaded {len(pages)} Golden Set page(s); "
        "model inference will run once before pipeline fan-out",
        flush=True,
    )
    manifest = exporter(
        images,
        output,
        progress=_progress(detector),
    )
    if detector == "orli_page_mask" and results_root is not None and orli_identity is not None:
        _persist_orli_evidence(
            manifest=manifest,
            results_root=results_root,
            identity=orli_identity,
        )
    elapsed = time.perf_counter() - started
    print(
        f"[learned-evidence][{detector}] SHARED EVIDENCE READY "
        f"pages={len(pages)} elapsed={elapsed:.2f}s path={manifest} source=inference",
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
    prep.add_argument("--results-root", type=Path, default=None)
    rebuild = sub.add_parser("rebuild-orli-index")
    rebuild.add_argument("--results-root", type=Path, required=True)
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
            results_root=args.results_root,
        )
        return 0
    if args.command == "rebuild-orli-index":
        target = rebuild_orli_index(results_root=args.results_root)
        print(f"Orli evidence index rebuilt: {target}", flush=True)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
