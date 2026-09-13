from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from hth.geometry import detector_dhsegment_page_mask, detector_eynollah_page_mask, detector_docextractor_page_mask, detector_pagenet_page_mask, detector_doc_ufcn_page_mask, detector_mask_rcnn_page_mask, detector_kraken_page_mask, detector_orli_page_mask
from hth.collection_cache import (
    EvidenceCacheArtifact,
    canonical_evidence_id,
    deterministic_evidence_bundle,
    download,
    fetch_url,
    materialize,
    publish,
    sha256_file,
    validate_evidence_bundle,
)
from hth.persistence import canonical_index_path, atomic_write_json


EXPORTERS = {
    "kraken_page_mask": detector_kraken_page_mask.export_precomputed_golden_set_evidence,
    "doc_ufcn_page_mask": detector_doc_ufcn_page_mask.export_precomputed_golden_set_evidence,
    "amsre_doc_ufcn_fusion": detector_doc_ufcn_page_mask.export_precomputed_golden_set_evidence,
    "mask_rcnn_page_mask": detector_mask_rcnn_page_mask.export_precomputed_golden_set_evidence,
    "orli_page_mask": detector_orli_page_mask.export_precomputed_golden_set_evidence,
    "dhsegment_page_mask": detector_dhsegment_page_mask.export_precomputed_golden_set_evidence,
    "eynollah_page_mask": detector_eynollah_page_mask.export_precomputed_golden_set_evidence,
    "docextractor_page_mask": detector_docextractor_page_mask.export_precomputed_golden_set_evidence,
    "pagenet_page_mask": detector_pagenet_page_mask.export_precomputed_golden_set_evidence,
}

ORLI_EVIDENCE_INDEX = Path("indexes") / "orli-evidence-index.json"  # compatibility alias; canonical path is owned by hth.persistence
ORLI_EVIDENCE_ROOT = Path("learned-evidence") / "orli_page_mask"
ORLI_PERSISTENCE_SCHEMA_VERSION = "1.0"
EVIDENCE_PERSISTENCE_SCHEMA_VERSION = "1.0"

PROVENANCE_ENVS = {
    "kraken_page_mask": "HTH_KRAKEN_PAGE_PROVENANCE",
    "doc_ufcn_page_mask": "HTH_DOC_UFCN_PAGE_PROVENANCE",
    "amsre_doc_ufcn_fusion": "HTH_DOC_UFCN_PAGE_PROVENANCE",
    "mask_rcnn_page_mask": "HTH_MASK_RCNN_PAGE_PROVENANCE",
    "orli_page_mask": "HTH_ORLI_PAGE_PROVENANCE",
    "dhsegment_page_mask": "HTH_DHSEGMENT_PAGE_PROVENANCE",
    "eynollah_page_mask": "HTH_EYNOLLAH_PAGE_PROVENANCE",
    "docextractor_page_mask": "HTH_DOCEXTRACTOR_PAGE_PROVENANCE",
    "pagenet_page_mask": "HTH_LEARNED_PAGE_MASK_PROVENANCE",
}

IMAGE_KEYERS = {
    "kraken_page_mask": detector_kraken_page_mask._image_key,
    "doc_ufcn_page_mask": detector_doc_ufcn_page_mask._image_key,
    "amsre_doc_ufcn_fusion": detector_doc_ufcn_page_mask._image_key,
    "mask_rcnn_page_mask": detector_mask_rcnn_page_mask._image_key,
    "orli_page_mask": detector_orli_page_mask._image_key,
    "dhsegment_page_mask": detector_dhsegment_page_mask._image_key,
    "eynollah_page_mask": detector_eynollah_page_mask._image_key,
    "docextractor_page_mask": detector_docextractor_page_mask._image_key,
    "pagenet_page_mask": detector_pagenet_page_mask._image_key,
}

EVIDENCE_REPRESENTATIONS = {
    "kraken_page_mask": "immutable-json",
    "doc_ufcn_page_mask": "doc-ufcn-page-polygons",
    "amsre_doc_ufcn_fusion": "doc-ufcn-page-polygons",
    "mask_rcnn_page_mask": "hjdataset-mask-rcnn-instances",
    "orli_page_mask": "immutable-json",
    "dhsegment_page_mask": "readonly-npy",
    "eynollah_page_mask": "eynollah-page-probability",
    "docextractor_page_mask": "docextractor-foreground-probability",
    "pagenet_page_mask": "pagenet-ohio-page-probability-256",
}
CACHE_DETECTORS = {
    # The fusion consumes exactly the immutable Doc-UFCN polygon evidence; its
    # classical AMSRE branch is parameter-dependent and is never cached here.
    "amsre_doc_ufcn_fusion": "doc_ufcn_page_mask",
}


def load_pages(golden_set: Path, image_root: Path, maximum_dimension: int):
    # Lazy import keeps runner -> learned_evidence reuse free of an import cycle.
    from hth.regression.runner import load_pages as runner_load_pages
    return runner_load_pages(golden_set, image_root, maximum_dimension)


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
    return sha256_file(path)


def _provenance(detector: str) -> dict:
    env_name = PROVENANCE_ENVS[detector]
    raw = str(os.environ.get(env_name) or "").strip()
    if not raw:
        raise RuntimeError(f"{env_name} is required for canonical {detector} evidence")
    payload = json.loads(Path(raw).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("model_id"):
        raise RuntimeError(f"{detector} model provenance has no model_id")
    return payload


def _artifact_hashes(value, prefix="") -> dict[str, str]:
    hashes = {}
    if isinstance(value, dict):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            item = value[key]
            if str(key).endswith("sha256") and isinstance(item, str) and len(item) == 64:
                hashes[path] = item.lower()
            else:
                hashes.update(_artifact_hashes(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hashes.update(_artifact_hashes(item, f"{prefix}[{index}]"))
    return hashes


def evidence_identity(*, detector: str, golden_set: Path, maximum_dimension: int, images) -> dict:
    if detector == "orli_page_mask":
        return _orli_identity(
            golden_set=golden_set,
            maximum_dimension=maximum_dimension,
            images=images,
        )
    provenance = _provenance(detector)
    cache_detector = CACHE_DETECTORS.get(detector, detector)
    hashes = _artifact_hashes(provenance)
    if not hashes:
        raise RuntimeError(f"{detector} model provenance has no artifact SHA-256 identity")
    return {
        "schema_version": EVIDENCE_PERSISTENCE_SCHEMA_VERSION,
        "detector": cache_detector,
        "model_id": provenance.get("model_id"),
        "model_artifact_hashes": hashes,
        "model_variant": provenance.get("model_variant") or provenance.get("variant"),
        "inference_backend": provenance.get("inference_backend"),
        "serving_contract": provenance.get("serving_contract") or provenance.get("input_contract"),
        "golden_set_sha256": _sha256_file(golden_set),
        "maximum_dimension": int(maximum_dimension),
        "image_keys": [IMAGE_KEYERS[detector](image) for image in images],
        "evidence_representation": EVIDENCE_REPRESENTATIONS[detector],
    }


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
    return canonical_evidence_id(identity)


def _orli_artifact_path(results_root: Path, evidence_id: str) -> Path:
    return Path(results_root) / ORLI_EVIDENCE_ROOT / evidence_id / "manifest.json"


def _orli_index_path(results_root: Path) -> Path:
    return canonical_index_path(results_root, "orli-evidence-index.json")


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
    atomic_write_json(target, payload)
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
    atomic_write_json(Path(manifest), payload)

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


def _attach_persistence(manifest: Path, identity: dict) -> tuple[Path, str]:
    manifest = Path(manifest)
    evidence_id = canonical_evidence_id(identity)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["persistence"] = {
        "schema_version": EVIDENCE_PERSISTENCE_SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "identity": identity,
    }
    atomic_write_json(manifest, payload)
    return manifest, evidence_id


def _cache_repository(explicit: str | None = None) -> str | None:
    value = explicit if explicit is not None else os.environ.get("HTH_EVIDENCE_CACHE_REPOSITORY", "")
    value = str(value or "").strip()
    return value or None


def _local_cache_bundle(spec: EvidenceCacheArtifact) -> Path | None:
    root = str(os.environ.get("HTH_EVIDENCE_LOCAL_CACHE_ROOT") or "").strip()
    if not root:
        return None
    return Path(root) / spec.detector / spec.asset_name


def _reuse_local_cache(*, output: Path, spec: EvidenceCacheArtifact) -> Path | None:
    bundle = _local_cache_bundle(spec)
    if bundle is None or not bundle.is_file():
        return None
    try:
        validated = validate_evidence_bundle(bundle)
        if validated["identity"] != spec.identity:
            raise RuntimeError("local evidence identity mismatch")
        manifest = materialize(bundle, output)
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(
            f"[learned-evidence][{spec.detector}] LOCAL CACHE REJECTED "
            f"evidence_id={spec.evidence_id[:12]} reason={type(exc).__name__}: {exc}",
            flush=True,
        )
        bundle.unlink(missing_ok=True)
        return None
    print(
        f"[learned-evidence][{spec.detector}] LOCAL CACHE HIT "
        f"evidence_id={spec.evidence_id[:12]} path={manifest}",
        flush=True,
    )
    return manifest


def _store_local_bundle(bundle: Path, spec: EvidenceCacheArtifact) -> Path | None:
    target = _local_cache_bundle(spec)
    if target is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    shutil.copy2(bundle, temporary)
    os.replace(temporary, target)
    return target


def _reuse_collection_cache(*, output: Path, spec: EvidenceCacheArtifact) -> Path | None:
    with tempfile.TemporaryDirectory(dir=Path(output).parent) as temp:
        bundle = Path(temp) / spec.asset_name
        try:
            download(spec, bundle, fetch=fetch_url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(
                    f"[learned-evidence][{spec.detector}] COLLECTION CACHE MISS "
                    f"evidence_id={spec.evidence_id[:12]}",
                    flush=True,
                )
                return None
            print(
                f"[learned-evidence][{spec.detector}] COLLECTION CACHE UNAVAILABLE "
                f"evidence_id={spec.evidence_id[:12]} status={exc.code}; regenerating",
                flush=True,
            )
            return None
        except (OSError, RuntimeError, json.JSONDecodeError) as exc:
            print(
                f"[learned-evidence][{spec.detector}] COLLECTION CACHE REJECTED "
                f"evidence_id={spec.evidence_id[:12]} reason={type(exc).__name__}: {exc}; regenerating",
                flush=True,
            )
            return None
        local = _store_local_bundle(bundle, spec)
        manifest = materialize(local or bundle, output)
    print(
        f"[learned-evidence][{spec.detector}] COLLECTION CACHE HIT "
        f"evidence_id={spec.evidence_id[:12]} pages={len(spec.identity['image_keys'])} path={manifest}",
        flush=True,
    )
    return manifest


def _publish_collection_cache(*, manifest: Path, spec: EvidenceCacheArtifact) -> None:
    with tempfile.TemporaryDirectory(dir=Path(manifest).parent.parent) as temp:
        bundle = Path(temp) / spec.asset_name
        deterministic_evidence_bundle(manifest, bundle)
        local = _store_local_bundle(bundle, spec)
        if local is not None:
            print(
                f"[learned-evidence][{spec.detector}] LOCAL CACHE FILLED "
                f"evidence_id={spec.evidence_id[:12]} path={local}",
                flush=True,
            )
        if not spec.repository:
            return
        try:
            status = publish(
                spec,
                bundle,
                evidence_manifest_sha256=_sha256_file(manifest),
            )
        except (OSError, RuntimeError, urllib.error.HTTPError) as exc:
            print(
                f"[learned-evidence][{spec.detector}] COLLECTION CACHE PUBLISH FAILED "
                f"evidence_id={spec.evidence_id[:12]} reason={type(exc).__name__}: {exc}",
                flush=True,
            )
            return
    print(
        f"[learned-evidence][{spec.detector}] COLLECTION CACHE {status.upper()} "
        f"evidence_id={spec.evidence_id[:12]} repository={spec.repository}",
        flush=True,
    )


def prepare_images(
    *,
    detector: str,
    golden_set: Path,
    maximum_dimension: int,
    images,
    output: Path,
    results_root: Path | None = None,
    cache_repository: str | None = None,
) -> tuple[Path, str]:
    exporter = EXPORTERS.get(detector)
    if exporter is None:
        raise ValueError(f"Detector does not support shared learned evidence: {detector}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = evidence_identity(
        detector=detector,
        golden_set=golden_set,
        maximum_dimension=maximum_dimension,
        images=images,
    )
    repository = _cache_repository(cache_repository)
    local_cache_enabled = bool(str(os.environ.get("HTH_EVIDENCE_LOCAL_CACHE_ROOT") or "").strip())
    spec = None
    if repository or local_cache_enabled:
        evidence_id = canonical_evidence_id(identity)
        spec = EvidenceCacheArtifact(repository or "", str(identity["detector"]), evidence_id, identity)
        reused = _reuse_local_cache(output=output, spec=spec)
        if reused is not None:
            return reused, "runner-local-cache"
        if repository:
            reused = _reuse_collection_cache(output=output, spec=spec)
            if reused is not None:
                return reused, "collection-cache"

    # Transitional read compatibility for the two already-migrated Orli
    # manifests. No new evidence is written to the results repository.
    if detector == "orli_page_mask" and results_root is not None:
        reused = _reuse_persistent_orli_evidence(
            output=output,
            results_root=results_root,
            identity=identity,
        )
        if reused is not None:
            return reused, "legacy-results-cache"

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = exporter(images, output, progress=_progress(detector))
    manifest, evidence_id = _attach_persistence(manifest, identity)
    if spec is not None:
        _publish_collection_cache(manifest=manifest, spec=spec)
    return manifest, "inference"


def prepare(
    *,
    detector: str,
    golden_set: Path,
    image_root: Path,
    maximum_dimension: int,
    output: Path,
    results_root: Path | None = None,
    cache_repository: str | None = None,
) -> Path:
    exporter = EXPORTERS.get(detector)
    if exporter is None:
        raise ValueError(f"Detector does not support shared learned evidence: {detector}")

    started = time.perf_counter()
    print(f"[learned-evidence][{detector}] preparing shared Golden Set evidence", flush=True)
    pages = load_pages(golden_set, image_root, maximum_dimension)
    images = [page["image"] for page in pages]

    manifest, source = prepare_images(
        detector=detector,
        golden_set=golden_set,
        maximum_dimension=maximum_dimension,
        images=images,
        output=output,
        results_root=results_root,
        cache_repository=cache_repository,
    )
    elapsed = time.perf_counter() - started
    print(
        f"[learned-evidence][{detector}] SHARED EVIDENCE READY "
        f"pages={len(pages)} elapsed={elapsed:.2f}s path={manifest} source={source}",
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
    prep.add_argument("--cache-repository", default=None)
    rebuild = sub.add_parser("rebuild-orli-index")
    rebuild.add_argument("--results-root", type=Path, required=True)
    sub.add_parser("supported")
    sub.add_parser("registry")
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
            cache_repository=args.cache_repository,
        )
        return 0
    if args.command == "rebuild-orli-index":
        target = rebuild_orli_index(results_root=args.results_root)
        print(f"Orli evidence index rebuilt: {target}", flush=True)
        return 0
    if args.command == "supported":
        print("\n".join(sorted(EXPORTERS)), flush=True)
        return 0
    if args.command == "registry":
        for detector in sorted(EXPORTERS):
            print(f"{detector}\t{CACHE_DETECTORS.get(detector, detector)}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
