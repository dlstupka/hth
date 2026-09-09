#!/usr/bin/env python3
"""Validate and seed legacy results-repository models, or purge their Git history."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hth.artifact_mirror import MirrorArtifact, download as download_mirror, publish
from hth.detector_lifecycle import ORLI_MODEL_ID, ORLI_MODEL_MIRROR, _validate_safetensors_file, _validate_zip_file


HASH_FILES = {
    "model_sha256": "model_filename",
    "parameters_sha256": "parameters.yml",
    "deploy_prototxt_sha256": "ohio_deploy.prototxt",
    "train_prototxt_sha256": "ohio_train_val.prototxt",
    "weights_sha256": "ohio_weights.caffemodel",
    "archive_sha256": "model.zip",
    "source_archive_sha256": "source.zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_dir(model_dir: Path) -> dict[str, object]:
    provenance_path = model_dir / "model-provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(f"{model_dir.name}: model-provenance.json is missing")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("model_id") != model_dir.name:
        raise RuntimeError(f"{model_dir.name}: provenance model_id does not match directory")
    if not provenance.get("license"):
        raise RuntimeError(f"{model_dir.name}: provenance license is missing")
    if not (provenance.get("upstream_repository") or provenance.get("model_repository")):
        raise RuntimeError(f"{model_dir.name}: authoritative repository is missing")
    verified = []
    for hash_field, filename_field in HASH_FILES.items():
        expected = provenance.get(hash_field)
        if not expected:
            continue
        filename = provenance.get(filename_field) if filename_field == "model_filename" else filename_field
        if not filename:
            raise RuntimeError(f"{model_dir.name}: {filename_field} is missing")
        artifact = model_dir / str(filename)
        if not artifact.is_file():
            raise RuntimeError(f"{model_dir.name}: recorded artifact {filename} is missing")
        actual = sha256(artifact)
        if actual != expected:
            raise RuntimeError(
                f"{model_dir.name}: {filename} SHA-256 mismatch; expected={expected} actual={actual}"
            )
        verified.append(str(filename))
    if not verified:
        raise RuntimeError(f"{model_dir.name}: provenance contains no verifiable artifact hashes")
    if model_dir.name == ORLI_MODEL_ID:
        _validate_safetensors_file(model_dir / ORLI_MODEL_MIRROR.asset_name)
    return {"provenance": provenance, "verified_files": verified}


def deterministic_bundle(model_dir: Path, destination: Path) -> None:
    files = sorted(
        path for path in model_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(model_dir).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def mirror_spec(model_dir: Path, provenance: dict[str, object], bundle: Path | None) -> MirrorArtifact:
    if model_dir.name == ORLI_MODEL_ID:
        return ORLI_MODEL_MIRROR
    normalized = re.sub(r"[^A-Z0-9]+", "-", model_dir.name.upper()).strip("-")
    repository = str(provenance.get("model_repository") or provenance.get("upstream_repository"))
    reference = str(
        provenance.get("model_doi") or provenance.get("model_source_reference")
        or provenance.get("model_release_version") or provenance.get("schema_version")
    )
    return MirrorArtifact(
        repository="dlstupka/hth-mirror",
        tag=f"HTH-MIRROR-{normalized}",
        asset_name=f"{model_dir.name}.zip",
        artifact_id=model_dir.name,
        authoritative_repository=repository,
        authoritative_reference=reference,
        license=str(provenance["license"]),
    )


def fetch(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "HTH-results-model-migration/1"})
    with urllib.request.urlopen(request) as response, Path(target).open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)


def verify_published_mirror(model_dir: Path, spec: MirrorArtifact, expected_artifact: Path) -> None:
    """Download the published artifact and validate both transport and model contents."""
    with tempfile.TemporaryDirectory() as temp:
        temp_root = Path(temp)
        downloaded = temp_root / spec.asset_name
        validator = _validate_safetensors_file if model_dir.name == ORLI_MODEL_ID else _validate_zip_file
        download_mirror(spec, downloaded, fetch=fetch, validator=validator)
        if sha256(downloaded) != sha256(expected_artifact):
            raise RuntimeError(f"{model_dir.name}: published artifact differs from verified local source")
        if model_dir.name != ORLI_MODEL_ID:
            extracted = temp_root / model_dir.name
            extracted.mkdir()
            with zipfile.ZipFile(downloaded) as archive:
                archive.extractall(extracted)
            validate_model_dir(extracted)
def seed(results_repo: Path, *, token: str | None, dry_run: bool, selected_models: list[str] | None = None) -> None:
    model_root = results_repo / "models"
    model_dirs = sorted(path for path in model_root.iterdir() if path.is_dir())
    if selected_models:
        requested = set(selected_models)
        model_dirs = [path for path in model_dirs if path.name in requested]
        missing = requested - {path.name for path in model_dirs}
        if missing:
            raise RuntimeError("Requested model cache(s) not found: " + ", ".join(sorted(missing)))
    if not model_dirs:
        raise RuntimeError(f"No model directories found under {model_root}")
    validated = []
    failures = []
    for model_dir in model_dirs:
        try:
            validated.append((model_dir, validate_model_dir(model_dir)))
        except Exception as exc:
            failures.append(str(exc))
    if failures:
        raise RuntimeError("Model verification failed; nothing was published:\n- " + "\n- ".join(failures))
    with tempfile.TemporaryDirectory() as temp:
        temp_root = Path(temp)
        for model_dir, result in validated:
            provenance = result["provenance"]
            if model_dir.name == ORLI_MODEL_ID:
                artifact = model_dir / ORLI_MODEL_MIRROR.asset_name
                bundle = None
            else:
                artifact = temp_root / f"{model_dir.name}.zip"
                deterministic_bundle(model_dir, artifact)
                bundle = artifact
            spec = mirror_spec(model_dir, provenance, bundle)
            print(
                f"Verified {model_dir.name}: {len(result['verified_files'])} provenance hash(es); "
                f"mirror={spec.tag}/{spec.asset_name} sha256={sha256(artifact)}"
            )
            if dry_run:
                continue
            status = publish(
                spec, artifact, token=token,
                authoritative_source={
                    "site": provenance.get("model_source_site") or "legacy validated results cache",
                    "url": provenance.get("model_url"),
                    "reference": provenance.get("model_source_reference") or provenance.get("model_doi"),
                    "provenance_sha256": sha256(model_dir / "model-provenance.json"),
                },
            )
            if status == "skipped-no-token":
                raise RuntimeError("HTH_RELEASES_TOKEN is required to seed hth-mirror")
            print(f"Published {model_dir.name}: status={status}")
            verify_published_mirror(model_dir, spec, artifact)
            print(f"Verified published mirror retrieval: model={model_dir.name} sha256={sha256(artifact)}")


def git(results_repo: Path, *args: str, capture: bool = False) -> str:
    process = subprocess.run(
        ["git", "-C", str(results_repo), *args], check=True,
        text=True, capture_output=capture,
    )
    return process.stdout if capture else ""


def verify_current_mirrors(results_repo: Path, selected_models: list[str] | None = None) -> None:
    model_root = results_repo / "models"
    model_dirs = sorted(path for path in model_root.iterdir() if path.is_dir())
    if selected_models:
        requested = set(selected_models)
        model_dirs = [path for path in model_dirs if path.name in requested]
        missing = requested - {path.name for path in model_dirs}
        if missing:
            raise RuntimeError("Requested model cache(s) not found: " + ", ".join(sorted(missing)))
    verified = 0
    for model_dir in model_dirs:
        substantive = [
            path for path in model_dir.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        ]
        if not substantive:
            print(f"Ignoring bytecode-only cache during mirror verification: {model_dir.name}")
            continue
        result = validate_model_dir(model_dir)
        provenance = result["provenance"]
        with tempfile.TemporaryDirectory() as temp:
            if model_dir.name == ORLI_MODEL_ID:
                artifact = model_dir / ORLI_MODEL_MIRROR.asset_name
                bundle = None
            else:
                artifact = Path(temp) / f"{model_dir.name}.zip"
                deterministic_bundle(model_dir, artifact)
                bundle = artifact
            spec = mirror_spec(model_dir, provenance, bundle)
            verify_published_mirror(model_dir, spec, artifact)
        verified += 1
        print(f"Verified downloadable mirror: model={model_dir.name}")
    if not verified:
        raise RuntimeError("No substantive model caches were verified")


def purge(results_repo: Path, *, confirmation: str, backup: Path) -> None:
    if confirmation != "PURGE-MODELS-AND-HISTORY":
        raise RuntimeError("--purge requires --confirm PURGE-MODELS-AND-HISTORY")
    if git(results_repo, "status", "--porcelain", capture=True).strip():
        raise RuntimeError("Results repository must have a clean working tree before history rewrite")
    # The destructive phase cannot start until every substantive current model
    # cache has a byte-identical, downloadable, provenance-valid mirror copy.
    verify_current_mirrors(results_repo)
    backup.parent.mkdir(parents=True, exist_ok=True)
    git(results_repo, "bundle", "create", str(backup), "--all")
    git(
        results_repo, "filter-branch", "--force", "--index-filter",
        "git rm -r --cached --ignore-unmatch models", "--prune-empty",
        "--tag-name-filter", "cat", "--", "--all",
    )
    refs = git(results_repo, "for-each-ref", "--format=%(refname)", "refs/original/", capture=True)
    for ref in refs.splitlines():
        git(results_repo, "update-ref", "-d", ref)
    git(results_repo, "reflog", "expire", "--expire=now", "--all")
    git(results_repo, "gc", "--prune=now")
    objects = git(results_repo, "rev-list", "--objects", "--all", capture=True)
    remaining = [line for line in objects.splitlines() if " models/" in line or line.endswith(" models")]
    if remaining:
        raise RuntimeError("Model paths remain in rewritten history")
    print(f"Purged models/ from local branches and tags. Recovery bundle: {backup}")
    print("No remote was modified. Review the rewritten repository before force-pushing branches and tags.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-repo", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", action="store_true")
    mode.add_argument("--verify", action="store_true", help="Verify current caches against downloadable mirror releases")
    mode.add_argument("--purge", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show seed plan without publishing")
    parser.add_argument("--model", action="append", default=[], help="Seed only this model ID; repeat as needed")
    parser.add_argument("--confirm", default="", help="Required literal for destructive history purge")
    parser.add_argument("--backup", type=Path, help="Recovery bundle path for --purge")
    args = parser.parse_args()
    results_repo = args.results_repo.resolve()
    if args.seed:
        seed(
            results_repo, token=os.environ.get("HTH_RELEASES_TOKEN"),
            dry_run=args.dry_run, selected_models=args.model,
        )
    elif args.verify:
        if args.dry_run:
            raise RuntimeError("--dry-run is not meaningful with --verify")
        verify_current_mirrors(results_repo, selected_models=args.model)
    else:
        if args.dry_run:
            raise RuntimeError("--dry-run applies only to --seed; --purge requires explicit confirmation")
        backup = args.backup or results_repo.parent / f"{results_repo.name}-before-model-purge.bundle"
        purge(results_repo, confirmation=args.confirm, backup=backup.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
