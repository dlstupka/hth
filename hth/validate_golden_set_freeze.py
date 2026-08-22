from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_freeze(*, freeze_path: Path, repository_root: Path) -> None:
    freeze = _load(freeze_path)
    if freeze.get("state") != "frozen":
        raise ValueError(f"freeze manifest is not frozen: {freeze_path}")

    relative = str(freeze.get("golden_set_path") or "").strip()
    if not relative:
        raise ValueError("freeze manifest is missing golden_set_path")
    golden_path = repository_root / relative
    data = golden_path.read_bytes()
    actual_sha = hashlib.sha256(data).hexdigest()
    expected_sha = str(freeze.get("golden_set_sha256") or "").strip()
    if actual_sha != expected_sha:
        raise ValueError(
            f"frozen Golden Set changed: {relative} sha256 {actual_sha} != {expected_sha}; "
            "create a new Golden Set ID instead of editing HTH-0001 in place"
        )

    golden = json.loads(data.decode("utf-8"))
    golden_id = str(golden.get("collection_id") or golden.get("id") or "").strip()
    frozen_id = str(freeze.get("golden_set_id") or "").strip()
    if golden_id != frozen_id:
        raise ValueError(f"Golden Set identity changed: {golden_id!r} != {frozen_id!r}")

    pages = golden.get("pages") or []
    ordinals = [int(page["global_ordinal"]) for page in pages]
    membership = freeze.get("membership") or {}
    expected_ordinals = [int(value) for value in membership.get("global_ordinals") or []]
    expected_count = int(membership.get("page_count") or 0)
    if ordinals != expected_ordinals or len(pages) != expected_count:
        raise ValueError(
            f"frozen Golden Set membership changed: {ordinals} != {expected_ordinals}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an immutable HTH Golden Set freeze manifest")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args()
    validate_freeze(freeze_path=args.freeze, repository_root=args.repository_root)
    print(f"Frozen Golden Set verified: {args.freeze}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
