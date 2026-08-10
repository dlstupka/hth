from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from hth.regression_dispatch import _dispatch, resolve_targets


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_without_exhaustive_matches_current_golden_set_and_detector_config() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        detectors = root / "detectors"
        detectors.mkdir()
        (detectors / "a.json").write_text('{"a":1}\n', encoding="utf-8")
        (detectors / "b.json").write_text('{"b":1}\n', encoding="utf-8")
        golden = root / "golden.json"
        golden.write_text('{"pages":[1]}\n', encoding="utf-8")
        index = root / "calibration-index.json"
        index.write_text(json.dumps({"entries": [
            {
                "detector_id": "a",
                "detector_config_sha256": _sha(detectors / "a.json"),
                "golden_set_sha256": _sha(golden),
                "search": {"exhaustive_complete": True},
            },
            {
                "detector_id": "b",
                "detector_config_sha256": "stale-config",
                "golden_set_sha256": _sha(golden),
                "search": {"exhaustive_complete": True},
            },
        ]}), encoding="utf-8")
        assert resolve_targets(detectors, index, golden, "all-without-exhaustive") == ["b"]


def test_all_without_exhaustive_returns_every_detector_without_index() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        detectors = root / "detectors"
        detectors.mkdir()
        (detectors / "a.json").write_text("{}\n", encoding="utf-8")
        golden = root / "golden.json"
        golden.write_text("{}\n", encoding="utf-8")
        assert resolve_targets(detectors, root / "missing.json", golden, "all-without-exhaustive") == ["a"]


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_dispatch_accepts_any_2xx_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "hth.regression_dispatch.urllib.request.urlopen",
        lambda request: _FakeResponse(200),
    )
    _dispatch(
        "https://api.github.test/dispatches",
        "token",
        "main",
        "border_energy",
        {"mode": "full"},
    )


def test_dispatch_rejects_non_2xx_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "hth.regression_dispatch.urllib.request.urlopen",
        lambda request: _FakeResponse(300),
    )
    try:
        _dispatch(
            "https://api.github.test/dispatches",
            "token",
            "main",
            "border_energy",
            {"mode": "full"},
        )
    except RuntimeError as exc:
        assert "Unexpected dispatch status 300" in str(exc)
    else:
        raise AssertionError("Expected non-2xx dispatch status to fail")
