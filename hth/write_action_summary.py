from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MAX_BYTES = 950 * 1024
_DETAIL_MARKERS = (
    "<summary><h3>Per-Detector Calibration Reports</h3></summary>",
    "<summary><h3>Per-Detector Regression Reports</h3></summary>",
)


def _remove_balanced_details(text: str, marker: str) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    marker_index = next((i for i, line in enumerate(lines) if marker in line), None)
    if marker_index is None:
        return text, False

    start = marker_index
    while start >= 0 and "<details" not in lines[start]:
        start -= 1
    if start < 0:
        return text, False

    depth = 0
    end = None
    for index in range(start, len(lines)):
        depth += lines[index].count("<details")
        depth -= lines[index].count("</details>")
        if depth == 0 and index > start:
            end = index + 1
            break
    if end is None:
        return text, False

    title = "Per-detector calibration reports" if "Calibration" in marker else "Per-detector regression reports"
    replacement = [
        f"### {title}\n",
        "\n",
        "Detailed per-detector reports are omitted from the GitHub Actions step summary to keep it within GitHub's size limit. The complete manifest is preserved in the uploaded regression artifact.\n",
        "\n",
    ]
    return "".join(lines[:start] + replacement + lines[end:]), True


def compact_manifest(text: str) -> tuple[str, list[str]]:
    removed: list[str] = []
    compacted = text
    for marker in _DETAIL_MARKERS:
        compacted, changed = _remove_balanced_details(compacted, marker)
        if changed:
            removed.append(marker)
    return compacted, removed


def _truncate_blocks(text: str, byte_budget: int) -> str:
    if byte_budget <= 0:
        return ""
    blocks = text.split("\n\n")
    kept: list[str] = []
    used = 0
    for block in blocks:
        candidate = block if not kept else "\n\n" + block
        size = len(candidate.encode("utf-8"))
        if used + size > byte_budget:
            break
        kept.append(block)
        used += size
    if kept:
        return "\n\n".join(kept).rstrip() + "\n"
    return ""


def append_bounded_summary(source: Path, destination: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> dict[str, object]:
    full_text = source.read_text(encoding="utf-8")
    compacted, removed = compact_manifest(full_text)

    existing_bytes = destination.stat().st_size if destination.exists() else 0
    available = max(0, max_bytes - existing_bytes)
    notice = (
        "\n> **GitHub Actions summary note:** Detailed per-detector report bodies were omitted from this view "
        "to stay within GitHub's step-summary size limit. The complete manifest is preserved in the uploaded regression artifact.\n"
    )
    notice_bytes = len(notice.encode("utf-8"))

    candidate = compacted
    truncated = False
    if len(candidate.encode("utf-8")) + notice_bytes > available:
        candidate = _truncate_blocks(candidate, max(0, available - notice_bytes))
        truncated = True

    if removed or truncated:
        output = candidate.rstrip() + "\n" + notice
    else:
        output = candidate if candidate.endswith("\n") else candidate + "\n"

    encoded = output.encode("utf-8")
    if len(encoded) > available:
        output = _truncate_blocks(output, available)
        encoded = output.encode("utf-8")
        truncated = True

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Append exact UTF-8 bytes rather than text-mode output.  On Windows,
    # text mode translates ``\n`` to ``\r\n``; budgeting the UTF-8 payload
    # before that translation can therefore exceed GitHub's byte limit.
    with destination.open("ab") as handle:
        handle.write(encoded)

    return {
        "source_bytes": len(full_text.encode("utf-8")),
        "written_bytes": len(encoded),
        "existing_bytes": existing_bytes,
        "max_bytes": max_bytes,
        "removed_detail_sections": len(removed),
        "truncated": truncated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a bounded HTH report to GitHub Actions step summary.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args()

    result = append_bounded_summary(args.source, args.destination, args.max_bytes)
    print(
        "GitHub step summary: "
        f"source={result['source_bytes']} bytes, wrote={result['written_bytes']} bytes, "
        f"existing={result['existing_bytes']} bytes, budget={result['max_bytes']} bytes, "
        f"detail_sections_omitted={result['removed_detail_sections']}, truncated={str(result['truncated']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
