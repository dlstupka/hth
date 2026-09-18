#!/usr/bin/env python3
"""Render the canonical runner-target catalog into GitHub workflow dispatch UI."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hth.runner_targets import load_runner_targets


WORKFLOW_ROOT = ROOT / ".github" / "workflows"
LEGACY_DEFAULTS = {
    "github-hosted": "github-hosted",
    "self-hosted-hth": "hth",
    "self-hosted-windows": "self-hosted-windows",
    "self-hosted-rhel8": "rhel8",
    "self-hosted-e7k": "e7k",
    "self-hosted-e9k": "e9k",
}
BEGIN_INPUT = "# BEGIN GENERATED RUNNER TARGET INPUT"
END_INPUT = "# END GENERATED RUNNER TARGET INPUT"
BEGIN_RUNS_ON = "# BEGIN GENERATED RUNNER TARGET ROUTING"
END_RUNS_ON = "# END GENERATED RUNNER TARGET ROUTING"
BEGIN_SETUP = "# BEGIN GENERATED RUNNER SETUP LABEL"
END_SETUP = "# END GENERATED RUNNER SETUP LABEL"
BEGIN_LABELS = "# BEGIN GENERATED RUNNER LABEL SET"
END_LABELS = "# END GENERATED RUNNER LABEL SET"


def _expression(targets: list[dict], field: str, *, indent: str) -> str:
    branches = []
    for target in targets:
        if target["id"] == "github-hosted":
            continue
        value = target[field]
        rendered = (
            f"fromJSON('{json.dumps(value, separators=(',', ':'))}')"
            if isinstance(value, list)
            else f"'{value}'"
        )
        branches.append(f"inputs.runner_target == '{target['id']}' && {rendered}")
    continuation = f"\n{indent}  || "
    return "${{ " + continuation.join(branches) + continuation + (
        "'ubuntu-latest' }}" if field == "runs_on" else "'github-hosted' }}"
    )


def _label_set_expression(targets: list[dict], *, indent: str) -> str:
    projected = [{**target, "label_set": ",".join(target["runs_on"])} for target in targets]
    return _expression(projected, "label_set", indent=indent)


def _input_block(targets: list[dict], default: str, *, reusable: bool) -> str:
    lines = [f"      {BEGIN_INPUT}", "      runner_target:"]
    if reusable:
        lines.extend([
            '        description: "Canonical execution runner target"',
            "        required: false",
            "        type: string",
            f'        default: "{default}"',
        ])
    else:
        lines.extend([
            '        description: "Execution runner target"',
            "        required: true",
            f"        default: {default}",
            "        type: choice",
            "        options:",
            *[f"          - {target['id']}" for target in targets],
        ])
    lines.append(f"      {END_INPUT}")
    return "\n".join(lines) + "\n"


def _runs_on_block(targets: list[dict]) -> str:
    return "\n".join([
        f"    {BEGIN_RUNS_ON}",
        "    runs-on: >-",
        "      " + _expression(targets, "runs_on", indent="      "),
        f"    {END_RUNS_ON}",
    ]) + "\n"


def _setup_block(targets: list[dict]) -> str:
    return "\n".join([
        f"      {BEGIN_SETUP}",
        "      HTH_SELECTED_RUNNER_LABEL: >-",
        "        " + _expression(targets, "setup_label", indent="        "),
        f"      {END_SETUP}",
    ]) + "\n"


def _core_setup_block(targets: list[dict]) -> str:
    return "\n".join([
        f"  {BEGIN_SETUP}",
        "  HTH_RUNNER_LABEL: >-",
        "    " + _expression(targets, "setup_label", indent="    "),
        f"  {END_SETUP}",
    ]) + "\n"


def _optimizer_label_blocks(targets: list[dict]) -> tuple[str, str]:
    labels = "\n".join([
        f"      {BEGIN_LABELS}",
        "      HTH_RUNNER_LABELS: >-",
        "        " + _label_set_expression(targets, indent="        "),
        f"      {END_LABELS}",
    ]) + "\n"
    label = "\n".join([
        f"      {BEGIN_SETUP}",
        "      HTH_RUNNER_LABEL: >-",
        "        " + _expression(targets, "setup_label", indent="        "),
        f"      {END_SETUP}",
    ]) + "\n"
    return labels, label


def _replace_region(text: str, begin: str, end: str, replacement: str, indent: int) -> tuple[str, bool]:
    pattern = rf"(?ms)^{' ' * indent}{re.escape(begin)}\n.*?^{' ' * indent}{re.escape(end)}\n"
    updated, count = re.subn(pattern, replacement, text)
    return updated, bool(count)


def render_workflow(path: Path, targets: list[dict]) -> str:
    text = path.read_text(encoding="utf-8")
    reusable = path.name == "_core-hth.yml"

    marker = re.search(rf"(?ms)^      {re.escape(BEGIN_INPUT)}\n.*?^      {re.escape(END_INPUT)}\n", text)
    if marker:
        current = marker.group(0)
        default_match = re.search(r"^        default: ['\"]?([^'\"\n]+)", current, re.MULTILINE)
        default = default_match.group(1) if default_match else "github-hosted"
        text = text[:marker.start()] + _input_block(targets, default, reusable=reusable) + text[marker.end():]
    else:
        legacy = re.search(
            r"(?m)^      runner:\n(?: {8}.*\n|[ \t]*\n)*"
            r"^      specific_runner:\n(?: {8}.*\n|[ \t]*\n)*"
            r"^      custom_runner_label:\n(?: {8}.*\n|[ \t]*\n)*",
            text,
        )
        if not legacy:
            return text
        default_match = re.search(r"^        default: ['\"]?([^'\"\n]+)", legacy.group(0), re.MULTILINE)
        legacy_default = default_match.group(1) if default_match else "github-hosted"
        default = LEGACY_DEFAULTS.get(legacy_default, legacy_default)
        text = text[:legacy.start()] + _input_block(targets, default, reusable=reusable) + text[legacy.end():]

    # Reusable-workflow callers pass one value instead of the legacy three-part selector.
    text = re.sub(
        r"(?m)^(      )runner: (.+)\n\1specific_runner: .+\n\1custom_runner_label: .+\n",
        lambda match: f"{match.group(1)}runner_target: {match.group(2).replace('inputs.runner', 'inputs.runner_target')}\n",
        text,
    )

    # Direct jobs route from the generated catalog expression.
    routed, replaced = _replace_region(text, BEGIN_RUNS_ON, END_RUNS_ON, _runs_on_block(targets), 4)
    text = routed
    if not replaced:
        text = re.sub(
            r"(?ms)^    runs-on: >-\n      \$\{\{\s+inputs\.specific_runner.*?^          \|\| 'ubuntu-latest' \}\}\n",
            _runs_on_block(targets),
            text,
        )

    setup, replaced = _replace_region(text, BEGIN_SETUP, END_SETUP, _setup_block(targets), 6)
    text = setup
    if not replaced:
        text = re.sub(
            r"(?ms)^      HTH_SELECTED_RUNNER_LABEL: >-\n        \$\{\{\s+inputs\.specific_runner.*?^          \|\| 'github-hosted' \}\}\n",
            _setup_block(targets),
            text,
        )

    if path.name == "execution-optimizer.yml":
        labels_block, label_block = _optimizer_label_blocks(targets)
        labels, replaced = _replace_region(text, BEGIN_LABELS, END_LABELS, labels_block, 6)
        text = labels
        if not replaced:
            text = re.sub(
                r"(?ms)^      HTH_RUNNER_LABELS: >-\n        \$\{\{\s+inputs\.specific_runner.*?^          \|\| 'ubuntu-latest' \}\}\n",
                labels_block,
                text,
            )
        label, replaced = _replace_region(text, BEGIN_SETUP, END_SETUP, label_block, 6)
        text = label
        if not replaced:
            text = re.sub(
                r"(?ms)^      HTH_RUNNER_LABEL: >-\n        \$\{\{\s+inputs\.specific_runner.*?^          \|\| 'github-hosted' \}\}\n",
                label_block,
                text,
            )

    if reusable:
        core_setup, replaced = _replace_region(
            text, BEGIN_SETUP, END_SETUP, _core_setup_block(targets), 2
        )
        text = core_setup
        if not replaced:
            text = text.replace(
                "  GIT_CONFIG_VALUE_0: main\n",
                "  GIT_CONFIG_VALUE_0: main\n" + _core_setup_block(targets),
                1,
            )

    # Dispatch helper CLIs and child workflows use the same single target.
    text = re.sub(
        r'(?m)^(\s*)--runner "([^\n]+)" \\\n\1--specific-runner "[^\n]+" \\\n\1--custom-runner-label "[^\n]+" \\\n',
        lambda match: f'{match.group(1)}--runner-target "{match.group(2).replace("inputs.runner", "inputs.runner_target")}" \\\n',
        text,
    )
    text = re.sub(
        r'(?m)^(\s*)--requested-runner "([^\n]+)" \\\n\1--specific-runner "[^\n]+" \\\n\1--custom-runner-label "[^\n]+" \\\n',
        lambda match: f'{match.group(1)}--runner-target "{match.group(2).replace("inputs.runner", "inputs.runner_target")}" \\\n',
        text,
    )
    text = text.replace("inputs.runner }}", "inputs.runner_target }}")
    text = text.replace("inputs.runner ||", "inputs.runner_target ||")
    text = text.replace("inputs.runner }}-", "inputs.runner_target }}-")
    return text


def synchronize(
    *, workflow_root: Path, catalog_path: Path, check: bool, repair: bool
) -> tuple[list[Path], bool]:
    targets = load_runner_targets(catalog_path)["targets"]
    rendered_by_path: dict[Path, str] = {}
    changed: list[Path] = []
    for path in sorted(workflow_root.glob("*.yml")):
        original = path.read_text(encoding="utf-8")
        rendered = render_workflow(path, targets)
        if rendered != original:
            changed.append(path)
            rendered_by_path[path] = rendered

    repaired = bool(check and repair and changed)
    if changed and (not check or repair):
        if repaired:
            names = ", ".join(path.name for path in changed)
            print(
                "::warning file=config/runner-targets.json::Runner-target workflow "
                f"rendering drift detected; automatically synchronizing: {names}",
                file=sys.stderr,
            )
        for path in changed:
            path.write_text(rendered_by_path[path], encoding="utf-8", newline="\n")

    if repaired:
        remaining = [
            path
            for path in sorted(workflow_root.glob("*.yml"))
            if render_workflow(path, targets) != path.read_text(encoding="utf-8")
        ]
        if remaining:
            names = ", ".join(path.name for path in remaining)
            raise RuntimeError(f"Runner-target repair did not converge: {names}")
        print("Runner-target workflow synchronization repaired and rechecked successfully.")
    return changed, repaired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="When --check finds drift, synchronize it, notify, and recheck",
    )
    parser.add_argument("--catalog", type=Path, default=ROOT / "config" / "runner-targets.json")
    parser.add_argument("--workflow-root", type=Path, default=WORKFLOW_ROOT)
    args = parser.parse_args()
    if args.repair and not args.check:
        parser.error("--repair requires --check")
    changed, repaired = synchronize(
        workflow_root=args.workflow_root,
        catalog_path=args.catalog,
        check=args.check,
        repair=args.repair,
    )
    if args.check and changed and not repaired:
        names = ", ".join(path.name for path in changed)
        raise SystemExit(f"Runner-target workflow rendering is stale: {names}")
    if not args.check:
        print(f"Synchronized runner targets in {len(changed)} workflow(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
