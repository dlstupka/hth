from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hth.runner_targets import load_runner_targets, resolve_runner_target, runner_target_ids


ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


class RunnerTargetTests(unittest.TestCase):
    def test_catalog_contains_every_supported_target_in_ui_order(self) -> None:
        self.assertEqual(
            runner_target_ids(),
            [
                "github-hosted",
                "self-hosted-linux",
                "self-hosted-windows",
                "hth",
                "rhel8",
                "e7k",
                "e9k",
                "192t",
                "96t",
                "32t",
            ],
        )

    def test_capacity_targets_resolve_to_exact_self_hosted_labels(self) -> None:
        for target_id in ("192t", "96t", "32t"):
            target = resolve_runner_target(target_id)
            self.assertEqual(
                target["runs_on"],
                ["self-hosted", "Linux", "X64", target_id],
            )
            self.assertEqual(target["setup_label"], target_id)

    def test_catalog_rejects_unknown_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown runner target"):
            resolve_runner_target("not-a-runner")

    def test_00_workflow_rendering_is_synchronized_with_catalog(self) -> None:
        completed = subprocess.run(
            [sys.executable, "tools/sync-runner-targets.py", "--check", "--repair"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if completed.stdout:
            print(completed.stdout, end="")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_mismatch_is_notified_repaired_and_retested(self) -> None:
        stale = """on:
  workflow_dispatch:
    inputs:
      # BEGIN GENERATED RUNNER TARGET INPUT
      runner_target:
        description: "Execution runner target"
        required: true
        default: github-hosted
        type: choice
        options:
          - github-hosted
      # END GENERATED RUNNER TARGET INPUT
jobs:
  example:
    runs-on: ubuntu-latest
"""
        with tempfile.TemporaryDirectory() as tmp:
            workflow_root = Path(tmp) / "workflows"
            workflow_root.mkdir()
            workflow = workflow_root / "example.yml"
            workflow.write_text(stale, encoding="utf-8")
            repair = subprocess.run(
                [
                    sys.executable,
                    "tools/sync-runner-targets.py",
                    "--check",
                    "--repair",
                    "--workflow-root",
                    str(workflow_root),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(repair.returncode, 0, repair.stdout + repair.stderr)
            self.assertIn("::warning", repair.stderr)
            self.assertIn("repaired and rechecked successfully", repair.stdout)
            repaired = workflow.read_text(encoding="utf-8")
            for target in load_runner_targets()["targets"]:
                self.assertIn(f"          - {target['id']}", repaired)
            recheck = subprocess.run(
                [
                    sys.executable,
                    "tools/sync-runner-targets.py",
                    "--check",
                    "--workflow-root",
                    str(workflow_root),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        self.assertEqual(recheck.returncode, 0, recheck.stdout + recheck.stderr)

    def test_legacy_last_input_migration_stops_at_input_block_boundary(self) -> None:
        script = ROOT / "tools" / "sync-runner-targets.py"
        spec = importlib.util.spec_from_file_location("sync_runner_targets", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        legacy = """on:
  workflow_dispatch:
    inputs:
      runner:
        default: github-hosted
      specific_runner:
        default: any
      custom_runner_label:
        default: ""
permissions:
  contents: read
jobs:
  example:
    runs-on: ubuntu-latest
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "example.yml"
            path.write_text(legacy, encoding="utf-8")
            rendered = module.render_workflow(path, load_runner_targets()["targets"])
        self.assertIn("runner_target:", rendered)
        self.assertIn("permissions:\n  contents: read\njobs:\n", rendered)

    def test_runner_enabled_workflows_have_only_one_selector(self) -> None:
        expected_options = [f"          - {target_id}" for target_id in runner_target_ids()]
        runner_workflows = 0
        for path in sorted(WORKFLOWS.glob("*.yml")):
            text = path.read_text(encoding="utf-8")
            if "# BEGIN GENERATED RUNNER TARGET INPUT" not in text:
                continue
            runner_workflows += 1
            self.assertEqual(text.count("      runner_target:\n"), 1, path.name)
            self.assertNotIn("      runner:\n", text, path.name)
            self.assertNotIn("      specific_runner:\n", text, path.name)
            self.assertNotIn("      custom_runner_label:\n", text, path.name)
            if path.name != "_core-hth.yml":
                for option in expected_options:
                    self.assertIn(option, text, path.name)
        self.assertGreater(runner_workflows, 10)

    def test_catalog_has_one_default(self) -> None:
        catalog = load_runner_targets()
        self.assertEqual(catalog["default_target"], "github-hosted")


if __name__ == "__main__":
    unittest.main()
