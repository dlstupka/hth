import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/regress-detector.yml",
    ROOT / ".github/workflows/execution-optimizer.yml",
)
PYTHON_ACTION = ROOT / ".github/actions/setup-hth-python/action.yml"
MANAGED_ACTION = ROOT / ".github/actions/setup-hth-managed-runtime/action.yml"
RUNTIME_MANAGER = ROOT / "tools" / "ensure-managed-runtime.sh"
RESULTS_CHECKOUT_PREP = ROOT / "tools" / "prepare-reusable-results-checkout.sh"


def _bash_executable() -> str | None:
    """Resolve Git Bash explicitly on Windows, avoiding the WSL app shim."""
    if os.name == "nt":
        git = shutil.which("git")
        candidates = []
        if git:
            git_root = Path(git).resolve().parent.parent
            candidates.extend((git_root / "bin" / "bash.exe", git_root / "usr" / "bin" / "bash.exe"))
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidates.append(program_files / "Git" / "bin" / "bash.exe")
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    return shutil.which("bash")


class RuntimeVerifyInstallWorkflowTests(unittest.TestCase):
    @contextmanager
    def _workspace(self):
        scratch = ROOT / ".test-reusable-results-checkout"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=scratch,
            ignore_cleanup_errors=os.name == "nt",
        ) as temporary_directory:
            yield Path(temporary_directory)
        try:
            scratch.rmdir()
        except OSError:
            pass

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )

    def _init_checkout(self, workspace: Path, origin: str = "https://github.com/owner/results.git") -> Path:
        checkout = workspace / "results-repo"
        checkout.mkdir()
        self._git("init", str(checkout))
        self._git("-C", str(checkout), "config", "user.name", "HTH test")
        self._git("-C", str(checkout), "config", "user.email", "hth-test@example.invalid")
        (checkout / "tracked.txt").write_text("committed\n", encoding="utf-8")
        self._git("-C", str(checkout), "add", "tracked.txt")
        self._git("-C", str(checkout), "commit", "-m", "seed")
        self._git("-C", str(checkout), "remote", "add", "origin", origin)
        return checkout

    def _checkout_helper(
        self,
        workspace: Path,
        mode: str = "prepare",
        expected_repository: str = "owner/results",
    ) -> subprocess.CompletedProcess[bytes]:
        bash = _bash_executable()
        self.assertIsNotNone(bash, "A POSIX bash executable is required")
        relative_workspace = workspace.relative_to(ROOT).as_posix()
        script = (
            f"GITHUB_WORKSPACE={shlex.quote(relative_workspace)} "
            "bash tools/prepare-reusable-results-checkout.sh "
            f"{shlex.quote(mode)} {shlex.quote(expected_repository)}\n"
        )
        return subprocess.run(
            [bash],
            input=script.encode("utf-8"),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_self_hosted_runtime_is_reusable_and_manually_wipeable(self):
        python_action = PYTHON_ACTION.read_text(encoding="utf-8")
        self.assertIn('runtime_root="/tmp/.ar/.hth-runtime"', python_action)
        self.assertIn('Reusing verified Python environment', python_action)
        self.assertIn('HTH_VENV_REUSED', python_action)
        self.assertIn('PIP_DISABLE_PIP_VERSION_CHECK=1', python_action)
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("clean_runner:", text, workflow.name)
            self.assertIn('description: "Advanced — Wipe runner workspace and rebuild from scratch"', text, workflow.name)
            self.assertIn("- name: Wipe runner workspace", text, workflow.name)
            self.assertIn('find "$GITHUB_WORKSPACE" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +', text, workflow.name)
            wipe = text.index("- name: Wipe runner workspace")
            checkout = text.index("- name: Checkout HTH pipeline", wipe)
            self.assertLess(wipe, checkout, workflow.name)
            self.assertIn('rm -rf "/tmp/.ar/.hth-runtime"', text, workflow.name)
            self.assertIn("uses: ./hth-pipeline/.github/actions/setup-hth-python", text, workflow.name)

    def test_reusable_results_checkout_is_validated_around_checkout_action(self):
        text = (ROOT / ".github/workflows/regress-detector.yml").read_text(encoding="utf-8")
        self.assertEqual(text.count("- name: Prepare reusable results checkout"), 3)
        self.assertEqual(text.count("- name: Verify reusable results checkout"), 3)
        for match in [
            pos for pos in range(len(text))
            if text.startswith("- name: Prepare reusable results checkout", pos)
        ]:
            checkout = text.index("- name: Checkout results repository", match)
            verify = text.index("- name: Verify reusable results checkout", checkout)
            self.assertLess(match, checkout)
            self.assertLess(checkout, verify)

        optimizer = (ROOT / ".github/workflows/execution-optimizer.yml").read_text(encoding="utf-8")
        self.assertEqual(optimizer.count("- name: Prepare reusable results checkout"), 1)
        self.assertEqual(optimizer.count("- name: Verify reusable results checkout"), 1)

        helper = RESULTS_CHECKOUT_PREP.read_text(encoding="utf-8")
        self.assertIn("rev-parse --verify 'HEAD^{commit}'", helper)
        self.assertIn("remote get-url origin", helper)
        self.assertIn('value="${value#*@github.com/}"', helper)
        self.assertIn('canonical_origin="https://github.com/${repository_slug}.git"', helper)
        self.assertIn('remote set-url origin "$canonical_origin"', helper)
        self.assertIn('git -C "$target" reset --hard HEAD', helper)
        self.assertIn('git -C "$target" clean -ffd', helper)
        self.assertIn('rm -rf -- "$target"', helper)
        self.assertNotIn(".hth-runtime", helper)

    def test_reusable_results_checkout_accepts_and_sanitizes_authenticated_origin(self):
        with self._workspace() as workspace:
            checkout = self._init_checkout(
                workspace,
                origin="https://x-access-token:secret-value@github.com/owner/results.git",
            )
            completed = self._checkout_helper(workspace)
            stdout = completed.stdout.decode("utf-8", errors="replace")
            stderr = completed.stderr.decode("utf-8", errors="replace")

            self.assertEqual(
                completed.returncode,
                0,
                f"stdout:\n{stdout}\nstderr:\n{stderr}",
            )
            self.assertNotIn("::warning::", stdout)
            self.assertIn("validated and cleaned", stdout)
            origin = self._git("-C", str(checkout), "remote", "get-url", "origin").stdout.strip()
            self.assertEqual(origin, "https://github.com/owner/results.git")

    def test_reusable_results_checkout_cleans_dirty_valid_checkout(self):
        with self._workspace() as workspace:
            checkout = self._init_checkout(workspace)
            (checkout / "tracked.txt").write_text("dirty\n", encoding="utf-8")
            (checkout / "untracked.txt").write_text("remove me\n", encoding="utf-8")

            completed = self._checkout_helper(workspace)
            stdout = completed.stdout.decode("utf-8", errors="replace")

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn("::warning::", stdout)
            self.assertEqual((checkout / "tracked.txt").read_text(encoding="utf-8"), "committed\n")
            self.assertFalse((checkout / "untracked.txt").exists())

    def test_invalid_reusable_checkout_states_are_recreated_in_scope(self):
        for state in ("partial", "unborn", "corrupt-head", "missing-origin", "wrong-origin"):
            with self.subTest(state=state), self._workspace() as workspace:
                sibling = workspace / "keep.txt"
                sibling.write_text("preserve\n", encoding="utf-8")
                checkout = workspace / "results-repo"
                if state == "partial":
                    (checkout / ".git").mkdir(parents=True)
                elif state == "unborn":
                    checkout.mkdir()
                    self._git("init", str(checkout))
                    self._git(
                        "-C", str(checkout), "remote", "add", "origin",
                        "https://github.com/owner/results.git",
                    )
                else:
                    checkout = self._init_checkout(workspace)
                    if state == "corrupt-head":
                        (checkout / ".git" / "HEAD").write_text(
                            "ref: refs/heads/missing\n", encoding="utf-8",
                        )
                    elif state == "missing-origin":
                        self._git("-C", str(checkout), "remote", "remove", "origin")
                    elif state == "wrong-origin":
                        self._git(
                            "-C", str(checkout), "remote", "set-url", "origin",
                            "https://github.com/other/results.git",
                        )

                completed = self._checkout_helper(workspace)
                stdout = completed.stdout.decode("utf-8", errors="replace")

                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("::warning::Reusable results checkout is invalid", stdout)
                self.assertFalse(checkout.exists())
                self.assertEqual(sibling.read_text(encoding="utf-8"), "preserve\n")

    def test_absent_reusable_checkout_is_left_for_checkout_action(self):
        with self._workspace() as workspace:
            completed = self._checkout_helper(workspace)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, b"")
            self.assertFalse((workspace / "results-repo").exists())

    def test_diverged_but_valid_checkout_is_reused_before_forced_checkout(self):
        with self._workspace() as workspace:
            checkout = self._init_checkout(workspace)
            self._git("-C", str(checkout), "update-ref", "refs/remotes/origin/main", "HEAD")
            (checkout / "local-history.txt").write_text("replacement history\n", encoding="utf-8")
            self._git("-C", str(checkout), "add", "local-history.txt")
            self._git("-C", str(checkout), "commit", "-m", "locally diverged history")

            completed = self._checkout_helper(workspace)
            stdout = completed.stdout.decode("utf-8", errors="replace")

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn("::warning::", stdout)
            self.assertTrue(checkout.is_dir())
            self.assertIn("validated and cleaned", stdout)

    def test_verify_rejects_dirty_checkout_without_deleting_it(self):
        with self._workspace() as workspace:
            checkout = self._init_checkout(workspace)
            (checkout / "untracked.txt").write_text("dirty\n", encoding="utf-8")

            completed = self._checkout_helper(workspace, mode="verify")
            stderr = completed.stderr.decode("utf-8", errors="replace")

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not clean immediately after checkout", stderr)
            self.assertTrue(checkout.is_dir())

    def test_runtime_is_built_once_then_specialized_steps_only_verify(self):
        manager = RUNTIME_MANAGER.read_text(encoding="utf-8")
        action = MANAGED_ACTION.read_text(encoding="utf-8")
        self.assertIn("- name: Verify / Install complete managed runtime", action)
        self.assertIn("tools/ensure-managed-runtime.sh", action)
        self.assertIn("- name: Verify dhSegment TensorFlow runtime", action)
        self.assertIn("- name: Verify Kraken historical-document segmentation runtime", action)
        self.assertNotIn("- name: Verify / Install dhSegment TensorFlow runtime", action)
        self.assertNotIn("- name: Verify / Install Kraken historical-document segmentation runtime", action)
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("uses: ./hth-pipeline/.github/actions/setup-hth-managed-runtime", text, workflow.name)
        self.assertIn("Managed runtime verified — using previous install; no install required.", manager)
        self.assertIn("Managed base runtime verified; augmenting only missing optional runtime layer(s).", manager)

    def test_runtime_verification_checks_target_versions_and_imports(self):
        manager = RUNTIME_MANAGER.read_text(encoding="utf-8")
        self.assertIn('Requirement(line)', manager)
        self.assertIn('SpecifierSet(">=2.18,<2.21")', manager)
        self.assertIn('import tensorflow as tf', manager)
        self.assertIn('version != "7.0.2"', manager)
        self.assertIn('from kraken.tasks.segmentation import SegmentationTaskModel', manager)
        self.assertIn('except metadata.PackageNotFoundError:', manager)
        self.assertIn("python -m pip check", manager)

    def test_github_hosted_stays_run_local(self):
        action = PYTHON_ACTION.read_text(encoding="utf-8")
        self.assertIn('venv_dir="$RUNNER_TEMP/hth-python-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"', action)
        self.assertIn('if [[ "${{ runner.environment }}" == "self-hosted" ]]; then', action)


if __name__ == "__main__":
    unittest.main()
