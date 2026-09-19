import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools" / "hardened-persistence.sh"
WORKFLOWS = ROOT / ".github" / "workflows"


@dataclass(frozen=True)
class PersistenceRepositories:
    root: Path
    remote: Path
    writer: Path
    runner_temp: Path
    racer: Path | None = None

    def bash_path(self, path: Path) -> str:
        return path.relative_to(ROOT).as_posix()


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


class HardenedPersistenceTests(unittest.TestCase):
    @staticmethod
    def _remove_readonly_and_retry(function, path, _error) -> None:
        os.chmod(path, stat.S_IWRITE)
        function(path)

    def _run_git(self, *args: str) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["git", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            self.fail(
                f"Git fixture command failed ({completed.returncode}): git {' '.join(args)}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return completed

    def _configure_checkout(self, checkout: Path) -> None:
        self._run_git("-C", str(checkout), "config", "user.name", "test")
        self._run_git("-C", str(checkout), "config", "user.email", "test@example.com")
        # Native Windows Git records a local clone origin as C:\\..., which
        # bash-invoked Git can misread as scp syntax ("host c"). This relative
        # URL works identically with native Git, Git Bash, and POSIX bash.
        self._run_git("-C", str(checkout), "remote", "set-url", "origin", "../remote.git")

    @contextmanager
    def _persistence_repositories(self, *, include_racer: bool = False):
        """Create a disposable local origin and configured writer checkouts."""
        scratch_root = ROOT / ".test-hardened-persistence"
        scratch_root.mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(dir=scratch_root))
        remote = root / "remote.git"
        seed = root / "seed"
        writer = root / "writer"
        racer = root / "racer" if include_racer else None
        runner_temp = root / "runner-temp"
        runner_temp.mkdir()

        try:
            self._run_git("init", "--bare", "--initial-branch=main", str(remote))
            self._run_git("clone", str(remote), str(seed))
            self._run_git("-C", str(seed), "config", "user.name", "test")
            self._run_git("-C", str(seed), "config", "user.email", "test@example.com")
            (seed / "base.txt").write_text("base\n", encoding="utf-8")
            self._run_git("-C", str(seed), "add", "base.txt")
            self._run_git("-C", str(seed), "commit", "-m", "seed")
            self._run_git("-C", str(seed), "push", "origin", "main")
            self._run_git("--git-dir", str(remote), "symbolic-ref", "HEAD", "refs/heads/main")

            self._run_git("clone", str(remote), str(writer))
            self._configure_checkout(writer)
            if racer is not None:
                self._run_git("clone", str(remote), str(racer))
                self._configure_checkout(racer)

            yield PersistenceRepositories(
                root=root,
                remote=remote,
                writer=writer,
                racer=racer,
                runner_temp=runner_temp,
            )
        finally:
            # Git Bash can retain a checkout handle briefly after it exits.
            for attempt in range(20):
                try:
                    shutil.rmtree(root, onexc=self._remove_readonly_and_retry)
                    break
                except PermissionError:
                    if attempt == 19:
                        self.fail(f"Git Bash did not release persistence fixture: {root}")
                    time.sleep(0.1)
            try:
                scratch_root.rmdir()
            except OSError:
                pass

    def _run_bash(self, script: str) -> subprocess.CompletedProcess[bytes]:
        # Bytes preserve LF on Windows; text-mode stdin would turn the first
        # line into `set -euo pipefail\r`, which bash rejects.
        bash = _bash_executable()
        self.assertIsNotNone(bash, "A POSIX bash executable is required")
        return subprocess.run(
            [bash],
            input=script.encode("utf-8"),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_helper_has_calibration_grade_collision_contract(self):
        text = HELPER.read_text(encoding="utf-8")
        self.assertIn("git -C \"$repo\" fetch origin \"$branch\"", text)
        self.assertIn("git -C \"$repo\" reset --hard \"origin/$branch\"", text)
        self.assertIn("max_attempts=\"${HTH_PERSIST_MAX_ATTEMPTS:-5}\"", text)
        self.assertIn("backoff_seconds=\"${HTH_PERSIST_BACKOFF_SECONDS:-5}\"", text)
        self.assertIn("non-fast-forward|fetch first|failed to push some refs", text)
        self.assertIn("refusing to misclassify and retry it", text)

    def test_canonical_staging_is_not_blocked_by_sparse_checkout(self):
        with self._persistence_repositories() as repos:
            writer_sh = repos.bash_path(repos.writer)
            self._run_git("-C", str(repos.writer), "sparse-checkout", "init", "--no-cone")
            self._run_git("-C", str(repos.writer), "sparse-checkout", "set", "--no-cone", "/base.txt")

            script = (
                "set -euo pipefail\n"
                "source tools/hardened-persistence.sh\n"
                f"mkdir -p \"{writer_sh}/metadata\"\n"
                f"printf '{{\"state\":\"canonical\"}}\\n' > \"{writer_sh}/metadata/resource-lifecycle.json\"\n"
                f"hth_results_stage \"{writer_sh}\" metadata/resource-lifecycle.json\n"
            )

            proc = self._run_bash(script)
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            self.assertEqual(proc.returncode, 0, stderr or stdout)
            staged = self._run_git(
                "-C", str(repos.writer), "diff", "--cached", "--name-only",
            ).stdout.splitlines()
            self.assertEqual(staged, ["metadata/resource-lifecycle.json"])
            sparse_patterns = self._run_git(
                "-C", str(repos.writer), "sparse-checkout", "list",
            ).stdout.splitlines()
            self.assertEqual(sparse_patterns, ["/base.txt"])

    def test_canonical_staging_requires_explicit_owned_paths(self):
        with self._persistence_repositories() as repos:
            writer_sh = repos.bash_path(repos.writer)
            script = (
                "set -uo pipefail\n"
                "source tools/hardened-persistence.sh\n"
                "status=0\n"
                f"hth_results_stage \"{writer_sh}\" || status=$?\n"
                "printf 'status=%s\\n' \"$status\"\n"
            )

            proc = self._run_bash(script)
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            self.assertEqual(proc.returncode, 0, stderr or stdout)
            self.assertIn("requires at least one owned repository-relative path", stdout)
            self.assertIn("status=2", stdout)

    def test_real_non_fast_forward_retry_preserves_concurrent_remote_write(self):
        with self._persistence_repositories(include_racer=True) as repos:
            self.assertIsNotNone(repos.racer)
            writer_sh = repos.bash_path(repos.writer)
            racer_sh = repos.bash_path(repos.racer)
            runner_temp_sh = repos.bash_path(repos.runner_temp)

            script = (
                "set -euo pipefail\n"
                "source tools/hardened-persistence.sh\n"
                "apply_writer() {\n"
                "  local attempt=\"$1\"\n"
                "  if [[ \"$attempt\" == \"1\" ]]; then\n"
                f"    printf 'racer\\n' > \"{racer_sh}/racer.txt\"\n"
                f"    git -C \"{racer_sh}\" add racer.txt\n"
                f"    git -C \"{racer_sh}\" commit -m racer >/dev/null\n"
                f"    git -C \"{racer_sh}\" push origin main >/dev/null\n"
                "  fi\n"
                f"  printf 'writer\\n' > \"{writer_sh}/writer.txt\"\n"
                f"  git -C \"{writer_sh}\" add writer.txt\n"
                "}\n"
                f"RUNNER_TEMP=\"{runner_temp_sh}\" "
                "GITHUB_RUN_ID=persistence-race-test "
                "GITHUB_RUN_ATTEMPT=1 "
                "HTH_PERSIST_BACKOFF_SECONDS=0 "
                f"hth_hardened_persist \"{writer_sh}\" main writer apply_writer Test\n"
            )

            proc = self._run_bash(script)
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            self.assertEqual(proc.returncode, 0, stderr or stdout)

            self._run_git("-C", str(repos.writer), "fetch", "origin", "main")
            self._run_git("-C", str(repos.writer), "reset", "--hard", "origin/main")
            self.assertEqual(
                (repos.writer / "racer.txt").read_text(encoding="utf-8"), "racer\n",
            )
            self.assertEqual(
                (repos.writer / "writer.txt").read_text(encoding="utf-8"), "writer\n",
            )

    def test_permanent_push_failure_is_not_retried(self):
        with self._persistence_repositories() as repos:
            self._run_git(
                "-C", str(repos.writer), "remote", "set-url", "--push", "origin",
                "../missing-remote.git",
            )
            writer_sh = repos.bash_path(repos.writer)
            runner_temp_sh = repos.bash_path(repos.runner_temp)
            script = (
                "set -uo pipefail\n"
                "source tools/hardened-persistence.sh\n"
                "apply_writer() {\n"
                f"  printf 'writer\\n' > \"{writer_sh}/writer.txt\"\n"
                f"  git -C \"{writer_sh}\" add writer.txt\n"
                "}\n"
                "status=0\n"
                f"RUNNER_TEMP=\"{runner_temp_sh}\" "
                "GITHUB_RUN_ID=persistence-permanent-failure-test "
                "GITHUB_RUN_ATTEMPT=1 HTH_PERSIST_BACKOFF_SECONDS=0 "
                f"hth_hardened_persist \"{writer_sh}\" main writer apply_writer Test || status=$?\n"
                "printf 'status=%s attempts=%s\\n' \"$status\" \"$HTH_PERSIST_ATTEMPTS\"\n"
            )

            proc = self._run_bash(script)
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            self.assertEqual(proc.returncode, 0, stderr or stdout)
            self.assertIn("refusing to misclassify and retry it", stdout)
            self.assertIn("status=1 attempts=1", stdout)
            self.assertNotIn("persistence attempt 2/", stdout)

    def test_all_results_repo_workflow_pushes_use_shared_helper(self):
        offenders = []
        for workflow in WORKFLOWS.glob("*.yml"):
            text = workflow.read_text(encoding="utf-8")
            if re.search(r"git(?: -C results-repo)? push", text):
                offenders.append(workflow.name)
        self.assertEqual(offenders, [], f"raw workflow pushes remain: {offenders}")

    def test_all_results_repo_writers_source_shared_helper(self):
        expected = {
            "_core-hth.yml": 3,
            "regress-detector.yml": 1,
            "execution-optimizer.yml": 1,
            "rebuild-historical-regression.yml": 1,
        }
        for name, minimum in expected.items():
            text = (WORKFLOWS / name).read_text(encoding="utf-8")
            self.assertGreaterEqual(
                text.count("source hth-pipeline/tools/hardened-persistence.sh"),
                minimum,
                name,
            )
            self.assertGreaterEqual(text.count("hth_hardened_persist"), minimum, name)


if __name__ == "__main__":
    unittest.main()
