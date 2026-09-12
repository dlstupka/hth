import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools" / "hardened-persistence.sh"
WORKFLOWS = ROOT / ".github" / "workflows"


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
    def test_helper_has_calibration_grade_collision_contract(self):
        text = HELPER.read_text(encoding="utf-8")
        self.assertIn("git -C \"$repo\" fetch origin \"$branch\"", text)
        self.assertIn("git -C \"$repo\" reset --hard \"origin/$branch\"", text)
        self.assertIn("max_attempts=\"${HTH_PERSIST_MAX_ATTEMPTS:-5}\"", text)
        self.assertIn("backoff_seconds=\"${HTH_PERSIST_BACKOFF_SECONDS:-5}\"", text)
        self.assertIn("non-fast-forward|fetch first|failed to push some refs", text)
        self.assertIn("refusing to misclassify and retry it", text)

    def test_real_non_fast_forward_retry_preserves_concurrent_remote_write(self):
        def run_quiet(args, **kwargs):
            return subprocess.run(
                args,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **kwargs,
            )

        # Keep the race fixture beneath the repository and use only paths
        # relative to ROOT inside bash.  That works with Git Bash, MSYS, WSL,
        # and POSIX bash without trying to translate Windows paths.
        scratch_root = ROOT / ".test-hardened-persistence"
        scratch_root.mkdir(exist_ok=True)
        # MSYS can release the final Git checkout handle a fraction after bash
        # exits. Let the context preserve a transiently locked tree on Windows,
        # then retry that exact test directory below instead of making a passed
        # persistence race fail during tempfile cleanup.
        with tempfile.TemporaryDirectory(
            dir=scratch_root, ignore_cleanup_errors=os.name == "nt",
        ) as td:
            root = Path(td)
            remote = root / "remote.git"
            seed = root / "seed"
            writer = root / "writer"
            racer = root / "racer"
            runner_temp = root / "runner-temp"
            runner_temp.mkdir()

            run_quiet(["git", "init", "--bare", "--initial-branch=main", str(remote)])
            run_quiet(["git", "clone", str(remote), str(seed)])
            run_quiet(["git", "-C", str(seed), "config", "user.name", "test"])
            run_quiet(["git", "-C", str(seed), "config", "user.email", "test@example.com"])
            (seed / "base.txt").write_text("base\n", encoding="utf-8")
            run_quiet(["git", "-C", str(seed), "add", "base.txt"])
            run_quiet(["git", "-C", str(seed), "commit", "-m", "seed"])
            run_quiet(["git", "-C", str(seed), "branch", "-M", "main"])
            run_quiet(["git", "-C", str(seed), "push", "origin", "main"])
            run_quiet(["git", "--git-dir", str(remote), "symbolic-ref", "HEAD", "refs/heads/main"])
            run_quiet(["git", "clone", str(remote), str(writer)])
            run_quiet(["git", "clone", str(remote), str(racer)])
            # Native Windows git records a local clone origin as C:\\..., which
            # bash-invoked git can misread as scp syntax ("host c"). Keep the
            # temporary bare remote as a simple path relative to each checkout;
            # ../remote.git is understood identically by native git and bash git
            # on Windows and POSIX.
            for checkout in (writer, racer):
                run_quiet(["git", "-C", str(checkout), "config", "user.name", "test"])
                run_quiet(["git", "-C", str(checkout), "config", "user.email", "test@example.com"])
                run_quiet(
                    ["git", "-C", str(checkout), "remote", "set-url", "origin", "../remote.git"]
                )

            rel_root = root.relative_to(ROOT).as_posix()
            writer_sh = f"{rel_root}/writer"
            racer_sh = f"{rel_root}/racer"
            runner_temp_sh = f"{rel_root}/runner-temp"

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

            # Feed bytes, not text, so Windows Python cannot translate the
            # script's LF newlines to CRLF while writing subprocess stdin.
            # Bash otherwise sees "pipefail\r" and rejects the first line.
            bash = _bash_executable()
            self.assertIsNotNone(bash, "A POSIX bash executable is required")
            proc = subprocess.run(
                [bash],
                input=script.encode("utf-8"),
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            self.assertEqual(proc.returncode, 0, stderr or stdout)

            run_quiet(["git", "-C", str(writer), "fetch", "origin", "main"])
            run_quiet(["git", "-C", str(writer), "reset", "--hard", "origin/main"])
            self.assertEqual((writer / "racer.txt").read_text(encoding="utf-8"), "racer\n")
            self.assertEqual((writer / "writer.txt").read_text(encoding="utf-8"), "writer\n")

        if root.exists():
            for attempt in range(20):
                try:
                    shutil.rmtree(root)
                    break
                except PermissionError:
                    if attempt == 19:
                        self.fail(f"Git Bash did not release race fixture: {root}")
                    time.sleep(0.1)

        try:
            scratch_root.rmdir()
        except OSError:
            pass

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
