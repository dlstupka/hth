import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools" / "hardened-persistence.sh"
WORKFLOWS = ROOT / ".github" / "workflows"


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
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            remote = root / "remote.git"
            seed = root / "seed"
            writer = root / "writer"
            racer = root / "racer"
            subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(remote)], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "clone", str(remote), str(seed)], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "-C", str(seed), "config", "user.name", "test"], check=True)
            subprocess.run(["git", "-C", str(seed), "config", "user.email", "test@example.com"], check=True)
            (seed / "base.txt").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(seed), "add", "base.txt"], check=True)
            subprocess.run(["git", "-C", str(seed), "commit", "-m", "seed"], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "-C", str(seed), "branch", "-M", "main"], check=True)
            subprocess.run(["git", "-C", str(seed), "push", "origin", "main"], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "--git-dir", str(remote), "symbolic-ref", "HEAD", "refs/heads/main"], check=True)
            subprocess.run(["git", "clone", str(remote), str(writer)], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "clone", str(remote), str(racer)], check=True, stdout=subprocess.DEVNULL)
            for checkout in (writer, racer):
                subprocess.run(["git", "-C", str(checkout), "config", "user.name", "test"], check=True)
                subprocess.run(["git", "-C", str(checkout), "config", "user.email", "test@example.com"], check=True)

            script = root / "race.sh"
            script.write_text(
                "set -euo pipefail\n"
                f"source \"{HELPER}\"\n"
                "apply_writer() {\n"
                "  local attempt=\"$1\"\n"
                "  if [[ \"$attempt\" == \"1\" ]]; then\n"
                f"    printf 'racer\\n' > \"{racer}/racer.txt\"\n"
                f"    git -C \"{racer}\" add racer.txt\n"
                f"    git -C \"{racer}\" commit -m racer >/dev/null\n"
                f"    git -C \"{racer}\" push origin main >/dev/null\n"
                "  fi\n"
                f"  printf 'writer\\n' > \"{writer}/writer.txt\"\n"
                f"  git -C \"{writer}\" add writer.txt\n"
                "}\n"
                f"HTH_PERSIST_BACKOFF_SECONDS=0 hth_hardened_persist \"{writer}\" main writer apply_writer Test\n",
                encoding="utf-8",
            )
            subprocess.run(["bash", str(script)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            subprocess.run(["git", "-C", str(writer), "fetch", "origin", "main"], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "-C", str(writer), "reset", "--hard", "origin/main"], check=True, stdout=subprocess.DEVNULL)
            self.assertEqual((writer / "racer.txt").read_text(encoding="utf-8"), "racer\n")
            self.assertEqual((writer / "writer.txt").read_text(encoding="utf-8"), "writer\n")

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
