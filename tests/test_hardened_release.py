from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _bash_executable() -> str | None:
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            candidate = Path(git).resolve().parent.parent / "bin" / "bash.exe"
            if candidate.is_file():
                return str(candidate)
    return shutil.which("bash")


class HardenedReleaseTests(unittest.TestCase):
    def _run(self, *, create_status: int, digest: str) -> subprocess.CompletedProcess[bytes]:
        bash = _bash_executable()
        self.assertIsNotNone(bash)
        scratch = ROOT / ".test-hardened-release"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            state = root / "state"
            fake_bin.mkdir()
            state.mkdir()
            gh = fake_bin / "gh"
            gh.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2" == "release view" ]]; then
  [[ -f "$FAKE_STATE/release" ]]
elif [[ "$1 $2" == "release create" ]]; then
  touch "$FAKE_STATE/release"
  printf '%s' "$FAKE_DIGEST" > "$FAKE_STATE/digest"
  exit "$FAKE_CREATE_STATUS"
elif [[ "$1" == api ]]; then
  cat "$FAKE_STATE/digest"
else
  exit 2
fi
""",
                encoding="utf-8",
                newline="\n",
            )
            gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
            asset = root / "asset.zip"
            asset.write_bytes(b"deterministic")
            relative_root = root.relative_to(ROOT).as_posix()
            script = f"""
set -euo pipefail
export PATH="{relative_root}/bin:$PATH"
export FAKE_STATE="{relative_root}/state"
export FAKE_CREATE_STATUS={create_status}
export FAKE_DIGEST={digest}
source tools/hardened-release.sh
hth_publish_immutable_release owner/results TAG "{relative_root}/asset.zip" asset.zip {'a' * 64} title notes
printf '%s' "$HTH_RELEASE_ACTIVITY"
"""
            completed = subprocess.run(
                [bash], input=script.encode("utf-8"), cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        try:
            scratch.rmdir()
        except OSError:
            pass
        return completed

    def test_release_creation_verifies_digest(self) -> None:
        completed = self._run(create_status=0, digest=f"sha256:{'a' * 64}")
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertTrue(completed.stdout.decode().endswith("CREATED"))

    def test_concurrent_creator_is_reused_only_after_digest_verification(self) -> None:
        completed = self._run(create_status=1, digest=f"sha256:{'a' * 64}")
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertTrue(completed.stdout.decode().endswith("REUSED"))

    def test_release_collision_fails_closed(self) -> None:
        completed = self._run(create_status=0, digest=f"sha256:{'b' * 64}")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Immutable release collision", completed.stdout.decode())


if __name__ == "__main__":
    unittest.main()
