from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from hth.regression.runner import file_sha256, print_environment_banner


class RegressionProvenanceTests(unittest.TestCase):
    def test_file_sha256_hashes_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "golden.json"
            source.write_bytes(b"golden-set\n")
            self.assertEqual(file_sha256(source), hashlib.sha256(b"golden-set\n").hexdigest())

    def test_environment_banner_prints_golden_set_hash(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            print_environment_banner(
                environment={},
                detector="hough",
                golden_set=Path("config/golden_set.json"),
                golden_set_sha256="abc123",
            )
        text = stream.getvalue()
        self.assertIn("Golden Set            : config/golden_set.json", text)
        self.assertIn("Golden Set SHA-256    : abc123", text)


if __name__ == "__main__":
    unittest.main()
