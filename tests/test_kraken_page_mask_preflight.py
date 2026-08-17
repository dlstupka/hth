import os
import unittest
from pathlib import Path
from unittest.mock import patch

from hth import kraken_page_mask_preflight as preflight


class KrakenPageMaskPreflightTests(unittest.TestCase):
    def test_missing_worker_model_environment_is_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HTH_KRAKEN_PAGE_MODEL is not set"):
                preflight.run(load_model=False)

    def test_shell_runs_preflight_after_lifecycle_source_and_enables_faulthandler(self):
        text = Path("tools/run-detector-regressions.sh").read_text(encoding="utf-8")
        source_pos = text.index('source "$lifecycle_env"')
        preflight_pos = text.index("python -m hth.kraken_page_mask_preflight")
        worker_pos = text.index('PYTHONFAULTHANDLER=1 \\\n    "${args[@]}"')
        self.assertLess(source_pos, preflight_pos)
        self.assertLess(preflight_pos, worker_pos)

    def test_missing_shard_directory_does_not_emit_secondary_find_error(self):
        text = Path("tools/run-detector-regressions.sh").read_text(encoding="utf-8")
        self.assertIn(
            'find "$shard_root" -mindepth 1 -maxdepth 1 -type d -name \'run-*\' 2>/dev/null',
            text,
        )


if __name__ == "__main__":
    unittest.main()
