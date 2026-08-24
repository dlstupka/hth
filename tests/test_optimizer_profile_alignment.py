import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OptimizerProfileAlignmentTests(unittest.TestCase):
    def test_published_single_run_heatmap_uses_current_execution(self):
        text = (ROOT / "hth" / "optimizer_store.py").read_text(encoding="utf-8")
        self.assertIn(
            'svg_path.write_text(render_heatmap_svg(current), encoding="utf-8")',
            text,
        )
        self.assertNotIn(
            'svg_path.write_text(render_heatmap_svg(compatible_historical), encoding="utf-8")',
            text,
        )


if __name__ == "__main__":
    unittest.main()
