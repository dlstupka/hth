import unittest
from pathlib import Path


class TopParameterReferenceReportingTests(unittest.TestCase):
    def test_manifest_has_reference_rows_and_last_build_column(self):
        text = Path("hth/write_regression_summary.py").read_text(encoding="utf-8")
        self.assertIn("| Rank | Last Build | Family ID | Parameter Set ID", text)
        self.assertIn('("Baseline*", baseline)', text)
        self.assertIn('("Best**", historic_best)', text)
        self.assertIn("detector's default parameter-set configuration", text)
        self.assertIn("historic best-known compatible parameter set prior to this regression run", text)
        self.assertIn("_last_build_for_parameter", text)
        self.assertIn("most recent known prior build", text)
        self.assertIn("return \"new\"", text)
        self.assertIn("`new` means this run is the first known evaluation", text)

    def test_runner_persists_reference_and_search_views(self):
        text = Path("hth/regression/runner.py").read_text(encoding="utf-8")
        materialization = Path("hth/regression/materialization.py").read_text(encoding="utf-8")
        self.assertIn('baseline_result["reference_roles"] = ["baseline"]', text)
        self.assertIn('historic_best_result["reference_roles"] = ["historic_best"]', text)
        self.assertIn('"historic_best": outcome.historic_best', materialization)
        self.assertIn('"search_top_parameter_sets": outcome.search_ranked[:5]', materialization)


if __name__ == "__main__":
    unittest.main()
