import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


class WorkflowResearchArtifactTests(unittest.TestCase):
    def test_all_checkout_workflows_apply_main_before_git_init(self) -> None:
        for path in WORKFLOWS.glob("*.yml"):
            text = path.read_text(encoding="utf-8")
            if "uses: actions/checkout@" not in text:
                continue
            with self.subTest(workflow=path.name):
                self.assertIn('GIT_CONFIG_COUNT: "1"', text)
                self.assertIn("GIT_CONFIG_KEY_0: init.defaultBranch", text)
                self.assertIn("GIT_CONFIG_VALUE_0: main", text)

    def test_report_writer_uploads_research_bundle_without_detector_verbose_trees(self) -> None:
        text = (WORKFLOWS / "_core-hth.yml").read_text(encoding="utf-8")
        self.assertIn("Assemble report research artifact", text)
        self.assertIn("Upload report research artifact", text)
        self.assertIn("results-repo/*-index.json", text)
        self.assertIn("optimizer-predictions.json", text)
        self.assertIn(
            "cp -a results-repo/indexes report-research-artifact/intelligence/indexes",
            text,
        )
        self.assertIn("for directory in reports execution-optimizer", text)
        assemble = text.split("- name: Assemble report research artifact", 1)[1].split(
            "- name: Upload report research artifact", 1
        )[0]
        copy_loop = assemble.split("for directory in reports execution-optimizer", 1)[1].split("done", 1)[0]
        self.assertNotIn("source-documents", copy_loop)
        self.assertIn('FREEZE_SOURCE="${{ steps.report_golden_set.outputs.freeze_path }}"', text)
        self.assertIn('cp "$FREEZE_SOURCE" "report-research-artifact/golden-set/${GOLDEN_ID}.freeze.json"', text)
        self.assertIn("selected-golden-set.json", text)
        self.assertIn("durable calibration records and raw evidence remain linked", text)


if __name__ == "__main__":
    unittest.main()
