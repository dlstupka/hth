from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.assess_perspective import estimate_line_convergence, recommend
from hth.canonical_build_evidence import canonical_hash


class PerspectiveAssessmentTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict[str, object]:
        root = Path(__file__).resolve().parents[1]
        return json.loads((root / "config/perspective-assessment.json").read_text(encoding="utf-8"))

    @staticmethod
    def _grid() -> np.ndarray:
        image = np.full((800, 600, 3), 255, dtype=np.uint8)
        for y in range(80, 760, 60):
            cv2.line(image, (40, y), (560, y), (0, 0, 0), 3)
        for x in range(60, 560, 60):
            cv2.line(image, (x, 40), (x, 760), (0, 0, 0), 3)
        return image

    def test_flat_grid_is_preserved(self) -> None:
        result = estimate_line_convergence(self._grid(), self._config())
        self.assertEqual(result["decision"], "preserve")
        self.assertLessEqual(result["maximum_convergence_degrees"], 0.35)

    def test_two_family_projective_convergence_is_candidate(self) -> None:
        image = self._grid()
        source = np.float32([[0, 0], [599, 0], [599, 799], [0, 799]])
        target = np.float32([[80, 40], [520, 0], [590, 760], [10, 799]])
        warped = cv2.warpPerspective(
            image,
            cv2.getPerspectiveTransform(source, target),
            (600, 800),
            borderValue=(255, 255, 255),
        )
        result = estimate_line_convergence(warped, self._config())
        self.assertEqual(result["decision"], "correction-candidate")
        self.assertEqual(result["confirming_family_count"], 2)

    def test_recommendation_requires_untampered_evidence(self) -> None:
        config = self._config()
        assessment = {
            "schema_version": "1.0",
            "assessment_type": "perspective-convergence-comparison",
            "status": "diagnostic-only",
            "canonical_normalization_result_identity": "a" * 64,
            "sample_identity": "b" * 64,
            "sample_page_count": 80,
            "config": config,
            "aggregate": {
                "preserve_pages": 70,
                "correction_candidate_pages": 0,
                "review_pages": 8,
                "inconclusive_pages": 2,
                "maximum_convergence_degrees": 0.7,
            },
            "pages": [],
        }
        assessment["assessment_identity"] = canonical_hash({
            key: assessment[key]
            for key in (
                "assessment_type",
                "canonical_normalization_result_identity",
                "sample_identity",
                "config",
                "pages",
            )
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "assessment.json"
            path.write_text(json.dumps(assessment), encoding="utf-8")
            policy = recommend(path, root / "perspective-policy.json")
            self.assertEqual(policy["status"], "skip-recommended")
            assessment["pages"] = [{"global_ordinal": 1}]
            path.write_text(json.dumps(assessment), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity"):
                recommend(path, root / "tampered.json")

    def test_workflow_is_diagnostic_and_persists_policy(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/assess-perspective.yml").read_text(encoding="utf-8")
        self.assertIn("python -m hth.assess_perspective evaluate", workflow)
        self.assertIn("python -m hth.assess_perspective recommend", workflow)
        self.assertIn("normalization/perspective-policy.json", workflow)
        sparse_checkout = workflow.split("sparse-checkout: |", 1)[1].split("sparse-checkout-cone-mode:", 1)[0]
        self.assertIn("/normalization/perspective/", sparse_checkout)
        self.assertIn("/normalization/perspective-policy.json", sparse_checkout)
        self.assertIn("runner_target:", workflow)
        self.assertNotIn("specific_runner:", workflow)
        self.assertNotIn("normalize_document_images", workflow)
        self.assertNotIn("git push", workflow)


if __name__ == "__main__":
    unittest.main()
