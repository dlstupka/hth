from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from hth.geometry import (
    detector_docextractor_page_mask,
    detector_eynollah_page_mask,
    detector_pagenet_page_mask,
)
from hth.geometry.model import Candidate
from hth.geometry.registry import detector_names
from hth.regression import learned_evidence, runner
from hth.regression.strategies.cartesian import generate


ROOT = Path(__file__).resolve().parents[1]


class HistoricalNeuralDetectorIntegrationTests(unittest.TestCase):
    def test_all_three_detectors_are_registered_and_configured(self):
        registered = set(detector_names())
        for detector in ("eynollah_page_mask", "pagenet_page_mask", "docextractor_page_mask"):
            self.assertIn(detector, registered)
            config = json.loads((ROOT / "config" / "detectors" / f"{detector}.json").read_text(encoding="utf-8"))
            self.assertEqual(config["detector"], detector)
            self.assertEqual(config["lifecycle"]["prepare"], detector)
            self.assertEqual(config["lifecycle"]["finalize"], detector)
            self.assertGreater(len(generate(config)), 1)

    def test_evidence_backed_detectors_use_parent_shared_evidence(self):
        for detector, module in (
            ("eynollah_page_mask", detector_eynollah_page_mask),
            ("docextractor_page_mask", detector_docextractor_page_mask),
            ("pagenet_page_mask", detector_pagenet_page_mask),
        ):
            self.assertIs(runner.PRECOMPUTED_EVIDENCE_PREPARERS[detector], module.precompute_golden_set_evidence)
            self.assertIs(runner.PRECOMPUTED_EVIDENCE_LOADERS[detector], module.load_precomputed_golden_set_evidence)
            self.assertIs(learned_evidence.EXPORTERS[detector], module.export_precomputed_golden_set_evidence)

    def test_evidence_mask_geometry_is_parameterized_without_reinference(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        probability = np.zeros((100, 150), dtype=np.float32)
        probability[10:90, 15:135] = 0.9
        with mock.patch.object(detector_eynollah_page_mask, "_infer", return_value=probability), \
             mock.patch.object(detector_eynollah_page_mask, "_provenance", return_value={"model_id": "test", "model_repository": "test"}):
            candidate = detector_eynollah_page_mask.detect(
                image_bgr=image,
                mask=None,
                parameters={"probability_threshold": 0.5, "minimum_page_area_fraction": 0.08, "close_kernel_fraction": 0.0, "page_padding_fraction": 0.0},
            )
        self.assertEqual(candidate.method, "eynollah_page_mask")
        self.assertEqual(candidate.status, "ok")
        self.assertGreater(candidate.bbox[2] - candidate.bbox[0], 200)
        self.assertGreater(candidate.bbox[3] - candidate.bbox[1], 140)

    def test_pagenet_explicit_identity_uses_cached_probability(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        probability = np.zeros((256, 256), dtype=np.float32)
        probability[24:232, 30:226] = 0.9
        provenance = {"model_id": "pagenet-ohio", "weights_sha256": "abc", "license": "BSD", "upstream_repository": "test"}
        with mock.patch.object(detector_pagenet_page_mask, "_probability", return_value=(probability, provenance)):
            candidate = detector_pagenet_page_mask.detect(
                image_bgr=image,
                mask=None,
                parameters={
                    "mask_threshold": 0.5,
                    "minimum_mask_area_fraction": 0.15,
                    "close_kernel_fraction": 0.0,
                    "polygon_epsilon_fraction": 0.012,
                    "bbox_padding_fraction": 0.0,
                },
            )
        self.assertEqual(candidate.method, "pagenet_page_mask")
        self.assertEqual(candidate.status, "ok")
        self.assertGreater(candidate.confidence, 0.5)
        self.assertEqual(candidate.confidence, candidate.score)
        self.assertEqual(candidate.diagnostics["explicit_detector"], "pagenet_page_mask")

    def test_eynollah_refinement_focuses_active_basin_and_preserves_zombie_audit(self):
        config = json.loads((ROOT / "config/detectors/eynollah_page_mask.json").read_text(encoding="utf-8"))
        self.assertEqual(len(generate(config)), 81)
        self.assertEqual(config["parameters"]["probability_threshold"]["values"], [0.245, 0.25, 0.255, 0.26, 0.2625, 0.265, 0.27, 0.275, 0.28])
        self.assertEqual(config["parameters"]["page_padding_fraction"]["values"], [0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01, 0.0105, 0.011, 0.0115])
        self.assertEqual(config["profiles"]["baseline"]["probability_threshold"], 0.2625)
        self.assertEqual(config["profiles"]["baseline"]["page_padding_fraction"], 0.0095)
        self.assertEqual(config["zombie_parameters"]["close_kernel_fraction"]["pinned_value"], 0.0)
        self.assertEqual(config["zombie_parameters"]["minimum_page_area_fraction"]["pinned_value"], 0.02)
        self.assertEqual(set(config["equivalence_parameters"]), {"close_kernel_fraction", "minimum_page_area_fraction"})

    def test_pagenet_enrolls_measured_minimum_area_zombie(self):
        config = json.loads((ROOT / "config/detectors/pagenet_page_mask.json").read_text(encoding="utf-8"))
        self.assertNotIn("minimum_mask_area_fraction", config["parameters"])
        zombie = config["zombie_parameters"]["minimum_mask_area_fraction"]
        self.assertEqual(zombie["pinned_value"], 0.04)
        self.assertEqual(zombie["classification"], "zombie")
        self.assertEqual(config["equivalence_parameters"], ["minimum_mask_area_fraction"])
        self.assertEqual(len(generate(config)), 25000)

    def test_docextractor_refinement_focuses_critical_interaction_basin(self):
        config = json.loads((ROOT / "config/detectors/docextractor_page_mask.json").read_text(encoding="utf-8"))
        self.assertEqual(len(generate(config)), 100)
        self.assertEqual(set(config["parameters"]), {"probability_threshold", "close_kernel_fraction", "minimum_page_area_fraction", "page_padding_fraction"})
        self.assertEqual(config["parameters"]["probability_threshold"]["values"], [0.2125, 0.215, 0.2175, 0.22, 0.2225, 0.225, 0.2275, 0.23, 0.2325, 0.235, 0.2375])
        self.assertEqual(config["parameters"]["close_kernel_fraction"]["values"], [0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032])
        self.assertIn(0.225, config["parameters"]["probability_threshold"]["values"])
        self.assertIn(0.028, config["parameters"]["close_kernel_fraction"]["values"])
        self.assertEqual(config["profiles"]["baseline"]["minimum_page_area_fraction"], 0.02)
        self.assertEqual(config["profiles"]["baseline"]["page_padding_fraction"], 0.0)
        self.assertEqual(config["parameters"]["minimum_page_area_fraction"]["values"], [0.02])
        self.assertEqual(config["parameters"]["page_padding_fraction"]["values"], [0.0])
        self.assertEqual(generate(config)[0], config["profiles"]["baseline"])

    def test_docextractor_managed_runtime_includes_upstream_toolz_dependency(self):
        runtime = (ROOT / "tools/ensure-managed-runtime.sh").read_text(encoding="utf-8")
        preflight = (ROOT / "hth/docextractor_page_mask_preflight.py").read_text(encoding="utf-8")
        self.assertIn('"toolz==1.0.0"', runtime)
        self.assertIn("import torch, gdown, toolz", runtime)
        self.assertIn("import toolz", preflight)

    def test_workflows_expose_all_three_detector_choices(self):
        for relative in (".github/workflows/regress-detector.yml", ".github/workflows/execution-optimizer.yml"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            for detector in ("eynollah_page_mask", "pagenet_page_mask", "docextractor_page_mask"):
                self.assertIn(f"- {detector}", text)


if __name__ == "__main__":
    unittest.main()
