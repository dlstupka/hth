from __future__ import annotations

import unittest

import numpy as np

from hth.geometry import detector_components


class ConnectedComponentsDetectorTests(unittest.TestCase):
    def test_detects_large_document_component(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        mask[20:180, 35:265] = 255

        candidate = detector_components.detect(image_bgr=image, mask=mask)

        self.assertEqual(candidate.status, "ok")
        self.assertEqual(candidate.method, "components")
        self.assertEqual(candidate.bbox, [35, 20, 265, 180])
        self.assertEqual(candidate.diagnostics["merged_components"], 1)
        self.assertEqual(
            candidate.diagnostics["parameters"],
            detector_components.BASELINE_PARAMETERS,
        )
        self.assertGreater(candidate.confidence, 0.7)

    def test_merges_nearby_page_fragments(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        mask[20:95, 35:265] = 255
        mask[101:180, 35:265] = 255

        candidate = detector_components.detect(image_bgr=image, mask=mask)

        self.assertEqual(candidate.status, "ok")
        self.assertEqual(candidate.bbox, [35, 20, 265, 180])
        self.assertEqual(candidate.diagnostics["merged_components"], 2)

    def test_parameter_overrides_change_merge_and_padding(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        mask[20:90, 35:265] = 255
        mask[105:180, 35:265] = 255

        candidate = detector_components.detect(
            image_bgr=image,
            mask=mask,
            parameters={
                "merge_gap_fraction": 0.01,
                "bbox_padding_fraction": 0.01,
            },
        )

        self.assertEqual(candidate.status, "ok")
        self.assertEqual(candidate.diagnostics["merged_components"], 1)
        self.assertEqual(candidate.bbox, [33, 103, 267, 182])
        self.assertEqual(
            candidate.diagnostics["parameters"]["merge_gap_fraction"], 0.01
        )

    def test_rejects_unknown_parameter(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaisesRegex(
            ValueError, "Unknown Connected Components parameters"
        ):
            detector_components.detect(
                image_bgr=image,
                mask=mask,
                parameters={"mystery": 1},
            )

    def test_tiny_components_are_a_normal_miss(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        mask[5:10, 5:10] = 255
        mask[50:54, 70:74] = 255

        candidate = detector_components.detect(image_bgr=image, mask=mask)

        self.assertIsNone(candidate.bbox)
        self.assertEqual(candidate.status, "no_candidate")
        self.assertEqual(candidate.diagnostics["reason"], "no_significant_components")


if __name__ == "__main__":
    unittest.main()
