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
        self.assertEqual(candidate.bbox, [34, 19, 266, 181])
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
        self.assertEqual(candidate.bbox, [34, 19, 266, 181])
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
        self.assertEqual(candidate.bbox, [32, 102, 268, 183])
        self.assertEqual(
            candidate.diagnostics["parameters"]["merge_gap_fraction"], 0.01
        )

    def test_morphology_connects_fragmented_ink_regions(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        for y in (40, 50, 60, 70):
            for x in range(45, 250, 14):
                mask[y:y + 6, x:x + 12] = 255

        without_morphology = detector_components.detect(
            image_bgr=image,
            mask=mask,
            parameters={
                "morphology_close_fraction": 0.0,
                "morphology_dilate_fraction": 0.0,
            },
        )
        with_morphology = detector_components.detect(image_bgr=image, mask=mask)

        self.assertLess(
            with_morphology.diagnostics["component_count"],
            without_morphology.diagnostics["component_count"],
        )
        self.assertGreater(
            with_morphology.diagnostics["morphology_dilate_kernel_px"], 1
        )

    def test_debug_images_include_morphology_and_labels(self) -> None:
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[20:30, 20:40] = 255
        mask[35:45, 45:65] = 255

        images = detector_components.debug_images(mask=mask)

        self.assertEqual(
            set(images), {"after-morphology.png", "component-labels.png"}
        )
        self.assertEqual(images["after-morphology.png"].shape, mask.shape)
        self.assertEqual(images["component-labels.png"].shape, (*mask.shape, 3))

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
