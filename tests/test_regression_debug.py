from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.regression.adapters.components import detect as components_detect
from hth.regression.adapters.contour import detect as contour_detect
from hth.regression.adapters.contour_quad import detect as contour_quad_detect
from hth.regression.adapters.consensus_quad import detect as consensus_quad_detect
from hth.regression.adapters.contour_components import detect as contour_components_detect
from hth.regression.adapters.contour_projection import detect as contour_projection_detect
from hth.regression.adapters.ransac import detect as ransac_detect
from hth.regression.runner import write_debug_artifacts


class RegressionDebugTests(unittest.TestCase):
    def test_run_forwards_selected_debug_level_to_artifact_writer(self) -> None:
        from hth.regression.runner import run

        source = inspect.getsource(run)
        self.assertIn("debug_level=debug_level", source)


    def test_verbose_promotes_failures_policy_to_complete_winner_debug(self) -> None:
        from hth.regression.runner import run

        source = inspect.getsource(run)
        self.assertIn('debug_level == "verbose" and debug_policy in {"none", "failures"}', source)
        self.assertIn('debug_policy = "winner"', source)

    def test_regression_adapter_populates_registry_provenance(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (280, 180), 255, -1)

        candidate = contour_detect(image_bgr=image, mask=mask)

        self.assertEqual(candidate.detector_name, "Contour")
        self.assertEqual(candidate.origin, "HTH")
        self.assertTrue(candidate.foundation)
        self.assertTrue(candidate.authors)
        self.assertTrue(candidate.version)
        self.assertTrue(candidate.repository)

    def test_failure_debug_directory_is_obvious_and_self_describing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.jpg"
            image = np.zeros((120, 180, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)
            page = {
                "global_ordinal": 6,
                "label": "title_or_index_sheet",
                "layout_type": "single_page",
                "image_path": str(image_path),
                "image": image,
                "mask": np.zeros((120, 180), dtype=np.uint8),
                "approved_bbox": [10, 10, 170, 110],
            }
            result = {
                "parameter_set_id": "baseline123",
                "pages": [{
                    "global_ordinal": 6,
                    "label": page["label"],
                    "layout_type": page["layout_type"],
                    "status": "no_candidate",
                    "iou": 0.0,
                    "candidate": {"diagnostics": {"reason": "no_plausible_contour"}},
                }],
            }

            outputs = write_debug_artifacts(
                root,
                "contour",
                "run-test",
                policy="failures",
                ranked=[result],
                pages=[page],
            )

            debug_root = root / "debug" / "contour" / "run-test"
            debug_page = debug_root / "baseline123" / "page-0006"
            self.assertTrue((debug_root / "README.txt").is_file())
            self.assertTrue((debug_page / "01-original.jpg").is_file())
            self.assertTrue((debug_page / "02-input-mask.png").is_file())
            self.assertTrue((debug_page / "03-overlay.jpg").is_file())
            diagnostics = json.loads((debug_page / "04-diagnostics.json").read_text())
            self.assertEqual(diagnostics["result"]["status"], "no_candidate")
            self.assertIn("debug/contour/run-test/README.txt", outputs)

    def test_components_debug_writes_detector_intermediate_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.jpg"
            image = np.zeros((120, 180, 3), dtype=np.uint8)
            mask = np.zeros((120, 180), dtype=np.uint8)
            mask[25:95, 30:150] = 255
            cv2.imwrite(str(image_path), image)
            candidate = components_detect(image_bgr=image, mask=mask)
            page = {
                "global_ordinal": 6,
                "label": "title_or_index_sheet",
                "layout_type": "single_page",
                "image_path": str(image_path),
                "image": image,
                "mask": mask,
                "approved_bbox": [10, 10, 170, 110],
            }
            result = {
                "parameter_set_id": "baseline123",
                "pages": [{
                    "global_ordinal": 6,
                    "label": page["label"],
                    "layout_type": page["layout_type"],
                    "status": "ok",
                    "iou": 0.5,
                    "candidate": candidate.__dict__,
                }],
            }

            write_debug_artifacts(
                root, "components", "run-test", policy="winner",
                ranked=[result], pages=[page],
            )

            debug_page = root / "debug" / "components" / "run-test" / "baseline123" / "page-0006"
            self.assertTrue((debug_page / "01-original.jpg").is_file())
            self.assertTrue((debug_page / "02-input-mask.png").is_file())
            self.assertTrue((debug_page / "03-after-morphology.png").is_file())
            self.assertTrue((debug_page / "04-component-labels.png").is_file())
            self.assertTrue((debug_page / "05-significant-components.png").is_file())
            self.assertTrue((debug_page / "06-selected-components.png").is_file())
            self.assertTrue((debug_page / "07-candidate-envelope.png").is_file())
            self.assertTrue((debug_page / "08-overlay.jpg").is_file())
            self.assertTrue((debug_page / "09-diagnostics.json").is_file())


    def test_contour_quad_debug_writes_comparable_research_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.jpg"
            image = np.zeros((240, 360, 3), dtype=np.uint8)
            mask = np.zeros((240, 360), dtype=np.uint8)
            cv2.rectangle(mask, (40, 30), (320, 210), 255, -1)
            cv2.rectangle(image, (40, 30), (320, 210), (255, 255, 255), 3)
            cv2.imwrite(str(image_path), image)
            candidate = contour_quad_detect(image_bgr=image, mask=mask)
            page = {
                "global_ordinal": 6,
                "label": "title_or_index_sheet",
                "layout_type": "single_page",
                "image_path": str(image_path),
                "image": image,
                "mask": mask,
                "approved_bbox": [35, 25, 325, 215],
            }
            result = {
                "parameter_set_id": "baseline123",
                "pages": [{
                    "global_ordinal": 6,
                    "label": page["label"],
                    "layout_type": page["layout_type"],
                    "status": "ok",
                    "iou": 0.8,
                    "candidate": candidate.__dict__,
                }],
            }

            write_debug_artifacts(
                root, "contour_quad", "run-test", policy="winner",
                ranked=[result], pages=[page],
            )

            debug_page = root / "debug" / "contour_quad" / "run-test" / "baseline123" / "page-0006"
            for filename in (
                "01-original.jpg",
                "02-input-mask.png",
                "03-after-morphology.png",
                "04-contour-hypotheses.png",
                "05-quadrilateral-hypotheses.png",
                "06-edge-evidence.png",
                "07-selected-quadrilateral.png",
                "08-overlay.jpg",
                "09-diagnostics.json",
            ):
                self.assertTrue((debug_page / filename).is_file(), filename)

    def test_contour_components_debug_writes_comparable_research_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.jpg"
            image = np.full((240, 360, 3), 30, dtype=np.uint8)
            mask = np.zeros((240, 360), dtype=np.uint8)
            cv2.rectangle(mask, (40, 30), (320, 210), 255, -1)
            cv2.rectangle(image, (40, 30), (320, 210), (235, 235, 235), -1)
            cv2.imwrite(str(image_path), image)
            candidate = contour_components_detect(image_bgr=image, mask=mask)
            page = {
                "global_ordinal": 6, "label": "title_or_index_sheet",
                "layout_type": "single_page", "image_path": str(image_path),
                "image": image, "mask": mask, "approved_bbox": [35, 25, 325, 215],
            }
            result = {
                "parameter_set_id": "baseline123",
                "pages": [{
                    "global_ordinal": 6, "label": page["label"],
                    "layout_type": page["layout_type"], "status": candidate.status,
                    "iou": 0.8, "candidate": candidate.__dict__,
                }],
            }
            write_debug_artifacts(root, "contour_components", "run-test", policy="winner", ranked=[result], pages=[page])
            debug_page = root / "debug" / "contour_components" / "run-test" / "baseline123" / "page-0006"
            for filename in (
                "01-original.jpg", "02-input-mask.png",
                "03-contour-hypotheses.png", "04-component-labels.png",
                "05-selected-components.png", "06-component-envelope.png",
                "07-component-evidence.png", "08-selected-quadrilateral.png",
                "09-overlay.jpg", "10-diagnostics.json",
            ):
                self.assertTrue((debug_page / filename).is_file(), filename)

    def test_contour_projection_debug_writes_comparable_research_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.jpg"
            image = np.full((240, 360, 3), 30, dtype=np.uint8)
            mask = np.zeros((240, 360), dtype=np.uint8)
            cv2.rectangle(mask, (40, 30), (320, 210), 255, -1)
            cv2.rectangle(image, (40, 30), (320, 210), (235, 235, 235), -1)
            for y in range(60, 195, 18):
                cv2.line(image, (70, y), (290, y), (30, 30, 30), 3)
            cv2.imwrite(str(image_path), image)
            candidate = contour_projection_detect(image_bgr=image, mask=mask)
            page = {
                "global_ordinal": 6, "label": "title_or_index_sheet",
                "layout_type": "single_page", "image_path": str(image_path),
                "image": image, "mask": mask, "approved_bbox": [35, 25, 325, 215],
            }
            result = {
                "parameter_set_id": "baseline123",
                "pages": [{
                    "global_ordinal": 6, "label": page["label"],
                    "layout_type": page["layout_type"], "status": candidate.status,
                    "iou": 0.8, "candidate": candidate.__dict__,
                }],
            }
            write_debug_artifacts(root, "contour_projection", "run-test", policy="winner", ranked=[result], pages=[page])
            debug_page = root / "debug" / "contour_projection" / "run-test" / "baseline123" / "page-0006"
            for filename in (
                "01-original.jpg", "02-input-mask.png",
                "03-contour-hypotheses.png", "04-warped-candidate.png",
                "05-projection-binary.png", "06-horizontal-projection.png",
                "07-selected-quadrilateral.png", "08-overlay.jpg",
                "09-diagnostics.json",
            ):
                self.assertTrue((debug_page / filename).is_file(), filename)

    def test_consensus_quad_debug_writes_comparable_research_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.jpg"
            image = np.zeros((240, 360, 3), dtype=np.uint8)
            mask = np.zeros((240, 360), dtype=np.uint8)
            cv2.rectangle(mask, (40, 30), (320, 210), 255, -1)
            cv2.rectangle(image, (40, 30), (320, 210), (255, 255, 255), 3)
            cv2.imwrite(str(image_path), image)
            candidate = consensus_quad_detect(image_bgr=image, mask=mask)
            page = {
                "global_ordinal": 6, "label": "title_or_index_sheet",
                "layout_type": "single_page", "image_path": str(image_path),
                "image": image, "mask": mask, "approved_bbox": [35, 25, 325, 215],
            }
            result = {
                "parameter_set_id": "baseline123",
                "pages": [{
                    "global_ordinal": 6, "label": page["label"],
                    "layout_type": page["layout_type"], "status": candidate.status,
                    "iou": 0.8, "candidate": candidate.__dict__,
                }],
            }
            write_debug_artifacts(root, "consensus_quad", "run-test", policy="winner", ranked=[result], pages=[page])
            debug_page = root / "debug" / "consensus_quad" / "run-test" / "baseline123" / "page-0006"
            for filename in (
                "01-original.jpg", "02-input-mask.png",
                "03-contour-quad-vote.png", "04-edge-contour-vote.png",
                "05-agreement-overlay.png", "06-selected-consensus.png",
                "07-overlay.jpg", "08-diagnostics.json",
            ):
                self.assertTrue((debug_page / filename).is_file(), filename)

    def test_ransac_debug_writes_detector_intermediate_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.jpg"
            image = np.zeros((240, 360, 3), dtype=np.uint8)
            mask = np.zeros((240, 360), dtype=np.uint8)
            cv2.rectangle(mask, (40, 30), (320, 210), 255, -1)
            cv2.imwrite(str(image_path), image)
            candidate = ransac_detect(image_bgr=image, mask=mask)
            page = {
                "global_ordinal": 6,
                "label": "title_or_index_sheet",
                "layout_type": "single_page",
                "image_path": str(image_path),
                "image": image,
                "mask": mask,
                "approved_bbox": [35, 25, 325, 215],
            }
            result = {
                "parameter_set_id": "baseline123",
                "pages": [{
                    "global_ordinal": 6,
                    "label": page["label"],
                    "layout_type": page["layout_type"],
                    "status": "ok",
                    "iou": 0.8,
                    "candidate": candidate.__dict__,
                }],
            }

            write_debug_artifacts(
                root, "ransac", "run-test", policy="winner",
                ranked=[result], pages=[page],
            )

            debug_page = root / "debug" / "ransac" / "run-test" / "baseline123" / "page-0006"
            for filename in (
                "01-original.jpg",
                "02-input-mask.png",
                "03-boundary-samples.png",
                "04-fitted-edge-models.png",
                "05-ransac-inliers.png",
                "06-candidate-quadrilateral.png",
                "07-overlay.jpg",
                "08-diagnostics.json",
            ):
                self.assertTrue((debug_page / filename).is_file(), filename)


    def test_new_detector_configs_preserve_winner_debug_artifacts(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        for detector_id in ("radial_edge", "adaptive_radial_edge", "border_energy"):
            config_path = repository_root / "config" / "detectors" / f"{detector_id}.json"
            config = json.loads(config_path.read_text())
            self.assertEqual(
                config["regression"]["debug_artifacts"],
                "winner",
                detector_id,
            )

    def test_verbose_radial_edge_writes_search_ray_evidence(self) -> None:
        from hth.regression.adapters.radial_edge import detect as radial_edge_detect
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.jpg"
            image = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.rectangle(image, (45, 30), (275, 210), (235, 235, 235), -1)
            mask = np.ones((240, 320), dtype=np.uint8) * 255
            cv2.imwrite(str(image_path), image)
            candidate = radial_edge_detect(image_bgr=image, mask=mask, parameters={"minimum_ray_support": 0.20, "gradient_percentile": 70.0, "maximum_radius_fraction": 0.90})
            page = {"global_ordinal": 6, "label": "page", "layout_type": "single_page", "image_path": str(image_path), "image": image, "mask": mask, "approved_bbox": [45, 30, 275, 210]}
            result = {"parameter_set_id": "winner", "pages": [{"global_ordinal": 6, "label": "page", "layout_type": "single_page", "status": "ok", "iou": 0.9, "candidate": candidate.__dict__}]}
            write_debug_artifacts(root, "radial_edge", "run-test", policy="winner", ranked=[result], pages=[page], debug_level="verbose")
            debug_page = root / "debug" / "radial_edge" / "run-test" / "winner" / "page-0006"
            self.assertTrue((debug_page / "05-radial-search-rays.png").is_file())
            self.assertTrue((debug_page / "06-accepted-rays.png").is_file())

    def test_verbose_debug_images_exist_for_gradient_grabcut_and_border_energy(self) -> None:
        from hth.geometry import detector_border_energy, detector_grabcut, detector_gradient_vote
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(image, (45, 30), (275, 210), (235, 235, 235), -1)
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.rectangle(mask, (40, 25), (280, 215), 255, -1)
        corners = [[45, 30], [275, 30], [275, 210], [45, 210]]
        gradient = detector_gradient_vote.debug_images(image_bgr=image, mask=mask, candidate_corners=corners, verbose=True)
        self.assertIn("vertical-gradient-votes.png", gradient)
        self.assertIn("horizontal-gradient-votes.png", gradient)
        self.assertIn("vote-maxima.png", gradient)
        grabcut = detector_grabcut.debug_images(image_bgr=image, mask=mask, candidate_corners=corners, verbose=True)
        self.assertIn("grabcut-labels.png", grabcut)
        self.assertIn("definite-foreground-seed.png", grabcut)
        self.assertIn("grabcut-contours.png", grabcut)
        border = detector_border_energy.debug_images(image_bgr=image, mask=mask, candidate_corners=corners, verbose=True)
        self.assertIn("border-sampling-bands.png", border)
        self.assertIn("side-energy-scores.png", border)


if __name__ == "__main__":
    unittest.main()
