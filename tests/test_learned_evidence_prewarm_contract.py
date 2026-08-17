import unittest
from pathlib import Path


class LearnedEvidencePrewarmContractTests(unittest.TestCase):
    def test_runner_prewarms_before_baseline_and_parameter_executor(self):
        text = Path("hth/regression/runner.py").read_text(encoding="utf-8")
        prewarm = text.index("evidence_preparer = PRECOMPUTED_EVIDENCE_PREPARERS.get(name)")
        baseline = text.index('progress.begin_evaluation("baseline")')
        executor = text.index("ThreadPoolExecutor(max_workers=args.threads")
        self.assertLess(prewarm, baseline)
        self.assertLess(prewarm, executor)
        self.assertIn("logical_golden_set(pages)", text)
        self.assertIn("PRECOMPUTED_EVIDENCE_LOADERS", text)
        self.assertIn("--precomputed-evidence", text)

    def test_kraken_cache_rechecks_inside_inference_lock(self):
        text = Path("hth/geometry/detector_kraken_page_mask.py").read_text(encoding="utf-8")
        lock = text.index("with _INFERENCE_LOCK:")
        recheck = text.index("with _EVIDENCE_CACHE_LOCK:", lock)
        predict = text.index("model.predict(", lock)
        self.assertLess(recheck, predict)

    def test_dhsegment_uses_cached_inference_in_proposal(self):
        text = Path("hth/geometry/detector_dhsegment_page_mask.py").read_text(encoding="utf-8")
        self.assertIn("probability, original_shape = _infer_evidence(image_bgr)", text)


if __name__ == "__main__":
    unittest.main()
