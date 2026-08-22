import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class DocumentDetectorReviewTests(unittest.TestCase):
    def test_approved_gen3_calibration_is_pinned(self):
        data=json.loads((ROOT/'config/document-detectors.json').read_text())
        gen3=data['detectors']['amsre_doc_ufcn_fusion']
        self.assertEqual(gen3['golden_set_id'],'HTH-0001')
        self.assertEqual(gen3['parameter_set_id'],'57b3edb3ac1c')
        self.assertEqual(gen3['parameters']['maximum_amsre_refined_support_fraction'],0.65)
        self.assertEqual(gen3['parameters']['minimum_corner_disagreement_fraction'],0.0075)

    def test_production_workflow_exposes_document_detector(self):
        text=(ROOT/'.github/workflows/preprocess.yml').read_text()
        self.assertIn('document_detector:',text)
        self.assertIn('amsre_doc_ufcn_fusion',text)
        core=(ROOT/'.github/workflows/_core-hth.yml').read_text()
        self.assertIn('run_document_detector.py',core)

    def test_workbench_knows_gen3(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn("amsre_doc_ufcn_fusion",text)
        self.assertIn('Fusion Gen3 — AMSRE + Doc-UFCN',text)

if __name__=='__main__': unittest.main()
