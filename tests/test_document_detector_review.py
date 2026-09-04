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

    def test_production_workflow_uses_preferred_rank_one_detector(self):
        text=(ROOT/'.github/workflows/preprocess.yml').read_text(encoding='utf-8')
        self.assertIn('document_detector: preferred',text)
        self.assertNotIn('amsre_doc_ufcn_fusion',text)
        core=(ROOT/'.github/workflows/_core-hth.yml').read_text(encoding='utf-8')
        self.assertIn('Resolve Rank #1 approved document detector',core)
        self.assertIn('run_document_detector.py',core)

    def test_workbench_knows_gen3(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn("amsre_doc_ufcn_fusion",text)
        self.assertIn('Fusion Gen3 — AMSRE + Doc-UFCN',text)

    def test_workbench_discovers_and_scrolls_all_loaded_detector_candidates(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn('class="detector-list"',text)
        self.assertIn('max-height:390px;overflow-y:auto',text)
        self.assertIn('function syncDetectorCatalog()',text)
        self.assertIn('registerDetector(method,raw)',text)
        self.assertIn("return detectorAliases[method]||method",text)
        self.assertIn('Checkboxes add or remove overlays',text)

    def test_workbench_supports_safe_versioned_golden_set_approval(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn('Replace membership with visible',text)
        self.assertIn("p.calibration_selected=!sourceCollection",text)
        self.assertIn("Every selected page must be reviewed and approved",text)
        self.assertIn("`${g.golden_set_id}.golden-set.json`",text)
        self.assertIn("`${g.golden_set_id}.freeze.json`",text)
        self.assertIn("golden_set_sha256:fileSha",text)

    def test_workbench_can_synthesize_a_source_draft_from_production_analysis(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn('function synthesizeSourceCollection(data)',text)
        self.assertIn("else if(analysisData){s.collection=synthesizeSourceCollection(analysisData)",text)
        self.assertIn("collection_id:'HTH-SOURCE-DRAFT'",text)
        self.assertIn("calibration_selected:false",text)
        self.assertIn('Source release manifest SHA-256',text)
        self.assertIn("/^HTH-GOLDEN-\\d{4,}$/",text)
        self.assertIn("tag:g.golden_set_id",text)
        self.assertIn("/^HTH-SOURCE-\\d{4,}$/",text)
        self.assertIn("/^[0-9a-f]{64}$/",text)

    def test_explicit_ordinal_entry_selects_the_matching_view_mode(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn("$('selectionOrdinals').oninput=()=>{$('imageSelectionMode').value='list'",text)
        self.assertIn("$('selectionOrdinals').disabled=false",text)
        self.assertIn('enable(true);syncImageSelectionControls();rebuild()',text)

    def test_page_approval_can_auto_advance_without_discard_prompt(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn("if(s.pageDirty&&!confirm('Discard unsaved page changes?'))return",text)
        self.assertIn('s.index=Math.max(0,Math.min(i,s.pages.length-1));s.pageDirty=false',text)
        self.assertNotIn("if(s.dirty&&!confirm('Discard unsaved changes?'))return",text)

    def test_auto_advance_is_limited_to_the_active_image_selection(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn('const ordered=[...s.visibleIndices.filter(i=>i>fromIndex)',text)
        self.assertIn("if(pageStatus(p)!=='approved'||!Array.isArray(p.physical_document_bbox))return i",text)
        self.assertNotIn("s.review.has(Number(p.global_ordinal))||pageStatus(p)==='fail'",text)

    def test_golden_set_review_state_is_not_inferred_from_detector_triage(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn("p.review_status=Array.isArray(p.physical_document_bbox)?'approved':'unreviewed'",text)
        self.assertIn("const review=!approved",text)
        self.assertIn("return 'review'",text)
        self.assertIn('pending human review',text)
        self.assertIn('detector-triaged',text)
        self.assertIn("p.review_status==='approved'&&Array.isArray(p.physical_document_bbox)",text)

if __name__=='__main__': unittest.main()
