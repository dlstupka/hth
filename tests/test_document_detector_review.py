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
        self.assertIn('canonicalDetectorCatalog',text)
        self.assertIn('Geometry detectors (47)',text)
        self.assertIn('syncDetectorCatalog();',text)
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
        self.assertIn("else if(analysisData&&!s.collection){s.collection=synthesizeSourceCollection(analysisData)",text)
        self.assertIn("collection_id:'HTH-SOURCE-DRAFT'",text)
        self.assertIn("calibration_selected:false",text)
        self.assertIn('Source release manifest SHA-256',text)
        self.assertIn("/^HTH-GOLDEN-\\d{4,}$/",text)
        self.assertIn("tag:g.golden_set_id",text)
        self.assertIn("/^HTH-SOURCE-\\d{4,}$/",text)
        self.assertIn("/^[0-9a-f]{64}$/",text)

    def test_workspace_inherits_immutable_source_provenance_from_build_info(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn('function buildSourceProvenance(text)',text)
        self.assertIn("yamlScalar(text,'source_release')||yamlScalar(text,'release')",text)
        self.assertIn("yamlScalar(text,'source_release_manifest_sha256')||yamlScalar(text,'release_manifest_sha256')",text)
        self.assertIn('mergeSourceProvenance(buildSourceProvenance',text)
        self.assertIn('current.tag||provenance.tag',text)
        self.assertIn("rejectGoldenField('goldenSourceReleaseTag'",text)
        self.assertIn('else if(analysisData&&!s.collection)',text)
        self.assertIn("const workspaceProvenance=s.collection?.source_release||s.collection?.source||null",text)
        self.assertIn("if(workspaceProvenance)mergeSourceProvenance(workspaceProvenance)",text)

    def test_explicit_ordinal_entry_selects_the_matching_view_mode(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn("$('selectionOrdinals').oninput=()=>{",text)
        self.assertIn("$('imageSelectionMode').value='list';syncImageSelectionControls();",text)
        self.assertIn("$('selectionOrdinals').disabled=false",text)
        self.assertIn('enable(true);syncImageSelectionControls();rebuild()',text)

    def test_ordinal_filter_rebuild_is_bounded_and_does_not_leak_thumbnail_urls(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn('thumbnailUrls:new Map()',text)
        self.assertIn('for(const url of s.thumbnailUrls.values())URL.revokeObjectURL(url)',text)
        self.assertIn('if(!visible)return;',text)
        self.assertIn('if(!s.thumbnailUrls.has(ordinal))s.thumbnailUrls.set(ordinal,URL.createObjectURL(f))',text)
        self.assertIn('setTimeout(rebuild,120)',text)
        self.assertIn("const explicitOrdinals=$('imageSelectionMode').value==='list'?parseOrdinalExpression",text)

    def test_json_exports_use_a_repeatable_download_lifecycle(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn('function downloadBlob(name,blob)',text)
        self.assertIn('document.body.appendChild(a);a.click()',text)
        self.assertIn('setTimeout(()=>{URL.revokeObjectURL(url);a.remove()},1000)',text)
        self.assertIn("downloadJson('reference_collection.json',s.collection)",text)

    def test_reference_export_reuses_a_writable_file_handle(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn('referenceExportHandle:null',text)
        self.assertIn("typeof window.showSaveFilePicker!=='function'",text)
        self.assertIn('s.referenceExportHandle=await window.showSaveFilePicker',text)
        self.assertIn('const writable=await s.referenceExportHandle.createWritable()',text)
        self.assertIn('try{await writable.write(contents)}finally{await writable.close()}',text)
        self.assertIn("button.textContent=result==='written'?'Saved to file ✓':'Downloaded ✓'",text)

    def test_freeze_export_opens_save_picker_before_async_hashing(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        picker=text.index("handle=await window.showSaveFilePicker({suggestedName:name")
        digest=text.index("fileSha=await sha256Hex(fileBytes)",picker)
        self.assertLess(picker,digest)
        self.assertIn("await writable.write(JSON.stringify(freeze,null,2)+'\\n')",text)
        self.assertIn("Saved freeze ✓",text)

    def test_pages_can_be_explicitly_unapproved_or_excluded(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn('id="unapprovePage"',text)
        self.assertIn('id="excludeCurrent"',text)
        self.assertIn("p.review_status='unreviewed'",text)
        self.assertIn("invalidateGoldenApproval('Page approval removed')",text)
        self.assertIn("invalidateGoldenApproval('Page excluded from Golden Set membership')",text)

    def test_golden_provenance_fields_allow_spaces_while_typing(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn('function invalidateGoldenApproval(reason,refreshFields=true)',text)
        self.assertIn('if(refreshFields)updateGoldenFields();else updateGoldenStatus()',text)
        self.assertIn("invalidateGoldenApproval('Golden Set identity or provenance changed',false)",text)

    def test_calibration_membership_actions_have_visible_confirmation(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn('id="calibrationMembershipStatus"',text)
        self.assertIn('function updateCalibrationMembershipStatus',text)
        self.assertIn('replaced with ${count} visible pages',text)

    def test_page_review_badge_is_not_confused_with_named_reviewer(self):
        text=(ROOT/'tools/reference-collection-editor-multidetector.html').read_text(encoding='utf-8')
        self.assertIn("Needs review: ${review?'Yes':'No'}",text)

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
