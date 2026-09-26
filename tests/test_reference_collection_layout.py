import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReferenceCollectionLayoutTests(unittest.TestCase):
    def test_layout_seed_is_draft_from_frozen_gs0002_not_invented_truth(self):
        detector = json.loads((ROOT / 'config/golden_sets/HTH-GOLDEN-0002.golden-set.json').read_text(encoding='utf-8'))
        freeze = json.loads((ROOT / 'config/golden_sets/HTH-GOLDEN-0002.freeze.json').read_text(encoding='utf-8'))
        layout = json.loads((ROOT / 'config/golden_sets/HTH-GOLDEN-0002-LAYOUT.draft.json').read_text(encoding='utf-8'))
        self.assertEqual(layout['layout_golden_set_id'], 'HTH-GOLDEN-0002-LAYOUT')
        self.assertEqual(layout['status'], 'draft')
        self.assertEqual(layout['coordinate_view'], 'source')
        self.assertEqual(layout['source_golden_set_sha256'], freeze['golden_set_sha256'])
        self.assertEqual(len(layout['pages']), 18)
        for source, seeded in zip(detector['pages'], layout['pages']):
            self.assertEqual(seeded['global_ordinal'], source['global_ordinal'])
            self.assertEqual(seeded['image_sha256'], source['image_sha256'])
            self.assertEqual(seeded['approved_document_bbox'], source['physical_document_bbox'])
            self.assertEqual(seeded['regions'], [])
            self.assertEqual(seeded['review_status'], 'unreviewed')

    def test_director_has_two_independent_editors_and_shared_workspace(self):
        director = (ROOT / 'tools/reference-collection-director.html').read_text(encoding='utf-8')
        self.assertIn('src="reference-collection-editor.html"', director)
        self.assertIn('src="reference-collection-layout.html"', director)
        self.assertIn("type:'HTH_REFERENCE_WORKSPACE',files", director)
        self.assertIn('id="bundle" type="file"', director)
        self.assertIn('readGs0002Bundle(file', director)
        self.assertIn('id="sourceRepo" type="url"', director)
        self.assertIn('id="goldenSetId" list="goldenSetOptions"', director)
        self.assertIn("type:'HTH_REFERENCE_RELEASE',release", director)
        self.assertIn('id="resultsRepo"', director)
        self.assertIn('id="updateResults"', director)
        self.assertNotIn('id="workspace" type="file" webkitdirectory', director)
        self.assertFalse((ROOT / 'tools/reference-collection-editor-multidetector.html').exists())

    def test_both_editors_default_to_gs0002_and_allow_manual_selection(self):
        defaults = (ROOT / 'tools/reference-collection-defaults.js').read_text(encoding='utf-8')
        detector = (ROOT / 'tools/reference-collection-editor.html').read_text(encoding='utf-8')
        layout = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('HTH-GOLDEN-0002-LAYOUT', defaults)
        self.assertIn('HTH-GOLDEN-0002', defaults)
        for editor in (detector, layout):
            self.assertIn('reference-collection-defaults.js', editor)
            self.assertIn('reference-collection-images.js', editor)
            self.assertIn('readGs0002Bundle(file', editor)
            self.assertIn('id="goldenSetChoice"', editor)
            self.assertIn('value="manual"', editor)
            self.assertIn('HTH_REFERENCE_WORKSPACE', editor)
            self.assertIn('id="releaseRepo" type="url"', editor)
            self.assertIn('id="releaseTag" list="releaseOptions"', editor)
            self.assertIn('HTH_REFERENCE_RELEASE', editor)
            self.assertIn('id="resultsRepo"', editor)
            self.assertIn('id="updateResults"', editor)
        self.assertIn('Open result repository workspace', detector)
        self.assertIn('not the results repository', layout)

    def test_offline_defaults_match_authoritative_files(self):
        lines = (ROOT / 'tools/reference-collection-defaults.js').read_text(encoding='utf-8').splitlines()
        bundled = {}
        for line in lines:
            stripped = line.strip()
            for key in ('detector', 'layout'):
                if stripped.startswith(f'{key}: '):
                    bundled[key] = json.loads(stripped.partition(': ')[2].rstrip(','))
        self.assertEqual(bundled['detector'], json.loads((ROOT / 'config/golden_sets/HTH-GOLDEN-0002.golden-set.json').read_text(encoding='utf-8')))
        self.assertEqual(bundled['layout'], json.loads((ROOT / 'config/golden_sets/HTH-GOLDEN-0002-LAYOUT.draft.json').read_text(encoding='utf-8')))

    def test_gs0002_bundle_import_uses_frozen_identity_and_per_image_hashes(self):
        freeze = json.loads((ROOT / 'config/golden_sets/HTH-GOLDEN-0002.freeze.json').read_text(encoding='utf-8'))
        defaults = (ROOT / 'tools/reference-collection-defaults.js').read_text(encoding='utf-8')
        importer = (ROOT / 'tools/reference-collection-images.js').read_text(encoding='utf-8')
        self.assertIn(freeze['image_bundle']['asset'], defaults)
        self.assertIn(str(freeze['image_bundle']['size']), defaults)
        self.assertIn(freeze['image_bundle']['sha256'], defaults)
        self.assertIn("sha256(bytes) !== bundle.sha256", importer)
        self.assertIn("sha256(imageBytes) !== record.sha256", importer)
        self.assertIn("record.size !== undefined && size !== record.size", importer)

    def test_layout_candidates_cannot_become_approved_truth_implicitly(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn("proposals=incoming", editor)
        self.assertIn('Adopt selected', editor)
        self.assertIn('Mark page reviewed', editor)
        self.assertIn("page().review_status='unreviewed'", editor)
        self.assertIn("collection.status='draft'", editor)
        self.assertIn('source image SHA-256 differs from the frozen Golden Set', editor)
        self.assertIn('candidate source identity mismatch', editor)

    def test_layout_page_tabs_stay_above_independently_scrolling_image(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertLess(editor.index('id="pages" class="pages"'), editor.index('class="viewport"'))
        self.assertIn('main{display:grid;grid-template-columns:minmax(0,1fr) 330px;flex:1;min-height:0}', editor)
        self.assertIn('.viewport{flex:1;min-height:0;overflow:auto', editor)

    def test_layout_zoom_changes_only_page_view(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        for control in ('zoomOut', 'zoomFit', 'zoomIn', 'zoomLevel'):
            self.assertIn(f'id="{control}"', editor)
        self.assertIn("$('viewport').addEventListener('wheel'", editor)
        self.assertIn('if(!e.ctrlKey||!image)return', editor)
        self.assertIn('function setZoom(value,clientX,clientY)', editor)
        self.assertIn('Math.round((e.clientX-rect.left)/zoom)', editor)

    def test_layout_zoom_follows_current_page_and_remembers_tweaked_pages(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn("let pageZooms=new Map(),defaultZoom={mode:'fit'}", editor)
        self.assertIn('pageZooms.get(p.global_ordinal)||defaultZoom', editor)
        self.assertIn('rememberZoom({mode:\'scale\',zoom})', editor)
        self.assertIn("$('zoomFit').onclick=()=>fit(true)", editor)
        self.assertIn('localStorage.setItem(zoomStorageKey()', editor)
        self.assertIn('restoreZoomPreferences();showResults()', editor)
        self.assertNotIn('page().zoom', editor)

    def test_layout_proposals_have_visible_outline_without_heavier_fill(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn("drawPath(r.boundary,'#704000','rgba(229,184,92,.07)',2.25,'#f3c76a')", editor)
        self.assertIn("drawPath(currentProposals[selected.index].boundary,'#0969da','rgba(229,184,92,.07)',2.25,'#dff5ff')", editor)
        self.assertIn("drawPath(r.boundary,'#006b3b','rgba(81,220,145,.05)',2,'#caffdf')", editor)
        self.assertIn("drawPath(r.boundary,'#004d2b','rgba(81,220,145,.05)',2,'#caffdf')", editor)
        self.assertLess(editor.index("drawPath(r.boundary,'#704000'"), editor.index("drawPath(currentProposals[selected.index].boundary,'#0969da'"))

    def test_layout_annotation_controls_are_visible_and_recoverable(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        for control in ('selectTool', 'rectangle', 'polygon', 'finishPolygon',
                        'cancelDrawing', 'undo', 'redo', 'deleteVertex',
                        'deleteRegion', 'dismissProposal', 'toolState',
                        'makeRectangle', 'addVertex'):
            self.assertIn(f'id="{control}"', editor)
        self.assertIn(".classList.toggle('tool-active'", editor)
        self.assertIn('function selectAt(p)', editor)
        self.assertIn('function drawDraft()', editor)
        self.assertIn('function capturePage(', editor)
        self.assertIn('function undo()', editor)
        self.assertIn('function redo()', editor)
        self.assertIn('if(!validPolygon(updated,image.naturalWidth,image.naturalHeight))', editor)
        self.assertIn('if(action.type===\'rectangle\')', editor)

    def test_review_advances_to_next_unreviewed_layout_page(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('function nextReviewIndex(fromIndex)', editor)
        self.assertIn("collection.pages[candidate].review_status!=='reviewed'", editor)
        self.assertIn('const next=nextReviewIndex(index)', editor)
        self.assertIn('load(next,`Page ${reviewedOrdinal} reviewed. Next page needing review: `)', editor)
        self.assertIn('All ${collection.pages.length} pages reviewed', editor)

    def test_rectangles_keep_axis_aligned_corner_and_edge_editing(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('function rectanglePoints(left,top,right,bottom)', editor)
        self.assertIn('function isRectangle(r)', editor)
        self.assertIn('function resizeRectangle(r,handle,p)', editor)
        self.assertIn("drag={type:'edge',index:edge,start:p,before:capturePage(),moved:false}", editor)
        self.assertIn("addRegion(points,$('kind').value,'rectangle')", editor)
        self.assertIn("r.shape='rectangle'", editor)
        self.assertIn("if(isRectangle(r)||r.boundary.length<=3)return", editor)

    def test_selected_layout_handles_have_one_pixel_nudges(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        for control in ('nudgeLeft', 'nudgeUp', 'nudgeDown', 'nudgeRight', 'nudgeStatus'):
            self.assertIn(f'id="{control}"', editor)
        self.assertIn('function canNudge(dx,dy)', editor)
        self.assertIn('function nudgeSelection(dx,dy,recordHistory=true)', editor)
        self.assertIn('function movePolygonEdge(r,index,dx,dy', editor)
        self.assertIn('nudgeSelection(...arrows[e.key])', editor)
        self.assertIn('selectedEdge%2===0?dy!==0:dx!==0', editor)
        self.assertIn('if(recordHistory)checkpoint();r.boundary=next.boundary', editor)

    def test_nudge_buttons_repeat_on_hold_as_one_undoable_edit(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('function installNudgeControl(id,dx,dy)', editor)
        self.assertIn('button.onpointerdown=e=>', editor)
        self.assertIn('nudgeSelection(dx,dy,false)', editor)
        self.assertIn('button.onpointerup=stop', editor)
        self.assertIn('button.onpointercancel=stop', editor)
        self.assertIn('button.onpointerleave=stop', editor)
        self.assertIn('if(e.detail===0)nudgeSelection(dx,dy)', editor)

    def test_add_vertex_splits_selected_polygon_edge_only(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('id="addVertex"', editor)
        self.assertIn('function insertVertexOnEdge(edge,point)', editor)
        self.assertIn('if(!r||isRectangle(r)||edge<0', editor)
        self.assertIn("$('addVertex').disabled=!verified||selected?.type!=='region'||selectedEdge<0||isRectangle", editor)
        self.assertIn('insertVertexOnEdge(selectedEdge,[Math.round((a[0]+b[0])/2)', editor)
        self.assertIn('selectedVertex=edge+1;selectedEdge=-1', editor)

    def test_layout_algorithm_and_golden_set_overlays_are_independent(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('id="showTruth" type="checkbox" checked', editor)
        self.assertIn('id="algorithmOverlays"', editor)
        self.assertIn("algorithmVisibility=new Map([['kraken',{label:'Kraken',visible:true}]])", editor)
        self.assertIn('function renderAlgorithmOverlays()', editor)
        self.assertIn("algorithm_id:'kraken'", editor)
        self.assertIn("if($('showTruth').checked)for", editor)
        self.assertIn('if(proposalShown(r)', editor)

    def test_layout_dismissals_are_draft_data_and_can_be_shown_or_restored(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('id="showDismissed" type="checkbox"', editor)
        self.assertIn('id="restoreProposal"', editor)
        self.assertIn('function isDismissed(candidate)', editor)
        self.assertIn("page().dismissed_proposal_ids??=[]", editor)
        self.assertIn('dismissed.push(candidate.id)', editor)
        self.assertIn('page().dismissed_proposal_ids=page().dismissed_proposal_ids.filter', editor)
        self.assertIn("id:await candidateId('kraken',kind,boundary)", editor)
        self.assertIn("JSON.stringify(collection,null,2)", editor)
        self.assertIn(".filter(candidate=>!isDismissed(candidate))", editor)
        self.assertNotIn("proposals.get(page().global_ordinal).splice(selected.index,1)", editor)

    def test_reopening_same_layout_draft_keeps_matching_kraken_import(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('id="openDraft" type="button"', editor)
        self.assertIn("$('manualJson').value=''", editor)
        self.assertIn('function sameProposalSource(data)', editor)
        self.assertIn('const keepProposals=sameProposalSource(data)', editor)
        self.assertIn('if(!keepProposals)proposals=new Map()', editor)
        self.assertIn("lastChoice='loaded'", editor)
        self.assertIn("$('goldenSetChoice').value=lastChoice", editor)
        self.assertIn("$('proposals').onchange=e=>{importProposals(e.target.files);e.target.value=''}", editor)

    def test_detector_page_thumbnails_stay_above_independently_scrolling_image(self):
        editor = (ROOT / 'tools/reference-collection-editor.html').read_text(encoding='utf-8')
        self.assertLess(editor.index('id="thumbnailBar" class="thumbnailbar"'), editor.index('id="canvasWrap" class="canvaswrap"'))
        self.assertIn('.workspace{min-width:0;min-height:0;display:grid;grid-template-rows:auto auto auto minmax(0,1fr)}', editor)
        self.assertIn('main{display:grid;grid-template-columns:minmax(0,1fr) 360px;flex:1;min-height:0}', editor)


if __name__ == '__main__':
    unittest.main()
