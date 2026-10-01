import json
import shutil
import subprocess
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
        self.assertEqual(layout['schema_version'], '0.2')
        self.assertEqual(layout['annotation_contract'], 'regions-reading-order-v2')
        self.assertEqual(layout['source_golden_set_sha256'], freeze['golden_set_sha256'])
        self.assertEqual(len(layout['pages']), 18)
        for source, seeded in zip(detector['pages'], layout['pages']):
            self.assertEqual(seeded['global_ordinal'], source['global_ordinal'])
            self.assertEqual(seeded['image_sha256'], source['image_sha256'])
            self.assertEqual(seeded['approved_document_bbox'], source['physical_document_bbox'])
            self.assertEqual(seeded['regions'], [])
            self.assertEqual(seeded['review_status'], 'unreviewed')
            self.assertEqual(seeded['reading_order'], [])
            self.assertEqual(seeded['reading_order_method'], 'spatial_suggestion')
            self.assertEqual(seeded['reading_order_status'], 'unreviewed')

    def test_region_reading_order_has_explicit_review_and_persistence(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        for control in ('readingOrder', 'regionLabel', 'spatialOrder', 'confirmOrder'):
            self.assertIn(f'id="{control}"', editor)
        self.assertIn('function spatialReadingOrder(p)', editor)
        self.assertIn('function validReadingOrder(p)', editor)
        self.assertIn('function normalizeLayoutDraft(data)', editor)
        self.assertIn("p.reading_order=[...ids];p.reading_order_method='creation_order'", editor)
        self.assertIn("p.reading_order_status==='unreviewed'&&p.reading_order_method==='spatial_suggestion'", editor)
        self.assertIn("reading_order_method:'creation_order'", editor)
        self.assertIn("page().reading_order_method='manual'", editor)
        self.assertIn("page().reading_order_status='reviewed'", editor)
        self.assertIn('page().reading_order_reviewed_at_utc=new Date().toISOString()', editor)
        self.assertIn('const data=normalizeLayoutDraft(raw)', editor)
        self.assertIn('region review is separate', editor.lower())
        self.assertIn("if(p.regions.length&&!validReadingOrder(p))", editor)
        self.assertIn("page().reading_order=page().reading_order.filter(id=>id!==removed.id)", editor)
        self.assertIn("p.reading_order_status='unreviewed'", editor)
        self.assertNotIn("page().review_status='unreviewed';delete page().reviewed_at_utc;page().reading_order_status", editor)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed to exercise layout editor functions')
    def test_creation_order_migration_and_multi_region_duplication(self):
        script = r"""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');
            const editor = fs.readFileSync('tools/reference-collection-layout.html', 'utf8');
            const line = name => editor.split('\n').find(row => row.startsWith(`function ${name}(`));
            const page = {
              global_ordinal: 1, regions: [
                {id:'r1', kind:'text', shape:'polygon', boundary:[[1,1],[11,1],[11,11],[1,11]]},
                {id:'r2', kind:'text', shape:'polygon', boundary:[[20,5],[30,5],[30,15],[20,15]]}
              ], region_notes: [], reading_order:['r2','r1'],
              reading_order_method:'spatial_suggestion', reading_order_status:'unreviewed'
            };
            const ctx = {structuredClone, page:()=>page, image:{naturalWidth:100,naturalHeight:100},
              verified:true, selectedRegionIds:new Set(['r1','r2']), selected:null,
              checkpoint:()=>{ctx.checkpoints++}, checkpoints:0, setMode:()=>{},
              $:()=>({value:''}),
              invalidation:()=>{}, draw:()=>{}, say:()=>{}};
            vm.createContext(ctx);
            vm.runInContext(['validReadingOrder','validRegionNotes','spatialReadingOrder',
              'normalizeLayoutDraft','validPolygon','clonePlacement','duplicateSelectedRegions'].map(line).join('\n'), ctx);
            const draft = {annotation_contract:'regions-reading-order-v2', pages:[page]};
            const migrated = ctx.normalizeLayoutDraft(draft);
            assert.deepEqual(Array.from(migrated.pages[0].reading_order), ['r1','r2']);
            assert.equal(migrated.pages[0].reading_order_method, 'creation_order');
            assert.deepEqual(page.reading_order, ['r2','r1']);
            for (const method of ['manual','spatial_suggestion']) {
              const confirmed = structuredClone(draft);
              confirmed.pages[0].reading_order_method = method;
              confirmed.pages[0].reading_order_status = 'reviewed';
              assert.deepEqual(Array.from(ctx.normalizeLayoutDraft(confirmed).pages[0].reading_order), ['r2','r1']);
            }
            page.regions[0].label = 'entry 1';
            page.reading_order = ['r1','r2'];
            page.reading_order_method = 'creation_order';
            vm.runInContext('duplicateSelectedRegions()', ctx);
            assert.equal(ctx.checkpoints, 1);
            assert.equal(page.regions.length, 4);
            assert.deepEqual(Array.from(page.reading_order), ['r1','r2','r3','r4']);
            assert.deepEqual(Array.from(ctx.selectedRegionIds), ['r3','r4']);
            assert.equal(ctx.selected.index, 3);
            assert.deepEqual(Array.from(page.regions[2].boundary[0]), [1,25]);
            assert.deepEqual(Array.from(page.regions[3].boundary[0]), [20,29]);
            assert.equal(Math.min(...page.regions.slice(2).flatMap(r=>r.boundary.map(p=>p[1]))),
              Math.max(...page.regions.slice(0,2).flatMap(r=>r.boundary.map(p=>p[1])))+10);
            assert.equal(page.regions[2].label, undefined);
            ctx.selectedRegionIds = new Set(['r3']);
            vm.runInContext('duplicateSelectedRegions()', ctx);
            assert.equal(page.regions.length, 5);
            ctx.selectedRegionIds = new Set(['r5']);
            ctx.image.naturalHeight = 60;
            vm.runInContext('duplicateSelectedRegions()', ctx);
            assert.equal(page.regions.length, 6);
            assert.deepEqual(Array.from(page.regions[5].boundary[0]), [21,45]);
            assert.equal(ctx.checkpoints, 3);
        """
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed to exercise clone placement')
    def test_clone_placement_falls_back_below_right_above_left(self):
        script = r"""
            const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
            const editor=fs.readFileSync('tools/reference-collection-layout.html','utf8');
            const line=name=>editor.split('\n').find(row=>row.startsWith(`function ${name}(`));
            const ctx={};vm.createContext(ctx);
            vm.runInContext(['validPolygon','clonePlacement'].map(line).join('\n'),ctx);
            const region=(x1,y1,x2,y2)=>({boundary:[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]});
            const check=(source,width,height,direction,first)=>{
              const placement=ctx.clonePlacement([source],width,height);
              assert.equal(placement.direction,direction);
              assert.deepEqual(Array.from(placement.copies[0].boundary[0]),first);
            };
            check(region(20,20,30,30),100,100,'below',[20,40]);
            check(region(10,70,20,80),100,100,'right',[30,70]);
            check(region(70,70,80,80),100,100,'above',[70,50]);
            check(region(70,0,80,10),100,20,'left',[50,0]);
            assert.equal(ctx.clonePlacement([region(0,0,9,9)],19,19),null);
            const group=[region(0,0,10,10),region(20,5,30,15)];
            const placement=ctx.clonePlacement(group,100,100);
            assert.equal(placement.direction,'below');
            assert.equal(Math.min(...placement.copies.flatMap(c=>c.boundary.map(p=>p[1]))),25);
            assert.deepEqual(Array.from(placement.copies[1].boundary[0]),[20,30]);
        """
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed to exercise polygon movement')
    def test_polygon_edge_drag_translates_whole_shape_with_image_clamping(self):
        script = r"""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');
            const editor = fs.readFileSync('tools/reference-collection-layout.html', 'utf8');
            const helper = editor.split('\n').find(row => row.startsWith('function moveWholePolygon('));
            const ctx = {image:{naturalWidth:100,naturalHeight:80}};
            vm.createContext(ctx);
            vm.runInContext(helper, ctx);
            const source = [[10,10],[30,10],[35,25],[20,30],[10,25]];
            const region = {boundary:source.map(point=>[...point])};
            ctx.moveWholePolygon(region, 12, 8, source);
            assert.deepEqual(region.boundary.map(p=>Array.from(p)), source.map(([x,y])=>[x+12,y+8]));
            ctx.moveWholePolygon(region, 1000, 1000, source);
            assert.deepEqual(region.boundary.map(p=>Array.from(p)), source.map(([x,y])=>[x+64,y+49]));
            ctx.moveWholePolygon(region, -1000, -1000, source);
            assert.deepEqual(region.boundary.map(p=>Array.from(p)), source.map(([x,y])=>[x-10,y-10]));
            assert.deepEqual(source, [[10,10],[30,10],[35,25],[20,30],[10,25]]);
            assert.match(editor, /if\(!verified\|\|e\.button!==0\)return/);
            assert.match(editor, /drag=\{type:'region-move',start:p,before:capturePage\(\),regionIds:\[\.\.\.selectedRegionIds\],moved:false\}/);
            assert.match(editor, /moveWholePolygon\(r,p\[0\]-drag\.start\[0\],p\[1\]-drag\.start\[1\],source\)/);
            assert.match(editor, /else movePolygonEdge\(r,drag\.index,p\[0\]-drag\.start\[0\],p\[1\]-drag\.start\[1\]/);
            assert.doesNotMatch(editor, /canvas\.oncontextmenu=/);
            assert.match(editor, /action\.type==='region-move'\?`\$\{affected\.length\} region\(s\) moved without changing their shapes/);
            const edgeDrag = editor.indexOf("const hit=$('showTruth').checked?nextGeometryHit(e):null;");
            const interiorDrag = editor.indexOf("if(hitsBoundary(page().regions[i].boundary,p)){const r=page().regions[i]");
            assert.ok(edgeDrag>=0 && interiorDrag>edgeDrag);
            const edgeCtx = {zoom:1, $:()=>({checked:true}),
              page:()=>({regions:[{shape:'rectangle',boundary:[[10,10],[30,10],[30,30],[10,30]]}]})};
            vm.createContext(edgeCtx);
            vm.runInContext(editor.split('\n').find(row=>row.startsWith('function distanceToSegment(')),edgeCtx);
            vm.runInContext(editor.split('\n').find(row=>row.startsWith('function hitsBoundary(')),edgeCtx);
            assert.equal(edgeCtx.hitsBoundary([[10,10],[30,10],[30,30],[10,30]],[20,20]),true);
            assert.equal(edgeCtx.hitsBoundary([[10,10],[30,10],[30,30],[10,30]],[80,70]),false);
        """
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed to exercise marquee and group movement')
    def test_marquee_selection_and_group_drag_geometry(self):
        script = r"""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');
            const editor = fs.readFileSync('tools/reference-collection-layout.html', 'utf8');
            const names = ['moveRegionGroup','selectionBounds','pointInPolygon','segmentsIntersect','polygonIntersectsSelection'];
            const ctx = {image:{naturalWidth:100,naturalHeight:80}};
            vm.createContext(ctx);
            for (const name of names) vm.runInContext(editor.split('\n').find(row=>row.startsWith(`function ${name}(`)),ctx);
            const square = [[10,10],[20,10],[20,20],[10,20]];
            assert.equal(ctx.polygonIntersectsSelection(square,[0,0],[15,15]),true);
            assert.equal(ctx.polygonIntersectsSelection(square,[12,12],[18,18]),true);
            assert.equal(ctx.polygonIntersectsSelection(square,[15,0],[16,30]),true);
            assert.equal(ctx.polygonIntersectsSelection(square,[30,30],[40,40]),false);
            const source = {regions:[
              {id:'a',boundary:[[10,10],[20,10],[20,20],[10,20]]},
              {id:'b',boundary:[[40,30],[50,30],[50,40],[40,40]]},
              {id:'c',boundary:[[60,50],[70,50],[70,60],[60,60]]}
            ]};
            const current = structuredClone(source);
            ctx.page = ()=>current;
            ctx.moveRegionGroup(['a','b'],source,1000,1000);
            assert.deepEqual(current.regions[0].boundary.map(p=>Array.from(p)),[[59,49],[69,49],[69,59],[59,59]]);
            assert.deepEqual(current.regions[1].boundary.map(p=>Array.from(p)),[[89,69],[99,69],[99,79],[89,79]]);
            assert.deepEqual(current.regions[2],source.regions[2]);
            ctx.moveRegionGroup(['a','b'],source,-1000,-1000);
            assert.deepEqual(current.regions[0].boundary.map(p=>Array.from(p)),[[0,0],[10,0],[10,10],[0,10]]);
            assert.deepEqual(current.regions[1].boundary.map(p=>Array.from(p)),[[30,20],[40,20],[40,30],[30,30]]);
            assert.match(editor, /drag=\{type:'marquee',start:p,current:p/);
            assert.match(editor, /polygonIntersectsSelection\(r\.boundary,action\.start,action\.current\)/);
            assert.match(editor, /if\(drag\.regionIds\.length>1\)moveRegionGroup/);
            assert.match(editor, /checkpoint\(action\.before\)/);
        """
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_notes_can_anchor_to_multiple_stable_region_ids(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        docs = (ROOT / 'docs/reference-collection-layout.md').read_text(encoding='utf-8')
        for control in ('notes', 'regionNoteText', 'regionNoteAnchors', 'regionNotes', 'saveRegionNote'):
            self.assertIn(f'id="{control}"', editor)
        self.assertIn('function validRegionNotes(p)', editor)
        self.assertIn('new Set(n.region_ids).size===n.region_ids.length', editor)
        self.assertIn('n.region_ids.every(id=>regionIds.has(id))', editor)
        self.assertIn('p.region_notes??=[]', editor)
        self.assertIn('region_notes:[]', editor)
        self.assertIn('region_ids:regionIds,text:textValue', editor)
        self.assertIn('if(!validRegionNotes(p))', editor)
        self.assertIn('note.region_ids=note.region_ids.filter(id=>id!==removed.id)', editor)
        self.assertIn('page().notes=[page().notes,...detached.map', editor)
        self.assertIn('`region_notes`', docs)

    def test_saving_and_reopening_draft_restores_last_viewed_page(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('last_viewed_page_ordinal:page().global_ordinal', editor)
        self.assertIn('data.pages.findIndex(p=>p.global_ordinal===data.last_viewed_page_ordinal)', editor)
        self.assertIn('load(index);say(`Layout draft loaded on page ${page().global_ordinal}.', editor)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed to exercise save view preservation')
    def test_saving_draft_preserves_canvas_position_after_picker_and_status(self):
        script = r"""
            const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
            const editor=fs.readFileSync('tools/reference-collection-layout.html','utf8');
            const line=name=>editor.split('\n').find(row=>row.startsWith(`function ${name}(`));
            const save=editor.slice(editor.indexOf('async function exportDraft('),
              editor.indexOf('\nfunction seedLayout('));
            const viewport={scrollLeft:413,scrollTop:907},buttons={export:{},saveDraftAs:{}},frames=[];
            const browser={scrollX:0,scrollY:580,scrollTo(x,y){this.scrollX=x;this.scrollY=y}};
            const page={global_ordinal:197,review_status:'unreviewed',regions:[],region_notes:[]};
            const ctx={collection:{layout_golden_set_id:'HTH-GOLDEN-0002-LAYOUT',pages:[page]},
              page:()=>page,zoom:3,fitMode:false,dirty:true,canvasViewInputRevision:0,
              window:browser,
              $:id=>id==='viewport'?viewport:buttons[id],
              say:()=>{viewport.scrollLeft=0;viewport.scrollTop=0;browser.scrollY=0},
              draftJsonSaver:{save:async()=>{viewport.scrollLeft=0;viewport.scrollTop=0;
                browser.scrollY=0;return{kind:'written',filename:'draft.json'}}},
              validRegionNotes:()=>true,validReadingOrder:()=>true,validPolygon:()=>true,
              requestAnimationFrame:callback=>frames.push(callback)};
            vm.createContext(ctx);
            vm.runInContext([line('canvasView'),line('restoreCanvasView'),save].join('\n'),ctx);
            vm.runInContext('exportDraft()',ctx).then(()=>{
              assert.deepEqual([viewport.scrollLeft,viewport.scrollTop],[413,907]);
              assert.equal(browser.scrollY,580);
              assert.equal(ctx.dirty,false);
              assert.equal(buttons.export.disabled,false);
              assert.equal(buttons.saveDraftAs.disabled,false);
              assert.equal(frames.length,1);
              viewport.scrollLeft=0;viewport.scrollTop=0;browser.scrollY=0;frames[0]();
              assert.deepEqual([viewport.scrollLeft,viewport.scrollTop],[413,907]);
              assert.equal(browser.scrollY,580);
              return vm.runInContext('exportDraft()',ctx);
            }).then(()=>{
              viewport.scrollLeft=22;viewport.scrollTop=33;browser.scrollY=44;
              ctx.canvasViewInputRevision++;
              frames[1]();
              assert.deepEqual([viewport.scrollLeft,viewport.scrollTop],[22,33]);
              assert.equal(browser.scrollY,44);
            }).catch(error=>{console.error(error);process.exitCode=1});
        """
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reading_order_rows_render_and_matching_suggestion_gives_feedback(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('for(const[position,id]of(p.reading_order||[]).entries())', editor)
        self.assertNotIn('for(const[id,position]of(p.reading_order||[]).entries())', editor)
        self.assertIn('number.textContent=`${position+1}.`', editor)
        self.assertIn('orderList.appendChild(row)', editor)
        self.assertIn('The layout suggestion already matches the numbered sequence below.', editor)
        self.assertIn('Check the numbered sequence below, then confirm it.', editor)

    def test_both_editors_share_confirmed_json_save_behavior(self):
        detector = (ROOT / 'tools/reference-collection-editor.html').read_text(encoding='utf-8')
        layout = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        saver = (ROOT / 'tools/reference-collection-save.js').read_text(encoding='utf-8')
        for editor in (detector, layout):
            self.assertIn('src="reference-collection-save.js"', editor)
            self.assertIn('HTH_REFERENCE_SAVE.createJsonSaver()', editor)
        self.assertIn('await window.showSaveFilePicker(', saver)
        self.assertIn('await writable.write(contents)', saver)
        self.assertIn('await writable.close()', saver)
        self.assertIn("return {kind: 'written'", saver)
        self.assertIn("return {kind: 'download-requested'", saver)
        self.assertIn('button id="export" type="button">Save draft JSON', layout)
        self.assertIn('button id="saveDraftAs" type="button">Save draft as…', layout)
        self.assertLess(layout.index('Import Kraken proposals'), layout.index('button id="openDraft"'))
        self.assertLess(layout.index('button id="openDraft"'), layout.index('button id="export"'))
        self.assertIn('draftJsonSaver.save(filename,data,{saveAs})', layout)
        self.assertIn("$('export').onclick=()=>exportDraft()", layout)
        self.assertIn("$('saveDraftAs').onclick=()=>exportDraft(true)", layout)
        self.assertIn('async function save(name, contents, {saveAs = false} = {})', saver)
        self.assertIn('if (saveAs || !destination)', saver)
        self.assertIn('handle = destination;', saver)
        self.assertIn('handleName = name;', saver)
        self.assertIn("if(result.kind==='written'){dirty=false", layout)
        self.assertIn('Browser download requested', layout)
        self.assertNotIn('Draft downloaded.', layout)

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

    def test_director_keeps_canvas_scrollbar_inside_visible_window(self):
        director = (ROOT / 'tools/reference-collection-director.html').read_text(encoding='utf-8')
        layout = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('body{margin:0;height:100vh;display:flex;flex-direction:column;overflow:hidden}', director)
        self.assertIn('.frame{width:100%;flex:1;min-height:0;border:0;display:block}', director)
        self.assertNotIn('height:calc(100vh - 164px)', director)
        self.assertIn('.viewport{flex:1;min-height:0;overflow:auto', layout)
        self.assertFalse((ROOT / 'tools/reference-collection-editor-multidetector.html').exists())

    def test_top_controls_can_be_hidden_without_hiding_side_panes(self):
        director = (ROOT / 'tools/reference-collection-director.html').read_text(encoding='utf-8')
        layout = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        detector = (ROOT / 'tools/reference-collection-editor.html').read_text(encoding='utf-8')
        for page in (director, layout):
            self.assertIn('id="toggleTopControls"', page)
            self.assertIn('Hide top controls', page)
            self.assertIn('Show top controls', page)
        self.assertIn('HTH_REFERENCE_TOP_CONTROLS', director)
        self.assertIn('HTH_REFERENCE_TOP_CONTROLS', layout)
        self.assertIn('HTH_REFERENCE_TOP_CONTROLS', detector)
        self.assertIn('body.top-controls-hidden .workspace>.pages', layout)
        self.assertIn('body.top-controls-hidden .workspace>.view-controls', layout)
        self.assertIn('body.top-controls-hidden .workspace>.overlay-controls', layout)
        self.assertIn('<aside class="tools-pane"', layout)
        self.assertIn('<aside class="metadata-pane"', layout)
        self.assertNotIn('body.top-controls-hidden aside', layout)
        self.assertIn('body.top-controls-hidden .workspace{grid-template-rows:minmax(0,1fr)}', detector)
        self.assertIn("if(image&&fitMode&&width!==lastViewportWidth)fit()", layout)

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
        self.assertIn('main{display:grid;grid-template-columns:330px minmax(0,1fr) 330px;flex:1;min-height:0}', editor)
        self.assertIn('.viewport{flex:1;min-height:0;overflow:auto', editor)

    def test_layout_tool_and_metadata_panes_and_action_order(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        tools = editor.split('<aside class="tools-pane"', 1)[1].split('</aside>', 1)[0]
        metadata = editor.split('<aside class="metadata-pane"', 1)[1].split('</aside>', 1)[0]
        for label in ('Region class', 'Region reading order', 'Review notes'):
            self.assertIn(f'<h2>{label}</h2>', metadata)
            self.assertNotIn(f'<h2>{label}</h2>', tools)
        self.assertIn('<details class="tool-help"><summary>Help</summary>', tools)
        self.assertIn('grid-template-columns:repeat(2,minmax(0,1fr))', editor)
        self.assertIn('.region-actions button{width:100%;height:48px', editor)
        actions = tools.split('<div class="region-actions">', 1)[1].split('</div>', 1)[0]
        for control in ('makeRectangle', 'convertPolygon', 'duplicateRegions',
                        'mirrorBoundary', 'addVertex', 'deleteVertex', 'deleteRegion'):
            self.assertIn(f'id="{control}"', actions)
        self.assertLess(actions.index('id="makeRectangle"'), actions.index('id="duplicateRegions"'))
        self.assertLess(actions.index('id="convertPolygon"'), actions.index('id="mirrorBoundary"'))
        self.assertLess(actions.index('id="deleteVertex"'), actions.index('id="deleteRegion"'))
        self.assertIn('id="duplicateRegions" type="button">Clone selected', actions)
        self.assertIn('id="deleteVertex" type="button">Delete vertex', actions)
        self.assertIn('id="deleteRegion" type="button">Delete region', actions)
        self.assertLess(tools.index('id="nudgeUp"'), tools.index('id="nudgeDown"'))
        self.assertIn('.nudge-controls #nudgeUp{grid-column:2;grid-row:2}', editor)
        self.assertIn('.nudge-controls #nudgeDown{grid-column:2;grid-row:3}', editor)

    def test_layout_zoom_changes_only_page_view(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        for control in ('zoomOut', 'zoomFit', 'zoomIn', 'zoomLevel'):
            self.assertIn(f'id="{control}"', editor)
        self.assertIn("$('viewport').addEventListener('wheel'", editor)
        self.assertIn('if(!e.ctrlKey||!image)return', editor)
        self.assertIn('function setZoom(value,clientX,clientY)', editor)
        self.assertIn('Math.round((e.clientX-rect.left)/zoom)', editor)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed to exercise undo restoration')
    def test_undo_restores_geometry_without_reloading_or_moving_canvas(self):
        script = r"""
            const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
            const editor=fs.readFileSync('tools/reference-collection-layout.html','utf8');
            const restore=editor.split('\n').find(row=>row.startsWith('function restorePage('));
            const viewport={scrollLeft:317,scrollTop:842};
            const ctx={structuredClone,index:0,collection:{pages:[{global_ordinal:1,regions:[]},
              {global_ordinal:2,regions:[]}],status:'reviewed'},proposals:new Map(),
              selected:{type:'region',index:0},selectedRegionIds:new Set(['r1']),
              selectedVertex:1,selectedEdge:-1,noteDraftAnchors:new Set(['r1']),
              noteDraftText:'old',editingNoteId:'n1',calls:[],
              $:()=>viewport,clearProposalSelection:()=>ctx.calls.push('clear'),
              setMode:()=>ctx.calls.push('mode'),renderLists:()=>ctx.calls.push('lists'),
              draw:()=>{ctx.calls.push('draw');viewport.scrollLeft=0;viewport.scrollTop=0},
              renderPages:()=>ctx.calls.push('pages'),load:()=>{throw Error('Undo reloaded the image')}};
            vm.createContext(ctx);vm.runInContext(restore,ctx);
            const same={ordinal:1,page:{global_ordinal:1,regions:[{id:'r1'}]},proposals:[]};
            vm.runInContext('restorePage(same)',Object.assign(ctx,{same}));
            assert.equal(ctx.collection.pages[0].regions[0].id,'r1');
            assert.deepEqual([viewport.scrollLeft,viewport.scrollTop],[317,842]);
            assert.deepEqual(ctx.calls,['clear','mode','lists','draw','pages']);
            assert.equal(ctx.selected,null);
            ctx.calls=[];ctx.selected={type:'region',index:0};
            const other={ordinal:2,page:{global_ordinal:2,regions:[{id:'r2'}]},proposals:[]};
            vm.runInContext('restorePage(other)',Object.assign(ctx,{other}));
            assert.equal(ctx.collection.pages[1].regions[0].id,'r2');
            assert.deepEqual([viewport.scrollLeft,viewport.scrollTop],[317,842]);
            assert.deepEqual(ctx.calls,['pages']);
            assert.equal(ctx.selected.index,0);
        """
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_layout_zoom_follows_current_page_and_remembers_tweaked_pages(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn("let pageZooms=new Map(),defaultZoom={mode:'fit'}", editor)
        self.assertIn('pageZooms.get(p.global_ordinal)||defaultZoom', editor)
        self.assertIn('rememberZoom({mode:\'scale\',zoom})', editor)
        self.assertIn("$('zoomFit').onclick=()=>fit(true)", editor)
        self.assertIn('localStorage.setItem(zoomStorageKey()', editor)
        self.assertIn('restoreZoomPreferences();showResults()', editor)
        self.assertNotIn('page().zoom', editor)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed to exercise layout zoom')
    def test_layout_zoom_reaches_eight_hundred_percent_without_huge_canvas_bitmap(self):
        script = r"""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');
            const editor = fs.readFileSync('tools/reference-collection-layout.html','utf8');
            const calls = [];
            const viewport = {clientWidth:600,clientHeight:400,scrollLeft:0,scrollTop:0,
              getBoundingClientRect:()=>({left:0,top:0})};
            const canvas = {style:{},parentElement:viewport,getBoundingClientRect:()=>({left:0,top:0})};
            const ctx = {image:{naturalWidth:1600,naturalHeight:1200},zoom:1,canvas,fitMode:true,
              ctx:{setTransform:(...a)=>calls.push(['transform',...a]),
                fillRect:(...a)=>calls.push(['fill',...a]),
                drawImage:(...a)=>calls.push(['image',...a])},
              updateZoom:()=>{},rememberZoom:()=>{}};
            vm.createContext(ctx);
            vm.runInContext(editor.split('\n').find(row=>row.startsWith('function validZoomPreference(')),ctx);
            vm.runInContext(editor.split('\n').find(row=>row.startsWith('function setZoom(')),ctx);
            const drawStart=editor.split('function draw(){')[1].split('const currentProposals=')[0];
            vm.runInContext('function draw(){'+drawStart+'}',ctx);
            assert.equal(ctx.validZoomPreference({mode:'scale',zoom:8}).zoom,8);
            assert.equal(ctx.validZoomPreference({mode:'scale',zoom:8.01}),null);
            ctx.setZoom(100);
            assert.equal(ctx.zoom,8);
            assert.equal(canvas.width,3200);
            assert.equal(canvas.height,2400);
            assert.equal(canvas.style.width,'12800px');
            assert.equal(canvas.style.height,'9600px');
            assert.deepEqual(calls.find(c=>c[0]==='transform'),['transform',.25,0,0,.25,0,0]);
            assert.deepEqual(calls.find(c=>c[0]==='image').slice(2),[0,0,12800,9600]);
        """
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_layout_proposals_have_visible_outline_without_heavier_fill(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn("drawPath(r.boundary,'#704000','rgba(229,184,92,.07)',2.25,'#f3c76a',overlayEffect(false))", editor)
        self.assertIn("if(selectedProposalIds.has(r.id)&&proposalShown(r))drawPath(r.boundary,'#0969da','rgba(229,184,92,.07)',2.25,'#dff5ff',overlayEffect(true))", editor)
        self.assertIn("selectedRegionIds.has(r.id)?'#004d2b':'#006b3b'", editor)
        self.assertIn("drawPath(r.boundary,'#004d2b','rgba(81,220,145,.05)',2,'#caffdf',effect)", editor)
        self.assertLess(editor.index("drawPath(r.boundary,'#704000'"), editor.index("drawPath(r.boundary,'#0969da'"))

    def test_layout_annotation_controls_are_visible_and_recoverable(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        for control in ('selectTool', 'rectangle', 'polygon', 'finishPolygon',
                        'cancelDrawing', 'undo', 'redo', 'deleteVertex',
                        'deleteRegion', 'dismissProposal', 'toolState',
                        'makeRectangle', 'addVertex'):
            self.assertIn(f'id="{control}"', editor)
        self.assertIn(".classList.toggle('tool-active'", editor)
        self.assertIn('function selectAt(p,event={})', editor)
        self.assertIn('function drawDraft()', editor)
        self.assertIn('function capturePage(', editor)
        self.assertIn('function undo()', editor)
        self.assertIn('function redo()', editor)
        self.assertIn('if(!validPolygon(updated,image.naturalWidth,image.naturalHeight))', editor)
        self.assertIn('if(action.type===\'rectangle\')', editor)

    def test_review_advances_only_after_regions_and_reading_order_are_reviewed(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn("function pageFullyReviewed(p){return p.review_status==='reviewed'&&p.reading_order_status==='reviewed'}", editor)
        self.assertIn('function nextIncompletePageIndex(fromIndex)', editor)
        self.assertIn('if(!pageFullyReviewed(collection.pages[candidate]))return candidate', editor)
        self.assertIn('function advanceAfterPageReview(reviewedOrdinal)', editor)
        self.assertIn('if(!pageFullyReviewed(page()))', editor)
        self.assertIn('Confirm reading order to continue.', editor)
        self.assertIn('Mark regions reviewed to continue.', editor)
        self.assertEqual(editor.count('advanceAfterPageReview(reviewedOrdinal)};'), 2)
        self.assertIn('All ${collection.pages.length} pages complete', editor)

    def test_rectangles_keep_axis_aligned_corner_and_edge_editing(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('function rectanglePoints(left,top,right,bottom)', editor)
        self.assertIn('function isRectangle(r)', editor)
        self.assertIn('function resizeRectangle(r,handle,p)', editor)
        self.assertIn("drag={type:hit.type,index:hit.index,start:p,before:capturePage(),moved:false}", editor)
        self.assertIn("addRegion(points,$('kind').value,'rectangle')", editor)
        self.assertIn("r.shape='rectangle'", editor)
        self.assertIn("if(isRectangle(r)||r.boundary.length<=3)return", editor)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed to exercise overlapping handle selection')
    def test_repeated_clicks_cycle_nearby_vertices_and_edges(self):
        script = r"""
            const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
            const editor=fs.readFileSync('tools/reference-collection-layout.html','utf8');
            const line=name=>editor.split('\n').find(row=>row.startsWith(`function ${name}(`));
            const blocks=['distanceToSegment','canvasPoint','nearbyGeometryHits','nextGeometryHit']
              .map(name=>name==='nearbyGeometryHits'||name==='nextGeometryHit'
                ?editor.slice(editor.indexOf(`function ${name}(`),editor.indexOf('\nfunction ',editor.indexOf(`function ${name}(`)+1))
                :line(name));
            const boundary=[[10,10],[30,10],[30,30],[10,30]];
            const page={global_ordinal:5,regions:[
              {id:'r1',boundary},{id:'r2',boundary:boundary.map(point=>[...point])}]};
            const ctx={zoom:2,geometryHitCycle:null,page:()=>page,
              canvas:{getBoundingClientRect:()=>({left:0,top:0})}};
            vm.createContext(ctx);vm.runInContext(blocks.join('\n'),ctx);
            const click={clientX:20,clientY:20};
            const candidates=vm.runInContext('nearbyGeometryHits([10,10])',ctx);
            assert.equal(candidates.length,6);
            assert.equal(candidates[0].type,'vertex');
            const chosen=[];
            for(let i=0;i<6;i++){
              const hit=ctx.nextGeometryHit(click);
              chosen.push(`${hit.type}:${hit.regionIndex}:${hit.index}`);
            }
            assert.equal(new Set(chosen).size,6);
            assert.equal(ctx.nextGeometryHit(click).type,'vertex');
            assert.equal(ctx.geometryHitCycle.index,0);
            assert.equal(vm.runInContext('nearbyGeometryHits([15.1,15.1]).length',ctx),0);
            assert.equal(ctx.nextGeometryHit({clientX:100,clientY:100}),null);
            assert.equal(ctx.geometryHitCycle,null);
            assert.match(editor,/if\(geometryHitCycle\?\.hits\.length>1.*e\.preventDefault\(\);return/);
        """
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selected_region_can_convert_between_rectangle_and_polygon(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('id="makeRectangle" type="button">Convert to rectangle', editor)
        self.assertIn('id="convertPolygon" type="button">Convert to polygon', editor)
        self.assertIn("$('convertPolygon').disabled=!verified||selected?.type!=='region'||!isRectangle", editor)
        self.assertIn("$('convertPolygon').onclick=()=>", editor)
        self.assertIn("if(!isRectangle(r))return;checkpoint();r.shape='polygon'", editor)
        self.assertIn('Rectangle converted to a polygon with the same four corners.', editor)
        self.assertIn("if(!r||isRectangle(r))return;const xs=", editor)
        self.assertIn('Polygon converted to its bounding rectangle.', editor)
        self.assertIn("${isRectangle(r)?'rectangle':'polygon'}", editor)

    def test_neighbor_boundary_mirror_copies_source_vertices_and_is_undoable(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        boundary = (ROOT / 'tools/reference-collection-boundary.js').read_text(encoding='utf-8')
        self.assertIn('src="reference-collection-boundary.js"', editor)
        self.assertIn('id="mirrorBoundary" type="button">Mirror boundary…', editor)
        self.assertIn("$('mirrorBoundary').onclick=()=>", editor)
        self.assertIn('if(mirrorSource){e.preventDefault();mirrorToEdge(p);return}', editor)
        self.assertIn('window.HTH_REFERENCE_BOUNDARY.mirror(source.boundary,sourceRef.edge,target.boundary,edge)', editor)
        self.assertIn('checkpoint();target.boundary=result.boundary;target.shape=\'polygon\'', editor)
        self.assertIn('if (!simplePolygon(boundary)) continue;', boundary)
        self.assertIn('const copied = sourceArc.indices.map(index => [...source[index]])', boundary)
        self.assertIn('const boundary = [...copied, ...rest.slice(1, -1)', boundary)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed to exercise geometry shortcuts')
    def test_geometry_shortcuts_delete_edges_and_mirror_copied_edges(self):
        script = r"""
            const assert=require('node:assert/strict'), fs=require('node:fs'), vm=require('node:vm');
            const editor=fs.readFileSync('tools/reference-collection-layout.html','utf8');
            const line=name=>editor.split('\n').find(row=>row.startsWith(`function ${name}(`));
            const region={id:'r1',shape:'polygon',boundary:[[0,0],[10,0],[12,5],[10,10],[0,10]]};
            const target={id:'r2',shape:'polygon',boundary:[[0,12],[10,12],[10,22],[0,22]]};
            const page={global_ordinal:1,regions:[region,target]};
            let mirrored=null, notices=[];
            const ctx={page:()=>page,image:{naturalWidth:100,naturalHeight:100},verified:true,
              selected:{type:'region',index:0},selectedVertex:-1,selectedEdge:0,
              selectedRegionIds:new Set(['r1']),geometryClipboard:null,mirrorSource:null,
              checkpoint:()=>{ctx.checkpoints++},checkpoints:0,invalidation:()=>{},
              renderLists:()=>{},draw:()=>{},say:(message)=>notices.push(message),
              $:()=>({checked:true}),window:{HTH_REFERENCE_BOUNDARY:{mirror:(source,seam,dest,edge)=>{
                mirrored={seam,edge};return{boundary:dest,copiedVertices:2};}}}};
            vm.createContext(ctx);
            vm.runInContext(['validPolygon','isRectangle','distanceToSegment',
              'applyMirroredBoundary','pasteCopiedEdge','deleteSelectedEdge',
              'copyGeometrySelection'].map(line).join('\n'),ctx);
            assert.equal(vm.runInContext('deleteSelectedEdge()',ctx),true);
            assert.equal(ctx.checkpoints,1);
            assert.deepEqual(Array.from(region.boundary[0]),[10,0]);
            assert.equal(ctx.selectedVertex,0);
            assert.equal(ctx.selectedEdge,-1);
            region.boundary=[[0,0],[10,0],[12,5],[10,10],[0,10]];
            ctx.selectedVertex=-1;ctx.selectedEdge=3;
            assert.equal(vm.runInContext('deleteSelectedEdge()',ctx),true);
            assert.deepEqual(Array.from(region.boundary.at(-1)),[10,10]);
            assert.equal(ctx.selectedVertex,3);
            region.boundary=[[0,0],[10,0],[10,10]];
            ctx.selectedVertex=-1;ctx.selectedEdge=0;
            assert.equal(vm.runInContext('deleteSelectedEdge()',ctx),false);
            assert.equal(ctx.checkpoints,2);
            region.boundary=[[0,0],[10,0],[10,10],[0,10]];
            ctx.selectedEdge=2;
            assert.equal(vm.runInContext('copyGeometrySelection()',ctx),true);
            assert.equal(ctx.geometryClipboard.type,'edge');
            ctx.selected={type:'region',index:1};ctx.selectedEdge=-1;
            assert.equal(vm.runInContext('pasteCopiedEdge()',ctx),true);
            assert.deepEqual(mirrored,{seam:2,edge:0});
            assert.equal(ctx.checkpoints,3);
            assert.deepEqual(Array.from(ctx.selectedRegionIds),['r2']);
        """
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed to exercise keyboard shortcuts')
    def test_geometry_keyboard_shortcuts_route_by_selection(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn("modified&&key==='c'&&copyGeometrySelection()", editor)
        self.assertIn("modified&&key==='v'&&pasteGeometrySelection()", editor)
        self.assertIn("modified&&key==='x'&&deleteGeometrySelection()", editor)
        self.assertIn("e.key==='Delete'&&deleteGeometrySelection()", editor)
        self.assertIn("if(selectedVertex>=0){$('deleteVertex').click();return true}", editor)
        self.assertIn('if(selectedEdge>=0){deleteSelectedEdge();return true}', editor)
        self.assertIn("$('deleteRegion').click();return true", editor)
        script = r"""
            const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
            const editor=fs.readFileSync('tools/reference-collection-layout.html','utf8');
            const handler=editor.split('\n').find(row=>row.startsWith('window.onkeydown=e=>'));
            const calls=[],ctx={window:{},document:{activeElement:{tagName:'BODY'}},
              mode:'select',copyGeometrySelection:()=>{calls.push('copy');return true},
              pasteGeometrySelection:()=>{calls.push('paste');return true},
              deleteGeometrySelection:()=>{calls.push('delete');return true},
              canNudge:()=>false,geometryClipboard:{type:'vertex'},selected:null};
            vm.createContext(ctx);vm.runInContext(handler,ctx);
            const key=(value,ctrl=false)=>{let prevented=false;ctx.window.onkeydown({key:value,
              ctrlKey:ctrl,metaKey:false,altKey:false,shiftKey:false,
              preventDefault:()=>{prevented=true}});return prevented};
            assert.equal(key('c',true),true);
            assert.equal(key('v',true),true);
            assert.equal(key('x',true),true);
            assert.equal(ctx.geometryClipboard,null);
            assert.equal(key('Delete'),true);
            assert.deepEqual(calls,['copy','paste','delete','delete']);
            ctx.document.activeElement={tagName:'INPUT'};
            assert.equal(key('Delete'),false);
            assert.equal(key('c',true),false);
            assert.equal(calls.length,4);
        """
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed to execute the boundary helper')
    def test_mirror_uses_facing_corners_when_extrema_are_on_opposite_side(self):
        script = """
            const assert = require('node:assert/strict');
            global.window = {};
            require('./tools/reference-collection-boundary.js');
            const source = [[0,0],[100,0],[99,40],[80,40],[70,50],[40,50],[1,40]];
            const target = [[10,70],[90,70],[90,100],[10,100]];
            const result = window.HTH_REFERENCE_BOUNDARY.mirror(source, 4, target, 0);
            assert.deepEqual(result.boundary, [[1,40],[40,50],[70,50],[80,40],[99,40],[90,100],[10,100]]);
            assert.equal(result.copiedVertices, 5);
            assert.deepEqual(target, [[10,70],[90,70],[90,100],[10,100]]);
        """
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

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
        self.assertIn('canNudge(...arrows[e.key])', editor)
        self.assertIn('moveRegionGroup(ids,before.page,dx,dy)', editor)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed to exercise group nudging')
    def test_arrow_nudges_selected_regions_together_and_respects_image_bounds(self):
        script = r"""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');
            const editor = fs.readFileSync('tools/reference-collection-layout.html','utf8');
            const original = {regions:[
              {id:'a',boundary:[[10,10],[20,10],[20,20],[10,20]]},
              {id:'b',boundary:[[50,30],[60,30],[60,40],[50,40]]},
              {id:'c',boundary:[[70,50],[80,50],[80,60],[70,60]]}
            ]};
            const current = structuredClone(original);
            let checkpoints = 0, invalidations = 0;
            const ctx = {image:{naturalWidth:100,naturalHeight:80},verified:true,
              selected:{type:'region',index:0},selectedRegionIds:new Set(['a','b']),
              selectedVertex:-1,selectedEdge:-1,collection:{pages:[current]},index:0,
              $:()=>({checked:true}),page:()=>current,structuredClone,
              capturePage:()=>({page:structuredClone(current)}),
              checkpoint:()=>{checkpoints++},invalidation:()=>{invalidations++},
              draw:()=>{},say:()=>{},validPolygon:()=>true};
            vm.createContext(ctx);
            for(const name of ['moveRegionGroup','canNudge','nudgeSelection'])
              vm.runInContext(editor.split('\n').find(row=>row.startsWith(`function ${name}(`)),ctx);
            assert.equal(ctx.nudgeSelection(1,0),true);
            assert.deepEqual(current.regions[0].boundary.map(p=>Array.from(p)),[[11,10],[21,10],[21,20],[11,20]]);
            assert.deepEqual(current.regions[1].boundary.map(p=>Array.from(p)),[[51,30],[61,30],[61,40],[51,40]]);
            assert.deepEqual(current.regions[2],original.regions[2]);
            assert.equal(checkpoints,1);
            assert.equal(invalidations,1);
            assert.equal(ctx.nudgeSelection(1,0,false),true);
            assert.equal(checkpoints,1);
            assert.deepEqual([...ctx.selectedRegionIds],['a','b']);
            ctx.moveRegionGroup(['a','b'],{regions:current.regions.map(r=>({id:r.id,boundary:r.boundary.map(p=>[...p])}))},-1000,0);
            assert.equal(ctx.nudgeSelection(-1,0),false);
            assert.equal(checkpoints,1);
        """
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_nudge_buttons_repeat_on_hold_as_one_undoable_edit(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('function installNudgeControl(id,dx,dy)', editor)
        self.assertIn('button.onpointerdown=e=>', editor)
        self.assertIn('nudgeSelection(dx,dy,false)', editor)
        self.assertIn('button.onpointerup=stop', editor)
        self.assertIn('button.onpointercancel=stop', editor)
        self.assertIn('button.onpointerleave=stop', editor)
        self.assertIn('if(e.detail===0)nudgeSelection(dx,dy)', editor)

    def test_add_vertex_defaults_to_right_midpoint_and_advances_selected_edge(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('id="addVertex"', editor)
        self.assertIn('function rightMidpointEdge(boundary)', editor)
        self.assertIn('function insertVertexOnEdge(edge,point,selectRightEdge=false)', editor)
        self.assertIn('if(!r||isRectangle(r)||edge<0', editor)
        self.assertIn("$('addVertex').disabled=!verified||selected?.type!=='region'||isRectangle", editor)
        self.assertIn('edge=hadSelectedEdge?selectedEdge:selectedVertex>=0?rightAdjacentEdge(r.boundary,selectedVertex):rightMidpointEdge(r.boundary)', editor)
        self.assertIn('selectedEdge=afterMidX>=beforeMidX?edge+1:edge', editor)
        self.assertIn('selectedVertex=edge+1;selectedEdge=-1', editor)

    def test_delete_vertex_selects_rightmost_adjacent_vertex_for_repeated_deletion(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn('const deleted=selectedVertex,updated=r.boundary.filter((_,i)=>i!==deleted)', editor)
        self.assertIn('const before=(deleted-1+updated.length)%updated.length,after=deleted%updated.length', editor)
        self.assertIn('selectedVertex=updated[after][0]>=updated[before][0]?after:before', editor)
        self.assertIn("if(!validPolygon(updated,image.naturalWidth,image.naturalHeight)){say('Deleting that vertex would make the region invalid.'", editor)

    def test_drawing_tools_remain_active_and_workpanes_are_left_aligned(self):
        layout = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        detector = (ROOT / 'tools/reference-collection-editor.html').read_text(encoding='utf-8')
        director = (ROOT / 'tools/reference-collection-director.html').read_text(encoding='utf-8')
        self.assertIn("if(addRegion(points,$('kind').value,'rectangle'))say('Rectangle added.", layout)
        self.assertIn('function finishPolygon()', layout)
        self.assertNotIn("if(addRegion(draftPoints.map(p=>[...p]),$('kind').value))setMode('select')", layout)
        self.assertIn('grid-template-columns:330px minmax(0,1fr) 330px', layout)
        self.assertIn('grid-template-columns:360px minmax(0,1fr)', detector)
        for page in (layout, detector, director):
            self.assertIn('.header-title{order:2;margin-left:auto}', page)

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
        self.assertIn('for(const id of chosen)dismissed.push(id)', editor)
        self.assertIn('page().dismissed_proposal_ids=page().dismissed_proposal_ids.filter', editor)
        self.assertIn("id:await candidateId('kraken',kind,boundary)", editor)
        self.assertIn("JSON.stringify({...collection,last_viewed_page_ordinal:page().global_ordinal},null,2)", editor)
        self.assertIn(".filter(candidate=>!isDismissed(candidate))", editor)
        self.assertNotIn("proposals.get(page().global_ordinal).splice(selected.index,1)", editor)

    def test_layout_proposals_support_multi_selection_and_batch_actions(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        for control in ('adoptSelected', 'dismissProposal', 'restoreProposal', 'adoptAll', 'dismissAll'):
            self.assertIn(f'id="{control}"', editor)
        self.assertIn('selectedProposalIds=new Set([candidate.id])', editor)
        self.assertIn('event.shiftKey&&proposalSelectionAnchorId', editor)
        self.assertIn('event.ctrlKey||event.metaKey', editor)
        self.assertIn('const visible=current.filter(proposalShown)', editor)
        self.assertIn('button.onclick=event=>selectProposal(i,event)', editor)
        self.assertIn('selectAt(p,e)', editor)
        self.assertIn('function adoptCandidates(candidates)', editor)
        self.assertIn('function dismissCandidates(candidates)', editor)
        self.assertIn('function restoreCandidates(candidates)', editor)
        self.assertIn("$('dismissAll').onclick=()=>dismissCandidates", editor)
        self.assertIn('checkpoint();const dismissed=page().dismissed_proposal_ids', editor)
        self.assertIn('checkpoint();const used=new Set(page().regions.map(r=>r.id))', editor)

    def test_overlay_emphasis_can_mute_or_highlight_selected_or_other_regions(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        for mode in ('normal', 'mute-selected', 'highlight-selected', 'mute-others', 'highlight-others'):
            self.assertIn(f'<option value="{mode}">', editor)
        self.assertIn('function overlayEffect(isSelected)', editor)
        self.assertIn("if(effect==='mute')ctx.globalAlpha=.13", editor)
        self.assertIn("if(effect==='highlight'){color='#e44cff'", editor)
        self.assertIn("$('overlayFocus').onchange=()=>{updateOverlayFocusInfo();draw()}", editor)
        self.assertIn('image pixels and draft JSON are unchanged',
                      (ROOT / 'docs/reference-collection-layout.md').read_text(encoding='utf-8'))

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
        self.assertIn('.workspace{grid-column:2;grid-row:1;min-width:0;min-height:0;display:grid;grid-template-rows:auto auto auto minmax(0,1fr)}', editor)
        self.assertIn('main{display:grid;grid-template-columns:360px minmax(0,1fr);flex:1;min-height:0}', editor)


if __name__ == '__main__':
    unittest.main()
