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
        self.assertIn("sha256(imageBytes) !== expected.get(number)", importer)

    def test_layout_candidates_cannot_become_approved_truth_implicitly(self):
        editor = (ROOT / 'tools/reference-collection-layout.html').read_text(encoding='utf-8')
        self.assertIn("proposals=incoming", editor)
        self.assertIn('Adopt selected', editor)
        self.assertIn('Mark page reviewed', editor)
        self.assertIn("page().review_status='unreviewed'", editor)
        self.assertIn("collection.status='draft'", editor)
        self.assertIn('source image SHA-256 differs from the frozen Golden Set', editor)
        self.assertIn('candidate source identity mismatch', editor)


if __name__ == '__main__':
    unittest.main()
