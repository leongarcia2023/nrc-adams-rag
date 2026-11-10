import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adams_rag.export import document_markdown
from adams_rag.extraction_review import filter_rows, for_version, record
from adams_rag.research import hybrid_retrieve, research
from adams_rag.retrieval import retrieve
from adams_rag.storage import StoragePause
from adams_rag.store import Store
from adams_rag.text_quality import get


class ExtractionReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name)/'test.sqlite')
        self.acc = 'ML24017A120'
        self.put(['Valve introduction with omitted sender', 'Valve table has corrupted values', 'Valve narrative retained'])
        self.old_hash = self.store.db.execute('SELECT text_hash FROM documents').fetchone()[0]

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def put(self,pages):
        with self.store.db:
            self.store.put({'AccessionNumber':self.acc,'DocumentTitle':'Valve report'}, 'pdf',pages)

    def review(self,page=2,disposition='exclude',text_hash=None):
        return record(self.store,self.acc,text_hash or self.old_hash,page,disposition,
                      ['numeric_ocr_error'],'Visual comparison found a corrupted exponent.')

    def test_all_retrieval_routes_exclude_known_bad_pages_without_deleting_text(self):
        self.review(1,'warn'); self.review()
        raw=[dict(r) for r in self.store.db.execute('SELECT c.*,d.title,d.url,d.provenance,d.document_date FROM chunks c JOIN documents d USING(accession)')]
        class Semantic:
            def search(self,*args):
                return list(reversed(raw))
        excluded=[]
        sources=hybrid_retrieve(self.store,'Valve',semantic=Semantic(),exclusions=excluded)
        self.assertEqual({s['page'] for s in sources},{1,3})
        self.assertEqual([r['page'] for r in excluded],[2])
        self.assertNotIn(2,{s['page'] for s in retrieve(self.store,'Valve')})
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM chunks').fetchone()[0],3)
        self.assertIn('corrupted values',document_markdown(self.store,self.acc))
        self.assertIn('reviewed_extraction_issue',document_markdown(self.store,self.acc))

    def test_whole_document_exclusion_prevents_generation_and_reports_gap(self):
        self.review(0)
        class Engine:
            def generate(self,*args):
                raise AssertionError('Known-bad text must not reach generation')
        result=research(self.store,'Valve',engine=Engine(),live=False)
        self.assertFalse(result['sources'])
        self.assertTrue(result['insufficient_evidence'])
        self.assertEqual(len(result['excluded_evidence']),1)
        self.assertTrue(any('Known extraction errors' in w for w in result['warnings']))

    def test_visual_source_with_no_first_page_text_keeps_source_and_reports_gap(self):
        self.put(['', 'Corrupted aerial photograph labels'])
        current_hash = self.store.db.execute('SELECT text_hash FROM documents').fetchone()[0]
        record(self.store, self.acc, current_hash, 0, 'exclude',
               ['unrepresented_visual_content', 'incorrect_text'],
               'Both pages are photographs; page 1 has no text and page 2 OCR misreads the labels.')
        result = research(self.store, self.acc, live=False)
        self.assertFalse(result['sources'])
        self.assertTrue(result['insufficient_evidence'])
        self.assertIn('unrepresented_visual_content', result['excluded_evidence'][0]['reasons'])
        self.assertEqual(self.store.db.execute('SELECT page FROM chunks').fetchone()[0], 2)
        self.assertIn('Corrupted aerial photograph labels', document_markdown(self.store, self.acc))
        self.assertEqual(self.store.db.execute('SELECT text_hash FROM documents').fetchone()[0], current_hash)

    def test_review_tracks_exact_text_version_and_old_exports_retain_it(self):
        self.review()
        old_version=self.store.db.execute('SELECT version FROM document_versions').fetchone()[0]
        self.put(['Valve revised first page','Valve corrected numeric value'])
        self.assertEqual({s['page'] for s in hybrid_retrieve(self.store,'Valve')},{1,2})
        self.assertNotIn('evidence_reviews',get(self.store.db,self.acc))
        self.assertIn('numeric_ocr_error',document_markdown(self.store,self.acc,old_version))
        self.assertNotIn('numeric_ocr_error',document_markdown(self.store,self.acc))
        with self.assertRaisesRegex(ValueError,'changed'):
            self.review(1)

    def test_metadata_refresh_and_missing_automated_diagnostic_keep_review(self):
        self.review()
        with self.store.db:
            self.store.put({'AccessionNumber':self.acc,'DocumentTitle':'Updated title'})
            self.store.db.execute('DELETE FROM document_quality')
        quality=get(self.store.db,self.acc)
        self.assertEqual(quality['state'],'needs_review')
        self.assertEqual(quality['evidence_reviews'][0]['page'],2)
        self.assertNotIn(2,{s['page'] for s in hybrid_retrieve(self.store,'Valve')})

    def test_reviews_are_idempotent_immutable_and_validate_page_identity(self):
        self.assertTrue(self.review())
        self.assertFalse(self.review())
        with self.assertRaisesRegex(ValueError,'immutable'):
            self.review(2,'warn')
        for page in [True,-1,99]:
            with self.assertRaises(ValueError):
                self.review(page)
        for reasons in [[],['unknown'],[{}]]:
            with self.assertRaises(ValueError):
                record(self.store,self.acc,self.old_hash,1,'exclude',reasons,'Observed issue')
        self.assertEqual(len(for_version(self.store.db,self.acc,self.old_hash)),1)

    def test_storage_stop_cannot_publish_review(self):
        with patch('adams_rag.extraction_review.require_space',side_effect=StoragePause('low')):
            with self.assertRaises(StoragePause):
                self.review()
        self.assertEqual(for_version(self.store.db,self.acc,self.old_hash),[])


if __name__=='__main__':
    unittest.main()
