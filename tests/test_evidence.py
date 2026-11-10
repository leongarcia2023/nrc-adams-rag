import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import parse_qs,urlsplit
import zlib

from adams_rag.evidence import read_snapshot,saved_run_view,sha256
from adams_rag.extraction_review import record
from adams_rag.research import hybrid_retrieve,research
from adams_rag.retrieval import retrieve
from adams_rag.store import Store


class EvidenceBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.store=Store(self.root/'db.sqlite')
        self.acc='ML24017A120'
        self.put(['First valve page','Second valve page with value 12'])

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def put(self,pages,**metadata):
        with self.store.db:
            self.store.put({'AccessionNumber':self.acc,'DocumentTitle':'Valve report',
                            'LocalFileSHA256':'a'*64,**metadata},'pdf',pages)

    def test_passages_resolve_to_the_exact_retained_page_and_offsets(self):
        for retriever in (hybrid_retrieve,retrieve):
            sources=retriever(self.store,'valve')
            self.assertEqual(len(sources),2)
            for source in sources:
                binding=source['source_version']
                query=parse_qs(urlsplit(binding['url']).query)
                self.assertEqual(query,{'accession':[self.acc],'version':[binding['version']]})
                row=self.store.db.execute('SELECT * FROM document_versions WHERE version=?',(binding['version'],)).fetchone()
                body=zlib.decompress(row['content']).decode()
                page=body.split('\f')[source['page']-1]
                self.assertEqual(page[binding['start_character']:binding['end_character']],source['text'])
                self.assertEqual(binding['text_sha256'],sha256(body))
                self.assertEqual(binding['passage_sha256'],sha256(source['text']))
                self.assertEqual(binding['source_file_sha256'],'a'*64)
                self.assertFalse(binding['source_fidelity_verified'])

    def test_metadata_refresh_keeps_text_version_separate_from_current_metadata(self):
        before=hybrid_retrieve(self.store,'valve')[0]['source_version']
        with self.store.db:
            self.store.put({'AccessionNumber':self.acc,'DocumentTitle':'New title'})
        source=hybrid_retrieve(self.store,'valve')[0];after=source['source_version']
        self.assertEqual(source['title'],'New title')
        self.assertEqual(before['version'],after['version'])
        self.assertEqual(before['retained_metadata_sha256'],after['retained_metadata_sha256'])
        self.assertNotEqual(after['retrieval_metadata_sha256'],after['retained_metadata_sha256'])

    def test_same_text_from_new_pdf_binds_to_new_file_version(self):
        before=hybrid_retrieve(self.store,'valve')[0]['source_version']
        self.put(['First valve page','Second valve page with value 12'],LocalFileSHA256='b'*64)
        after=hybrid_retrieve(self.store,'valve')[0]['source_version']
        self.assertEqual(before['text_sha256'],after['text_sha256'])
        self.assertNotEqual(before['version'],after['version'])
        self.assertEqual(after['source_file_sha256'],'b'*64)

    def test_concurrent_replacement_cannot_mix_old_passage_with_new_metadata(self):
        old=hybrid_retrieve(self.store,'valve')[0]['source_version']['version']
        writer=Store(self.root/'db.sqlite')
        class ReplacingSemantic:
            def search(inner,*args):
                with writer.db:
                    writer.put({'AccessionNumber':self.acc,'DocumentTitle':'Replacement','LocalFileSHA256':'b'*64},
                               'pdf',['New valve replacement with value 24'])
                return []
        try:
            sources=hybrid_retrieve(self.store,'valve',semantic=ReplacingSemantic())
            self.assertEqual({s['title'] for s in sources},{'Valve report'})
            self.assertEqual({s['source_version']['version'] for s in sources},{old})
            self.assertFalse(self.store.db.in_transaction)
            self.assertEqual(self.store.db.execute('SELECT title FROM documents').fetchone()[0],'Replacement')
        finally:
            writer.db.close()

    def test_read_snapshot_never_rolls_back_its_callers_transaction(self):
        self.store.db.execute('BEGIN IMMEDIATE')
        self.store.db.execute("UPDATE documents SET title='Caller update'")
        with read_snapshot(self.store.db):
            self.assertTrue(self.store.db.in_transaction)
        self.assertTrue(self.store.db.in_transaction)
        self.store.db.rollback()
        self.assertEqual(self.store.db.execute('SELECT title FROM documents').fetchone()[0],'Valve report')

    def test_failed_retrieval_releases_owned_snapshot(self):
        class BrokenSemantic:
            def search(self,*args):raise RuntimeError('failed')
        with self.assertRaises(RuntimeError):
            hybrid_retrieve(self.store,'valve',semantic=BrokenSemantic())
        self.assertFalse(self.store.db.in_transaction)

    def test_missing_retained_version_cannot_reach_generation(self):
        with self.store.db:self.store.db.execute('DELETE FROM document_versions')
        class Engine:
            def generate(self,*args):raise AssertionError('Unbound text reached generation')
        result=research(self.store,'valve',engine=Engine(),live=False)
        self.assertEqual(result['sources'],[])
        self.assertTrue(result['insufficient_evidence'])
        self.assertEqual({r['reason'] for r in result['provenance_issues']},{'retained_version_missing'})

    def test_corrupt_retained_content_and_wrong_page_are_excluded(self):
        with self.store.db:
            self.store.db.execute('UPDATE document_versions SET content=?',(zlib.compress(b'Changed bytes'),))
        result=research(self.store,'valve',live=False,generate=False)
        self.assertEqual(result['sources'],[])
        self.assertEqual({r['reason'] for r in result['provenance_issues']},{'retained_text_checksum_mismatch'})
        self.put(['Restored valve page','Other valve page'])
        with self.store.db:self.store.db.execute('UPDATE chunks SET page=99')
        issues=[]
        self.assertEqual(hybrid_retrieve(self.store,'valve',source_issues=issues),[])
        self.assertEqual({r['reason'] for r in issues},{'retained_page_missing'})

    def test_saved_answer_gets_new_review_warning_without_rewriting_history(self):
        result=research(self.store,'valve',live=False,generate=False)
        original=self.store.db.execute('SELECT result FROM research_runs WHERE id=?',(result['id'],)).fetchone()[0]
        source=result['sources'][0]
        record(self.store,self.acc,source['source_version']['text_sha256'],source['page'],
               'exclude',['numeric_ocr_error'],'A later visual check found an incorrect value.')
        view=saved_run_view(self.store,result['id'])
        update=next(r for r in view['review_updates']['sources'] if r['citation']==source['citation'])
        self.assertEqual(update['state'],'excluded_by_review')
        self.assertTrue(update['review_changed_since_run'])
        self.assertTrue(any('now excluded' in w for w in view['warnings']))
        self.assertEqual(view['sources'],result['sources'])
        self.assertEqual(self.store.db.execute('SELECT result FROM research_runs WHERE id=?',(result['id'],)).fetchone()[0],original)

    def test_saved_run_retains_old_version_after_replacement(self):
        result=research(self.store,'valve',live=False,generate=False)
        self.put(['Different valve source'],LocalFileSHA256='b'*64)
        view=saved_run_view(self.store,result['id'])
        self.assertEqual(view['sources'],result['sources'])
        self.assertTrue(all(not r['current_source_matches'] for r in view['review_updates']['sources']))
        self.assertTrue(any('has changed since this run' in w for w in view['warnings']))

    def test_old_unversioned_runs_are_not_guessed_from_current_documents(self):
        result=research(self.store,'valve',live=False,generate=False)
        for source in result['sources']:source.pop('source_version')
        with self.store.db:self.store.db.execute('UPDATE research_runs SET result=? WHERE id=?',(json.dumps(result),result['id']))
        view=saved_run_view(self.store,result['id'])
        self.assertEqual(view['sources'],result['sources'])
        self.assertEqual({r['state'] for r in view['review_updates']['sources']},{'unversioned_saved_evidence'})
        self.assertEqual(len(view['review_updates']['warnings']),1)
        self.assertIsNone(saved_run_view(self.store,'unknown'))

    def test_tampered_saved_passage_is_reported_as_binding_mismatch(self):
        result=research(self.store,'valve',live=False,generate=False)
        result['sources'][0]['text']='Not the saved passage'
        with self.store.db:self.store.db.execute('UPDATE research_runs SET result=? WHERE id=?',(json.dumps(result),result['id']))
        view=saved_run_view(self.store,result['id'])
        self.assertEqual(view['review_updates']['sources'][0]['state'],'retained_binding_unavailable_or_mismatched')

    def test_tampered_file_binding_and_retained_metadata_are_detected(self):
        result=research(self.store,'valve',live=False,generate=False)
        result['sources'][0]['source_version']['source_file_sha256']='b'*64
        with self.store.db:self.store.db.execute('UPDATE research_runs SET result=? WHERE id=?',(json.dumps(result),result['id']))
        view=saved_run_view(self.store,result['id'])
        self.assertEqual(view['review_updates']['sources'][0]['state'],'retained_binding_unavailable_or_mismatched')
        with self.store.db:
            self.store.db.execute("UPDATE document_versions SET metadata=json_set(metadata,'$.DocumentTitle','Unexpected alteration')")
        result=research(self.store,'valve',live=False,generate=False)
        self.assertEqual(result['sources'],[])
        self.assertEqual({r['reason'] for r in result['provenance_issues']},{'retained_version_checksum_mismatch'})


if __name__=='__main__':unittest.main()
