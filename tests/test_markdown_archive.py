import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from adams_rag.evidence import SourceBindings, saved_run_view
from adams_rag.extraction_review import filter_rows, record
from adams_rag.export import document_markdown
from adams_rag.markdown_archive import (decode_markdown, export_batch, inspect_document,
    render_markdown, run, setup, saved_markdown, prune_reviewed_pdf)
from adams_rag.research import hybrid_retrieve, research
from adams_rag.retrieval import retrieve
from adams_rag.storage import StoragePause
from adams_rag.store import Store, digest


class MarkdownArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / 'test.sqlite')
        self.out = self.root / 'markdown'; self.out.mkdir()
        setup(self.store.db)

    def tearDown(self):
        self.store.db.close(); self.tmp.cleanup()

    def put(self, acc='ML000000001', text='Valve inspection text.', **kwargs):
        with self.store.db:
            self.store.put(dict(AccessionNumber=acc, DocumentTitle='Valve inspection', content=text, **kwargs))
        return self.store.db.execute('SELECT rowid,* FROM documents WHERE accession=?', (acc,)).fetchone()

    def test_lossless_plain_text_unicode_fences_and_source_provenance(self):
        row = self.put(text='  Valve 123 μSv\n~~~text\n# Literal heading\n<script>foo</script>\n\f end  ')
        before = [tuple(r) for r in self.store.db.execute('SELECT * FROM chunks')]
        self.assertEqual(export_batch(self.store, [row], self.out, {}), 1)
        md = saved_markdown(self.store, row['accession'])
        header, body = decode_markdown(md)
        self.assertEqual(digest(body), row['text_hash'])
        self.assertEqual(header['accession'], row['accession'])
        self.assertIn('https://', header['source_url'])
        self.assertFalse(header['review']['source_fidelity_verified'])
        self.assertEqual(header['review']['state'], 'ready')
        self.assertEqual(document_markdown(self.store, row['accession']), md)
        self.assertEqual(document_markdown(self.store, row['accession'], header['retained_version']), md)
        self.assertEqual(before, [tuple(r) for r in self.store.db.execute('SELECT * FROM chunks')])
        SourceBindings(self.store.db).document(row['accession'])

    def test_damaged_text_and_reviews_are_held_out_across_retrieval_routes(self):
        good = self.put()
        bad = self.put('ML000000002', text='Valve '\
            + '\ufffd' * 6 + ' inspection is unreadable')
        export_batch(self.store, [good, bad], self.out, {})
        class Semantic:
            def search(other, *args):
                return [dict(r) for r in self.store.db.execute('SELECT c.*,d.title,d.url,d.provenance,d.document_date FROM chunks c JOIN documents d USING(accession)')]
        exclusions = []
        self.assertEqual({r['accession'] for r in hybrid_retrieve(self.store, 'Valve', semantic=Semantic(), exclusions=exclusions)}, {good['accession']})
        self.assertEqual({r['accession'] for r in retrieve(self.store, 'Valve')}, {good['accession']})
        self.assertIn('replacement_characters', exclusions[0]['reasons'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM chunks').fetchone()[0], 2)
        self.assertIn('holdout/', self.store.db.execute('SELECT member FROM markdown_checks WHERE accession=?', (bad['accession'],)).fetchone()[0])

    def test_pdf_pages_are_reversible_and_ocr_is_not_certified(self):
        pdf = self.root / 'source.pdf'; pdf.write_bytes(b'original fixture')
        pages = [' Valve\fembedded control ', 'second page\n']
        with self.store.db:
            self.store.put({'AccessionNumber':'ML000000003', 'LocalFileSHA256':digest('original fixture'),
                'ExtractedPageCount':2, 'PageExtraction':[{'page':1,'method':'apple_vision_ocr'}, {'page':2,'method':'embedded_text'}]}, 'pdf', pages)
        row = self.store.db.execute('SELECT rowid,* FROM documents').fetchone()
        text, extracted, retained, report = inspect_document(self.store, row, {row['accession']:pdf})
        self.assertEqual(report['state'], 'holdout')
        self.assertIn('machine_ocr', report['reasons'])
        self.assertFalse(report['source_fidelity_verified'])
        md = render_markdown(row, text, extracted, retained, report)
        self.assertEqual(decode_markdown(md)[1], '\f'.join(pages))
        self.assertTrue(pdf.exists())

    def test_pruning_rejects_automatic_ocr_and_fabricated_review_metadata(self):
        acc = 'ML000000003'; recovery = self.root / 'recovery'; recovery.mkdir()
        pdf = recovery / (acc + '.pdf'); pdf.write_text('PDF fixture')
        with self.store.db:
            self.store.put({'AccessionNumber':acc,'Url':'https://www.nrc.gov/docs/ML0000/'+acc+'.pdf',
                'LocalFileSHA256':digest('PDF fixture'),'ExtractedPageCount':1,
                'PageExtraction':[{'page':1,'method':'reviewed_transcription','correction_id':'fabricated',
                                   'source_pdf_sha256':digest('PDF fixture')}]}, 'pdf', ['Valve exact content.'])
        row = self.store.db.execute('SELECT rowid,* FROM documents').fetchone()
        export_batch(self.store, [row], self.out, {acc:pdf})
        with self.assertRaisesRegex(ValueError, 'Full-source review'):
            prune_reviewed_pdf(self.store, acc)
        self.assertTrue(pdf.exists())

    def test_pruning_requires_full_page_review_and_retains_source_link_and_text(self):
        acc = 'ML000000003'; recovery = self.root / 'recovery'; recovery.mkdir()
        pdf = recovery / (acc + '.pdf'); pdf.write_text('PDF fixture')
        text = 'Valve exact content.'; checksum = digest('PDF fixture')
        manifest = json.dumps({'pages':[{'page':1,'text':text}]}, sort_keys=True)
        key = digest(manifest)
        with self.store.db:
            self.store.db.execute('CREATE TABLE extraction_corrections(id TEXT,accession TEXT,pdf_sha256 TEXT,manifest TEXT)')
            self.store.db.execute('INSERT INTO extraction_corrections VALUES(?,?,?,?)', (key,acc,checksum,manifest))
            self.store.put({'AccessionNumber':acc,'Url':'https://www.nrc.gov/docs/ML0000/'+acc+'.pdf',
                'LocalFileSHA256':checksum,'ExtractedPageCount':1,
                'PageExtraction':[{'page':1,'method':'reviewed_transcription','correction_id':key,
                                   'source_pdf_sha256':checksum}]}, 'pdf', [text])
        row = self.store.db.execute('SELECT rowid,* FROM documents').fetchone()
        export_batch(self.store, [row], self.out, {acc:pdf})
        self.assertTrue(prune_reviewed_pdf(self.store, acc)['markdown_retained'])
        self.assertFalse(pdf.exists())
        header, body = decode_markdown(saved_markdown(self.store, acc))
        self.assertEqual(body, text); self.assertTrue(header['source_url'].endswith(acc+'.pdf'))
        self.assertEqual(self.store.db.execute('SELECT state FROM markdown_pruned_pdf').fetchone()[0], 'removed_after_verified_export')
        with self.store.db:
            self.store.put({'AccessionNumber':acc,'DocumentTitle':'Refreshed catalog title'}, 'api_document_unverified')
        row = self.store.db.execute('SELECT rowid,* FROM documents').fetchone()
        export_batch(self.store, [row], self.out, {})
        header, body = decode_markdown(saved_markdown(self.store, acc))
        self.assertEqual(body, text)
        self.assertEqual(header['review']['state'], 'ready')
        self.assertTrue(header['review']['source_pdf_previously_verified'])
        self.assertTrue(header['review']['source_fidelity_verified'])

    def test_missing_text_metadata_is_exported_as_holdout_without_fake_text(self):
        row = self.put(text='')
        export_batch(self.store, [row], self.out, {})
        header, body = decode_markdown(saved_markdown(self.store, row['accession']))
        self.assertEqual(body, '')
        self.assertEqual(header['review']['reasons'], ['no_full_text_available'])

    def test_saved_answers_gain_current_holdout_warning_without_rewriting_history(self):
        row = self.put(text='Valve inspection '\
            + '\ufffd' * 6 + ' incomplete source')
        result = research(self.store, 'Valve', live=False, generate=False)
        self.assertTrue(result['sources'])
        original = self.store.db.execute('SELECT result FROM research_runs WHERE id=?', (result['id'],)).fetchone()[0]
        export_batch(self.store, [row], self.out, {})
        view = saved_run_view(self.store, result['id'])
        self.assertEqual(view['review_updates']['sources'][0]['state'], 'held_out_by_document_check')
        self.assertTrue(any('held out' in s for s in view['warnings']))
        self.assertEqual(original, self.store.db.execute('SELECT result FROM research_runs WHERE id=?', (result['id'],)).fetchone()[0])

    def test_changed_source_or_review_invalidates_eligibility_and_resume_rechecks(self):
        row = self.put()
        run(self.store, self.out)
        record(self.store, row['accession'], row['text_hash'], 0, 'exclude', ['omitted_text'], 'Missing content')
        self.assertIsNone(saved_markdown(self.store, row['accession']))
        self.assertFalse(hybrid_retrieve(self.store, 'Valve'))
        run(self.store, self.out)
        self.assertEqual(self.store.db.execute('SELECT state FROM markdown_checks').fetchone()[0], 'holdout')
        self.put(text='Valve replacement content.')
        self.assertFalse(hybrid_retrieve(self.store, 'Valve'))
        run(self.store, self.out)
        self.assertTrue(hybrid_retrieve(self.store, 'Valve'))
        count = len(list(self.out.glob('*.zip')))
        run(self.store, self.out)
        self.assertEqual(len(list(self.out.glob('*.zip'))), count)

    def test_corrupt_chunk_is_held_out_and_archive_tampering_is_rejected(self):
        row = self.put()
        with self.store.db:
            self.store.db.execute("UPDATE chunks SET text='invented' WHERE accession=?", (row['accession'],))
        export_batch(self.store, [row], self.out, {})
        check = self.store.db.execute('SELECT * FROM markdown_checks').fetchone()
        self.assertIn('chunk_coverage_mismatch', json.loads(check['report'])['reasons'])
        with zipfile.ZipFile(check['archive'], 'w') as z:
            z.writestr(check['member'], 'tampered text')
        with self.assertRaises(ValueError):
            saved_markdown(self.store, row['accession'])

    def test_storage_pause_publishes_no_partial_check_or_file(self):
        row = self.put()
        with patch('adams_rag.markdown_archive.require_space', side_effect=[10**9, StoragePause('low')]):
            with self.assertRaises(StoragePause):
                export_batch(self.store, [row], self.out, {})
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM markdown_checks').fetchone()[0], 0)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_failure_in_final_verification_cannot_publish_manifest(self):
        row = self.put()
        with patch('adams_rag.markdown_archive.decode_markdown', side_effect=[({}, 'Valve inspection text.'), ValueError('corrupt')]):
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                export_batch(self.store, [row], self.out, {})
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM markdown_checks').fetchone()[0], 0)
        self.assertFalse(list(self.out.glob('*')))


if __name__ == '__main__':
    unittest.main()
