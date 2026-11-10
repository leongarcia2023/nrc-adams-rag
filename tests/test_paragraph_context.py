import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

from adams_rag.store import Store
from adams_rag.evidence import SourceBindings, SourceBindingError, read_snapshot, saved_run_view, sha256
from adams_rag.extraction_review import record
from adams_rag.research import hybrid_retrieve, research


class ParagraphContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'db.sqlite'
        self.store = Store(self.path)
        self.acc = 'ML24017A120'
        self.anchor = 'Workers raised concerns. The most common issues involved'
        self.paragraph = self.anchor + ' resources and delays. These had not discouraged staff from raising concerns.'
        self.body = self.paragraph + '\n\nA different section starts here.'

    def tearDown(self):
        self.store.db.close(); self.tmp.cleanup()

    def prepare(self, body=None, anchor=None, pages=None):
        doc = {'AccessionNumber': self.acc, 'DocumentTitle': 'Inspection report',
               'content': body or self.body, 'DocumentDate': '2023-12-15', 'DocketNumber': ['05000373']}
        with self.store.db:
            self.store.put(doc, 'pdf' if pages is not None else 'api_document_unverified', pages)
            first = self.store.db.execute('SELECT id FROM chunks WHERE accession=? ORDER BY ordinal LIMIT 1', (self.acc,)).fetchone()[0]
            self.store.db.execute('UPDATE chunks SET text=? WHERE id=?', (anchor or self.anchor, first))
        return self.rows()[0]

    def rows(self):
        return [{**dict(r), 'retrieval_score': 1 / (60 + r['ordinal'])} for r in self.store.db.execute(
            'SELECT c.*,d.title,d.url,d.provenance,d.document_date FROM chunks c JOIN documents d USING(accession) ORDER BY ordinal')]

    def retrieve(self, rows, **kwargs):
        with patch('adams_rag.research.rank_candidates', return_value=rows):
            return hybrid_retrieve(self.store, 'work environment', **kwargs)

    def test_exact_retained_paragraph_is_added_without_next_section(self):
        row = self.prepare(); result = self.retrieve([row])
        self.assertEqual(len(result), 1)
        source = result[0]; binding = source['source_version']
        self.assertEqual(source['text'], self.paragraph)
        self.assertEqual(source['ordinal'], row['ordinal'])
        self.assertEqual(binding['passage_sha256'], sha256(self.paragraph))
        self.assertFalse(binding['source_fidelity_verified'])
        retained = self.store.db.execute('SELECT content FROM document_versions WHERE version=?', (binding['version'],)).fetchone()[0]
        body = zlib.decompress(retained).decode()
        self.assertEqual(body[binding['start_character']:binding['end_character']], self.paragraph)
        self.assertEqual(source['context_expansion']['anchor_passage_sha256'], sha256(self.anchor))
        self.assertEqual(self.store.db.execute('SELECT text FROM chunks WHERE id=?', (row['id'],)).fetchone()[0], self.anchor)

    def test_already_complete_sentence_heading_and_oversized_tail_are_unchanged(self):
        examples = [
            ('Complete sentence.', 'Complete sentence. Another sentence.\n\nNext.'),
            ('Section heading', 'Section heading\n\nNew unrelated paragraph.'),
            ('Partial', 'Partial ' + 'word ' * 201 + 'done.\n\nNext.'),
        ]
        for anchor, body in examples:
            with self.subTest(anchor=anchor):
                row = self.prepare(body, anchor)
                self.assertIsNone(SourceBindings(self.store.db).paragraph_completion(row))

    def test_ambiguous_repeated_anchor_is_not_assigned_a_continuation(self):
        row = self.prepare(body=self.body + '\n\n' + self.anchor + ' a different result.')
        self.assertIsNone(SourceBindings(self.store.db).paragraph_completion(row))

    def test_windows_line_endings_still_stop_at_the_paragraph_boundary(self):
        row = self.prepare(body=self.paragraph + '\r\n \r\nA different section starts here.')
        self.assertEqual(self.retrieve([row])[0]['text'], self.paragraph)

    def test_pdf_boundary_and_excluded_page_are_never_crossed(self):
        row = self.prepare(pages=[self.anchor, ' resources on a different page.'])
        text_hash = self.store.db.execute('SELECT text_hash FROM documents').fetchone()[0]
        record(self.store, self.acc, text_hash, 2, 'exclude', ['incorrect_text'], 'Page text is incorrect.')
        source = self.retrieve([row])[0]
        self.assertEqual(source['text'], self.anchor)
        self.assertNotIn('context_expansion', source)
        # In ordinary retrieval, the known-bad page remains excluded as well.
        self.assertNotIn(2, {s['page'] for s in hybrid_retrieve(self.store, 'resources')})

    def test_pdf_extension_within_same_page_preserves_page_and_review(self):
        row = self.prepare(pages=[self.body, 'Other page.'])
        text_hash = self.store.db.execute('SELECT text_hash FROM documents').fetchone()[0]
        record(self.store, self.acc, text_hash, 1, 'warn', ['omitted_text'], 'Some original content may be omitted.')
        source = self.retrieve([row])[0]
        self.assertEqual(source['page'], 1)
        self.assertEqual(source['text'], self.paragraph)
        self.assertEqual(source['source_version']['offset_scope'], 'physical_pdf_page')
        self.assertEqual(source['text_quality']['evidence_reviews'][0]['disposition'], 'warn')

    def test_existing_neighbor_coverage_avoids_duplicate_tail(self):
        row = self.prepare()
        with self.store.db:
            self.store.db.execute('INSERT INTO chunks(accession,ordinal,page,text) VALUES(?,?,?,?)',
                                  (self.acc, 2, None, self.paragraph))
        rows = self.rows(); sources = self.retrieve(rows)
        self.assertEqual([s['text'] for s in sources], [self.anchor, self.paragraph])
        self.assertFalse(any('context_expansion' in s for s in sources))

    def test_total_context_budget_never_displaces_original_anchor(self):
        anchor = ' '.join(['word'] * 4195)
        body = anchor + ' ' + ' '.join(['tail'] * 10) + '.\n\nNext.'
        row = self.prepare(body=body, anchor=anchor)
        sources = self.retrieve([row], top_k=1)
        self.assertEqual(sources[0]['text'], anchor)
        self.assertLessEqual(sum(len(s['text'].split()) for s in sources), 4200)

    def test_stale_rows_or_corrupt_retained_text_cannot_expand(self):
        row = self.prepare(); bindings = SourceBindings(self.store.db)
        changed = {**row, 'text': row['text'] + ' changed'}
        with self.assertRaises(SourceBindingError):
            bindings.paragraph_completion(changed)
        with self.store.db:
            self.store.db.execute('UPDATE document_versions SET content=?', (zlib.compress(b'Wrong text'),))
        with self.assertRaises(SourceBindingError):
            SourceBindings(self.store.db).paragraph_completion(row)

    def test_concurrent_source_replacement_keeps_one_snapshot(self):
        row = self.prepare(); writer = Store(self.path)
        try:
            with read_snapshot(self.store.db):
                bindings = SourceBindings(self.store.db); _, before = bindings.bind(row)
                with writer.db:
                    writer.put({'AccessionNumber': self.acc, 'DocumentTitle': 'Replacement', 'content': 'Different result.'},
                               'api_document_unverified')
                text, after, _ = bindings.paragraph_completion(row)
                self.assertEqual(text, self.paragraph)
                self.assertEqual(after['version'], before['version'])
            self.assertEqual(self.store.db.execute('SELECT title FROM documents').fetchone()[0], 'Replacement')
        finally:
            writer.db.close()

    def test_saved_expanded_span_remains_bound_after_source_replacement(self):
        row = self.prepare()
        with patch('adams_rag.research.rank_candidates', return_value=[row]):
            result = research(self.store, 'work environment', live=False, generate=False)
        with self.store.db:
            self.store.put({'AccessionNumber': self.acc, 'content': 'New version.'}, 'api_document_unverified')
        view = saved_run_view(self.store, result['id'])
        self.assertEqual(view['sources'], result['sources'])
        update = view['review_updates']['sources'][0]
        self.assertNotEqual(update['state'], 'retained_binding_unavailable_or_mismatched')
        self.assertFalse(update['current_source_matches'])


if __name__ == '__main__':
    unittest.main()
