from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from adams_rag.evidence import read_snapshot
from adams_rag.extraction_review import record
from adams_rag.research import hybrid_retrieve, rank_candidates
from adams_rag.semantic_index import SemanticIndex
from adams_rag.store import Store


class Encoder:
    def passage_embed(self, texts, **kwargs):
        return [np.asarray([1., 0.], dtype=np.float32) for _ in texts]


class CoverageFusionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / 'db.sqlite')
        self.fresh = 'ML20024A884'
        self.older = [f'OLDER{i:02}' for i in range(20)]
        with self.store.db:
            for key in [self.fresh, *self.older]:
                self.store.put({'AccessionNumber': key, 'DocumentTitle': 'Equipment record ' + key,
                                'content': 'Retained equipment observation for source ' + key})
        self.index = SemanticIndex(self.store, self.root / 'index', 'active-test-profile', Encoder())
        self.index.sync(accessions=self.older)
        rows = self.store.db.execute('''SELECT c.*,d.title,d.url,d.provenance,d.document_date,d.metadata
            FROM chunks c JOIN documents d ON d.accession=c.accession''').fetchall()
        self.rows = {r['accession']: dict(r) for r in rows}
        self.keyword = [self.rows[a] for a in [self.fresh, *self.older]]
        self.dense = [self.rows[a] for a in self.older]

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def retrieve(self, question='equipment observation'):
        with patch.object(self.store, 'lexical', return_value=self.keyword), \
                patch.object(self.index, 'search', return_value=self.dense):
            return hybrid_retrieve(self.store, question, self.index, top_k=3)

    def test_unembedded_top_keyword_hit_survives_partial_coverage(self):
        class UnknownCoverage:
            def search(inner, *args):
                return self.dense
        with patch.object(self.store, 'lexical', return_value=self.keyword):
            unadjusted = hybrid_retrieve(self.store, 'equipment observation', UnknownCoverage(), top_k=3)
        self.assertNotIn(self.fresh, [s['accession'] for s in unadjusted])
        adjusted = self.retrieve()
        self.assertEqual(adjusted[0]['accession'], self.fresh)
        self.assertEqual(adjusted[0]['source_version']['state'], 'matched_retained_text')

    def test_relevant_hit_remains_found_after_embedding_is_available(self):
        self.assertEqual(self.retrieve()[0]['accession'], self.fresh)
        self.index.sync(accessions=[self.fresh])
        self.dense.insert(0, self.rows[self.fresh])
        self.assertEqual(self.retrieve()[0]['accession'], self.fresh)

    def test_indexed_passage_absent_from_semantic_results_gets_no_missing_coverage_boost(self):
        self.index.sync(accessions=[self.fresh])
        self.assertEqual(self.retrieve()[0]['accession'], self.older[0])
        self.assertNotIn(self.fresh, [s['accession'] for s in self.retrieve()])

    def test_coverage_is_profile_specific_and_handles_batched_ids(self):
        alternate = SemanticIndex(self.store, self.root / 'index', 'other-profile', Encoder())
        alternate.sync(accessions=[self.fresh])
        first, last = self.rows[self.older[0]]['id'], self.rows[self.older[-1]]['id']
        fresh = self.rows[self.fresh]['id']
        ids = [first, *range(10000, 10500), last, fresh, first]
        self.assertEqual(self.index.covered_chunk_ids(ids), {first, last})
        self.assertEqual(alternate.covered_chunk_ids(ids), {fresh})
        self.assertEqual(self.index.covered_chunk_ids([]), set())
        for bad in ([1, True], [0], [-1], [1.0], ['1']):
            with self.assertRaises(ValueError):
                self.index.covered_chunk_ids(bad)

    def test_coverage_uses_same_read_snapshot_as_candidates(self):
        target = self.rows[self.older[0]]['id']
        writer = Store(self.store.path)
        try:
            with read_snapshot(self.store.db):
                self.assertEqual(self.index.covered_chunk_ids([target]), {target})
                with writer.db:
                    writer.db.execute('DELETE FROM dense_vectors WHERE model=? AND chunk_id=?', (self.index.model, target))
                self.assertEqual(self.index.covered_chunk_ids([target]), {target})
            self.assertEqual(self.index.covered_chunk_ids([target]), set())
        finally:
            writer.db.close()

    def test_semantic_only_and_keyword_only_results_are_preserved(self):
        with patch.object(self.store, 'lexical', return_value=[]), patch.object(self.index, 'search', return_value=self.dense):
            rows = hybrid_retrieve(self.store, 'paraphrased concept', self.index, top_k=3)
        self.assertEqual([r['accession'] for r in rows], self.older[:3])
        with patch.object(self.store, 'lexical', return_value=self.keyword):
            rows = rank_candidates(self.store, 'equipment observation', semantic=None)
        self.assertEqual([r['accession'] for r in rows], [self.fresh, *self.older])

    def test_review_exclusion_still_removes_boosted_unembedded_source(self):
        text_hash = self.store.db.execute('SELECT text_hash FROM documents WHERE accession=?', (self.fresh,)).fetchone()[0]
        record(self.store, self.fresh, text_hash, 0, 'exclude', ['incorrect_text'], 'Test reviewed extraction error')
        self.assertNotIn(self.fresh, [s['accession'] for s in self.retrieve()])

    def test_exact_identifier_evidence_keeps_priority(self):
        with self.store.db:
            self.store.put({'AccessionNumber': 'EXACT', 'content': 'Valve 1E12-F025C failed the specified test.'})
        rows = self.retrieve('What happened to valve 1E12-F025C?')
        self.assertEqual(rows[0]['accession'], 'EXACT')

    def test_adjusted_tie_retains_prior_fused_preference(self):
        with self.store.db:
            for n in range(61):
                self.store.put({'AccessionNumber': f'PENDING{n:02}', 'content': f'New observation {n}'})
        keyword = [dict(r) for r in self.store.db.execute('''SELECT c.*,d.title,d.url,d.provenance,d.document_date
            FROM chunks c JOIN documents d ON d.accession=c.accession WHERE c.accession LIKE 'PENDING%'
            ORDER BY c.accession''')]
        keyword.append(self.rows[self.fresh])
        with patch.object(self.store, 'lexical', return_value=keyword), \
                patch.object(self.index, 'search', return_value=[self.rows[self.older[0]]]):
            rows = rank_candidates(self.store, 'equipment observation', self.index)
        positions = {r['accession']: i for i, r in enumerate(rows)}
        self.assertLess(positions[self.older[0]], positions[self.fresh])
        self.assertEqual(rows[positions[self.older[0]]]['retrieval_score'], rows[positions[self.fresh]]['retrieval_score'])


if __name__ == '__main__':
    unittest.main()
