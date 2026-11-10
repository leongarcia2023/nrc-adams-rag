from pathlib import Path
import tempfile
import unittest

from adams_rag.identifiers import extract_identifiers
from adams_rag.store import Store


class LexicalRankingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "research.sqlite")
        with self.store.db:
            for i in range(60):
                self.store.put({
                    "AccessionNumber": f"ML24017A{i:03}",
                    "DocumentTitle": "Valve test" if i % 3 else "Inspection report",
                    "DocumentDate": "2024-03-01" if i >= 50 else "2020-01-01",
                    "DocketNumber": ["05000373"] if i >= 55 else ["05000374"],
                    "content": ("relief valve test pressure " * (i % 7 + 1)) +
                               ("instrument observation " * (i % 5)) +
                               ("1E12-F025C " if i % 2 else "2B21-F016 ") +
                               ("1E12-F025C 2B21-F016 " if i >= 58 else ""),
                })

    def tearDown(self):
        self.store.db.close()
        self.temp.cleanup()

    def reference(self, question, limit, **filters):
        # The previous exhaustive ranking is an oracle for the optimized
        # query's scores, selected rows, metadata and filter placement.
        where, params = self.store._retrieval_filters(**filters)
        return [dict(r) for r in self.store.db.execute("""
            SELECT c.*,d.title,d.url,d.provenance,d.document_date,
                   bm25(chunks_fts,1.0,2.0,4.0) AS score
            FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
            JOIN documents d ON d.accession=c.accession
            WHERE chunks_fts MATCH ?""" + where + " ORDER BY score LIMIT ?",
            [self.store._fts_expression(question), *params, limit])]

    def test_weighted_ranking_and_result_fields_match_exhaustive_query(self):
        for query in ["relief valve test pressure", "inspection instrument", "ML24017A055"]:
            for limit in [1, 7, 200]:
                for filters in [{}, {"docket": "05000373"},
                                {"date_from": "2024-01-01", "date_to": "2024-12-31"},
                                {"docket": "05000373", "date_from": "2024-01-01"}]:
                    with self.subTest(query=query, limit=limit, filters=filters):
                        self.assertEqual(self.store.lexical(query, limit, **filters),
                                         self.reference(query, limit, **filters))

    def test_filters_are_applied_before_a_small_candidate_limit(self):
        rows = self.store.lexical("relief valve test pressure", 3,
                                  docket="05000373", date_from="2024-01-01")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(int(row["accession"][-3:]) >= 55 for row in rows))

    def test_empty_and_no_match_queries_remain_empty(self):
        for query in ["", "what is the", "unfindabletoken"]:
            self.assertEqual(self.store.lexical(query), [])
        self.assertEqual(self.store.lexical("valve", docket="05000999"), [])

    def test_identifier_lookup_preserves_hit_priority_relevance_and_filters(self):
        question = "How were relief valve 1E12-F025C and valve 2B21-F016 tested?"
        identifiers = sorted(extract_identifiers(question))
        expression = " OR ".join([self.store._fts_expression(question),
                                   self.store._fts_expression(" ".join(identifiers))])
        for filters in [{}, {"docket": "05000373", "date_from": "2024-01-01"}]:
            where, params = self.store._retrieval_filters(**filters)
            for limit in [1, 7, 200]:
                with self.subTest(filters=filters, limit=limit):
                    expected = [dict(r) for r in self.store.db.execute("""
                        WITH exact AS (SELECT chunk_id,count(*) AS identifier_hits
                          FROM chunk_identifiers WHERE identifier IN (?,?) GROUP BY chunk_id)
                        SELECT c.*,d.title,d.url,d.provenance,d.document_date,d.metadata,exact.identifier_hits
                        FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
                        JOIN exact ON exact.chunk_id=c.id
                        JOIN documents d ON d.accession=c.accession
                        WHERE chunks_fts MATCH ?""" + where + """
                        ORDER BY identifier_hits DESC,bm25(chunks_fts,1.0,2.0,4.0) LIMIT ?""",
                        (*identifiers, expression, *params, limit))]
                    actual = self.store.identifier_matches(question, limit, **filters)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual[0]["identifier_hits"], 2)


if __name__ == "__main__":
    unittest.main()
