from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from adams_rag.extraction_review import record
from adams_rag.profile_regression import compare_retrieval, QueryCache, validate_cases
from adams_rag.semantic_index import SemanticIndex
from adams_rag.store import Store


class Encoder:
    def passage_embed(self, texts, **kwargs):
        return [np.asarray([1., 0.] if "valve" in text else [0., 1.], dtype=np.float32) for text in texts]

    def query_embed(self, question):
        return iter([np.asarray([1., 0.], dtype=np.float32)])


class ProfileRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "research.sqlite")
        with self.store.db:
            self.store.put({"AccessionNumber": "A", "content": "valve evidence", "DocketNumber": ["05000373"], "DocumentDate": "2024-01-17"})
            self.store.put({"AccessionNumber": "B", "content": "pump evidence", "DocketNumber": ["05000001"], "DocumentDate": "1999-01-01"})
        self.baseline = SemanticIndex(self.store, self.root / "index", "baseline", Encoder())
        self.candidate = SemanticIndex(self.store, self.root / "index", "candidate", Encoder())
        self.baseline.sync()
        self.cases = [{"id": "equipment", "question": "equipment", "relevant": ["A"]}]

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def compare(self, cases=None):
        return compare_retrieval(self.store, self.baseline, self.candidate, cases or self.cases, k=1)

    def test_missing_candidate_coverage_never_runs_or_passes_retrieval(self):
        with patch("adams_rag.profile_regression.rank_candidates", side_effect=AssertionError("Not ready")):
            result = self.compare()
        self.assertEqual(result["state"], "coverage_pending")
        self.assertFalse(result["known_source_regressions_passed"])
        self.assertEqual(result["cases"], [])

    def test_paired_ranks_record_a_regression_even_when_vector_coverage_matches(self):
        self.candidate.sync()
        self.candidate.encoder = QueryCache(["equipment"], [np.asarray([0., 1.])])
        result = self.compare()
        self.assertEqual(result["state"], "evaluated")
        self.assertEqual(result["cases"][0]["baseline"]["relevant_ranks"], {"A": 1})
        self.assertEqual(result["cases"][0]["candidate"]["relevant_ranks"], {"A": None})
        self.assertFalse(result["known_source_regressions_passed"])

    def test_matching_results_retain_filters_and_do_not_certify_general_accuracy(self):
        self.candidate.sync()
        result = self.compare([{**self.cases[0], "docket": "05000373", "date_from": "2024-01-01", "date_to": "2024-12-31"}])
        self.assertTrue(result["known_source_regressions_passed"])
        self.assertFalse(result["independent_quality_benchmark"])
        self.assertFalse(result["generation_evaluated"])
        self.assertEqual(result["collection_passages"], 2)
        self.assertFalse(self.store.db.in_transaction)

    def test_review_exclusions_apply_to_both_profiles(self):
        self.candidate.sync()
        text_hash = self.store.db.execute("SELECT text_hash FROM documents WHERE accession='A'").fetchone()[0]
        record(self.store, "A", text_hash, 0, "exclude", ["incorrect_text"], "Observed incorrect extraction")
        result = self.compare([{**self.cases[0], "relevant": ["B"]}])
        self.assertTrue(result["known_source_regressions_passed"])
        for label in ("baseline", "candidate"):
            self.assertEqual(result["cases"][0][label]["excluded_evidence"][0]["accession"], "A")

    def test_concurrent_source_replacement_cannot_change_one_side_of_the_comparison(self):
        from adams_rag.research import rank_candidates
        self.candidate.sync()
        writer = Store(self.store.path)
        changed = [False]
        def rank_then_change(*args, **kwargs):
            result = rank_candidates(*args, **kwargs)
            if not changed[0]:
                changed[0] = True
                with writer.db:
                    writer.put({"AccessionNumber": "A", "content": "replacement pump evidence"})
            return result
        try:
            with patch("adams_rag.profile_regression.rank_candidates", side_effect=rank_then_change):
                result = self.compare()
            self.assertTrue(result["known_source_regressions_passed"])
            self.assertNotEqual(result["snapshot_fingerprints"]["baseline"], self.baseline.fingerprint())
        finally:
            writer.db.close()

    def test_unlabelled_duplicate_or_malformed_queries_are_rejected(self):
        for cases in ([], self.cases * 2, [{**self.cases[0], "relevant": []}], [{**self.cases[0], "date_from": "bad"}]):
            with self.assertRaises(ValueError):
                validate_cases(cases)
        for values in ([], [np.asarray([np.nan, 0.])], [np.zeros((1, 2))]):
            with self.assertRaises(ValueError):
                QueryCache(["question"], values)


if __name__ == "__main__":
    unittest.main()
