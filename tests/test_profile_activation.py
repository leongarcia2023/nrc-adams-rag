import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from usearch.index import Index

from adams_rag.extraction_review import record
from adams_rag.index_audit import audit_profiles
from adams_rag.mlx_embeddings import MODEL
from adams_rag.profile_activation import activate
from adams_rag.profile_regression import compare_retrieval, retrieval_fingerprint
from adams_rag.semantic_index import DEFAULT_EMBEDDING, SemanticIndex, active_model
from adams_rag.storage import StoragePause
from adams_rag.store import Store


class Encoder:
    def passage_embed(self, texts, **kwargs):
        result = np.zeros(384, dtype=np.float32); result[0] = 1
        return [result for _ in texts]

    def query_embed(self, question):
        return iter(self.passage_embed([question]))


class ProfileActivationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.directory = self.root / "index"
        self.store = Store(self.root / "research.sqlite")
        with self.store.db:
            self.store.put({"AccessionNumber": "A", "content": "valve evidence"})
        self.baseline = SemanticIndex(self.store, self.directory, DEFAULT_EMBEDDING, Encoder())
        self.candidate = SemanticIndex(self.store, self.directory, MODEL, Encoder())
        self.baseline.sync(); self.candidate.sync()
        self.cases = [{"id": "valve", "question": "valve", "relevant": ["A"]}]
        self.audit = audit_profiles(self.store.db, self.store.path, self.directory, DEFAULT_EMBEDDING, MODEL)
        self.regression = compare_retrieval(self.store, self.baseline, self.candidate, self.cases)

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def activate(self):
        return activate(self.store, self.directory, self.audit, self.regression, self.cases)

    def rejected(self, pattern=None):
        with self.assertRaisesRegex(ValueError, pattern or ".+"):
            self.activate()
        self.assertEqual(active_model(self.directory), DEFAULT_EMBEDDING)

    def test_current_matching_audits_atomically_select_profile_and_keep_old_vectors(self):
        result = self.activate()
        self.assertEqual((result["state"], active_model(self.directory)), ("activated", MODEL))
        self.assertEqual(self.baseline.status()["vectors"], 1)
        self.assertFalse(result["full_archive_verified"])
        self.assertFalse(list(self.directory.glob("active-profile-*.tmp")))

    def test_metadata_only_change_invalidates_retrieval_check(self):
        with self.store.db:
            self.store.db.execute("UPDATE documents SET document_date='2024-01-01' WHERE accession='A'")
        self.rejected("collection or its reviews changed")

    def test_new_keyword_only_document_invalidates_retrieval_check(self):
        with self.store.db:
            self.store.put({"AccessionNumber": "B", "content": "valve observations"})
        self.rejected("collection or its reviews changed")

    def test_new_review_invalidates_retrieval_check_without_changing_vectors(self):
        text_hash = self.store.db.execute("SELECT text_hash FROM documents WHERE accession='A'").fetchone()[0]
        record(self.store, "A", text_hash, 0, "warn", ["incorrect_text"], "Observed extraction error")
        self.rejected("collection or its reviews changed")

    def test_same_generation_graph_edit_cannot_reuse_old_checks(self):
        graph = Index.restore(str(self.candidate.path), view=False)
        key = self.store.db.execute("SELECT id FROM chunks").fetchone()[0]
        graph.remove(key)
        vector = np.zeros(384, dtype=np.float32); vector[1] = 1
        graph.add(key, vector); graph.save(str(self.candidate.path))
        self.rejected("graph state changed")

    def test_vector_revision_cannot_reuse_old_checks(self):
        with self.store.db:
            self.store.db.execute("UPDATE dense_vectors SET vector=vector WHERE model=?", (MODEL,))
        self.rejected("graph state changed")

    def test_failed_or_different_query_checks_do_not_change_active_profile(self):
        old = copy.deepcopy(self.regression)
        for changes in ({"known_source_regressions_passed": False}, {"queries_sha256": "wrong"}, {"cases": []}):
            self.regression = {**old, **changes}
            self.rejected()

    def test_low_storage_prevents_configuration_write(self):
        with patch("adams_rag.profile_activation.require_space", side_effect=StoragePause("low")):
            with self.assertRaises(StoragePause):
                self.activate()
        self.assertEqual(active_model(self.directory), DEFAULT_EMBEDDING)

    def test_retrieval_generation_rolls_back_with_source_transaction(self):
        before = retrieval_fingerprint(self.store.db)
        with self.assertRaises(RuntimeError):
            with self.store.db:
                self.store.db.execute("UPDATE chunks SET text='uncommitted'")
                self.assertNotEqual(before, retrieval_fingerprint(self.store.db))
                raise RuntimeError("rollback")
        self.assertEqual(before, retrieval_fingerprint(self.store.db))
        self.assertEqual(self.activate()["state"], "activated")


if __name__ == "__main__":
    unittest.main()
