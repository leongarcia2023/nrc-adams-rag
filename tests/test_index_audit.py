import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from usearch.index import Index

from adams_rag.index_audit import audit_profiles
from adams_rag.semantic_index import SemanticIndex
from adams_rag.store import Store


class Encoder:
    def passage_embed(self, texts, **kwargs):
        return [np.asarray([1., 0.], dtype=np.float32) for _ in texts]


class IndexAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "research.sqlite")
        with self.store.db:
            for accession in ["A", "B"]:
                self.store.put({"AccessionNumber": accession, "content": "Valve evidence."})
        self.baseline = SemanticIndex(self.store, self.root / "index", "baseline", Encoder())
        self.candidate = SemanticIndex(self.store, self.root / "index", "candidate", Encoder())

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def audit(self):
        return audit_profiles(self.store.db, self.store.path, self.root / "index", "baseline", "candidate")

    def ready(self):
        self.baseline.sync(); self.candidate.sync()

    def test_all_stored_inputs_vectors_and_graphs_match_without_certifying_quality(self):
        self.ready()
        result = self.audit()
        self.assertTrue(result["coverage_and_integrity_passed"])
        self.assertEqual(result["coverage"], {"baseline_vectors": 2, "missing_passages": 0, "different_input_hashes": 0})
        self.assertFalse(result["retrieval_quality_verified"])
        self.assertFalse(result["full_archive_verified"])
        self.assertFalse(self.store.db.in_transaction)

    def test_equal_vector_counts_do_not_establish_coverage(self):
        self.baseline.sync(accessions=["A"])
        self.candidate.sync(accessions=["B"])
        result = self.audit()
        self.assertTrue(all(r["passed"] for r in result["profiles"].values()))
        self.assertFalse(result["coverage_and_integrity_passed"])
        self.assertEqual(result["coverage"]["missing_passages"], 1)

    def test_stale_input_hash_and_nonfinite_vector_are_detected(self):
        self.ready()
        with self.store.db:
            self.store.db.execute("UPDATE dense_vectors SET content_hash='stale',vector=? WHERE model='candidate'",
                                  (np.asarray([np.nan, 0.], dtype=np.float32).tobytes(),))
        result = self.audit()
        self.assertFalse(result["coverage_and_integrity_passed"])
        errors = result["profiles"]["candidate"]["errors"]
        self.assertEqual(errors["embedding_input_hash_mismatch"], 2)
        self.assertEqual(errors["invalid_vector_values"], 2)
        self.assertEqual(result["coverage"]["different_input_hashes"], 2)

    def test_graph_wrong_values_cannot_hide_behind_matching_manifest_and_count(self):
        self.ready()
        graph = Index.restore(str(self.candidate.path), view=False)
        key = self.store.db.execute("SELECT id FROM chunks ORDER BY id LIMIT 1").fetchone()[0]
        graph.remove(key)
        graph.add(key, np.asarray([-1., 0.], dtype=np.float32))
        graph.save(str(self.candidate.path))
        result = self.audit()
        self.assertEqual(result["profiles"]["candidate"]["errors"], {"graph_vector_mismatch": 1})
        self.assertFalse(result["coverage_and_integrity_passed"])

    def test_graph_wrong_keys_cannot_hide_behind_matching_count(self):
        self.ready()
        graph = Index.restore(str(self.candidate.path), view=False)
        key = self.store.db.execute("SELECT id FROM chunks ORDER BY id LIMIT 1").fetchone()[0]
        graph.remove(key)
        graph.add(999999, np.asarray([1., 0.], dtype=np.float32))
        graph.save(str(self.candidate.path))
        result = self.audit()
        self.assertEqual(result["profiles"]["candidate"]["errors"], {"passage_missing_from_graph": 1})

    def test_empty_or_unpublished_profile_fails_without_touching_active_configuration(self):
        self.baseline.sync()
        self.assertFalse(self.audit()["coverage_and_integrity_passed"])
        self.candidate.sync()
        manifest = json.loads(self.candidate.manifest.read_text())
        manifest["snapshot_generation"] -= 1
        self.candidate.manifest.write_text(json.dumps(manifest))
        self.assertIn("graph_manifest_not_current", self.audit()["profiles"]["candidate"]["errors"])
        self.assertFalse((self.root / "index/ACTIVE_MODEL.json").exists())

    def test_malformed_vector_shape_is_reported(self):
        self.ready()
        with self.store.db:
            self.store.db.execute("UPDATE dense_vectors SET vector=x'00' WHERE model='candidate'")
        self.assertEqual(self.audit()["profiles"]["candidate"]["errors"]["invalid_vector_shape"], 2)

    def test_concurrent_source_replacement_does_not_mix_audit_snapshots(self):
        self.ready()
        writer = Store(self.store.path)
        original = Index.get
        changed = [False]
        def get_then_change(graph, *args, **kwargs):
            result = original(graph, *args, **kwargs)
            if not changed[0]:
                changed[0] = True
                with writer.db:
                    writer.put({"AccessionNumber": "A", "content": "Replacement pump text."})
            return result
        try:
            with patch.object(Index, "get", get_then_change):
                result = self.audit()
            self.assertTrue(result["coverage_and_integrity_passed"])
            self.assertNotEqual(result["profiles"]["candidate"]["snapshot_fingerprint"], self.candidate.fingerprint())
            self.assertFalse(self.audit()["coverage_and_integrity_passed"])
        finally:
            writer.db.close()

    def test_existing_read_transaction_remains_owned_by_caller(self):
        self.ready()
        self.store.db.execute("BEGIN")
        self.assertTrue(self.audit()["coverage_and_integrity_passed"])
        self.assertTrue(self.store.db.in_transaction)
        self.store.db.rollback()


if __name__ == "__main__":
    unittest.main()
