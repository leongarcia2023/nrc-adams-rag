import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import hashlib
import json

try:
    import numpy as np
    import usearch
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

from adams_rag.store import Store
from adams_rag.semantic_index import SemanticIndex


class Encoder:
    def passage_embed(self, texts, **kwargs):
        return (np.asarray([1., 0.] if "valve" in t else [0., 1.], dtype=np.float32) for t in texts)

    def query_embed(self, text):
        return iter([np.asarray([1., 0.] if "valve" in text else [0., 1.], dtype=np.float32)])


@unittest.skipUnless(AVAILABLE, "Optional ANN dependencies are not installed")
class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "db.sqlite")
        with self.store.db:
            for id_, text in [("A", "valve evidence"), ("B", "pump evidence")]:
                self.store.put({"AccessionNumber": id_, "content": text, "DocumentTitle": "Report", "DocketNumber": ["05000373"]})
        self.index = SemanticIndex(self.store, self.root / "index", "test-vectors", Encoder())

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def test_persistent_index_and_idempotent_resume(self):
        self.assertEqual(self.index.sync()["embedded"], 2)
        self.assertEqual(self.index.sync()["embedded"], 0)
        reopened = SemanticIndex(self.store, self.root / "index", "test-vectors", Encoder())
        self.assertEqual(reopened.search("valve", 1)[0]["accession"], "A")

    def test_priority_preserves_later_served_passages_before_fifo_and_then_continues(self):
        baseline = SemanticIndex(self.store, self.root / "index", "baseline", Encoder())
        baseline.sync(accessions=["B"])
        with self.store.db:
            self.store.put({"AccessionNumber": "C", "content": "New valve evidence"})
        first = self.index.sync(limit=1, priority_model="baseline")
        keys = [r[0] for r in self.store.db.execute("SELECT c.accession FROM dense_vectors v JOIN chunks c ON c.id=v.chunk_id WHERE v.model=?", (self.index.model,))]
        self.assertEqual(keys, ["B"])
        self.assertEqual(first["priority_selected"], 1)
        second = self.index.sync(limit=2, priority_model="baseline")
        self.assertEqual((second["priority_selected"], second["embedded"]), (0, 2))
        self.assertEqual(baseline.status()["vectors"], 1)
        self.assertEqual(self.index.status()["vectors"], 3)

    def test_priority_and_fifo_fill_one_budget_without_duplicates(self):
        baseline = SemanticIndex(self.store, self.root / "index", "baseline", Encoder())
        baseline.sync(accessions=["B"])
        result = self.index.sync(limit=2, priority_model="baseline")
        self.assertEqual((result["priority_selected"], result["embedded"]), (1, 2))
        self.assertEqual(self.index.status()["vectors"], 2)

    def test_priority_respects_accession_scope(self):
        baseline = SemanticIndex(self.store, self.root / "index", "baseline", Encoder())
        baseline.sync(accessions=["B"])
        result = self.index.sync(limit=2, accessions=["A"], priority_model="baseline")
        self.assertEqual((result["priority_selected"], result["embedded"]), (0, 1))
        self.assertEqual(self.index.coverage_against("baseline")["missing_passages"], 1)

    def test_pending_selection_refills_old_holes_before_new_passages_and_respects_scope(self):
        with self.store.db:
            for number in range(30):
                self.store.put({"AccessionNumber": f"C{number:02}", "content": f"pump record {number}"})
        self.index.sync()
        with self.store.db:
            # Removing embeddings can leave holes far behind the newest ID.
            self.store.db.execute("DELETE FROM dense_vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE accession IN ('B','C17'))")
            self.store.put({"AccessionNumber": "LATEST", "content": "new valve record"})
        self.assertEqual(self.index.sync(limit=1, accessions=["C17", "LATEST"])["embedded"], 1)
        selected = [row[0] for row in self.store.db.execute("""SELECT c.accession FROM dense_vectors v
            JOIN chunks c ON c.id=v.chunk_id WHERE c.accession IN ('B','C17','LATEST')""")]
        self.assertEqual(selected, ["C17"])
        self.assertEqual(self.index.sync(limit=1)["embedded"], 1)
        pending = [row[0] for row in self.store.db.execute("""SELECT c.accession FROM chunks c
            LEFT JOIN dense_vectors v ON v.chunk_id=c.id AND v.model=? WHERE v.chunk_id IS NULL""", (self.index.model,))]
        self.assertEqual(pending, ["LATEST"])

    def test_low_storage_does_not_build_pending_index_or_start_embedding(self):
        from adams_rag.storage import StoragePause
        self.store.db.execute("DROP INDEX IF EXISTS chunks_pending_ids")
        with patch('adams_rag.semantic_index.require_space', side_effect=StoragePause('low')) as space:
            with self.assertRaises(StoragePause):
                self.index.sync()
        self.assertEqual(space.call_args.args[0], self.store.path.parent)
        self.assertIsNone(self.store.db.execute("SELECT 1 FROM sqlite_master WHERE name='chunks_pending_ids'").fetchone())
        self.assertEqual(self.index.status()["vectors"], 0)

    def test_cancellation_after_larger_save_group_keeps_commits_and_resumes_without_skips(self):
        with self.store.db:
            for number in range(128):
                self.store.put({"AccessionNumber": f"GROUP{number:03}", "content": f"valve evidence {number}"})
        def stop_after_group(done, total):
            if done >= 64:
                raise InterruptedError('requested stop')
        with self.assertRaises(InterruptedError):
            self.index.sync(limit=1000, batch_size=64, progress=stop_after_group)
        self.assertEqual(self.index.status()["vectors"], 64)
        self.assertEqual(self.index.sync(limit=1000, batch_size=64)["embedded"], 66)
        self.assertEqual(self.index.status()["vectors"], 130)
        self.assertEqual(len(self.index.index), 130)
        missing = self.store.db.execute("""SELECT count(*) FROM chunks c LEFT JOIN dense_vectors v
            ON v.chunk_id=c.id AND v.model=? WHERE v.chunk_id IS NULL""", (self.index.model,)).fetchone()[0]
        self.assertEqual(missing, 0)

    def test_storage_drop_during_encoding_does_not_publish_uncommitted_group(self):
        from adams_rag.storage import StoragePause
        with self.store.db:
            for number in range(128):
                self.store.put({"AccessionNumber": f"LOW{number:03}", "content": f"pump evidence {number}"})
        calls = [0]
        class EncoderWithStorageDrop(Encoder):
            def passage_embed(inner, texts, **kwargs):
                calls[0] += 1
                return super().passage_embed(texts, **kwargs)
        def check_space(*args, **kwargs):
            if calls[0] >= 2:
                raise StoragePause('storage dropped during the second group')
        self.index.encoder = EncoderWithStorageDrop()
        with patch('adams_rag.semantic_index.require_space', side_effect=check_space):
            with self.assertRaises(StoragePause):
                self.index.sync(limit=1000, batch_size=64)
        self.assertEqual(self.index.status()["vectors"], 64)
        self.index.encoder = Encoder()
        self.assertEqual(self.index.sync(limit=1000, batch_size=64)["embedded"], 66)

    def test_coverage_distinguishes_equal_counts_from_equal_passage_sets_and_inputs(self):
        baseline = SemanticIndex(self.store, self.root / "index", "baseline", Encoder())
        baseline.sync(accessions=["B"])
        self.index.sync(accessions=["A"])
        report = self.index.coverage_against("baseline")
        self.assertEqual((report["baseline_vectors"], report["missing_passages"], report["different_input_hashes"]), (1, 1, 0))
        self.index.sync(priority_model="baseline")
        self.assertEqual(self.index.coverage_against("baseline")["missing_passages"], 0)
        with self.store.db:
            self.store.db.execute("UPDATE dense_vectors SET content_hash='different' WHERE model=? AND chunk_id=(SELECT id FROM chunks WHERE accession='B')", (self.index.model,))
        self.assertEqual(self.index.coverage_against("baseline")["different_input_hashes"], 1)
        self.assertFalse(report["retrieval_quality_verified"])

    def test_invalid_priority_is_rejected_before_embedding(self):
        for priority in (self.index.model, "", [], True):
            with self.assertRaises(ValueError):
                self.index.sync(priority_model=priority)
        self.assertEqual(self.index.status()["vectors"], 0)

    def test_replacement_cannot_return_deleted_vector(self):
        self.index.sync()
        with self.store.db:
            self.store.put({"AccessionNumber": "A", "content": "replacement pump evidence", "DocumentTitle": "Report"})
        self.index.sync()
        results = self.index.search("pump")
        self.assertEqual(len(results), 2)
        self.assertTrue(all("valve" not in r["text"] for r in results))

    def test_metadata_filters_apply_to_semantic_results(self):
        self.index.sync()
        self.assertFalse(self.index.search("valve", docket="05000001"))
        self.assertEqual(len(self.index.search("valve", docket="05000373")), 2)
        self.assertFalse(self.index.search("valve", date_from="2024-01-01"))

    def test_metadata_title_refresh_invalidates_dense_vector(self):
        self.index.sync()
        with self.store.db:
            self.store.put({"AccessionNumber": "A", "content": "", "DocumentTitle": "Changed title"}, "api_search_excerpt")
        self.assertEqual(self.index.status()["vectors"], 1)
        self.assertEqual(self.index.sync()["embedded"], 1)

    def test_append_only_update_adds_only_new_vectors(self):
        self.index.sync()
        with self.store.db:
            self.store.put({"AccessionNumber": "C", "content": "Another valve report"})
        self.index.sync()
        manifest = json.loads(self.index.manifest.read_text())
        self.assertEqual((manifest["build_mode"], manifest["vectors_added"], manifest["vectors_removed"]),
                         ("incremental", 1, 0))
        self.assertEqual(len(self.index.index), 3)

    def test_same_key_vector_replacement_and_deletion_update_incrementally(self):
        self.index.sync()
        with self.store.db:
            self.store.db.execute("UPDATE dense_vectors SET vector=? WHERE chunk_id=(SELECT id FROM chunks WHERE accession='A')",
                                  (np.asarray([0., 1.], dtype=np.float32).tobytes(),))
            self.store.db.execute("DELETE FROM dense_vectors WHERE chunk_id=(SELECT id FROM chunks WHERE accession='B')")
        self.index.rebuild(incremental=True)
        manifest = json.loads(self.index.manifest.read_text())
        self.assertEqual((manifest["build_mode"], manifest["vectors_added"], manifest["vectors_removed"]),
                         ("incremental", 1, 2))
        self.assertEqual(len(self.index.index), 1)
        self.assertEqual(self.index.search("pump")[0]["accession"], "A")

    def test_old_manifest_is_rebuilt_before_incremental_updates(self):
        self.index.sync()
        manifest = json.loads(self.index.manifest.read_text())
        manifest.pop("tracking_version")
        self.index.manifest.write_text(json.dumps(manifest))
        self.index.rebuild(incremental=True)
        self.assertEqual(json.loads(self.index.manifest.read_text())["build_mode"], "full")

    def test_incremental_rebuild_keeps_concurrent_new_vectors_for_next_snapshot(self):
        from usearch.index import Index
        self.index.sync()
        with self.store.db:
            id_ = self.store.db.execute("SELECT id FROM chunks WHERE accession='A'").fetchone()[0]
            self.store.db.execute("UPDATE dense_vectors SET vector=? WHERE chunk_id=?", (np.asarray([0., 1.], dtype=np.float32).tobytes(), id_))
        writer = Store(self.root / "db.sqlite")
        original = Index.add
        changed = [False]
        def add_then_update(index, *args, **kwargs):
            result = original(index, *args, **kwargs)
            if not changed[0]:
                changed[0] = True
                with writer.db:
                    writer.put({"AccessionNumber": "C", "content": "pump evidence"})
                    id_ = writer.db.execute("SELECT id FROM chunks WHERE accession='C'").fetchone()[0]
                    source = writer.db.execute("SELECT d.title,c.text FROM chunks c JOIN documents d USING(accession) WHERE c.id=?", (id_,)).fetchone()
                    writer.db.execute("INSERT INTO dense_vectors VALUES(?,?,?,?,?)", (id_, "test-vectors", 2,
                        np.asarray([0., 1.], dtype=np.float32).tobytes(), hashlib.sha256((source[0]+'\n'+source[1]).encode()).hexdigest()))
            return result
        try:
            with patch.object(Index, "add", add_then_update):
                self.index.rebuild(incremental=True)
            self.assertEqual(len(self.index.index), 2)
            self.assertNotEqual(self.index.generation, self.index.fingerprint())
            self.assertEqual(len(self.index.search("pump")), 3)
        finally:
            writer.db.close()

    def test_vector_with_wrong_source_hash_is_withheld_from_search(self):
        self.index.sync()
        with self.store.db:
            self.store.db.execute("UPDATE dense_vectors SET content_hash='wrong' WHERE chunk_id=(SELECT id FROM chunks WHERE accession='A')")
        self.assertEqual([r['accession'] for r in self.index.search('valve')],['B'])

    def test_whole_graph_request_cannot_lose_nodes_to_approximate_traversal(self):
        from types import SimpleNamespace
        self.index.sync()
        native=self.index.index.search
        missing=self.store.db.execute("SELECT id FROM chunks WHERE accession='A'").fetchone()[0]
        def partial(vector,**kwargs):
            result=native(vector,**kwargs)
            if not kwargs.get('exact'):
                return SimpleNamespace(keys=np.asarray([k for k in result.keys if int(k)!=missing],dtype=np.uint64))
            return result
        with patch.object(self.index.index,'search',side_effect=partial):
            matches=self.index.search('valve',limit=2)
        self.assertEqual({r['accession'] for r in matches},{'A','B'})

    def test_revision_during_embedding_cannot_insert_stale_vectors(self):
        writer = Store(self.root / "db.sqlite")
        class RevisingEncoder(Encoder):
            def passage_embed(inner, texts, **kwargs):
                with writer.db:
                    writer.put({"AccessionNumber": "A", "content": "New pump evidence", "DocumentTitle": "Revision"})
                return super().passage_embed(texts, **kwargs)
        self.index.encoder = RevisingEncoder()
        try:
            self.assertEqual(self.index.sync()["embedded"], 1)
            self.assertEqual(self.index.status()["vectors"], 1)
        finally:
            writer.db.close()

    def test_rebuild_manifest_describes_same_database_snapshot_as_vectors(self):
        from usearch.index import Index
        self.index.sync()
        writer = Store(self.root / "db.sqlite")
        original = Index.add
        changed = [False]
        def add_then_update(index, *args, **kwargs):
            result = original(index, *args, **kwargs)
            if not changed[0]:
                changed[0] = True
                with writer.db:
                    writer.put({"AccessionNumber": "C", "content": "pump evidence", "DocumentTitle": "Report"})
                    id_ = writer.db.execute("SELECT id FROM chunks WHERE accession='C'").fetchone()[0]
                    writer.db.execute("INSERT INTO dense_vectors VALUES(?,?,?,?,?)", (id_, "test-vectors", 2,
                        np.asarray([0., 1.], dtype=np.float32).tobytes(), hashlib.sha256(b"Report\npump evidence").hexdigest()))
            return result
        try:
            with patch.object(Index, "add", add_then_update):
                self.index.rebuild()
            self.assertEqual(len(self.index.index), 2)
            self.assertNotEqual(self.index.generation, self.index.fingerprint())
            self.assertEqual(len(self.index.search("pump")), 3)
        finally:
            writer.db.close()


if __name__ == "__main__":
    unittest.main()
