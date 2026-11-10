from dataclasses import replace
import json
from pathlib import Path
import tempfile
import sqlite3
import unittest
from unittest.mock import patch

import numpy as np

from adams_rag.embedding_windows import WindowPlanner, MODEL, POLICY, text_hash
from adams_rag.storage import StoragePause
from adams_rag.store import Store
from adams_rag.window_store import WindowStore, StaleWindowInput

FIXTURE_INDEX = Path(__file__).resolve().parent / '.cache/index'


class WindowStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (FIXTURE_INDEX / "mlx-bge/tokenizer.json").is_file():
            raise unittest.SkipTest("Run python scripts/prepare_test_assets.py for tokenizer integration tests")
        cls.planner = WindowPlanner(FIXTURE_INDEX)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "research.sqlite")
        self.acc = "ML24017A120"
        with self.store.db:
            self.store.put({"AccessionNumber": self.acc, "DocumentTitle": "Sample valve record",
                            "content": "Valve set-pressure 500 psig. " * 90})
        self.chunk = self.store.db.execute("SELECT min(id) FROM chunks").fetchone()[0]
        self.windows = WindowStore(self.store, self.planner)
        self.plan = self.windows.plan(self.chunk)
        self.assertGreater(len(self.plan.windows), 1)
        self.assertTrue(self.plan.windows[0].base_compatible)

    def tearDown(self):
        self.store.db.close()
        self.temp.cleanup()

    @staticmethod
    def vector(index=0):
        result = np.zeros(384, dtype="<f4")
        result[index] = 1
        return result

    def base(self, model=MODEL, content_hash=None, vector=None):
        vector = self.vector() if vector is None else vector
        with self.store.db:
            self.store.db.execute("INSERT OR REPLACE INTO dense_vectors VALUES(?,?,?,?,?)",
                (self.chunk, model, 384, vector.tobytes(), content_hash or self.plan.source_hash))

    def complete(self):
        self.base()
        self.windows.register(self.chunk, self.plan)
        self.windows.commit(self.chunk, self.plan,
            {w.ordinal: self.vector(w.ordinal) for w in self.windows.missing(self.chunk, self.plan)})

    def test_partial_progress_survives_reopen_and_reuses_exact_prefix(self):
        self.base()
        self.windows.register(self.chunk, self.plan)
        state = self.windows.resolve(self.chunk)
        self.assertEqual(state["state"], "pending")
        self.assertEqual(state["windows"][0]["storage"], "base")
        missing = self.windows.missing(self.chunk, self.plan)
        self.assertEqual([w.ordinal for w in missing], list(range(2, len(self.plan.windows)+1)))
        self.windows.commit(self.chunk, self.plan, {missing[0].ordinal: self.vector(2)})
        generation = self.windows.generation()
        self.store.db.close()
        self.store = Store(Path(self.temp.name) / "research.sqlite")
        self.windows = WindowStore(self.store, self.planner)
        self.assertFalse(self.windows.register(self.chunk, self.plan))
        self.assertEqual(self.windows.generation(), generation)
        self.assertEqual(self.windows.resolve(self.chunk)["windows"][1]["storage"], "stored")
        remaining = self.windows.missing(self.chunk, self.plan)
        if remaining:
            self.windows.commit(self.chunk, self.plan, {w.ordinal: self.vector(w.ordinal) for w in remaining})
        self.assertEqual(self.windows.resolve(self.chunk)["state"], "complete")

    def test_short_input_needs_no_extra_vector_when_base_is_present(self):
        with self.store.db:
            self.store.put({"AccessionNumber": "ML24017A121", "DocumentTitle": "Short", "content": "Valve value 12."})
        chunk = self.store.db.execute("SELECT id FROM chunks WHERE accession='ML24017A121'").fetchone()[0]
        plan = self.windows.plan(chunk)
        with self.store.db:
            self.store.db.execute("INSERT INTO dense_vectors VALUES(?,?,?,?,?)", (chunk, MODEL, 384, self.vector().tobytes(), plan.source_hash))
        self.windows.register(chunk, plan)
        self.assertEqual(self.windows.resolve(chunk)["state"], "complete")
        self.assertEqual(self.windows.missing(chunk, plan), [])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM embedding_windows WHERE chunk_id=? AND vector IS NOT NULL", (chunk,)).fetchone()[0], 0)

    def test_missing_or_wrong_profile_prefix_does_not_count_as_complete(self):
        self.windows.register(self.chunk, self.plan)
        self.base(model="unrelated-model")
        self.assertEqual(self.windows.resolve(self.chunk)["windows"][0]["storage"], "missing")
        self.base(content_hash="0"*64)
        self.assertEqual(self.windows.resolve(self.chunk)["windows"][0]["storage"], "missing")
        self.base(vector=np.full(384, np.nan, dtype="<f4"))
        self.assertEqual(self.windows.resolve(self.chunk)["windows"][0]["storage"], "missing")

    def test_full_window_set_can_be_completed_before_base_indexing(self):
        self.windows.register(self.chunk, self.plan)
        self.windows.commit(self.chunk, self.plan, {w.ordinal: self.vector(w.ordinal) for w in self.plan.windows})
        state = self.windows.resolve(self.chunk)
        self.assertEqual(state["state"], "complete")
        self.assertEqual({w["storage"] for w in state["windows"]}, {"stored"})

    def test_identical_retry_preserves_vectors_and_generation(self):
        self.complete()
        generation = self.windows.generation()
        self.assertEqual(self.windows.commit(self.chunk, self.plan, {1: self.vector(8), 2: self.vector(9)}), 0)
        self.assertEqual(self.windows.generation(), generation)
        self.assertEqual(self.windows.resolve(self.chunk)["windows"][0]["vector"], self.vector().tobytes())

    def test_source_revision_between_inference_and_commit_is_rejected(self):
        self.windows.register(self.chunk, self.plan)
        with self.store.db:
            self.store.db.execute("UPDATE chunks SET text='Different retained passage' WHERE id=?", (self.chunk,))
        with self.assertRaises(StaleWindowInput):
            self.windows.commit(self.chunk, self.plan, {2: self.vector(2)})
        self.assertEqual(self.windows.resolve(self.chunk)["state"], "no_plan")

    def test_title_change_invalidates_windows_without_deleting_source_text(self):
        self.complete()
        text = self.store.db.execute("SELECT text FROM chunks WHERE id=?", (self.chunk,)).fetchone()[0]
        ids = [w["id"] for w in self.windows.resolve(self.chunk)["windows"]]
        generation = self.windows.generation()
        with self.store.db:
            self.store.put({"AccessionNumber": self.acc, "DocumentTitle": "Revised title"})
        self.assertEqual(self.windows.resolve(self.chunk)["state"], "no_plan")
        self.assertEqual(self.store.db.execute("SELECT text FROM chunks WHERE id=?", (self.chunk,)).fetchone()[0], text)
        self.assertGreater(self.windows.generation(), generation)
        changed = {r[0] for r in self.store.db.execute("SELECT window_id FROM window_changes WHERE generation>?", (generation,))}
        self.assertTrue(set(ids) <= changed)
        with self.assertRaises(StaleWindowInput):
            self.windows.register(self.chunk, self.plan)

    def test_base_insert_update_delete_advance_reused_window_generation(self):
        self.windows.register(self.chunk, self.plan)
        before = self.windows.generation()
        self.base()
        self.assertGreater(self.windows.generation(), before)
        before = self.windows.generation()
        with self.store.db:
            self.store.db.execute("UPDATE dense_vectors SET vector=? WHERE chunk_id=? AND model=?", (self.vector(9).tobytes(), self.chunk, MODEL))
        self.assertGreater(self.windows.generation(), before)
        self.assertEqual(self.windows.resolve(self.chunk)["windows"][0]["vector"], self.vector(9).tobytes())
        before = self.windows.generation()
        with self.store.db:
            self.store.db.execute("DELETE FROM dense_vectors WHERE chunk_id=? AND model=?", (self.chunk, MODEL))
        self.assertGreater(self.windows.generation(), before)
        self.assertEqual(self.windows.resolve(self.chunk)["windows"][0]["storage"], "missing")

    def test_chunk_deletion_cascades_and_leaves_graph_removal_record(self):
        self.complete()
        ids = {w["id"] for w in self.windows.resolve(self.chunk)["windows"]}
        before = self.windows.generation()
        with self.store.db:
            self.store.db.execute("DELETE FROM chunks WHERE id=?", (self.chunk,))
        self.assertEqual(self.windows.resolve(self.chunk)["state"], "no_plan")
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM embedding_windows").fetchone()[0], 0)
        changed = {r[0] for r in self.store.db.execute("SELECT window_id FROM window_changes WHERE generation>?", (before,))}
        self.assertTrue(ids <= changed)

    def test_malformed_or_partial_vector_batch_cannot_commit(self):
        self.windows.register(self.chunk, self.plan)
        for bad in [np.full(384, np.nan), np.zeros(384), np.ones(3), np.ones((1,384))]:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                self.windows.commit(self.chunk, self.plan, {1: self.vector(), 2: bad})
            self.assertTrue(all(w["vector"] is None for w in self.windows.resolve(self.chunk)["windows"]))

    def test_storage_stop_leaves_previous_vectors_and_plan_unchanged(self):
        self.base()
        self.windows.register(self.chunk, self.plan)
        before = self.windows.resolve(self.chunk)
        with patch("adams_rag.window_store.require_space", side_effect=StoragePause("test reserve")):
            with self.assertRaises(StoragePause):
                self.windows.commit(self.chunk, self.plan, {2: self.vector(2)})
        self.assertEqual(self.windows.resolve(self.chunk), before)

    def test_forged_plan_and_corrupt_stored_offsets_are_rejected(self):
        forged = replace(self.plan, windows=(replace(self.plan.windows[0], end_character=1), *self.plan.windows[1:]))
        with self.assertRaises(StaleWindowInput):
            self.windows.register(self.chunk, forged)
        self.windows.register(self.chunk, self.plan)
        with self.store.db:
            self.store.db.execute("UPDATE embedding_windows SET end_character=1 WHERE ordinal=1")
        self.assertEqual(self.windows.resolve(self.chunk)["state"], "invalid_plan")
        with self.assertRaises(ValueError):
            self.windows.register(self.chunk, self.plan)
        with self.assertRaises(StaleWindowInput):
            self.windows.commit(self.chunk, self.plan, {2: self.vector(2)})

    def test_register_does_not_commit_callers_transaction(self):
        self.store.db.execute("BEGIN IMMEDIATE")
        self.windows.register(self.chunk, self.plan)
        self.assertTrue(self.store.db.in_transaction)
        self.store.db.rollback()
        self.assertEqual(self.windows.resolve(self.chunk)["state"], "no_plan")

    def test_reader_uses_one_snapshot_during_concurrent_title_revision(self):
        self.complete()
        writer = Store(self.store.path)
        original = self.windows._source
        changed = [False]
        def source(chunk_id):
            value = original(chunk_id)
            if not changed[0]:
                changed[0] = True
                with writer.db:
                    writer.put({"AccessionNumber": self.acc, "DocumentTitle": "Concurrent title revision"})
            return value
        try:
            with patch.object(self.windows, "_source", side_effect=source):
                view = self.windows.resolve(self.chunk)
            self.assertEqual(view["state"], "complete")
            self.assertEqual(view["source_hash"], self.plan.source_hash)
            self.assertEqual(self.windows.resolve(self.chunk)["state"], "no_plan")
            self.assertFalse(self.store.db.in_transaction)
        finally:
            writer.db.close()

    def test_pinned_tokenizer_rejects_changed_file(self):
        root = Path(self.temp.name) / "other"
        (root / "mlx-bge").mkdir(parents=True)
        (root / "mlx-bge/tokenizer.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "pinned"):
            WindowPlanner(root)

    def test_unrecognized_schema_is_not_silently_reinitialized(self):
        with self.store.db:
            self.store.db.execute("DELETE FROM window_storage_schema")
        with self.assertRaisesRegex(ValueError, "explicit migration"):
            WindowStore(self.store, self.planner)
        with self.assertRaisesRegex(ValueError, "unsupported schema"):
            WindowStore(self.store, self.planner, initialize=False)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM window_storage_schema").fetchone()[0], 0)

    def test_profile_identity_cannot_change_under_existing_window_references(self):
        self.complete()
        for sql in ["UPDATE embedding_window_profiles SET tokenizer_hash='wrong'", "DELETE FROM embedding_window_profiles"]:
            with self.subTest(sql=sql), self.assertRaisesRegex(sqlite3.IntegrityError, "immutable"):
                with self.store.db:
                    self.store.db.execute(sql)
        self.assertEqual(self.windows.resolve(self.chunk)["state"], "complete")


if __name__ == "__main__":
    unittest.main()
