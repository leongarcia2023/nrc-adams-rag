import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from adams_rag.embedding_windows import MODEL, WindowPlanner
from adams_rag.store import Store
from adams_rag.storage import StoragePause
from adams_rag.window_backfill import WindowBackfill

FIXTURE_INDEX = Path(__file__).resolve().parent / '.cache/index'


class Encoder:
    def __init__(self):
        self.calls = []
        self.after = lambda: None

    def passage_embed(self, texts, **kwargs):
        self.calls.append(list(texts))
        self.after()
        values = np.zeros((len(texts), 384), dtype='<f4')
        values[:, 0] = 1
        return iter(values)


class WindowBackfillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (FIXTURE_INDEX / "mlx-bge/tokenizer.json").is_file():
            raise unittest.SkipTest("Run python scripts/prepare_test_assets.py for tokenizer integration tests")
        cls.planner = WindowPlanner(FIXTURE_INDEX)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / 'research.sqlite')
        self.encoder = Encoder()
        self.backfill = WindowBackfill(self.store, self.planner, self.encoder)
        self.chunk = self.add('A', long=True)
        self.plan = self.backfill.windows.plan(self.chunk)
        self.assertGreater(len(self.plan.windows), 2)

    def tearDown(self):
        self.store.db.close()
        self.temp.cleanup()

    def add(self, accession, long=False, base=True):
        text = 'RV-493.2-A set-pressure=500psig. ' * 90 if long else 'Valve test value 12.'
        with self.store.db:
            self.store.put({'AccessionNumber': accession, 'DocumentTitle': 'Document ' + accession, 'content': text})
        chunks = self.store.db.execute('SELECT id FROM chunks WHERE accession=?', (accession,)).fetchall()
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0][0]
        if base:
            plan = self.backfill.windows.plan(chunk)
            vector = np.zeros(384, dtype='<f4'); vector[0] = 1
            with self.store.db:
                self.store.db.execute('INSERT INTO dense_vectors VALUES(?,?,?,?,?)', (chunk, MODEL, 384, vector.tobytes(), plan.source_hash))
        return chunk

    def test_reuses_prefix_and_marks_short_passage_without_encoding_it(self):
        short = self.add('SHORT')
        result = self.backfill.sync()
        self.assertEqual(result['embedded'], len(self.plan.windows) - 1)
        self.assertEqual(result['completed_passages'], 2)
        self.assertEqual(self.backfill.windows.resolve(short)['windows'][0]['storage'], 'base')
        self.assertEqual(self.backfill.pending(), [])
        self.assertEqual(self.backfill.sync()['selected_passages'], 0)
        self.assertEqual(result['role'], 'window_preparation')
        self.assertNotIn('retrieval_activated', result)

    def test_vector_limit_and_reopening_resume_only_missing_vectors(self):
        first = self.backfill.sync(vector_limit=1)
        self.assertEqual(first['embedded'], 1)
        self.assertTrue(first['vector_limit_reached'])
        self.assertEqual(first['completed_passages'], 0)
        self.store.db.close()
        self.store = Store(self.root / 'research.sqlite')
        self.backfill = WindowBackfill(self.store, self.planner, self.encoder)
        second = self.backfill.sync()
        self.assertEqual(second['embedded'], len(self.plan.windows) - 2)
        self.assertEqual(second['completed_passages'], 1)
        self.assertEqual(sum(map(len, self.encoder.calls)), len(self.plan.windows) - 1)

    def test_yield_after_eight_commits_then_resume_remaining_work(self):
        for n in range(4):
            self.add('LONG' + str(n), long=True)
        expected = sum(len(self.backfill.windows.plan(chunk).windows) - 1 for chunk in self.backfill.pending())
        waiting = [False]
        self.encoder.after = lambda: waiting.__setitem__(0, True)
        first = self.backfill.sync(should_yield=lambda: waiting[0])
        self.assertEqual(first['embedded'], 8)
        self.assertTrue(first['yielded_for_interactive'])
        self.assertEqual(first['stored_window_vectors'], 8)
        self.encoder.after = lambda: None
        second = self.backfill.sync()
        self.assertEqual(first['embedded'] + second['embedded'], expected)
        self.assertEqual(self.backfill.pending(), [])
        self.assertTrue(all(len(batch) <= 8 for batch in self.encoder.calls))

    def test_queued_query_before_first_batch_keeps_work_pending(self):
        result = self.backfill.sync(should_yield=lambda: True)
        self.assertTrue(result['yielded_for_interactive'])
        self.assertEqual(result['embedded'], 0)
        self.assertEqual(self.encoder.calls, [])
        self.assertEqual(self.backfill.pending(), [self.chunk])

    def test_source_change_during_inference_is_requeued_without_stale_commit(self):
        def revise():
            with self.store.db:
                self.store.put({'AccessionNumber': 'A', 'DocumentTitle': 'Changed during inference'})
        self.encoder.after = revise
        result = self.backfill.sync()
        self.assertEqual(result['embedded'], 0)
        self.assertEqual(result['stale_passages'], 1)
        self.assertEqual(self.backfill.pending(), [self.chunk])
        self.encoder.after = lambda: None
        self.assertEqual(self.backfill.sync()['completed_passages'], 1)

    def test_missing_reused_base_invalidates_marker_and_repairs_only_that_window(self):
        self.backfill.sync()
        with self.store.db:
            self.store.db.execute('DELETE FROM dense_vectors WHERE chunk_id=?', (self.chunk,))
        self.assertEqual(self.backfill.pending(), [self.chunk])
        repair = self.backfill.sync()
        self.assertEqual(repair['embedded'], 1)
        self.assertEqual(repair['completed_passages'], 1)
        self.assertEqual(self.backfill.windows.resolve(self.chunk)['windows'][0]['storage'], 'stored')

    def test_corrupt_vector_invalidates_completion_and_is_replaced(self):
        self.backfill.sync()
        with self.store.db:
            self.store.db.execute('UPDATE embedding_windows SET vector=? WHERE chunk_id=? AND ordinal=2',
                                  (np.full(384, np.nan, dtype='<f4').tobytes(), self.chunk))
        self.assertEqual(self.backfill.pending(), [self.chunk])
        self.assertEqual(self.backfill.sync()['embedded'], 1)
        self.assertEqual(self.backfill.pending(), [])

    def test_unfinished_plan_cannot_be_marked_complete(self):
        self.backfill.windows.register(self.chunk, self.plan)
        self.assertFalse(self.backfill.mark_complete(self.chunk))
        self.assertEqual(self.backfill.pending(), [self.chunk])

    def test_cancel_after_encoding_does_not_commit_unchecked_batch(self):
        cancel = [False]
        self.encoder.after = lambda: cancel.__setitem__(0, True)
        def progress():
            if cancel[0]:
                raise InterruptedError('stop')
        with self.assertRaises(InterruptedError):
            self.backfill.sync(progress=progress)
        self.assertEqual(self.backfill.status()['stored_window_vectors'], 0)
        self.assertEqual(self.backfill.pending(), [self.chunk])

    def test_storage_pause_preserves_already_committed_batch(self):
        for n in range(4):
            self.add('LONG' + str(n), long=True)
        def progress():
            if self.backfill.status()['stored_window_vectors'] >= 8:
                raise StoragePause('reserve')
        with self.assertRaises(StoragePause):
            self.backfill.sync(progress=progress)
        self.assertEqual(self.backfill.status()['stored_window_vectors'], 8)
        self.assertTrue(self.backfill.pending())
        self.backfill.sync()
        self.assertEqual(self.backfill.pending(), [])

    def test_invalid_batch_rejected_before_any_vector_is_written(self):
        invalid = [np.zeros(384, dtype='<f4')] * (len(self.plan.windows) - 1)
        with patch.object(self.encoder, 'passage_embed', return_value=invalid):
            with self.assertRaises(ValueError):
                self.backfill.sync()
        self.assertEqual(self.backfill.status()['stored_window_vectors'], 0)
        self.assertEqual(self.backfill.pending(), [self.chunk])

    def test_revised_old_id_is_not_skipped_after_later_passages_complete(self):
        later = self.add('LATER')
        self.backfill.sync()
        with self.store.db:
            self.store.put({'AccessionNumber': 'A', 'DocumentTitle': 'A new title'})
        self.assertEqual(self.backfill.pending(), [self.chunk])
        self.assertGreater(later, self.chunk)
        self.backfill.sync()
        self.assertEqual(self.backfill.pending(), [])

    def test_nonindexed_source_is_skipped_until_status_changes(self):
        other = self.add('LATER')
        with self.store.db:
            self.store.db.execute("UPDATE documents SET status='unavailable_text' WHERE accession='A'")
        self.assertEqual(self.backfill.pending(), [other])
        self.backfill.sync()
        with self.store.db:
            self.store.db.execute("UPDATE documents SET status='indexed' WHERE accession='A'")
        self.assertEqual(self.backfill.pending(), [self.chunk])

    def test_invalid_limits_reject_without_planning(self):
        for args in ({'limit': True}, {'limit': 257}, {'vector_limit': True}, {'vector_limit': 129}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.backfill.sync(**args)
        self.assertEqual(self.backfill.status()['planned_passages'], 0)

    def test_reuse_only_completes_short_inputs_and_leaves_long_or_unembedded_inputs_pending(self):
        short = self.add('SHORT')
        without_base = self.add('NO_BASE', base=False)
        before = [tuple(r) for r in self.store.db.execute('SELECT * FROM dense_vectors ORDER BY chunk_id,model')]
        result = self.backfill.prepare_existing()
        self.assertEqual(result['selected_passages'], 2)
        self.assertEqual(result['completed_this_job'], 1)
        self.assertEqual(result['pending_this_job'], 1)
        self.assertEqual(result['embedded'], 0)
        self.assertEqual(self.encoder.calls, [])
        self.assertEqual(result['stored_window_vectors'], 0)
        self.assertEqual(before, [tuple(r) for r in self.store.db.execute('SELECT * FROM dense_vectors ORDER BY chunk_id,model')])
        self.assertEqual(self.backfill.windows.resolve(short)['windows'][0]['storage'], 'base')
        self.assertEqual(self.backfill.windows.resolve(self.chunk)['state'], 'pending')
        self.assertEqual(self.backfill.windows.resolve(without_base)['state'], 'no_plan')
        self.assertEqual(self.backfill.pending(), [self.chunk, without_base])
        # Reopening skips the already planned long input instead of stalling on it.
        self.store.db.close()
        self.store = Store(self.root / 'research.sqlite')
        self.backfill = WindowBackfill(self.store, self.planner, self.encoder)
        later = self.add('LATER')
        self.assertEqual(self.backfill.prepare_existing()['selected_passages'], 1)
        self.assertEqual(self.backfill.windows.resolve(later)['state'], 'complete')
        self.assertEqual(self.backfill.prepare_existing()['selected_passages'], 0)
        # The normal encoder can finish the same retained plans and missing base.
        self.backfill.sync()
        self.assertEqual(self.backfill.pending(), [])

    def test_reuse_only_rejects_stale_corrupt_and_wrong_profile_bases(self):
        stale = self.add('STALE'); corrupt = self.add('CORRUPT'); other = self.add('OTHER')
        with self.store.db:
            self.store.db.execute('UPDATE dense_vectors SET content_hash=? WHERE chunk_id=?', ('0' * 64, stale))
            self.store.db.execute('UPDATE dense_vectors SET vector=? WHERE chunk_id=?',
                                 (np.full(384, np.nan, dtype='<f4').tobytes(), corrupt))
            self.store.db.execute('UPDATE dense_vectors SET model=? WHERE chunk_id=?', ('different-profile', other))
        result = self.backfill.prepare_existing()
        self.assertEqual(result['completed_this_job'], 0)
        self.assertEqual(result['selected_passages'], 3)
        self.assertEqual(self.backfill.windows.resolve(other)['state'], 'no_plan')
        for chunk in (stale, corrupt):
            self.assertEqual(self.backfill.windows.resolve(chunk)['state'], 'pending')
        self.assertEqual(self.encoder.calls, [])

    def test_reuse_only_source_change_during_planning_cannot_publish_stale_plan(self):
        register = self.backfill.windows.register
        def revise(chunk, plan):
            with self.store.db:
                self.store.put({'AccessionNumber': 'A', 'DocumentTitle': 'Changed before registration'})
            return register(chunk, plan)
        with patch.object(self.backfill.windows, 'register', side_effect=revise):
            result = self.backfill.prepare_existing()
        self.assertEqual(result['stale_passages'], 1)
        self.assertEqual(result['completed_this_job'], 0)
        self.assertEqual(self.backfill.windows.resolve(self.chunk)['state'], 'no_plan')

    def test_reuse_only_storage_pause_preserves_plans_and_retry_advances(self):
        short = self.add('SHORT')
        def progress():
            if self.backfill.status()['planned_passages']:
                raise StoragePause('reserve')
        with self.assertRaises(StoragePause):
            self.backfill.prepare_existing(progress=progress)
        self.assertEqual(self.backfill.status()['planned_passages'], 1)
        resumed = self.backfill.prepare_existing()
        self.assertEqual(resumed['selected_passages'], 1)
        self.assertEqual(self.backfill.windows.resolve(short)['state'], 'complete')
        self.assertEqual(self.encoder.calls, [])

    def test_reuse_only_limits_and_nonindexed_source(self):
        for limit in (True, 0, 257):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.backfill.prepare_existing(limit)
        with self.store.db:
            self.store.db.execute("UPDATE documents SET status='unavailable_text' WHERE accession='A'")
        self.assertEqual(self.backfill.prepare_existing()['selected_passages'], 0)
        with self.store.db:
            self.store.db.execute("UPDATE documents SET status='indexed' WHERE accession='A'")
        self.assertEqual(self.backfill.prepare_existing()['selected_passages'], 1)

    def test_worker_reuse_only_does_not_call_model_or_claim_all_windows_complete(self):
        from adams_rag.window_worker import main
        output = io.StringIO()
        with patch('adams_rag.window_worker.run_operation', side_effect=AssertionError('model loaded')) as operation, \
                patch('adams_rag.window_worker.signal.signal'), contextlib.redirect_stdout(output):
            main(['--db', str(self.store.path), '--directory', str(FIXTURE_INDEX),
                  '--state-directory', str(self.root / 'reuse-worker'), '--reuse-only', '--jobs', '2'])
        operation.assert_not_called()
        last = json.loads(output.getvalue().splitlines()[-1])
        self.assertEqual(last['mode'], 'reuse_only')
        self.assertEqual(last['state'], 'caught_up')
        self.assertIn('missing window vectors may still remain', last['detail'])
        self.assertEqual(self.backfill.pending(), [self.chunk])

    def test_worker_reuse_stop_keeps_unstarted_plan_pending(self):
        from adams_rag.window_worker import main
        directory = self.root / 'reuse-stop'; directory.mkdir(); (directory / 'STOP').touch()
        with patch('adams_rag.window_worker.run_operation') as operation, \
                patch('adams_rag.window_worker.signal.signal'), contextlib.redirect_stdout(io.StringIO()):
            main(['--db', str(self.store.path), '--directory', str(FIXTURE_INDEX),
                  '--state-directory', str(directory), '--reuse-only'])
        operation.assert_not_called()
        self.assertEqual(self.backfill.status()['planned_passages'], 0)
        self.assertEqual(json.loads((directory / 'STATUS.json').read_text())['state'], 'paused')

    def test_worker_zero_work_yield_does_not_consume_job_limit(self):
        from adams_rag.window_worker import main
        responses = [{'selected_passages': 0, 'embedded': 0, 'yielded_for_interactive': True},
                     {'selected_passages': 1, 'embedded': 1, 'yielded_for_interactive': False}]
        output = io.StringIO()
        with patch('adams_rag.window_worker.run_operation', side_effect=responses) as operation, \
                patch('adams_rag.window_worker.signal.signal'), contextlib.redirect_stdout(output):
            main(['--db', str(self.store.path), '--directory', str(FIXTURE_INDEX),
                  '--state-directory', str(self.root / 'worker')])
        self.assertEqual(operation.call_count, 2)
        states = [json.loads(line)['state'] for line in output.getvalue().splitlines()]
        self.assertEqual(states[-1], 'job_limit')
        self.assertNotIn('caught_up', states)

    def test_worker_stop_file_is_checked_before_creating_schema(self):
        from adams_rag.window_worker import main
        directory = self.root / 'stopped'; directory.mkdir(); (directory / 'STOP').touch()
        other = self.root / 'never-created.sqlite'
        with patch('adams_rag.window_worker.signal.signal'), contextlib.redirect_stdout(io.StringIO()):
            main(['--db', str(other), '--directory', str(FIXTURE_INDEX), '--state-directory', str(directory)])
        self.assertFalse(other.exists())
        self.assertEqual(json.loads((directory / 'STATUS.json').read_text())['state'], 'paused')


if __name__ == '__main__':
    unittest.main()
