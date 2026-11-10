import contextlib
import fcntl
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adams_rag.index_worker import main
from adams_rag.mlx_embeddings import EmbeddingBusy
from adams_rag.storage import StoragePause


class WindowScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.directory=self.root/'index';self.directory.mkdir()
        self.args=['--db',str(self.root/'research.sqlite'),'--directory',str(self.directory),
                   '--backend','mlx','--window-jobs','2','--window-passages','256','--window-vectors','128']

    def tearDown(self):
        self.temp.cleanup()

    def run_worker(self,responses,args=None):
        stream=io.StringIO()
        with patch('adams_rag.index_worker.run_operation',side_effect=responses) as operation, \
                patch('adams_rag.index_worker.signal.signal'),patch('adams_rag.index_worker.time.sleep'),contextlib.redirect_stdout(stream):
            main(args or self.args)
        return operation.call_args_list,[json.loads(line) for line in stream.getvalue().splitlines()]

    @staticmethod
    def base(count=1000,yielded=False):
        return {'embedded':count,'yielded_for_interactive':yielded}

    @staticmethod
    def windows(selected=256,embedded=128,yielded=False):
        return {'embedded':embedded,'selected_passages':selected,'yielded_for_interactive':yielded}

    def test_bounded_window_jobs_alternate_with_prefix_work(self):
        calls,states=self.run_worker([self.base(),self.windows(),self.windows(),self.base(0),self.windows(0,0)])
        self.assertEqual([call.args[1]['operation'] for call in calls],['sync','window_sync','window_sync','sync','window_sync'])
        self.assertEqual(calls[1].args[1]['limit'],256)
        self.assertEqual(calls[1].args[1]['vector_limit'],128)
        self.assertEqual(states[-1]['state'],'complete')
        self.assertEqual(states[-1]['window_backfill']['remaining_jobs'],0)

    def test_empty_prefix_queue_does_not_finish_while_windows_remain(self):
        calls,states=self.run_worker([self.base(0),self.windows(10,0),self.windows(0,0),self.base(0),self.windows(0,0)])
        self.assertEqual(len(calls),5)
        self.assertEqual(states[-1]['state'],'complete')

    def test_busy_slot_retries_same_window_phase(self):
        calls,states=self.run_worker([self.base(),EmbeddingBusy('shared slot'),self.windows(),self.windows(0,0),self.base(0),self.windows(0,0)])
        self.assertEqual([c.args[1]['operation'] for c in calls][:4],['sync','window_sync','window_sync','window_sync'])
        self.assertIn('waiting_for_model',[s['state'] for s in states])

    def test_query_yield_keeps_window_allowance_and_committed_progress(self):
        calls,states=self.run_worker([self.base(0),self.windows(12,8,True),self.windows(4,2),self.windows(0,0),self.base(0),self.windows(0,0)])
        self.assertEqual([c.args[1]['operation'] for c in calls],['sync','window_sync','window_sync','window_sync','sync','window_sync'])
        waiting=next(s for s in states if s['state']=='waiting_for_model')
        self.assertEqual(waiting['window_backfill']['remaining_jobs'],2)
        self.assertEqual(waiting['window_backfill']['last_job']['embedded'],8)

    def test_zero_work_prefix_yield_does_not_skip_to_window_phase(self):
        calls,_=self.run_worker([self.base(0,True),self.base(0),self.windows(0,0)])
        self.assertEqual([c.args[1]['operation'] for c in calls],['sync','sync','window_sync'])

    def test_partial_prefix_progress_allows_window_turn_after_query_yield(self):
        calls,_=self.run_worker([self.base(8,True),self.windows(0,0),self.base(0),self.windows(0,0)])
        self.assertEqual([c.args[1]['operation'] for c in calls],['sync','window_sync','sync','window_sync'])

    def test_window_stop_before_start_does_not_stop_prefix_indexing(self):
        directory=self.directory/'windows';directory.mkdir();(directory/'STOP').touch()
        calls,states=self.run_worker([self.base(8),self.base(0)])
        self.assertEqual([c.args[1]['operation'] for c in calls],['sync','sync'])
        self.assertTrue(states[-1]['window_backfill']['stop_requested'])

    def test_window_stop_during_job_returns_to_prefix_work(self):
        operations=[]
        def run(directory,payload,**kwargs):
            operations.append(payload['operation'])
            if len(operations)==1:return self.base(8)
            if payload['operation']=='window_sync':
                (self.directory/'windows/STOP').touch()
                self.assertTrue(kwargs['cancelled']())
                raise InterruptedError('window stop')
            return self.base(0)
        _,states=self.run_worker(run)
        self.assertEqual(operations,['sync','window_sync','sync'])
        self.assertEqual(states[-1]['state'],'complete')

    def test_global_stop_during_window_job_pauses_whole_indexer(self):
        operations=[]
        def run(directory,payload,**kwargs):
            operations.append(payload['operation'])
            if len(operations)==1:return self.base(8)
            (self.directory/'STOP').touch()
            self.assertTrue(kwargs['cancelled']())
            raise InterruptedError('global stop')
        _,states=self.run_worker(run)
        self.assertEqual(operations,['sync','window_sync'])
        self.assertEqual(states[-1]['state'],'paused')

    def test_storage_pause_in_window_phase_stops_without_starting_another_job(self):
        calls,states=self.run_worker([self.base(8),StoragePause('reserve')])
        self.assertEqual(len(calls),2)
        self.assertEqual(states[-1]['state'],'storage_pause')

    def test_restart_resumes_pending_windows_then_checks_current_prefix_queue(self):
        self.run_worker([self.base(8), StoragePause('reserve')])
        calls, states = self.run_worker([self.windows(), self.windows(), self.base(0), self.windows(0, 0)])
        self.assertEqual([c.args[1]['operation'] for c in calls],
                         ['window_sync', 'window_sync', 'sync', 'window_sync'])
        self.assertEqual(states[0]['window_backfill']['resumed_jobs_at_start'], 2)
        self.assertEqual(states[-1]['state'], 'complete')

    def test_restart_keeps_only_the_unfinished_window_job(self):
        self.run_worker([self.base(8), self.windows(10, 3), StoragePause('reserve')])
        calls, states = self.run_worker([self.windows(0, 0), self.base(0), self.windows(0, 0)])
        self.assertEqual([c.args[1]['operation'] for c in calls], ['window_sync', 'sync', 'window_sync'])
        self.assertEqual(states[0]['window_backfill']['resumed_jobs_at_start'], 1)
        self.assertEqual(states[0]['window_backfill']['last_job']['embedded'], 3)

    def test_invalid_or_incompatible_checkpoint_starts_with_current_prefix_work(self):
        self.run_worker([self.base(8), StoragePause('reserve')])
        path = self.directory / 'BUILD_STATUS.json'
        original = json.loads(path.read_text())
        variants = [[], {'backend': 'cpu'}, {'model': 'different'},
                    {'window_backfill': {**original['window_backfill'], 'remaining_jobs': True}},
                    {'window_backfill': {**original['window_backfill'], 'remaining_jobs': 3}},
                    {'window_backfill': {**original['window_backfill'], 'phase': 'prefix'}},
                    {'window_backfill': {**original['window_backfill'], 'vectors_per_job': 64}}]
        for changes in variants:
            with self.subTest(changes=changes):
                path.write_text(json.dumps({**original, **changes} if isinstance(changes, dict) else changes))
                calls, states = self.run_worker([self.base(0), self.windows(0, 0)])
                self.assertEqual(calls[0].args[1]['operation'], 'sync')
                self.assertEqual(states[0]['window_backfill']['resumed_jobs_at_start'], 0)
        path.write_text('incomplete json')
        calls, _ = self.run_worker([self.base(0), self.windows(0, 0)])
        self.assertEqual(calls[0].args[1]['operation'], 'sync')
        path.write_text(json.dumps(original))
        calls, states = self.run_worker([self.base(0), self.windows(0, 0)], self.args + ['--window-jobs', '1'])
        self.assertEqual(calls[0].args[1]['operation'], 'sync')
        self.assertEqual(states[0]['window_backfill']['resumed_jobs_at_start'], 0)

    def test_window_stop_still_overrides_a_restored_turn(self):
        self.run_worker([self.base(8), StoragePause('reserve')])
        (self.directory / 'windows/STOP').touch()
        calls, states = self.run_worker([self.base(0)])
        self.assertEqual([c.args[1]['operation'] for c in calls], ['sync'])
        self.assertTrue(states[-1]['window_backfill']['stop_requested'])

    def test_existing_standalone_window_lock_prevents_second_owner(self):
        directory=self.directory/'windows';directory.mkdir()
        with (directory/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_worker([])
        self.assertFalse((self.root/'research.sqlite').exists())

    def test_invalid_window_configuration_rejects_before_starting_jobs(self):
        for extra in (['--window-jobs','9'],['--window-vectors','0'],['--window-passages','257'],['--backend','cpu']):
            with self.subTest(extra=extra),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                self.run_worker([],self.args+extra)
        self.assertFalse((self.root/'research.sqlite').exists())


if __name__=='__main__':
    unittest.main()
