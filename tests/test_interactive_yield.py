import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from adams_rag.inference_guard import inference_slot, interactive_waiting
from adams_rag.semantic_index import SemanticIndex
from adams_rag.storage import StoragePause
from adams_rag.store import Store


class Encoder:
    def __init__(self):
        self.inputs = []
        self.after_encode = lambda: None

    def passage_embed(self, texts, **kwargs):
        self.inputs.extend(texts)
        self.after_encode()
        return iter([np.asarray([1., 0.], dtype=np.float32) for _ in texts])

    def query_embed(self, question):
        return iter([np.asarray([1., 0.], dtype=np.float32)])


class InteractiveYieldTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / 'research.sqlite')
        self.encoder = Encoder()
        self.index = SemanticIndex(self.store, self.root / 'index', 'test-yield', self.encoder)
        with self.store.db:
            for n in range(2):
                self.store.put({'AccessionNumber': f'BASE{n}', 'content': f'valve baseline {n}'})

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def test_queued_interactive_caller_is_detected_without_stealing_its_gate(self):
        path = self.root / 'inference.lock'
        waiting, acquired = threading.Event(), threading.Event()
        failures = []

        def caller():
            try:
                with inference_slot(path, monitor=waiting.set, wait_seconds=3):
                    acquired.set()
            except Exception as error:
                failures.append(error)

        with inference_slot(path, foreground=False):
            self.assertFalse(interactive_waiting(path))
            thread = threading.Thread(target=caller)
            thread.start()
            self.assertTrue(waiting.wait(2))
            self.assertTrue(interactive_waiting(path))
            self.assertFalse(acquired.is_set())
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(failures)
        self.assertTrue(acquired.is_set())
        self.assertFalse(interactive_waiting(path))

    def test_yield_keeps_commits_query_publishes_them_and_resume_has_no_skips(self):
        self.index.sync()
        self.encoder.inputs.clear()
        prior_manifest = self.index.manifest.read_bytes()
        with self.store.db:
            for n in range(20):
                self.store.put({'AccessionNumber': f'NEW{n:02}', 'content': f'valve new passage {n}'})
        pending = [False]
        self.encoder.after_encode = lambda: pending.__setitem__(0, True)
        result = self.index.sync(limit=100, should_yield=lambda: pending[0])
        self.assertEqual(result['embedded'], 8)
        self.assertTrue(result['yielded_for_interactive'])
        self.assertTrue(result['graph_publication_deferred'])
        self.assertEqual(self.index.status()['vectors'], 10)
        self.assertEqual(self.index.manifest.read_bytes(), prior_manifest)
        fresh = SemanticIndex(self.store, self.root / 'index', 'test-yield', Encoder())
        matches = fresh.search('valve', limit=50)
        self.assertEqual(len(matches), 10)
        self.assertEqual(len(fresh.index), 10)
        pending[0] = False
        self.encoder.after_encode = lambda: None
        resumed = self.index.sync(limit=100, should_yield=lambda: pending[0])
        self.assertEqual(resumed['embedded'], 12)
        self.assertFalse(resumed['yielded_for_interactive'])
        self.assertEqual(len(self.index.index), 22)
        self.assertEqual(len(self.encoder.inputs), 20)
        self.assertEqual(len(set(self.encoder.inputs)), 20)
        self.assertEqual(self.index.generation, self.index.fingerprint())

    def test_queue_before_first_batch_does_no_encoding_and_retains_pending_work(self):
        result = self.index.sync(should_yield=lambda: True)
        self.assertEqual(result['embedded'], 0)
        self.assertTrue(result['yielded_for_interactive'])
        self.assertEqual(self.encoder.inputs, [])
        self.assertFalse(self.index.path.exists())
        self.assertEqual(self.index.sync()['embedded'], 2)

    def test_cancellation_takes_precedence_over_yield(self):
        called = []
        def cancelled(*_):
            raise InterruptedError('user stop')
        with self.assertRaises(InterruptedError):
            self.index.sync(progress=cancelled, should_yield=lambda: called.append(True))
        self.assertFalse(called)
        self.assertEqual(self.index.status()['vectors'], 0)

    def test_storage_stop_takes_precedence_over_yield(self):
        called = []
        with patch('adams_rag.semantic_index.require_space', side_effect=StoragePause('reserve')):
            with self.assertRaises(StoragePause):
                self.index.sync(should_yield=lambda: called.append(True))
        self.assertFalse(called)
        self.assertEqual(self.index.status()['vectors'], 0)

    def test_last_batch_completion_publishes_normally(self):
        pending = [False]
        self.encoder.after_encode = lambda: pending.__setitem__(0, True)
        result = self.index.sync(should_yield=lambda: pending[0])
        self.assertEqual(result['embedded'], 2)
        self.assertFalse(result['yielded_for_interactive'])
        self.assertEqual(len(self.index.index), 2)
        self.assertEqual(json.loads(self.index.manifest.read_text())['generation'], self.index.fingerprint())

    def test_index_worker_does_not_treat_zero_work_yield_as_completion(self):
        from adams_rag.index_worker import main
        responses = [{'embedded': 0, 'yielded_for_interactive': True},
                     {'embedded': 2, 'yielded_for_interactive': False},
                     {'embedded': 0, 'yielded_for_interactive': False}]
        output = io.StringIO()
        with patch('adams_rag.index_worker.run_operation', side_effect=responses) as operation, \
                patch('adams_rag.index_worker.signal.signal'), contextlib.redirect_stdout(output):
            main(['--db', str(self.root / 'research.sqlite'), '--directory', str(self.root / 'index'), '--backend', 'mlx'])
        self.assertEqual(operation.call_count, 3)
        states = [json.loads(line)['state'] for line in output.getvalue().splitlines()]
        self.assertIn('waiting_for_model', states)
        self.assertEqual(states[-1], 'complete')


if __name__ == '__main__':
    unittest.main()
