import json
import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from adams_rag.inference_guard import run_guarded,inference_slot,InferenceBusy
from adams_rag.models import MLXEngine
from adams_rag.storage import StoragePause


class InferenceGuardTests(unittest.TestCase):
    def test_real_child_returns_data_then_exits(self):
        command = [sys.executable, "-c", "import json,sys; d=json.load(sys.stdin); print(json.dumps({'ok':True,'result':{'value':d['value']}}))"]
        result = run_guarded(command, {"value": "source data"}, preflight=lambda: None, monitor=lambda: None)
        self.assertEqual(result, {"value": "source data"})

    def test_slow_reader_receives_large_input_across_monitor_intervals(self):
        command=[sys.executable,'-c',"import json,sys,time; time.sleep(.6); d=json.load(sys.stdin); print(json.dumps({'ok':True,'result':{'length':len(d['text'])}}))"]
        checks=[]
        result=run_guarded(command,{'text':'x'*1500000},preflight=lambda:None,
                           monitor=lambda:checks.append(True),timeout=5)
        self.assertEqual(result,{'length':1500000})
        self.assertGreater(len(checks),3)

    def test_output_allowance_stops_a_chatty_child_while_it_is_running(self):
        command=[sys.executable,'-c',"import sys,time; sys.stdout.write('x'*1100000); sys.stdout.flush(); time.sleep(30)"]
        with self.assertRaisesRegex(ValueError,'output allowance'):
            run_guarded(command,{},preflight=lambda:None,monitor=lambda:None,timeout=3)

    def test_failed_preflight_never_starts_process(self):
        with patch("adams_rag.inference_guard.subprocess.Popen") as start:
            with self.assertRaises(StoragePause):
                run_guarded(["unused"], {}, preflight=lambda: self.low(), monitor=lambda: None)
            start.assert_not_called()

    @staticmethod
    def low():
        raise StoragePause("reserve reached")

    def test_space_drop_terminates_and_reaps_only_the_child(self):
        original = subprocess.Popen
        children = []
        def start(*args, **kwargs):
            child = original(*args, **kwargs); children.append(child); return child
        calls = [0]
        def monitor():
            calls[0] += 1
            if calls[0] >= 2:
                self.low()
        with patch("adams_rag.inference_guard.subprocess.Popen", side_effect=start):
            with self.assertRaises(StoragePause):
                run_guarded([sys.executable, "-c", "import sys,time; sys.stdin.read(); time.sleep(30)"], {},
                            preflight=lambda: None, monitor=monitor)
        self.assertIsNotNone(children[0].poll())
        self.assertTrue(children[0].stdin.closed)
        self.assertTrue(children[0].stdout.closed)

    def test_timeout_and_cancellation_end_child_without_result(self):
        command = [sys.executable, "-c", "import sys,time; sys.stdin.read(); time.sleep(30)"]
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            run_guarded(command, {}, preflight=lambda: None, monitor=lambda: None, timeout=.1)
        checks = iter([False, False, True])
        with self.assertRaises(InterruptedError):
            run_guarded(command, {}, preflight=lambda: None, monitor=lambda: None,
                        cancelled=lambda: next(checks, True))

    def test_failed_or_malformed_worker_output_never_becomes_an_answer(self):
        for code in ["print('not json')", "print('{\"ok\":false}')", "raise SystemExit(2)"]:
            with self.subTest(code=code), self.assertRaises((RuntimeError, ValueError)):
                run_guarded([sys.executable, "-c", code], {}, preflight=lambda: None, monitor=lambda: None)

    def test_model_headroom_checks_include_system_temporary_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"; model.mkdir()
            (model / "weights.safetensors").write_bytes(b"1234")
            (model / "resource-manifest.json").write_text(json.dumps({"files": [{"path": "weights.safetensors", "bytes": 4}]}))
            engine = MLXEngine(model, storage_path=tmp)
            with patch("adams_rag.models.require_space") as check:
                engine.check_space(loading=True)
            self.assertEqual({str(c.args[0]) for c in check.call_args_list},
                             {str(Path(tmp).resolve()), str(Path(tempfile.gettempdir()).resolve())})
            self.assertTrue(all(c.kwargs["extra_bytes"] == 4 + 1024**3 for c in check.call_args_list))
            with patch("adams_rag.models.shutil.disk_usage") as disk:
                disk.return_value.free = 4_000_000_000
                required = engine.availability()["required_free_bytes"]
                disk.return_value.free = required - 1
                self.assertEqual(engine.availability()["state"], "headroom_needed")
                disk.return_value.free = required
                self.assertEqual(engine.availability()["state"], "ready")

    def test_real_worker_rejects_invalid_supervisor_without_loading_gpu_model(self):
        result = subprocess.run([sys.executable, "-m", "adams_rag.model_worker"], input=json.dumps({"parent_pid": -1}),
                                text=True, capture_output=True, timeout=5)
        response = json.loads(result.stdout)
        self.assertIs(response['ok'], False)
        self.assertEqual(response['diagnostics']['stage'], 'input')
        self.assertIsInstance(response['diagnostics']['elapsed_ms'], int)
        self.assertEqual(result.stderr, "")

    def test_parent_watch_ends_orphan_worker_without_gpu_initialization(self):
        result = subprocess.run([sys.executable, "-c", "from adams_rag.model_worker import watch_parent; watch_parent(-1)"], timeout=5)
        self.assertEqual(result.returncode, 70)

    def test_separate_model_instances_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"; model.mkdir()
            (model / "weights.safetensors").write_bytes(b"1234")
            (model / "resource-manifest.json").write_text(json.dumps({"files": [{"path": "weights.safetensors", "bytes": 4}]}))
            with inference_slot(Path(tmp)/'inference.lock'):
                with patch("adams_rag.inference_guard.subprocess.Popen") as start:
                    with self.assertRaisesRegex(RuntimeError, "Another local model"):
                        MLXEngine(model).complete("test", {})
                    start.assert_not_called()

    def test_interactive_request_gets_next_slot_ahead_of_background_indexing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'inference.lock'
            waiting,acquired,finished=threading.Event(),threading.Event(),threading.Event()
            errors=[]
            def foreground():
                try:
                    with inference_slot(path,monitor=waiting.set,wait_seconds=3):
                        acquired.set()
                        finished.wait(3)
                except Exception as error:
                    errors.append(error)
            with inference_slot(path,foreground=False):
                thread=threading.Thread(target=foreground);thread.start()
                self.assertTrue(waiting.wait(2))
                with self.assertRaises(InferenceBusy):
                    with inference_slot(path,foreground=False):
                        self.fail('Background work overtook the queued query')
                self.assertFalse(acquired.is_set())
            self.assertTrue(acquired.wait(2))
            finished.set();thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertFalse(errors)

    def test_storage_pause_while_waiting_releases_interactive_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'inference.lock'
            with inference_slot(path,foreground=False):
                with self.assertRaises(StoragePause):
                    with inference_slot(path,monitor=self.low):
                        self.fail('Low storage should stop queued inference')
            with inference_slot(path,foreground=False):
                pass


if __name__ == "__main__":
    unittest.main()
