import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from adams_rag.service_supervisor import supervise


class ServiceSupervisorTests(unittest.TestCase):
    def test_crashes_restart_with_a_bounded_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            count = Path(directory) / 'starts'
            code = "import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.write_text(p.read_text()+'x' if p.exists() else 'x');sys.exit(1)"
            result = supervise([sys.executable, '-c', code, str(count)], Path(directory)/'STOP', max_restarts=2, delay=0)
            self.assertEqual(result, 1)
            self.assertEqual(count.read_text(), 'xxx')

    def test_clean_exit_and_existing_stop_do_not_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory)/'STOP'
            self.assertEqual(supervise([sys.executable, '-c', 'pass'], stop, delay=0), 0)
            stop.touch()
            self.assertEqual(supervise(['/does/not/exist'], stop), 0)

    def test_signal_stops_child_without_restarting(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory)/'child.pid'
            child_code = "import os,pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(60)"
            parent = subprocess.Popen([sys.executable, '-m', 'adams_rag.service_supervisor',
                '--stop-file', str(Path(directory)/'STOP'), '--', sys.executable, '-c', child_code, str(pidfile)])
            try:
                deadline = time.monotonic()+10
                while not pidfile.exists() and time.monotonic()<deadline:
                    time.sleep(.05)
                self.assertTrue(pidfile.exists())
                child_pid = int(pidfile.read_text())
                parent.terminate()
                self.assertEqual(parent.wait(timeout=20), 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if parent.poll() is None:
                    parent.terminate()
                    parent.wait(timeout=20)
