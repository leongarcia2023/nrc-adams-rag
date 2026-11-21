"""Run one model call in a disposable process with a monitored storage reserve."""
import json
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import selectors
import subprocess
import time

from .model_diagnostics import ModelWorkerFailure


class InferenceBusy(RuntimeError):
    pass


def interactive_waiting(path):
    """Called by a background GPU holder to detect a caller holding the priority gate."""
    gate = Path(path).with_name('interactive.lock')
    with gate.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False  # Closing this descriptor releases the probe lock.


@contextmanager
def inference_slot(path, *, foreground=True, monitor=lambda: None, wait_seconds=60):
    """A queued interactive call takes priority after the current background job."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_name('interactive.lock').open('a') as gate, path.open('a') as lock:
        # interactive_waiting() briefly probes this gate. A foreground caller
        # must not mistake that millisecond probe for another active question.
        gate_deadline = time.monotonic() + (.1 if foreground else 0)
        while True:
            try:
                fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= gate_deadline:
                    raise InferenceBusy('Another local model call is active; interactive work has priority') from None
                monitor()
                time.sleep(.005)
        deadline=time.monotonic()+wait_seconds
        while True:
            monitor()
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not foreground or time.monotonic()>=deadline:
                    raise InferenceBusy('Another local model call is active; retry after it finishes') from None
                time.sleep(min(.25,max(0,deadline-time.monotonic())))
        if not foreground:
            # An interactive caller can now queue, blocking the next background job.
            fcntl.flock(gate,fcntl.LOCK_UN)
        yield


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def run_guarded(command, payload, *, preflight, monitor, timeout=300, cancelled=None):
    if not 0 < timeout <= 1800:
        raise ValueError("Model timeout must be positive and at most 30 minutes")
    data = json.dumps(payload, ensure_ascii=False).encode()
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Model input exceeds the bounded context allowance")
    preflight()
    if cancelled and cancelled():
        raise InterruptedError("Local model call cancelled")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)
    started = time.monotonic()
    try:
        # Resume both directions explicitly. On some Python versions retrying
        # communicate(input=None) after a timeout never resumes a partial stdin
        # write, leaving a slow-starting child waiting forever for the final bytes.
        output = bytearray()
        sent = 0
        with selectors.DefaultSelector() as selector:
            for stream, event in ((process.stdin, selectors.EVENT_WRITE), (process.stdout, selectors.EVENT_READ)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, event)
            while selector.get_map() or process.poll() is None:
                if cancelled and cancelled():
                    raise InterruptedError("Local model call cancelled")
                monitor()
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise RuntimeError("Local model call timed out; retrieved evidence is retained")
                for key, _ in selector.select(min(.25, remaining)):
                    if key.fileobj is process.stdin:
                        try:
                            sent += os.write(key.fd, data[sent:sent + 4096])
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            sent = len(data)
                        if sent == len(data):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                    else:
                        try:
                            block = os.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if block:
                            output.extend(block)
                            if len(output) > 1024 * 1024:
                                raise ValueError("Local model response exceeds the output allowance")
                        else:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
        monitor()
        if process.returncode != 0:
            raise RuntimeError("Local model process ended before returning a result; retrieved evidence is retained")
        if len(output) > 1024 * 1024:
            raise ValueError("Local model response exceeds the output allowance")
        try:
            response = json.loads(output)
        except (ValueError, UnicodeDecodeError):
            raise ValueError("Local model returned an invalid response") from None
        if isinstance(response, dict) and response.get('ok') is False:
            raise ModelWorkerFailure(response.get('diagnostics'))
        if not isinstance(response, dict) or response.get("ok") is not True or not isinstance(response.get("result"), dict):
            raise RuntimeError("Local model did not produce a valid result; retrieved evidence is retained")
        return response["result"]
    finally:
        stop_process(process)
        for stream in (process.stdin, process.stdout):
            if stream:
                stream.close()
