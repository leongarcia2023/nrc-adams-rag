"""Restart a workspace web process after crashes without renting resources.

This supervisor survives a closed terminal when launched with nohup, but cannot
keep the workspace itself alive or restart it after a platform shutdown.
"""
import argparse
from pathlib import Path
import signal
import subprocess
import time
import os


def supervise(command, stop_file, *, max_restarts=5, delay=5, stable_seconds=600):
    stop_file = Path(stop_file)
    stopping = False
    previous = {}

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    def terminate(child):
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=15)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, request_stop)
    restarts = 0
    child = None
    try:
        while not stopping and not stop_file.exists():
            started = time.monotonic()
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
            while child.poll() is None:
                if stopping or stop_file.exists():
                    terminate(child)
                    return 0
                time.sleep(.2)
            if child.returncode == 0:
                return 0
            if time.monotonic() - started >= stable_seconds:
                restarts = 0
            if restarts >= max_restarts:
                print('Web restart limit reached; inspect the service log before resuming.', flush=True)
                return 1
            restarts += 1
            print(f'Web process exited ({child.returncode}); restart {restarts}/{max_restarts}.', flush=True)
            deadline = time.monotonic() + min(delay * 2 ** (restarts - 1), 60)
            while time.monotonic() < deadline and not stopping and not stop_file.exists():
                time.sleep(.2)
        return 0
    finally:
        if child is not None:
            terminate(child)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop-file', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Supply the web process command after --')
    raise SystemExit(supervise(command, args.stop_file))


if __name__ == '__main__':
    main()
