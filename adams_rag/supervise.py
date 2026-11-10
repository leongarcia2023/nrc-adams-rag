"""Run one catalog worker; recover from a small number of process-level exits."""
import argparse
import datetime
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path


TERMINAL_STATES = {"finished", "needs_storage", "needs_review", "stopped"}


def should_restart(returncode, state, stop_requested, consecutive_restarts):
    # Application validation errors require review. Only external termination
    # (negative subprocess return code) is eligible for an automatic restart.
    return (returncode < 0 and not stop_requested and state not in TERMINAL_STATES
            and consecutive_restarts < 3)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--directory", required=True)
    args, _ = parser.parse_known_args(argv)
    directory = Path(args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "supervisor.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A supervisor already holds this catalog lock")
    restarts = 0
    while not (directory / "STOP").exists():
        # A separately launched worker should never be duplicated or replaced.
        with (directory / "worker.lock").open("a+") as worker_lock:
            try:
                fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("A catalog worker is already running")
        started = time.monotonic()
        child = subprocess.Popen([sys.executable, "-u", "-m", "adams_rag.catalog", *argv], stdin=subprocess.DEVNULL)
        def report(state, **details):
            result = {"updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "state": state, "worker_pid": child.pid, "consecutive_restarts": restarts, **details}
            temp = directory / "supervisor.json.tmp"
            temp.write_text(json.dumps(result, indent=2) + "\n")
            temp.replace(directory / "supervisor.json")
        report("running")
        code = child.wait()
        if time.monotonic() - started > 600:
            restarts = 0
        try:
            state = json.loads((directory / "status.json").read_text()).get("worker_state")
        except (OSError, ValueError):
            state = None
        if not should_restart(code, state, (directory / "STOP").exists(), restarts):
            report("stopped", exit_code=code, catalog_state=state)
            return
        restarts += 1
        report("restart_wait", exit_code=code, retry_seconds=60)
        for _ in range(60):
            if (directory / "STOP").exists():
                report("stopped", exit_code=code, catalog_state="stopped")
                return
            time.sleep(1)


if __name__ == "__main__":
    main()
