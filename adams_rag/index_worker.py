"""Resumable local embedding worker with a stop file, disk reserve and progress report."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time

from .semantic_index import SemanticIndex
from .store import Store
from .storage import StoragePause, require_space, reserve_gb
from .mlx_embeddings import MODEL as GPU_MODEL, EmbeddingBusy, run_operation
from .semantic_index import DEFAULT_EMBEDDING, active_model
from .inference_guard import inference_slot, interactive_waiting, InferenceBusy


def pending_window_turn(directory, model, args):
    """Resume scheduling only; actual pending passages always come from the DB."""
    if args.backend != 'mlx' or not args.window_jobs:
        return 0, None
    try:
        path = directory / 'BUILD_STATUS.json'
        if path.stat().st_size > 64 * 1024:
            return 0, None
        previous = json.loads(path.read_text())
        if not isinstance(previous, dict) or previous.get('backend') != 'mlx' or previous.get('model') != model:
            return 0, None
        state = previous.get('window_backfill')
        if not isinstance(state, dict) or state.get('phase') != 'windows':
            return 0, None
        settings = {'jobs_per_prefix_job': args.window_jobs, 'passages_per_job': args.window_passages,
                    'vectors_per_job': args.window_vectors}
        if any(type(state.get(key)) is not int or state[key] != value for key, value in settings.items()):
            return 0, None
        remaining = state.get('remaining_jobs')
        if type(remaining) is not int or not 1 <= remaining <= args.window_jobs:
            return 0, None
        last = state.get('last_job')
        return remaining, last if isinstance(last, dict) else None
    except (OSError, ValueError):
        return 0, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/research.sqlite")
    parser.add_argument("--directory", default="data/index")
    parser.add_argument("--follow", action="store_true", help="Wait for newly ingested passages after catching up")
    parser.add_argument("--backend", choices=("cpu", "mlx"), default="cpu")
    parser.add_argument("--window-jobs", type=int, default=0, help="Run 0–8 bounded window jobs after each prefix job")
    parser.add_argument("--window-passages", type=int, default=256)
    parser.add_argument("--window-vectors", type=int, default=128)
    args = parser.parse_args(argv)
    if (not 0 <= args.window_jobs <= 8 or not 1 <= args.window_passages <= 256
            or not 1 <= args.window_vectors <= 128 or (args.window_jobs and args.backend != 'mlx')):
        parser.error('Window backfill requires the MLX backend, 0–8 jobs, 1–256 passages and 1–128 vectors')
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "worker.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit("An indexing worker is already running")
    window_directory, window_lock = directory / 'windows', None
    if args.window_jobs:
        try:
            require_space(directory)
            window_directory.mkdir(parents=True, exist_ok=True)
            window_lock = (window_directory / 'worker.lock').open('a')
            fcntl.flock(window_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            if window_lock:
                window_lock.close()
            lock.close()
            raise
    stopping = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    store = Store(args.db)
    index = SemanticIndex(store, directory, model=GPU_MODEL if args.backend == "mlx" else DEFAULT_EMBEDDING)
    started = time.monotonic()
    initial = index.status()["vectors"]
    last_update = [0]
    window_remaining, last_window = pending_window_turn(directory, index.model, args)
    resumed_window_jobs = window_remaining
    # Even after restoring a window turn, check the current prefix queue before
    # declaring this local pass idle. The status file is not coverage evidence.
    prefix_idle, window_progress = False, False
    def report(state, detail=None):
        vectors = index.status()["vectors"]
        total = store.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        status = {"state": state, "pid": os.getpid(), "vectors": vectors, "chunks": total,
                  "model": index.model, "backend": args.backend,
                  "reserve_bytes": int(reserve_gb() * 1e9), "free_bytes": shutil.disk_usage(directory).free,
                  "embedded_this_run": vectors - initial,
                  "elapsed_seconds": round(time.monotonic() - started),
                  "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "detail": detail}
        if args.window_jobs:
            status['window_backfill'] = {'jobs_per_prefix_job': args.window_jobs,
                'phase': 'windows' if window_remaining else 'prefix', 'remaining_jobs': window_remaining,
                'passages_per_job': args.window_passages, 'vectors_per_job': args.window_vectors,
                'last_job': last_window, 'stop_requested': (window_directory / 'STOP').exists(),
                'resumed_jobs_at_start': resumed_window_jobs}
        active = active_model(directory)
        if active != index.model:
            status["active_profile_coverage"] = index.coverage_against(active)
        temporary = directory / "BUILD_STATUS.json.tmp"
        temporary.write_text(json.dumps(status, indent=2))
        os.replace(temporary, directory / "BUILD_STATUS.json")
        print(json.dumps(status), flush=True)
    def progress(done, total):
        if time.monotonic() - last_update[0] > 30:
            report("running")
            last_update[0] = time.monotonic()
        if stopping[0] or (directory / "STOP").exists():
            raise InterruptedError("Indexing paused; committed vectors are retained")
        require_space(directory)
    def wait(seconds):
        for _ in range(seconds):
            if stopping[0] or (directory / 'STOP').exists():
                return
            require_space(directory)
            time.sleep(1)
    try:
        report("starting")
        while not stopping[0] and not (directory / "STOP").exists():
            progress(0, 0)
            is_window = window_remaining > 0
            if args.backend == "mlx":
                try:
                    active = active_model(directory)
                    if is_window and (window_directory / 'STOP').exists():
                        result = {'selected_passages': 0, 'embedded': 0, 'window_stop_requested': True}
                    else:
                        payload = ({'operation': 'window_sync', 'db': str(store.path.resolve()),
                            'state_directory': str(window_directory.resolve()), 'limit': args.window_passages,
                            'vector_limit': args.window_vectors} if is_window else
                            {"operation": "sync", "db": str(store.path.resolve()), "limit": 1000,
                             "priority_model": active if active != index.model else None})
                        result = run_operation(directory, payload,
                            cancelled=lambda: stopping[0] or (directory / "STOP").exists()
                                or (is_window and (window_directory / 'STOP').exists()),
                            monitor=lambda: progress(0, 0))
                except EmbeddingBusy as error:
                    report("waiting_for_model", str(error))
                    wait(10)
                    continue
                except InterruptedError:
                    if is_window and (window_directory / 'STOP').exists() and not stopping[0] and not (directory / 'STOP').exists():
                        result = {'selected_passages': 0, 'embedded': 0, 'window_stop_requested': True}
                    else:
                        raise
            else:
                inference_lock = store.path.parent / 'models' / 'inference.lock'
                try:
                    with inference_slot(inference_lock, foreground=False, monitor=lambda: progress(0, 0)):
                        result = index.sync(1000, progress=progress,
                            should_yield=lambda: interactive_waiting(inference_lock))
                except InferenceBusy as error:
                    report("waiting_for_model", str(error))
                    wait(10)
                    continue
            if is_window:
                last_window = result
                if result.get('yielded_for_interactive'):
                    report('waiting_for_model', 'Window backfill yielded after committed batches')
                    continue  # Retry this phase; a queued query does not consume its work allowance.
                window_progress = window_progress or result['selected_passages'] > 0
                window_remaining -= 1
                if not result['selected_passages']:
                    window_remaining = 0
                report('running', 'Window job finished; committed vectors retained')
                if window_remaining or not prefix_idle or window_progress:
                    continue
                # Both queues were idle in this cycle. New sources are picked up
                # by follow mode; this is not a full-archive completion claim.
            else:
                prefix_idle = not result['embedded']
                window_progress = False
                if args.window_jobs and (result['embedded'] or not result.get('yielded_for_interactive')):
                    window_remaining = args.window_jobs
                    continue
            if result.get("yielded_for_interactive"):
                report("waiting_for_model", "Yielded between committed batches for an interactive request")
                continue  # The priority gate supplies the ordinary busy/retry path.
            if prefix_idle:
                if args.follow:
                    report("waiting_for_text")
                    wait(30)
                    continue
                report("complete")
                return
        report("paused")
    except InterruptedError as error:
        report("paused", str(error))
    except StoragePause as error:
        report("storage_pause", str(error))
    except Exception as error:
        report("failed", type(error).__name__)
        raise
    finally:
        store.db.close()
        if window_lock:
            window_lock.close()
        lock.close()


if __name__ == "__main__":
    main()
