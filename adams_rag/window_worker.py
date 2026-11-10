"""Prepare complete embedding windows with resumable batches; does not enable search."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time

from .embedding_windows import MODEL, POLICY, WindowPlanner
from .mlx_embeddings import EmbeddingBusy, run_operation
from .storage import StoragePause, require_space, reserve_gb
from .store import Store
from .window_backfill import WindowBackfill


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='data/research.sqlite')
    parser.add_argument('--directory', default='data/index')
    parser.add_argument('--state-directory')
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--vector-limit', type=int, default=128)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--reuse-only', action='store_true',
                        help='Plan saved prefix inputs without loading a model; missing window vectors stay pending')
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 256 or not 1 <= args.vector_limit <= 128 or not 1 <= args.jobs <= 10000:
        parser.error('Use 1–256 passages, 1–128 vectors and 1–10,000 jobs')
    directory = Path(args.directory).resolve()
    state_directory = Path(args.state_directory).resolve() if args.state_directory else directory / 'windows'
    require_space(directory)
    state_directory.mkdir(parents=True, exist_ok=True)
    lock = (state_directory / 'worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit('A window worker is already running here')
    stopping = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    store, backfill = None, None
    started = time.monotonic()
    last_report = [0]

    def cancelled():
        return stopping[0] or (directory / 'STOP').exists() or (state_directory / 'STOP').exists()

    def report(state, detail=None, result=None):
        status = {'state': state, 'pid': os.getpid(), 'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'model': MODEL, 'policy': POLICY, 'db': str(Path(args.db).resolve()),
                  'elapsed_seconds': round(time.monotonic() - started),
                  'free_bytes': shutil.disk_usage(directory).free, 'reserve_bytes': int(reserve_gb() * 1e9),
                  'detail': detail, 'last_job': result, 'role': 'window_preparation',
                  'mode': 'reuse_only' if args.reuse_only else 'embedding',
                  'full_archive_verified': False, **(backfill.status() if backfill else {})}
        temporary = state_directory / 'STATUS.json.tmp'
        temporary.write_text(json.dumps(status, indent=2))
        os.replace(temporary, state_directory / 'STATUS.json')
        print(json.dumps(status), flush=True)
        last_report[0] = time.monotonic()

    def progress():
        if cancelled():
            raise InterruptedError('Window indexing stopped; committed vectors retained')
        require_space(Path(args.db).resolve().parent)
        if time.monotonic() - last_report[0] > 30:
            report('running')

    def wait(seconds):
        for _ in range(seconds):
            if cancelled():
                return
            require_space(Path(args.db).resolve().parent)
            time.sleep(1)

    try:
        # A stop request is checked before creating any schema in the database.
        progress()
        store = Store(args.db)
        backfill = WindowBackfill(store, WindowPlanner(directory))
        report('starting')
        completed_jobs = 0
        while not cancelled():
            progress()
            try:
                if args.reuse_only:
                    result = backfill.prepare_existing(args.limit, progress=progress)
                else:
                    result = run_operation(directory, {'operation': 'window_sync', 'db': str(store.path.resolve()),
                        'state_directory': str(state_directory), 'limit': args.limit, 'vector_limit': args.vector_limit},
                        cancelled=cancelled, monitor=progress)
            except EmbeddingBusy as error:
                report('waiting_for_model', str(error))
                wait(10)
                continue
            report('running', result=result)
            if result.get('yielded_for_interactive'):
                continue
            completed_jobs += 1
            if not result['selected_passages']:
                if args.follow:
                    report('waiting_for_prefix_vectors' if args.reuse_only else 'waiting_for_passages', result=result)
                    wait(30)
                    continue
                detail = ('No unplanned saved prefix inputs; missing window vectors may still remain'
                          if args.reuse_only else 'No pending indexed passages at this observation')
                report('caught_up', detail, result)
                return
            if not args.follow and completed_jobs >= args.jobs:
                report('job_limit', 'Committed progress retained; more passages may remain', result)
                return
        report('paused', 'Committed progress retained')
    except InterruptedError as error:
        report('paused', str(error))
    except StoragePause as error:
        report('storage_pause', str(error))
    except Exception as error:
        report('failed', type(error).__name__)
        raise
    finally:
        if store:
            store.db.close()
        lock.close()


if __name__ == '__main__':
    main()
