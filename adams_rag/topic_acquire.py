"""Resumable priority acquisition alongside the full catalog backfill.

Search pages are discovery only. A separate Get Document response, identifier
check, and existing extraction gate are required before text becomes evidence.
Search exhaustion is an observation, not proof of complete archive coverage.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3

from .client import AdamsClient, AdamsHTTPError, BudgetStop, unpack_search
from .config import load_api_key
from .identity import is_legacy
from .ingest import archive
from .markdown_archive import prepare_fetched
from .query_terms import contains_subject, MILESTONE
from .research import ACCESSION
from .storage import require_space, reserve_gb, StoragePause
from .store import Store
from .transport import PersistentOpener


def acquire(store, client, directory, query, title=None, *, max_documents=1000,
            max_pages=30, min_free_gb=None, cancelled=lambda: False):
    if not query.strip() or len(query) > 500 or not 1 <= max_pages <= 100 or not 1 <= max_documents <= 10000:
        raise ValueError('Use a query and bounded page/document allowances')
    directory = Path(directory)
    require_space(store.path.parent, min_free_gb)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = sqlite3.connect(directory / 'queue.sqlite')
        db.row_factory = sqlite3.Row
        try:
            db.executescript('''CREATE TABLE IF NOT EXISTS job(
                id INTEGER PRIMARY KEY CHECK(id=1), query TEXT, title TEXT,
                offset INTEGER DEFAULT 0, state TEXT DEFAULT 'discovering',
                last_page TEXT, reported_count INTEGER);
                CREATE TABLE IF NOT EXISTS candidates(accession TEXT PRIMARY KEY,
                title TEXT, document_date TEXT, priority INTEGER, outcome TEXT DEFAULT 'pending');''')
            with db:
                db.execute('INSERT OR IGNORE INTO job(id,query,title) VALUES(1,?,?)', (query, title))
            job = db.execute('SELECT * FROM job').fetchone()
            if (job['query'], job['title']) != (query, title):
                raise ValueError('This priority queue belongs to a different query')
            counts = {'downloaded_this_run': 0, 'already_present_this_run': 0}
            def report(state, detail=None):
                job = dict(db.execute('SELECT * FROM job').fetchone())
                result = dict(state=state, query=query, title_filter=title, **counts,
                    outcomes=dict(db.execute('SELECT outcome,count(*) FROM candidates GROUP BY outcome')),
                    discovery_state=job['state'], examined_search_records=job['offset'],
                    reported_matches=job['reported_count'], full_archive_verified=False,
                    updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), detail=detail)
                temporary = directory / 'STATUS.json.tmp'
                temporary.write_text(json.dumps(result, indent=2) + '\n')
                os.replace(temporary, directory / 'STATUS.json')
                return result
            report('discovering')
            try:
                for _ in range(max_pages):
                    job = db.execute('SELECT * FROM job').fetchone()
                    if job['state'] != 'discovering':
                        break
                    if cancelled():
                        return report('stopped')
                    require_space(store.path.parent, min_free_gb)
                    response = client.search(q=query, skip=job['offset'], library='main')
                    docs = unpack_search(response)
                    ids = [d['AccessionNumber'] for d in docs]
                    signature = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                    if docs and (signature == job['last_page'] or len(ids) != len(set(ids))):
                        return report('discovery_blocked', 'Repeated page or duplicate identifiers; cursor preserved')
                    with db:
                        for document in docs:
                            accession = document['AccessionNumber']
                            name = document.get('DocumentTitle') or ''
                            if is_legacy(document) or not ACCESSION.fullmatch(accession):
                                continue
                            if title and not contains_subject(name, [title]):
                                continue
                            priority = min(4, len(MILESTONE.findall(name)))
                            db.execute('INSERT OR IGNORE INTO candidates(accession,title,document_date,priority) VALUES(?,?,?,?)',
                                (accession, name, document.get('DocumentDate'), priority))
                        db.execute('UPDATE job SET offset=offset+?,last_page=?,reported_count=?,state=? WHERE id=1',
                            (len(docs), signature, response.get('count'),
                             'discovering' if docs else 'search_exhausted_unverified'))
                    report('discovering')
                for _ in range(max_documents):
                    if cancelled():
                        return report('stopped')
                    row = db.execute("SELECT * FROM candidates WHERE outcome='pending' ORDER BY priority DESC,document_date DESC,accession LIMIT 1").fetchone()
                    if row is None:
                        return report('complete_observed_queue' if db.execute('SELECT state FROM job').fetchone()[0] != 'discovering' else 'discovery_limit')
                    require_space(store.path.parent, min_free_gb)
                    accession = row['accession']
                    existing = store.db.execute('SELECT status FROM documents WHERE accession=?', (accession,)).fetchone()
                    if existing and existing[0] == 'indexed':
                        outcome = 'already_indexed'
                        prepare_fetched(store, [accession])
                        counts['already_present_this_run'] += 1
                    else:
                        try:
                            response = client.request(accession=accession)
                        except AdamsHTTPError as error:
                            if error.status not in (404, 410):
                                raise
                            outcome = 'unavailable_' + str(error.status)
                        else:
                            document = response.get('document') if isinstance(response, dict) else None
                            if (not isinstance(document, dict) or is_legacy(document)
                                    or str(document.get('AccessionNumber', '')).upper() != accession.upper()):
                                raise ValueError('Unexpected document identity; queue item remains pending')
                            require_space(store.path.parent, min_free_gb,
                                extra_bytes=max(32 * 1024 * 1024, len(json.dumps(response).encode()) * 12))
                            archive(store, response)
                            with store.db:
                                outcome = store.put(document, 'api_document_unverified')
                                store.db.execute('INSERT INTO retrieval_events(accession,outcome) VALUES(?,?)', (accession, outcome))
                            if outcome == 'indexed':
                                prepare_fetched(store, [accession])
                                counts['downloaded_this_run'] += 1
                    with db:
                        db.execute('UPDATE candidates SET outcome=? WHERE accession=?', (outcome, accession))
                    report('acquiring')
                return report('document_limit')
            except StoragePause as error:
                return report('storage_pause', str(error))
            except BudgetStop as error:
                return report('budget_pause', str(error))
            except Exception as error:
                report('error', str(error))
                raise
        finally:
            db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--rate-db', required=True)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--title', help='Only prioritize records containing this phrase in their title')
    parser.add_argument('--max-documents', type=int, default=1000)
    parser.add_argument('--max-pages', type=int, default=30)
    parser.add_argument('--min-free-gb', type=float)
    parser.add_argument('--rps', type=float, default=2)
    parser.add_argument('--daily-limit', type=int, default=90000)
    args = parser.parse_args(argv)
    if not 0 < args.rps <= 10:
        parser.error('Use a request rate above zero and at most 10')
    args.min_free_gb = reserve_gb(args.min_free_gb)
    # prepare_fetched follows the same reserve as the acquisition worker.
    os.environ['ADAMS_MIN_FREE_GB'] = str(args.min_free_gb)
    stopping = [False]
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopping.__setitem__(0, True))
    store, rate, transport = Store(args.db), Store(args.rate_db), PersistentOpener()
    try:
        client = AdamsClient(rate, load_api_key(), max_requests=args.max_pages + args.max_documents * 6 + 10,
            rps=args.rps, daily_limit=args.daily_limit, opener=transport, include_search_content=False)
        result = acquire(store, client, args.directory, args.query, args.title,
            max_documents=args.max_documents, max_pages=args.max_pages, min_free_gb=args.min_free_gb,
            cancelled=lambda: stopping[0] or (Path(args.directory) / 'STOP').exists())
        print(json.dumps(result), flush=True)
    finally:
        transport.close(); store.db.close(); rate.db.close()


if __name__ == '__main__':
    main()
