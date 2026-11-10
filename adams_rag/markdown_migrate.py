"""Make verified Markdown canonical and remove exactly covered raw API caches."""
import argparse
import base64
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import zipfile

from .identity import is_legacy, source_key
from .markdown_archive import decode_markdown, render_markdown
from .page_boundaries import retained_pdf_pages
from .source_content import MARKER, offload, read_text
from .storage import StoragePause, require_space
from .store import Store, digest


def historical_check(store, row):
    """Preserve old citation versions separately from current retrieval eligibility."""
    body = read_text(row['content'], store.db)
    metadata = json.loads(row['metadata'])
    current = dict(store.db.execute('SELECT * FROM documents WHERE accession=?', (row['accession'],)).fetchone())
    current.update(metadata=row['metadata'], text_hash=row['text_hash'], provenance=row['provenance'],
        title=metadata.get('DocumentTitle') or current['title'])
    pages = retained_pdf_pages(body, metadata) if row['provenance'] == 'pdf' else None
    report = dict(state='holdout', reasons=['historical_version'], source_fidelity_verified=False)
    markdown = render_markdown(current, body, pages, row, report)
    if decode_markdown(markdown)[1] != body:
        raise ValueError('Historical source round trip failed')
    directory = store.path.parent / 'markdown'
    require_space(directory, extra_bytes=max(32 * 1024**2, len(markdown.encode()) * 2))
    path = directory / ('history-' + uuid.uuid4().hex + '.zip')
    temporary = path.with_suffix('.partial')
    member = 'history/' + row['version'] + '.md'
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, markdown)
    with zipfile.ZipFile(temporary) as archive:
        if archive.read(member).decode() != markdown:
            raise ValueError('Stored historical Markdown differs')
    with temporary.open('rb') as file:
        os.fsync(file.fileno())
    os.replace(temporary, path)
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(archive=str(path.resolve()), member=member, markdown_sha256=digest(markdown))


def migrate_versions(store, progress):
    cursor = 0
    while True:
        rows = store.db.execute('SELECT rowid,* FROM document_versions WHERE rowid>? ORDER BY rowid LIMIT 100', (cursor,)).fetchall()
        if not rows:
            break
        require_space(store.path.parent)
        with store.db:
            for row in rows:
                cursor = row['rowid']
                if row['content'].startswith(MARKER):
                    continue
                check = store.db.execute('SELECT * FROM markdown_checks WHERE accession=? AND text_hash=?',
                    (row['accession'], row['text_hash'])).fetchone()
                if check and json.loads(check['report']).get('retained_version') != row['version']:
                    check = None
                if check is None:
                    check = historical_check(store, row)
                    progress['historical_versions_exported'] += 1
                progress['redundant_database_payload_bytes'] += offload(store.db, row, check)
                progress['versions_migrated'] += 1
        store.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        if cursor % 2000 < 100:
            print(json.dumps(progress), flush=True)
    with store.db:
        store.db.execute('CREATE TABLE IF NOT EXISTS markdown_storage(id INTEGER PRIMARY KEY CHECK(id=1), enabled INTEGER NOT NULL)')
        store.db.execute('INSERT OR REPLACE INTO markdown_storage VALUES(1,1)')


def covered_raw(store, path):
    """Only individual API packets with losslessly retained text AND metadata."""
    if path.is_symlink() or path.stat().st_size > 128 * 1024**2:
        return None, 'unsafe_or_oversized'
    with gzip.open(path, 'rb') as file:
        data = file.read(128 * 1024**2 + 1)
    if len(data) > 128 * 1024**2:
        return None, 'oversized'
    raw = data.decode()
    if digest(raw) + '.json.gz' != path.name:
        return None, 'raw_checksum_mismatch'
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get('document'), dict) or set(payload) - {'document','highlights','score','semanticSearch'}:
        return None, 'unique_or_unrecognized_packet'
    doc = payload['document']
    acc = source_key(doc) if is_legacy(doc) else doc.get('AccessionNumber')
    check = store.db.execute('SELECT state FROM markdown_checks WHERE accession=?', (acc,)).fetchone()
    if not check or check['state'] != 'ready':
        return None, 'held_out_or_not_exported'
    body = doc.get('content', doc.get('text', '')) or ''
    if not isinstance(body, str) or not body.strip():
        return None, 'no_full_text'
    if doc.get('contentEncoding') == 'base64':
        body = base64.b64decode(body, validate=True).decode()
    for row in store.db.execute('SELECT * FROM document_versions WHERE accession=? AND text_hash=?', (acc,digest(body))):
        if not row['content'].startswith(MARKER):
            continue
        metadata = json.loads(row['metadata'])
        if any(k not in metadata or metadata[k] != v for k,v in doc.items() if k not in {'content','text'}):
            continue
        if read_text(row['content'], store.db) != body:
            continue
        return dict(accession=acc, version=row['version'], raw_sha256=digest(raw)), None
    return None, 'text_or_metadata_not_covered'


def prune_raw(store, progress, audit):
    skipped = Counter()
    # Append durable receipts BEFORE unlink; an interrupted prepared record is
    # harmless and the existing path can be verified again on the next run.
    with (audit / 'raw-removal-receipts.jsonl').open('a') as log:
        batch = []
        def remove_batch():
            for path, stat, receipt in batch:
                log.write(json.dumps(dict(path=str(path), bytes=stat.st_size, links=stat.st_nlink,
                    state='verified_removal_prepared', **receipt)) + '\n')
            log.flush(); os.fsync(log.fileno())
            for path, stat, receipt in batch:
                current = path.stat()
                if (current.st_ino, current.st_mtime_ns, current.st_size) != (stat.st_ino, stat.st_mtime_ns, stat.st_size):
                    raise ValueError('Raw cache changed during cleanup')
                path.unlink()
                progress['raw_files_removed'] += 1
                progress['raw_bytes_removed'] += stat.st_size
            batch.clear()
        n = 0
        for n, path in enumerate((store.path.parent / 'raw').rglob('*.json.gz'), 1):
            if n % 100 == 1:
                require_space(store.path.parent, extra_bytes=1024**2)
            try:
                stat = path.stat()
                receipt, reason = covered_raw(store, path)
            except (ValueError, OSError, KeyError):
                receipt, reason = None, 'invalid_or_unreadable_packet'
            if receipt:
                batch.append((path, stat, receipt))
            else:
                skipped[reason] += 1
            if len(batch) >= 100:
                remove_batch()
            if n % 2000 == 0:
                progress['raw_files_scanned'] = n
                print(json.dumps(progress), flush=True)
        remove_batch()
        progress['raw_files_scanned'] = n
    progress['raw_skipped'] = dict(skipped)


def run(db_path):
    store = Store(db_path)
    audit = store.path.parent / 'audit' / 'canonical-migration'
    audit.mkdir(parents=True, exist_ok=True)
    progress = dict(state='running', started=time.time(), versions_migrated=0, historical_versions_exported=0,
        redundant_database_payload_bytes=0, raw_files_removed=0, raw_bytes_removed=0,
        free_before=shutil.disk_usage(store.path.parent).free)
    try:
        migrate_versions(store, progress)
        prune_raw(store, progress, audit)
        progress['state'] = 'complete'
    except StoragePause as error:
        progress.update(state='storage_pause', error=str(error))
    except BaseException as error:
        progress.update(state='failed', error=str(error))
        raise
    finally:
        store.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        progress['sqlite_reusable_bytes'] = store.db.execute('PRAGMA freelist_count').fetchone()[0] * store.db.execute('PRAGMA page_size').fetchone()[0]
        progress['free_after'] = shutil.disk_usage(store.path.parent).free
        (audit / 'status.json').write_text(json.dumps(progress, indent=2) + '\n')
        store.db.close()
        print(json.dumps(progress), flush=True)
    return progress


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='data/research.sqlite')
    run(Path(parser.parse_args().db))
