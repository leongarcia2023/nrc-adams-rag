"""Resumable, lossless Markdown exports and a version-bound retrieval holdout gate.

Automated screening is not visual OCR verification. Only complete, recorded
page reviews qualify a PDF for removal; uncertain sources stay retained.
"""
from .source_content import read_text
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import signal
import time
import uuid
import zipfile
import zlib
from urllib.parse import urlparse

from .extraction_review import for_version
from .identity import is_legacy, record_id
from .page_boundaries import retained_pdf_pages
from .storage import StoragePause, require_space
from .store import Store, chunk_text, digest, source_url
from .text_quality import assess

POLICY = 1


def setup(db):
    db.executescript("""
    CREATE TABLE IF NOT EXISTS markdown_checks(
      accession TEXT PRIMARY KEY, text_hash TEXT, metadata_hash TEXT NOT NULL,
      policy INTEGER NOT NULL, state TEXT NOT NULL, report TEXT NOT NULL,
      archive TEXT NOT NULL, member TEXT NOT NULL, markdown_sha256 TEXT NOT NULL,
      checked TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
    CREATE INDEX IF NOT EXISTS markdown_checks_by_state ON markdown_checks(state);
    CREATE INDEX IF NOT EXISTS markdown_verified_sources ON markdown_checks(accession)
      WHERE json_extract(report,'$.source_fidelity_verified')=1;
    CREATE TABLE IF NOT EXISTS markdown_policy(id INTEGER PRIMARY KEY CHECK(id=1), enabled INTEGER NOT NULL);
    INSERT OR IGNORE INTO markdown_policy VALUES(1,0);
    CREATE TABLE IF NOT EXISTS markdown_pruned_pdf(
      path TEXT PRIMARY KEY, accession TEXT NOT NULL, source_sha256 TEXT NOT NULL,
      bytes INTEGER NOT NULL, state TEXT NOT NULL, updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
    CREATE TRIGGER IF NOT EXISTS markdown_checks_insert AFTER INSERT ON markdown_checks
    BEGIN UPDATE retrieval_state SET generation=generation+1 WHERE id=1; END;
    CREATE TRIGGER IF NOT EXISTS markdown_checks_update AFTER UPDATE ON markdown_checks
    BEGIN UPDATE retrieval_state SET generation=generation+1 WHERE id=1; END;
    CREATE TRIGGER IF NOT EXISTS markdown_checks_delete AFTER DELETE ON markdown_checks
    BEGIN UPDATE retrieval_state SET generation=generation+1 WHERE id=1; END;
    CREATE TRIGGER IF NOT EXISTS markdown_policy_update AFTER UPDATE ON markdown_policy
    BEGIN UPDATE retrieval_state SET generation=generation+1 WHERE id=1; END;
    CREATE TRIGGER IF NOT EXISTS markdown_document_changed AFTER UPDATE ON documents
    WHEN old.text_hash IS NOT new.text_hash OR old.metadata<>new.metadata OR old.provenance<>new.provenance OR old.status<>new.status
    BEGIN DELETE FROM markdown_checks WHERE accession=new.accession; END;
    CREATE TRIGGER IF NOT EXISTS markdown_document_deleted AFTER DELETE ON documents
    BEGIN DELETE FROM markdown_checks WHERE accession=old.accession; END;
    CREATE TRIGGER IF NOT EXISTS markdown_review_insert AFTER INSERT ON extraction_reviews
    BEGIN DELETE FROM markdown_checks WHERE accession=new.accession; END;
    CREATE TRIGGER IF NOT EXISTS markdown_review_update AFTER UPDATE ON extraction_reviews
    BEGIN DELETE FROM markdown_checks WHERE accession=new.accession; END;
    CREATE TRIGGER IF NOT EXISTS markdown_review_delete AFTER DELETE ON extraction_reviews
    BEGIN DELETE FROM markdown_checks WHERE accession=old.accession; END;
    """)


def gate_enabled(db):
    return bool(db.execute("SELECT 1 FROM sqlite_master WHERE name='markdown_policy'").fetchone()
                and db.execute('SELECT enabled FROM markdown_policy WHERE id=1').fetchone()[0])


def filter_rows(db, rows, exclusions=None):
    if not rows or not db.execute("SELECT 1 FROM sqlite_master WHERE name='markdown_checks'").fetchone():
        return rows
    strict = gate_enabled(db)
    blocked = set()
    seen = {(r['accession'], r['text_hash'], r['page']) for r in (exclusions or [])}
    for start in range(0, len(rows), 300):
        ids = [r['id'] for r in rows[start:start + 300]]
        found = db.execute("""SELECT c.id,d.accession,d.text_hash,q.state,q.report,q.policy
            FROM chunks c JOIN documents d USING(accession)
            LEFT JOIN markdown_checks q ON q.accession=d.accession AND q.text_hash IS d.text_hash
            WHERE c.id IN (""" + ','.join('?' for _ in ids) + ')', ids)
        for r in found:
            pending = r['state'] is None or r['policy'] != POLICY
            if (pending and not strict) or (not pending and r['state'] == 'ready'):
                continue
            blocked.add(r['id'])
            identity = (r['accession'], r['text_hash'], 0)
            if exclusions is not None and identity not in seen:
                seen.add(identity)
                report = json.loads(r['report']) if not pending else {}
                exclusions.append(dict(accession=r['accession'], text_hash=r['text_hash'], page=0,
                    disposition='exclude', reasons=report.get('reasons') or ['markdown_check_pending'],
                    note='Held out of retrieval pending extraction review or Markdown validation.'))
    return [r for r in rows if r['id'] not in blocked]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def reviewed_pages_match(db, accession, pages, methods, pdf_hash):
    if not pages or len(methods) != len(pages) or not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='extraction_corrections'").fetchone():
        return False
    manifests = {}
    for n, method in enumerate(methods, 1):
        key = method.get('correction_id')
        if method.get('page') != n or method.get('method') != 'reviewed_transcription' or method.get('source_pdf_sha256') != pdf_hash or not key:
            return False
        if key not in manifests:
            row = db.execute('SELECT manifest FROM extraction_corrections WHERE id=? AND accession=? AND pdf_sha256=?',
                             (key, accession, pdf_hash)).fetchone()
            if not row or digest(row[0]) != key:
                return False
            manifests[key] = json.loads(row[0])
        reviewed = next((p for p in manifests[key]['pages'] if p['page'] == n), None)
        if reviewed is None or reviewed['text'] != pages[n - 1]:
            return False
    return True


def inspect_document(store, row, pdf_paths):
    """Check exact stored text, every chunk, page boundaries and source diagnostics."""
    metadata = json.loads(row['metadata'])
    report = dict(policy=POLICY, reasons=[], integrity_verified=False,
                  source_fidelity_verified=False, automated_screen=True,
                  verification_scope='Retained text, chunk consistency and automated extraction diagnostics')
    reviews = for_version(store.db, row['accession'], row['text_hash'])
    report['reviews_sha256'] = digest(json.dumps(reviews, sort_keys=True))
    reasons = report['reasons']
    body, pages, retained = '', None, None
    if row['status'] != 'indexed':
        reasons.append('no_full_text_available')
    else:
        retained = store.db.execute("""SELECT * FROM document_versions WHERE accession=? AND text_hash=?
            AND provenance=? ORDER BY captured DESC,rowid DESC LIMIT 1""",
            (row['accession'], row['text_hash'], row['provenance'])).fetchone()
        try:
            if not retained:
                raise ValueError('retained_version_missing')
            if digest(retained['text_hash'] + retained['metadata']) != retained['version']:
                raise ValueError('retained_version_checksum_mismatch')
            body = read_text(retained['content'], store.db)
            if digest(body) != row['text_hash']:
                raise ValueError('retained_text_checksum_mismatch')
            original = json.loads(retained['metadata'])
            if row['provenance'] == 'pdf':
                pages = retained_pdf_pages(body, original)
                if original.get('LocalFileSHA256') != metadata.get('LocalFileSHA256'):
                    raise ValueError('pdf_identity_mismatch')
            expected = ((n, text) for n, content in (enumerate(pages, 1) if pages is not None else [(None, body)])
                        for text in chunk_text(content))
            actual = store.db.execute('SELECT page,text,ordinal FROM chunks WHERE accession=? ORDER BY ordinal', (row['accession'],))
            count = 0
            for number, pair in enumerate(itertools.zip_longest(expected, actual), 1):
                want, got = pair
                if want is None or got is None or (got['page'], got['text']) != want or got['ordinal'] != number:
                    raise ValueError('chunk_coverage_mismatch')
                count += 1
            report.update(integrity_verified=True, checked_chunks=count, retained_version=retained['version'])
        except (ValueError, UnicodeError, zlib.error) as e:
            reasons.append(str(e) if isinstance(e, ValueError) else 'retained_text_unreadable')
        quality = assess(body, metadata, row['provenance'], row['status'], pages, text_available=bool(retained))
        report['quality'] = quality
        reasons.extend(quality['flags'])
        if reviews:
            reasons.append('reviewed_extraction_issue')
            report['page_reviews'] = reviews
        # Reject obvious broken encodings without mistaking punctuation-heavy
        # tables or dotted contents pages alone for OCR corruption.
        if '\x00' in body:
            reasons.append('nul_characters')
        if re.search(r'\(cid:\d+\)', body):
            reasons.append('unmapped_pdf_characters')
        if row['provenance'] == 'pdf':
            source = pdf_paths.get(row['accession'])
            expected_hash = metadata.get('LocalFileSHA256')
            if source and file_hash(source) == expected_hash:
                report['source_pdf'] = str(source.resolve())
                report['source_pdf_sha256'] = expected_hash
                report['source_pdf_hash_verified'] = True
            else:
                removed = store.db.execute("SELECT 1 FROM markdown_pruned_pdf WHERE accession=? AND source_sha256=? AND state='removed_after_verified_export'",
                                           (row['accession'], expected_hash)).fetchone()
                if removed and source is None:
                    report['source_pdf_previously_verified'] = True
                    report['source_pdf_retention'] = 'removed_after_verified_export'
                    report['source_pdf_sha256'] = expected_hash
                else:
                    reasons.append('source_pdf_missing_or_changed')
            methods = metadata.get('PageExtraction', [])
            completely_reviewed = reviewed_pages_match(store.db, row['accession'], pages, methods, expected_hash)
            if completely_reviewed and (report.get('source_pdf_hash_verified') or report.get('source_pdf_previously_verified')) and report['integrity_verified'] and not reasons:
                report['source_fidelity_verified'] = True
                report['verification_scope'] += '; every physical page has a recorded source comparison'
            report['physical_pages'] = len(pages) if pages is not None else None
    report['reasons'] = sorted(set(reasons))
    report['state'] = 'holdout' if reasons else 'ready'
    report['hard_to_scan'] = bool(set(reasons) & {'machine_ocr', 'replacement_characters', 'sparse_text_for_page_count',
        'blank_pdf_pages', 'reviewed_extraction_issue', 'nul_characters', 'unmapped_pdf_characters'})
    return body, pages, dict(retained) if retained else None, report


def render_markdown(row, body, pages, retained, report):
    """Exact source blocks are reversible, including whitespace and literal fences."""
    m = json.loads(row['metadata'])
    public_id = m.get('SourceAccession') or m.get('AccessionNumber') or row['accession']
    library = 'supplemental' if row['provenance'] == 'community_regulatory_text' else ('legacy' if is_legacy(m) else 'main')
    url = source_url(m.get('Url') or row['url'], public_id)
    header = dict(format='nrc-markdown-1', accession=public_id, source_key=row['accession'],
        title=row['title'], source_url=url, library=library,
        nrc_record_id=record_id({**m, 'AccessionNumber': public_id}) if library != 'supplemental' else None,
        document_date=row['document_date'], provenance=row['provenance'], dataset=m.get('Dataset'),
        source_text_sha256=row['text_hash'], exported_text_sha256=digest(body),
        source_pdf_sha256=m.get('LocalFileSHA256'),
        retained_version=retained['version'] if retained else None,
        captured_utc=retained['captured'] if retained else None,
        review=report, page_mapping='physical_pdf_pages' if pages is not None else 'unavailable',
        content_lengths=[len(p) for p in pages] if pages is not None else [len(body)])
    sections = pages if pages is not None else [body]
    fence = '~' * max(3, 1 + max((len(x) for x in re.findall(r'~+', body)), default=0))
    header['source_fence'] = fence
    # JSON is valid YAML, so this is a portable Markdown front matter block.
    result = '---\n' + json.dumps(header, ensure_ascii=False, separators=(',', ':')) + '\n---\n\n'
    title = re.sub(r'([\\`*_{}\[\]<>#])', r'\\\1', ' '.join(row['title'].split()))
    result += '# ' + title + '\n\n'
    result += '**Source:** <' + url.replace('<', '%3C').replace('>', '%3E') + '>\n\n'
    result += '**Accession:** ' + str(public_id) + '  \n**Status:** ' + report['state'] + '\n\n'
    if report['reasons']:
        result += '**Held out:** ' + ', '.join(report['reasons']) + '\n\n'
    for n, text in enumerate(sections, 1):
        result += ('## PDF page ' + str(n) if pages is not None else '## Source text') + '\n\n'
        result += fence + 'text\n' + text + '\n' + fence + '\n\n'
    return result


def decode_markdown(markdown):
    """Round-trip check independent of rendering, using explicit character spans."""
    first, metadata, marker, remainder = markdown.split('\n', 3)
    if first != '---' or marker != '---':
        raise ValueError('markdown_header_invalid')
    header = json.loads(metadata)
    fence = header['source_fence']
    values, position = [], 0
    for length in header['content_lengths']:
        start = remainder.index(fence + 'text\n', position) + len(fence) + 5
        end = start + length
        if remainder[end:end + len(fence) + 2] != '\n' + fence + '\n':
            raise ValueError('markdown_source_boundary_invalid')
        values.append(remainder[start:end]); position = end + len(fence) + 2
    body = ('\f' if header['page_mapping'] == 'physical_pdf_pages' else '').join(values)
    if digest(body) != header['exported_text_sha256']:
        raise ValueError('markdown_source_checksum_mismatch')
    return header, body


def member_name(row, report):
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', row['accession'])[:100]
    return report['state'] + '/' + safe + '-' + digest(row['accession'])[:12] + '.md'


def export_batch(store, rows, directory, pdf_paths, cancelled=lambda: False):
    require_space(directory)
    name = 'documents-' + uuid.uuid4().hex + '.zip'
    temporary, destination = directory / (name + '.partial'), directory / name
    results = []
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for row in rows:
                if cancelled():
                    break
                body, pages, retained, report = inspect_document(store, row, pdf_paths)
                markdown = render_markdown(row, body, pages, retained, report)
                require_space(directory, extra_bytes=max(32 * 1024**2, len(markdown.encode()) * 2))
                header, recovered = decode_markdown(markdown)
                if recovered != body:
                    raise ValueError('Markdown round trip changed source text')
                member = member_name(row, report)
                archive.writestr(member, markdown.encode())
                results.append((row, report, member, digest(markdown)))
        if not results:
            temporary.unlink(); return 0
        # Reopen the physical ZIP, verify every CRC, source hash, and Markdown hash
        # before publishing any manifest row or changing retrieval eligibility.
        with zipfile.ZipFile(temporary) as archive:
            for row, report, member, checksum in results:
                content = archive.read(member).decode()
                decode_markdown(content)
                if digest(content) != checksum:
                    raise ValueError('Stored Markdown checksum mismatch')
        with temporary.open('rb') as f:
            os.fsync(f.fileno())
        os.replace(temporary, destination)
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        require_space(directory)
        with store.db:
            for row, report, member, checksum in results:
                current = store.db.execute('SELECT text_hash,metadata,provenance,status FROM documents WHERE accession=?', (row['accession'],)).fetchone()
                if not current or any(current[k] != row[k] for k in ('text_hash', 'metadata', 'provenance', 'status')):
                    continue
                if digest(json.dumps(for_version(store.db, row['accession'], row['text_hash']), sort_keys=True)) != report['reviews_sha256']:
                    continue
                store.db.execute('''INSERT INTO markdown_checks(accession,text_hash,metadata_hash,policy,state,report,archive,member,markdown_sha256)
                    VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(accession) DO UPDATE SET text_hash=excluded.text_hash,
                    metadata_hash=excluded.metadata_hash,policy=excluded.policy,state=excluded.state,report=excluded.report,
                    archive=excluded.archive,member=excluded.member,markdown_sha256=excluded.markdown_sha256,checked=CURRENT_TIMESTAMP''',
                    (row['accession'], row['text_hash'], digest(row['metadata']), POLICY, report['state'],
                     json.dumps(report, separators=(',', ':')), str(destination.resolve()), member, checksum))
                # Once migration is enabled, a durable, verified Markdown member
                # replaces the duplicate full-text BLOB for each new version.
                if report['integrity_verified'] and store.db.execute("SELECT 1 FROM sqlite_master WHERE name='markdown_storage'").fetchone():
                    if store.db.execute('SELECT enabled FROM markdown_storage WHERE id=1').fetchone()[0]:
                        from .source_content import offload
                        version = store.db.execute('SELECT * FROM document_versions WHERE accession=? AND version=?',
                            (row['accession'], report['retained_version'])).fetchone()
                        offload(store.db, version, dict(archive=str(destination.resolve()), member=member, markdown_sha256=checksum))
        return len(results)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def saved_markdown(store, accession):
    if not store.db.execute("SELECT 1 FROM sqlite_master WHERE name='markdown_checks'").fetchone():
        return None
    row = store.db.execute('SELECT * FROM markdown_checks WHERE accession=? AND policy=?', (accession, POLICY)).fetchone()
    if not row:
        return None
    with zipfile.ZipFile(row['archive']) as archive:
        text = archive.read(row['member']).decode()
    header, _ = decode_markdown(text)
    if digest(text) != row['markdown_sha256'] or header['source_key'] != accession:
        raise ValueError('Saved Markdown integrity check failed')
    return text


def prune_reviewed_pdf(store, accession):
    """Remove one managed PDF only after rechecking every recorded page review.

    Call explicitly after export. No uncertain or merely machine-screened PDF
    qualifies. The durable removal ledger survives interruption after unlink.
    """
    require_space(store.path.parent)
    with store.db:
        store.db.execute('BEGIN IMMEDIATE')
        row = store.db.execute('SELECT rowid,* FROM documents WHERE accession=?', (accession,)).fetchone()
        if not row or row['provenance'] != 'pdf':
            raise ValueError('Only a managed, fully reviewed PDF can be removed')
        source = store.path.parent / 'recovery' / (accession + '.pdf')
        if not re.fullmatch(r'ML[A-Z0-9]+', accession) or source.is_symlink() or not source.is_file():
            raise ValueError('The managed PDF is unavailable or unsafe to remove')
        url = urlparse(row['url'])
        if url.scheme != 'https' or url.hostname not in {'www.nrc.gov', 'nrc.gov'} or not url.path.endswith('/' + accession + '.pdf'):
            raise ValueError('A matching official NRC download URL is required')
        markdown = saved_markdown(store, accession)
        if markdown is None:
            raise ValueError('A verified Markdown export is required')
        header, retained_body = decode_markdown(markdown)
        body, pages, retained, report = inspect_document(store, row, {accession: source})
        if (report['state'] != 'ready' or not report['source_fidelity_verified'] or body != retained_body
                or header['source_text_sha256'] != row['text_hash'] or header['retained_version'] != retained['version']):
            raise ValueError('Full-source review and lossless Markdown verification are required')
        size, path = source.stat().st_size, str(source.absolute())
        store.db.execute('''INSERT INTO markdown_pruned_pdf(path,accession,source_sha256,bytes,state)
            VALUES(?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET state=excluded.state,updated=CURRENT_TIMESTAMP''',
            (path, accession, report['source_pdf_sha256'], size, 'removal_prepared'))
    # Recheck the file immediately before unlink; source checks never authorize
    # deleting a replacement file at the same path.
    with store.db:
        store.db.execute('BEGIN IMMEDIATE')
        current = store.db.execute('SELECT text_hash,metadata FROM documents WHERE accession=?', (accession,)).fetchone()
        check = store.db.execute('SELECT state FROM markdown_checks WHERE accession=?', (accession,)).fetchone()
        if not check or check['state'] != 'ready' or current['text_hash'] != row['text_hash'] or current['metadata'] != row['metadata']:
            raise ValueError('The source changed before removal')
        if source.is_symlink() or file_hash(source) != report['source_pdf_sha256']:
            raise ValueError('The original PDF changed before removal')
        source.unlink()
        store.db.execute("UPDATE markdown_pruned_pdf SET state='removed_after_verified_export',updated=CURRENT_TIMESTAMP WHERE path=?", (path,))
    return dict(accession=accession, removed_pdf_bytes=size, text_sha256=row['text_hash'],
                source_pdf_sha256=report['source_pdf_sha256'], markdown_retained=True)


def write_indexes(store, directory):
    """Small, plain CSV indexes make each source and holdout easy to locate."""
    require_space(directory, extra_bytes=64 * 1024**2)
    fields = ['accession', 'source_key', 'title', 'source_url', 'provenance', 'state', 'hard_to_scan', 'reasons',
              'source_fidelity_verified', 'archive', 'markdown_file', 'text_sha256']
    with (directory / 'sources.csv.partial').open('w', newline='') as all_f, (directory / 'holdouts.csv.partial').open('w', newline='') as bad_f:
        writers = [csv.writer(f) for f in (all_f, bad_f)]
        for w in writers:
            w.writerow(fields)
        for number, r in enumerate(store.db.execute('SELECT d.accession,d.title,d.url,d.provenance,d.metadata,q.* FROM markdown_checks q CROSS JOIN documents d ON d.accession=q.accession ORDER BY q.rowid')):
            if number % 1000 == 0:
                require_space(directory)
            report = json.loads(r['report'])
            m = json.loads(r['metadata'])
            public_id = m.get('SourceAccession') or m.get('AccessionNumber') or r['accession']
            values = [public_id, r['accession'], r['title'], r['url'], r['provenance'], r['state'], report['hard_to_scan'],
                      '; '.join(report['reasons']), report['source_fidelity_verified'], Path(r['archive']).name, r['member'], r['text_hash']]
            # CSV opened in spreadsheet applications must not execute document titles.
            values = ["'" + x if isinstance(x, str) and x.startswith(('=', '+', '-', '@')) else x for x in values]
            writers[0].writerow(values)
            if r['state'] == 'holdout':
                writers[1].writerow(values)
    for name in ('sources.csv', 'holdouts.csv'):
        os.replace(directory / (name + '.partial'), directory / name)


def run(store, directory, batch_size=500, limit=None, cancelled=lambda: False):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    require_space(directory)
    setup(store.db)
    pdf_paths = {p.stem: p for root in (store.path.parent / 'audit' / 'source-pilot', Path('examples'), store.path.parent / 'recovery')
                 if root.exists() for p in root.glob('*.pdf')}
    total, started, position = 0, time.monotonic(), 0
    state, detail = 'running', None
    def status():
        counts = dict(store.db.execute('SELECT state,count(*) FROM markdown_checks GROUP BY state'))
        value = dict(state=state, detail=detail, processed_this_run=total, counts=counts,
            updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_seconds=round(time.monotonic()-started),
            free_bytes=shutil.disk_usage(directory).free, source_fidelity_verified_documents=store.db.execute(
                "SELECT count(*) FROM markdown_checks WHERE json_extract(report,'$.source_fidelity_verified')=1").fetchone()[0],
            complete_ocr_verification=False)
        temporary = directory / 'STATUS.json.partial'; temporary.write_text(json.dumps(value, indent=2) + '\n')
        os.replace(temporary, directory / 'STATUS.json'); print(json.dumps(value), flush=True)
    try:
        while not cancelled() and (limit is None or total < limit):
            require_space(directory)
            rows = store.db.execute('''SELECT d.rowid,d.* FROM documents d LEFT JOIN markdown_checks q USING(accession)
                WHERE d.rowid>? AND (q.accession IS NULL OR q.policy<>?) ORDER BY d.rowid LIMIT ?''',
                (position, POLICY, min(batch_size, limit-total) if limit is not None else batch_size)).fetchall()
            if not rows:
                pending = store.db.execute('SELECT 1 FROM documents d LEFT JOIN markdown_checks q USING(accession) WHERE q.accession IS NULL OR q.policy<>? LIMIT 1', (POLICY,)).fetchone()
                if pending:
                    position = 0
                    continue
                state = 'complete'
                with store.db:
                    store.db.execute('UPDATE markdown_policy SET enabled=1 WHERE id=1')
                break
            total += export_batch(store, rows, directory, pdf_paths, cancelled)
            position = rows[-1]['rowid']
            status()
        if cancelled():
            state = 'stopped'
        elif state != 'complete':
            state = 'batch_limit'
        write_indexes(store, directory)
    except StoragePause as error:
        state, detail = 'storage_pause', str(error)
    except Exception as error:
        state, detail = 'failed', type(error).__name__
        raise
    finally:
        status()
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='data/research.sqlite')
    parser.add_argument('--directory', default='data/markdown')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=500)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 1000 or (args.limit is not None and args.limit < 1):
        parser.error('Use a positive limit and a batch size of 1–1000')
    directory = Path(args.directory); directory.mkdir(parents=True, exist_ok=True)
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    with (directory / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Store(args.db)
        try:
            run(store, directory, args.batch_size, args.limit, lambda: stop[0] or (directory / 'STOP').exists())
        finally:
            store.db.close()


if __name__ == '__main__':
    main()
