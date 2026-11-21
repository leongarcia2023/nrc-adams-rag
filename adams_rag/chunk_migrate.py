"""Resume a lossless chunk-reference migration, preserving IDs and search data.

Run against an offline copy first. This releases database pages for reuse; use
SQLite VACUUM on that copy to return the released space to the filesystem.
"""
import argparse
import itertools
import json
import time

from .chunk_storage import reference, source_id
from .evidence import SourceBindings
from .store import Store
from .storage import require_space


def convert_document(store, accession):
    value = SourceBindings(store.db).document(accession)
    retained, body = value['retained'], value['body']
    rows = store.db.execute('SELECT id,ordinal,page,CAST(text AS BLOB) AS stored_text,typeof(text) AS kind '
                            'FROM chunks WHERE accession=? ORDER BY ordinal', (accession,)).fetchall()
    if not rows:
        raise ValueError('indexed_document_has_no_chunks')
    contexts, offsets, previous = {}, {}, {}
    offset = 0
    for page, context in (enumerate(value['pages'], 1) if value['pages'] is not None else [(None, body)]):
        contexts[page], offsets[page] = context, offset
        offset += len(context) + 1
    planned = []
    for ordinal, row in enumerate(rows, 1):
        page = row['page']
        if row['ordinal'] != ordinal or page not in contexts:
            raise ValueError('chunk_position_does_not_match_retained_version')
        if row['kind'] == 'text':
            text = row['stored_text'].decode()
        else:
            text = store.db.execute('SELECT text FROM chunks WHERE id=?', (row['id'],)).fetchone()[0]
        start = contexts[page].find(text, previous.get(page, 0))
        if start < 0:
            start = contexts[page].find(text)
        if start < 0 or not text.strip():
            raise ValueError('chunk_text_does_not_match_retained_version')
        previous[page] = start + 1
        start += offsets[page]
        end = start + len(text)
        if body[start:end] != text:
            raise ValueError('chunk_position_does_not_match_retained_version')
        planned.append((ordinal, page, start, end, text))
    source = source_id(store.db, accession, retained['version'])
    changed, removed, added = 0, 0, 0
    for row, (_, _, start, end, text) in zip(rows, planned):
        if row['kind'] != 'text':
            continue
        pointer = reference(source, start, end, text)
        store.db.execute('UPDATE chunks SET text=? WHERE id=?', (pointer, row['id']))
        changed += 1; removed += len(row['stored_text']); added += len(pointer)
    # Verify through the same reader the app, embeddings and citations will use.
    actual = store.db.execute('SELECT id,text FROM chunks WHERE accession=? ORDER BY ordinal', (accession,))
    for row, planned_row in zip(actual, planned):
        if row['text'] != planned_row[4]:
            raise ValueError('chunk_reference_round_trip_failed')
    return changed, removed, added


def migrate(store, *, batch_size=100, max_documents=None, report=None):
    if not 1 <= batch_size <= 1000:
        raise ValueError('Use a batch size between 1 and 1000')
    with store.db:
        store.db.execute('''CREATE TABLE IF NOT EXISTS chunk_migration(
            id INTEGER PRIMARY KEY CHECK(id=1),last_accession TEXT NOT NULL,
            documents INTEGER NOT NULL,chunks INTEGER NOT NULL,old_bytes INTEGER NOT NULL,
            reference_bytes INTEGER NOT NULL,holdouts INTEGER NOT NULL,state TEXT NOT NULL)''')
        store.db.execute("INSERT OR IGNORE INTO chunk_migration VALUES(1,'',0,0,0,0,0,'pending')")
        store.db.execute('''CREATE TABLE IF NOT EXISTS chunk_migration_holdouts(
            accession TEXT PRIMARY KEY,reason TEXT NOT NULL)''')
        store.db.execute('UPDATE chunk_storage_policy SET enabled=1 WHERE id=1')
    start, done = time.monotonic(), 0
    while max_documents is None or done < max_documents:
        require_space(store.path.parent)
        state = dict(store.db.execute('SELECT * FROM chunk_migration WHERE id=1').fetchone())
        limit = batch_size if max_documents is None else min(batch_size, max_documents - done)
        # Seek along the accession key. The status index otherwise makes SQLite
        # scan and sort the entire indexed collection again for every batch.
        accessions = [r[0] for r in store.db.execute("SELECT accession FROM documents INDEXED BY sqlite_autoindex_documents_1 WHERE accession>? AND status='indexed' ORDER BY accession LIMIT ?", (state['last_accession'], limit))]
        if not accessions:
            with store.db:
                store.db.execute("UPDATE chunk_migration SET state='complete' WHERE id=1")
            break
        with store.db:
            # The connection context does not itself start a transaction.
            # Begin before DDL/savepoints so releasing a document savepoint
            # cannot commit it ahead of the batch cursor or trigger restore.
            store.db.execute('BEGIN IMMEDIATE')
            # This is a physical representation change, not new embedding input.
            # Suppress only this invalidation trigger, in the same transaction.
            trigger = store.db.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name='window_source_chunk_update_v1'").fetchone()
            if trigger:
                store.db.execute('DROP TRIGGER window_source_chunk_update_v1')
            for accession in accessions:
                store.db.execute('SAVEPOINT compact_document')
                try:
                    changed, old_bytes, new_bytes = convert_document(store, accession)
                except (ValueError, UnicodeError, OSError) as error:
                    store.db.execute('ROLLBACK TO compact_document')
                    store.db.execute('INSERT OR REPLACE INTO chunk_migration_holdouts VALUES(?,?)', (accession, str(error)))
                    state['holdouts'] += 1
                else:
                    state['chunks'] += changed; state['old_bytes'] += old_bytes; state['reference_bytes'] += new_bytes
                finally:
                    store.db.execute('RELEASE compact_document')
                state['documents'] += 1; state['last_accession'] = accession; done += 1
            if trigger:
                store.db.execute(trigger[0])
            store.db.execute("UPDATE chunk_migration SET last_accession=?,documents=?,chunks=?,old_bytes=?,reference_bytes=?,holdouts=?,state='running' WHERE id=1",
                             tuple(state[k] for k in ['last_accession','documents','chunks','old_bytes','reference_bytes','holdouts']))
        store.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        if report:
            report({**state, 'state':'running', 'elapsed_seconds':round(time.monotonic()-start, 1)})
    state = dict(store.db.execute('SELECT * FROM chunk_migration WHERE id=1').fetchone())
    state['elapsed_seconds'] = round(time.monotonic()-start, 1)
    if report:
        report(state)
    return state


def repack_chunks(store):
    """Pack the small references into new pages, releasing old pages for reuse.

    This transactional table replacement needs space for the compact chunks,
    rather than a second copy of the entire database. Other services must be
    stopped. It does not shrink the database file; the freed pages can hold
    future documents, metadata, indexes or vectors without growing that file.
    """
    db = store.db
    if db.in_transaction:
        raise ValueError('Finish the caller transaction before repacking')
    state = db.execute('SELECT state FROM chunk_migration WHERE id=1').fetchone()
    if not state or state[0] != 'complete':
        raise ValueError('Complete the chunk conversion before repacking')
    count, payload = db.execute('SELECT count(*),coalesce(sum(length(CAST(text AS BLOB))),0) FROM chunks').fetchone()
    # Include indexes, table overhead, and WAL writes, with bounded cache use.
    require_space(store.path.parent, extra_bytes=64 * 1024**2 + (payload + count * 96) * 2)
    schema = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone()[0]
    dependent = [r[0] for r in db.execute("SELECT sql FROM sqlite_master WHERE tbl_name='chunks' AND type IN ('index','trigger') AND sql IS NOT NULL")]
    sequence = db.execute("SELECT seq FROM sqlite_sequence WHERE name='chunks'").fetchone()
    page_size = db.execute('PRAGMA page_size').fetchone()[0]
    before = db.execute('PRAGMA freelist_count').fetchone()[0] * page_size
    foreign_keys = db.execute('PRAGMA foreign_keys').fetchone()[0]
    legacy = db.execute('PRAGMA legacy_alter_table').fetchone()[0]
    secure = db.execute('PRAGMA secure_delete').fetchone()[0]
    try:
        db.execute('PRAGMA foreign_keys=OFF')
        db.execute('PRAGMA legacy_alter_table=ON')
        # The old pages remain internal free space. Avoid journaling gigabytes
        # solely to erase duplicated public text; original sources are retained.
        db.execute('PRAGMA secure_delete=OFF')
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE TABLE chunks_repack_stage ' + schema[schema.index('('):])
        db.execute('INSERT INTO chunks_repack_stage(id,accession,ordinal,page,text) SELECT id,accession,ordinal,page,text FROM chunks ORDER BY id')
        columns = 'id,accession,ordinal,page,typeof(text),CAST(text AS BLOB) AS stored_text'
        left = db.execute('SELECT '+columns+' FROM chunks ORDER BY id')
        right = db.execute('SELECT '+columns+' FROM chunks_repack_stage ORDER BY id')
        for old, new in itertools.zip_longest(left, right):
            if old is None or new is None or tuple(old) != tuple(new):
                raise ValueError('Repacked chunk rows differ')
        db.execute('ALTER TABLE chunks RENAME TO chunks_repack_old')
        db.execute('ALTER TABLE chunks_repack_stage RENAME TO chunks')
        db.execute('DROP TABLE chunks_repack_old')
        for statement in dependent:
            db.execute(statement)
        if sequence:
            db.execute("UPDATE sqlite_sequence SET seq=max(seq,?) WHERE name='chunks'", (sequence[0],))
        if db.execute('PRAGMA foreign_key_check').fetchone() is not None:
            raise ValueError('Repacking changed a foreign-key relationship')
        if [tuple(r) for r in db.execute("PRAGMA integrity_check('chunks')")] != [('ok',)]:
            raise ValueError('Repacked chunk table failed integrity checks')
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.execute(f'PRAGMA foreign_keys={foreign_keys}')
        db.execute(f'PRAGMA legacy_alter_table={legacy}')
        db.execute(f'PRAGMA secure_delete={secure}')
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    after = db.execute('PRAGMA freelist_count').fetchone()[0] * page_size
    return {'state':'repacked','chunks':count,'reusable_bytes_before':before,
            'reusable_bytes_after':after,'database_file_bytes':store.path.stat().st_size,
            'filesystem_shrunk':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--max-documents', type=int)
    parser.add_argument('--repack', action='store_true', help='Repack completed references into reusable database space; stop other services first')
    args = parser.parse_args()
    store = Store(args.db)
    try:
        if args.repack:
            print(json.dumps(repack_chunks(store)), flush=True)
        else:
            migrate(store, batch_size=args.batch_size, max_documents=args.max_documents,
                    report=lambda value: print(json.dumps(value), flush=True))
    finally:
        store.db.close()


if __name__ == '__main__':
    main()
