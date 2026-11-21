"""Resume a lossless chunk-reference migration, preserving IDs and search data.

Run against an offline copy first. This releases database pages for reuse; use
SQLite VACUUM on that copy to return the released space to the filesystem.
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--max-documents', type=int)
    args = parser.parse_args()
    store = Store(args.db)
    try:
        migrate(store, batch_size=args.batch_size, max_documents=args.max_documents,
                report=lambda value: print(json.dumps(value), flush=True))
    finally:
        store.db.close()


if __name__ == '__main__':
    main()
