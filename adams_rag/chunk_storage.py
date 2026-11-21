"""Lossless chunk references into immutable compressed source versions.

The chunks table retains its IDs, indexes and foreign keys. Its text column can
contain either legacy text or a small binary reference. Application connections
resolve references in their row factory, so callers still receive exact text.
Raw SQLite clients must use install_reader() before consuming chunk text.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sqlite3
import struct

from .source_content import MARKER as MARKDOWN_MARKER, database_root, read_text

MARKER = b'ADAMS-CHUNK-V1\0'
SPAN = struct.Struct('!QQQ32s')


def setup(db):
    db.execute('''CREATE TABLE IF NOT EXISTS chunk_sources(
        id INTEGER PRIMARY KEY, accession TEXT NOT NULL, version TEXT NOT NULL,
        UNIQUE(accession,version),
        FOREIGN KEY(accession,version) REFERENCES document_versions(accession,version) ON DELETE RESTRICT)''')
    db.execute('''CREATE TABLE IF NOT EXISTS chunk_storage_policy(
        id INTEGER PRIMARY KEY CHECK(id=1), enabled INTEGER NOT NULL CHECK(enabled IN(0,1)))''')
    db.execute('INSERT OR IGNORE INTO chunk_storage_policy VALUES(1,0)')


def enabled(db):
    return bool(db.execute('SELECT enabled FROM chunk_storage_policy WHERE id=1').fetchone()[0])


def source_id(db, accession, version):
    db.execute('INSERT OR IGNORE INTO chunk_sources(accession,version) VALUES(?,?)', (accession, version))
    return db.execute('SELECT id FROM chunk_sources WHERE accession=? AND version=?', (accession, version)).fetchone()[0]


def reference(source, start, end, text):
    if not (type(source) is int and source > 0 and type(start) is int and type(end) is int
            and 0 <= start < end and end - start == len(text)):
        raise ValueError('Invalid source span')
    return MARKER + SPAN.pack(source, start, end, hashlib.sha256(text.encode()).digest())


class Reader:
    def __init__(self, db, max_cache_bytes=32 * 1024**2):
        self.db, self.max_cache_bytes = db, max_cache_bytes
        self.cache, self.bytes, self.epoch = OrderedDict(), 0, None

    def _fingerprint(self, content):
        if not content.startswith(MARKDOWN_MARKER):
            return None
        ref = json.loads(content[len(MARKDOWN_MARKER):])
        root = (database_root(self.db) / 'markdown').resolve()
        relative = Path(ref['archive'])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError('Markdown archive path escapes the collection')
        s = path.stat()
        return (s.st_ino, s.st_mtime_ns, s.st_size)

    def resolve(self, value):
        if isinstance(value, str) or value is None:
            return value
        if not isinstance(value, bytes) or not value.startswith(MARKER) or len(value) != len(MARKER) + SPAN.size:
            raise ValueError('Invalid chunk reference')
        source, start, end, expected = SPAN.unpack(value[len(MARKER):])
        epoch = (self.db.total_changes, self.db.execute('PRAGMA data_version').fetchone()[0])
        if epoch != self.epoch:
            self.cache.clear(); self.bytes = 0; self.epoch = epoch
        entry = self.cache.pop(source, None)
        if entry is not None:
            self.bytes -= entry[3]
            if self._fingerprint(entry[0]) != entry[1]:
                entry = None
        if entry is None:
            row = self.db.execute('''SELECT v.* FROM chunk_sources s JOIN document_versions v
                ON v.accession=s.accession AND v.version=s.version WHERE s.id=?''', (source,)).fetchone()
            if row is None:
                raise ValueError('Chunk source version missing')
            if hashlib.sha256((row['text_hash'] + row['metadata']).encode()).hexdigest() != row['version']:
                raise ValueError('Chunk source metadata checksum mismatch')
            body = read_text(row['content'], self.db)
            if hashlib.sha256(body.encode()).hexdigest() != row['text_hash']:
                raise ValueError('Chunk source text checksum mismatch')
            entry = (row['content'], self._fingerprint(row['content']), body,
                     len(body.encode()) + len(row['content']))
        if entry[3] <= self.max_cache_bytes:
            self.cache[source] = entry; self.bytes += entry[3]
            while self.bytes > self.max_cache_bytes:
                self.bytes -= self.cache.popitem(last=False)[1][3]
        body = entry[2]
        if not 0 <= start < end <= len(body):
            raise ValueError('Chunk source offsets invalid')
        text = body[start:end]
        if hashlib.sha256(text.encode()).digest() != expected:
            raise ValueError('Chunk source passage checksum mismatch')
        return text

    def row(self, cursor, values):
        if any(column[0] == 'text' and isinstance(value, bytes)
               for column, value in zip(cursor.description, values)):
            values = tuple(self.resolve(value) if column[0] == 'text' and isinstance(value, bytes) else value
                           for column, value in zip(cursor.description, values))
        return sqlite3.Row(cursor, values)


def install_reader(db):
    reader = Reader(db)
    db.row_factory = reader.row
    return reader
