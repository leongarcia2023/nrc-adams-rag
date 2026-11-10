"""Version-pinned text in either the legacy BLOB or the canonical Markdown archive."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import threading
import zipfile
import zlib

MARKER = b'ADAMS-MARKDOWN-V1\0'
_local = threading.local()


def reference(db, row, check):
    root = database_root(db) / 'markdown'
    archive = Path(check['archive']).resolve().relative_to(root.resolve())
    return MARKER + json.dumps(dict(archive=str(archive), member=check['member'],
        markdown_sha256=check['markdown_sha256'], accession=row['accession'],
        version=row['version'], text_hash=row['text_hash']), sort_keys=True).encode()


def database_root(db):
    path = next(r[2] for r in db.execute('PRAGMA database_list') if r[1] == 'main')
    if not path:
        raise ValueError('Markdown references require a file-backed database')
    return Path(path).resolve().parent


def read_text(content, db):
    """Fail closed on a missing, changed, or incorrectly bound canonical source."""
    if not content.startswith(MARKER):
        return zlib.decompress(content).decode()
    from .markdown_archive import decode_markdown
    try:
        ref = json.loads(content[len(MARKER):])
        root = (database_root(db) / 'markdown').resolve()
        relative = Path(ref['archive'])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError('Markdown archive path escapes the collection')
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino)
        cache = getattr(_local, 'archives', None)
        if cache is None:
            cache = _local.archives = OrderedDict()
        archive = cache.pop(key, None)
        if archive is None:
            archive = zipfile.ZipFile(path)
        cache[key] = archive
        while len(cache) > 4:
            cache.popitem(last=False)[1].close()
        info = archive.getinfo(ref['member'])
        if info.file_size > 128 * 1024**2:
            raise ValueError('Markdown source exceeds the supported read size')
        markdown = archive.read(info).decode()
        if hashlib.sha256(markdown.encode()).hexdigest() != ref['markdown_sha256']:
            raise ValueError('Markdown source checksum mismatch')
        header, body = decode_markdown(markdown)
        if (header['source_key'], header['retained_version'], header['source_text_sha256']) != (
                ref['accession'], ref['version'], ref['text_hash']):
            raise ValueError('Markdown source version mismatch')
        if hashlib.sha256(body.encode()).hexdigest() != ref['text_hash']:
            raise ValueError('Markdown source text mismatch')
        return body
    except (OSError, KeyError, zipfile.BadZipFile, UnicodeError) as error:
        raise ValueError('Canonical Markdown source unavailable or invalid') from error


def offload(db, row, check):
    """Replace only a byte-for-byte covered retained body, keeping all metadata."""
    if row['content'].startswith(MARKER):
        return 0
    pointer = reference(db, row, check)
    if read_text(pointer, db) != read_text(row['content'], db):
        raise ValueError('Markdown does not cover the retained source text')
    changed = db.execute('UPDATE document_versions SET content=? WHERE accession=? AND version=? AND content=?',
        (pointer, row['accession'], row['version'], row['content'])).rowcount
    return len(row['content']) - len(pointer) if changed else 0
