"""Small, bounded helpers for the interactive read path."""
import copy
import datetime as dt
import json
import time
from collections import OrderedDict


def coverage_snapshot(store):
    """Use an explicitly dated worker snapshot rather than scanning every table.

    This is a coverage label, never evidence or an input to retrieval. CLI audits
    retain Store.status() for exact, current counts.
    """
    try:
        value = json.loads((store.path.parent / 'archive' / 'STATUS.json').read_text())
        documents, chunks = value['indexed_documents'], value['chunks']
        observed = dt.datetime.fromisoformat(value['updated_utc'])
        if (type(documents) is not int or type(chunks) is not int or min(documents, chunks) < 0
                or observed.tzinfo is None):
            raise ValueError('Invalid coverage snapshot')
        return {'documents': {'indexed': documents}, 'chunks': chunks,
                'observed_utc': value['updated_utc'], 'count_source': 'archive_worker_snapshot',
                'counts_are_current': False, 'full_archive_verified': False}
    except (OSError, ValueError, KeyError, TypeError):
        return store.status()


def collection_revision(store):
    """Source changes and extraction holdouts invalidate answer reuse."""
    identity = store.db.execute('SELECT value FROM collection_identity WHERE id=1').fetchone()[0]
    generation = store.db.execute('SELECT generation FROM retrieval_state WHERE id=1').fetchone()[0]
    return identity, generation


class AnswerCache:
    """Memory-only, exact-request reuse. No failures, partial answers or disk copy."""
    def __init__(self, limit=16, ttl=120, clock=time.monotonic):
        self.entries = OrderedDict()
        self.limit, self.ttl, self.clock = limit, ttl, clock

    def get(self, key, revision):
        entry = self.entries.get(key)
        if entry is None:
            return None
        saved, old_revision, result = entry
        age = self.clock() - saved
        if old_revision != revision or age >= self.ttl:
            self.entries.pop(key, None)
            return None
        self.entries.move_to_end(key)
        value = copy.deepcopy(result)
        value['answer_cache'] = {'hit': True, 'age_seconds': round(age, 2),
                                 'source_checked_utc': result.get('created_utc')}
        return value

    def put(self, key, revision, result):
        if result.get('mode') != 'generated' or not result.get('answer'):
            return
        self.entries[key] = self.clock(), revision, copy.deepcopy(result)
        self.entries.move_to_end(key)
        while len(self.entries) > self.limit:
            self.entries.popitem(last=False)


def cache_key(payload, engine, model):
    fields = {'question': payload['question'].strip(), 'live': payload.get('live', True),
              'docket': payload.get('docket'), 'date_from': payload.get('date_from'),
              'date_to': payload.get('date_to'), 'top_k': payload.get('top_k', 10),
              'max_documents': payload.get('max_documents', 8), 'embedding_model': model,
              'engine': type(engine).__name__, 'model': str(getattr(engine, 'model', '')),
              'cpu_only': bool(getattr(engine, 'cpu_only', False))}
    return json.dumps(fields, sort_keys=True)
