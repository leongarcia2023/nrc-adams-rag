"""An additional ANN graph for stored windows, merged with existing prefix search."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

from .embedding_windows import MODEL, POLICY, text_hash
from .evidence import read_snapshot, SourceBindings, SourceBindingError, report_binding_issue
from .extraction_review import filter_rows
from .semantic_index import SemanticIndex, matches_filters
from .storage import require_space
from .window_store import WindowStore


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def query_vector(value):
    import numpy as np
    value = np.asarray(value, dtype='<f4')
    if value.shape != (384,) or WindowStore._vector(value.tobytes(), 384) is None:
        raise ValueError('Use a finite normalized query from the pinned 384-dimensional model')
    return value


class WindowIndex:
    """Only additional stored vectors live here; compatible prefixes stay in the base graph."""
    def __init__(self, store, directory):
        self.store, self.db = store, store.db
        self.windows = WindowStore(store, None, initialize=False)
        self.directory = Path(directory) / 'windows'
        if not self.directory.exists():
            require_space(Path(directory))
            self.directory.mkdir(parents=True, exist_ok=True)
        self.key = 'extra-' + text_hash(MODEL + '\n' + POLICY)[:16]
        self.manifest = self.directory / (self.key + '.json')
        self.lock_path = self.directory / (self.key + '.lock')
        self.index, self.loaded = None, None

    def snapshot(self):
        identity = self.db.execute('SELECT value FROM collection_identity WHERE id=1').fetchone()[0]
        return {'identity': text_hash(str(self.store.path.resolve()) + ':' + identity),
                'model': MODEL, 'policy': POLICY, 'dimension': 384,
                'generation': self.windows.generation(), 'format': 1}

    def count(self):
        return self.db.execute('SELECT count(*) FROM embedding_windows WHERE profile_id=? AND vector IS NOT NULL',
                               (self.windows.profile,)).fetchone()[0]

    def _manifest(self):
        try:
            data = json.loads(self.manifest.read_text())
            graph = data.get('graph')
            if (not isinstance(graph, str) or Path(graph).name != graph
                    or not graph.startswith(self.key + '-') or not graph.endswith('.usearch')):
                return None
            if file_hash(self.directory / graph) != data.get('graph_sha256'):
                return None
            return data
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def _resolved(self, chunk_id, cache):
        if chunk_id not in cache:
            current = self.windows.resolve(chunk_id)
            if current['state'] not in ('complete', 'pending'):
                raise ValueError('Window plan failed integrity checks; repair it before publication')
            cache[chunk_id] = {w['id']:w for w in current['windows']}
        return cache[chunk_id]

    def _build(self, snapshot, previous):
        import numpy as np
        from usearch.index import Index
        count = self.count()
        require_space(self.directory, extra_bytes=max(32 * 1024**2, count * 1400))
        compatible = (previous is not None and all(previous.get(k) == v for k,v in snapshot.items() if k != 'generation')
                      and type(previous.get('generation')) is int and previous['generation'] <= snapshot['generation'])
        index = Index.restore(str(self.directory / previous['graph']), view=False) if compatible else None
        if index is not None and len(index) != previous.get('vectors'):
            index = None
        incremental = index is not None
        if not incremental:
            index = Index(ndim=384, metric='cos', dtype='f16', connectivity=16, expansion_add=64, expansion_search=80)
            cursor = self.db.execute('SELECT id,chunk_id,vector FROM embedding_windows WHERE profile_id=? AND vector IS NOT NULL ORDER BY id', (self.windows.profile,))
        else:
            cursor = self.db.execute('''SELECT c.window_id,w.chunk_id,w.vector FROM window_changes c
                LEFT JOIN embedding_windows w ON w.id=c.window_id AND w.profile_id=c.profile_id
                WHERE c.profile_id=? AND c.generation>? ORDER BY c.generation,c.window_id''',
                (self.windows.profile, previous['generation']))
        added, removed = 0, 0
        while rows := cursor.fetchmany(512):
            require_space(self.directory, extra_bytes=max(32 * 1024**2, count * 1400))
            if incremental:
                removed += int(np.sum(index.remove(np.asarray([r[0] for r in rows], dtype=np.uint64), compact=False, threads=4)))
            keys, vectors, cache = [], [], {}
            for key, chunk, raw in rows:
                if raw is None:
                    continue
                window = self._resolved(chunk, cache).get(key)
                if window is None or window['storage'] != 'stored' or window['vector'] != raw:
                    raise ValueError('Stored window vector failed integrity checks')
                keys.append(key); vectors.append(np.frombuffer(raw, dtype='<f4'))
            if keys:
                index.add(np.asarray(keys, dtype=np.uint64), np.stack(vectors), threads=4)
                added += len(keys)
        if len(index) != count:
            raise ValueError('Window graph count differs from its database snapshot')
        name = self.key + '-' + str(snapshot['generation']) + '-' + uuid.uuid4().hex + '.usearch'
        graph = self.directory / name
        temporary = graph.with_suffix('.tmp')
        temporary_manifest = self.manifest.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        published = False
        try:
            require_space(self.directory, extra_bytes=max(32 * 1024**2, count * 1400))
            index.save(str(temporary))
            os.replace(temporary, graph)
            data = {**snapshot, 'graph': name, 'graph_sha256': file_hash(graph), 'vectors': count,
                    'build_mode': 'incremental' if incremental else 'full', 'vectors_added': added, 'vectors_removed': removed}
            temporary_manifest.write_text(json.dumps(data))
            # Readers hold the same lock while opening the referenced immutable
            # graph. A crash before this swap leaves the previous pair intact.
            os.replace(temporary_manifest, self.manifest)
            published = True
            self.index, self.loaded = index, snapshot
            if previous and previous['graph'] != name:
                (self.directory / previous['graph']).unlink(missing_ok=True)
            return data
        finally:
            temporary.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)
            if not published:
                graph.unlink(missing_ok=True)

    def ensure(self, read_only=False):
        from usearch.index import Index
        with read_snapshot(self.db):
            snapshot = self.snapshot()
            if self.index is not None and self.loaded == snapshot:
                return
            with self.lock_path.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                previous = self._manifest()
                compatible = (previous and all(previous.get(k) == v for k,v in snapshot.items() if k != 'generation')
                              and type(previous.get('generation')) is int and previous['generation'] <= snapshot['generation'])
                if compatible and (read_only or previous['generation'] == snapshot['generation']):
                    index = Index.restore(str(self.directory / previous['graph']), view=True)
                    if len(index) == previous.get('vectors') and (read_only or len(index) == self.count()):
                        self.index, self.loaded = index, {**snapshot, 'generation': previous['generation']}
                        return
                if read_only:
                    raise RuntimeError('No compatible published window index; using keyword search until background publication')
                self._build(snapshot, previous)

    def candidates(self, vector, limit=80, read_only=False):
        vector = query_vector(vector)
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Use 1–1,000 parent passages')
        with read_snapshot(self.db):
            self.ensure(read_only=read_only)
            if not len(self.index):
                return []
            count = min(len(self.index), 4096, max(128, limit * 32))
            matches = self.index.search(vector, count=count, threads=2, exact=count == len(self.index))
            parents, cache = {}, {}
            for key in matches.keys:
                raw = self.db.execute('SELECT chunk_id FROM embedding_windows WHERE id=? AND profile_id=?',
                                      (int(key), self.windows.profile)).fetchone()
                if raw is None:
                    continue
                window = self._resolved(raw[0], cache).get(int(key))
                if not window or window['storage'] != 'stored' or window['vector'] is None:
                    continue
                import numpy as np
                value = np.frombuffer(window['vector'], dtype='<f4')
                score = float(vector @ value / (np.linalg.norm(vector) * np.linalg.norm(value)))
                if raw[0] in parents and parents[raw[0]]['semantic_score'] >= score:
                    continue
                row = self.db.execute('''SELECT c.*,d.title,d.url,d.provenance,d.document_date,d.metadata
                    FROM chunks c JOIN documents d ON d.accession=c.accession WHERE c.id=? AND d.status='indexed' ''', (raw[0],)).fetchone()
                if row:
                    parents[raw[0]] = {**dict(row), 'semantic_score': score,
                        'window_match': {'window_id': int(key), **window['spec'], 'offset_scope': 'title_plus_passage'}}
            return sorted(parents.values(), key=lambda row: (-row['semantic_score'], row['id']))


class WindowSemanticIndex(SemanticIndex):
    """One semantic ranking from existing prefixes and additional windows."""
    def __init__(self, store, directory, encoder=None):
        super().__init__(store, directory, model=MODEL, encoder=encoder)
        self.window_index = WindowIndex(store, directory)

    def search(self, question, limit=40, docket=None, date_from=None, date_to=None):
        return self.search_with_diagnostics(question, limit, docket, date_from, date_to)

    def search_with_diagnostics(self, question, limit=40, docket=None, date_from=None, date_to=None,
                                exclusions=None, source_issues=None):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Use 1–1,000 parent passages')
        vector = query_vector(next(iter(self._encoder().query_embed(question))))
        with read_snapshot(self.store.db):
            base = super().search_vector(vector, min(1000, limit * 4), docket, date_from, date_to)
            additional = self.window_index.candidates(vector, limit, read_only=self.read_only)
            merged = {}
            for row in base + additional:
                if matches_filters(row, docket, date_from, date_to):
                    old = merged.get(row['id'])
                    if old is None or row['semantic_score'] > old['semantic_score']:
                        merged[row['id']] = row
            rows = sorted(merged.values(), key=lambda row: (-row['semantic_score'], row['id']))
            rows = filter_rows(self.store.db, rows, exclusions)
            bindings, result = SourceBindings(self.store.db), []
            for row in rows:
                try:
                    _, binding = bindings.bind(row)
                except SourceBindingError as error:
                    report_binding_issue(source_issues, row, error)
                    continue
                result.append({**row, 'source_version': binding})
                if len(result) >= limit:
                    break
            return result

    def covered_chunk_ids(self, chunk_ids):
        ids = list(chunk_ids)
        with read_snapshot(self.store.db):
            covered = super().covered_chunk_ids(ids)
            for chunk in set(ids) - covered:
                view = self.window_index.windows.resolve(chunk)
                if any(w['storage'] == 'stored' and w['vector'] is not None for w in view['windows']):
                    covered.add(chunk)
            return covered
