"""Versioned local embeddings and a persistent approximate-nearest-neighbor index."""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import uuid
from .storage import require_space

DEFAULT_EMBEDDING = "BAAI/bge-small-en-v1.5"
EMBEDDING_REPO = "Qdrant/bge-small-en-v1.5-onnx-Q"
EMBEDDING_REVISION = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"


def cpu_threads():
    value = int(os.environ.get("ADAMS_EMBEDDING_THREADS", "4"))
    if not 1 <= value <= 64:
        raise ValueError("ADAMS_EMBEDDING_THREADS must be between 1 and 64")
    return value


def active_model(directory):
    path = Path(directory) / "ACTIVE_MODEL.json"
    if not path.exists():
        return DEFAULT_EMBEDDING
    from .mlx_embeddings import MODEL
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or set(value) != {"version", "model"}
            or type(value["version"]) is not int or value["version"] != 1
            or value["model"] not in (DEFAULT_EMBEDDING, MODEL)):
        raise ValueError("Invalid active embedding profile")
    return value["model"]


def window_search_enabled(directory):
    """Validate the explicit reader configuration, separately from preparation."""
    path = Path(directory) / 'WINDOW_SEARCH.json'
    if not path.exists():
        return False
    from .embedding_windows import MODEL, POLICY
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or set(value) != {'version','model','policy','enabled'}
            or type(value['version']) is not int or value['version'] != 1
            or type(value['enabled']) is not bool or value['model'] != MODEL or value['policy'] != POLICY):
        raise ValueError('Invalid window search configuration')
    if not value['enabled']:
        return False
    if active_model(directory) != MODEL:
        raise ValueError('Window search requires the same pinned active embedding model')
    return True


def search_index(store, directory, encoder=None):
    """Select an explicitly enabled, compatible window search profile."""
    if not window_search_enabled(directory):
        return SemanticIndex(store, directory, encoder=encoder)
    from .window_index import WindowSemanticIndex
    return WindowSemanticIndex(store, directory, encoder)


class SemanticIndex:
    def __init__(self, store, directory, model=None, encoder=None):
        model = model if model is not None else active_model(directory)
        self.store, self.directory, self.model = store, Path(directory), model
        self.directory.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self.index = None
        self.generation = None
        self.key = hashlib.sha256(model.encode()).hexdigest()[:16]
        self.path = self.directory / (self.key + ".usearch")
        self.manifest = self.directory / (self.key + ".json")
        store.db.executescript("""
          CREATE TABLE IF NOT EXISTS dense_vectors(
            chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            model TEXT NOT NULL, dimension INTEGER NOT NULL, vector BLOB NOT NULL,
            content_hash TEXT NOT NULL, PRIMARY KEY(chunk_id,model));
        """)

    def _encoder(self):
        if self.encoder is None:
            from .mlx_embeddings import MODEL, GuardedEncoder
            if self.model == MODEL:
                self.encoder = GuardedEncoder(self.directory)
                return self.encoder
            import onnxruntime
            onnxruntime.disable_telemetry_events()
            options = {}
            if self.model == DEFAULT_EMBEDDING:
                folder = self.directory / "models" / ("models--" + EMBEDDING_REPO.replace("/", "--")) / "snapshots" / EMBEDDING_REVISION
                required = ["model_optimized.onnx", "tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json"]
                if not all((folder / name).is_file() for name in required):
                    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
                    from huggingface_hub import snapshot_download
                    folder = Path(snapshot_download(EMBEDDING_REPO, revision=EMBEDDING_REVISION,
                        cache_dir=str(self.directory / "models"), allow_patterns=required, token=False))
                options = {"specific_model_path": str(folder), "local_files_only": True}
            from fastembed import TextEmbedding
            self.encoder = TextEmbedding(self.model, cache_dir=str(self.directory / "models"), threads=cpu_threads(), **options)
        return self.encoder

    def sync(self, limit=2000, batch_size=8, accessions=None, progress=None, priority_model=None, should_yield=None):
        import numpy as np
        if not 1 <= limit <= 1_000_000 or not 1 <= batch_size <= 256:
            raise ValueError("Use a positive embedding limit and a batch size between 1 and 256")
        if priority_model is not None and (not isinstance(priority_model, str) or not priority_model.strip()
                                           or priority_model == self.model):
            raise ValueError("The priority model must be a different embedding profile")
        # A narrow covering index lets pending selection walk passage IDs without
        # rereading the growing text corpus. Build it once, with room for sorting.
        if not self.store.db.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='chunks_pending_ids'").fetchone():
            count = self.store.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
            require_space(self.store.path.parent, extra_bytes=max(32 * 1024 * 1024, count * 64))
            with self.store.db:
                self.store.db.execute("CREATE INDEX IF NOT EXISTS chunks_pending_ids ON chunks(id)")
        params = [self.model]
        condition = ""
        if accessions:
            condition = " AND c.accession IN (" + ",".join("?" for _ in accessions) + ")"
            params.extend(accessions)
        params.append(limit)
        rows = []
        # Determine the bounded set before loading titles and passage text. The
        # materialized CTE prevents SQLite from joining every already-indexed
        # passage to its document on each supervised job.
        load_selected = """ SELECT c.id,c.text,d.title FROM pending p
            CROSS JOIN chunks c ON c.id=p.id
            JOIN documents d ON d.accession=c.accession ORDER BY c.id"""
        if priority_model is not None:
            # First preserve the passages served by the active profile, even if
            # interactive ingestion embedded them far ahead of the FIFO cursor.
            rows = self.store.db.execute("""WITH pending AS MATERIALIZED (SELECT c.id FROM dense_vectors p
              JOIN chunks c ON c.id=p.chunk_id
              LEFT JOIN dense_vectors v ON v.chunk_id=c.id AND v.model=?
              WHERE p.model=? AND v.chunk_id IS NULL""" + condition + " ORDER BY p.chunk_id LIMIT ?)" + load_selected,
              [self.model, priority_model, *params[1:]]).fetchall()
        priority_selected = len(rows)
        if len(rows) < limit:
            chosen = {row["id"] for row in rows}
            pending = self.store.db.execute("""WITH pending AS MATERIALIZED (SELECT c.id FROM chunks c
              LEFT JOIN dense_vectors v ON v.chunk_id=c.id AND v.model=?
              WHERE v.chunk_id IS NULL""" + condition + " ORDER BY c.id LIMIT ?)" + load_selected, params).fetchall()
            rows.extend(row for row in pending if row["id"] not in chosen)
            rows = rows[:limit]
        embedded, yielded = 0, False
        for start in range(0, len(rows), batch_size):
            require_space(self.directory)
            if progress:
                progress(start, len(rows))
            if should_yield and should_yield():
                yielded = True
                break
            batch = rows[start:start + batch_size]
            texts = [r["title"] + "\n" + r["text"] for r in batch]
            vectors = list(self._encoder().passage_embed(texts, batch_size=batch_size))
            if len(vectors) != len(batch):
                raise ValueError("Embedding count mismatch")
            require_space(self.directory)
            with self.store.db:
                self.store.db.execute("BEGIN IMMEDIATE")
                for row, text, vector in zip(batch, texts, vectors):
                    current = self.store.db.execute("SELECT d.title,c.text FROM chunks c JOIN documents d ON d.accession=c.accession WHERE c.id=?", (row["id"],)).fetchone()
                    if not current or current[0] + "\n" + current[1] != text:
                        continue  # Source was revised while this batch was being embedded.
                    vector = np.asarray(vector, dtype=np.float32)
                    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                        raise ValueError("Invalid embedding")
                    self.store.db.execute("INSERT OR REPLACE INTO dense_vectors VALUES(?,?,?,?,?)",
                        (row["id"], self.model, len(vector), vector.tobytes(), hashlib.sha256(text.encode()).hexdigest()))
                    embedded += 1
            if progress:
                progress(min(start + batch_size, len(rows)), len(rows))
        # Committed vectors and their change ledger are durable. A queued search
        # can rebuild from its own snapshot; the next background job also picks
        # up all unpublished changes. Avoid holding the GPU for graph I/O here.
        if rows and not yielded:
            self.rebuild(incremental=True)
        return {"embedded": embedded, "model": self.model, "priority_model": priority_model,
                "priority_selected": priority_selected, "yielded_for_interactive": yielded,
                "graph_publication_deferred": yielded, **self.status()}

    def coverage_against(self, model):
        """Compare stored passage identities and inputs, not retrieval quality."""
        row = self.store.db.execute("""SELECT count(*) AS baseline_vectors,
            coalesce(sum(v.chunk_id IS NULL),0) AS missing_passages,
            coalesce(sum(v.chunk_id IS NOT NULL AND v.content_hash<>b.content_hash),0) AS different_input_hashes
            FROM dense_vectors b LEFT JOIN dense_vectors v ON v.chunk_id=b.chunk_id AND v.model=?
            WHERE b.model=?""", (self.model, model)).fetchone()
        return {"baseline_model": model, "candidate_model": self.model, **dict(row),
                "retrieval_quality_verified": False}

    def covered_chunk_ids(self, chunk_ids):
        """Passages with vectors in this profile, within the caller's read snapshot."""
        ids = list(chunk_ids)
        if any(type(value) is not int or value < 1 for value in ids):
            raise ValueError("Use positive integer passage IDs")
        ids = list(dict.fromkeys(ids))
        covered = set()
        # Stay below conservative SQLite parameter limits even for larger callers.
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            covered.update(row[0] for row in self.store.db.execute(
                "SELECT chunk_id FROM dense_vectors WHERE model=? AND chunk_id IN (" +
                ",".join("?" for _ in batch) + ")", (self.model, *batch)))
        return covered

    def status(self):
        row = self.store.db.execute("SELECT count(*),coalesce(max(chunk_id),0) FROM dense_vectors WHERE model=?", (self.model,)).fetchone()
        return {"vectors": row[0], "last_chunk_id": row[1], "index_bytes": self.path.stat().st_size if self.path.exists() else 0}

    def fingerprint(self):
        # Database triggers advance the generation even for same-count replacements.
        # This avoids scanning every vector on each interactive retrieval.
        identity = self.store.db.execute("SELECT value FROM collection_identity WHERE id=1").fetchone()[0]
        row = self.store.db.execute("SELECT generation FROM vector_generations WHERE model=?", (self.model,)).fetchone()
        return hashlib.sha256(f"{self.store.path.resolve()}:{identity}:{row[0] if row else 0}".encode()).hexdigest()

    def rebuild(self, incremental=False):
        # Different API/CLI processes may index simultaneously. Publish one coherent
        # vector snapshot and manifest at a time; readers never see half a generation.
        with (self.directory / (self.key + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._rebuild_locked(incremental)

    def _rebuild_locked(self, incremental=False):
        import numpy as np
        from usearch.index import Index
        # A new index is written beside the old one before atomic publication.
        count = self.status()["vectors"]
        require_space(self.directory, extra_bytes=max(32 * 1024 * 1024, count * 1400,
                      self.path.stat().st_size if self.path.exists() else 0))
        own_transaction = not self.store.db.in_transaction
        if own_transaction:
            self.store.db.execute("BEGIN")
        try:
            first = self.store.db.execute("SELECT dimension FROM dense_vectors WHERE model=? LIMIT 1", (self.model,)).fetchone()
            if not first:
                self.index = None
                return
            identity = self.store.db.execute("SELECT value FROM collection_identity WHERE id=1").fetchone()[0]
            identity = hashlib.sha256(f"{self.store.path.resolve()}:{identity}".encode()).hexdigest()
            snapshot_generation = self.store.db.execute("SELECT generation FROM vector_generations WHERE model=?", (self.model,)).fetchone()[0]
            index, previous = None, {}
            if incremental and self.path.exists() and self.manifest.exists():
                previous = json.loads(self.manifest.read_text())
                if (previous.get("tracking_version") == 1 and previous.get("identity") == identity
                        and previous.get("model") == self.model and previous.get("dimension") == first[0]
                        and type(previous.get("snapshot_generation")) is int
                        and previous["snapshot_generation"] <= snapshot_generation):
                    index = Index.restore(str(self.path), view=False)
            updated, removed = 0, 0
            if index is None:
                index = Index(ndim=first[0], metric="cos", dtype="f16", connectivity=16, expansion_add=64, expansion_search=80)
                cursor = self.store.db.execute("SELECT chunk_id,dimension,vector FROM dense_vectors WHERE model=? ORDER BY chunk_id", (self.model,))
                build_mode = "full"
            else:
                cursor = self.store.db.execute("""SELECT c.chunk_id,v.dimension,v.vector FROM vector_changes c
                    LEFT JOIN dense_vectors v ON v.model=c.model AND v.chunk_id=c.chunk_id
                    WHERE c.model=? AND c.generation>? ORDER BY c.generation,c.chunk_id""",
                    (self.model, previous["snapshot_generation"]))
                build_mode = "incremental"
            while rows := cursor.fetchmany(1024):
                if build_mode == "incremental":
                    keys = np.asarray([r[0] for r in rows], dtype=np.uint64)
                    removed += int(np.sum(index.remove(keys, compact=False, threads=cpu_threads())))
                    rows = [r for r in rows if r[2] is not None]
                if not rows:
                    continue
                if any(r[1] != first[0] for r in rows):
                    raise ValueError("Embedding dimension changed")
                index.add(np.asarray([r[0] for r in rows], dtype=np.uint64),
                          np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows]), threads=cpu_threads())
                updated += len(rows)
            expected = self.store.db.execute("SELECT count(*) FROM dense_vectors WHERE model=?", (self.model,)).fetchone()[0]
            if len(index) != expected:
                raise ValueError("Semantic index count differs from its database snapshot")
            generation = self.fingerprint()
        finally:
            if own_transaction:
                self.store.db.rollback()  # Release the read-only snapshot.
        temporary = str(self.path) + "." + uuid.uuid4().hex + ".tmp"
        require_space(self.directory, extra_bytes=max(32 * 1024 * 1024, expected * (first[0] * 2 + 512)))
        try:
            index.save(temporary)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        temporary_manifest = Path(str(self.manifest) + ".tmp")
        temporary_manifest.write_text(json.dumps({"model": self.model, "dimension": first[0], "generation": generation,
            "identity": identity, "snapshot_generation": snapshot_generation, "tracking_version": 1,
            "build_mode": build_mode, "vectors_added": updated, "vectors_removed": removed}))
        os.replace(temporary_manifest, self.manifest)
        self.index, self.generation = index, generation

    def search(self, question, limit=40, docket=None, date_from=None, date_to=None):
        if not self.status()["vectors"]:
            return []
        vector = next(iter(self._encoder().query_embed(question)))
        return self.search_vector(vector, limit, docket, date_from, date_to)

    def search_vector(self, vector, limit=40, docket=None, date_from=None, date_to=None):
        """Reuse one query embedding across compatible semantic candidate indexes."""
        import numpy as np
        from usearch.index import Index
        if not self.status()["vectors"]:
            return []
        generation = self.fingerprint()
        if self.generation != generation:
            if self.path.exists() and self.manifest.exists():
                with (self.directory / (self.key + ".lock")).open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_SH)
                    data = json.loads(self.manifest.read_text())
                    if data.get("generation") == generation and data.get("model") == self.model:
                        self.index = Index.restore(str(self.path), view=True)
                        self.generation = generation
            if self.generation != generation:
                self.rebuild(incremental=True)
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("Invalid semantic query vector")
        # Overfetch for metadata filters; this is bounded retrieval, never an exhaustive query.
        count = min(len(self.index), max(limit, limit * 8))
        # Approximate traversal may omit nodes even when count requests the
        # whole graph. Use exhaustive enumeration in that bounded small case.
        matches = self.index.search(vector, count=count, threads=2, exact=count == len(self.index))
        result = []
        for key in matches.keys:
            row = self.store.db.execute("""SELECT c.*,d.title,d.url,d.provenance,d.document_date,d.metadata,
                v.vector,v.content_hash FROM chunks c JOIN documents d ON d.accession=c.accession
                JOIN dense_vectors v ON v.chunk_id=c.id AND v.model=? WHERE c.id=? AND d.status='indexed'""",
                (self.model, int(key))).fetchone()
            if row and matches_filters(dict(row), docket, date_from, date_to):
                if row['content_hash'] != hashlib.sha256((row['title']+'\n'+row['text']).encode()).hexdigest():
                    continue
                value = np.frombuffer(row['vector'], dtype=np.float32)
                if value.shape != vector.shape or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
                    continue
                item = {k:row[k] for k in row.keys() if k not in ('vector','content_hash')}
                item['semantic_score'] = float(vector @ value / (np.linalg.norm(vector) * np.linalg.norm(value)))
                result.append(item)
                if len(result) >= limit:
                    break
        return result


def matches_filters(row, docket=None, date_from=None, date_to=None):
    metadata = json.loads(row.get("metadata", "{}"))
    if docket and docket not in metadata.get("DocketNumber", []):
        return False
    date = (row.get("document_date") or "")[:10]
    if date_from and (not date or date < date_from):
        return False
    if date_to and (not date or date > date_to):
        return False
    return True
