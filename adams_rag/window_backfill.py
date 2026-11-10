"""Bounded, resumable preparation of complete passage windows; no search activation."""
from collections import defaultdict

from .embedding_windows import MODEL
from .storage import require_space
from .window_store import WindowStore, StaleWindowInput, atomic


class WindowBackfill:
    def __init__(self, store, planner, encoder=None):
        self.store, self.db, self.encoder = store, store.db, encoder
        self.windows = WindowStore(store, planner)
        self._initialize()

    def _initialize(self):
        # Completion is an optimization for finding work, never query evidence.
        # Every relevant mutation removes the marker; a low ID revised later
        # therefore reenters the queue without resetting a high-water cursor.
        require_space(self.store.path.parent)
        statements = [
            """CREATE TABLE IF NOT EXISTS window_backfill_completed(
                chunk_id INTEGER NOT NULL,profile_id INTEGER NOT NULL,
                PRIMARY KEY(chunk_id,profile_id),
                FOREIGN KEY(chunk_id,profile_id) REFERENCES embedding_window_sets(chunk_id,profile_id) ON DELETE CASCADE)""",
        ]
        for table, prefix in (("embedding_windows", "window"), ("embedding_window_sets", "set")):
            for operation, references in (("insert", ("new",)), ("delete", ("old",)), ("update", ("old", "new"))):
                body = " ".join(f"DELETE FROM window_backfill_completed WHERE chunk_id={ref}.chunk_id AND profile_id={ref}.profile_id;" for ref in references)
                statements.append(f"CREATE TRIGGER IF NOT EXISTS window_backfill_{prefix}_{operation}_v1 AFTER {operation.upper()} ON {table} BEGIN {body} END")
        for operation, references in (("insert", ("new",)), ("delete", ("old",)), ("update", ("old", "new"))):
            body = " ".join(f"""DELETE FROM window_backfill_completed WHERE chunk_id={ref}.chunk_id
                AND profile_id IN (SELECT w.profile_id FROM embedding_windows w
                    JOIN embedding_window_profiles p ON p.id=w.profile_id
                    WHERE w.chunk_id={ref}.chunk_id AND p.model={ref}.model
                    AND w.base_compatible=1 AND w.vector IS NULL);""" for ref in references)
            statements.append(f"CREATE TRIGGER IF NOT EXISTS window_backfill_base_{operation}_v1 AFTER {operation.upper()} ON dense_vectors BEGIN {body} END")
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='chunks_pending_ids' AND type='index'").fetchone():
            count = self.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
            require_space(self.store.path.parent, extra_bytes=max(32 * 1024**2, count * 64))
            statements.append("CREATE INDEX IF NOT EXISTS chunks_pending_ids ON chunks(id)")
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='window_stored_profile' AND type='index'").fetchone():
            count = self.db.execute("SELECT count(*) FROM embedding_windows").fetchone()[0]
            require_space(self.store.path.parent, extra_bytes=max(32 * 1024**2, count * 64))
            # Frequent progress reports should count a narrow index instead of
            # scanning the pages holding every stored 1,536-byte vector.
            statements.append("CREATE INDEX IF NOT EXISTS window_stored_profile ON embedding_windows(profile_id) WHERE vector IS NOT NULL")
        with atomic(self.db):
            for statement in statements:
                self.db.execute(statement)

    def pending(self, limit=128):
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("Use 1–256 passages per window job")
        # Read the covering ID index first. Only unmarked candidates need their
        # document status loaded; completed passage bodies are not reread.
        return [row[0] for row in self.db.execute("""SELECT c.id FROM chunks c INDEXED BY chunks_pending_ids
            WHERE NOT EXISTS (SELECT 1 FROM window_backfill_completed b WHERE b.chunk_id=c.id AND b.profile_id=?)
            AND EXISTS (SELECT 1 FROM chunks current JOIN documents d ON d.accession=current.accession
                        WHERE current.id=c.id AND d.status='indexed')
            ORDER BY c.id LIMIT ?""", (self.windows.profile, limit))]

    def mark_complete(self, chunk_id):
        require_space(self.store.path.parent)
        with atomic(self.db):
            if self.windows.resolve(chunk_id)["state"] != "complete":
                return False
            self.db.execute("INSERT OR IGNORE INTO window_backfill_completed VALUES(?,?)", (chunk_id, self.windows.profile))
        return True

    def status(self):
        profile = self.windows.profile
        return {"planned_passages": self.db.execute("SELECT count(*) FROM embedding_window_sets WHERE profile_id=?", (profile,)).fetchone()[0],
                "completed_passages": self.db.execute("SELECT count(*) FROM window_backfill_completed WHERE profile_id=?", (profile,)).fetchone()[0],
                "stored_window_vectors": self.db.execute("SELECT count(*) FROM embedding_windows WHERE profile_id=? AND vector IS NOT NULL", (profile,)).fetchone()[0],
                "window_generation": self.windows.generation(), "role": "window_preparation"}

    def prepare_existing(self, limit=128, progress=None):
        """Plan existing prefix inputs without loading an encoder or copying vectors.

        A short, exact-compatible input can be complete using its saved prefix.
        Longer inputs remain pending in the ordinary GPU queue. Selecting only
        unplanned inputs lets this pass advance past those incomplete passages;
        it does not use a cursor that could hide later source revisions.
        """
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("Use 1–256 passages per window job")
        selected = [row[0] for row in self.db.execute("""SELECT v.chunk_id FROM dense_vectors v
            WHERE v.model=? AND NOT EXISTS (
                SELECT 1 FROM embedding_window_sets s WHERE s.chunk_id=v.chunk_id AND s.profile_id=?)
            AND EXISTS (SELECT 1 FROM chunks c JOIN documents d ON d.accession=c.accession
                        WHERE c.id=v.chunk_id AND d.status='indexed')
            ORDER BY v.chunk_id LIMIT ?""", (MODEL, self.windows.profile, limit))]
        result = {"selected_passages": len(selected), "planned_this_job": 0, "embedded": 0,
                  "completed_this_job": 0, "pending_this_job": 0, "stale_passages": 0,
                  "mode": "reuse_only", "yielded_for_interactive": False}
        for chunk in selected:
            require_space(self.store.path.parent)
            if progress:
                progress()
            try:
                plan = self.windows.plan(chunk)
                self.windows.register(chunk, plan)
                result["planned_this_job"] += 1
                # resolve() checks the current source hash, profile, exact input
                # compatibility, vector shape and normalization before reuse.
                if self.mark_complete(chunk):
                    result["completed_this_job"] += 1
                else:
                    result["pending_this_job"] += 1
            except StaleWindowInput:
                result["stale_passages"] += 1
        return {**result, **self.status()}

    def sync(self, limit=128, vector_limit=128, progress=None, should_yield=None):
        import numpy as np
        if type(vector_limit) is not int or not 1 <= vector_limit <= 128:
            raise ValueError("Use 1–128 new vectors per window job")
        selected = self.pending(limit)
        result = {"selected_passages": len(selected), "planned_this_job": 0, "embedded": 0,
                  "completed_this_job": 0, "stale_passages": 0, "yielded_for_interactive": False,
                  "vector_limit_reached": False}
        completed, stale, batch = set(), set(), []

        def check():
            require_space(self.store.path.parent)
            if progress:
                progress()
            if should_yield and should_yield():
                result["yielded_for_interactive"] = True
                return False
            return True

        def flush():
            if not batch:
                return True
            if not check():
                return False
            if self.encoder is None:
                raise ValueError("Window embedding requires the supervised encoder")
            vectors = list(self.encoder.passage_embed([item[2].text for item in batch], batch_size=8))
            if len(vectors) != len(batch):
                raise ValueError("Window embedding count mismatch")
            encoded = []
            for vector in vectors:
                array = np.asarray(vector, dtype="<f4")
                if array.shape != (384,) or WindowStore._vector(array.tobytes(), 384) is None:
                    raise ValueError("Invalid window embedding")
                encoded.append(array)
            # Validate the whole returned batch before writing any of it. Each
            # passage commit then checks source identity again under its lock.
            require_space(self.store.path.parent)
            if progress:
                progress()
            grouped, plans = defaultdict(dict), {}
            for (chunk, plan, window), vector in zip(batch, encoded):
                grouped[chunk][window.ordinal] = vector
                plans[chunk] = plan
            for chunk, payload in grouped.items():
                try:
                    result["embedded"] += self.windows.commit(chunk, plans[chunk], payload)
                    if self.mark_complete(chunk):
                        completed.add(chunk)
                except StaleWindowInput:
                    stale.add(chunk)
            batch.clear()
            return True

        requested = 0
        for chunk in selected:
            if not check():
                break
            try:
                plan = self.windows.plan(chunk)
                self.windows.register(chunk, plan)
                result["planned_this_job"] += 1
                missing = self.windows.missing(chunk, plan)
            except StaleWindowInput:
                stale.add(chunk)
                continue
            if not missing:
                if self.mark_complete(chunk):
                    completed.add(chunk)
                continue
            for window in missing:
                batch.append((chunk, plan, window))
                requested += 1
                if len(batch) == 8 or requested == vector_limit:
                    if not flush():
                        break
                if chunk in stale or requested == vector_limit:
                    break
            if result["yielded_for_interactive"]:
                break
            if requested == vector_limit:
                result["vector_limit_reached"] = True
                break
        if not result["yielded_for_interactive"]:
            flush()
        result["completed_this_job"] = len(completed)
        result["stale_passages"] = len(stale)
        return {**result, **self.status()}
