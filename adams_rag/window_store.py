"""Staged durable window vectors. Creating this store does not activate retrieval."""
from contextlib import contextmanager

from .embedding_windows import canonical, text_hash, WindowPlan, POLICY
from .evidence import read_snapshot
from .mlx_embeddings import MODEL, FILES
from .storage import require_space


class StaleWindowInput(ValueError):
    pass


@contextmanager
def atomic(db):
    own = not db.in_transaction
    db.execute("BEGIN IMMEDIATE" if own else "SAVEPOINT window_store_write")
    try:
        yield
    except BaseException:
        if own:
            db.rollback()
        else:
            db.execute("ROLLBACK TO window_store_write")
            db.execute("RELEASE window_store_write")
        raise
    else:
        if own:
            db.commit()
        else:
            db.execute("RELEASE window_store_write")


class WindowStore:
    def __init__(self, store, planner, initialize=True):
        self.store, self.db, self.planner = store, store.db, planner
        if not self.db.execute("PRAGMA foreign_keys").fetchone()[0]:
            raise ValueError("Window storage requires foreign-key enforcement")
        if initialize:
            self.initialize()
        elif self._schema_version() != 2:
            raise ValueError("Window storage is unavailable or has an unsupported schema")
        row = self.db.execute("SELECT id,tokenizer_hash FROM embedding_window_profiles WHERE model=? AND policy=?", (MODEL, POLICY)).fetchone()
        if row is None or row["tokenizer_hash"] != FILES["tokenizer.json"]:
            raise ValueError("Window profile does not match the pinned tokenizer")
        self.profile = row["id"]

    def _schema_version(self):
        exists = self.db.execute("SELECT 1 FROM sqlite_master WHERE name='window_storage_schema'").fetchone()
        row = self.db.execute("SELECT version FROM window_storage_schema WHERE id=1").fetchone() if exists else None
        return row[0] if row else None

    def initialize(self):
        require_space(self.store.path.parent)
        old = self.db.execute("SELECT 1 FROM sqlite_master WHERE name='embedding_windows'").fetchone()
        versioned = self.db.execute("SELECT 1 FROM sqlite_master WHERE name='window_storage_schema'").fetchone()
        if (old or versioned) and self._schema_version() != 2:
            raise ValueError("Unrecognized staged window schema; an explicit migration is required")
        statements = [
            "CREATE TABLE IF NOT EXISTS window_storage_schema(id INTEGER PRIMARY KEY CHECK(id=1),version INTEGER NOT NULL)",
            "INSERT OR IGNORE INTO window_storage_schema VALUES(1,2)",
            """CREATE TABLE IF NOT EXISTS embedding_window_profiles(
                id INTEGER PRIMARY KEY AUTOINCREMENT,model TEXT NOT NULL,policy TEXT NOT NULL,tokenizer_hash TEXT NOT NULL,
                UNIQUE(model,policy))""",
            """CREATE TRIGGER IF NOT EXISTS window_profile_update_v2 BEFORE UPDATE ON embedding_window_profiles
                BEGIN SELECT RAISE(ABORT,'Window profile identity is immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS window_profile_delete_v2 BEFORE DELETE ON embedding_window_profiles
                BEGIN SELECT RAISE(ABORT,'Window profile identity is immutable'); END""",
            """CREATE TABLE IF NOT EXISTS embedding_window_sets(
                chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                profile_id INTEGER NOT NULL REFERENCES embedding_window_profiles(id),
                source_hash BLOB NOT NULL CHECK(typeof(source_hash)='blob' AND length(source_hash)=32),
                plan_hash BLOB NOT NULL CHECK(typeof(plan_hash)='blob' AND length(plan_hash)=32),
                input_characters INTEGER NOT NULL,input_tokens INTEGER NOT NULL,expected_windows INTEGER NOT NULL,
                PRIMARY KEY(chunk_id,profile_id))""",
            """CREATE TABLE IF NOT EXISTS embedding_windows(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER NOT NULL,profile_id INTEGER NOT NULL,ordinal INTEGER NOT NULL,
                plan_hash BLOB NOT NULL CHECK(typeof(plan_hash)='blob' AND length(plan_hash)=32),
                start_character INTEGER NOT NULL,end_character INTEGER NOT NULL,input_tokens INTEGER NOT NULL,
                input_hash BLOB NOT NULL CHECK(typeof(input_hash)='blob' AND length(input_hash)=32),
                token_hash BLOB NOT NULL CHECK(typeof(token_hash)='blob' AND length(token_hash)=32),
                base_compatible INTEGER NOT NULL CHECK(base_compatible IN (0,1)),
                dimension INTEGER,vector BLOB,
                UNIQUE(chunk_id,profile_id,ordinal),
                FOREIGN KEY(chunk_id,profile_id) REFERENCES embedding_window_sets(chunk_id,profile_id) ON DELETE CASCADE,
                CHECK((dimension IS NULL AND vector IS NULL) OR
                      (dimension=384 AND vector IS NOT NULL AND length(vector)=1536)))""",
            """CREATE TABLE IF NOT EXISTS window_generations(
                profile_id INTEGER PRIMARY KEY,generation INTEGER NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS window_changes(
                profile_id INTEGER NOT NULL,window_id INTEGER NOT NULL,generation INTEGER NOT NULL,
                PRIMARY KEY(profile_id,window_id))""",
            "CREATE INDEX IF NOT EXISTS window_changes_since ON window_changes(profile_id,generation)",
            """CREATE TRIGGER IF NOT EXISTS window_source_chunk_update_v1 AFTER UPDATE ON chunks
                WHEN old.text IS NOT new.text OR old.accession IS NOT new.accession
                BEGIN DELETE FROM embedding_window_sets WHERE chunk_id=new.id; END""",
            """CREATE TRIGGER IF NOT EXISTS window_source_document_update_v1 AFTER UPDATE ON documents
                WHEN old.title IS NOT new.title OR old.status IS NOT new.status
                BEGIN DELETE FROM embedding_window_sets WHERE chunk_id IN
                    (SELECT id FROM chunks WHERE accession=new.accession); END""",
        ]
        for operation, reference in (("insert", "new"), ("delete", "old"), ("update", "new")):
            statements.append(f"""CREATE TRIGGER IF NOT EXISTS window_vector_{operation}_v1
                AFTER {operation.upper()} ON embedding_windows BEGIN
                INSERT INTO window_generations VALUES({reference}.profile_id,1)
                ON CONFLICT(profile_id) DO UPDATE SET generation=generation+1;
                INSERT INTO window_changes VALUES({reference}.profile_id,{reference}.id,
                    (SELECT generation FROM window_generations WHERE profile_id={reference}.profile_id))
                ON CONFLICT(profile_id,window_id) DO UPDATE SET generation=excluded.generation;
                END""")
        # A reused prefix can appear, change, or disappear without a window-row write.
        for operation, references in (("insert", ("new",)), ("delete", ("old",)), ("update", ("old", "new"))):
            body = ""
            for reference in references:
                where = f"w.chunk_id={reference}.chunk_id AND p.model={reference}.model AND w.base_compatible=1 AND w.vector IS NULL"
                body += f"""
                    INSERT INTO window_generations(profile_id,generation)
                    SELECT DISTINCT w.profile_id,1 FROM embedding_windows w
                    JOIN embedding_window_profiles p ON p.id=w.profile_id WHERE {where}
                    ON CONFLICT(profile_id) DO UPDATE SET generation=generation+1;
                    INSERT INTO window_changes(profile_id,window_id,generation)
                    SELECT w.profile_id,w.id,g.generation FROM embedding_windows w
                    JOIN embedding_window_profiles p ON p.id=w.profile_id
                    JOIN window_generations g ON g.profile_id=w.profile_id WHERE {where}
                    ON CONFLICT(profile_id,window_id) DO UPDATE SET generation=excluded.generation;
                """
            statements.append(f"CREATE TRIGGER IF NOT EXISTS window_base_{operation}_v1 AFTER {operation.upper()} ON dense_vectors BEGIN {body} END")
        with atomic(self.db):
            for statement in statements:
                self.db.execute(statement)
            self.db.execute("INSERT OR IGNORE INTO embedding_window_profiles(model,policy,tokenizer_hash) VALUES(?,?,?)",
                            (MODEL, POLICY, FILES["tokenizer.json"]))

    def _source(self, chunk_id):
        if type(chunk_id) is not int or chunk_id < 1:
            raise ValueError("Use a positive passage ID")
        row = self.db.execute("""SELECT c.text,d.title,d.status FROM chunks c
            JOIN documents d ON d.accession=c.accession WHERE c.id=?""", (chunk_id,)).fetchone()
        if row is None or row["status"] != "indexed":
            raise StaleWindowInput("Window source is no longer indexed")
        return row["title"] + "\n" + row["text"]

    def plan(self, chunk_id):
        with read_snapshot(self.db):
            return self.planner.plan(self._source(chunk_id))

    def _check_source(self, chunk_id, plan):
        if text_hash(self._source(chunk_id)) != plan.source_hash:
            raise StaleWindowInput("Window source changed after planning or embedding")

    def register(self, chunk_id, plan):
        if not isinstance(plan, WindowPlan) or plan.model != MODEL or plan.policy != POLICY:
            raise ValueError("Unsupported window plan")
        source = self._source(chunk_id)
        if plan != self.planner.plan(source):
            raise StaleWindowInput("Window plan does not match the current source and tokenizer")
        serialized = canonical(plan.manifest())
        require_space(self.store.path.parent, extra_bytes=max(32 * 1024 * 1024, len(serialized.encode()) * 12))
        with atomic(self.db):
            self._check_source(chunk_id, plan)
            old = self.db.execute("SELECT plan_hash FROM embedding_window_sets WHERE chunk_id=? AND profile_id=?",
                                  (chunk_id, self.profile)).fetchone()
            fingerprint = bytes.fromhex(plan.fingerprint)
            if old and old[0] == fingerprint:
                # Do not silently repair a corrupt same-hash set or erase committed vectors.
                state = self._resolve(chunk_id)
                if state["state"] == "invalid_plan":
                    raise ValueError("Stored window plan failed integrity checks")
                return False
            self.db.execute("DELETE FROM embedding_window_sets WHERE chunk_id=? AND profile_id=?", (chunk_id, self.profile))
            self.db.execute("INSERT INTO embedding_window_sets VALUES(?,?,?,?,?,?,?)",
                (chunk_id, self.profile, bytes.fromhex(plan.source_hash), fingerprint,
                 plan.input_characters, plan.input_tokens, len(plan.windows)))
            self.db.executemany("""INSERT INTO embedding_windows
                (chunk_id,profile_id,ordinal,plan_hash,start_character,end_character,input_tokens,
                 input_hash,token_hash,base_compatible) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                [(chunk_id, self.profile, w.ordinal, fingerprint, w.start_character, w.end_character,
                  w.input_tokens, bytes.fromhex(w.input_hash), bytes.fromhex(w.token_hash), int(w.base_compatible))
                 for w in plan.windows])
        return True

    @staticmethod
    def _vector(value, dimension):
        import numpy as np
        if dimension != 384 or not isinstance(value, bytes) or len(value) != 1536:
            return None
        vector = np.frombuffer(value, dtype="<f4")
        if not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1) > .001:
            return None
        return value

    def _resolve(self, chunk_id):
        row = self.db.execute("SELECT * FROM embedding_window_sets WHERE chunk_id=? AND profile_id=?",
                              (chunk_id, self.profile)).fetchone()
        if row is None:
            return {"state": "no_plan", "windows": []}
        try:
            source = self._source(chunk_id)
        except StaleWindowInput:
            return {"state": "stale_source", "windows": []}
        if bytes.fromhex(text_hash(source)) != row["source_hash"]:
            return {"state": "stale_source", "windows": []}
        try:
            windows = self.db.execute("SELECT * FROM embedding_windows WHERE chunk_id=? AND profile_id=? ORDER BY ordinal",
                                      (chunk_id, self.profile)).fetchall()
            if len(windows) != row["expected_windows"] or not windows:
                raise ValueError
            specs = [{"ordinal": w["ordinal"], "start_character": w["start_character"],
                      "end_character": w["end_character"], "input_tokens": w["input_tokens"],
                      "input_hash": w["input_hash"].hex(), "token_hash": w["token_hash"].hex(),
                      "base_compatible": bool(w["base_compatible"])} for w in windows]
            manifest = {"model": MODEL, "policy": POLICY, "source_hash": row["source_hash"].hex(),
                        "input_characters": row["input_characters"], "input_tokens": row["input_tokens"],
                        "tokenizer_sha256": FILES["tokenizer.json"], "windows": specs}
            if (bytes.fromhex(text_hash(canonical(manifest))) != row["plan_hash"]
                    or row["input_characters"] != len(source)):
                raise ValueError
            covered = 0
            for number, (window, spec) in enumerate(zip(windows, specs), 1):
                start, end = spec["start_character"], spec["end_character"]
                if (window["plan_hash"] != row["plan_hash"]
                        or window["ordinal"] != number or spec["ordinal"] != number
                        or type(start) is not int or type(end) is not int or not 0 <= start <= covered <= end <= len(source)
                        or spec["input_hash"] != text_hash(source[start:end])
                        or type(spec["input_tokens"]) is not int or not 1 <= spec["input_tokens"] <= 512
                        or type(spec["base_compatible"]) is not bool
                        or bool(window["base_compatible"]) != spec["base_compatible"]
                        or (number > 1 and spec["base_compatible"])):
                    raise ValueError
                covered = end
            if covered != len(source):
                raise ValueError
        except (KeyError, TypeError, ValueError, AttributeError):
            return {"state": "invalid_plan", "windows": []}
        base = self.db.execute("SELECT * FROM dense_vectors WHERE chunk_id=? AND model=?", (chunk_id, MODEL)).fetchone()
        base_vector = self._vector(base["vector"], base["dimension"]) if base and base["content_hash"] == row["source_hash"].hex() else None
        resolved = []
        for window, spec in zip(windows, specs):
            vector, storage = None, "missing"
            if window["vector"] is not None:
                vector = self._vector(window["vector"], window["dimension"])
                storage = "stored" if vector is not None else "invalid"
            elif window["base_compatible"] and base_vector is not None:
                vector, storage = base_vector, "base"
            resolved.append({"id": window["id"], "ordinal": window["ordinal"], "spec": spec,
                             "storage": storage, "vector": vector})
        return {"state": "complete" if all(w["vector"] is not None for w in resolved) else "pending",
                "plan_hash": row["plan_hash"].hex(), "source_hash": row["source_hash"].hex(), "windows": resolved}

    def resolve(self, chunk_id):
        with read_snapshot(self.db):
            return self._resolve(chunk_id)

    def missing(self, chunk_id, plan):
        current = self.resolve(chunk_id)
        if current.get("plan_hash") != plan.fingerprint:
            raise StaleWindowInput("Window plan was replaced or invalidated")
        missing = {w["ordinal"] for w in current["windows"] if w["vector"] is None}
        return [w for w in plan.windows if w.ordinal in missing]

    def commit(self, chunk_id, plan, vectors):
        """Publish a bounded subset atomically, rechecking source identity after inference."""
        import numpy as np
        if (not isinstance(vectors, dict) or not 1 <= len(vectors) <= 128
                or any(type(n) is not int or not 1 <= n <= len(plan.windows) for n in vectors)):
            raise ValueError("Use 1–128 planned window vectors keyed by ordinal")
        encoded = {}
        for ordinal, vector in vectors.items():
            array = np.asarray(vector, dtype="<f4")
            if array.shape != (384,) or self._vector(array.tobytes(), 384) is None:
                raise ValueError("Window vectors must be finite normalized 384-dimensional values")
            encoded[ordinal] = array.tobytes()
        require_space(self.store.path.parent, extra_bytes=32 * 1024 * 1024 + len(encoded) * 1536)
        with atomic(self.db):
            self._check_source(chunk_id, plan)
            current = self._resolve(chunk_id)
            if current.get("plan_hash") != plan.fingerprint:
                raise StaleWindowInput("Window plan was replaced or invalidated during inference")
            written = 0
            for ordinal, value in encoded.items():
                window = current["windows"][ordinal - 1]
                if window["vector"] is not None:
                    continue  # Preserve durable or compatible existing embeddings on retries.
                self.db.execute("UPDATE embedding_windows SET dimension=384,vector=? WHERE id=?", (value, window["id"]))
                written += 1
        return written

    def generation(self):
        row = self.db.execute("SELECT generation FROM window_generations WHERE profile_id=?", (self.profile,)).fetchone()
        return row[0] if row else 0
