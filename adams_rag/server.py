"""Loopback-only RAG API. One inference/ingestion job at a time on a personal Mac."""
from __future__ import annotations

from .source_content import read_text
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import uuid
from urllib.parse import parse_qs, urlsplit

from .client import AdamsClient
from .config import load_api_key
from .models import model_ready
from .ollama_engine import make_engine
from .research import catalog_coverage, research
from .semantic_index import SemanticIndex, active_model, search_index, window_search_enabled
from .store import Store
from .transport import PersistentOpener


class Application:
    def __init__(self, db, rate_db, catalog, model_path, index_dir, ollama_model=None):
        self.db, self.rate_db, self.catalog = Path(db), Path(rate_db), Path(catalog)
        self.index_dir, self.model_path = Path(index_dir), Path(model_path)
        self.engine = make_engine(self.model_path, storage_path=self.db.parent, ollama_model=ollama_model)
        self.encoder = None
        self.encoder_model = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.jobs = {}

    def status(self):
        store = Store(self.db)
        try:
            generation = self.engine.availability() if self.engine else {"state": "model_unavailable"}
            return {"collection": store.status(), "catalog": catalog_coverage(self.catalog),
                    "semantic_retrieval_model": active_model(self.index_dir),
                    "semantic_window_search_enabled": window_search_enabled(self.index_dir),
                    "local_model_installed": generation["state"] != "model_unavailable",
                    "generation": generation,
                    "adams_connected": bool(load_api_key()), "preparation": self.preparation(),
                    "capabilities": ["ingestion", "version_history", "catalog_lookup", "legacy_digitization_links", "keyword_retrieval", "identifier_retrieval", "semantic_retrieval",
                                     "live_adams_discovery", "package_members", "context_assembly", "cited_generation", "evidence_export"],
                    "limitations": ["Base catalog counts reconciled at recorded dates; newer observations are separate and full-text coverage is partial.",
                                    "Retrieval is bounded and cannot establish exhaustive answers or current regulatory applicability.",
                                    "Citation identifiers are validated; source entailment is not automatically proven."]}
        finally:
            store.db.close()

    def preparation(self):
        result = {}
        try:
            value = json.loads((self.db.parent / 'markdown' / 'STATUS.json').read_text())
            if isinstance(value, dict):
                result['markdown'] = {key: value.get(key) for key in ('state', 'counts', 'updated_utc', 'detail',
                    'source_fidelity_verified_documents', 'complete_ocr_verification')}
        except (OSError, ValueError):
            pass
        try:
            value = json.loads((self.db.parent / "archive" / "STATUS.json").read_text())
            result["archive"] = {key: value.get(key) for key in (
                "state", "library", "updated_utc", "detail", "retry_at_utc", "new_documents_this_run",
                "free_bytes", "reserve_bytes", "coverage", "cursors")}
        except (OSError, ValueError):
            pass
        try:
            value = json.loads((self.db.parent / "recovery" / "STATUS.json").read_text())
            result["pdf_recovery"] = {key: value.get(key) for key in (
                "state", "pid", "updated_utc", "detail", "current_document", "attempted",
                "recovered", "free_bytes", "reserve_bytes")}
        except (OSError, ValueError):
            pass
        progress = self.index_dir / "BUILD_STATUS.json"
        try:
            value = json.loads(progress.read_text())
            result["indexing"] = {key: value.get(key) for key in ("state", "vectors", "chunks", "updated_utc", "detail", "model", "backend", "active_profile_coverage", "window_backfill")}
        except (OSError, ValueError):
            pass
        try:
            value = json.loads((self.index_dir / "windows" / "STATUS.json").read_text())
            if isinstance(value, dict):
                result["windows"] = {key: value.get(key) for key in (
                    "state", "updated_utc", "detail", "model", "policy", "mode",
                    "planned_passages", "completed_passages", "stored_window_vectors")}
        except (OSError, ValueError):
            pass
        if not model_ready(self.model_path):
            for path in self.model_path.glob("*.progress.json"):
                try:
                    value = json.loads(path.read_text())
                    done = sum(min(value["block_size"], value["size"] - offset) for offset in value["completed"])
                    result["model_download"] = {"percent": round(done / value["size"] * 100)}
                except (OSError, ValueError, KeyError, TypeError):
                    pass
        return result

    def submit(self, payload, generate=True):
        allowed = {"question", "live", "docket", "date_from", "date_to", "top_k", "max_documents"}
        if set(payload) - allowed:
            raise ValueError("Unknown request fields")
        if not isinstance(payload.get("question"), str) or not 1 <= len(payload["question"].strip()) <= 2000:
            raise ValueError("Enter a question between 1 and 2,000 characters")
        if "live" in payload and not isinstance(payload["live"], bool):
            raise ValueError("live must be true or false")
        from .research import validate_filters
        validate_filters(payload.get("docket"), payload.get("date_from"), payload.get("date_to"))
        for key, default in [("top_k", 10), ("max_documents", 8)]:
            value = payload.get(key, default)
            if type(value) is not int or not 1 <= value <= 20:
                raise ValueError(f"{key} must be between 1 and 20")
        with self.lock:
            if any(j["state"] in {"queued", "running"} for j in self.jobs.values()):
                raise RuntimeError("A research run is already active. Wait for it to finish before submitting another.")
            job_id = str(uuid.uuid4())
            self.jobs[job_id] = {"id": job_id, "state": "queued", "stage": "Queued", "detail": ""}
            if len(self.jobs) > 30:
                del self.jobs[next(iter(self.jobs))]
        self.pool.submit(self.run, job_id, payload, generate)
        return self.jobs[job_id].copy()

    def run(self, job_id, payload, generate):
        store, rate, transport = None, None, None
        def progress(stage, detail):
            with self.lock:
                self.jobs[job_id].update(state="running", stage=stage, detail=detail)
        try:
            progress("Starting retrieval", "Preparing the local collection")
            if self.engine is None and model_ready(self.model_path):
                self.engine = MLXEngine(self.model_path, storage_path=self.db.parent)
            store = Store(self.db)
            semantic = search_index(store, self.index_dir)
            if self.encoder_model == semantic.model:
                semantic.encoder = self.encoder
            client = None
            key = load_api_key()
            if key and payload.get("live", True):
                rate = Store(self.rate_db)
                transport = PersistentOpener()
                client = AdamsClient(rate, key, max_requests=30, rps=2, daily_limit=90000,
                                     opener=transport, include_search_content=False)
            result = research(store, engine=self.engine, semantic=semantic, client=client,
                              catalog=self.catalog, generate=generate, progress=progress, **payload)
            self.encoder = semantic.encoder
            self.encoder_model = semantic.model
            with self.lock:
                self.jobs[job_id].update(state="finished", stage="Finished", detail="", result=result)
        except Exception as error:
            # Expected errors have user-facing messages. Unknown dependency failures
            # expose only their class, never arbitrary exception bodies/credentials.
            message = str(error) if isinstance(error, (ValueError, RuntimeError)) else f"Run failed ({type(error).__name__}); saved records are retained."
            with self.lock:
                self.jobs[job_id].update(state="failed", stage="Run failed", detail=message)
        finally:
            if transport:
                transport.close()
            if store:
                store.db.close()
            if rate:
                rate.db.close()


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # Query text and source content do not enter access logs.

        def valid_host(self):
            return self.headers.get("Host") in {f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}"}

        def send(self, value, status=200, content_type="application/json; charset=utf-8"):
            data = json.dumps(value, ensure_ascii=False).encode() if content_type.startswith("application/json") else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.valid_host():
                return self.send({"error": "Invalid local host"}, 403)
            route = urlsplit(self.path)
            try:
                if route.path == "/api/status":
                    return self.send(app.status())
                if route.path in {"/v1/catalog/legacy-links", "/api/catalog/legacy-links"}:
                    from .legacy_links import lookup
                    query = parse_qs(route.query, keep_blank_values=True)
                    if set(query) - {"accession", "limit"} or any(len(v) != 1 for v in query.values()):
                        raise ValueError("Unknown or repeated legacy-link parameters")
                    return self.send(lookup(app.catalog, query.get("accession", [None])[0],
                                            limit=int(query.get("limit", ["20"])[0])))
                if route.path in {"/v1/catalog", "/api/catalog"}:
                    from .catalog_collection import lookup_available
                    query = parse_qs(route.query, keep_blank_values=True)
                    allowed = {"accession", "library", "record_id", "limit", "cursor"}
                    if set(query) - allowed or any(len(values) != 1 for values in query.values()):
                        raise ValueError("Unknown or repeated catalog lookup parameters")
                    values = {key: val[0] for key, val in query.items()}
                    if "limit" in values:
                        values["limit"] = int(values["limit"])
                    return self.send(lookup_available(app.catalog, app.db.parent / "updates", **values))
                if route.path.startswith("/api/jobs/"):
                    with app.lock:
                        job = app.jobs.get(route.path.rsplit("/", 1)[-1])
                        return self.send(job.copy() if job else {"error": "Unknown job"}, 200 if job else 404)
                if route.path == "/api/history":
                    store = Store(app.db)
                    try:
                        exists = store.db.execute("SELECT 1 FROM sqlite_master WHERE name='research_runs'").fetchone()
                        rows = store.db.execute("SELECT id,created,json_extract(result,'$.question') AS question FROM research_runs ORDER BY created DESC LIMIT 20").fetchall() if exists else []
                        return self.send([dict(r) for r in rows])
                    finally:
                        store.db.close()
                if route.path.startswith("/api/runs/"):
                    store = Store(app.db)
                    try:
                        from .evidence import saved_run_view
                        result = saved_run_view(store,route.path.rsplit("/", 1)[-1])
                        return self.send(result if result else {"error": "Unknown run"}, 200 if result else 404)
                    finally:
                        store.db.close()
                if route.path == "/api/document/version":
                    query = parse_qs(route.query)
                    store = Store(app.db)
                    try:
                        row = store.db.execute("SELECT * FROM document_versions WHERE accession=? AND version=?",
                            (query.get("accession", [""])[0], query.get("version", [""])[0])).fetchone()
                        if not row:
                            return self.send({"error": "Unknown document version"}, 404)
                        value = dict(row)
                        value["content"] = read_text(value["content"], store.db)
                        value["metadata"] = json.loads(value["metadata"])
                        from .text_quality import assess
                        from .page_boundaries import retained_pdf_pages
                        value["text_quality"] = assess(value["content"], value["metadata"], value["provenance"], "indexed",
                            retained_pdf_pages(value["content"], value["metadata"]) if value["provenance"] == "pdf" else None)
                        from .extraction_review import annotate
                        value["text_quality"] = annotate(value["text_quality"],store.db,value["accession"],value["text_hash"])
                        return self.send(value)
                    finally:
                        store.db.close()
                if route.path == "/api/document/markdown":
                    from .export import document_markdown
                    query = parse_qs(route.query)
                    accession = query.get("accession", [""])[0]
                    store = Store(app.db)
                    try:
                        return self.send(document_markdown(store, accession, query.get('version', [None])[0]).encode(), content_type="text/markdown; charset=utf-8")
                    finally:
                        store.db.close()
                if route.path in ('/api/markdown/sources.csv', '/api/markdown/holdouts.csv'):
                    path = app.db.parent / 'markdown' / route.path.rsplit('/', 1)[-1]
                    if not path.is_file():
                        return self.send({'error': 'The source index is still being prepared'}, 404)
                    return self.send(path.read_bytes(), content_type='text/csv; charset=utf-8')
                if route.path == "/api/document":
                    accession = parse_qs(route.query).get("accession", [""])[0]
                    store = Store(app.db)
                    try:
                        row = store.db.execute("SELECT * FROM documents WHERE accession=?", (accession,)).fetchone()
                        if not row:
                            return self.send({"error": "Document is not in the local collection"}, 404)
                        from .text_quality import get as text_quality
                        check = store.db.execute("SELECT 1 FROM sqlite_master WHERE name='markdown_checks'").fetchone()
                        check = store.db.execute("SELECT state,report,archive,member,checked FROM markdown_checks WHERE accession=?", (accession,)).fetchone() if check else None
                        return self.send({"document": dict(row),
                            "markdown_check": dict(check) if check else None,
                            "text_quality": text_quality(store.db, accession),
                            "chunks": [dict(r) for r in store.db.execute("SELECT ordinal,page,text FROM chunks WHERE accession=? ORDER BY ordinal", (accession,))],
                            "versions": [dict(r) for r in store.db.execute("SELECT version,text_hash,provenance,captured FROM document_versions WHERE accession=? ORDER BY captured DESC", (accession,))]})
                    finally:
                        store.db.close()
                assets = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/style.css": ("style.css", "text/css; charset=utf-8")}
                for font in ("cmu-serif-500-roman.woff2", "cmu-serif-700-roman.woff2", "cmu-serif-500-italic.woff2"):
                    assets["/fonts/" + font] = ("fonts/" + font, "font/woff2")
                if route.path in assets:
                    name, mime = assets[route.path]
                    path = Path(__file__).parent / "web" / name
                    if path.exists():
                        return self.send(path.read_bytes(), content_type=mime)
                return self.send({"error": "Not found"}, 404)
            except (ValueError, RuntimeError) as error:
                return self.send({"error": str(error)}, 400)
            except Exception:
                return self.send({"error": "Unable to read local state"}, 500)

        def do_POST(self):
            origins = {f"http://localhost:{self.server.server_port}", f"http://127.0.0.1:{self.server.server_port}"}
            if not self.valid_host() or (self.headers.get("Origin") and self.headers["Origin"] not in origins):
                return self.send({"error": "Only same-origin local requests are accepted"}, 403)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send({"error": "Use application/json"}, 415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 16384:
                    return self.send({"error": "Request size must be between 1 and 16,384 bytes"}, 413)
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                if self.path not in {"/api/research", "/v1/retrieve", "/v1/rag"}:
                    return self.send({"error": "Not found"}, 404)
                return self.send(app.submit(payload, generate=self.path != "/v1/retrieve"), 202)
            except (ValueError, TypeError) as error:
                return self.send({"error": str(error)}, 400)
            except RuntimeError as error:
                return self.send({"error": str(error)}, 409)
    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description="Local NRC RAG API and research workspace")
    p.add_argument("--db", default="data/research.sqlite")
    p.add_argument("--rate-db", default="data/adams.sqlite")
    p.add_argument("--catalog", default="../adams-catalog/catalog.sqlite")
    p.add_argument("--model-path", default="data/models/answer")
    p.add_argument("--ollama-model", help="Use an installed local Ollama model instead of MLX")
    p.add_argument("--index-dir", default="data/index")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)
    app = Application(args.db, args.rate_db, args.catalog, args.model_path, args.index_dir, args.ollama_model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(app))
    print(f"NRC RAG listening at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
