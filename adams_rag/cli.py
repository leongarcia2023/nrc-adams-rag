from __future__ import annotations

import argparse
import json
import sys

from .client import AdamsClient, BudgetStop
from .config import load_api_key
from .ingest import crawl, hydrate, import_jsonl, import_pdf, live_search
from .retrieval import Ollama, ask, embed_pending
from .store import Store


def parser():
    p = argparse.ArgumentParser(description="Resumable NRC ADAMS cache and cited local RAG prototype")
    p.add_argument("--db", default="data/adams.sqlite", help="Local SQLite database path")
    p.add_argument("--json", action="store_true", help="Print question results as structured JSON")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show indexed documents, gaps, and crawl status")
    export = commands.add_parser("export-document", help="Export a retained source version as Markdown")
    export.add_argument("accession")
    export.add_argument("--version")
    export.add_argument("--output", required=True)
    imp = commands.add_parser("import-jsonl", help="Import records with accession/title/content or BEIR _id/title/text")
    imp.add_argument("path")
    pdf = commands.add_parser("import-pdf", help="Extract a local PDF with physical page citations")
    pdf.add_argument("path")
    pdf.add_argument("--accession", required=True)
    pdf.add_argument("--title")
    pdf.add_argument("--url")
    pdf.add_argument("--docket", action="append")
    pdf.add_argument("--ocr", action="store_true", help="Use local macOS OCR on pages without embedded text")
    pdf.add_argument("--max-pages", type=int, default=50, help="Bound PDF processing (1–5,000 pages)")
    pdf.add_argument("--max-mib", type=int, default=25, help="Per-PDF file allowance (1–512 MiB)")
    pdf.add_argument("--max-new-ocr-pages", type=int, default=20, help="New OCR pages per document/run (1–500)")
    collect = commands.add_parser("crawl", help="Collect daily DateAddedTimestamp partitions; rerun to resume")
    collect.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    collect.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    collect.add_argument("--max-offset", type=int, default=1000, help="Local safety ceiling, NOT a verified APS cap")
    collect.add_argument("--refresh", action="store_true", help="Revisit dates; use for recent overlapping update windows")
    collect.add_argument("--docket")
    fetch = commands.add_parser("hydrate", help="Fetch document text for search excerpts and records missing text")
    fetch.add_argument("--limit", type=int, default=20)
    fetch.add_argument("--retry-missing", action="store_true", help="Retry records previously returned without text")
    live = commands.add_parser("live-search", help="Search all requested APS libraries now and cache the first result page")
    live.add_argument("query", help="APS keywords/Boolean query; no automatic question rewriting")
    live.add_argument("--docket")
    for cmd in (collect, fetch, live):
        cmd.add_argument("--max-requests", type=int, default=20, help="Per-run budget, including retries")
        cmd.add_argument("--rps", type=float, default=0.5, help="Local pacing setting; confirm your NRC allowance")
        cmd.add_argument("--daily-limit", type=int, default=5000, help="Local rolling 24-hour budget; NOT an NRC quota")
    for cmd in (collect, live):
        cmd.add_argument("--library", choices=["main", "legacy", "both"], default="both")
    emb = commands.add_parser("embed", help="Embed pending chunks using a locally installed Ollama model")
    emb.add_argument("--model", required=True)
    emb.add_argument("--limit", type=int, default=100)
    emb.add_argument("--batch-size", type=int, default=16)
    query = commands.add_parser("ask", help="Retrieve cited evidence; optionally generate an answer with local Ollama")
    query.add_argument("question")
    query.add_argument("--model", help="Installed Ollama chat model; omit for evidence-only search")
    query.add_argument("--embedding-model", help="Same installed model used by the embed command")
    query.add_argument("--docket")
    query.add_argument("--top-k", type=int, default=6)
    for name in ("research", "retrieve"):
        r = commands.add_parser(name, help="Full RAG pipeline" if name == "research" else "Retrieve reusable grounded context without generating an answer")
        r.add_argument("question")
        r.add_argument("--offline", action="store_true", help="Use only locally indexed text")
        r.add_argument("--model-path", default="data/models/answer")
        r.add_argument("--ollama-model", help="Use an installed local Ollama model instead of MLX")
        r.add_argument("--index-dir", default="data/index")
        r.add_argument("--rate-db", default="data/adams.sqlite")
        r.add_argument("--catalog", default="../adams-catalog/catalog.sqlite")
        r.add_argument("--docket")
        r.add_argument("--date-from")
        r.add_argument("--date-to")
        r.add_argument("--top-k", type=int, default=10)
        r.add_argument("--max-documents", type=int, default=8)
        r.add_argument("--no-semantic", action="store_true")
        r.add_argument("--output", help="Save the complete evidence package as JSON")
    source = commands.add_parser("add-regulatory-dataset", help="Fetch and import the public NRC regulatory dataset with provenance")
    source.add_argument("--directory", default="data/resources/nrc-regulatory")
    source.add_argument("--import-only", action="store_true")
    fermi = commands.add_parser("import-fermibench", help="Import an authorized local FermiBench copy")
    fermi.add_argument("directory")
    back = commands.add_parser("backfill", help="Resume bounded full-text ingestion from the completed catalog")
    back.add_argument("--catalog", default="../adams-catalog/catalog.sqlite")
    back.add_argument("--rate-db", default="data/adams.sqlite")
    back.add_argument("--library", choices=["main", "legacy"], default="main")
    back.add_argument("--max-documents", type=int, default=20)
    back.add_argument("--max-requests", type=int, default=25)
    back.add_argument("--rps", type=float, default=.5)
    back.add_argument("--daily-limit", type=int, default=5000)
    back.add_argument("--min-free-gb", type=float)
    model = commands.add_parser("setup-model", help="Download a pinned open local answer model")
    model.add_argument("--directory", default="data/models/answer")
    model.add_argument("--repo", default="mlx-community/Qwen3-4B-Instruct-2507-4bit")
    model.add_argument("--revision")
    index = commands.add_parser("index", help="Resume local embeddings and build the persistent ANN index")
    index.add_argument("--directory", default="data/index")
    index.add_argument("--limit", type=int, default=2000)
    identifiers = commands.add_parser("index-identifiers", help="Resume exact technical-identifier indexing for older collections")
    identifiers.add_argument("--limit", type=int, default=2000)
    evaluation = commands.add_parser("evaluate", help="Evaluate document retrieval against external relevance labels")
    evaluation.add_argument("--queries", required=True)
    evaluation.add_argument("--labels", required=True)
    evaluation.add_argument("--index-dir", default="data/index")
    evaluation.add_argument("--keyword-only", action="store_true")
    evaluation.add_argument("--output", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    store = Store(args.db)
    try:
        if args.command in {"crawl", "hydrate", "live-search"}:
            client = AdamsClient(store, load_api_key(), args.max_requests, args.rps, args.daily_limit)
        if args.command == "status":
            result = store.status()
        elif args.command == "export-document":
            from pathlib import Path
            from .export import document_markdown
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(document_markdown(store, args.accession, args.version))
            result = {"accession": args.accession, "output": str(path), "format": "markdown"}
        elif args.command == "import-jsonl":
            result = {"imported": import_jsonl(store, args.path), **store.status()}
        elif args.command == "import-pdf":
            from .ingest import PDFExtractionPending
            try:
                result = import_pdf(store, args.path, args.accession, args.title, args.url, args.docket,
                                    ocr=args.ocr, max_pages=args.max_pages,max_new_ocr_pages=args.max_new_ocr_pages,
                                    max_bytes=args.max_mib * 1024 * 1024)
            except PDFExtractionPending as error:
                result = {"state":"ocr_pending",**error.progress,"detail":str(error)}
        elif args.command == "crawl":
            result = crawl(store, client, args.start, args.end, args.library, args.docket, args.max_offset, args.refresh)
        elif args.command == "hydrate":
            result = hydrate(store, client, args.limit, args.retry_missing)
        elif args.command == "live-search":
            result = {"cached_results": live_search(store, client, args.query, args.library, args.docket), **store.status()}
        elif args.command == "embed":
            result = embed_pending(store, Ollama(), args.model, args.limit, args.batch_size)
        elif args.command == "ask":
            result = ask(store, args.question, args.model, args.embedding_model, docket=args.docket, top_k=args.top_k)
        elif args.command in {"research", "retrieve"}:
            from pathlib import Path
            from .ollama_engine import make_engine
            from .research import research
            from .semantic_index import search_index
            from .transport import PersistentOpener
            engine = make_engine(args.model_path, storage_path=store.path.parent, ollama_model=args.ollama_model)
            semantic = None if args.no_semantic else search_index(store, args.index_dir)
            rate, transport, live_client = None, None, None
            try:
                if not args.offline and load_api_key():
                    rate, transport = Store(args.rate_db), PersistentOpener()
                    live_client = AdamsClient(rate, load_api_key(), 30, 2, 90000, opener=transport, include_search_content=False)
                result = research(store, args.question, engine=engine, semantic=semantic, client=live_client,
                    live=not args.offline, catalog=args.catalog, docket=args.docket, date_from=args.date_from,
                    date_to=args.date_to, top_k=args.top_k, max_documents=args.max_documents,
                    generate=args.command == "research",
                    progress=lambda stage, detail: print(stage + ": " + detail, file=sys.stderr))
            finally:
                if transport:
                    transport.close()
                if rate:
                    rate.db.close()
            if args.output:
                Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "add-regulatory-dataset":
            from .resources import fetch_dataset, import_regulatory
            if not args.import_only:
                fetch_dataset(args.directory)
            result = import_regulatory(store, args.directory)
        elif args.command == "import-fermibench":
            from .resources import import_fermibench
            result = import_fermibench(store, args.directory)
        elif args.command == "backfill":
            from .backfill import backfill
            from .transport import PersistentOpener
            rate, transport = Store(args.rate_db), PersistentOpener()
            try:
                client = AdamsClient(rate, load_api_key(), args.max_requests, args.rps,
                                     args.daily_limit, opener=transport, include_search_content=False)
                result = backfill(store, client, args.catalog, args.max_documents, args.library,
                                  args.min_free_gb,
                                  progress=lambda accession, done, total: print(f"Reading {accession} ({done + 1}/{total})", file=sys.stderr))
            finally:
                rate.db.close()
                transport.close()
        elif args.command == "setup-model":
            from .resources import setup_model
            result = setup_model(args.directory, args.repo, args.revision)
        elif args.command == "index-identifiers":
            from .identifiers import sync_identifiers
            result = {"passages_indexed": sync_identifiers(store, args.limit)}
        elif args.command == "index":
            from .semantic_index import SemanticIndex
            result = SemanticIndex(store, args.directory).sync(args.limit,
                progress=lambda done, total: print(f"Embedded {done}/{total}", file=sys.stderr))
        elif args.command == "evaluate":
            from pathlib import Path
            from .evaluate import evaluate
            from .semantic_index import search_index
            result = evaluate(store, args.queries, args.labels,
                semantic=None if args.keyword_only else search_index(store, args.index_dir))
            Path(args.output).write_text(json.dumps(result, indent=2))
        if args.command == "ask" and not args.json:
            print(render_answer(result))
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    except BudgetStop as error:
        print(json.dumps({"paused": str(error), "status": store.status()}, indent=2))
    except (ValueError, RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
    finally:
        store.db.close()


def render_answer(result):
    lines = ["ADAMS local evidence", "", result["question"], ""]
    if result.get("answer"):
        lines.extend([result["answer"], ""])
    else:
        lines.extend(["Cited excerpts — no language model was used.", ""])
    for source in result["sources"]:
        location = f"PDF page {source['page']}" if source["page"] else f"text chunk {source['ordinal']} (page unknown)"
        lines.extend([f"[{source['citation']}] {source['accession']} · {location}",
                      source["title"], source["url"], "", source["text"], ""])
    coverage = result["coverage"]
    lines.append(f"Local cache: {coverage['documents'].get('indexed', 0)} indexed documents, {coverage['chunks']} chunks. Full archive coverage has not been verified.")
    lines.append(result["note"])
    return "\n".join(lines)
