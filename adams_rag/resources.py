"""Auditable reuse of public resources; no gated-resource access workaround."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse
from .storage import require_space

REGULATORY_DATASET = "davenporten/nrc-regulatory-embeddings"
FERMIBENCH = "atomic-canyon/FermiBench"


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def fetch_dataset(directory, repo=REGULATORY_DATASET):
    from huggingface_hub import HfApi, hf_hub_download
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=False).dataset_info(repo, files_metadata=True)
    if info.gated:
        raise RuntimeError(f"{repo} requires account access approval. Import an authorized local copy with import-fermibench; no access controls were bypassed.")
    files = [f.rfilename for f in info.siblings if f.rfilename.endswith((".parquet", "README.md", "LICENSE"))]
    required = sum(f.size or 0 for f in info.siblings if f.rfilename in files)
    require_space(directory, extra_bytes=required + 32 * 1024 * 1024)
    manifest = {"id": repo, "revision": info.sha, "url": f"https://huggingface.co/datasets/{repo}",
                "license": info.card_data.to_dict().get("license") if info.card_data else None, "files": [],
                "status": "community_text_unverified", "full_adams": False}
    for name in files:
        path = hf_hub_download(repo, name, repo_type="dataset", revision=info.sha,
                               local_dir=str(directory), token=False)
        manifest["files"].append({"path": name, "bytes": Path(path).stat().st_size, "sha256": file_hash(path)})
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def import_regulatory(store, directory):
    import pyarrow.parquet as pq
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["id"] != REGULATORY_DATASET:
        raise ValueError("Unexpected dataset manifest")
    groups, short, duplicates, seen = {}, 0, 0, set()
    for entry in manifest["files"]:
        if not entry["path"].endswith(".parquet"):
            continue
        path = directory / entry["path"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Dataset checksum mismatch")
        source = pq.ParquetFile(path)
        columns = [name for name in source.schema_arrow.names if name != "embedding"]
        for batch in source.iter_batches(batch_size=512, columns=columns):
            for row in batch.to_pylist():
                text = row.get("text") or ""
                if len(text.strip()) < 80:
                    short += 1
                    continue
                identity = (row.get("source_url"), re.sub(r"\s+", " ", text).strip())
                if identity in seen:
                    duplicates += 1
                    continue
                seen.add(identity)
                groups.setdefault(row["document_id"], []).append(row)
    chunks, imported = 0, 0
    for doc_id, rows in groups.items():
        rows.sort(key=lambda r: r.get("chunk_index", 0))
        first = rows[0]
        url = first.get("source_url") or ""
        if urlparse(url).scheme != "https" or urlparse(url).hostname not in {"nrc.gov", "www.nrc.gov", "ecfr.gov", "www.ecfr.gov"}:
            continue
        accession = re.search(r"ML\d{5}[A-Za-z0-9]\d{3}", url, re.I)
        document = {"AccessionNumber": "regulations:" + doc_id,
                    "SourceAccession": accession.group(0).upper() if accession else first.get("guide_id") or first.get("section_id"),
                    "DocumentTitle": first.get("title") or first.get("document") or "Regulatory text",
                    "DocumentDate": str(first.get("revision_date") or "") or None,
                    "Url": url, "content": "\n\n".join(r["text"] for r in rows),
                    "DocumentType": first.get("document_type"), "section_id": first.get("section_id"),
                    "Dataset": manifest["id"], "DatasetRevision": manifest["revision"],
                    "QualityNote": "Community extraction; completeness and current applicability are unverified.",
                    "SourceChunkIds": [r["id"] for r in rows]}
        with store.db:
            store.put(document, "community_regulatory_text")
        imported += 1
        chunks += len(rows)
    manifest.update(imported_documents=imported, accepted_source_chunks=chunks,
                    skipped_short_chunks=short, skipped_duplicate_chunks=duplicates,
                    embeddings_reused=False, reason="Re-embed locally; supplied vectors require a different query model.")
    with store.db:
        store.db.execute("INSERT OR REPLACE INTO datasets(id,manifest) VALUES(?,?)", (manifest["id"], json.dumps(manifest)))
    return manifest


def import_fermibench(store, directory):
    directory = Path(directory)
    count = 0
    with (directory / "corpus.jsonl").open() as f:
        for line in f:
            doc = json.loads(line)
            url = doc["_id"]
            if url.startswith("www.nrc.gov/"):
                url = "https://" + url
            if urlparse(url).hostname not in {"www.nrc.gov", "nrc.gov"} or urlparse(url).scheme != "https":
                raise ValueError("FermiBench record does not have an official NRC source URL")
            match = re.search(r"ML\d{5}[A-Za-z0-9]\d{3}", url, re.I)
            if not match:
                raise ValueError("FermiBench record lacks a source accession")
            with store.db:
                store.put({"AccessionNumber": match.group(0).upper(), "DocumentTitle": doc["title"],
                           "content": doc["text"], "Url": url, "BenchmarkId": doc["_id"], "Dataset": FERMIBENCH}, "fermibench")
            count += 1
    manifest = {"id": FERMIBENCH, "license": "apache-2.0", "imported_documents": count,
                "sha256": file_hash(directory / "corpus.jsonl"), "full_adams": False}
    with store.db:
        store.db.execute("INSERT OR REPLACE INTO datasets(id,manifest) VALUES(?,?)", (FERMIBENCH, json.dumps(manifest)))
    return manifest


def setup_model(directory, repo, revision=None):
    from huggingface_hub import HfApi, hf_hub_download
    from .download import ranged_download
    import sys
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    from .models import model_ready
    if model_ready(directory):
        installed = json.loads((directory / "resource-manifest.json").read_text())
        if installed.get("id") == repo and (not revision or installed.get("revision") == revision):
            if all(file_hash(directory / f["path"]) == f["sha256"] for f in installed["files"]):
                return installed
    info = HfApi(token=False).model_info(repo, revision=revision, files_metadata=True)
    if info.gated:
        raise RuntimeError("Model access requires account approval")
    require_space(directory, extra_bytes=3_000_000_000)
    files = []
    for entry in info.siblings:
        name = entry.rfilename
        if not name.endswith((".safetensors", ".json", ".txt", ".jinja")) and name not in {"LICENSE", "README.md"}:
            continue
        path = directory / name
        if entry.size and entry.size > 64 * 1024 * 1024 and entry.lfs:
            ranged_download(f"https://huggingface.co/{repo}/resolve/{info.sha}/{name}", path,
                            entry.size, entry.lfs.sha256,
                            progress=lambda done, total: print(f"Model download: {done / total:.0%}", file=sys.stderr, flush=True))
        else:
            hf_hub_download(repo, name, revision=info.sha, local_dir=str(directory), token=False)
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": file_hash(path)})
    manifest = {"id": repo, "revision": info.sha, "license": info.card_data.to_dict().get("license") if info.card_data else None,
                "local_only": True, "files": files}
    (directory / "resource-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
