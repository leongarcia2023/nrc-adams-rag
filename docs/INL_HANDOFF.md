# INL handoff

The repository can start from an empty collection. An existing project snapshot
is stored separately in the owner's Stanford Google Drive. The public repository
does not grant access to that Drive folder or contain its account credentials.
Arrange archive access directly with the owner; cloning the code does not fetch
the archive automatically.

## Restore the existing snapshot

The November 8, 2026 snapshot contains approximately 27.5 GB of project data,
including the catalog, research database, Markdown archives, retained PDFs,
indexes, and model resources. It is not the entire ADAMS full-text archive.

Obtain these records with the archive: `RESTORE.txt`, `FINAL-REPORT.json`,
`VERIFIED-ARCHIVE.json`, `ARCHIVE-MANIFEST.json`, and
`verified-removals.jsonl`. They identify the transferred objects and checksums.
The large tar is only part of the snapshot: the research database, one PDF,
and one model-weight file were transferred individually.

1. Keep all app and worker processes stopped. Allow roughly 45 GB for a
   conventional download-and-extract restore, plus working headroom.
2. Download the tar and individual files. Verify their SHA-256 checksums against
   the transfer receipts. Extract into a new staging directory, preserving the
   recorded relative paths. Check the member hashes against the manifest.
3. Restore all separately transferred files into their recorded paths. The
   expected layout under staging is `adams-rag/data/` and `adams-catalog/`.
4. Move the restored `adams-rag/data/` into this checkout as `data/` and
   `adams-catalog/` as `data/catalog/`. Do not merge with an unrelated collection.
   Retained Markdown references resolve relative to the research database;
   keep `data/markdown/` beside `data/research.sqlite`.
5. Use `--catalog data/catalog/catalog.sqlite` for the server and relevant
   commands. Defaults otherwise refer to the older sibling catalog layout.
6. Check the database, source access, and index configuration before resuming
   acquisition. If offload markers were copied separately, remove them only
   after verifying the restoration. This code repository includes no such marker.

The transferred embedding configuration may select the MLX profile and window
index. That profile requires Apple silicon. On Linux, retain those files as part
of the backup, use `--no-semantic` for an initial CLI retrieval check, and build
the default CPU profile in a **new** index directory:

```sh
python -m adams_rag.index_worker --db data/research.sqlite \
  --directory data/index-cpu --backend cpu
python -m adams_rag.server --catalog data/catalog/catalog.sqlite \
  --index-dir data/index-cpu --ollama-model "$NRC_OLLAMA_MODEL"
```

The new directory selects the CPU model without changing or mislabelling the
transferred MLX vectors. This rebuild takes additional space and time. A moved
catalog can also start a new acquisition cursor because cursor identity includes
its path; already indexed documents are skipped rather than duplicated.

## Check the installation

Run the synthetic smoke test, then the unit/integration suite described in the
README. Verify a real document's title, source URL, version, text, and citations
against its original before reviewing broader questions. Test the selected
answer model using examples with missing evidence and conflicting or historical
records, as well as straightforward questions.

The Ollama adapter is new in this handoff. Its local transport, pipeline
integration and citation rejection are tested; performance and answer quality
with INL's chosen model and hardware still need a local evaluation. The existing
MLX path and Apple OCR remain available on supported Macs.

The server has no multi-user authentication or production hosting layer. It
binds to loopback for a single workstation. For a remote workstation, use an
approved SSH tunnel; shared deployment needs an authenticated service wrapper
and an explicit concurrency/storage design.
