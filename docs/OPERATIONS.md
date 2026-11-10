# Collection operations

Run these commands from the repository with its Python environment active.
Use one acquisition host per subscription and a shared local request ledger.
The ledger coordinates processes on that host, not different machines. Adding
workers on other hosts does not multiply the NRC subscription's allowance.

## Inventory the public catalog

```sh
python -m adams_rag.catalog \
  --directory data/catalog --rate-db data/adams.sqlite \
  --requests 5000 --rps 2 --daily-limit 90000 --prefetch
```

Rerun the same command to resume after its request budget. The collector splits
accession-prefix partitions, includes complementary branches and undated
records, and reconciles returned IDs with API counts for both public libraries.
Do not change the catalog path between resumptions.

```sh
python -m adams_rag.catalog --directory data/catalog \
  --rate-db data/adams.sqlite --status
```

All request and daily budgets in these examples are local pacing settings,
not a statement of NRC's quota. NRC support advised the project that the
standard API allowance is 10 requests/second and that no bulk export is offered.
The continuous text worker currently caps its setting at 6 requests/second.
Server cooldowns and retry/backoff still apply. Start at 2, measure, and stay
within the allowance for the subscription being used.

## Acquire full text and build the index

Once catalog enumeration and reconciliation complete:

```sh
python -m adams_rag.archive_worker \
  --db data/research.sqlite --catalog data/catalog/catalog.sqlite \
  --rate-db data/adams.sqlite --directory data/archive \
  --library main --batch-size 200 --rps 2 --daily-limit 90000 \
  --prefetch --prefetch-depth 2
```

In a second terminal, using the same environment and storage setting:

```sh
python -m adams_rag.index_worker --db data/research.sqlite \
  --directory data/index --backend cpu --follow
```

On Apple silicon, `--backend mlx` is available as a separate embedding profile.
Do not relabel vectors from one model as vectors from another. Profile activation
and optional window indexing are separate operations. The older `embed` / `ask`
Ollama vector-scan commands are for small pilots (50,000-vector ceiling); use
`index_worker` and `research` for the larger ANN-backed collection.

Acquisition and indexing have separate checkpoints and locks. The downloader
records missing-text outcomes; it does not silently turn snippets into source
text. `--library legacy` handles the legacy catalog but may produce metadata-only
records. PDF recovery and OCR are separate stages, not guaranteed recovery of
every legacy record.

Graceful stop files:

```sh
touch data/archive/STOP
touch data/index/STOP
```

Wait for workers to exit. Remove only the corresponding STOP file when ready to
resume. Progress is recorded in `data/archive/STATUS.json` and
`data/index/BUILD_STATUS.json`. Workers do not install background services or
automatically restart after reboot.

## Markdown and extraction quality

```sh
python -m adams_rag.text_quality --db data/research.sqlite --limit 1000
python -m adams_rag.markdown_archive \
  --db data/research.sqlite --directory data/markdown
```

Repeat bounded assessments until caught up. Markdown is stored in compressed
archives with source URLs, accessions, retained versions and integrity checks.
Individual documents can be exported with `export-document`.

`python -m adams_rag.markdown_migrate --db data/research.sqlite` is an optional
space-saving migration: it replaces covered full-text payloads with references
to verified Markdown and removes covered raw caches. Keep a backup before using
it. Afterwards, `research.sqlite` and `data/markdown/` must move together. Do not
delete PDFs merely because Markdown exists; extraction integrity and fidelity
to the original are separate checks. The source-review modules retain exclusions
and version-bound page corrections.

## Ongoing additions

```sh
python -m adams_rag.recent --start 2026-11-01 --end 2026-11-02 \
  --directory data/updates/2026-11-01_to_2026-11-02 \
  --rate-db data/adams.sqlite --requests 20
```

Choose the desired dates. This collects metadata using Date Added, with bounded
1–31-day windows and resumable partitions. After a supplement is reconciled,
run `backfill` against its `catalog.sqlite`. Use overlapping windows for delayed
publication and deduplicate by source identity; do not add overlapping raw counts.
Date Added alone does not detect all historical edits or withdrawals.

## Storage and scaling

Use a local filesystem for the live SQLite database and request ledger. Do not
run the database from a Google Drive sync folder or share its WAL files among
machines. Keep acquisition coordinated; use separate databases for independent
deployments. A distributed ingestion queue and shared production vector service
are not implemented by this repository.

Use `ADAMS_MIN_FREE_GB` consistently across workers. Set model and dataset caches
on the intended volume as well. Ollama's own storage and memory are managed by
Ollama, outside the application's storage guard. Observe actual disk growth and
throughput before extrapolating to all ADAMS records.
