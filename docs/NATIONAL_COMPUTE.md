# National Compute deployment

National Compute is the target hosting platform for this project. The deployment
constraint is no personal out-of-pocket spending: production compute, storage and
model serving must fit confirmed included resources or a funded organization
allocation. Membership access alone does not establish resource funding. Confirm
the allocation's duration, service restrictions and overage treatment before
provisioning permanent capacity. Do not enable external model subscriptions or
automatic credit purchases as part of this deployment.

The model API can be an Ollama server running on National Compute itself. The
existing adapter already supports this over a private loopback SSH tunnel; no
third-party model API subscription is necessary. The CPU workspace is a prototype,
not a verified always-on public hosting service. Permanent industry access still
needs funded persistent hosting, storage, a public HTTPS endpoint, authentication,
monitoring, and an adequately sized model service on National Compute.

## Enablement request for the National Compute administrator

Provisioning, resource funding and application deployment are separate steps.
The following are requested capabilities, not claims that these resources already
exist or are included in Pro. The capacity figures are starting targets for a pilot,
not a measured full-archive requirement or a National Compute SKU.

| Enable | Initial target and acceptance condition |
| --- | --- |
| Sponsored project organization | Attribute CPU, model serving, storage, network and any control-plane charges to the sponsor. Confirm the coverage period and renewal before expiry. No personal card charges, credit advances, personal debt or automatic purchases on exhaustion; uncovered usage must be refused or paused. |
| Permanent application hosting | Provision a managed Kubernetes cluster using the organization's grandfathered entitlement, or an equivalent supported permanent host. Target at least 4 CPU cores and 16 GiB RAM for the initial collection. It must run with the browser closed, restart workloads after host maintenance, and avoid idle suspension. Confirm actual CPU-worker size: NC's standard worker cannot be resized. |
| Durable writable disk | Start with 250 GiB of persistent SSD for the current database/index and expansion, with a funded path to TiB-scale storage as the corpus grows. The current SQLite WAL database needs a supported local/block filesystem on one host; do not put it on the shared NFS volume. The disk must survive process and compute replacement. |
| Archive and backup storage | Provision the organization's private S3 bucket (initial 2 TiB soft cap, expandable) or equivalent sponsored durable store. Resolve the Public Research organization restriction explicitly, potentially by using a sponsor's managed project organization. Preserve timestamped, consistent backups separately: NC buckets currently have no automatic versioning or backups. |
| Model API | Prefer an existing sponsored, private inference endpoint with a suitable open model and documented concurrency/usage allowance. An Ollama endpoint is already supported through a loopback SSH tunnel. Otherwise allocate reserved model-serving capacity, potentially one GPU or a supported GPU share, and benchmark model size/quality before selecting it. This must be funded for the service lifetime; a preemptible four-hour Public Research rental is not a permanent endpoint. Marshall's interactive model allowance alone is not an app API. |
| Industry access | Enable an external load balancer in a supported region, a stable HTTPS hostname, certificate renewal and restricted administration. Keep inference and storage private; only expose the authenticated web application. Confirm network/egress charges are covered. |

If a sub-node GPU allocation is used, NC must enable its supported Base Load
packing and provide the funded reserved capacity; ordinary market jobs request
whole nodes. Do not submit a GPU job merely to discover its cost.

The application still needs operational work before broad public access: individual
accounts and quotas, durable request handling, monitored restart and restore tests,
and source-reviewed NRC answer evaluation. Resource provisioning does not certify
answer quality or complete the archive. Retain the existing private workspace while
those checks run.

Platform references, checked November 2026:
[managed CPU workers and networking](https://docs.nationalcompute.com/kubernetes/),
[bucket eligibility and durability limits](https://docs.nationalcompute.com/object-storage/),
[funding scopes, expiry and unpaid usage](https://docs.nationalcompute.com/credits/),
[SQLite WAL filesystem requirements](https://sqlite.org/wal.html).

The app runs in the persistent Marshall workspace. Its database, Markdown archives,
request ledger, and checkpoints stay together on that volume. A rented GPU can
serve the answer model over an SSH tunnel; it does not need the archive or NRC key.

The November 2026 deployment restored a 27.5 GB snapshot containing 139,041 indexed
documents and 1,990,929 chunks. The catalog contains about 5.46 million records.
These are different coverage measures: the full catalog is not full-text coverage.
Only part of the chunk collection has semantic vectors; keyword retrieval covers
the indexed text. The restored MLX vectors remain available for Apple deployments,
while Linux uses its separate compatible CPU embedding profile.

## Persistent layout

| Path under `/data/workspace` | Contents |
| --- | --- |
| `nrc-adams-rag` | Public code and Python environment |
| `nrc-snapshot/adams-rag/data` | Research database, Markdown, acquisition and index state |
| `nrc-snapshot/adams-catalog/catalog.sqlite` | Reconciled catalog |
| `nrc-runtime` | Ollama runtime, models, service logs and local configuration |
| `nrc-runtime/private/adams-api.key` | Owner-only NRC credential; excluded from Git |

The observed workspace has 50 GiB of persistent storage, about 7 GiB of memory,
and a 1.5-CPU quota. Pro membership does not provide enough storage for the entire
ADAMS archive. Services default to a 5 GB free-space reserve. Expand storage before
attempting the full collection; a GPU node's temporary disk is not an archive backup.

### Expanding storage on the existing plan

The included workspace disk and a site's shared volume are separate resources.
The console does not offer a workspace-disk resize control. Ask NC support to
expand the existing persistent disk to 250 GiB initially, preserving its contents,
or provide an additional persistent block disk accessible from this workspace.
Request the monthly storage price and whether the current plan or sponsor covers
it before authorizing a charge. This is an initial growth allowance, not a claim
that 250 GiB holds the complete archive and index.

Pro's site storage is provisioned on request. Buying a higher membership level
is not the only question to ask, and access to a resource does not establish that
its usage is included. NC's documented object-storage bucket excludes Public
Research organizations; resolving that requires NC. See the current
[membership](https://docs.nationalcompute.com/membership/),
[storage](https://docs.nationalcompute.com/object-storage/) and
[support](https://docs.nationalcompute.com/support/) documentation.

An in-place disk expansion preserves the existing paths. If NC attaches another
disk, stop writers, copy and verify the entire snapshot, then change
`NRC_SNAPSHOT_ROOT` in the private runtime configuration. Keep the original until
the new collection passes integrity and citation checks. The SQLite database
requires a local/block filesystem with supported locking; do not move it to NFS
or an object-storage mount. Do not treat additional storage as approval to rent
GPUs, upgrade the membership, or enable external model calls.

## Start or resume

After installing the Python research dependencies, restoring and verifying the
snapshot, and installing Ollama and the chosen local model, save this configuration
as `/data/workspace/nrc-runtime/workspace.env`. Replace the two proxy placeholders
with the URL returned by Marshall's `port_open` operation for port 8765.

```sh
NRC_PUBLIC_ORIGIN=https://YOUR-SHARD.workspaces.nationalcompute.com
NRC_BASE_PATH=/user/YOUR-WORKSPACE/port/8765/
NRC_MODEL=qwen3:4b-instruct-2507-q4_K_M
ADAMS_OLLAMA_THREADS=2
ADAMS_OLLAMA_CPU_ONLY=1
ADAMS_OLLAMA_KEEP_ALIVE_SECONDS=120
ADAMS_EMBEDDING_THREADS=1
ADAMS_OLLAMA_TIMEOUT=1800
ADAMS_MIN_FREE_GB=5
```

Then run:

```sh
bash /data/workspace/nrc-adams-rag/scripts/start_workspace.sh
```

This starts the app, CPU model service, acquisition, CPU indexing, and Markdown
preparation. The Markdown worker checks new text and records extraction holdouts;
it does not certify the accuracy of NRC's OCR against every original page. Indexing
runs at a lower CPU priority than interactive answers. Existing worker locks
prevent duplicate writers. Logs append under `nrc-runtime`; worker status files
live in `data/archive/STATUS.json`, `data/index-cpu/BUILD_STATUS.json`, and
`data/markdown/STATUS.json`.
The CPU indexer also releases its inference slot between committed batches when
an answer is waiting. The default local workspace profile uses two generation
threads and one embedding thread. It keeps model weights in RAM for 120 seconds
after a reply, allowing nearby requests to reuse the loaded model, then unloads
them. This uses the existing workspace's memory and does not allocate a GPU or
buy extra CPU capacity. Set `ADAMS_OLLAMA_KEEP_ALIVE_SECONDS=0` to unload
immediately. Accepted values are 0–600 seconds. A separately funded GPU tunnel
must set `ADAMS_OLLAMA_CPU_ONLY=0`; skipping the local model launcher defaults to
that setting. These controls do not make a small CPU workspace an industry-scale
inference service.
Processes survive closing the browser, but may need this command again after a
workspace restart. A saved `STOP` file continues to pause its worker. Rename that
specific file when intentionally resuming; the launcher does not clear stop files.
The web process has bounded crash recovery: five restarts with increasing delays,
reset after ten minutes of stable running. A clean exit is not restarted. Creating
`/data/workspace/nrc-runtime/web.STOP` stops the supervised web process and prevents
another launch; it leaves the archive workers alone. Rename that file to resume.
This handles an application crash, not a sleeping, stopped or restarted workspace.
The launcher selects the repository directory before starting the supervisor.

The web service queues up to eight questions behind one active worker (adjust
`NRC_MAX_QUEUED` from 0 to 20). It searches retained documents without starting
NRC requests from visitors; acquisition remains a separate background worker.
Queued requests do not survive a process restart. The workspace launcher rejects
an external model-provider selection, so a hosted API configuration cannot
silently change this deployment's funding route.

The AMD test produced a cited answer with this Qwen3 model. A focused CPU test
using one retrieved passage also produced a cited answer, taking 456 seconds
end to end while background workers were active. Broader CPU reliability is
unproven. That CPU answer correctly described the cited code requirements, but
omitted the passage's statement that the licensee failed to perform additional
testing and a causal evaluation. Successful generation and valid citation IDs
are not a complete answer-quality evaluation. The initial Qwen3 test exceeded
five minutes, and a smaller Qwen2.5 1.5B
test exceeded 15 minutes. The configuration above allows up to 30 minutes per
model call and limits its CPU threads to two. Use a separately budgeted model
server for a responsive interface. Retrieval and ingestion can continue on the
CPU without a rented GPU.
On a CPU-limited workspace that exposes many host cores, set `NRC_CPUSET` to two
allowed CPU IDs (inspect `os.sched_getaffinity(0)` first). Sharing that CPU set lets
the lower background-worker priorities take effect; spreading every process across
all host cores can consume the shared quota before the answer model gets its turn.
Leave this unset on larger machines unless measurements show contention.

Open the app by clicking its port link **inside the Marshall console**. That click
authenticates the browser; opening a copied URL in a fresh browser may show a
sign-in message. The app retains exact origin checks and a loopback-only listener.

The service launcher uses a 9-request/second acquisition ceiling. NRC support
advised a standard 10-request/second allowance and no bulk export. Backoff and the
shared request ledger remain active. The daily setting is a local budget, not an
NRC promise. Run one acquisition host per subscription. See [operations](OPERATIONS.md)
for Date Added supplements, extraction review, and Markdown maintenance.

## Optional AMD test

Request a node through Marshall only after approving its price and duration.
Use the assigned SSH destination and workspace-managed authentication. Install
the official [Ollama Linux runtime and ROCm package](https://docs.ollama.com/linux)
on the node, bind it to `127.0.0.1:11434`, and pull a local model there.

```sh
ssh -N -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:11435:127.0.0.1:11434 ubuntu@ASSIGNED_IP
```

Restart only the web process with `ADAMS_OLLAMA_URL=http://127.0.0.1:11435`
and `NRC_MODEL=qwen3:4b-instruct-2507-q4_K_M`. Keep the model endpoint private.
Set `NRC_START_LOCAL_MODEL=0` in the workspace configuration when the model is
served on a separately funded National Compute node. This prevents future launcher
runs from starting an unnecessary CPU model server; it does not stop an already
running server, create a tunnel, rent capacity, or release a node.
Before releasing the GPU, switch the web process back to the CPU configuration
and save logs or other outputs to the persistent workspace.

At the observed price, an eight-GPU AMD node costs $24 per hour. Billing ends only
when the lease ends; stopping a model process or closing SSH does not end it.
The platform offered no automatic spending cap. Explicitly release through Marshall
and verify the released state. Recheck current terms before another rental.
# CPU questions and live discovery

The CPU-only Ollama adapter limits each generation prompt to 4,500 UTF-8 bytes
of system instructions and JSON input, four source excerpts, and 384 output
tokens. It records the exact excerpt ranges used. This is a bounded reading
budget, not a promise that every relevant fact was supplied to the model.
Retrieved full passages remain available in the answer's source list. CPU model
requests stop after at most 240 seconds; evidence is retained on timeout.

The UI distinguishes reading source context from drafting, shows elapsed time,
and exposes sources before generation completes. Streamed model tokens remain
private until the complete structured answer passes citation validation.

Set `NRC_LIVE_RETRIEVAL=1` in the private workspace environment to allow bounded
NRC APS lookups for user questions. The default launcher remains offline. Live
requests use the existing private NRC key and shared rate database, with at most
30 API requests per question. Ordinary title-cased multiword names and quoted
phrases remain phrase anchors; an absent company name is not relaxed into common
individual words. Live discovery is still bounded, so an answer must distinguish
a dated application milestone from current license issuance.

## Compact chunk storage

`adams_rag.chunk_migrate` replaces duplicate passage bodies with version-pinned
character ranges and SHA-256 checksums. The source stays in its existing compressed
Markdown archive or compressed retained-version record. Chunk IDs, physical pages,
FTS entries, vectors, extraction reviews and saved citations are preserved.
The migration compares every converted passage with its retained source and reads
it back through the application reader. A mismatch leaves that document's chunks
unchanged and records a migration holdout; it does not approve its extraction.

Build an offline copy using SQLite's backup API, keep its `markdown` directory
available beside it, then run:

```sh
python -m adams_rag.chunk_migrate --db /path/to/offline-copy/research.sqlite
```

The durable cursor makes this resumable. It also enables compact storage for
future ingestion. Run SQLite integrity/foreign-key checks and retrieval checks on
the copy before switching. Use `VACUUM INTO` on the migrated copy to reclaim free
pages and fragmentation. IDs in chunk references are explicit integer primary
keys and survive compaction. Temporary build space is not a durable backup: do
not remove the active database until a verified replacement or recoverable backup
exists on persistent storage. Stop all database users during the final switch.

Application `Store` connections resolve both old text and compact references.
Read-only tools that use raw SQLite must call
`adams_rag.chunk_storage.install_reader(connection)` before consuming `text`
columns. Select `CAST(text AS BLOB) AS stored_text` when inspecting raw storage.
Old application versions cannot interpret the new binary references. Missing or
changed canonical sources fail closed instead of yielding fabricated passages.
