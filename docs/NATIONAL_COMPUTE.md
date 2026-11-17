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
Processes survive closing the browser, but may need this command again after a
workspace restart. A saved `STOP` file continues to pause its worker. Rename that
specific file when intentionally resuming; the launcher does not clear stop files.

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
