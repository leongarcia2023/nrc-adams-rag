# NRC ADAMS RAG

A runnable research prototype for public NRC documents: catalog discovery,
resumable text acquisition, versioned Markdown sources, chunking, keyword and
semantic retrieval, and locally generated answers with source citations.

This repository contains the application and tests. It does **not** contain the
downloaded archive, API keys, model weights, or a complete copy of ADAMS. The
existing collection can be transferred separately, or rebuilt with an NRC APS
API subscription. See [the INL handoff guide](docs/INL_HANDOFF.md).
For the persistent Marshall workspace, private app URL configuration, and optional
AMD model server, see [the National Compute deployment guide](docs/NATIONAL_COMPUTE.md).

## What runs where

| Capability | Linux / Windows through WSL2 | Apple silicon macOS |
| --- | --- | --- |
| Catalog, download, chunking, Markdown export | Yes | Yes |
| Keyword and CPU semantic retrieval | Yes | Yes |
| Browser interface and full RAG with local Ollama | Yes | Yes |
| MLX answer model and MLX embeddings | No | Yes |
| Embedded PDF text extraction | Yes | Yes |
| Built-in scanned-page OCR through Apple Vision | No | Yes |

Use Python 3.12. Native Windows is not supported: workers use POSIX file locks;
run the Python application inside WSL2. For the Ollama path, run Ollama in that
same Linux environment so the default loopback connection works. Linux GPU
support depends on the chosen Ollama installation and hardware. The Python
semantic index uses its CPU backend by default.

## Install and check

```sh
git clone https://github.com/leongarcia2023/nrc-adams-rag.git
cd nrc-adams-rag
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[research]'
python scripts/smoke_test.py
```

The smoke test creates a tiny synthetic document in a temporary directory,
checks import, retrieval and Markdown export, then removes its temporary data.
It needs no credentials, model, or NRC connection.

The examples below use paths relative to this checkout. Put `data/` on a volume
with room to grow, or supply absolute paths using the corresponding command
options. Set the storage floor for the machine:

```sh
export ADAMS_MIN_FREE_GB=5
```

Workers stop before crossing this reserve plus their next operation's estimated
space allowance. The default is 2 GB; the old Mac's 0.5 GB override is not shipped.

## Connect an NRC subscription

Obtain an APS API subscription through [NRC ADAMS](https://adams-search.nrc.gov/).
Store the key in an owner-only file outside the checkout, then set its absolute
path in the shell running the application:

```sh
export ADAMS_API_KEY_FILE="$HOME/.config/nrc-adams/api.key"
chmod 600 "$ADAMS_API_KEY_FILE"
```

Create that file with your own key first. `ADAMS_API_KEY` is also supported for
managed environments. No key is required for offline retrieval over existing
data. The application does not automatically load `.env` files.

## Acquire a small starting collection

Start with a bounded query and fetch its document text:

```sh
python -m adams_rag --db data/research.sqlite live-search \
  'LaSalle relief valve testing' --library main --max-requests 5
python -m adams_rag --db data/research.sqlite hydrate \
  --limit 20 --max-requests 25
python -m adams_rag --db data/research.sqlite index --limit 2000
```

Search metadata alone is not answer evidence; `hydrate` acquires source text.
Indexing downloads the pinned embedding model on first use. Repeat the indexing
command until pending work is complete. For the full catalog, ongoing updates,
and resumable archive workers, use [the operations guide](docs/OPERATIONS.md).

For retrieval without downloading embedding weights:

```sh
python -m adams_rag --db data/research.sqlite --json retrieve \
  'What did the inspection find about valve testing?' --offline --no-semantic
```

## Generate answers locally

Install [Ollama](https://docs.ollama.com/) and a local text model approved for
your machine. Start its local server and set the exact installed model name:

```sh
ollama list
export NRC_OLLAMA_MODEL='your-installed-model-name'
python -m adams_rag --db data/research.sqlite --json research \
  'What did the inspection find about valve testing?' \
  --offline --ollama-model "$NRC_OLLAMA_MODEL"
```

Replace `your-installed-model-name`; this is not a downloadable model tag.
Choose a model that can produce JSON and accept a 16,384-token context. The
adapter uses [Ollama's local chat API](https://docs.ollama.com/api/chat), validates
the answer structure and citation IDs, and withholds malformed output. Ollama
manages model memory and GPU resources; the MLX child-process disk monitor does
not manage an external Ollama process. Only loopback model connections are
accepted, and remote/cloud model descriptors are rejected. Run Ollama with its
cloud features disabled if local-only processing is required.

Launch the one-box browser interface:

```sh
python -m adams_rag.server --ollama-model "$NRC_OLLAMA_MODEL"
```

Open **http://127.0.0.1:8765/**. The server binds only to loopback and handles one
research job at a time. See [API examples](docs/API.md) for using the same pipeline
from another program.

On an Apple silicon Mac, the existing MLX path is also available:

```sh
python -m pip install '.[apple]'
python -m adams_rag --db data/research.sqlite setup-model
python -m adams_rag.server
```

The setup command downloads the pinned Qwen3 MLX model. Install `.[ocr]` only
when using Apple's local OCR. The macOS lock file records the original machine's
dependencies; do not install that lock file on Linux.

## Tests

```sh
python -m pip install '.[test]'
python scripts/prepare_test_assets.py
python -m unittest discover -s tests
python scripts/smoke_test.py
```

Test preparation downloads a checksum-verified 711 KB tokenizer, excluded from
Git. It does not download model weights or NRC records. Without it, three window
integration-test classes are explicitly skipped. CI runs the full suite on
Linux and macOS with the tokenizer present.

The Ollama transport and its integration with retrieval are tested against a
local simulated server. Those checks do not establish the answer quality or
performance of a model selected by INL; evaluate that model against reviewed
source passages before relying on its answers.

## Coverage and provenance

Metadata coverage, downloaded text, accepted extraction, chunks, and embeddings
are separate counts. This is a partial full-text collection and an ongoing
full-archive project. A completed catalog scan does not mean every document has
usable text or vectors. Legacy records, missing text, scanned pages, tables and
withdrawn or revised material need separate handling.

Each evidence package retains source identity and text-version checksums.
Markdown export preserves the retained text and source links; automated checks
do not certify that OCR matches every original page. Reviewed bad extractions
are held out of retrieval. A valid citation ID also does not prove that the
source supports every generated claim, or that a historical document establishes
a current requirement.

Third-party code, fonts, datasets and model references are documented in
[THIRD_PARTY.md](THIRD_PARTY.md), with bundled license notices retained.
