# Use the RAG pipeline from another program

Start the server as documented in the README. All routes are on
`http://127.0.0.1:8765`. Requests return JSON. The server permits one active
research job and returns HTTP 409 if another is already running.
`--max-queued N` permits up to N waiting jobs (0–20), still with one active worker.
`--offline-only` forces retained-corpus retrieval for all clients. `GET /healthz`
checks process liveness without calling a model or scanning the collection.
See [hosted API deployment](HOSTED_API.md) for explicit hosted-model activation
and its persistent spending allowance. The localhost server requires an
authenticated HTTPS proxy before sharing it with a pilot team.

Retrieve reusable evidence without generating an answer:

```sh
curl -sS http://127.0.0.1:8765/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"question":"What did the inspection find about valve testing?","live":false,"top_k":10}'
```

Run retrieval and source-linked generation:

```sh
curl -sS http://127.0.0.1:8765/v1/rag \
  -H 'Content-Type: application/json' \
  -d '{"question":"What did the inspection find about valve testing?","live":false,"top_k":10}'
```

Both return HTTP 202 with a job `id`. Poll `GET /api/jobs/<id>` until the
state is `finished` or `failed`; a finished job includes its `result`. Results
contain passages, source/version bindings, coverage information and warnings.
Generation adds the answer and citation checks when successful. Inspect the
result's `mode` and warnings: evidence retained after generation fails is not
a generated answer.

Additional request fields are `docket`, `date_from`, `date_to`, and
`max_documents`. `top_k` and `max_documents` must be integers from 1 to 20.
`live:true` permits bounded NRC discovery and acquisition when a key is
configured. The browser's default live behavior can therefore make API calls;
use `live:false` explicitly for offline programmatic evaluation.

`GET /api/status` reports collection and model status. `GET /v1/catalog` searches
the local metadata catalog. The API is a reusable retrieval/context/generation
interface; the browser is one client of it.
