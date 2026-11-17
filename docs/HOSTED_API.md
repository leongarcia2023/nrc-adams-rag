# Hosted model API and permanent pilot

The hosted adapter uses Groq for answer generation while the NRC archive,
retrieval, source bindings, and citation validation remain in this application.
Only the question and selected evidence are sent to Groq. Local MLX and Ollama
remain the default. Setting the provider alone cannot start paid inference.

This is prepared code, not an activated hosting subscription. Tests use simulated
provider responses. No real Groq key or paid answer-quality benchmark has been
used to validate this adapter yet.

## Cost basis, checked November 17, 2026

[Groq's production model list](https://console.groq.com/docs/models) publishes
these standard rates in USD per million tokens:

| Model | Input | Output | 1,000 example questions | 10,000 example questions |
| --- | ---: | ---: | ---: | ---: |
| GPT-OSS 20B | $0.075 | $0.30 | $1.05 | $10.50 |
| GPT-OSS 120B | $0.15 | $0.60 | $2.10 | $21.00 |

The examples assume 10,000 input tokens and 1,000 billed output tokens **in total
per question**, across any planning and answer calls. Output includes reasoning.
These are workload estimates, not a fixed price per question. Multi-report
questions can generate several calls. No paid search tools or embeddings API
are used by this adapter. Quality on NRC questions remains unmeasured for both
models. Start evaluation with 120B; compare 20B before choosing on cost alone.

[Gemini 2.5 Flash-Lite](https://aistudio.google.com/docs/pricing#gemini-2.5-flash-lite)
lists $0.10 input / $0.40 output, or $1.40 per 1,000 questions under the same
assumptions. It is a comparison, not an implemented backend.

The [OVHcloud US configurator](https://us.ovhcloud.com/vps/configurator/?planCode=vps-2027-model2)
showed VPS-2 2027, 4 vCPU, 8 GB RAM, 75 GB NVMe, IPv4 and standard daily backup
at **$10/month before tax, no commitment**, in Beauharnois, Canada. The $8.50
headline price required twelve months paid up front. No order was placed.
This is the cheapest suitable candidate verified in this comparison, not an
exhaustive claim about every provider. Regional availability can change.

That gives a current-corpus pilot estimate of **$11.05 or $12.10/month** at
1,000 example questions, depending on the model. Allow about $15/month before
tax as a planning figure for the 120B pilot and a modest separate backup.
At 10,000 example questions, the corresponding totals are $20.50 or $31/month.
An existing subdomain adds no domain-registration charge. An independent
[Backblaze B2](https://www.backblaze.com/cloud-storage/pricing) backup is listed
from $6.95/TB-month with the first 10 GB free; retained versions and download
overages affect its bill. Backup configuration is still deployment work.

These prices cover a low-traffic pilot serving the current partial corpus.
75 GB is not capacity for the full ADAMS archive. A complete build needs more
archive storage and a measured search-index memory/disk budget. Object storage
cannot replace the local disk holding the current SQLite database and vector
index. Do not promise full-archive hosting at the pilot price.

National Compute is another candidate where managed CPU hosting is available.
Obtain an account-specific CPU, storage, and public-endpoint quote first. Pro
membership alone is not a hosting/storage price. Grandfathered access can avoid
a membership upgrade, but does not establish that compute is free. A Public
Research organization must confirm bucket eligibility separately. Marshall's
model allowance is not a documented application inference API.

## Explicit activation after approval

The process user must own a model key file with mode 600. Keep it outside the
checkout and never put the key in command arguments, logs, or browser code.
The private environment selects the provider, key file, and budget:

```sh
ADAMS_MODEL_PROVIDER=groq
ADAMS_API_MODEL=openai/gpt-oss-120b
ADAMS_MODEL_API_KEY_FILE=/absolute/private/model-api.key
ADAMS_MODEL_BUDGET_DB=/absolute/persistent/model-budget.sqlite
ADAMS_ALLOW_HOSTED_MODEL=0
ADAMS_MODEL_MONTHLY_USD=0
```

After provider/data-transfer and spending approval, an operator sets the allow
flag to `1` and the monthly allowance to the approved amount. Do not pass
`--ollama-model` with the hosted provider. The status endpoint reports
`configured`, not `ready`, because checking status performs no network call
and cannot prove the key works. `local_model_installed` remains false.

Requests go only to `https://api.groq.com/openai/v1/chat/completions` through
certificate-verified HTTPS. Redirects and environment proxies are not used.
Client requests cannot choose a different host, model, key, or budget.
Provider error bodies are withheld. Incomplete output and invalid citation IDs
are withheld by the shared validation path.

## Spending controls

Before dispatch the application reserves a conservative cost based on the model's
entire accepted context window and the request's maximum output. SQLite makes
the reservation atomic across processes sharing the ledger. A valid usage receipt
replaces the reservation with the calculated actual token cost. Timeouts, crashes,
missing usage, and error responses retain the reservation. There are no automatic
paid retries. Review uncertain entries against the provider bill; never clear
them merely to allow more requests.

The allowance resets by UTC calendar month. Keep the ledger on persistent local
disk and in backups. All serving instances must share the same supported ledger
or receive separate explicitly allocated budgets; copying it into several
independent replicas does not enforce a shared limit. The rates are versioned
in code and must be updated if the provider changes prices.

This control covers requests through this application at the configured rates,
not taxes, other applications using the account, or provider billing changes.
Also configure the provider's own limit when approved. Groq documents a
[10–15 minute delay](https://console.groq.com/docs/spend-limits) in its account
spend tracking, so that control alone can overshoot under load.

## Deployment prepared for a small pilot

The templates in `deploy/pilot/` do not install or activate anything:

- `nrc-rag.service` runs the application as an unprivileged Linux account,
  restarts on failure, and listens only on loopback. It accepts eight queued
  questions behind one active worker. This is bounded queuing, not parallel
  research execution. Pending jobs are in memory and do not survive a restart;
  completed evidence runs and the spending ledger do.
- `service.env.example` starts with hosted inference disabled and a zero budget.
- `Caddyfile` provides HTTPS and a required pilot login. It blocks global question
  history and operational status at the proxy. It does not provide separate
  customer accounts: saved-run and job UUIDs act as bearer links within the
  authenticated pilot. Use only a trusted team that understands that sharing model.

After approving hosting, provision Linux with Python 3.11 or newer and Caddy from
their official sources. Create the `nrc` service account, install this checkout
and its `research` dependencies under `/srv/nrc/app`, and restore a verified,
consistent database/Markdown/index snapshot under `/srv/nrc/data`. Keep all source
files in the snapshot together. Place the catalog at
`/srv/nrc/data/catalog/catalog.sqlite`; verify relocated Markdown downloads.
Prepare query-embedding weights before accepting traffic so a first question does
not trigger a model download. Install no local answer model.

Set `NRC_PUBLIC_ORIGIN` to the exact HTTPS origin in `/etc/nrc-rag/service.env`.
Set the domain, pilot username and password hash in Caddy's private service
configuration. Generate the hash interactively with `caddy hash-password`; validate
the Caddyfile before exposing it. The password must not be committed. See
[Caddy authentication](https://caddyserver.com/docs/caddyfile/directives/basic_auth).

Validate the systemd unit on the target host with `systemd-analyze verify`, then
enable the service only after the activation decision. Configure a firewall for
HTTPS/HTTP and restricted administration. Add an external health check and test
reboot recovery, backup restoration, and a real source-reviewed answer. These
Linux deployment templates have not yet been executed on a purchased server.

`GET /healthz` is a cheap liveness check; it makes no model call. It is not a
readiness or quality test. The service template forces offline retrieval even
when the browser sends `live:true`. Acquisition runs separately with the existing
NRC rate limiter, and verified snapshots are promoted to serving storage.

Before anonymous/public industry access, add per-user authorization and quotas,
durable jobs, retention controls, load testing and reviewed NRC evaluation cases.
Measure the growing index before selecting full-archive hardware. A single VPS
can run continuously but cannot provide redundant availability by itself.
