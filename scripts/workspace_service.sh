#!/usr/bin/env bash
# Run one persistent-workspace service. Use nohup or your process manager outside this script.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
workspace_root=${NRC_WORKSPACE_ROOT:-/data/workspace}
snapshot_root=${NRC_SNAPSHOT_ROOT:-$workspace_root/nrc-snapshot}
runtime_dir=${NRC_RUNTIME_DIR:-$workspace_root/nrc-runtime}
data_dir=$snapshot_root/adams-rag/data
catalog_db=$snapshot_root/adams-catalog/catalog.sqlite
python_bin=${NRC_PYTHON:-$repo_dir/.venv/bin/python}
export ADAMS_MIN_FREE_GB=${ADAMS_MIN_FREE_GB:-5}
export HF_HUB_DISABLE_XET=1
mkdir -p "$runtime_dir"
if [[ -f "$runtime_dir/private/adams-api.key" ]]; then
  export ADAMS_API_KEY_FILE="$runtime_dir/private/adams-api.key"
fi
case "${1:-web}" in
  model)
    export OLLAMA_HOST=127.0.0.1:11434
    export OLLAMA_MODELS="$runtime_dir/models"
    export OLLAMA_NO_CLOUD=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=0
    exec "$runtime_dir/bin/ollama" serve
    ;;
  web|download|index|markdown)
    [[ -s "$data_dir/research.sqlite" && -s "$catalog_db" ]] || {
      echo 'Restore and verify the snapshot before starting services.' >&2; exit 1;
    }
    ;;
  *) echo 'Usage: workspace_service.sh model|web|download|index|markdown' >&2; exit 2 ;;
esac
case "${1:-web}" in
  web)
    : "${NRC_PUBLIC_ORIGIN:?Set the exact authenticated workspace HTTPS origin}"
    : "${NRC_BASE_PATH:?Set the authenticated workspace port path, including trailing slash}"
    exec "$python_bin" -m adams_rag.server \
      --db "$data_dir/research.sqlite" --rate-db "$data_dir/adams.sqlite" \
      --catalog "$catalog_db" --index-dir "$data_dir/index-cpu" \
      --ollama-model "${NRC_MODEL:-qwen3:4b-instruct-2507-q4_K_M}" \
      --public-origin "$NRC_PUBLIC_ORIGIN" --base-path "$NRC_BASE_PATH"
    ;;
  download)
    [[ -n "${ADAMS_API_KEY_FILE:-}" || -n "${ADAMS_API_KEY:-}" ]] || {
      echo 'Install the NRC key privately before starting acquisition.' >&2; exit 1;
    }
    exec "$python_bin" -u -m adams_rag.archive_worker \
      --db "$data_dir/research.sqlite" --catalog "$catalog_db" \
      --rate-db "$data_dir/adams.sqlite" --directory "$data_dir/archive" \
      --library both --batch-size 200 --prefetch --prefetch-depth 2 \
      --rps "${NRC_RPS:-9}" --daily-limit "${NRC_DAILY_LIMIT:-777600}" \
      --min-free-gb "$ADAMS_MIN_FREE_GB"
    ;;
  index)
    exec "$python_bin" -u -m adams_rag.index_worker \
      --db "$data_dir/research.sqlite" --directory "$data_dir/index-cpu" --backend cpu --follow
    ;;
  markdown)
    while [[ ! -f "$data_dir/markdown/STOP" ]]; do
      "$python_bin" -u -m adams_rag.markdown_archive --db "$data_dir/research.sqlite" \
        --directory "$data_dir/markdown" --limit 2000 --batch-size 200
      state=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' \
        "$data_dir/markdown/STATUS.json")
      [[ "$state" == storage_pause || "$state" == stopped ]] && break
      sleep 60
    done
    ;;
esac
