#!/usr/bin/env bash
# Start CPU services on an already installed and restored Linux workspace.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
runtime_dir=${NRC_RUNTIME_DIR:-${NRC_WORKSPACE_ROOT:-/data/workspace}/nrc-runtime}
if [[ -f "$runtime_dir/workspace.env" ]]; then
  set -a
  source "$runtime_dir/workspace.env"
  set +a
fi
: "${NRC_PUBLIC_ORIGIN:?Configure NRC_PUBLIC_ORIGIN in workspace.env}"
: "${NRC_BASE_PATH:?Configure NRC_BASE_PATH in workspace.env}"
export ADAMS_MODEL_PROVIDER=${ADAMS_MODEL_PROVIDER:-local}
[[ "$ADAMS_MODEL_PROVIDER" == local ]] || {
  echo 'This launcher uses the National Compute model service; external model providers are not enabled here.' >&2
  exit 1
}
case "${NRC_START_LOCAL_MODEL:-1}" in
  0|1) ;;
  *) echo 'NRC_START_LOCAL_MODEL must be 0 or 1.' >&2; exit 1 ;;
esac
export NRC_MODEL=${NRC_MODEL:-qwen3:4b-instruct-2507-q4_K_M}
# Fit the included workspace. These settings do not provision any resources.
export ADAMS_EMBEDDING_THREADS=${ADAMS_EMBEDDING_THREADS:-1}
export ADAMS_OLLAMA_THREADS=${ADAMS_OLLAMA_THREADS:-2}
export ADAMS_OLLAMA_KEEP_ALIVE_SECONDS=${ADAMS_OLLAMA_KEEP_ALIVE_SECONDS:-120}
export ADAMS_OLLAMA_CPU_ONLY=${ADAMS_OLLAMA_CPU_ONLY:-${NRC_START_LOCAL_MODEL:-1}}
mkdir -p "$runtime_dir"
launch() {
  local service=$1 priority=$2
  local command=(bash "$repo_dir/scripts/workspace_service.sh" "$service")
  if [[ "$service" == web ]]; then
    command=("${NRC_PYTHON:-$repo_dir/.venv/bin/python}" -u -m adams_rag.service_supervisor
      --stop-file "$runtime_dir/web.STOP" -- "${command[@]}")
  fi
  if [[ -n "${NRC_CPUSET:-}" ]]; then
    command=(taskset -c "$NRC_CPUSET" "${command[@]}")
  fi
  nohup nice -n "$priority" flock -n "$runtime_dir/$service.service.lock" \
    "${command[@]}" \
    >> "$runtime_dir/$service.log" 2>&1 < /dev/null &
}
if [[ "${NRC_START_LOCAL_MODEL:-1}" == 1 ]] && ! curl -fsS --max-time 3 http://127.0.0.1:11434/api/version >/dev/null; then
  launch model 0
fi
if [[ ! -f "$runtime_dir/web.STOP" ]] && ! curl -fsS --max-time 3 http://127.0.0.1:8765/ >/dev/null; then
  launch web 0
fi
launch download 10
launch index 15
launch markdown 15
echo "Services launched; check $runtime_dir/*.log and the archive/index status files."
