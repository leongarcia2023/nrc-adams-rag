#!/usr/bin/env bash
# Start CPU services on an already installed and restored Linux workspace.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runtime_dir=${NRC_RUNTIME_DIR:-${NRC_WORKSPACE_ROOT:-/data/workspace}/nrc-runtime}
if [[ -f "$runtime_dir/workspace.env" ]]; then
  set -a
  source "$runtime_dir/workspace.env"
  set +a
fi
: "${NRC_PUBLIC_ORIGIN:?Configure NRC_PUBLIC_ORIGIN in workspace.env}"
: "${NRC_BASE_PATH:?Configure NRC_BASE_PATH in workspace.env}"
export NRC_MODEL=${NRC_MODEL:-qwen2.5:1.5b-instruct-q4_K_M}
mkdir -p "$runtime_dir"
launch() {
  local service=$1 priority=$2
  local command=(bash "$repo_dir/scripts/workspace_service.sh" "$service")
  if [[ -n "${NRC_CPUSET:-}" ]]; then
    command=(taskset -c "$NRC_CPUSET" "${command[@]}")
  fi
  nohup nice -n "$priority" flock -n "$runtime_dir/$service.service.lock" \
    "${command[@]}" \
    >> "$runtime_dir/$service.log" 2>&1 < /dev/null &
}
if ! curl -fsS --max-time 3 http://127.0.0.1:11434/api/version >/dev/null; then
  launch model 0
fi
if ! curl -fsS --max-time 3 http://127.0.0.1:8765/ >/dev/null; then
  launch web 0
fi
launch download 10
launch index 15
launch markdown 15
echo "Services launched; check $runtime_dir/*.log and the archive/index status files."
