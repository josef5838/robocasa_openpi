#!/usr/bin/env bash
# Run one frozen task through native, native_plus_ours, and ours.
set -euo pipefail
if [[ $# -lt 1 ]]; then echo "Usage: $0 {validate|config|train} --task TASK --data-root DIR [options]" >&2; exit 2; fi
command=$1; shift
case "$command" in validate|config|train) ;; *) echo "Unknown command: $command" >&2; exit 2;; esac
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
cd "$repo_root"
run_study() {
  if [[ -n "${PYTHON:-}" ]]; then
    [[ -x "$PYTHON" ]] || { echo "PYTHON is not executable: $PYTHON" >&2; exit 1; }
    env PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m openpi.robocasa365.study "$@"
  else
    uv run python -m openpi.robocasa365.study "$@"
  fi
}
if [[ "$command" == validate ]]; then
  run_study validate "$@"
  exit $?
fi
for arm in native native_plus_ours ours; do
  echo "=== $command: arm=$arm ==="
  run_study "$command" --arm "$arm" "$@"
done
