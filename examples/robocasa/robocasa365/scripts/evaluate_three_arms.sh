#!/usr/bin/env bash
# Derive held-out native fixtures and evaluate native, native_plus_ours, and ours.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {plan|evaluate} --task TASK --data-root DIR --output-root DIR [options]" >&2
  exit 2
fi

command=$1
shift
case "$command" in
  plan|evaluate) ;;
  *) echo "Unknown command: $command" >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
cd "$repo_root"
run_evaluator() {
  if [[ -n "${PYTHON:-}" ]]; then
    [[ -x "$PYTHON" ]] || { echo "PYTHON is not executable: $PYTHON" >&2; exit 1; }
    env PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m openpi.robocasa365.evaluate "$@"
  else
    uv run python -m openpi.robocasa365.evaluate "$@"
  fi
}

for arm in native native_plus_ours ours; do
  echo "=== $command: arm=$arm ==="
  run_evaluator "$command" --arm "$arm" "$@"
done
