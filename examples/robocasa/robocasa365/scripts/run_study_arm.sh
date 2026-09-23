#!/usr/bin/env bash
# Run exactly one frozen RoboCasa365 task/arm/seed job.
#
# This is the scheduler-friendly counterpart to run_three_arms.sh: it never
# loops over arms, so independent jobs can be assigned to independent GPUs.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {validate|config|train} --task TASK --arm {native|native_plus_ours|ours} --data-root DIR [options]" >&2
  exit 2
fi

command=$1
shift
case "$command" in
  validate|config|train) ;;
  *) echo "Unknown command: $command" >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
cd "$repo_root"

if [[ -n "${PYTHON:-}" ]]; then
  [[ -x "$PYTHON" ]] || { echo "PYTHON is not executable: $PYTHON" >&2; exit 1; }
  exec env PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
    -m openpi.robocasa365.study "$command" "$@"
fi

exec uv run python -m openpi.robocasa365.study "$command" "$@"
