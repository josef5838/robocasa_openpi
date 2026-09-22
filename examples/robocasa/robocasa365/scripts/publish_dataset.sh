#!/usr/bin/env bash
# Safe by default: preflight only. Add --upload to publish after validation.
# Set PYTHON to an existing OpenPI environment to bypass `uv` setup if needed.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
cd "$repo_root"
if [[ -n "${PYTHON:-}" ]]; then
  [[ -x "$PYTHON" ]] || { echo "PYTHON is not executable: $PYTHON" >&2; exit 1; }
  exec env PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
    examples/robocasa/robocasa365/scripts/publish_dataset.py "$@"
fi
exec uv run --with huggingface_hub python examples/robocasa/robocasa365/scripts/publish_dataset.py "$@"
