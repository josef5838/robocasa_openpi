#!/usr/bin/env bash
# Download exactly the training release, including converted data and stats.
# Set PYTHON to an existing OpenPI environment to bypass `uv` setup if needed.
set -euo pipefail
usage() {
  echo "Usage: $0 ORG/DATASET_REPO COMMIT_SHA DESTINATION" >&2
  echo "   or: $0 --config RELEASE_CONFIG DESTINATION" >&2
}
[[ $# -ge 1 ]] || { usage; exit 2; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
cd "$repo_root"
if [[ "$1" == "--config" ]]; then
  [[ $# -eq 3 ]] || { usage; exit 2; }
  args=(--release-config "$2" --data-root "$3")
else
  [[ $# -eq 3 ]] || { usage; exit 2; }
  args=(--repo-id "$1" --revision "$2" --data-root "$3")
fi
if [[ -n "${PYTHON:-}" ]]; then
  [[ -x "$PYTHON" ]] || { echo "PYTHON is not executable: $PYTHON" >&2; exit 1; }
  exec env PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
    -m openpi.robocasa365.study download "${args[@]}"
fi
exec uv run --with huggingface_hub python -m openpi.robocasa365.study download "${args[@]}"
