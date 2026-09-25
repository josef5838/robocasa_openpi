#!/usr/bin/env bash
set -euo pipefail

task=${1:?task is required}
gpu=${2:?gpu index is required}
case "$task:$gpu" in
  TurnSinkSpout:0|OpenToasterOvenDoor:1) ;;
  *) echo "Unsupported task/GPU assignment: $task:$gpu" >&2; exit 2 ;;
esac

repo=/data/users/haoliang/robocasa_openpi
data=/data/users/haoliang/experiments/robocasa365-feas/data/pi05_current_raw_20260922
python=/data/users/haoliang/experiments/robocasa365-feas/repos/openpi/.venv/bin/python
study="$repo/examples/robocasa/robocasa365/configs/robocasa365_20260922_articraft_extension.json"
output="$repo/examples/robocasa/robocasa365/artifacts/articraft_extension_seed0_20260925"
log="$output/${task}_articraft_gpu${gpu}.log"
status="$output/${task}_articraft_gpu${gpu}.exit"

mkdir -p "$output"
rm -f "$status"
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu"
export JAX_COMPILATION_CACHE_DIR="/tmp/pi05-robocasa365-jax-${task}-articraft-gpu${gpu}"
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"

printf 'START %s task=%s arm=articraft gpu=%s host=%s\n' "$(date -Is)" "$task" "$gpu" "$(hostname)" >> "$log"
if "$python" examples/robocasa/robocasa365/scripts/run_task_arm.py \
  --study "$study" --task "$task" --arm articraft \
  --data-root "$data" --output-root "$output" >> "$log" 2>&1; then
  result=0
else
  result=$?
fi
printf 'EXIT %s task=%s status=%s\n' "$(date -Is)" "$task" "$result" >> "$log"
printf '%s\n' "$result" > "$status"
exit "$result"
