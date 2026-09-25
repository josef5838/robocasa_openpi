#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
checkpoint_root="$repo_root/checkpoints"
study="$repo_root/examples/robocasa/robocasa365/configs/robocasa365_20260922.json"
python=${OPENPI_PYTHON:-/data/users/haoliang/experiments/robocasa365-feas/repos/openpi/.venv/bin/python}
data_root=${ROBOCASA365_DATA_ROOT:-/data/users/haoliang/experiments/robocasa365-feas/data/pi05_current_raw_20260922}
output_root=${ROBOCASA365_EVAL_OUTPUT_ROOT:-$repo_root/examples/robocasa/robocasa365/artifacts/hf_checkpoints_eval10_seed20260925}
log_root="$output_root/logs"

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

launch_workers() {
  [[ -x "$python" ]] || fail "Python environment not executable: $python"
  [[ -d "$checkpoint_root" ]] || fail "Checkpoint directory not found: $checkpoint_root"
  [[ -f "$data_root/manifests/raw_datasets.json" ]] || fail "Dataset manifests not found under: $data_root"
  mkdir -p "$log_root"
  for gpu in 0 1; do
    pid_file="$output_root/gpu${gpu}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
      fail "GPU $gpu worker is already running with PID $(<"$pid_file")"
    fi
  done
  for gpu in 0 1; do
    log_file="$log_root/gpu${gpu}.log"
    nohup bash "$script_dir/evaluate_hf_checkpoints_2gpu.sh" worker "$gpu" \
      </dev/null >>"$log_file" 2>&1 &
    printf '%s\n' "$!" >"$output_root/gpu${gpu}.pid"
    printf 'Started GPU %s worker, pid %s, log %s\n' "$gpu" "$!" "$log_file"
  done
}

run_worker() {
  gpu=${1:?GPU index required}
  [[ "$gpu" == 0 || "$gpu" == 1 ]] || fail "GPU must be 0 or 1"
  [[ -x "$python" ]] || fail "Python environment not executable: $python"
  [[ -f "$data_root/manifests/raw_datasets.json" ]] || fail "Dataset manifests not found under: $data_root"
  [[ -f "$study" ]] || fail "Study definition not found: $study"

  export CUDA_VISIBLE_DEVICES="$gpu"
  export MUJOCO_GL=egl
  export WANDB_MODE=disabled
  export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
  export JAX_COMPILATION_CACHE_DIR="/tmp/pi05-robocasa365-eval10-gpu${gpu}"
  mkdir -p "$log_root"
  cd "$repo_root"

  "$python" -c 'import jax; devices=jax.devices(); print("JAX devices:", devices, flush=True); assert any(d.platform == "gpu" for d in devices), "CUDA GPU unavailable"'

  mapfile -t checkpoint_dirs < <(find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -name 'pi05_robocasa365_*' | sort)
  ((${#checkpoint_dirs[@]} == 24)) || fail "Expected 24 checkpoints, found ${#checkpoint_dirs[@]}"

  worker_failed=0
  checkpoint_index=0
  for checkpoint_dir in "${checkpoint_dirs[@]}"; do
    assigned_gpu=$((checkpoint_index % 2))
    checkpoint_index=$((checkpoint_index + 1))
    [[ "$assigned_gpu" == "$gpu" ]] || continue

    checkpoint_name=${checkpoint_dir##*/}
    task_arm=${checkpoint_name#pi05_robocasa365_}
    task=${task_arm%%_*}
    arm_suffix=${task_arm#"$task"_}
    arm=${arm_suffix%_lora_b64_8k_fixed_instruction}
    checkpoint="$checkpoint_dir/seed-0/7999"
    [[ -f "$checkpoint/_CHECKPOINT_METADATA" ]] || {
      printf 'SKIP missing committed checkpoint: %s\n' "$checkpoint" >&2
      worker_failed=1
      continue
    }

    job_log="$log_root/gpu${gpu}_${task}_${arm}.log"
    printf 'START %s gpu=%s task=%s arm=%s checkpoint=%s\n' "$(date -Is)" "$gpu" "$task" "$arm" "$checkpoint" | tee -a "$job_log"
    if "$python" -m openpi.robocasa365.evaluate evaluate \
      --study "$study" \
      --task "$task" \
      --arm "$arm" \
      --data-root "$data_root" \
      --output-root "$output_root" \
      --eval-root "$output_root/evaluations" \
      --checkpoint "$checkpoint" \
      --seed 0 \
      --seed-offset 20260925 \
      --total-episodes 10 \
      --replan-steps 5 \
      --video-mode all \
      --video-fps 30 >>"$job_log" 2>&1; then
      printf 'DONE %s gpu=%s task=%s arm=%s\n' "$(date -Is)" "$gpu" "$task" "$arm" | tee -a "$job_log"
    else
      result=$?
      printf 'FAILED %s gpu=%s task=%s arm=%s exit=%s\n' "$(date -Is)" "$gpu" "$task" "$arm" "$result" | tee -a "$job_log"
      worker_failed=1
    fi
  done
  exit "$worker_failed"
}

case "${1:-launch}" in
  launch) launch_workers ;;
  worker) shift; run_worker "$@" ;;
  *) fail "Usage: $0 [launch|worker GPU]" ;;
esac
