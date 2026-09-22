# RoboCasa365 Pi0.5 three-arm study

This repository runs the fixed-contract RoboCasa365 Pi0.5 comparison across three training arms: `native`, `native_plus_ours`, and `ours`. For every selected task and seed, the model, optimizer, batch size, schedule, LoRA ranks, vision-tower setting, instruction, and normalization policy are fixed. Dataset composition is the only experimental variable.

The frozen study definition is [examples/robocasa/robocasa365/configs/robocasa365_20260922.json](examples/robocasa/robocasa365/configs/robocasa365_20260922.json).

## Requirements

Training requires a CUDA GPU with at least 80 GB of memory. Evaluation requires a CUDA GPU with at least 8 GB of memory. A compatible RoboCasa checkout must be importable in the OpenPI environment.

## Installation

```bash
git clone https://github.com/robocasa-benchmark/openpi.git robocasa_openpi
cd robocasa_openpi
uv sync --frozen
uv run python -m pip install -e /path/to/robocasa
```

If the locked environment cannot build PyAV on the machine, set `PYTHON=/path/to/openpi/.venv/bin/python`. The download, training, and evaluation shell wrappers all honor it, so use the same known-working OpenPI environment throughout the workflow instead of mixing environments.

## Download checkpoints and data

The study fine-tunes the fixed Pi0.5 base checkpoint. Prefetch it before training to verify access:

```bash
examples/robocasa/robocasa365/scripts/prefetch_pi05_base.sh
```

Download one immutable RoboCasa365 data release containing converted LeRobot assets, manifests, and normalization artifacts. Copy the example configuration, set its Hugging Face dataset repository and immutable commit SHA, then download it:

```bash
cp examples/robocasa/robocasa365/configs/data_release.example.json \
  examples/robocasa/robocasa365/configs/data_release.json
# Edit data_release.json: repo_id and revision.
examples/robocasa/robocasa365/scripts/download_dataset.sh \
  --config examples/robocasa/robocasa365/configs/data_release.json \
  examples/robocasa/robocasa365/data/release
```

Validate the selected task before training. This checks every arm manifest, the LeRobot data contract, and the uploaded normalization artifacts.

```bash
examples/robocasa/robocasa365/scripts/run_three_arms.sh validate \
  --task TurnSinkSpout \
  --data-root examples/robocasa/robocasa365/data/release
```

## Dataset composition

Every asset contributes 50 demonstrations. `native_plus_ours` is the exact deterministic concatenation of the corresponding native and generated manifests; the source assets are stored once in the release.

| Task | Native | Ours | Native + ours |
| --- | ---: | ---: | ---: |
| OpenElectricKettleLid | 8 assets / 400 episodes | 10 assets / 500 episodes | 18 assets / 900 episodes |
| OpenFridgeDrawer | 5 assets / 250 episodes | 1 asset / 50 episodes | 6 assets / 300 episodes |
| OpenStandMixerHead | 9 assets / 450 episodes | 8 assets / 400 episodes | 17 assets / 850 episodes |
| OpenToasterOvenDoor | 6 assets / 300 episodes | 3 assets / 150 episodes | 9 assets / 450 episodes |
| SlideDishwasherRack | 10 assets / 500 episodes | 1 asset / 50 episodes | 11 assets / 550 episodes |
| SlideOvenRack | 3 assets / 150 episodes | 2 assets / 100 episodes | 5 assets / 250 episodes |
| TurnOnStove | 10 assets / 500 episodes | 5 assets / 250 episodes | 15 assets / 750 episodes |
| TurnSinkSpout | 10 assets / 500 episodes | 5 assets / 250 episodes | 15 assets / 750 episodes |
| **All tasks** | **61 assets / 3,050 episodes** | **35 assets / 1,750 episodes** | **96 assets / 4,800 episodes** |

## Train

Train all three arms for one task and one seed together. The contract is Pi0.5 with batch size 64, 8,000 steps, peak learning rate `5e-5`, PaliGemma LoRA rank 16, action-expert LoRA rank 32, and a trainable vision tower. The command writes manifests, configuration digests, checkpoints, and copied normalization assets below the output root.

```bash
COMMON=(
  --task TurnSinkSpout
  --data-root examples/robocasa/robocasa365/data/release
  --output-root examples/robocasa/robocasa365/artifacts/turnsink_seed0
  --seed 0
)

examples/robocasa/robocasa365/scripts/run_three_arms.sh config "${COMMON[@]}"
examples/robocasa/robocasa365/scripts/run_three_arms.sh train "${COMMON[@]}"
```

Repeat the same procedure with seeds `1` and `2`. Add `--resume` to resume an interrupted training command. Checkpoints are stored at:

```text
<output-root>/checkpoints/pi05_robocasa365_<task>_<arm>_lora_b64_8k_fixed_instruction/seed-<seed>/<step>/
```

## Evaluate

Evaluation uses only held-out native fixtures that match the structural variant of the native training fixtures. For a native or combined arm, its native training fixtures are excluded. For the generated-only arm, all compatible native fixtures remain held out. The evaluator reconstructs the simulator from the portable `converted/.../demo.hdf5` recording included in each release asset (not an author-machine absolute path), uses the fixed training instruction, three 224px camera inputs, five-step replanning, and normalization statistics saved in the checkpoint.

First inspect the immutable test split for every arm. This command does not run policy rollouts and does not require a checkpoint, so run it before training to review the exact evaluation protocol:

```bash
examples/robocasa/robocasa365/scripts/evaluate_three_arms.sh plan \
  --task TurnSinkSpout \
  --data-root examples/robocasa/robocasa365/data/release \
  --output-root examples/robocasa/robocasa365/artifacts/turnsink_seed0 \
  --seed 0 \
  --robocasa-assets /path/to/robocasa/robocasa/models/assets
```

After reviewing the emitted `held_out_test_assets`, run evaluation. It loads the latest committed checkpoint for each arm and runs exactly 50 rollouts per held-out asset. The process is resumable only when the saved protocol matches.

```bash
examples/robocasa/robocasa365/scripts/evaluate_three_arms.sh evaluate \
  --task TurnSinkSpout \
  --data-root examples/robocasa/robocasa365/data/release \
  --output-root examples/robocasa/robocasa365/artifacts/turnsink_seed0 \
  --seed 0 \
  --robocasa-assets /path/to/robocasa/robocasa/models/assets
```

Per-asset results and the pooled task success rate are written to:

```text
<output-root>/evaluations/<task>/<arm>/seed-<seed>/<checkpoint-step>/stats.json
```

The `held_out.success_rate` field is the per-task, per-arm success rate across all held-out rollouts. `held_out.assets` reports the same metric for each fixture.

## Repository entry points

- [Study configuration](examples/robocasa/robocasa365/configs/robocasa365_20260922.json)
- [Three-arm trainer](examples/robocasa/robocasa365/scripts/run_three_arms.sh)
- [Three-arm evaluator](examples/robocasa/robocasa365/scripts/evaluate_three_arms.py)
- [Held-out evaluation implementation](src/openpi/robocasa365/evaluate.py)
