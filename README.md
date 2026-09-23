# RoboCasa365 Pi0.5 three-arm study

This repository has one execution design: run one `<task, arm, seed-0>` job on one GPU. That job trains its checkpoint, evaluates it immediately on the pinned shared held-out native fixtures, and saves its success count and success rate. The complete experiment is 8 tasks × 3 arms × one seed = 24 independent GPU jobs.

The frozen study definition is [examples/robocasa/robocasa365/configs/robocasa365_20260922.json](examples/robocasa/robocasa365/configs/robocasa365_20260922.json). Dataset composition is the only experimental variable across `native`, `native_plus_ours`, and `ours`.

## Requirements

Each job needs one CUDA GPU with at least 80 GB for training. RoboCasa and robosuite must be importable in the same Python environment; evaluation additionally needs the RoboCasa kitchen assets.

## Installation

```bash
git clone https://github.com/robocasa-benchmark/openpi.git robocasa_openpi
cd robocasa_openpi
uv sync --frozen
```

### Install robosuite and RoboCasa

Keep the two external source checkouts in `third_party/`. The revisions below are the locally validated pair for this study. Install robosuite first, then RoboCasa, into the OpenPI environment.

```bash
mkdir -p third_party
git clone https://github.com/ARISE-Initiative/robosuite third_party/robosuite
git -C third_party/robosuite checkout 7736d3248abba04abe00b5c3882c4059ed08ce1b
uv run python -m pip install -e third_party/robosuite

git clone https://github.com/robocasa/robocasa third_party/robocasa
git -C third_party/robocasa checkout 4f8a2980def75a55dff96b990745b83540425f09
uv run python -m pip install -e third_party/robocasa
uv run python -c "import robocasa; print(robocasa.__file__)"
```

Initialize RoboCasa and download its kitchen assets before running jobs. The asset download is roughly 10 GB.

```bash
uv run python -m robocasa.scripts.setup_macros
uv run python -m robocasa.scripts.download_kitchen_assets
```

If `uv sync` cannot build PyAV on the machine, use a known-working environment instead. Set `PYTHON=/path/to/openpi/.venv/bin/python` and launch the job runner as `PYTHONPATH=src "$PYTHON" examples/robocasa/robocasa365/scripts/run_task_arm.py ...`.

## Download checkpoint and data

Prefetch the fixed Pi0.5 base checkpoint:

```bash
examples/robocasa/robocasa365/scripts/prefetch_pi05_base.sh
```

Download one immutable RoboCasa365 data release containing converted LeRobot assets, manifests, and normalization artifacts:

```bash
cp examples/robocasa/robocasa365/configs/data_release.example.json \
  examples/robocasa/robocasa365/configs/data_release.json
# Edit data_release.json: repo_id and revision.
examples/robocasa/robocasa365/scripts/download_dataset.sh \
  --config examples/robocasa/robocasa365/configs/data_release.json \
  examples/robocasa/robocasa365/data/release
```

## Dataset composition

Each asset contributes 50 demonstrations. `native_plus_ours` is the deterministic concatenation of the matching native and ours manifests.

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

## Run the 24 jobs

[configs/jobs_seed0.tsv](examples/robocasa/robocasa365/configs/jobs_seed0.tsv) is the complete job manifest: 24 unique task/arm rows, all with seed `0`. Assign one row to one GPU. With 64 GPUs, launch all 24 rows concurrently; no job shares a GPU with another job.

The only job command is:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python examples/robocasa/robocasa365/scripts/run_task_arm.py \
  --task TurnSinkSpout \
  --arm native \
  --data-root examples/robocasa/robocasa365/data/release \
  --output-root examples/robocasa/robocasa365/artifacts/full_study
```

Use the `task` and `arm` values from one TSV row in that command. The runner validates the arm data, trains the fixed Pi0.5 contract, loads its latest committed checkpoint, and runs all 50-rollout held-out evaluations on the same inherited `CUDA_VISIBLE_DEVICES` card. Add `--resume` to recover an interrupted job; training resumes and evaluation continues only missing saved rollouts.

## Evaluation fixtures and results

All three arms of a task evaluate on the same native-held-out fixture set. The exact IDs are read directly from [held_out_native_assets_20260922.json](examples/robocasa/robocasa365/configs/held_out_native_assets_20260922.json), which is validated against the local RoboCasa registry before evaluation.

| Task | Shared held-out native fixtures | Rollouts per arm |
| --- | --- | ---: |
| OpenElectricKettleLid | ElectricKettle001, ElectricKettle004, ElectricKettle015, ElectricKettle016, ElectricKettle018, ElectricKettle019, ElectricKettle023, ElectricKettle024, ElectricKettle025 | 450 |
| OpenFridgeDrawer | Refrigerator034, Refrigerator042, Refrigerator045, Refrigerator049, Refrigerator053, Refrigerator054, Refrigerator055, Refrigerator056, Refrigerator057, Refrigerator058, Refrigerator067 | 550 |
| OpenStandMixerHead | StandMixer004, StandMixer005, StandMixer010, StandMixer011, StandMixer014, StandMixer017, StandMixer019, StandMixer021, StandMixer024, StandMixer027, StandMixer029, StandMixer030 | 600 |
| OpenToasterOvenDoor | ToasterOven009, ToasterOven017, ToasterOven039, ToasterOven049, ToasterOven062 | 250 |
| SlideDishwasherRack | Dishwasher043, Dishwasher044, Dishwasher062, Dishwasher067 | 200 |
| SlideOvenRack | Oven031, Oven036, Oven037, Oven038, Oven046, Oven047, Oven050, Oven052, Oven054 | 450 |
| TurnOnStove | Stove068 | 50 |
| TurnSinkSpout | Sink003, Sink014, Sink015, Sink017, Sink027, Sink030, Sink037, Sink046, Sink047, Sink048, Sink051, Sink053 | 600 |
| **All tasks** | **63 fixtures shared by every arm** | **3,150** |

After a job succeeds, its compact result is stored at:

```text
<output-root>/results/<task>/<arm>/seed-0.json
```

It contains `num_successes`, `num_rollouts`, `success_rate`, the checkpoint path, and the detailed evaluation `stats.json` path. The detailed per-fixture outcomes remain under:

```text
<output-root>/evaluations/<task>/<arm>/seed-0/<checkpoint-step>/stats.json
```

## Entry points

- [24-job manifest](examples/robocasa/robocasa365/configs/jobs_seed0.tsv)
- [Single-card train-and-evaluate runner](examples/robocasa/robocasa365/scripts/run_task_arm.py)
- [Frozen study configuration](examples/robocasa/robocasa365/configs/robocasa365_20260922.json)
- [Frozen held-out fixture list](examples/robocasa/robocasa365/configs/held_out_native_assets_20260922.json)
