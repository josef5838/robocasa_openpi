# RoboCasa365 Pi0.5 three-arm study

This directory is the self-contained experiment entry point in the RoboCasa OpenPI fork. It compares `native`, `native_plus_ours`, and `ours` for a single selected task. The arm dataset is the only experimental variable.

The immutable contract is in [`configs/robocasa365_20260922.json`](configs/robocasa365_20260922.json): Pi0.5 base weights, global batch size 64, peak learning rate `5e-5`, PaliGemma LoRA rank 16, action-expert LoRA rank 32, trainable vision tower, and one fixed language instruction derived from the selected task name. The runner rejects changed contract values.

## Dataset composition

The table is derived from the frozen manifests in `robocasa365-current-20260922`. Every asset contributes 50 demonstrations. `native_plus_ours` is the exact deterministic concatenation of the two source manifests; it does not duplicate converted data on disk.

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

## Install

Clone your fork, enter it, and create the locked environment:

```bash
git clone git@github.com:YOUR_ACCOUNT/robocasa_openpi.git
cd robocasa_openpi
uv sync --frozen
```

This RoboCasa fork expects the `robocasa` Python package to be importable. In the benchmark development environment it is already present; otherwise install the matching local checkout into this environment:

```bash
uv run python -m pip install -e /path/to/robocasa
```

If `uv sync` tries to build PyAV and fails on a machine with incompatible FFmpeg development headers, use an already working OpenPI environment for release commands instead:

```bash
export PYTHON=/path/to/openpi/.venv/bin/python
```

The publication, download, training, and evaluation shell wrappers honor this variable.

## Publish and download one self-contained data release

Do not commit dataset data to this Git repository. The local [`data/`](data/) directory is Git-ignored and is only a convenient download destination. Publish one Hugging Face *dataset* repository containing deterministic archive shards, manifests, and `normalization_artifacts/full_population/`. `bundle_release.py` packs complete converted asset trees into checksum-indexed `bundles/**`; the publisher uploads those bundles, `BUNDLE_INDEX.json`, manifests, normalization artifacts, `RELEASE.json`, and `README.md`—not an expanded `converted/**` tree, raw recordings, logs, or caches.

First validate the expanded source release, then build the deterministic bundles. The builder validates every asset used by this frozen study, verifies archive checksums by extracting into a temporary directory, and preserves the existing manifest paths:

```bash
examples/robocasa/robocasa365/scripts/bundle_release.py \
  --source-root /path/to/pi05_current_raw_20260922 \
  --output-root /path/to/robocasa365-bundled-release
```

After reviewing the bundle preflight, publish with an authenticated `HF_TOKEN`. Use `--replace-remote` only when intentionally replacing a prior release tree in the same repository:

```bash
export HF_TOKEN=hf_...
examples/robocasa/robocasa365/scripts/publish_dataset.sh \
  --data-root /path/to/robocasa365-bundled-release \
  --repo-id ORG/robocasa365-current-20260922 \
  --private --replace-remote --upload
```

Bundling reduces the release from tens of thousands of small Hub objects to a few dozen archive files. The downloader verifies every archive SHA-256 and extracts the original `converted/` tree locally before validation or training.

Copy the resulting immutable Hugging Face commit SHA into a local, ignored release config, then download the complete training release in one command:

```bash
cp examples/robocasa/robocasa365/configs/data_release.example.json \
  examples/robocasa/robocasa365/configs/data_release.json
# Edit repo_id and revision (a 40-character commit SHA).
examples/robocasa/robocasa365/scripts/download_dataset.sh \
  --config examples/robocasa/robocasa365/configs/data_release.json \
  examples/robocasa/robocasa365/data/release
```

The downloaded release contains bundle archives, manifests, normalization artifacts, and `RELEASE.json`; the download command verifies and extracts `converted/` automatically. Validate a task before training:

```bash
examples/robocasa/robocasa365/scripts/run_three_arms.sh validate \
  --task TurnSinkSpout \
  --data-root examples/robocasa/robocasa365/data/release
```

All three checks must report `missing_or_invalid: 0`.

## Normalization artifacts and the combined arm

Each arm has its own deterministic, full-population normalization artifact:

```text
normalization_artifacts/full_population/<task>/<arm>/norm_stats.json
```

The trainer loads that path directly and snapshots it into the output checkpoint. It does not require, use, or create symlinks. The Hugging Face release contains all converted source assets once; `native_plus_ours` remains a manifest over those source assets and has its own uploaded stats. The publisher rejects a stale artifact, an invalid source asset, or a combined manifest that differs from exact `native + ours`.

If you need to regenerate artifacts before publishing, use the exact deterministic implementation:

```bash
uv run python examples/robocasa/robocasa365/scripts/compute_norm_stats.py \
  --data-root examples/robocasa/robocasa365/data/release \
  --study examples/robocasa/robocasa365/configs/robocasa365_20260922.json
```

## Pretrained Pi0.5 weights

The fixed base URI is `gs://openpi-assets/checkpoints/pi05_base/params`. The trainer downloads it on first use. To prefetch it and verify access:

```bash
examples/robocasa/robocasa365/scripts/prefetch_pi05_base.sh
```

## Train all three arms for one task

Use one task and seed for the complete three-arm comparison. `native_plus_ours` uses deterministic concatenation; the other two use their single source. The following uses precomputed release stats automatically:

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

The frozen design runs each task and arm once with seed `0`. Add `--resume` to the train command to resume an interrupted arm. For each invocation, the runner records the study digest, manifests, source revision, task, arm, seed, and relevant environment settings under `<output-root>/runs/`.

For a nonstandard location of uploaded norm stats, add `--norm-stats-root /path/to/full_population` to either command.
