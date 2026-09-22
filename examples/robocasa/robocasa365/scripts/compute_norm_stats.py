#!/usr/bin/env python3
"""Compute deterministic release normalization artifacts without decoding camera video.

This implements the pinned RoboCasa Pi0.5 loader's state/action transforms and
50-step terminal-padded action windows. Unlike the fork's stochastic mixture
sampler, each valid source timestep appears exactly once in the population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize

STATE_ORDER = np.r_[np.arange(7, 14), np.arange(0, 7), np.arange(14, 16)]
ACTION_ORDER = np.r_[np.arange(5, 12), np.arange(0, 5)]
MODEL_DIM = 32
ACTION_HORIZON = 50


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vectors(parquet: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(parquet, columns=["observation.state", "action"])
    state = np.asarray(table["observation.state"].combine_chunks().values).reshape(-1, 16)[:, STATE_ORDER]
    action = np.asarray(table["action"].combine_chunks().values).reshape(-1, 12)[:, ACTION_ORDER]
    return np.pad(state, ((0, 0), (0, MODEL_DIM - state.shape[1]))), np.pad(
        action, ((0, 0), (0, MODEL_DIM - action.shape[1]))
    )


def compute(data_root: Path, task: str, arm: str, output_root: Path, overwrite: bool) -> None:
    manifest_path = data_root / "manifests" / "tasks" / task / f"{arm}.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("task") != task or manifest.get("arm") != arm:
        raise ValueError(f"Manifest identity does not match: {manifest_path}")
    destination = output_root / task / arm
    stats_path = destination / "norm_stats.json"
    if stats_path.exists() and not overwrite:
        print(f"exists, skipped: {stats_path}")
        return
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    episode_count = 0
    for entry in manifest["datasets"]:
        dataset = data_root / entry["lerobot_target"]
        for parquet in sorted((dataset / "data").rglob("*.parquet")):
            state, action = vectors(parquet)
            if len(action) == 0:
                raise ValueError(f"Empty episode: {parquet}")
            windows = np.minimum(np.arange(len(action))[:, None] + np.arange(ACTION_HORIZON), len(action) - 1)
            stats["state"].update(state)
            stats["actions"].update(action[windows])
            episode_count += 1
    result = {key: value.get_statistics() for key, value in stats.items()}
    normalize.save(destination, result)
    metadata = {
        "schema_version": 1,
        "method": "deterministic_full_population_parquet_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "task": task,
        "arm": arm,
        "manifest_sha256": sha256(manifest_path),
        "episode_count": episode_count,
        "state_vector_count": stats["state"]._count,
        "action_vector_count": stats["actions"]._count,
        "state_dim": MODEL_DIM,
        "action_dim": MODEL_DIM,
        "action_horizon": ACTION_HORIZON,
        "state_transform": "[eef_position,eef_rotation,base_position,base_rotation,gripper], then zero-pad",
        "action_transform": "[eef_position,eef_rotation,gripper,base_motion,control_mode], then zero-pad",
        "terminal_action_padding": "repeat_last",
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": task, "arm": arm, "episodes": episode_count, "output": str(stats_path)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--study", type=Path, default=Path("configs/robocasa365_20260922.json"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--task", action="append")
    parser.add_argument("--arm", action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    tasks = args.task or study["tasks"]
    arms = args.arm or study["arms"]
    for task in tasks:
        if task not in study["tasks"]:
            parser.error(f"task is outside frozen study: {task}")
    for arm in arms:
        if arm not in study["arms"]:
            parser.error(f"arm is outside frozen study: {arm}")
    output_root = args.output_root or args.data_root / "normalization_artifacts" / "full_population"
    for task in tasks:
        for arm in arms:
            compute(args.data_root, task, arm, output_root, args.overwrite)


if __name__ == "__main__":
    main()
