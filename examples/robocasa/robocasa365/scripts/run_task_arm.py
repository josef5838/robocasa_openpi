#!/usr/bin/env python3
"""Run one frozen RoboCasa365 task/arm job: train, then evaluate on the same GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_STUDY = REPO_ROOT / "examples/robocasa/robocasa365/configs/robocasa365_20260922.json"
SEED = 0


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--task", required=True)
    parser.add_argument("--arm", required=True, choices=("native", "native_plus_ours", "ours", "articraft"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_src = str(REPO_ROOT / "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)
    os.environ["PYTHONPATH"] = repo_src + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    common = [
        "--study", str(args.study),
        "--task", args.task,
        "--arm", args.arm,
        "--data-root", str(args.data_root),
        "--output-root", str(args.output_root),
        "--seed", str(SEED),
    ]

    train_command = [sys.executable, "-m", "openpi.robocasa365.study", "train", *common]
    if args.resume:
        train_command.append("--resume")
    run(train_command)

    run([
        sys.executable,
        "-m",
        "openpi.robocasa365.evaluate",
        "evaluate",
        *common,
    ])

    from openpi.robocasa365.evaluate import latest_checkpoint

    checkpoint = latest_checkpoint(args.output_root, args.task, args.arm, SEED)
    stats_path = args.output_root / "evaluations" / args.task / args.arm / f"seed-{SEED}" / checkpoint.name / "stats.json"
    stats = json.loads(stats_path.read_text())
    held_out = stats["held_out"]
    summary_path = args.output_root / "results" / args.task / args.arm / f"seed-{SEED}.json"
    summary = {
        "task": args.task,
        "arm": args.arm,
        "seed": SEED,
        "checkpoint": str(checkpoint.resolve()),
        "stats_path": str(stats_path.resolve()),
        "num_held_out_assets": len(held_out["assets"]),
        "num_rollouts": held_out["num_episodes"],
        "num_successes": held_out["num_successes"],
        "success_rate": held_out["success_rate"],
    }
    atomic_json(summary_path, summary)
    print(
        "RESULT task={} arm={} successes={}/{} success_rate={}".format(
            args.task, args.arm, summary["num_successes"], summary["num_rollouts"], summary["success_rate"]
        ),
        flush=True,
    )
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
