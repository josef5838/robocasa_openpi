"""Pinned, task-level RoboCasa365 Pi0.5 study commands.

The study fixes the model contract and task instruction across native,
native_plus_ours, and ours. Dataset locations and their release normalization
artifacts are supplied at runtime; no dataset payload belongs in this Git repo.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STUDY = REPO_ROOT / "examples/robocasa/robocasa365/configs/robocasa365_20260922.json"
# The immutable release deliberately contains only reproducibility inputs, not
# conversion logs, source recordings, or local Hugging Face cache files.
RELEASE_ALLOW_PATTERNS = [
    "bundles/**",
    "BUNDLE_INDEX.json",
    "manifests/**",
    "normalization_artifacts/full_population/**",
    "RELEASE.json",
    "README.md",
]
FROZEN_MODEL_CONTRACT = {
    "model_type": "Pi0Config(pi05=True)",
    "base_weights": "gs://openpi-assets/checkpoints/pi05_base/params",
    "batch_size": 64,
    "peak_learning_rate": 5e-5,
    "paligemma_variant": "gemma_2b_lora",
    "paligemma_lora_rank": 16,
    "action_expert_variant": "gemma_300m_lora",
    "action_expert_lora_rank": 32,
    "vision_tower_trainable": True,
}
REQUIRED_SLICES = {
    "state": {
        "base_position": (0, 3),
        "base_rotation": (3, 7),
        "end_effector_position_relative": (7, 10),
        "end_effector_rotation_relative": (10, 14),
        "gripper_qpos": (14, 16),
    },
    "action": {
        "base_motion": (0, 4),
        "control_mode": (4, 5),
        "end_effector_position": (5, 8),
        "end_effector_rotation": (8, 11),
        "gripper_close": (11, 12),
    },
}


@dataclass(frozen=True)
class FixedTaskInstruction:
    instruction: str

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "prompt": np.asarray(self.instruction)}


@dataclass(frozen=True)
class FixedPromptDataFactory:
    base_factory: object
    instruction: str

    def create(self, assets_dirs: Path, model_config: object):
        data = self.base_factory.create(assets_dirs, model_config)
        transforms = dataclasses.replace(
            data.model_transforms, inputs=(FixedTaskInstruction(self.instruction), *data.model_transforms.inputs)
        )
        return dataclasses.replace(data, model_transforms=transforms)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}: {exc}") from exc


def manifest_path(data_root: Path, task: str, arm: str) -> Path:
    return data_root / "manifests/tasks" / task / f"{arm}.json"


def load_arm(data_root: Path, task: str, arm: str) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path(data_root, task, arm))
    if manifest.get("task") != task or manifest.get("arm") != arm:
        raise ValueError(f"Manifest identity does not match its path: {manifest_path(data_root, task, arm)}")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"Manifest has no datasets: {manifest_path(data_root, task, arm)}")
    return datasets


def validate_asset(data_root: Path, entry: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    target = entry.get("lerobot_target")
    if not isinstance(target, str):
        return ["manifest entry has no lerobot_target"]
    path = data_root / target
    paths = [path / "meta/modality.json", path / "meta/info.json", path / "meta/stats.json"]
    if not all(item.is_file() for item in paths):
        return [f"missing LeRobot metadata under {path}"]
    modality, info, stats = (read_json(item) for item in paths)
    errors = []
    for group, fields in REQUIRED_SLICES.items():
        for field, expected in fields.items():
            value = modality.get(group, {}).get(field)
            actual = (value.get("start"), value.get("end")) if isinstance(value, dict) else None
            if actual != expected:
                errors.append(f"{group}.{field} slice is {actual}, expected {expected}")
    features = info.get("features", {})
    wanted = {"observation.state": [contract["state_dim"]], "action": [contract["action_dim"]]}
    wanted.update({f"observation.images.{camera}": contract["image_size"] for camera in contract["cameras"]})
    for name, shape in wanted.items():
        if features.get(name, {}).get("shape") != shape:
            errors.append(f"{name} shape is {features.get(name, {}).get('shape')}, expected {shape}")
    if info.get("total_episodes") != contract["episodes_per_asset"]:
        errors.append(f"total_episodes is {info.get('total_episodes')}, expected {contract['episodes_per_asset']}")
    if not isinstance(modality.get("annotation", {}).get("human.task_description"), dict):
        errors.append("missing annotation.human.task_description")
    errors.extend(
        f"missing normalization source stats for {name}"
        for name in ("observation.state", "action")
        if name not in stats
    )
    return errors


def validate(data_root: Path, study: dict[str, Any], task: str | None = None, arm: str | None = None) -> dict[str, Any]:
    tasks = [task] if task else study["tasks"]
    arms = [arm] if arm else study["arms"]
    report: dict[str, Any] = {"ready": 0, "missing_or_invalid": 0, "checks": []}
    for current_task in tasks:
        if current_task not in study["tasks"]:
            raise ValueError(f"Task is outside the frozen study: {current_task}")
        for current_arm in arms:
            if current_arm not in study["arms"]:
                raise ValueError(f"Arm is outside the frozen study: {current_arm}")
            failures = [
                {"asset": item.get("asset"), "errors": errors}
                for item in load_arm(data_root, current_task, current_arm)
                if (errors := validate_asset(data_root, item, study["data_contract"]))
            ]
            assets = len(load_arm(data_root, current_task, current_arm))
            report["checks"].append({"task": current_task, "arm": current_arm, "assets": assets, "failures": failures})
            report["ready"] += assets - len(failures)
            report["missing_or_invalid"] += len(failures)
    return report


def require_ready(data_root: Path, study: dict[str, Any], task: str, arm: str) -> None:
    report = validate(data_root, study, task, arm)
    failures = report["checks"][0]["failures"]
    if failures:
        raise RuntimeError(json.dumps({"dataset_not_ready": failures}, indent=2))


def norm_stats_path(norm_stats_root: Path, task: str, arm: str) -> Path:
    return norm_stats_root / task / arm / "norm_stats.json"


def build_config(
    data_root: Path,
    norm_stats_root: Path,
    output_root: Path,
    study: dict[str, Any],
    task: str,
    arm: str,
    seed: int,
    *,
    resume: bool,
):
    if study.get("model_contract") != FROZEN_MODEL_CONTRACT or study["training"].get("batch_size") != 64:
        raise ValueError("Study model contract differs from the required frozen Pi0.5 contract")
    from openpi.models import pi0_config
    from openpi.training import config as configs
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    base = configs.get_config("pi05_libero")
    model = pi0_config.Pi0Config(
        pi05=True,
        max_token_len=200,
        paligemma_variant=FROZEN_MODEL_CONTRACT["paligemma_variant"],
        action_expert_variant=FROZEN_MODEL_CONTRACT["action_expert_variant"],
    )
    datasets = [
        {"path": str((data_root / item["lerobot_target"]).resolve()), "filter_key": None}
        for item in load_arm(data_root, task, arm)
    ]
    name = f"pi05_robocasa365_{task}_{arm}_lora_b64_8k_fixed_instruction"
    factory = configs.LeRobotRobocasaDataConfig(
        repo_id=name,
        data_dirs=datasets,
        dataset_mode="concat" if arm == "native_plus_ours" else "mixture",
        assets=configs.AssetsConfig(assets_dir=str(norm_stats_root.resolve()), asset_id=f"{task}/{arm}"),
    )
    training = study["training"]
    return dataclasses.replace(
        base,
        name=name,
        exp_name=f"seed-{seed}",
        model=model,
        data=FixedPromptDataFactory(factory, study["task_instructions"][task]),
        weight_loader=weight_loaders.CheckpointWeightLoader(FROZEN_MODEL_CONTRACT["base_weights"]),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=FROZEN_MODEL_CONTRACT["peak_learning_rate"], decay_steps=30000, decay_lr=5e-6
        ),
        optimizer=optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=model.get_freeze_filter(),
        seed=seed,
        num_train_steps=training["steps"],
        batch_size=training["batch_size"],
        num_workers=training["num_workers"],
        save_interval=training["save_interval"],
        wandb_enabled=training["wandb_enabled"],
        fsdp_devices=1,
        assets_base_dir=str(output_root / "assets"),
        checkpoint_base_dir=str(output_root / "checkpoints"),
        resume=resume,
        policy_metadata={
            "study": study["name"],
            "release": study["release"],
            "task": task,
            "arm": arm,
            "seed": seed,
            "fixed_instruction": study["task_instructions"][task],
            "model_contract": FROZEN_MODEL_CONTRACT,
        },
    )


def write_run_record(
    output_root: Path, study_path: Path, data_root: Path, task: str, arm: str, seed: int, command: str
) -> Path:
    run_dir = output_root / "runs" / f"{task}-{arm}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifests = [manifest_path(data_root, task, arm)]
    if arm == "native_plus_ours":
        manifests += [manifest_path(data_root, task, name) for name in ("native", "ours")]

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "command": command,
        "task": task,
        "arm": arm,
        "seed": seed,
        "study_path": str(study_path.resolve()),
        "study_sha256": digest(study_path),
        "data_root": str(data_root.resolve()),
        "openpi_revision": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "manifests": [{"path": str(path.resolve()), "sha256": digest(path)} for path in manifests],
        "environment": {
            key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "JAX_COMPILATION_CACHE_DIR", "WANDB_MODE")
        },
    }
    record = run_dir / f"{command}.json"
    atomic_write(record, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return record


def openpi_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"robocasa365_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import OpenPI script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract a bundle without allowing absolute, traversing, or link entries."""
    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = Path(member.name)
        target = (destination / member_path).resolve()
        if member_path.is_absolute() or ".." in member_path.parts or not target.is_relative_to(destination):
            raise ValueError(f"Unsafe bundle member path: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"Unsupported link or device in bundle: {member.name}")
    archive.extractall(destination)


def extract_bundles(destination: Path) -> dict[str, int]:
    """Verify and expand every indexed archive into the normal converted layout."""
    index_path = destination / "BUNDLE_INDEX.json"
    index = read_json(index_path)
    if index.get("schema_version") != 1 or index.get("archive_format") != "tar":
        raise ValueError(f"Unsupported bundle index: {index_path}")
    bundles = index.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise ValueError(f"Bundle index contains no archives: {index_path}")

    extracted = 0
    for bundle in bundles:
        archive_name = bundle.get("path")
        expected_sha256 = bundle.get("sha256")
        expected_size = bundle.get("size_bytes")
        if not isinstance(archive_name, str) or not isinstance(expected_sha256, str) or not isinstance(expected_size, int):
            raise ValueError(f"Invalid bundle entry in {index_path}: {bundle}")
        archive_path = (destination / archive_name).resolve()
        if not archive_path.is_relative_to(destination) or not archive_path.is_file():
            raise FileNotFoundError(f"Missing release bundle: {archive_name}")
        if archive_path.stat().st_size != expected_size:
            raise ValueError(f"Bundle size mismatch: {archive_name}")
        if sha256_file(archive_path) != expected_sha256:
            raise ValueError(f"Bundle SHA-256 mismatch: {archive_name}")
        with tarfile.open(archive_path, mode="r:") as archive:
            _safe_extract(archive, destination)
        extracted += 1
    return {"bundles_extracted": extracted}


def download(repo_id: str, revision: str, destination: Path) -> None:
    if (
        not revision
        or len(revision) not in range(40, 65)
        or any(char not in "0123456789abcdefABCDEF" for char in revision)
    ):
        raise ValueError("--revision must be an immutable 40-64 character Git commit SHA")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
        allow_patterns=RELEASE_ALLOW_PATTERNS,
    )
    extraction = extract_bundles(destination)
    atomic_write(
        destination / ".pi05-data-source.json",
        json.dumps({"repo_id": repo_id, "revision": revision, **extraction}, indent=2) + "\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "config", "train", "download"))
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("PI05_DATA_ROOT"))
    parser.add_argument("--norm-stats-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts/robocasa365")
    parser.add_argument("--task")
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repo-id")
    parser.add_argument("--revision")
    parser.add_argument("--release-config", type=Path)
    args = parser.parse_args()
    if args.command == "download":
        if args.release_config is not None:
            release = read_json(args.release_config)
            args.repo_id = args.repo_id or release.get("repo_id")
            args.revision = args.revision or release.get("revision")
        if not args.repo_id or not args.revision or args.data_root is None:
            parser.error("download requires --repo-id/--revision (or --release-config) and --data-root")
        download(args.repo_id, args.revision, args.data_root)
        return
    if args.data_root is None:
        parser.error("--data-root or PI05_DATA_ROOT is required")
    study = read_json(args.study)
    if args.command == "validate":
        print(json.dumps(validate(args.data_root, study, args.task, args.arm), indent=2))
        return
    if args.task not in study["tasks"] or args.arm not in study["arms"] or args.seed not in study["seeds"]:
        parser.error("train/config requires a frozen --task, --arm, and --seed")
    require_ready(args.data_root, study, args.task, args.arm)
    norm_root = args.norm_stats_root or args.data_root / "normalization_artifacts/full_population"
    stats = norm_stats_path(norm_root, args.task, args.arm)
    if not stats.is_file():
        raise FileNotFoundError(f"Missing release normalization artifact: {stats}")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/pi05-robocasa365-jax")
    config = build_config(
        args.data_root, norm_root, args.output_root, study, args.task, args.arm, args.seed, resume=args.resume
    )
    record = write_run_record(
        args.output_root, args.study, args.data_root, args.task, args.arm, args.seed, args.command
    )
    print(f"run record: {record}")
    if args.command == "config":
        print(config)
    else:
        openpi_script("train").main(config)


if __name__ == "__main__":
    main()
