"""Evaluate RoboCasa365 checkpoints on held-out, structurally matched native assets."""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import dataclasses
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from openpi.robocasa365.study import DEFAULT_STUDY, build_config, load_arm, read_json


CAMERAS = ("robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right")
PROTOCOL_VERSION = 1
# Target fixture registry, native model-name prefix.  Keeping this explicit
# prevents a name match in an unrelated registry from becoming a test asset.
TASK_SPECS = {
    "OpenElectricKettleLid": ("electric_kettle", "ElectricKettle"),
    "OpenFridgeDrawer": ("fridge_bottom_freezer", "Refrigerator"),
    "OpenStandMixerHead": ("stand_mixer", "StandMixer"),
    "OpenToasterOvenDoor": ("toaster_oven", "ToasterOven"),
    "SlideDishwasherRack": ("dishwasher", "Dishwasher"),
    "SlideOvenRack": ("oven", "Oven"),
    "TurnOnStove": ("stove", "Stove"),
    "TurnSinkSpout": ("sink", "Sink"),
}


@dataclass(frozen=True)
class Asset:
    asset_id: str
    model: Path
    sha256: str
    signature: dict[str, list]


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    seed: int
    success: bool
    steps: int
    layout_id: int
    style_id: int
    prompt: str


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def structural_signature(path: Path) -> dict[str, list]:
    """Topology relevant to appliance manipulation; reject incompatible models."""
    root = ET.parse(path).getroot()
    return {
        "joints": sorted((x.get("name"), x.get("type", "hinge")) for x in root.findall(".//joint")),
        "regions": sorted(x.get("name") for x in root.findall(".//geom")
                          if x.get("class") == "region" and any(w in x.get("name", "") for w in ("basin", "slot"))),
    }


def load_registry_assets(assets_root: Path, fixture_key: str, prefix: str) -> dict[str, Asset]:
    import yaml

    registry_path = assets_root / "fixtures/fixture_registry" / f"{fixture_key}.yaml"
    registry = yaml.safe_load(registry_path.read_text())
    result: dict[str, Asset] = {}
    for asset_id, entry in registry.items():
        if not isinstance(asset_id, str) or not asset_id.startswith(prefix) or not isinstance(entry, dict):
            continue
        model = assets_root / entry["xml"]
        model = model / "model.xml" if model.is_dir() else model
        if not model.is_file():
            raise FileNotFoundError(f"Registry model is missing: {model}")
        result[asset_id] = Asset(asset_id, model.resolve(), hashlib.sha256(model.read_bytes()).hexdigest(), structural_signature(model))
    if not result:
        raise ValueError(f"No {prefix} assets in {registry_path}")
    return result


def build_asset_split(data_root: Path, task: str, arm: str, assets_root: Path) -> tuple[list[Asset], list[Asset]]:
    """Label unselected native models in the training variant as test assets."""
    fixture_key, prefix = TASK_SPECS[task]
    candidates = load_registry_assets(assets_root, fixture_key, prefix)
    arm_entries = load_arm(data_root, task, arm)
    training_ids = sorted({item.get("asset") for item in arm_entries if item.get("asset") in candidates})
    # An ``ours`` arm has no native training instances.  Native instances still
    # provide the task's structural variant, but none are excluded from testing.
    reference_ids = training_ids or sorted(
        item["asset"] for item in load_arm(data_root, task, "native") if item.get("asset") in candidates
    )
    if not reference_ids:
        raise ValueError(f"No native {prefix} reference assets for {task}/{arm}")
    signatures = {json.dumps(candidates[asset].signature, sort_keys=True) for asset in reference_ids}
    if len(signatures) != 1:
        raise ValueError(f"Native references do not share one structural variant: {reference_ids}")
    training = [candidates[asset] for asset in training_ids]
    held_out = [asset for name, asset in sorted(candidates.items())
                if name not in set(training_ids) and asset.signature == candidates[reference_ids[0]].signature]
    if not held_out:
        raise ValueError(f"No held-out native asset remains for {task}/{arm}")
    return training, held_out


def recording_path(entry: dict[str, Any], data_root: Path) -> Path:
    """Locate the portable recording bundled beside each converted asset.

    The absolute source path and optional staging ``raw/`` location are kept
    only for author-side compatibility. A downloaded release must be able to
    evaluate without either of them.
    """
    target = entry.get("lerobot_target")
    bundled = data_root / Path(target).parent / "demo.hdf5" if isinstance(target, str) else None
    candidates = (bundled, Path(entry["source_hdf5"]) if isinstance(entry.get("source_hdf5"), str) else None,
                  data_root / entry["raw_hdf5"] if isinstance(entry.get("raw_hdf5"), str) else None)
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No portable recording for manifest asset {entry.get('asset')}")


def training_context(data_root: Path, task: str, arm: str) -> dict[str, Any]:
    """Use precisely the environment configuration recorded for training."""
    import h5py

    env_meta = None
    prompts: collections.Counter[str] = collections.Counter()
    episode_metas: list[dict[str, Any]] = []
    for entry in load_arm(data_root, task, arm):
        with h5py.File(recording_path(entry, data_root), "r") as dataset:
            current = json.loads(dataset["data"].attrs["env_args"])
            if env_meta is not None and current != env_meta:
                raise ValueError(f"{task}/{arm} recordings have different env_args")
            env_meta = current
            for demo in dataset["data"].values():
                prompts[json.loads(demo.attrs["ep_meta"])["lang"]] += 1
                episode_metas.append(json.loads(demo.attrs["ep_meta"]))
    if env_meta is None:
        raise ValueError(f"Empty manifest: {task}/{arm}")
    return {"env_meta": env_meta, "training_prompt_counts": dict(sorted(prompts.items())), "episode_metas": episode_metas}


@contextlib.contextmanager
def fixed_fixture(fixture_key: str, model: Path):
    from robocasa.models.scenes import scene_builder

    previous = scene_builder.FIXTURES[fixture_key]

    class FixedFixture(previous):
        def __init__(self, xml=None, *args, **kwargs):
            del xml
            super().__init__(xml=str(model), *args, **kwargs)

    scene_builder.FIXTURES[fixture_key] = FixedFixture
    try:
        yield
    finally:
        scene_builder.FIXTURES[fixture_key] = previous


def create_environment(env_meta: dict[str, Any]):
    import robocasa  # noqa: F401
    from robocasa.utils.robomimic import robomimic_env_utils
    return robomimic_env_utils.create_env_for_data_processing(
        env_meta=copy.deepcopy(env_meta), camera_names=list(CAMERAS), camera_height=256, camera_width=256,
        reward_shaping=False,
    )


def policy_input(obs: dict[str, Any], prompt: str) -> dict[str, Any]:
    from openpi_client import image_tools
    from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter
    mapped = PandaOmronKeyConverter.map_obs_in_eval(obs)
    state = np.concatenate((mapped["state.end_effector_position_relative"], mapped["state.end_effector_rotation_relative"],
                            mapped["state.base_position"], mapped["state.base_rotation"], mapped["state.gripper_qpos"]))

    def image(camera: str) -> np.ndarray:
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs[f"{camera}_image"]), 224, 224))
    return {"observation/image": image("robot0_agentview_left"), "observation/wrist_image": image("robot0_eye_in_hand"),
            "observation/right_image": image("robot0_agentview_right"), "observation/state": state, "prompt": prompt}


def run_asset(policy: Any, task: str, fixture_key: str, asset: Asset, context: dict[str, Any], output_dir: Path,
              trials: int, seed_offset: int, replan_steps: int, video_mode: str) -> None:
    import imageio.v2 as imageio
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    asset_dir, results_path = output_dir / asset.asset_id, output_dir / asset.asset_id / "episodes.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    if len(results) > trials:
        raise ValueError(f"{results_path} has more than {trials} episodes")
    with fixed_fixture(fixture_key, asset.model):
        env = create_environment(context["env_meta"])
        try:
            for episode in range(len(results), trials):
                episode_seed = seed_offset + int(hashlib.sha256(asset.asset_id.encode()).hexdigest()[:8], 16) + episode
                np.random.seed(episode_seed)
                env.env.rng = np.random.default_rng(episode_seed)
                episode_meta = context["episode_metas"][episode % len(context["episode_metas"])]
                env.env.set_ep_meta(copy.deepcopy(episode_meta))
                obs, plan, success, frames = env.reset(unset_ep_meta=False), collections.deque(), False, []
                meta = env.env.get_ep_meta()
                for step in range(int(get_task_horizon(task))):
                    if not plan:
                        actions = np.asarray(policy.infer(policy_input(obs, meta["lang"]))["actions"])
                        if actions.ndim != 2 or actions.shape[1] != 12 or len(actions) < replan_steps:
                            raise ValueError(f"Invalid policy action chunk: {actions.shape}")
                        plan.extend(actions[:replan_steps])
                    obs, _, _, _ = env.step(np.asarray(plan.popleft()).copy())
                    success = bool(env.is_success()["task"])
                    if video_mode != "none" and (step % 2 == 0 or success):
                        frames.append(np.asarray(obs["robot0_agentview_left_image"]).copy())
                    if success:
                        break
                results.append(dataclasses.asdict(EpisodeResult(episode, episode_seed, success, step + 1,
                    int(meta["layout_id"]), int(meta["style_id"]), meta["lang"])))
                atomic_json(results_path, results)
                if video_mode == "all" or (video_mode == "representative" and not any(x["success"] == success for x in results[:-1])):
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    imageio.mimwrite(asset_dir / f"rollout_{episode:03d}_{'success' if success else 'failure'}.mp4", frames, fps=10)
                logging.info("%s %d/%d success=%s", asset.asset_id, episode + 1, trials, success)
        finally:
            env.env.close()


def summarize(output_dir: Path, assets: list[Asset]) -> dict[str, Any]:
    all_results, per_asset = [], {}
    for asset in assets:
        path = output_dir / asset.asset_id / "episodes.json"
        results = json.loads(path.read_text()) if path.exists() else []
        successes = sum(x["success"] for x in results)
        per_asset[asset.asset_id] = {"num_episodes": len(results), "num_successes": successes,
                                    "success_rate": successes / len(results) if results else None}
        all_results.extend(results)
    successes = sum(x["success"] for x in all_results)
    return {"num_episodes": len(all_results), "num_successes": successes,
            "success_rate": successes / len(all_results) if all_results else None, "assets": per_asset}


def latest_checkpoint(output_root: Path, task: str, arm: str, seed: int) -> Path:
    name = f"pi05_robocasa365_{task}_{arm}_lora_b64_8k_fixed_instruction"
    root = output_root / "checkpoints" / name / f"seed-{seed}"
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "_CHECKPOINT_METADATA").is_file()] if root.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"No committed checkpoint under {root}")
    return max(candidates, key=lambda p: int(p.name))


def asset_record(asset: Asset) -> dict[str, Any]:
    return {"asset_id": asset.asset_id, "model": str(asset.model), "sha256": asset.sha256, "signature": asset.signature}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "evaluate"))
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--robocasa-assets", type=Path)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--assets", nargs="+")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=3000)
    parser.add_argument("--video-mode", choices=("representative", "all", "none"), default="representative")
    args = parser.parse_args()
    if args.num_trials != 50 or args.replan_steps != 5:
        raise ValueError("The frozen protocol requires exactly 50 trials and 5-step replanning")
    study = read_json(args.study)
    if args.task not in study["tasks"] or args.arm not in study["arms"] or args.seed not in study["seeds"]:
        parser.error("task, arm, or seed is outside the frozen study")
    if args.robocasa_assets is None:
        import robocasa
        args.robocasa_assets = Path(robocasa.models.assets_root)
    training, held_out = build_asset_split(args.data_root, args.task, args.arm, args.robocasa_assets)
    if args.assets:
        unknown = set(args.assets) - {x.asset_id for x in held_out}
        if unknown:
            raise ValueError(f"Requested assets are not derived held-out assets: {sorted(unknown)}")
        held_out = [x for x in held_out if x.asset_id in set(args.assets)]
    context = training_context(args.data_root, args.task, args.arm)
    # Planning intentionally precedes training: it proves the held-out split and
    # portable recording context without requiring a checkpoint to exist.
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    if args.command == "evaluate":
        checkpoint = checkpoint or latest_checkpoint(args.output_root, args.task, args.arm, args.seed).resolve()
        if not (checkpoint / "_CHECKPOINT_METADATA").is_file():
            raise FileNotFoundError(f"Not a committed checkpoint: {checkpoint}")
    elif checkpoint is not None and not (checkpoint / "_CHECKPOINT_METADATA").is_file():
        raise FileNotFoundError(f"Not a committed checkpoint: {checkpoint}")
    fixture_key, _ = TASK_SPECS[args.task]
    protocol = {"protocol_version": PROTOCOL_VERSION, "task": args.task, "arm": args.arm, "training_seed": args.seed,
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "data_root": str(args.data_root.resolve()), "fixture_key": fixture_key,
                "training_assets": [asset_record(x) for x in training], "held_out_test_assets": [asset_record(x) for x in held_out],
                "num_trials_per_asset": 50, "replan_steps": 5,
                "preprocessing": "Pi0.5 training transforms; 224px three-camera observations; checkpoint normalization",
                "simulator": "recorded training env_args with only the target fixture XML replaced",
                "training_prompt_counts": context["training_prompt_counts"]}
    if args.command == "plan":
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return
    assert checkpoint is not None
    output_dir = (args.eval_root or args.output_root / "evaluations") / args.task / args.arm / f"seed-{args.seed}" / checkpoint.name
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError(f"Refusing to mix a different protocol into {output_dir}")
    if not protocol_path.exists() and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing nonempty output directory without protocol: {output_dir}")
    if not protocol_path.exists():
        atomic_json(protocol_path, protocol)
    from openpi.policies.policy_config import create_trained_policy
    config = build_config(args.data_root, args.data_root / "normalization_artifacts/full_population", args.output_root,
                          study, args.task, args.arm, args.seed, resume=False)
    policy = create_trained_policy(config, checkpoint)
    for asset in held_out:
        run_asset(policy, args.task, fixture_key, asset, context, output_dir, 50, args.seed_offset, 5, args.video_mode)
        atomic_json(output_dir / "stats.json", {**protocol, "held_out": summarize(output_dir, held_out)})
    atomic_json(output_dir / "stats.json", {**protocol, "held_out": summarize(output_dir, held_out)})
    print(output_dir / "stats.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    main()
