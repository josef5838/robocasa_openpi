#!/usr/bin/env python3
"""Preflight and publish the bundled RoboCasa365 Hugging Face release.

Build the release first with ``bundle_release.py``. This publisher uploads only
archive shards plus the manifests, normalization artifacts, release metadata,
and dataset card; it never uploads the expanded ``converted/`` tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from openpi.robocasa365 import study


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def check_normalization_artifacts(data_root: Path, frozen_study: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    norm_root = data_root / "normalization_artifacts/full_population"
    for task in frozen_study["tasks"]:
        for arm in frozen_study["arms"]:
            manifest = data_root / "manifests/tasks" / task / f"{arm}.json"
            artifact_dir = norm_root / task / arm
            stats = artifact_dir / "norm_stats.json"
            metadata = artifact_dir / "metadata.json"
            if not stats.is_file() or not metadata.is_file():
                errors.append(f"missing normalization artifact for {task}/{arm}")
                continue
            info = read_json(metadata)
            if info.get("task") != task or info.get("arm") != arm:
                errors.append(f"normalization identity mismatch: {metadata}")
            if info.get("manifest_sha256") != sha256_file(manifest):
                errors.append(f"normalization artifact is stale for manifest: {manifest}")
            if arm == "native_plus_ours":
                combined = read_json(manifest).get("datasets")
                native = read_json(manifest.with_name("native.json")).get("datasets")
                ours = read_json(manifest.with_name("ours.json")).get("datasets")
                if combined != native + ours:
                    errors.append(f"combined manifest is not exact native + ours: {manifest}")
    return errors


def check_bundle_index(data_root: Path, frozen_study: dict[str, Any]) -> tuple[list[str], int]:
    errors: list[str] = []
    index_path = data_root / "BUNDLE_INDEX.json"
    if not index_path.is_file():
        return [f"missing bundle index: {index_path}"], 0
    index = read_json(index_path)
    bundles = index.get("bundles")
    if index.get("schema_version") != 1 or index.get("archive_format") != "tar" or not isinstance(bundles, list):
        return [f"invalid bundle index: {index_path}"], 0

    asset_roots: set[str] = set()
    seen_archives: set[str] = set()
    for bundle in bundles:
        archive_name = bundle.get("path")
        expected_sha256 = bundle.get("sha256")
        expected_size = bundle.get("size_bytes")
        roots = bundle.get("asset_roots")
        if not isinstance(archive_name, str) or not isinstance(expected_sha256, str) or not isinstance(expected_size, int):
            errors.append(f"invalid bundle entry: {bundle}")
            continue
        archive_path = (data_root / archive_name).resolve()
        if archive_name in seen_archives or not archive_path.is_relative_to(data_root.resolve()):
            errors.append(f"unsafe or duplicate bundle path: {archive_name}")
            continue
        seen_archives.add(archive_name)
        if not archive_path.is_file() or archive_path.stat().st_size != expected_size:
            errors.append(f"missing or size-mismatched bundle: {archive_name}")
        elif sha256_file(archive_path) != expected_sha256:
            errors.append(f"checksum mismatch for bundle: {archive_name}")
        if not isinstance(roots, list) or not all(isinstance(root, str) for root in roots):
            errors.append(f"invalid asset roots for bundle: {archive_name}")
        else:
            asset_roots.update(roots)

    for task in frozen_study["tasks"]:
        for arm in frozen_study["arms"]:
            for entry in read_json(data_root / "manifests/tasks" / task / f"{arm}.json").get("datasets", []):
                target = entry.get("lerobot_target")
                if not isinstance(target, str):
                    errors.append(f"manifest has no LeRobot target: {task}/{arm}")
                    continue
                asset_root = Path(target).parent.as_posix()
                if asset_root not in asset_roots:
                    errors.append(f"manifest target is absent from bundles: {target}")
    return errors, len(bundles)


def preflight(data_root: Path, study_path: Path) -> dict[str, Any]:
    frozen_study = study.read_json(study_path)
    required = [
        data_root / "bundles",
        data_root / "BUNDLE_INDEX.json",
        data_root / "manifests",
        data_root / "normalization_artifacts/full_population",
        data_root / "RELEASE.json",
        data_root / "README.md",
    ]
    errors = [f"missing release path: {path}" for path in required if not path.exists()]
    bundle_errors, bundle_count = check_bundle_index(data_root, frozen_study)
    errors.extend(bundle_errors)
    errors.extend(check_normalization_artifacts(data_root, frozen_study))
    release = read_json(data_root / "RELEASE.json") if (data_root / "RELEASE.json").is_file() else {}
    if release.get("bundle_index", {}).get("sha256") != sha256_file(data_root / "BUNDLE_INDEX.json"):
        errors.append("RELEASE.json does not attest to BUNDLE_INDEX.json")
    if errors:
        raise RuntimeError("Bundled Hugging Face release preflight failed:\n" + "\n".join(f"- {error}" for error in errors))
    return {
        "preflight": "passed",
        "bundle_count": bundle_count,
        "release_metadata": str(data_root / "RELEASE.json"),
        "upload_patterns": study.RELEASE_ALLOW_PATTERNS,
    }


def upload(data_root: Path, repo_id: str, *, private: bool, replace_remote: bool) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=data_root,
        allow_patterns=study.RELEASE_ALLOW_PATTERNS,
        delete_patterns="*" if replace_remote else None,
        commit_message="Publish bundled RoboCasa365 release",
    )
    return f"completed upload to {result.repo_url} at commit {result.oid}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--study", type=Path, default=study.DEFAULT_STUDY)
    parser.add_argument("--repo-id", help="Hugging Face dataset repository, e.g. ORG/robocasa365-current-20260922")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--replace-remote", action="store_true", help="Delete all existing remote files in this commit before upload")
    parser.add_argument("--upload", action="store_true", help="Perform the external upload after preflight")
    args = parser.parse_args()
    if args.upload and not args.repo_id:
        parser.error("--upload requires --repo-id")
    summary = preflight(args.data_root, args.study)
    if args.upload:
        summary["upload_result"] = upload(
            args.data_root, args.repo_id, private=args.private, replace_remote=args.replace_remote
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
