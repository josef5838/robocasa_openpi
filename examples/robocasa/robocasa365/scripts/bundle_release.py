#!/usr/bin/env python3
"""Build a portable, deterministic archive release for RoboCasa365.

Each archive contains complete ``converted/<task>/<group>/<asset>`` trees.
Extracting every indexed archive restores the exact layout consumed by the
existing manifests and training code while reducing Hub file-count overhead.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any

from openpi.robocasa365 import study

MIB = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def asset_directories(source_root: Path) -> list[Path]:
    converted = source_root / "converted"
    assets = sorted(
        (path for path in converted.glob("*/*/*") if path.is_dir()),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    if not assets:
        raise ValueError(f"No converted asset directories found in {converted}")
    for asset in assets:
        if any(path.is_symlink() for path in asset.rglob("*")):
            raise ValueError(f"Symlinks are not supported in a portable release bundle: {asset}")
    return assets


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def shard_assets(source_root: Path, assets: list[Path], target_bytes: int) -> list[list[Path]]:
    """Greedily pack sorted assets, never mixing task/group boundaries."""
    shards: list[list[Path]] = []
    current: list[Path] = []
    current_key: tuple[str, str] | None = None
    current_size = 0
    for asset in assets:
        relative = asset.relative_to(source_root)
        key = (relative.parts[1], relative.parts[2])
        size = directory_size(asset)
        if current and (key != current_key or current_size + size > target_bytes):
            shards.append(current)
            current = []
            current_size = 0
        current.append(asset)
        current_key = key
        current_size += size
    if current:
        shards.append(current)
    return shards


def normalized_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    return info


def create_archive(source_root: Path, output_root: Path, assets: list[Path], ordinal: int) -> dict[str, Any]:
    first = assets[0].relative_to(source_root)
    task, group = first.parts[1:3]
    archive_relative = Path("bundles") / task / group / f"shard-{ordinal:03d}.tar"
    archive_path = output_root / archive_relative
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(".tar.partial")
    with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT, dereference=False) as archive:
        for asset in assets:
            archive.add(
                asset,
                arcname=asset.relative_to(source_root).as_posix(),
                recursive=True,
                filter=normalized_tarinfo,
            )
    temporary.replace(archive_path)
    return {
        "path": archive_relative.as_posix(),
        "sha256": sha256_file(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "asset_roots": [asset.relative_to(source_root).as_posix() for asset in assets],
    }


def copy_release_metadata(source_root: Path, output_root: Path) -> None:
    for relative in (Path("manifests"), Path("normalization_artifacts/full_population")):
        shutil.copytree(source_root / relative, output_root / relative)
    shutil.copy2(source_root / "README.md", output_root / "README.md")


def release_payload(study_path: Path, frozen_study: dict[str, Any], bundle_index: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "release_layout": {
            "bundles": "deterministic tar shards; extract into the release root before training",
            "converted": "created by archive extraction; manifests retain their original LeRobot targets",
            "manifests": "task/arm manifests; combined manifests reference source assets",
            "normalization_artifacts": "deterministic full-population stats for each frozen task/arm",
        },
        "bundle_index": {
            "path": "BUNDLE_INDEX.json",
            "sha256": sha256_file(bundle_index),
        },
        "study": {
            "name": frozen_study["name"],
            "release": frozen_study["release"],
            "path": "examples/robocasa/robocasa365/configs/robocasa365_20260922.json",
            "sha256": sha256_file(study_path),
            "tasks": frozen_study["tasks"],
            "arms": frozen_study["arms"],
        },
        "normalization_root": "normalization_artifacts/full_population",
        "publication_allow_patterns": study.RELEASE_ALLOW_PATTERNS,
    }


def verify_bundle(output_root: Path, frozen_study: dict[str, Any]) -> dict[str, Any]:
    """Validate archive checksums, extraction safety, and all training inputs."""
    with tempfile.TemporaryDirectory(prefix="robocasa365-bundle-verify-", dir=output_root.parent) as temporary:
        destination = Path(temporary)
        for relative in (Path("BUNDLE_INDEX.json"), Path("manifests"), Path("normalization_artifacts/full_population")):
            source = output_root / relative
            target = destination / relative
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        for archive in (output_root / "bundles").rglob("*.tar"):
            target = destination / archive.relative_to(output_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.hardlink_to(archive)
        extraction = study.extract_bundles(destination)
        report = study.validate(destination, frozen_study)
        if report["missing_or_invalid"]:
            raise RuntimeError(f"Archive extraction failed release validation: {json.dumps(report, indent=2)}")
        return {**extraction, "validated_assets": report["ready"]}


def build(source_root: Path, output_root: Path, study_path: Path, target_mib: int) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("Output root must not be the source root or one of its parents")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle directory: {output_root}")
    frozen_study = study.read_json(study_path)
    source_report = study.validate(source_root, frozen_study)
    if source_report["missing_or_invalid"]:
        raise RuntimeError(f"Source release validation failed: {json.dumps(source_report, indent=2)}")

    assets = asset_directories(source_root)
    shards = shard_assets(source_root, assets, target_mib * MIB)
    output_root.mkdir(parents=True)
    copy_release_metadata(source_root, output_root)
    bundle_entries = [create_archive(source_root, output_root, shard, ordinal) for ordinal, shard in enumerate(shards)]
    index_path = output_root / "BUNDLE_INDEX.json"
    index = {
        "schema_version": 1,
        "archive_format": "tar",
        "target_shard_size_mib": target_mib,
        "source_asset_directories": len(assets),
        "bundles": bundle_entries,
    }
    write_json(index_path, index)
    write_json(output_root / "RELEASE.json", release_payload(study_path, frozen_study, index_path))
    verification = verify_bundle(output_root, frozen_study)
    return {
        "output_root": str(output_root),
        "assets": len(assets),
        "bundles": len(bundle_entries),
        "bundle_bytes": sum(item["size_bytes"] for item in bundle_entries),
        "verification": verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--study", type=Path, default=study.DEFAULT_STUDY)
    parser.add_argument("--target-shard-mib", type=int, default=512)
    args = parser.parse_args()
    if args.target_shard_mib < 64:
        parser.error("--target-shard-mib must be at least 64")
    print(json.dumps(build(args.source_root, args.output_root, args.study, args.target_shard_mib), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
