#!/usr/bin/env python
"""Run metadata helpers for pi0.5 offline-inference visualizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def checkpoint_display_name(checkpoint: str | None) -> str:
    if not checkpoint:
        return "unknown"
    path = Path(checkpoint)
    if path.name == "pretrained_model":
        return path.parent.name
    return path.name or str(path)


def load_run_info(data_dir: Path) -> dict[str, Any]:
    metrics_path = data_dir / "metrics.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text())
            return {
                "checkpoint": data.get("checkpoint"),
                "checkpoint_name": checkpoint_display_name(data.get("checkpoint")),
                "dataset_root": data.get("dataset_root"),
                "cameras": data.get("cameras"),
                "camera_source": data.get("camera_source"),
            }
        except Exception as exc:
            print(f"[WARN] Could not read run metadata from {metrics_path}: {exc}")

    return {
        "checkpoint": None,
        "checkpoint_name": data_dir.name,
        "dataset_root": None,
        "cameras": None,
        "camera_source": None,
    }


def load_episode_metadata(episode_dir: Path) -> dict[str, Any]:
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text())
    except Exception as exc:
        print(f"[WARN] Could not read episode metadata from {metadata_path}: {exc}")
        return {}


def merge_episode_info(run_info: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    merged = dict(run_info)
    if metadata.get("checkpoint"):
        merged["checkpoint"] = metadata["checkpoint"]
        merged["checkpoint_name"] = checkpoint_display_name(metadata["checkpoint"])
    if metadata.get("cameras"):
        merged["cameras"] = metadata["cameras"]
    if metadata.get("camera_source"):
        merged["camera_source"] = metadata["camera_source"]
    return merged


def short_run_title(run_info: dict[str, Any]) -> str:
    title = f"CKPT: {run_info.get('checkpoint_name') or 'unknown'}"
    cameras = run_info.get("cameras")
    if cameras:
        camera_names = ", ".join(str(camera).split(".")[-1] for camera in cameras)
        source = run_info.get("camera_source")
        suffix = f" ({source})" if source else ""
        title += f" | Cameras: {camera_names}{suffix}"
    return title
