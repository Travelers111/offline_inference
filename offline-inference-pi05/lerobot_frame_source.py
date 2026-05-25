#!/usr/bin/env python
"""Frame source helpers for pi0.5 LeRobot offline-inference visualizers."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "lerobot" / "src").exists():
            return candidate
    return start.parents[1]


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def install_numpy_pickle_compat() -> None:
    """Allow numpy-2 pickles to be read by environments exposing numpy.core."""
    if "numpy._core" not in sys.modules and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
    try:
        import numpy.core.multiarray as multiarray
        import numpy.core.numeric as numeric

        sys.modules.setdefault("numpy._core.multiarray", multiarray)
        sys.modules.setdefault("numpy._core.numeric", numeric)
    except Exception:
        pass


def image_to_uint8_hwc(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.dtype != torch.uint8:
            tensor = (tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
            return tensor[:3].permute(1, 2, 0).numpy()
        return tensor.numpy().astype(np.uint8, copy=False)

    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).round().astype(np.uint8)
    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array[:3], (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    return array.astype(np.uint8, copy=False)


def camera_label(camera: str) -> str:
    return camera.split(".")[-1]


def resize_nearest_rgb(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.size == 0:
        return image
    height, width = image.shape[:2]
    if height == target_height:
        return image
    scale = target_height / max(height, 1)
    new_width = max(1, int(round(width * scale)))
    y_idx = np.clip(np.round(np.linspace(0, height - 1, target_height)).astype(int), 0, height - 1)
    x_idx = np.clip(np.round(np.linspace(0, width - 1, new_width)).astype(int), 0, width - 1)
    return image[y_idx][:, x_idx]


def camera_strip(frames: list[np.ndarray], target_height: int = 220) -> np.ndarray | None:
    if not frames:
        return None
    resized = [resize_nearest_rgb(frame, target_height) for frame in frames if frame is not None]
    if not resized:
        return None
    pad = np.full((target_height, 8, 3), 255, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for frame in resized:
        if parts:
            parts.append(pad)
        parts.append(frame)
    return np.concatenate(parts, axis=1)


class EpisodeImageSource:
    """Reads visualization frames from images.pkl or directly from the LeRobot dataset."""

    def __init__(
        self,
        saved_images: dict[str, np.ndarray] | None = None,
        dataset: Any | None = None,
        dataset_from_index: int = 0,
        cameras: list[str] | None = None,
    ) -> None:
        self.saved_images = saved_images or {}
        self.dataset = dataset
        self.dataset_from_index = int(dataset_from_index)
        self.cameras = cameras or sorted(self.saved_images)

    @classmethod
    def from_episode_dir(cls, episode_dir: Path, trajectory_data: dict[str, Any]) -> "EpisodeImageSource | None":
        metadata = {}
        metadata_path = episode_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())

        images_path = episode_dir / "images.pkl"
        if images_path.exists():
            install_numpy_pickle_compat()
            with images_path.open("rb") as f:
                saved_images = pickle.load(f)
            cameras = list(metadata.get("cameras") or sorted(saved_images))
            return cls(saved_images=saved_images, cameras=cameras)

        dataset_root = trajectory_data.get("source_dataset_root")
        if not dataset_root:
            return None

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            root = Path(dataset_root).expanduser().resolve()
            repo_id = metadata.get("dataset_repo_id") or f"local/{root.name}"
            dataset = LeRobotDataset(
                repo_id,
                root=root,
                video_backend=metadata.get("video_backend", "pyav"),
                download_videos=False,
                return_uint8=False,
            )
            cameras = metadata.get("cameras") or [
                key for key in dataset.meta.features if key.startswith("observation.images.")
            ]
            return cls(
                dataset=dataset,
                dataset_from_index=int(trajectory_data.get("dataset_from_index", 0)),
                cameras=list(cameras),
            )
        except Exception as exc:
            print(f"[WARN] Could not open LeRobot image source for {episode_dir.name}: {exc}")
            return None

    def available_cameras(self) -> list[str]:
        return list(self.cameras)

    def get_frames(self, timestep: int) -> list[tuple[str, np.ndarray]]:
        frames: list[tuple[str, np.ndarray]] = []
        if self.saved_images:
            for camera in self.cameras:
                values = self.saved_images.get(camera)
                if values is not None and 0 <= timestep < len(values):
                    frames.append((camera_label(camera), image_to_uint8_hwc(values[timestep])))
            return frames

        if self.dataset is None:
            return frames

        item = self.dataset[self.dataset_from_index + int(timestep)]
        for camera in self.cameras:
            if camera in item:
                frames.append((camera_label(camera), image_to_uint8_hwc(item[camera])))
        return frames
