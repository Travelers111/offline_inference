#!/usr/bin/env python3
"""Image frame loading helpers for easy-mirro offline visualizers."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - visualizers can still run without images.
    cv2 = None

try:
    import h5py
except Exception:  # pragma: no cover - visualizers can still run without images.
    h5py = None


IMAGE_PREFIX = "/observations/images/"


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


def load_pickle(path: Path) -> Any:
    install_numpy_pickle_compat()
    with path.open("rb") as f:
        return pickle.load(f)


def load_metadata(episode_dir: Path) -> dict[str, Any]:
    json_path = episode_dir / "metadata.json"
    if json_path.exists():
        return json.loads(json_path.read_text())
    pkl_path = episode_dir / "metadata.pkl"
    if pkl_path.exists():
        return load_pickle(pkl_path)
    return {}


def decode_hdf5_image(raw: Any, compressed: bool) -> np.ndarray | None:
    """Decode one HDF5 image frame to the same BGR-like array used by eval."""
    if isinstance(raw, (bytes, bytearray, memoryview)):
        raw = np.frombuffer(raw, dtype=np.uint8)
    arr = np.asarray(raw)
    encoded = compressed or (arr.ndim == 1 and arr.dtype == np.uint8)
    if encoded:
        if cv2 is None:
            return None
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        image = arr
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 4 and cv2 is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if np.nanmax(image) <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def bgr_to_display_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return frame[..., ::-1]
    return frame


def resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    if frame.shape[0] == target_height:
        return frame
    scale = target_height / max(frame.shape[0], 1)
    target_width = max(1, int(round(frame.shape[1] * scale)))
    if cv2 is not None:
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    row_idx = np.linspace(0, frame.shape[0] - 1, target_height).astype(int)
    col_idx = np.linspace(0, frame.shape[1] - 1, target_width).astype(int)
    return frame[row_idx][:, col_idx]


def camera_strip(frames: list[np.ndarray], target_height: int = 180, gap: int = 8) -> np.ndarray | None:
    if not frames:
        return None
    resized = [resize_to_height(frame, target_height) for frame in frames]
    if len(resized) == 1:
        return resized[0]
    spacer = np.full((target_height, gap, 3), 245, dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for idx, frame in enumerate(resized):
        if idx:
            pieces.append(spacer)
        pieces.append(frame)
    return np.concatenate(pieces, axis=1)


class EpisodeImageSource:
    """Read only the model-used camera frames for one evaluated episode."""

    def __init__(self, episode_dir: Path, trajectory_data: dict[str, Any] | None = None):
        self.episode_dir = Path(episode_dir)
        self.trajectory_data = trajectory_data or {}
        self.metadata = load_metadata(self.episode_dir)
        self._images_pkl: dict[str, np.ndarray] | None = None
        self._h5 = None
        self._compressed = False
        self.source_hdf5_path = self._resolve_source_hdf5_path()
        self.camera_names = self._resolve_camera_names()
        self._load_images_pkl_if_present()

    def _resolve_source_hdf5_path(self) -> Path | None:
        path = self.metadata.get("source_hdf5_path") or self.trajectory_data.get("source_hdf5_path")
        if not path:
            return None
        resolved = Path(path).expanduser()
        return resolved if resolved.exists() else None

    def _resolve_camera_names(self) -> list[str]:
        names = self.metadata.get("cameras") or self.trajectory_data.get("cameras") or []
        return [str(name) for name in names if name]

    def _load_images_pkl_if_present(self) -> None:
        images_path = self.episode_dir / "images.pkl"
        if not images_path.exists():
            return
        loaded = load_pickle(images_path)
        self._images_pkl = {str(key): value for key, value in loaded.items() if value is not None}
        if not self.camera_names:
            self.camera_names = sorted(self._images_pkl)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()

    def _open_hdf5(self):
        if self._h5 is not None:
            return self._h5
        if h5py is None or self.source_hdf5_path is None:
            return None
        self._h5 = h5py.File(self.source_hdf5_path, "r")
        self._compressed = bool(self._h5.attrs.get("compress", False))
        return self._h5

    def available_cameras(self) -> list[str]:
        available = []
        if self._images_pkl is not None:
            available.extend([camera for camera in self.camera_names if camera in self._images_pkl])
            return available
        root = self._open_hdf5()
        if root is None:
            return []
        for camera in self.camera_names:
            if f"{IMAGE_PREFIX}{camera}" in root:
                available.append(camera)
        return available

    def get_frame(self, camera: str, timestep: int) -> np.ndarray | None:
        if self._images_pkl is not None and camera in self._images_pkl:
            frames = self._images_pkl[camera]
            if len(frames) == 0:
                return None
            return frames[min(max(int(timestep), 0), len(frames) - 1)]

        root = self._open_hdf5()
        if root is None:
            return None
        key = f"{IMAGE_PREFIX}{camera}"
        if key not in root:
            return None
        dataset = root[key]
        if len(dataset) == 0:
            return None
        idx = min(max(int(timestep), 0), len(dataset) - 1)
        return decode_hdf5_image(dataset[idx], self._compressed)

    def get_frames(self, timestep: int) -> list[tuple[str, np.ndarray]]:
        frames = []
        for camera in self.available_cameras():
            frame = self.get_frame(camera, timestep)
            if frame is not None:
                frames.append((camera, frame))
        return frames
