#!/usr/bin/env python3
"""Offline action-chunk evaluation for easy-mirro Pi0 checkpoints.

The default path intentionally matches the training contract used by
easy-mirro-dual-new_1/scripts/train_pi0_* and easy-mirro-blackboard:

  - input state: /observations/qpos[t], a 10D/20D absolute frame-0 pose
  - label source: /action[t:t+chunk_size], absolute future actions
  - delta checkpoints: compare model output to chunk-wise SE3 delta,
    inv(T_qpos[t]) @ T_action[t+k]
  - absolute checkpoints: compare model output directly to HDF5 /action chunks
  - all model outputs are unnormalized with the checkpoint dataset_stats.pkl

Raw collection HDF5 files without /observations/qpos and /action are rejected by
default because converting them here can silently diverge from the training
preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_EASY_MIRRO_ROOT = (
    WORKSPACE_ROOT / "easy-mirro-dual-new_1"
    if (WORKSPACE_ROOT / "easy-mirro-dual-new_1").exists()
    else WORKSPACE_ROOT / "easy-mirro-fz-zeros"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "offline_inference_output"
FALLBACK_TASK = "Separate the two connected blocks and place each block into the box that matches its color."
DEFAULT_TASK = "auto"
DEFAULT_CAMERAS = "auto"
IMAGE_FEATURE_PREFIX = "observation.images."
SUPPORTED_ACTION_MODES = {"delta", "absolute"}
ZERO_POSE_GRIPPER_STATE_MODES = {"zero_pose_gripper", "zero_pose_keep_gripper"}
SUPPORTED_STATE_MODES = {"absolute", *ZERO_POSE_GRIPPER_STATE_MODES}
POSE_BLOCK_SIZE = 10
TRAINING_LAYOUT_SOURCES = {"temp/qpos_action", "observations/qpos_action"}
RAW_FRAME0_SOURCES = {"state/6d_rot_frame0_next_action", "state/joint_position_frame0_next_action"}
# Legacy raw state/6d_rot files are not the already-clean
# /observations/qpos training layout. Their xyz values are in a shared raw
# Cartesian basis; matching the checkpoint qpos statistics requires shifting the
# right first frame to the origin and applying the physical left-base offset
# along raw x. Rotations are still expressed relative to the right first frame.
RAW_LEFT_BASE_OFFSET = np.array([0.61, 0.0, 0.0], dtype=np.float64)


def sanitize_experiment_name(value: str) -> str:
    """Make a readable, filesystem-safe experiment name."""
    cleaned = []
    for ch in value.strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("_")
        else:
            cleaned.append("_")
    name = "".join(cleaned).strip("._-")
    return name or "unnamed"


def default_output_dir(checkpoint: str | Path, eval_data_dir: str | Path) -> Path:
    checkpoint_name = sanitize_experiment_name(Path(checkpoint).expanduser().resolve().name)
    data_name = sanitize_experiment_name(Path(eval_data_dir).expanduser().resolve().name)
    return DEFAULT_OUTPUT_ROOT / f"{checkpoint_name}__{data_name}"


def parse_episode_selection(value: str | None) -> list[int] | None:
    if value is None or value.strip().lower() in {"", "all"}:
        return None
    selected: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            selected.extend(range(int(start), int(end) + 1))
        else:
            selected.append(int(item))
    return sorted(set(selected))


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_cameras(value: str) -> list[str]:
    return [camera.strip() for camera in value.split(",") if camera.strip()]


def camera_names_from_image_features(image_features: list[str]) -> list[str]:
    camera_names: list[str] = []
    for feature in image_features:
        if not feature.startswith(IMAGE_FEATURE_PREFIX):
            raise ValueError(f"Unsupported image feature in checkpoint config: {feature}")
        camera = feature[len(IMAGE_FEATURE_PREFIX) :]
        if not camera:
            raise ValueError(f"Empty camera name in checkpoint image feature: {feature}")
        camera_names.append(camera)
    if not camera_names:
        raise ValueError("Checkpoint config does not define image_features; pass --cameras explicitly.")
    return camera_names


def resolve_camera_names(cameras: str, checkpoint: str | Path) -> tuple[list[str], str]:
    requested = cameras.strip()
    if requested.lower() not in {"", "auto", "checkpoint", "ckpt"}:
        camera_names = parse_cameras(requested)
        if not camera_names:
            raise ValueError("No cameras were requested.")
        return camera_names, "cli"

    pretrained_path, _ = resolve_checkpoint(checkpoint)
    saved_cfg = load_checkpoint_config(pretrained_path)
    image_features = saved_cfg.get("image_features")
    if not isinstance(image_features, list):
        raise ValueError(
            f"Checkpoint config {pretrained_path / 'config.json'} does not contain a list image_features; "
            "pass --cameras cam_left_wrist,cam_right_wrist or another explicit list."
        )
    return camera_names_from_image_features(image_features), "checkpoint"


def add_easy_mirro_to_path(easy_mirro_root: Path) -> None:
    root = easy_mirro_root.expanduser().resolve()
    training_dir = root / "training"
    for path in (root, training_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def find_hdf5_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root, _, filenames in os.walk(data_dir):
        if "pointcloud" in root or "pointclouds" in root:
            continue
        for filename in filenames:
            if not filename.endswith(".hdf5"):
                continue
            if filename.startswith("._") or "features" in filename:
                continue
            files.append(Path(root) / filename)
    return sorted(files)


def resolve_checkpoint(path: str | Path) -> tuple[Path, Path]:
    """Return (pretrained_path, stats_path).

    Accepts either a concrete checkpoint directory containing safetensors, or a
    train output/root directory containing checkpoint-* subdirectories.
    """
    base = Path(path).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {base}")

    if list(base.glob("*.safetensors")):
        pretrained_path = base
    else:
        checkpoints = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
        if not checkpoints:
            raise FileNotFoundError(f"No safetensors or checkpoint-* directories found under {base}")
        pretrained_path = sorted(checkpoints, key=lambda p: int(p.name.split("-")[-1]))[-1]

    stats_candidates = [
        pretrained_path / "dataset_stats.pkl",
        pretrained_path.parent / "dataset_stats.pkl",
        base / "dataset_stats.pkl",
    ]
    for candidate in stats_candidates:
        if candidate.exists():
            return pretrained_path, candidate

    searched = "\n".join(f"  - {candidate}" for candidate in stats_candidates)
    raise FileNotFoundError(f"Could not find dataset_stats.pkl. Searched:\n{searched}")


def load_checkpoint_config(pretrained_path: Path) -> dict[str, Any]:
    config_path = pretrained_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing checkpoint config.json: {config_path}")
    return json.loads(config_path.read_text())


def read_stats(stats_path: Path) -> dict[str, Any]:
    with stats_path.open("rb") as f:
        stats = pickle.load(f)
    required = ("qpos_min", "qpos_max", "action_min", "action_max")
    missing = [key for key in required if key not in stats]
    if missing:
        raise KeyError(f"dataset_stats.pkl missing keys required by mixed normalization: {missing}")
    for key in required:
        arr = np.asarray(stats[key])
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values found in dataset stats key: {key}")
    return stats


def action_mode_from_stats(stats: dict[str, Any]) -> str | None:
    mode = stats.get("action_mode")
    if isinstance(mode, str) and mode in SUPPORTED_ACTION_MODES:
        return mode
    if "use_delta" in stats:
        return "delta" if bool(stats["use_delta"]) else "absolute"
    return None


def resolve_action_mode(requested: str, saved_cfg: dict[str, Any], stats: dict[str, Any]) -> tuple[str, str]:
    requested = (requested or "auto").strip().lower()
    if requested in SUPPORTED_ACTION_MODES:
        return requested, "cli"
    if requested not in {"auto", "checkpoint", "ckpt", ""}:
        raise ValueError(f"Unsupported action mode: {requested!r}. Use auto, delta, or absolute.")

    config_mode = saved_cfg.get("action_mode")
    if config_mode is not None and config_mode not in SUPPORTED_ACTION_MODES:
        raise ValueError(f"Unsupported checkpoint config action_mode={config_mode!r}")

    stats_mode = action_mode_from_stats(stats)
    if config_mode and stats_mode and config_mode != stats_mode:
        raise ValueError(
            "Checkpoint config and dataset_stats disagree on action mode: "
            f"config={config_mode!r}, stats={stats_mode!r}. Refusing to guess."
        )
    if config_mode:
        return str(config_mode), "checkpoint_config"
    if stats_mode:
        return stats_mode, "dataset_stats"
    return "delta", "legacy_default"


def canonical_state_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    mode = str(mode)
    if mode == "absolute":
        return "absolute"
    if mode in ZERO_POSE_GRIPPER_STATE_MODES:
        return "zero_pose_gripper"
    return mode


def resolve_state_mode(saved_cfg: dict[str, Any], stats: dict[str, Any]) -> tuple[str, str]:
    config_mode = saved_cfg.get("state_mode")
    stats_mode = stats.get("state_mode")
    if config_mode is not None and config_mode not in SUPPORTED_STATE_MODES:
        raise ValueError(f"Unsupported checkpoint config state_mode={config_mode!r}")
    if stats_mode is not None and stats_mode not in SUPPORTED_STATE_MODES:
        raise ValueError(f"Unsupported dataset_stats state_mode={stats_mode!r}")
    if config_mode and stats_mode and canonical_state_mode(config_mode) != canonical_state_mode(stats_mode):
        raise ValueError(
            "Checkpoint config and dataset_stats disagree on state mode: "
            f"config={config_mode!r}, stats={stats_mode!r}. Refusing to guess."
        )
    if config_mode:
        return str(config_mode), "checkpoint_config"
    if stats_mode:
        return str(stats_mode), "dataset_stats"
    return "absolute", "legacy_default"


def build_model(
    easy_mirro_root: Path,
    checkpoint: str | Path,
    camera_names: list[str],
    device: str,
    dtype: str,
) -> tuple[Any, dict[str, Any], Path, Path]:
    add_easy_mirro_to_path(easy_mirro_root)
    import models  # noqa: F401  Registers MIRRO with transformers AutoModel.
    from training.utils import load_vla_model

    pretrained_path, stats_path = resolve_checkpoint(checkpoint)
    saved_cfg = load_checkpoint_config(pretrained_path)
    stats = read_stats(stats_path)
    action_mode, _ = resolve_action_mode("auto", saved_cfg, stats)
    state_mode, _ = resolve_state_mode(saved_cfg, stats)

    image_features = [f"{IMAGE_FEATURE_PREFIX}{camera}" for camera in camera_names]
    if saved_cfg.get("image_features") and saved_cfg["image_features"] != image_features:
        print(f"[WARN] camera features differ from checkpoint config:")
        print(f"       checkpoint: {saved_cfg['image_features']}")
        print(f"       requested:  {image_features}")

    init_kwargs = {
        "vla_model_name": saved_cfg.get("vla_model_name", "mirro"),
        "generate_reasoning": False,
        "state_dim": int(saved_cfg.get("state_dim", 20)),
        "action_dim": int(saved_cfg.get("action_dim", 20)),
        "max_state_dim": int(saved_cfg.get("max_state_dim", 32)),
        "max_action_dim": int(saved_cfg.get("max_action_dim", 32)),
        "chunk_size": int(saved_cfg.get("chunk_size", saved_cfg.get("n_action_steps", 50))),
        "n_action_steps": int(saved_cfg.get("n_action_steps", saved_cfg.get("chunk_size", 50))),
        "pretrained_path": str(pretrained_path),
        "resize_imgs_with_padding": tuple(saved_cfg.get("resize_imgs_with_padding", (224, 224))),
        "image_features": image_features,
        "empty_cameras": int(saved_cfg.get("empty_cameras", 0)),
        "attention_implementation": saved_cfg.get("attention_implementation", "eager"),
        "num_steps": int(saved_cfg.get("num_steps", 10)),
        "proj_width": int(saved_cfg.get("proj_width", 1024)),
        "tokenizer_max_length": int(saved_cfg.get("tokenizer_max_length", 96)),
        "norm_mode": saved_cfg.get("norm_mode", stats.get("norm_mode", "mixed")),
        "action_mode": action_mode,
        "state_mode": state_mode,
        "adapt_to_pi_aloha": bool(saved_cfg.get("adapt_to_pi_aloha", False)),
        "use_cache": bool(saved_cfg.get("use_cache", True)),
        "freeze_vision_encoder": bool(saved_cfg.get("freeze_vision_encoder", True)),
        "train_expert_only": bool(saved_cfg.get("train_expert_only", False)),
        "train_state_proj": bool(saved_cfg.get("train_state_proj", True)),
        "stop_gradient": bool(saved_cfg.get("stop_gradient", False)),
    }

    model = load_vla_model(init_kwargs, device=device, data_stats=stats)
    model.eval()
    if dtype == "float32":
        model.to(dtype=torch.float32)
    elif dtype == "bfloat16":
        model.to(dtype=torch.bfloat16)
        model.model.action_out_proj = model.model.action_out_proj.to(torch.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return model, stats, pretrained_path, stats_path


# ======================== SE3 helpers ========================


def rot6d_to_mat(r: np.ndarray) -> np.ndarray:
    a1, a2 = r[..., :3], r[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-2)


def pose9d_to_se3(v: np.ndarray) -> np.ndarray:
    se3 = np.zeros(v.shape[:-1] + (4, 4), dtype=np.float64)
    se3[..., :3, :3] = rot6d_to_mat(v[..., 3:9])
    se3[..., :3, 3] = v[..., :3]
    se3[..., 3, 3] = 1.0
    return se3


def invert_se3(T: np.ndarray) -> np.ndarray:
    Rt = np.swapaxes(T[..., :3, :3], -2, -1)
    Ti = np.zeros_like(T)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    Ti[..., 3, 3] = 1.0
    return Ti


def se3_to_state10(se3: np.ndarray, gripper: float) -> np.ndarray:
    result = np.empty(10, dtype=np.float64)
    result[:3] = se3[:3, 3]
    result[3:6] = se3[0, :3]
    result[6:9] = se3[1, :3]
    result[9] = gripper
    return result


def se3_batch_to_state10(se3: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    result = np.empty((se3.shape[0], 10), dtype=np.float32)
    result[:, :3] = se3[:, :3, 3]
    result[:, 3:6] = se3[:, 0, :3]
    result[:, 6:9] = se3[:, 1, :3]
    result[:, 9] = gripper
    return result


def pose_block_offsets(dim: int) -> range:
    if dim <= 0 or dim % POSE_BLOCK_SIZE != 0:
        raise ValueError(f"Pose/action dimension must be a positive multiple of 10, got {dim}.")
    return range(0, dim, POSE_BLOCK_SIZE)


def state_for_model(qpos: np.ndarray, state_mode: str) -> np.ndarray:
    if state_mode == "absolute":
        return np.asarray(qpos, dtype=np.float32)
    if state_mode not in ZERO_POSE_GRIPPER_STATE_MODES:
        raise ValueError(f"Unsupported state_mode={state_mode!r}")
    model_state = np.zeros_like(qpos, dtype=np.float32)
    for off in pose_block_offsets(qpos.shape[-1]):
        model_state[..., off + 9] = qpos[..., off + 9]
    return model_state


def make_next_state_action(qpos: np.ndarray) -> np.ndarray:
    action = np.array(qpos, copy=True)
    if len(action) > 1:
        action[:-1] = qpos[1:]
    action[-1] = qpos[-1]
    return action.astype(np.float32)


def raw_dual_state_to_frame0(raw_state20: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert raw shared-frame poses to the training frame-0 convention."""
    raw_state20 = np.asarray(raw_state20, dtype=np.float32)
    if raw_state20.ndim != 2 or raw_state20.shape[1] < 20:
        raise ValueError(f"Expected raw dual-arm state shape (T, >=20), got {raw_state20.shape}")

    left_raw = raw_state20[:, :10]
    right_raw = raw_state20[:, 10:20]
    T_left_raw = pose9d_to_se3(left_raw[:, :9])
    T_right_raw = pose9d_to_se3(right_raw[:, :9])
    T_right0 = T_right_raw[0]
    R_right0_inv = T_right0[:3, :3].T
    left_rot_local = np.einsum("ij,tjk->tik", R_right0_inv, T_left_raw[:, :3, :3])
    right_rot_local = np.einsum("ij,tjk->tik", R_right0_inv, T_right_raw[:, :3, :3])

    left_pos_local = left_raw[:, :3].astype(np.float64) + RAW_LEFT_BASE_OFFSET - right_raw[0, :3].astype(np.float64)
    right_pos_local = right_raw[:, :3].astype(np.float64) - right_raw[0, :3].astype(np.float64)

    T_left_local = np.repeat(np.eye(4, dtype=np.float64)[None], len(raw_state20), axis=0)
    T_right_local = np.repeat(np.eye(4, dtype=np.float64)[None], len(raw_state20), axis=0)
    T_left_local[:, :3, :3] = left_rot_local
    T_right_local[:, :3, :3] = right_rot_local
    T_left_local[:, :3, 3] = left_pos_local
    T_right_local[:, :3, 3] = right_pos_local

    qpos = np.zeros((len(raw_state20), 20), dtype=np.float32)
    qpos[:, :10] = se3_batch_to_state10(T_left_local, left_raw[:, 9])
    qpos[:, 10:20] = se3_batch_to_state10(T_right_local, right_raw[:, 9])
    metadata = {
        "raw_transform_mode": "shared_xyz_right0_origin_left_x_offset_relative_rot",
        "raw_frame0_origin_offset": T_right0[:3, 3].astype(float).tolist(),
        "raw_frame0_origin_rotation": np.concatenate([T_right0[0, :3], T_right0[1, :3]]).astype(float).tolist(),
        "raw_left_base_offset": RAW_LEFT_BASE_OFFSET.astype(float).tolist(),
        "right_frame0_xyz_norm": float(np.linalg.norm(qpos[0, 10:13])),
        "right_frame0_rot6d": qpos[0, 13:19].astype(float).tolist(),
    }
    return qpos, metadata


def compute_action_delta(qpos_vec: np.ndarray, action_seq: np.ndarray) -> np.ndarray:
    action_dim = int(action_seq.shape[-1])
    if qpos_vec.shape[-1] < action_dim:
        raise ValueError(f"qpos dim {qpos_vec.shape[-1]} is smaller than action dim {action_dim}.")
    delta = np.zeros_like(action_seq, dtype=np.float32)
    for off in pose_block_offsets(action_dim):
        pose_slice = slice(off, off + 9)
        gripper_idx = off + 9
        T_inv = invert_se3(pose9d_to_se3(qpos_vec[pose_slice]))
        T_delta = np.einsum("ij,kjl->kil", T_inv, pose9d_to_se3(action_seq[:, pose_slice]))
        delta[:, off : off + 3] = T_delta[:, :3, 3]
        delta[:, off + 3 : off + 9] = np.concatenate([T_delta[:, 0, :3], T_delta[:, 1, :3]], axis=-1)
        delta[:, gripper_idx] = action_seq[:, gripper_idx]
    return delta


def delta_chunk_to_absolute(state: np.ndarray, delta_chunk: np.ndarray) -> np.ndarray:
    action_dim = int(delta_chunk.shape[-1])
    if state.shape[-1] < action_dim:
        raise ValueError(f"state dim {state.shape[-1]} is smaller than delta action dim {action_dim}.")
    absolute = np.zeros_like(delta_chunk, dtype=np.float32)
    for off in pose_block_offsets(action_dim):
        pose_slice = slice(off, off + 9)
        gripper_idx = off + 9
        T_state = pose9d_to_se3(state[pose_slice])
        T_delta = pose9d_to_se3(delta_chunk[:, pose_slice])
        T_abs = np.einsum("ij,kjl->kil", T_state, T_delta)
        absolute[:, off : off + 3] = T_abs[:, :3, 3]
        absolute[:, off + 3 : off + 9] = np.concatenate([T_abs[:, 0, :3], T_abs[:, 1, :3]], axis=-1)
        absolute[:, gripper_idx] = delta_chunk[:, gripper_idx]
    return absolute


def pad_vector(state: np.ndarray, max_dim: int = 32) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] > max_dim:
        raise ValueError(f"Cannot pad vector with dim {state.shape[-1]} to max_dim={max_dim}")
    padded = np.zeros(max_dim, dtype=np.float32)
    padded[: state.shape[-1]] = state
    return padded


# ======================== Data and inference ========================


def decode_image(raw: np.ndarray, compressed: bool) -> np.ndarray:
    if compressed and isinstance(raw, (bytes, bytearray, memoryview)):
        raw = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if compressed else raw
    if image is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.astype(np.uint8, copy=False)


def read_episode_task(root: h5py.File, hdf5_path: Path) -> str:
    h5_instruction = root.attrs.get("language_instruction")
    if isinstance(h5_instruction, bytes):
        h5_instruction = h5_instruction.decode("utf-8")
    if h5_instruction:
        return str(h5_instruction)
    task_name = hdf5_path.parent.name
    blackboard_tasks = {"wiping_blackboard_single", "wiping_whiteboard_single", "clean_processed_mirro", "blackboard_testdata"}
    if task_name in blackboard_tasks or "blackboard" in str(hdf5_path).lower() or "whiteboard" in str(hdf5_path).lower():
        return "Pick up the eraser and wipe off the black marker marks on the whiteboard."
    return FALLBACK_TASK


def read_qpos_actions(
    root: h5py.File,
    hdf5_path: Path,
    allow_raw_fallback: bool = False,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    if "temp" in root and "/temp/qpos" in root and "/temp/action" in root:
        return root["/temp/qpos"][()], root["/temp/action"][()], "temp/qpos_action", {}
    if "/observations/qpos" in root and "/action" in root:
        return root["/observations/qpos"][()], root["/action"][()], "observations/qpos_action", {}
    if not allow_raw_fallback:
        raise KeyError(
            f"{hdf5_path} is not in the training evaluation layout. Expected "
            "/observations/qpos and /action. Convert raw collection files with the "
            "same preprocessing used for training before running offline eval."
        )
    if "/state/6d_rot/left" in root and "/state/6d_rot/right" in root:
        raw_state = np.concatenate(
            [root["/state/6d_rot/left"][()], root["/state/6d_rot/right"][()]],
            axis=1,
        )
        qpos, transform_metadata = raw_dual_state_to_frame0(raw_state)
        return qpos, make_next_state_action(qpos), "state/6d_rot_frame0_next_action", transform_metadata
    if "/state/joint_position/left" in root and "/state/joint_position/right" in root:
        raw_state = np.concatenate(
            [root["/state/joint_position/left"][()], root["/state/joint_position/right"][()]],
            axis=1,
        )
        qpos, transform_metadata = raw_dual_state_to_frame0(raw_state)
        return qpos, make_next_state_action(qpos), "state/joint_position_frame0_next_action", transform_metadata
    raise KeyError(f"No supported qpos/action layout found in {hdf5_path}")


def load_episode(
    hdf5_path: Path,
    camera_names: list[str],
    allow_raw_fallback: bool,
    state_dim: int,
    action_dim: int,
    state_mode: str,
    task_arg: str,
) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as root:
        compressed = bool(root.attrs.get("compress", False))
        task = read_episode_task(root, hdf5_path) if task_arg == "auto" else task_arg
        qpos, actions, state_source, transform_metadata = read_qpos_actions(
            root,
            hdf5_path,
            allow_raw_fallback=allow_raw_fallback,
        )

        qpos = np.asarray(qpos, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if qpos.ndim != 2 or actions.ndim != 2:
            raise ValueError(f"qpos/action must be rank-2 arrays: {hdf5_path}, got {qpos.shape} and {actions.shape}")
        if qpos.shape[-1] < state_dim or actions.shape[-1] < action_dim:
            raise ValueError(
                f"Expected qpos/action dims at least {state_dim}/{action_dim}, "
                f"got {qpos.shape[-1]}/{actions.shape[-1]} in {hdf5_path}"
            )
        if state_dim < action_dim:
            raise ValueError(f"state_dim={state_dim} is smaller than action_dim={action_dim}")
        pose_block_offsets(action_dim)
        qpos = qpos[:, :state_dim]
        actions = actions[:, :action_dim]
        episode_len = int(min(len(qpos), len(actions)))
        qpos = qpos[:episode_len]
        actions = actions[:episode_len]
        model_qpos = state_for_model(qpos, state_mode)

        images: dict[str, np.ndarray | None] = {}
        for camera in camera_names:
            key = f"/observations/images/{camera}"
            if key not in root:
                print(f"[WARN] Camera {camera} not found in {hdf5_path}")
                images[camera] = None
                continue
            raw_images = root[key][()]
            decoded = [decode_image(raw_images[t], compressed) for t in range(episode_len)]
            images[camera] = np.stack(decoded, axis=0)

    return {
        "images": images,
        "qpos": qpos,
        "model_qpos": model_qpos,
        "actions": actions,
        "episode_len": episode_len,
        "task": task,
        "state_source": state_source,
        "transform_metadata": transform_metadata,
    }


def preprocess_image_batch(
    episode_images: dict[str, np.ndarray | None],
    camera_names: list[str],
    start: int,
    end: int,
    device: str,
    center_crop: bool,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for camera in camera_names:
        frames = episode_images.get(camera)
        if frames is None:
            continue
        frames = frames[start:end]
        frames = frames.astype(np.float32) / 255.0
        frames = np.transpose(frames, (0, 3, 1, 2))
        tensor = torch.from_numpy(frames).to(device=device, dtype=dtype)
        if center_crop:
            original_size = tensor.shape[-2:]
            ratio = 0.95
            h0 = int(original_size[0] * (1 - ratio) / 2)
            h1 = int(original_size[0] * (1 + ratio) / 2)
            w0 = int(original_size[1] * (1 - ratio) / 2)
            w1 = int(original_size[1] * (1 + ratio) / 2)
            tensor = tensor[..., h0:h1, w0:w1]
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=original_size,
                mode="bilinear",
                antialias=True,
            )
        tensors[camera] = tensor
    return tensors


def build_ground_truth_delta_chunks(
    qpos: np.ndarray,
    actions: np.ndarray,
    chunk_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    gt_action_chunks: list[np.ndarray] = []
    gt_delta_chunks: list[np.ndarray] = []
    valid_lengths: list[int] = []
    episode_len = actions.shape[0]
    action_dim = actions.shape[-1]
    for t in range(episode_len):
        end = min(t + chunk_size, episode_len)
        valid = end - t
        action_chunk = np.zeros((chunk_size, action_dim), dtype=np.float32)
        delta_chunk = np.zeros((chunk_size, action_dim), dtype=np.float32)
        action_chunk[:valid] = actions[t:end]
        delta_chunk[:valid] = compute_action_delta(qpos[t], actions[t:end])
        gt_action_chunks.append(action_chunk)
        gt_delta_chunks.append(delta_chunk)
        valid_lengths.append(valid)
    return gt_action_chunks, gt_delta_chunks, valid_lengths


def run_batched_inference(
    model: Any,
    episode_data: dict[str, Any],
    task: str,
    camera_names: list[str],
    device: str,
    batch_size: int,
    center_crop: bool,
    input_dtype: torch.dtype,
) -> np.ndarray:
    episode_len = episode_data["episode_len"]
    model_qpos = episode_data["model_qpos"]
    chunk_size = int(model.config.n_action_steps)
    action_dim = int(model.config.action_dim)
    max_state_dim = int(getattr(model.config, "max_state_dim", 32))

    missing_cameras = [camera for camera in camera_names if episode_data["images"].get(camera) is None]
    if missing_cameras:
        raise ValueError(
            "Episode is missing camera images required by the model/eval request: "
            f"{missing_cameras}. Requested cameras: {camera_names}"
        )

    states = np.stack([pad_vector(model_qpos[t], max_state_dim) for t in range(episode_len)], axis=0)

    pred_chunks: list[np.ndarray] = []

    for start in tqdm(range(0, episode_len, batch_size), desc="frames", leave=False):
        end = min(start + batch_size, episode_len)
        image_tensors = preprocess_image_batch(
            episode_data["images"],
            camera_names,
            start,
            end,
            device,
            center_crop,
            input_dtype,
        )
        states_tensor = torch.from_numpy(states[start:end]).to(device=device, dtype=input_dtype)
        batch: dict[str, Any] = {
            "observation.state": states_tensor,
            "task": [task] * (end - start),
            "reasoning": None,
            "is_s1": True,
            "is_vl_data": torch.zeros(end - start, dtype=torch.bool, device=device),
            "generate_reasoning": False,
        }
        for camera, tensor in image_tensors.items():
            batch[f"observation.images.{camera}"] = tensor

        batch = model.normalize_inputs(batch)
        state_after_norm = batch.get("observation.state")
        if state_after_norm is not None and not torch.isfinite(state_after_norm).all():
            print("[WARN] Non-finite observation.state after normalize_inputs; applying nan_to_num safeguard.")
            batch["observation.state"] = torch.nan_to_num(state_after_norm, nan=0.0, posinf=10.0, neginf=-10.0)

        images, img_masks = model.prepare_images(batch)
        state = model.prepare_state(batch)
        lang_tokens, lang_masks, _ = model.prepare_language(batch, eval=True)

        with torch.inference_mode():
            normalized_actions, _ = model.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
            if not torch.isfinite(normalized_actions).all():
                print("[WARN] Non-finite actions before unnormalize; applying nan_to_num safeguard.")
                normalized_actions = torch.nan_to_num(normalized_actions, nan=0.0, posinf=1e3, neginf=-1e3)
            actions = model.normalize_targets.unnormalize(normalized_actions)
            actions = actions[:, :, : model.config.action_dim]
            actions = torch.nan_to_num(actions, nan=0.0, posinf=1e3, neginf=-1e3)

        action_np = actions.detach().cpu().float().numpy()
        for batch_index in range(end - start):
            action_chunk = action_np[batch_index, :chunk_size, :action_dim].astype(np.float32)
            pred_chunks.append(action_chunk)

    return np.stack(pred_chunks, axis=0)


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> dict[str, Any]:
    dim = min(predictions.shape[-1], ground_truth.shape[-1])
    diff = predictions[..., :dim] - ground_truth[..., :dim]
    pred_diff = np.diff(predictions[..., :dim], axis=0) if len(predictions) > 1 else np.zeros_like(predictions)
    return {
        "mse": float(np.mean(diff**2)),
        "mae": float(np.mean(np.abs(diff))),
        "per_dim_mse": np.mean(diff**2, axis=0).astype(float).tolist(),
        "max_error": float(np.max(np.abs(diff))),
        "smoothness": float(np.mean(np.var(pred_diff, axis=0))),
    }


def build_trajectory_pairs(
    qpos: np.ndarray,
    model_qpos: np.ndarray,
    gt_actions: np.ndarray,
    pred_chunks: np.ndarray,
    action_mode: str,
    metric_drop_last_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if action_mode not in SUPPORTED_ACTION_MODES:
        raise ValueError(f"Unsupported action_mode={action_mode!r}")
    chunk_size = pred_chunks.shape[1]
    gt_action_chunks, gt_delta_chunks, valid_lengths = build_ground_truth_delta_chunks(qpos, gt_actions, chunk_size)
    pairs: list[dict[str, Any]] = []

    for t in range(gt_actions.shape[0]):
        pair = {
            "timestep": t,
            "observation_state": model_qpos[t].astype(np.float32),
            "raw_observation_qpos": qpos[t].astype(np.float32),
            "ground_truth_action_chunk": gt_action_chunks[t],
            "ground_truth_delta_chunk": gt_delta_chunks[t],
            "valid_length": valid_lengths[t],
        }
        if action_mode == "delta":
            predicted_delta = pred_chunks[t].astype(np.float32)
            predicted_absolute = delta_chunk_to_absolute(qpos[t], predicted_delta).astype(np.float32)
            pair["predicted_delta_chunk"] = predicted_delta
            pair["ground_truth_chunk"] = gt_action_chunks[t]
            pair["predicted_chunk"] = predicted_absolute
            pair["predicted_action_chunk"] = predicted_absolute
        else:
            absolute_pred = pred_chunks[t].astype(np.float32)
            pair["ground_truth_chunk"] = gt_action_chunks[t]
            pair["predicted_chunk"] = absolute_pred
            pair["predicted_action_chunk"] = absolute_pred
        pairs.append(pair)

    metric_end = max(0, gt_actions.shape[0] - metric_drop_last_frames) or gt_actions.shape[0]
    if action_mode == "delta":
        first_step_delta = compute_metrics(
            pred_chunks[:metric_end, 0],
            np.stack([pair["ground_truth_delta_chunk"][0] for pair in pairs[:metric_end]], axis=0),
        )

        valid_pred_delta = []
        valid_gt_delta = []
        for pair in pairs[:metric_end]:
            valid = pair["valid_length"]
            valid_pred_delta.append(pair["predicted_delta_chunk"][:valid])
            valid_gt_delta.append(pair["ground_truth_delta_chunk"][:valid])
        first_step_absolute_from_delta = compute_metrics(
            np.stack([pair["predicted_action_chunk"][0] for pair in pairs[:metric_end]], axis=0),
            np.stack([pair["ground_truth_action_chunk"][0] for pair in pairs[:metric_end]], axis=0),
        )
        valid_pred_absolute = []
        valid_gt_absolute = []
        for pair in pairs[:metric_end]:
            valid = pair["valid_length"]
            valid_pred_absolute.append(pair["predicted_action_chunk"][:valid])
            valid_gt_absolute.append(pair["ground_truth_action_chunk"][:valid])

        return pairs, {
            "first_step_delta": first_step_delta,
            "valid_chunk_delta": compute_metrics(np.concatenate(valid_pred_delta), np.concatenate(valid_gt_delta)),
            "first_step_absolute_from_delta": first_step_absolute_from_delta,
            "valid_chunk_absolute_from_delta": compute_metrics(
                np.concatenate(valid_pred_absolute),
                np.concatenate(valid_gt_absolute),
            ),
        }

    first_step_absolute = compute_metrics(
        pred_chunks[:metric_end, 0],
        np.stack([pair["ground_truth_action_chunk"][0] for pair in pairs[:metric_end]], axis=0),
    )
    valid_pred_absolute = []
    valid_gt_absolute = []
    for pair in pairs[:metric_end]:
        valid = pair["valid_length"]
        valid_pred_absolute.append(pair["predicted_action_chunk"][:valid])
        valid_gt_absolute.append(pair["ground_truth_action_chunk"][:valid])

    return pairs, {
        "first_step_absolute": first_step_absolute,
        "valid_chunk_absolute": compute_metrics(np.concatenate(valid_pred_absolute), np.concatenate(valid_gt_absolute)),
    }


def save_episode_output(
    output_dir: Path,
    episode_name: str,
    trajectory_data: dict[str, Any],
    metadata: dict[str, Any],
    images: dict[str, np.ndarray | None],
    save_images: bool,
) -> None:
    episode_dir = output_dir / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)
    with (episode_dir / "trajectory_pairs.pkl").open("wb") as f:
        pickle.dump(trajectory_data, f)
    with (episode_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f)
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    if save_images:
        packed = {key: value for key, value in images.items() if value is not None}
        with (episode_dir / "images.pkl").open("wb") as f:
            pickle.dump(packed, f)


def evaluate_one_episode(
    args: argparse.Namespace,
    model: Any,
    hdf5_path: Path,
    camera_names: list[str],
    device: str,
    input_dtype: torch.dtype,
) -> dict[str, Any]:
    episode_data = load_episode(
        hdf5_path,
        camera_names,
        allow_raw_fallback=args.allow_raw_fallback,
        state_dim=args.resolved_state_dim,
        action_dim=args.resolved_action_dim,
        state_mode=args.resolved_state_mode,
        task_arg=args.task,
    )
    if episode_data["state_source"] not in TRAINING_LAYOUT_SOURCES:
        print(
            f"[WARN] {hdf5_path.name}: state_source={episode_data['state_source']}. "
            "Raw fallback is enabled; this is not the strict training-layout evaluation path."
        )
    if args.limit_frames is not None:
        limit = min(args.limit_frames, episode_data["episode_len"])
        episode_data["qpos"] = episode_data["qpos"][:limit]
        episode_data["model_qpos"] = episode_data["model_qpos"][:limit]
        episode_data["actions"] = episode_data["actions"][:limit]
        episode_data["episode_len"] = limit
        for camera, frames in list(episode_data["images"].items()):
            if frames is not None:
                episode_data["images"][camera] = frames[:limit]

    pred_chunks = run_batched_inference(
        model=model,
        episode_data=episode_data,
        task=episode_data["task"],
        camera_names=camera_names,
        device=device,
        batch_size=args.batch_size,
        center_crop=not args.no_center_crop,
        input_dtype=input_dtype,
    )
    pairs, metrics = build_trajectory_pairs(
        qpos=episode_data["qpos"],
        model_qpos=episode_data["model_qpos"],
        gt_actions=episode_data["actions"],
        pred_chunks=pred_chunks,
        action_mode=args.resolved_action_mode,
        metric_drop_last_frames=args.metric_drop_last_frames,
    )

    episode_name = hdf5_path.stem
    action_dim = int(pred_chunks.shape[-1])
    if args.resolved_action_mode == "delta":
        prediction_space = f"chunkwise_se3_delta_{action_dim}d"
        ground_truth_space = f"chunkwise_se3_delta_{action_dim}d"
        model_raw_output_space = f"normalized_then_unnormalized_chunkwise_se3_delta_{action_dim}d"
        absolute_reconstruction_space = f"state_composed_absolute_action_{action_dim}d"
    else:
        prediction_space = f"chunkwise_absolute_action_{action_dim}d"
        ground_truth_space = f"hdf5_action_chunk_{action_dim}d"
        model_raw_output_space = f"normalized_then_unnormalized_chunkwise_absolute_action_{action_dim}d"
        absolute_reconstruction_space = None
    trajectory_data = {
        "pairs": pairs,
        "chunk_size": int(pred_chunks.shape[1]),
        "state_dim": int(episode_data["qpos"].shape[-1]),
        "action_dim": action_dim,
        "episode_len": int(episode_data["episode_len"]),
        "source_hdf5_path": str(hdf5_path),
        "cameras": camera_names,
        "action_mode": args.resolved_action_mode,
        "prediction_space": prediction_space,
        "ground_truth_space": ground_truth_space,
        "model_raw_output_space": model_raw_output_space,
        "absolute_reconstruction_space": absolute_reconstruction_space,
    }
    metadata = {
        "episode_name": episode_name,
        "episode_len": int(episode_data["episode_len"]),
        "task": episode_data["task"],
        "cameras": camera_names,
        "chunk_size": int(pred_chunks.shape[1]),
        "state_dim": int(episode_data["qpos"].shape[-1]),
        "action_dim": action_dim,
        "action_mode": args.resolved_action_mode,
        "action_mode_source": args.action_mode_source,
        "state_mode": args.resolved_state_mode,
        "state_mode_source": args.state_mode_source,
        "metrics": metrics,
        "source_hdf5_path": str(hdf5_path),
        "state_source": episode_data["state_source"],
        "uses_training_ground_truth_layout": episode_data["state_source"] in TRAINING_LAYOUT_SOURCES,
        "transform_metadata": episode_data.get("transform_metadata", {}),
        "primary_metric_space": prediction_space,
        "absolute_reconstruction_space": absolute_reconstruction_space,
    }
    save_episode_output(
        Path(args.output_dir),
        episode_name,
        trajectory_data,
        metadata,
        episode_data["images"],
        args.save_images,
    )
    return metadata


def write_summary(output_dir: Path, args: argparse.Namespace, episode_metrics: list[dict[str, Any]]) -> None:
    fallback_sources = sorted(
        {
            m.get("state_source")
            for m in episode_metrics
            if not m.get("uses_training_ground_truth_layout", False)
        }
    )
    action_mode = args.resolved_action_mode
    action_dim = int(args.resolved_action_dim)
    if action_mode == "delta":
        first_key = "first_step_delta"
        chunk_key = "valid_chunk_delta"
        primary_metric_space = f"chunkwise_se3_delta_{action_dim}d"
        avg_prefix = "delta"
    elif action_mode == "absolute":
        first_key = "first_step_absolute"
        chunk_key = "valid_chunk_absolute"
        primary_metric_space = f"chunkwise_absolute_action_{action_dim}d"
        avg_prefix = "absolute"
    else:
        raise ValueError(f"Unsupported action_mode={action_mode!r}")

    summary = {
        "checkpoint": str(args.resolved_checkpoint),
        "dataset_stats": str(args.resolved_stats),
        "eval_data_dir": str(Path(args.eval_data_dir).expanduser().resolve()),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "cameras": list(getattr(args, "resolved_cameras", parse_cameras(args.cameras))),
        "camera_source": getattr(args, "camera_source", "cli"),
        "action_mode": action_mode,
        "action_mode_source": getattr(args, "action_mode_source", "auto"),
        "state_mode": getattr(args, "resolved_state_mode", "absolute"),
        "state_mode_source": getattr(args, "state_mode_source", "auto"),
        "state_dim": int(args.resolved_state_dim),
        "action_dim": action_dim,
        "episodes": len(episode_metrics),
        f"avg_first_step_{avg_prefix}_mse": float(np.mean([m["metrics"][first_key]["mse"] for m in episode_metrics])),
        f"avg_first_step_{avg_prefix}_mae": float(np.mean([m["metrics"][first_key]["mae"] for m in episode_metrics])),
        f"avg_valid_chunk_{avg_prefix}_mse": float(np.mean([m["metrics"][chunk_key]["mse"] for m in episode_metrics])),
        f"avg_valid_chunk_{avg_prefix}_mae": float(np.mean([m["metrics"][chunk_key]["mae"] for m in episode_metrics])),
        "fallback_state_sources": fallback_sources,
        "primary_metric_space": primary_metric_space,
        "episode_metrics": episode_metrics,
    }
    if action_mode == "delta" and "first_step_absolute_from_delta" in episode_metrics[0]["metrics"]:
        summary["avg_first_step_absolute_from_delta_mse"] = float(
            np.mean([m["metrics"]["first_step_absolute_from_delta"]["mse"] for m in episode_metrics])
        )
        summary["avg_first_step_absolute_from_delta_mae"] = float(
            np.mean([m["metrics"]["first_step_absolute_from_delta"]["mae"] for m in episode_metrics])
        )
        summary["avg_valid_chunk_absolute_from_delta_mse"] = float(
            np.mean([m["metrics"]["valid_chunk_absolute_from_delta"]["mse"] for m in episode_metrics])
        )
        summary["avg_valid_chunk_absolute_from_delta_mae"] = float(
            np.mean([m["metrics"]["valid_chunk_absolute_from_delta"]["mae"] for m in episode_metrics])
        )
    unique_tasks = sorted({str(meta.get("task", "")) for meta in episode_metrics if meta.get("task")})
    summary["task"] = unique_tasks[0] if len(unique_tasks) == 1 else "per-episode"
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    if action_mode == "delta":
        comparison_lines = [
            "Primary comparison: model predicted delta action vs GT chunk-wise SE3 delta action",
            "GT delta construction: inv(T_qpos[t]) @ T_action[t+k], using /observations/qpos and /action",
            f"Average first-step delta MSE: {summary['avg_first_step_delta_mse']:.8f}",
            f"Average first-step delta MAE: {summary['avg_first_step_delta_mae']:.8f}",
            f"Average valid-chunk delta MSE: {summary['avg_valid_chunk_delta_mse']:.8f}",
            f"Average valid-chunk delta MAE: {summary['avg_valid_chunk_delta_mae']:.8f}",
        ]
        if "avg_first_step_absolute_from_delta_mse" in summary:
            comparison_lines.extend(
                [
                    "",
                    "Secondary comparison: delta prediction composed with input state into absolute action",
                    "Absolute reconstruction: T_abs_pred[t+k] = T_qpos[t] @ T_delta_pred[t+k]",
                    f"Average first-step absolute-from-delta MSE: {summary['avg_first_step_absolute_from_delta_mse']:.8f}",
                    f"Average first-step absolute-from-delta MAE: {summary['avg_first_step_absolute_from_delta_mae']:.8f}",
                    f"Average valid-chunk absolute-from-delta MSE: {summary['avg_valid_chunk_absolute_from_delta_mse']:.8f}",
                    f"Average valid-chunk absolute-from-delta MAE: {summary['avg_valid_chunk_absolute_from_delta_mae']:.8f}",
                ]
            )
    else:
        comparison_lines = [
            "Primary comparison: model predicted absolute action vs GT HDF5 /action chunk",
            "GT absolute construction: /action[t:t+chunk_size], using /observations/qpos[t] only as model state input",
            f"Average first-step absolute MSE: {summary['avg_first_step_absolute_mse']:.8f}",
            f"Average first-step absolute MAE: {summary['avg_first_step_absolute_mae']:.8f}",
            f"Average valid-chunk absolute MSE: {summary['avg_valid_chunk_absolute_mse']:.8f}",
            f"Average valid-chunk absolute MAE: {summary['avg_valid_chunk_absolute_mae']:.8f}",
        ]

    lines = [
        "EASY-MIRRO OFFLINE INFERENCE SUMMARY",
        "=" * 60,
        f"Checkpoint: {summary['checkpoint']}",
        f"Dataset stats: {summary['dataset_stats']}",
        f"Eval data dir: {summary['eval_data_dir']}",
        f"Output dir: {summary['output_dir']}",
        f"Task: {summary['task']}",
        f"Cameras: {summary['cameras']} ({summary['camera_source']})",
        f"Action mode: {summary['action_mode']} ({summary['action_mode_source']})",
        f"State mode: {summary['state_mode']} ({summary['state_mode_source']})",
        f"State/action dim: {summary['state_dim']}/{summary['action_dim']}",
        f"Episodes: {summary['episodes']}",
        "",
        *comparison_lines,
    ]
    if fallback_sources:
        lines.extend(
            [
                "",
                "WARNING:",
                "  Some episodes did not contain canonical /observations/qpos + /action.",
                "  Raw fallback was explicitly enabled. Prefer preprocessing eval data into the training layout.",
                "  Converted state sources:",
                f"  {fallback_sources}",
            ]
        )
        lines.append("")
    lines.append("")
    lines.append("Per episode:")
    for meta in episode_metrics:
        metrics = meta["metrics"]
        if action_mode == "delta":
            lines.append(
                f"  {meta['episode_name']}: "
                f"first_delta_mse={metrics['first_step_delta']['mse']:.8f}, "
                f"chunk_delta_mse={metrics['valid_chunk_delta']['mse']:.8f}, "
                f"first_abs_from_delta_mse={metrics['first_step_absolute_from_delta']['mse']:.8f}, "
                f"chunk_abs_from_delta_mse={metrics['valid_chunk_absolute_from_delta']['mse']:.8f}, "
                f"state_source={meta.get('state_source')}"
            )
        else:
            lines.append(
                f"  {meta['episode_name']}: "
                f"first_absolute_mse={metrics['first_step_absolute']['mse']:.8f}, "
                f"chunk_absolute_mse={metrics['valid_chunk_absolute']['mse']:.8f}, "
                f"state_source={meta.get('state_source')}"
            )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def list_episodes(files: list[Path], allow_raw_fallback: bool) -> None:
    print(f"Found {len(files)} HDF5 episodes")
    for idx, path in enumerate(files):
        try:
            with h5py.File(path, "r") as root:
                _, actions, state_source, _ = read_qpos_actions(
                    root,
                    path,
                    allow_raw_fallback=allow_raw_fallback,
                )
                length = len(actions)
            print(f"  {idx:06d}: len={length:5d}  source={state_source:28s}  {path}")
        except Exception as exc:
            print(f"  {idx:06d}: ERROR {exc}  {path}")


def main(args: argparse.Namespace) -> None:
    args.easy_mirro_root = str(Path(args.easy_mirro_root).expanduser().resolve())
    if args.output_dir:
        args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    else:
        args.output_dir = str(default_output_dir(args.checkpoint, args.eval_data_dir).resolve())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_names, camera_source = resolve_camera_names(args.cameras, args.checkpoint)
    args.resolved_cameras = camera_names
    args.camera_source = camera_source
    all_files = find_hdf5_files(Path(args.eval_data_dir).expanduser().resolve())
    if not all_files:
        raise FileNotFoundError(f"No HDF5 files found under {args.eval_data_dir}")
    if args.list_episodes:
        list_episodes(all_files, allow_raw_fallback=args.allow_raw_fallback)
        return

    selected = parse_episode_selection(args.episodes)
    if selected is not None:
        all_files = [path for idx, path in enumerate(all_files) if idx in set(selected)]
    if args.files:
        requested = {name for name in args.files}
        all_files = [path for path in all_files if path.name in requested or path.stem in requested]
    if args.random_episodes is not None and args.random_episodes < len(all_files):
        random.seed(args.seed)
        all_files = sorted(random.sample(all_files, args.random_episodes))
    if args.max_episodes is not None:
        all_files = all_files[: args.max_episodes]
    if not all_files:
        raise ValueError("No episodes selected for evaluation.")

    device = resolve_device(args.device)
    model_dtype = args.dtype
    if model_dtype == "auto":
        model_dtype = "float32"
    input_dtype = torch.float32 if model_dtype == "float32" else torch.bfloat16

    model, stats, resolved_checkpoint, resolved_stats = build_model(
        easy_mirro_root=Path(args.easy_mirro_root),
        checkpoint=args.checkpoint,
        camera_names=camera_names,
        device=device,
        dtype=model_dtype,
    )
    args.resolved_checkpoint = resolved_checkpoint
    args.resolved_stats = resolved_stats
    saved_cfg = load_checkpoint_config(resolved_checkpoint)
    action_mode, action_mode_source = resolve_action_mode(args.action_mode, saved_cfg, stats)
    state_mode, state_mode_source = resolve_state_mode(saved_cfg, stats)
    args.resolved_action_mode = action_mode
    args.action_mode_source = action_mode_source
    args.resolved_state_mode = state_mode
    args.state_mode_source = state_mode_source
    args.resolved_state_dim = int(getattr(model.config, "state_dim", saved_cfg.get("state_dim", stats.get("state_dim", 20))))
    args.resolved_action_dim = int(getattr(model.config, "action_dim", saved_cfg.get("action_dim", stats.get("action_dim", 20))))
    if getattr(model.config, "action_mode", action_mode) != action_mode:
        raise ValueError(
            f"Model config action_mode={getattr(model.config, 'action_mode', None)!r} "
            f"does not match resolved action_mode={action_mode!r}"
        )
    model_state_mode = getattr(model.config, "state_mode", state_mode)
    if canonical_state_mode(model_state_mode) != canonical_state_mode(state_mode):
        raise ValueError(
            f"Model config state_mode={model_state_mode!r} "
            f"does not match resolved state_mode={state_mode!r}"
        )

    print("=" * 70)
    print("easy-mirro offline inference")
    print(f"Checkpoint: {resolved_checkpoint}")
    print(f"Dataset stats: {resolved_stats}")
    print(f"Eval data dir: {Path(args.eval_data_dir).expanduser().resolve()}")
    print(f"Output dir: {output_dir}")
    print(f"Device/dtype: {device}/{model_dtype}")
    print(f"Cameras: {camera_names} ({camera_source})")
    print(f"Action mode: {action_mode} ({action_mode_source})")
    print(f"State mode: {state_mode} ({state_mode_source})")
    print(f"State/action dim: {args.resolved_state_dim}/{args.resolved_action_dim}")
    print(f"Chunk size: {model.config.n_action_steps}")
    print(f"Episodes: {len(all_files)}")
    print("=" * 70)

    episode_metrics = []
    for hdf5_path in tqdm(all_files, desc="episodes"):
        try:
            metadata = evaluate_one_episode(
                args=args,
                model=model,
                hdf5_path=hdf5_path,
                camera_names=camera_names,
                device=device,
                input_dtype=input_dtype,
            )
            episode_metrics.append(metadata)
        except Exception as exc:
            print(f"[ERROR] Failed to process {hdf5_path}: {exc}")
            if args.strict:
                raise
            import traceback

            traceback.print_exc()

    if not episode_metrics:
        raise RuntimeError("No episodes were successfully evaluated.")
    write_summary(output_dir, args, episode_metrics)
    print(f"Saved results to: {output_dir}")
    print(f"Summary: {output_dir / 'summary.txt'}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline inference for easy-mirro Pi0 checkpoints")
    parser.add_argument("--checkpoint", "--ckpt", required=True, help="Checkpoint dir or output root with checkpoint-*")
    parser.add_argument("--eval-data-dir", required=True, help="Directory containing HDF5 evaluation episodes")
    parser.add_argument("--easy-mirro-root", default=str(DEFAULT_EASY_MIRRO_ROOT), help="Easy-MIRRO training repo root")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. If omitted, writes to ../offline_inference_output/"
            "<checkpoint_name>__<eval_data_dir_name>."
        ),
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Language instruction, or 'auto' to read HDF5 attrs['language_instruction'] with a task fallback.",
    )
    parser.add_argument(
        "--cameras",
        default=DEFAULT_CAMERAS,
        help=(
            "Comma-separated model camera names, or 'auto'/'checkpoint' to read checkpoint config image_features. "
            "Default: auto."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    parser.add_argument(
        "--action-mode",
        default="auto",
        choices=["auto", "checkpoint", "delta", "absolute"],
        help="Output action space to evaluate. Default auto reads checkpoint config/dataset_stats.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--episodes", default=None, help="Episode indices from sorted HDF5 list, e.g. 0,2,4 or 0-3")
    parser.add_argument("--files", nargs="+", default=None, help="Specific HDF5 basenames or stems to evaluate")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--random-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-frames", type=int, default=None, help="Debug mode: only evaluate first N frames per episode")
    parser.add_argument("--metric-drop-last-frames", type=int, default=0)
    parser.add_argument("--save-images", action="store_true", help="Save decoded frames to images.pkl")
    parser.add_argument("--no-center-crop", action="store_true", help="Disable 95%% center crop used by online inference")
    parser.add_argument("--list-episodes", action="store_true", help="List selected data directory episodes and exit")
    parser.add_argument(
        "--require-training-layout",
        action="store_true",
        help="Deprecated no-op: canonical /observations/qpos + /action is now required by default.",
    )
    parser.add_argument(
        "--allow-raw-fallback",
        action="store_true",
        help=(
            "Legacy/debug only: allow raw state/6d_rot or state/joint_position files to be converted inside "
            "the evaluator. Off by default because it is not the strict training-layout path."
        ),
    )
    parser.add_argument("--strict", action="store_true", help="Raise immediately on first episode failure")
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
