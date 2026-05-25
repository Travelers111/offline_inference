#!/usr/bin/env python
"""Offline inference for LeRobot pi0.5 UMI datasets.

This evaluates a pi0.5 checkpoint on a local LeRobot v3 dataset and writes
trajectory-pair files compatible with the older offline-inference workflow.

Default contract for the blackboard pi0.5 checkpoint:

  - input state: LeRobot `observation.state[t]`, a 10D absolute pose
  - label source: LeRobot `action[t:t+chunk_size]`, absolute future actions
  - model output: unnormalized chunk-wise SE(3) relative action
  - visualized output: relative action composed with input state into absolute action
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

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

from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402
from lerobot.utils.feature_utils import dataset_to_policy_features  # noqa: E402
from lerobot.utils.pose6d import state_action_to_relative_pose_action_np  # noqa: E402
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "lerobot_blackboard_testdata"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "ckpt" / "inference_pi05_45000" / "pretrained_model"
DEFAULT_TOKENIZER = PROJECT_ROOT / "ckpt" / "inference_pi05_45000" / "paligemma-3b-pt-224-tokenizer"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "offline_inference_output"
DEFAULT_DATASET_REPO_ID = "local/lerobot_blackboard_testdata"
POSE_BLOCK_SIZE = 10


def parse_offsets(value: str) -> list[int]:
    value = value.strip()
    if value.startswith("["):
        parsed = json.loads(value)
        return [int(v) for v in parsed]
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def sanitize_experiment_name(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in {"-", "_", "."}:
            cleaned.append(char)
        elif char.isspace():
            cleaned.append("_")
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._-") or "unnamed"


def default_output_dir(checkpoint: str | Path, dataset_root: str | Path) -> Path:
    checkpoint_name = sanitize_experiment_name(Path(checkpoint).expanduser().resolve().name)
    if checkpoint_name == "pretrained_model":
        checkpoint_name = sanitize_experiment_name(Path(checkpoint).expanduser().resolve().parent.name)
    data_name = sanitize_experiment_name(Path(dataset_root).expanduser().resolve().name)
    return DEFAULT_OUTPUT_ROOT / f"{checkpoint_name}__{data_name}"


def parse_episode_selection(value: str | None) -> list[int] | None:
    if value is None or value.strip().lower() in {"", "all"}:
        return None
    episodes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(part))
    return sorted(set(episodes))


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model_dir(path: str | Path) -> Path:
    """Accept a pretrained_model dir, a checkpoint dir, or a train output dir."""
    base = Path(path).expanduser().resolve()
    candidates = [
        base,
        base / "pretrained_model",
        base / "checkpoints" / "last" / "pretrained_model",
        base / "last" / "pretrained_model",
    ]
    for candidate in candidates:
        if (candidate / "config.json").exists() and (candidate / "model.safetensors").exists():
            return candidate.resolve()
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find config.json and model.safetensors under {base}.\nSearched:\n{searched}"
    )


def resolve_action_mode(cfg) -> str:
    return "delta" if bool(getattr(cfg, "use_relative_actions", False)) else "absolute"


def pose_block_offsets(action_dim: int) -> range:
    if action_dim <= 0 or action_dim % POSE_BLOCK_SIZE != 0:
        raise ValueError(f"Action dim must be a positive multiple of 10, got {action_dim}")
    return range(0, action_dim, POSE_BLOCK_SIZE)


def read_processor_step_names(model_dir: Path, filename: str) -> list[str]:
    path = model_dir / filename
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    names = []
    for step in data.get("steps", []):
        names.append(step.get("registry_name") or step.get("class", ""))
    return names


def checkpoint_has_usable_processors(model_dir: Path, use_relative_actions: bool) -> bool:
    if not (model_dir / "policy_preprocessor.json").exists():
        return False
    if not (model_dir / "policy_postprocessor.json").exists():
        return False
    if not use_relative_actions:
        return True
    pre_steps = read_processor_step_names(model_dir, "policy_preprocessor.json")
    post_steps = read_processor_step_names(model_dir, "policy_postprocessor.json")
    return (
        "relative_pose_actions_processor" in pre_steps
        and "absolute_pose_actions_processor" in post_steps
    )


def get_checkpoint_image_keys(model_dir: Path) -> list[str]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return []

    data = json.loads(config_path.read_text())
    input_features = data.get("input_features") or {}
    image_keys = []
    for key, spec in input_features.items():
        feature_type = spec.get("type") if isinstance(spec, dict) else None
        if feature_type == "VISUAL" or key.startswith(f"{OBS_IMAGES}."):
            image_keys.append(key)
    return sorted(image_keys)


def normalize_camera_key(name: str) -> str:
    name = name.strip()
    if name.startswith(f"{OBS_IMAGES}."):
        return name
    return f"{OBS_IMAGES}.{name}"


def get_image_keys(
    dataset: LeRobotDataset,
    requested: str | None,
    checkpoint_keys: list[str],
) -> tuple[list[str], str]:
    available = sorted(key for key in dataset.meta.features if key.startswith(f"{OBS_IMAGES}."))
    available_set = set(available)
    request = (requested or "checkpoint").strip().lower()

    if request in {"", "auto", "checkpoint"}:
        if checkpoint_keys:
            missing = [key for key in checkpoint_keys if key not in available_set]
            if missing:
                raise KeyError(
                    "Checkpoint expects camera(s) missing from the dataset: "
                    f"{missing}. Available dataset cameras: {available}"
                )
            return sorted(checkpoint_keys), "checkpoint"
        return available, "dataset"

    if request in {"all", "dataset"}:
        return available, "dataset"

    image_keys = []
    for name in requested.split(","):
        if not name.strip():
            continue
        key = normalize_camera_key(name)
        if key not in available_set:
            raise KeyError(f"Requested camera {key} not found. Available: {available}")
        if key not in image_keys:
            image_keys.append(key)
    if not image_keys:
        raise ValueError("No cameras selected.")
    return image_keys, "explicit"


def build_pi05_config(
    args: argparse.Namespace,
    model_dir: Path,
    dataset: LeRobotDataset,
    image_keys: list[str],
):
    cfg = PreTrainedConfig.from_pretrained(model_dir)
    if cfg.type != "pi05":
        raise ValueError(f"Expected a pi05 checkpoint, got policy type: {cfg.type}")

    device = resolve_device(args.device)
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if device.startswith("cuda") else "float32"

    cfg.pretrained_path = model_dir
    cfg.device = device
    cfg.dtype = dtype
    cfg.tokenizer_name = str(Path(args.tokenizer_path).expanduser().resolve())
    if args.use_relative_actions is not None:
        cfg.use_relative_actions = args.use_relative_actions
    if args.relative_action_mode is not None:
        cfg.relative_action_mode = args.relative_action_mode
    if args.pose_arm_offsets is not None:
        cfg.pose_arm_offsets = args.pose_arm_offsets
    if args.pose_arm_stride is not None:
        cfg.pose_arm_stride = args.pose_arm_stride
    cfg.normalization_mapping = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.STATE: NormalizationMode.MIXED_QUANTILES,
        FeatureType.ACTION: NormalizationMode.MIXED_QUANTILES,
    }
    cfg.gradient_checkpointing = False
    cfg.compile_model = False

    # Re-infer state/action from the target LeRobot dataset, but keep visual
    # inputs limited to the resolved camera set. This lets a single-camera
    # checkpoint run on a dataset that still contains the unused second camera.
    features = dataset_to_policy_features(dataset.meta.features)
    missing_image_features = [key for key in image_keys if key not in features]
    if missing_image_features:
        raise KeyError(f"Selected camera feature(s) missing from policy features: {missing_image_features}")

    input_features = {key: features[key] for key in image_keys}
    for key, ft in features.items():
        if ft.type is FeatureType.ACTION:
            continue
        if key.startswith(f"{OBS_IMAGES}."):
            continue
        input_features[key] = ft

    cfg.input_features = input_features
    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}

    action_names = dataset.meta.features.get(ACTION, {}).get("names")
    if action_names is not None and hasattr(cfg, "action_feature_names"):
        cfg.action_feature_names = list(action_names)

    return cfg


def build_processors(args: argparse.Namespace, cfg, model_dir: Path, dataset: LeRobotDataset):
    source = args.processor_source
    if source == "auto":
        use_checkpoint = checkpoint_has_usable_processors(model_dir, cfg.use_relative_actions)
        resolved_source = "checkpoint" if use_checkpoint else "dataset-config"
    elif source == "checkpoint":
        if not checkpoint_has_usable_processors(model_dir, cfg.use_relative_actions):
            raise FileNotFoundError(
                f"{model_dir} does not contain processor configs compatible with "
                f"use_relative_actions={cfg.use_relative_actions}."
            )
        resolved_source = "checkpoint"
    else:
        resolved_source = "dataset-config"

    if resolved_source == "checkpoint":
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(model_dir),
            preprocessor_overrides={
                "device_processor": {"device": cfg.device},
                "tokenizer_processor": {"tokenizer_name": cfg.tokenizer_name},
            },
            postprocessor_overrides={
                "device_processor": {"device": "cpu"},
            },
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=None,
            dataset_stats=dataset.meta.stats,
        )

    return preprocessor, postprocessor, resolved_source

def episode_rows(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int, int]:
    ep = dataset.meta.episodes[episode_index]
    start = int(ep["dataset_from_index"])
    end = int(ep["dataset_to_index"])
    length = int(ep["length"])
    return start, end, length


def tensor_list_to_numpy(values: list[Any]) -> np.ndarray:
    arrays = []
    for value in values:
        if isinstance(value, torch.Tensor):
            arrays.append(value.detach().cpu().numpy())
        else:
            arrays.append(np.asarray(value))
    return np.stack(arrays).astype(np.float32)


def load_episode_feature_array(dataset: LeRobotDataset, key: str, start: int, end: int) -> np.ndarray:
    rows = dataset.hf_dataset[start:end]
    return tensor_list_to_numpy(rows[key])


def load_episode_state_action_arrays(
    dataset: LeRobotDataset,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = load_episode_feature_array(dataset, OBS_STATE, start, end)
    actions = load_episode_feature_array(dataset, ACTION, start, end)
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError(f"Expected rank-2 state/action arrays, got {states.shape} and {actions.shape}")
    if states.shape[0] != actions.shape[0]:
        raise ValueError(f"State/action length mismatch: {states.shape[0]} vs {actions.shape[0]}")
    return states.astype(np.float32), actions.astype(np.float32)


def make_inference_batch(
    items: list[dict[str, Any]],
    image_keys: list[str],
    task_override: str | None,
) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key in image_keys:
        batch[key] = torch.stack([item[key] for item in items], dim=0)
    batch[OBS_STATE] = torch.stack([item[OBS_STATE] for item in items], dim=0)
    batch["task"] = [task_override or item["task"] for item in items]
    return batch


def image_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu()
    if image.dtype != torch.uint8:
        image = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return image.permute(1, 2, 0).numpy()


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> dict[str, Any]:
    dim = min(predictions.shape[-1], ground_truth.shape[-1])
    predictions = predictions[..., :dim]
    ground_truth = ground_truth[..., :dim]
    diff = predictions - ground_truth
    pred_diff = np.diff(predictions, axis=0) if len(predictions) > 1 else np.zeros_like(predictions)
    return {
        "mse": float(np.mean(diff**2)),
        "mae": float(np.mean(np.abs(diff))),
        "per_dim_mse": np.mean(diff**2, axis=0).astype(float).tolist(),
        "max_error": float(np.max(np.abs(diff))),
        "smoothness": float(np.mean(np.var(pred_diff, axis=0))),
    }


def build_ground_truth_chunks(
    states: np.ndarray,
    actions: np.ndarray,
    chunk_size: int,
    arm_offsets: list[int],
    arm_stride: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    episode_len = actions.shape[0]
    action_dim = actions.shape[-1]
    pose_block_offsets(action_dim)
    absolute_chunks: list[np.ndarray] = []
    delta_chunks: list[np.ndarray] = []
    valid_lengths: list[int] = []

    for t in range(episode_len):
        end = min(t + chunk_size, episode_len)
        valid = end - t
        absolute_chunk = np.zeros((chunk_size, action_dim), dtype=np.float32)
        delta_chunk = np.zeros((chunk_size, action_dim), dtype=np.float32)
        absolute_chunk[:valid] = actions[t:end]
        delta_chunk[:valid] = state_action_to_relative_pose_action_np(
            actions[t:end],
            states[t],
            arm_offsets=arm_offsets,
            arm_stride=arm_stride,
        )
        if valid < chunk_size:
            absolute_chunk[valid:] = absolute_chunk[valid - 1]
            delta_chunk[valid:] = delta_chunk[valid - 1]
        absolute_chunks.append(absolute_chunk)
        delta_chunks.append(delta_chunk)
        valid_lengths.append(valid)

    return absolute_chunks, delta_chunks, valid_lengths


def valid_chunk_arrays(
    pairs: list[dict[str, Any]],
    pred_key: str,
    gt_key: str,
    metric_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_values = []
    gt_values = []
    for pair in pairs[:metric_end]:
        valid = pair["valid_length"]
        pred_values.append(pair[pred_key][:valid])
        gt_values.append(pair[gt_key][:valid])
    return np.concatenate(pred_values), np.concatenate(gt_values)


def build_trajectory_pairs(
    states: np.ndarray,
    gt_actions: np.ndarray,
    pred_delta_chunks: np.ndarray,
    pred_absolute_chunks: np.ndarray,
    chunk_size: int,
    metric_drop_last_frames: int,
    action_mode: str,
    arm_offsets: list[int],
    arm_stride: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episode_len = gt_actions.shape[0]
    action_dim = gt_actions.shape[-1]
    gt_action_chunks, gt_delta_chunks, valid_lengths = build_ground_truth_chunks(
        states=states,
        actions=gt_actions,
        chunk_size=chunk_size,
        arm_offsets=arm_offsets,
        arm_stride=arm_stride,
    )
    pairs = []

    for t in range(episode_len):
        pair = {
            "timestep": t,
            "observation_state": states[t].astype(np.float32),
            "raw_observation_state": states[t].astype(np.float32),
            "ground_truth_action_chunk": gt_action_chunks[t],
            "ground_truth_delta_chunk": gt_delta_chunks[t],
            "ground_truth_chunk": gt_action_chunks[t],
            "predicted_action_chunk": pred_absolute_chunks[t].astype(np.float32),
            "predicted_chunk": pred_absolute_chunks[t].astype(np.float32),
            "valid_length": valid_lengths[t],
        }
        if action_mode == "delta":
            pair["predicted_delta_chunk"] = pred_delta_chunks[t].astype(np.float32)
        pairs.append(pair)

    metric_end = max(0, episode_len - metric_drop_last_frames) or episode_len

    metrics: dict[str, Any] = {}
    if action_mode == "delta":
        metrics["first_step_delta"] = compute_metrics(
            np.stack([pair["predicted_delta_chunk"][0] for pair in pairs[:metric_end]], axis=0),
            np.stack([pair["ground_truth_delta_chunk"][0] for pair in pairs[:metric_end]], axis=0),
        )
        pred_valid_delta, gt_valid_delta = valid_chunk_arrays(
            pairs, "predicted_delta_chunk", "ground_truth_delta_chunk", metric_end
        )
        metrics["valid_chunk_delta"] = compute_metrics(pred_valid_delta, gt_valid_delta)
        metrics["first_step_absolute_from_delta"] = compute_metrics(
            np.stack([pair["predicted_action_chunk"][0] for pair in pairs[:metric_end]], axis=0),
            np.stack([pair["ground_truth_action_chunk"][0] for pair in pairs[:metric_end]], axis=0),
        )
        pred_valid_abs, gt_valid_abs = valid_chunk_arrays(
            pairs, "predicted_action_chunk", "ground_truth_action_chunk", metric_end
        )
        metrics["valid_chunk_absolute_from_delta"] = compute_metrics(pred_valid_abs, gt_valid_abs)
        metrics["first_step_absolute"] = metrics["first_step_absolute_from_delta"]
        metrics["valid_chunk_absolute"] = metrics["valid_chunk_absolute_from_delta"]
        metrics["first_step"] = metrics["first_step_delta"]
        metrics["valid_chunk"] = metrics["valid_chunk_delta"]
        return pairs, metrics

    metrics["first_step_absolute"] = compute_metrics(
        np.stack([pair["predicted_action_chunk"][0] for pair in pairs[:metric_end]], axis=0),
        np.stack([pair["ground_truth_action_chunk"][0] for pair in pairs[:metric_end]], axis=0),
    )
    pred_valid_abs, gt_valid_abs = valid_chunk_arrays(
        pairs, "predicted_action_chunk", "ground_truth_action_chunk", metric_end
    )
    metrics["valid_chunk_absolute"] = compute_metrics(pred_valid_abs, gt_valid_abs)
    metrics["first_step"] = metrics["first_step_absolute"]
    metrics["valid_chunk"] = metrics["valid_chunk_absolute"]
    return pairs, metrics


def save_episode_output(
    output_dir: Path,
    episode_name: str,
    trajectory_data: dict[str, Any],
    metadata: dict[str, Any],
    images: dict[str, list[np.ndarray]] | None,
) -> None:
    episode_dir = output_dir / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)

    with (episode_dir / "trajectory_pairs.pkl").open("wb") as f:
        pickle.dump(trajectory_data, f)

    with (episode_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f)

    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    if images:
        packed_images = {key: np.stack(frames, axis=0) for key, frames in images.items()}
        with (episode_dir / "images.pkl").open("wb") as f:
            pickle.dump(packed_images, f)


def evaluate_episode(
    args: argparse.Namespace,
    dataset: LeRobotDataset,
    policy,
    preprocessor,
    relative_postprocessor,
    postprocessor,
    image_keys: list[str],
    episode_index: int,
) -> dict[str, Any]:
    start, end, episode_len = episode_rows(dataset, episode_index)
    if args.limit_frames is not None:
        end = min(end, start + args.limit_frames)
        episode_len = end - start

    states, gt_actions = load_episode_state_action_arrays(dataset, start, end)
    pred_delta_chunks: list[np.ndarray] = []
    pred_absolute_chunks: list[np.ndarray] = []
    saved_images: dict[str, list[np.ndarray]] | None = (
        {key: [] for key in image_keys} if args.save_images else None
    )
    tasks_seen: list[str] = []

    policy.reset()

    frame_ranges = range(start, end, args.batch_size)
    for batch_start in tqdm(frame_ranges, desc=f"episode_{episode_index:06d}", leave=False):
        batch_end = min(batch_start + args.batch_size, end)
        items = [dataset[i] for i in range(batch_start, batch_end)]
        tasks_seen.extend(item["task"] for item in items)

        if saved_images is not None:
            for item in items:
                for key in image_keys:
                    saved_images[key].append(image_to_uint8_hwc(item[key]))

        batch = make_inference_batch(items, image_keys, args.task)
        processed_batch = preprocessor(batch)
        with torch.inference_mode():
            policy_actions = policy.predict_action_chunk(
                processed_batch,
                num_steps=args.num_inference_steps,
            )
            relative_actions = relative_postprocessor(policy_actions)
            absolute_actions = postprocessor(policy_actions)
        pred_delta_chunks.append(relative_actions.detach().cpu().float().numpy())
        pred_absolute_chunks.append(absolute_actions.detach().cpu().float().numpy())

    pred_delta_array = np.concatenate(pred_delta_chunks, axis=0)
    pred_absolute_array = np.concatenate(pred_absolute_chunks, axis=0)
    chunk_size = int(pred_delta_array.shape[1])
    action_dim = int(pred_delta_array.shape[2])
    action_mode = args.resolved_action_mode
    pairs, metrics = build_trajectory_pairs(
        states=states,
        gt_actions=gt_actions,
        pred_delta_chunks=pred_delta_array,
        pred_absolute_chunks=pred_absolute_array,
        chunk_size=chunk_size,
        metric_drop_last_frames=args.resolved_metric_drop_last_frames,
        action_mode=action_mode,
        arm_offsets=args.pose_arm_offsets,
        arm_stride=args.pose_arm_stride,
    )

    episode_name = f"episode_{episode_index:06d}"
    unique_tasks = sorted(set(tasks_seen))
    if action_mode == "delta":
        prediction_space = f"chunkwise_se3_delta_{action_dim}d"
        ground_truth_space = f"chunkwise_se3_delta_{action_dim}d"
        model_raw_output_space = f"normalized_then_unnormalized_chunkwise_se3_delta_{action_dim}d"
        absolute_reconstruction_space = f"state_composed_absolute_action_{action_dim}d"
        primary_metric_space = prediction_space
    else:
        prediction_space = f"chunkwise_absolute_action_{action_dim}d"
        ground_truth_space = f"lerobot_action_chunk_{action_dim}d"
        model_raw_output_space = f"normalized_then_unnormalized_chunkwise_absolute_action_{action_dim}d"
        absolute_reconstruction_space = None
        primary_metric_space = prediction_space
    trajectory_data = {
        "pairs": pairs,
        "chunk_size": chunk_size,
        "state_dim": int(states.shape[-1]),
        "action_dim": action_dim,
        "episode_len": episode_len,
        "source_dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "episode_index": episode_index,
        "dataset_from_index": start,
        "dataset_to_index": end,
        "action_mode": action_mode,
        "state_mode": "absolute",
        "prediction_space": prediction_space,
        "ground_truth_space": ground_truth_space,
        "model_raw_output_space": model_raw_output_space,
        "absolute_reconstruction_space": absolute_reconstruction_space,
    }
    metadata = {
        "episode_name": episode_name,
        "episode_index": episode_index,
        "episode_len": episode_len,
        "checkpoint": str(args.resolved_model_dir),
        "task": args.task or (unique_tasks[0] if len(unique_tasks) == 1 else unique_tasks),
        "dataset_tasks": unique_tasks,
        "cameras": image_keys,
        "camera_source": getattr(args, "resolved_camera_source", "unknown"),
        "chunk_size": chunk_size,
        "state_dim": int(states.shape[-1]),
        "action_dim": action_dim,
        "action_mode": action_mode,
        "state_mode": "absolute",
        "action_names": dataset.meta.features.get(ACTION, {}).get("names"),
        "metrics": metrics,
        "primary_metric_space": primary_metric_space,
        "absolute_reconstruction_space": absolute_reconstruction_space,
    }

    save_episode_output(Path(args.output_dir), episode_name, trajectory_data, metadata, saved_images)
    return metadata


def write_summary(output_dir: Path, args: argparse.Namespace, episode_metrics: list[dict[str, Any]]) -> None:
    action_mode = getattr(args, "resolved_action_mode", "delta")
    if action_mode == "delta":
        first_key = "first_step_delta"
        chunk_key = "valid_chunk_delta"
        avg_prefix = "delta"
        primary_metric_space = f"chunkwise_se3_delta_{episode_metrics[0]['action_dim']}d"
    else:
        first_key = "first_step_absolute"
        chunk_key = "valid_chunk_absolute"
        avg_prefix = "absolute"
        primary_metric_space = f"chunkwise_absolute_action_{episode_metrics[0]['action_dim']}d"

    first_mse = [m["metrics"][first_key]["mse"] for m in episode_metrics]
    first_mae = [m["metrics"][first_key]["mae"] for m in episode_metrics]
    chunk_mse = [m["metrics"][chunk_key]["mse"] for m in episode_metrics]
    chunk_mae = [m["metrics"][chunk_key]["mae"] for m in episode_metrics]

    summary = {
        "checkpoint": str(args.resolved_model_dir),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "processor_source": args.resolved_processor_source,
        "cameras": getattr(args, "resolved_cameras", []),
        "camera_source": getattr(args, "resolved_camera_source", "unknown"),
        "action_mode": action_mode,
        "state_mode": "absolute",
        "primary_metric_space": primary_metric_space,
        "metric_drop_last_frames": args.resolved_metric_drop_last_frames,
        "episodes": len(episode_metrics),
        f"avg_first_step_{avg_prefix}_mse": float(np.mean(first_mse)),
        f"avg_first_step_{avg_prefix}_mae": float(np.mean(first_mae)),
        f"avg_valid_chunk_{avg_prefix}_mse": float(np.mean(chunk_mse)),
        f"avg_valid_chunk_{avg_prefix}_mae": float(np.mean(chunk_mae)),
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
        summary["avg_first_step_absolute_mse"] = summary["avg_first_step_absolute_from_delta_mse"]
        summary["avg_first_step_absolute_mae"] = summary["avg_first_step_absolute_from_delta_mae"]
        summary["avg_valid_chunk_absolute_mse"] = summary["avg_valid_chunk_absolute_from_delta_mse"]
        summary["avg_valid_chunk_absolute_mae"] = summary["avg_valid_chunk_absolute_from_delta_mae"]
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    if action_mode == "delta":
        comparison_lines = [
            "Primary comparison: model predicted relative SE(3) action vs GT chunk-wise relative SE(3) action",
            "GT relative construction: inv(T_observation.state[t]) @ T_action[t+k]",
            f"Average first-step delta MSE: {summary['avg_first_step_delta_mse']:.8f}",
            f"Average first-step delta MAE: {summary['avg_first_step_delta_mae']:.8f}",
            f"Average valid-chunk delta MSE: {summary['avg_valid_chunk_delta_mse']:.8f}",
            f"Average valid-chunk delta MAE: {summary['avg_valid_chunk_delta_mae']:.8f}",
        ]
        if "avg_first_step_absolute_from_delta_mse" in summary:
            comparison_lines.extend(
                [
                    "",
                    "Absolute-space comparison: relative prediction composed with input state into absolute action",
                    "Absolute reconstruction: T_abs_pred[t+k] = T_state[t] @ T_delta_pred[t+k]",
                    f"Average first-step absolute MSE: {summary['avg_first_step_absolute_mse']:.8f}",
                    f"Average first-step absolute MAE: {summary['avg_first_step_absolute_mae']:.8f}",
                    f"Average valid-chunk absolute MSE: {summary['avg_valid_chunk_absolute_mse']:.8f}",
                    f"Average valid-chunk absolute MAE: {summary['avg_valid_chunk_absolute_mae']:.8f}",
                    "",
                    "Backward-compatible absolute-from-delta aliases:",
                    f"Average first-step absolute-from-delta MSE: {summary['avg_first_step_absolute_from_delta_mse']:.8f}",
                    f"Average first-step absolute-from-delta MAE: {summary['avg_first_step_absolute_from_delta_mae']:.8f}",
                    f"Average valid-chunk absolute-from-delta MSE: {summary['avg_valid_chunk_absolute_from_delta_mse']:.8f}",
                    f"Average valid-chunk absolute-from-delta MAE: {summary['avg_valid_chunk_absolute_from_delta_mae']:.8f}",
                ]
            )
    else:
        comparison_lines = [
            "Primary comparison: model predicted absolute action vs GT LeRobot action chunk",
            "GT absolute construction: action[t:t+chunk_size]",
            f"Average first-step absolute MSE: {summary['avg_first_step_absolute_mse']:.8f}",
            f"Average first-step absolute MAE: {summary['avg_first_step_absolute_mae']:.8f}",
            f"Average valid-chunk absolute MSE: {summary['avg_valid_chunk_absolute_mse']:.8f}",
            f"Average valid-chunk absolute MAE: {summary['avg_valid_chunk_absolute_mae']:.8f}",
        ]

    lines = [
        "LEROBOT PI0.5 OFFLINE INFERENCE SUMMARY",
        "=" * 60,
        f"Checkpoint: {summary['checkpoint']}",
        f"Dataset root: {summary['dataset_root']}",
        f"Output dir: {summary['output_dir']}",
        f"Processor source: {summary['processor_source']}",
        f"Cameras: {summary['cameras']} ({summary['camera_source']})",
        f"Action mode: {summary['action_mode']}",
        f"State mode: {summary['state_mode']}",
        f"Metric drop last frames: {summary['metric_drop_last_frames']}",
        f"Episodes: {summary['episodes']}",
        "",
        *comparison_lines,
        "",
        "Per episode:",
    ]
    for m in episode_metrics:
        metrics = m["metrics"]
        if action_mode == "delta":
            lines.append(
                f"  {m['episode_name']}: "
                f"first_delta_mse={metrics['first_step_delta']['mse']:.8f}, "
                f"chunk_delta_mse={metrics['valid_chunk_delta']['mse']:.8f}, "
                f"first_absolute_mse={metrics['first_step_absolute']['mse']:.8f}, "
                f"chunk_absolute_mse={metrics['valid_chunk_absolute']['mse']:.8f}, "
                f"first_abs_from_delta_mse={metrics['first_step_absolute_from_delta']['mse']:.8f}, "
                f"chunk_abs_from_delta_mse={metrics['valid_chunk_absolute_from_delta']['mse']:.8f}"
            )
        else:
            lines.append(
                f"  {m['episode_name']}: "
                f"first_absolute_mse={metrics['first_step_absolute']['mse']:.8f}, "
                f"chunk_absolute_mse={metrics['valid_chunk_absolute']['mse']:.8f}"
            )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def list_episodes(dataset: LeRobotDataset) -> None:
    print(f"Dataset: {dataset.root}")
    print(f"Total episodes: {dataset.num_episodes}, total frames: {dataset.num_frames}, fps: {dataset.fps}")
    for ep_idx in range(dataset.meta.total_episodes):
        start, end, length = episode_rows(dataset, ep_idx)
        print(f"  episode_{ep_idx:06d}: rows=[{start}, {end}), length={length}")


def main(args: argparse.Namespace) -> None:
    model_dir = resolve_model_dir(args.checkpoint)
    args.resolved_model_dir = model_dir
    if args.output_dir:
        args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    else:
        args.output_dir = str(default_output_dir(args.checkpoint, args.dataset_root).resolve())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=Path(args.dataset_root).expanduser().resolve(),
        video_backend=args.video_backend,
        download_videos=False,
        return_uint8=False,
    )
    if args.list_episodes:
        list_episodes(dataset)
        return

    checkpoint_cameras = get_checkpoint_image_keys(model_dir)
    image_keys, camera_source = get_image_keys(dataset, args.cameras, checkpoint_cameras)
    args.resolved_cameras = image_keys
    args.resolved_camera_source = camera_source
    cfg = build_pi05_config(args, model_dir, dataset, image_keys)
    args.pose_arm_offsets = list(getattr(cfg, "pose_arm_offsets", [0]))
    args.pose_arm_stride = int(getattr(cfg, "pose_arm_stride", 10))
    args.resolved_action_mode = resolve_action_mode(cfg)
    default_drop = int(getattr(cfg, "drop_n_last_frames", 0) or 0)
    args.resolved_metric_drop_last_frames = (
        default_drop if args.metric_drop_last_frames is None else int(args.metric_drop_last_frames)
    )

    print(f"Checkpoint: {model_dir}")
    print(f"Dataset: {dataset.root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device/dtype: {cfg.device}/{cfg.dtype}")
    print(f"Cameras: {image_keys} ({camera_source})")
    print(f"State mode: absolute")
    print(f"Action mode: {args.resolved_action_mode}")
    print(f"Relative actions: {cfg.use_relative_actions} ({cfg.relative_action_mode})")
    print(f"Metric drop last frames: {args.resolved_metric_drop_last_frames}")

    policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, postprocessor, processor_source = build_processors(args, cfg, model_dir, dataset)
    relative_postprocessor = postprocessor[:1]
    args.resolved_processor_source = processor_source
    print(f"Processor source: {processor_source}")

    all_episode_indices = list(range(dataset.meta.total_episodes))
    selected = parse_episode_selection(args.episodes)
    if selected is not None:
        all_episode_indices = [idx for idx in all_episode_indices if idx in set(selected)]
    if args.random_episodes is not None and args.random_episodes < len(all_episode_indices):
        random.seed(args.seed)
        all_episode_indices = sorted(random.sample(all_episode_indices, args.random_episodes))
    if args.max_episodes is not None:
        all_episode_indices = all_episode_indices[: args.max_episodes]
    if not all_episode_indices:
        raise ValueError("No episodes selected for evaluation.")

    print(f"Evaluating episodes: {all_episode_indices}")
    episode_metrics = []
    for episode_index in tqdm(all_episode_indices, desc="Evaluating episodes"):
        metadata = evaluate_episode(
            args=args,
            dataset=dataset,
            policy=policy,
            preprocessor=preprocessor,
            relative_postprocessor=relative_postprocessor,
            postprocessor=postprocessor,
            image_keys=image_keys,
            episode_index=episode_index,
        )
        episode_metrics.append(metadata)

    write_summary(Path(args.output_dir), args, episode_metrics)
    print(f"Saved results to: {args.output_dir}")
    print(f"Summary: {Path(args.output_dir) / 'summary.txt'}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline inference for LeRobot pi0.5 UMI/blackboard datasets")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="pi0.5 checkpoint path")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Local LeRobot dataset root",
    )
    parser.add_argument("--dataset-repo-id", type=str, default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--tokenizer-path", type=str, default=str(DEFAULT_TOKENIZER))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. If omitted, writes to offline_inference_output/"
            "<checkpoint_name>__<dataset_root_name>."
        ),
    )
    parser.add_argument("--task", type=str, default=None, help="Override task prompt for every frame")
    parser.add_argument(
        "--cameras",
        type=str,
        default="checkpoint",
        help=(
            "Camera selection: 'checkpoint'/'auto' uses checkpoint visual inputs, "
            "'all'/'dataset' uses all dataset cameras, or pass comma-separated names."
        ),
    )
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float32"])
    parser.add_argument("--video-backend", type=str, default="pyav")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--episodes", type=str, default=None, help="Examples: '0,2,4', '0-3', or 'all'")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--random-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-frames", type=int, default=None, help="Debug mode: evaluate first N frames")
    parser.add_argument("--save-images", action="store_true", help="Also save decoded RGB frames to images.pkl")
    parser.add_argument(
        "--metric-drop-last-frames",
        type=int,
        default=None,
        help="Ignore the last N frames when aggregating metrics. Default reads checkpoint drop_n_last_frames.",
    )
    parser.add_argument(
        "--processor-source",
        type=str,
        default="auto",
        choices=["auto", "checkpoint", "dataset-config"],
        help="auto uses saved processors when compatible, otherwise rebuilds from dataset stats/config",
    )
    parser.add_argument(
        "--use-relative-actions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint use_relative_actions. Omit to use the checkpoint config.",
    )
    parser.add_argument("--relative-action-mode", type=str, default=None, choices=["se3_pose", "elementwise"])
    parser.add_argument("--pose-arm-offsets", type=parse_offsets, default=None)
    parser.add_argument("--pose-arm-stride", type=int, default=None)
    parser.add_argument("--list-episodes", action="store_true", help="Print dataset episode ranges and exit")
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
