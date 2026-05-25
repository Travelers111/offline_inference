#!/usr/bin/env python
"""Render static PNG previews for pi0.5 offline inference outputs."""

from __future__ import annotations

import argparse
import html
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lerobot_frame_source import EpisodeImageSource, camera_strip
from viz_run_info import load_episode_metadata, load_run_info, merge_episode_info, short_run_title


ACTION_LABELS = [
    "x",
    "y",
    "z",
    "rot0",
    "rot1",
    "rot2",
    "rot3",
    "rot4",
    "rot5",
    "gripper",
]
SPACE_KEYS = {
    "absolute": ("ground_truth_chunk", "predicted_chunk"),
    "delta": ("ground_truth_delta_chunk", "predicted_delta_chunk"),
}
POSE_BLOCK_SIZE = 10
DEFAULT_POSITION_SCALE = 10.0
DEFAULT_POSITION_UNIT = "dm"


def pose_block_offsets(action_dim: int) -> list[int]:
    if action_dim <= 0 or action_dim % POSE_BLOCK_SIZE != 0:
        raise ValueError(f"Action dim must be a positive multiple of 10, got {action_dim}")
    return list(range(0, action_dim, POSE_BLOCK_SIZE))


def available_spaces(data: dict) -> list[str]:
    first_pair = data["pairs"][0]
    spaces = []
    for space in ("delta", "absolute"):
        gt_key, pred_key = SPACE_KEYS[space]
        if gt_key in first_pair and pred_key in first_pair:
            spaces.append(space)
    if not spaces:
        raise KeyError("trajectory_pairs.pkl does not contain supported action-space keys")
    return spaces


def display_action(action: np.ndarray, action_dim: int, position_scale: float) -> np.ndarray:
    shown = np.asarray(action, dtype=np.float32).copy()
    for off in pose_block_offsets(action_dim):
        shown[..., off : off + 3] *= position_scale
    return shown


def load_trajectory(path: Path) -> dict:
    install_numpy_pickle_compat()
    with path.open("rb") as f:
        return pickle.load(f)


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


def choose_snapshot_indices(data: dict) -> list[int]:
    pairs = data["pairs"]
    episode_len = data["episode_len"]
    chunk_size = data["chunk_size"]
    last_full = max(0, min(len(pairs) - 1, episode_len - chunk_size))
    candidates = [0, len(pairs) // 2, last_full]
    return sorted(set(int(idx) for idx in candidates if 0 <= idx < len(pairs)))


def compute_axis_limits(data: dict, space: str, position_scale: float):
    gt_key, pred_key = SPACE_KEYS[space]
    action_dim = int(data["action_dim"])
    xyz = []
    for pair in data["pairs"]:
        valid = pair["valid_length"]
        gt = display_action(pair[gt_key], action_dim, position_scale)
        pred = display_action(pair[pred_key], action_dim, position_scale)
        for off in pose_block_offsets(action_dim):
            xyz.append(gt[:valid, off : off + 3])
            xyz.append(pred[:valid, off : off + 3])
    all_xyz = np.concatenate(xyz, axis=0)
    lo = np.nanmin(all_xyz, axis=0)
    hi = np.nanmax(all_xyz, axis=0)
    center = (lo + hi) / 2.0
    radius = max(float(np.nanmax(hi - lo)) / 2.0, 0.05 * position_scale)
    radius *= 1.15
    return tuple(zip(center - radius, center + radius, strict=True))


def render_trajectory(
    data: dict,
    image_source: EpisodeImageSource | None,
    episode_name: str,
    run_info: dict,
    output_path: Path,
    space: str,
    position_scale: float,
    position_unit: str,
) -> None:
    indices = choose_snapshot_indices(data)
    xlim, ylim, zlim = compute_axis_limits(data, space, position_scale)
    gt_key, pred_key = SPACE_KEYS[space]
    action_dim = int(data["action_dim"])
    blocks = pose_block_offsets(action_dim)
    has_images = image_source is not None and bool(image_source.available_cameras())

    rows = 2 if has_images else 1
    fig = plt.figure(figsize=(6 * len(indices), 8.8 if has_images else 6))
    for plot_idx, pair_idx in enumerate(indices, start=1):
        pair = data["pairs"][pair_idx]
        gt = display_action(pair[gt_key], action_dim, position_scale)
        pred = display_action(pair[pred_key], action_dim, position_scale)
        valid = pair["valid_length"]
        timestep = pair["timestep"]

        ax = fig.add_subplot(rows, len(indices), plot_idx, projection="3d")
        for block_idx, off in enumerate(blocks):
            label = "Arm" if len(blocks) == 1 else f"Block {block_idx}"
            ax.plot(
                gt[:valid, off],
                gt[:valid, off + 1],
                gt[:valid, off + 2],
                color="tab:blue",
                lw=2,
                label=f"GT {label}",
            )
            ax.scatter(gt[:valid, off], gt[:valid, off + 1], gt[:valid, off + 2], color="tab:blue", s=14, alpha=0.7)
            ax.plot(
                pred[:valid, off],
                pred[:valid, off + 1],
                pred[:valid, off + 2],
                color="tab:red",
                lw=2,
                ls="--",
                label=f"Pred {label}",
            )
            ax.scatter(pred[:valid, off], pred[:valid, off + 1], pred[:valid, off + 2], color="tab:red", s=14, alpha=0.7)
            ax.scatter(*gt[0, off : off + 3], color="tab:blue", marker="^", s=90, edgecolors="black")
            ax.scatter(*pred[0, off : off + 3], color="tab:red", marker="x", s=90)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel(f"X ({position_unit})")
        ax.set_ylabel(f"Y ({position_unit})")
        ax.set_zlabel(f"Z ({position_unit})")
        ax.set_title(f"t={timestep}, valid={valid}/{data['chunk_size']}")
        if plot_idx == 1:
            ax.legend(loc="upper left")

        if has_images:
            img_ax = fig.add_subplot(rows, len(indices), len(indices) + plot_idx)
            frame_items = image_source.get_frames(timestep) if image_source is not None else []
            strip = camera_strip([frame for _, frame in frame_items], target_height=190)
            if strip is not None:
                img_ax.imshow(strip)
                img_ax.set_title(
                    f"frame {timestep}: " + " | ".join(name for name, _ in frame_items),
                    fontsize=9,
                )
            else:
                img_ax.text(0.5, 0.5, "No frame", ha="center", va="center")
            img_ax.axis("off")

    fig.suptitle(f"{short_run_title(run_info)}\n{episode_name}: {space} action chunk previews", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def first_step_arrays(data: dict, space: str, position_scale: float) -> tuple[np.ndarray, np.ndarray]:
    gt_key, pred_key = SPACE_KEYS[space]
    action_dim = int(data["action_dim"])
    gt = np.stack([pair[gt_key][0] for pair in data["pairs"]], axis=0)
    pred = np.stack([pair[pred_key][0] for pair in data["pairs"]], axis=0)
    return display_action(gt, action_dim, position_scale), display_action(pred, action_dim, position_scale)


def render_waveform(
    data: dict,
    episode_name: str,
    run_info: dict,
    output_path: Path,
    space: str,
    position_scale: float,
) -> None:
    gt, pred = first_step_arrays(data, space, position_scale)
    t = np.arange(len(data["pairs"]))
    dims = min(gt.shape[1], len(ACTION_LABELS))

    fig, axes = plt.subplots(dims, 1, figsize=(16, 1.8 * dims), sharex=True)
    if dims == 1:
        axes = [axes]

    snapshot_indices = choose_snapshot_indices(data)
    for dim, ax in enumerate(axes):
        ax.plot(t, gt[:, dim], color="steelblue", lw=1.2, label="GT first-step")
        ax.plot(t, pred[:, dim], color="tomato", lw=1.2, ls="--", label="Pred first-step")
        for snap_idx in snapshot_indices:
            ax.axvline(snap_idx, color="orange", lw=0.8, alpha=0.6)
        ax.set_ylabel(ACTION_LABELS[dim], rotation=0, labelpad=24)
        ax.grid(True, alpha=0.2)
        if dim == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("timestep")
    fig.suptitle(f"{short_run_title(run_info)}\n{episode_name}: first-step {space} GT vs Pred waveforms", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def episode_dirs(data_dir: Path, episode: str | None) -> list[Path]:
    episodes = sorted(path for path in data_dir.iterdir() if (path / "trajectory_pairs.pkl").exists())
    if episode:
        episodes = [path for path in episodes if path.name == episode]
    if not episodes:
        raise FileNotFoundError(f"No episode outputs found under {data_dir}")
    return episodes


def write_index(
    output_dir: Path,
    rendered: list[tuple[str, dict[str, tuple[Path, Path]]]],
    run_info: dict,
) -> Path:
    rows = []
    for episode, spaces in rendered:
        parts = [f"<h2>{html.escape(episode)}</h2>"]
        for space, (traj, wave) in spaces.items():
            parts.append(
                "<h3>{space}</h3>"
                '<p><a href="{traj}">trajectory png</a> | <a href="{wave}">waveform png</a></p>'
                '<img src="{traj}" style="max-width:100%; border:1px solid #ddd;">'
                '<img src="{wave}" style="max-width:100%; border:1px solid #ddd; margin-top:12px;">'.format(
                    space=html.escape(space),
                    traj=html.escape(traj.name),
                    wave=html.escape(wave.name),
                )
            )
        rows.append("\n".join(parts))
    index = output_dir / "index.html"
    index.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>pi0.5 offline inference preview</title></head>"
        "<body style='font-family:sans-serif; margin:24px;'>"
        "<h1>pi0.5 offline inference preview</h1>"
        f"<p><strong>{html.escape(short_run_title(run_info))}</strong></p>"
        + "\n".join(rows)
        + "</body></html>\n"
    )
    return index


def main():
    parser = argparse.ArgumentParser(description="Render static pi0.5 offline inference previews")
    parser.add_argument("-d", "--data_dir", "--data-dir", dest="data_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", "--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("-i", "--episode", type=str, default=None)
    parser.add_argument("--space", choices=["auto", "delta", "absolute", "all"], default="all")
    parser.add_argument("--position-scale", type=float, default=DEFAULT_POSITION_SCALE)
    parser.add_argument("--position-unit", type=str, default=DEFAULT_POSITION_UNIT)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_run_info = load_run_info(data_dir)

    rendered = []
    for episode_dir in episode_dirs(data_dir, args.episode):
        data = load_trajectory(episode_dir / "trajectory_pairs.pkl")
        run_info = merge_episode_info(base_run_info, load_episode_metadata(episode_dir))
        image_source = EpisodeImageSource.from_episode_dir(episode_dir, data)
        if args.space == "all":
            spaces = available_spaces(data)
        elif args.space == "auto":
            spaces = [available_spaces(data)[0]]
        else:
            if args.space not in available_spaces(data):
                raise KeyError(f"{episode_dir.name} does not contain {args.space!r} action-space keys")
            spaces = [args.space]
        rendered_spaces = {}
        for space in spaces:
            traj_path = output_dir / f"{episode_dir.name}_{space}_trajectory.png"
            wave_path = output_dir / f"{episode_dir.name}_{space}_waveform.png"
            render_trajectory(
                data,
                image_source,
                episode_dir.name,
                run_info,
                traj_path,
                space=space,
                position_scale=args.position_scale,
                position_unit=args.position_unit,
            )
            render_waveform(
                data,
                episode_dir.name,
                run_info,
                wave_path,
                space=space,
                position_scale=args.position_scale,
            )
            rendered_spaces[space] = (traj_path, wave_path)
            print(f"Rendered {episode_dir.name} [{space}]:")
            print(f"  {traj_path}")
            print(f"  {wave_path}")
        rendered.append((episode_dir.name, rendered_spaces))

    index = write_index(output_dir, rendered, base_run_info)
    print(f"Index: {index}")


if __name__ == "__main__":
    main()
