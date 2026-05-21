#!/usr/bin/env python3
"""Render static previews for easy-mirro pose-block offline inference outputs."""

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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  Registers the 3D projection.

from image_frame_source import EpisodeImageSource, bgr_to_display_rgb, camera_strip


LOCAL_DIM_LABELS = ["x", "y", "z", "r0", "r1", "r2", "r3", "r4", "r5", "g"]
POSE_BLOCK_SIZE = 10
BLOCK_COLORS = [
    ("tab:blue", "tab:red"),
    ("tab:cyan", "tab:orange"),
    ("tab:green", "tab:purple"),
    ("tab:brown", "tab:pink"),
]
SPACE_KEYS = {
    "absolute": ("ground_truth_chunk", "predicted_chunk"),
    "delta": ("ground_truth_delta_chunk", "predicted_delta_chunk"),
}
DEFAULT_POSITION_SCALE = 10.0
DEFAULT_POSITION_UNIT = "dm"


def pose_block_offsets(action_dim: int) -> list[int]:
    if action_dim <= 0 or action_dim % POSE_BLOCK_SIZE != 0:
        raise ValueError(f"Action dim must be a positive multiple of 10, got {action_dim}")
    return list(range(0, action_dim, POSE_BLOCK_SIZE))


def dim_label(dim: int, action_dim: int, position_unit: str = DEFAULT_POSITION_UNIT) -> str:
    block = dim // POSE_BLOCK_SIZE
    local = dim % POSE_BLOCK_SIZE
    local_label = LOCAL_DIM_LABELS[local]
    if local < 3:
        local_label = f"{local_label} ({position_unit})"
    if action_dim == POSE_BLOCK_SIZE:
        return local_label
    prefix = "L" if block == 0 and action_dim == 20 else "R" if block == 1 and action_dim == 20 else f"B{block}"
    return f"{prefix}_{local_label}"


def display_action(action: np.ndarray, action_dim: int, position_scale: float) -> np.ndarray:
    shown = np.asarray(action, dtype=np.float32).copy()
    for off in pose_block_offsets(action_dim):
        shown[..., off : off + 3] *= position_scale
    return shown


def available_spaces(data: dict) -> list[str]:
    first_pair = data["pairs"][0]
    spaces = []
    for space in ("delta", "absolute"):
        gt_key, pred_key = SPACE_KEYS[space]
        if gt_key in first_pair and pred_key in first_pair:
            spaces.append(space)
    if not spaces:
        raise KeyError("No supported GT/pred trajectory keys found in trajectory_pairs.pkl")
    return spaces


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


def load_pickle(path: Path):
    install_numpy_pickle_compat()
    with path.open("rb") as f:
        return pickle.load(f)


def episode_dirs(data_dir: Path, episode: str | None) -> list[Path]:
    episodes = sorted(path for path in data_dir.iterdir() if (path / "trajectory_pairs.pkl").exists())
    if episode:
        episodes = [path for path in episodes if path.name == episode]
    if not episodes:
        raise FileNotFoundError(f"No episode outputs found under {data_dir}")
    return episodes


def snapshot_indices(data: dict) -> list[int]:
    pairs = data["pairs"]
    episode_len = data["episode_len"]
    chunk_size = data["chunk_size"]
    last_full = max(0, min(len(pairs) - 1, episode_len - chunk_size))
    return sorted(set(idx for idx in [0, len(pairs) // 2, last_full] if 0 <= idx < len(pairs)))


def axis_limits(data: dict, space: str, position_scale: float):
    gt_key, pred_key = SPACE_KEYS[space]
    blocks = pose_block_offsets(int(data["action_dim"]))
    xyz = []
    for pair in data["pairs"]:
        gt = display_action(pair[gt_key], int(data["action_dim"]), position_scale)
        pred = display_action(pair[pred_key], int(data["action_dim"]), position_scale)
        valid = pair["valid_length"]
        for off in blocks:
            xyz_slice = slice(off, off + 3)
            xyz.extend([gt[:valid, xyz_slice], pred[:valid, xyz_slice]])
    all_xyz = np.concatenate(xyz, axis=0)
    lo = np.nanmin(all_xyz, axis=0)
    hi = np.nanmax(all_xyz, axis=0)
    center = (lo + hi) / 2.0
    radius = max(float(np.nanmax(hi - lo)) / 2.0, 0.05 * position_scale)
    radius *= 1.12
    return tuple(zip(center - radius, center + radius))


def render_trajectory(
    data: dict,
    image_source: EpisodeImageSource | None,
    episode_name: str,
    output_path: Path,
    space: str,
    position_scale: float,
    position_unit: str,
) -> None:
    indices = snapshot_indices(data)
    gt_key, pred_key = SPACE_KEYS[space]
    action_dim = int(data["action_dim"])
    xlim, ylim, zlim = axis_limits(data, space, position_scale)
    blocks = pose_block_offsets(action_dim)
    image_cameras = image_source.available_cameras() if image_source is not None else []
    has_images = bool(image_cameras)

    rows = 2 if has_images else 1
    fig = plt.figure(figsize=(6 * len(indices), 6.4 + (3.2 if has_images else 0)))
    for col, pair_idx in enumerate(indices, start=1):
        pair = data["pairs"][pair_idx]
        gt = display_action(pair[gt_key], action_dim, position_scale)
        pred = display_action(pair[pred_key], action_dim, position_scale)
        valid = pair["valid_length"]
        timestep = pair["timestep"]

        ax = fig.add_subplot(rows, len(indices), col, projection="3d")
        for block_idx, off in enumerate(blocks):
            gt_color, pred_color = BLOCK_COLORS[block_idx % len(BLOCK_COLORS)]
            xyz_slice = slice(off, off + 3)
            label = "Arm" if len(blocks) == 1 else f"Block{block_idx}"
            ax.plot(gt[:valid, off], gt[:valid, off + 1], gt[:valid, off + 2], color=gt_color, lw=2, label=f"GT {label}")
            ax.plot(pred[:valid, off], pred[:valid, off + 1], pred[:valid, off + 2], color=pred_color, lw=2, ls="--", label=f"Pred {label}")
            ax.scatter(*gt[0, xyz_slice], color=gt_color, marker="^", s=70, edgecolors="black")
            ax.scatter(*pred[0, xyz_slice], color=pred_color, marker="x", s=70)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_xlabel(f"X ({position_unit})")
        ax.set_ylabel(f"Y ({position_unit})")
        ax.set_zlabel(f"Z ({position_unit})")
        ax.set_title(f"t={timestep}, valid={valid}/{data['chunk_size']}")
        if col == 1:
            ax.legend(loc="upper left", fontsize=8)

        if has_images:
            img_ax = fig.add_subplot(rows, len(indices), len(indices) + col)
            frame_items = image_source.get_frames(timestep) if image_source is not None else []
            strip = camera_strip([frame for _, frame in frame_items], target_height=190)
            if strip is not None:
                img_ax.imshow(bgr_to_display_rgb(strip))
                img_ax.set_title(
                    f"HDF5 frame t={timestep}: " + " | ".join(name for name, _ in frame_items),
                    fontsize=9,
                )
            else:
                img_ax.text(0.5, 0.5, "No HDF5 frame", ha="center", va="center")
            img_ax.axis("off")

    subtitle = f"{episode_name}: {data['action_dim']}D {space} action chunks, xyz in {position_unit}"
    if has_images:
        subtitle += " | bottom row shows model input camera frames at each timestep"
    fig.suptitle(subtitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def first_step_arrays(data: dict, space: str, position_scale: float) -> tuple[np.ndarray, np.ndarray]:
    gt_key, pred_key = SPACE_KEYS[space]
    gt = np.stack([pair[gt_key][0] for pair in data["pairs"]], axis=0)
    pred = np.stack([pair[pred_key][0] for pair in data["pairs"]], axis=0)
    action_dim = int(data["action_dim"])
    gt = display_action(gt, action_dim, position_scale)
    pred = display_action(pred, action_dim, position_scale)
    return gt, pred


def render_waveform(
    data: dict,
    episode_name: str,
    output_path: Path,
    space: str,
    position_scale: float,
    position_unit: str,
    image_source: EpisodeImageSource | None = None,
) -> None:
    gt, pred = first_step_arrays(data, space, position_scale)
    t = np.arange(len(data["pairs"]))

    action_dim = int(data["action_dim"])
    image_items = []
    if image_source is not None and image_source.available_cameras():
        for idx in snapshot_indices(data):
            timestep = data["pairs"][idx]["timestep"]
            frame_items = image_source.get_frames(timestep)
            strip = camera_strip([frame for _, frame in frame_items], target_height=140)
            if strip is not None:
                image_items.append((timestep, frame_items, strip))

    if image_items:
        fig = plt.figure(figsize=(18, max(13, action_dim * 1.3 + 3.0)))
        gs = fig.add_gridspec(action_dim + 1, 1, height_ratios=[2.4] + [1] * action_dim)
        image_ax = fig.add_subplot(gs[0, 0])
        spacer = np.full((image_items[0][2].shape[0], 12, 3), 245, dtype=np.uint8)
        pieces = []
        for idx, (_, _, strip) in enumerate(image_items):
            if idx:
                pieces.append(spacer)
            pieces.append(strip)
        image_ax.imshow(bgr_to_display_rgb(np.concatenate(pieces, axis=1)))
        camera_label = " | ".join(name for name, _ in image_items[0][1])
        time_label = "  /  ".join(f"t={timestep}" for timestep, _, _ in image_items)
        image_ax.set_title(f"HDF5 model input frames: {camera_label}   ({time_label})", fontsize=10)
        image_ax.axis("off")
        axes = []
        shared_ax = None
        for row in range(action_dim):
            ax = fig.add_subplot(gs[row + 1, 0], sharex=shared_ax)
            if shared_ax is None:
                shared_ax = ax
            axes.append(ax)
    else:
        fig, axes = plt.subplots(action_dim, 1, figsize=(18, max(10, action_dim * 1.3)), sharex=True)
    axes = np.atleast_1d(axes)
    for dim, ax in enumerate(axes):
        ax.plot(t, gt[:, dim], color="steelblue", lw=1.0, label="GT")
        ax.plot(t, pred[:, dim], color="tomato", lw=1.0, ls="--", label="Pred")
        ax.set_ylabel(dim_label(dim, action_dim, position_unit), rotation=0, labelpad=32, fontsize=8)
        ax.grid(True, alpha=0.2)
        if dim == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("timestep")
    fig.suptitle(f"{episode_name}: first-step {action_dim}D {space} action, xyz in {position_unit}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def write_index(output_dir: Path, rendered: list[tuple[str, dict[str, tuple[Path, Path]]]]) -> Path:
    rows = []
    for episode, space_files in rendered:
        block = [f"<h2>{html.escape(episode)}</h2>"]
        for space in space_files:
            traj, wave = space_files[space]
            block.append(
                "<h3>{space}</h3>"
                '<p><a href="{traj}">trajectory</a> | <a href="{wave}">waveform</a></p>'
                '<img src="{traj}" style="max-width:100%; border:1px solid #ddd;">'
                '<img src="{wave}" style="max-width:100%; border:1px solid #ddd; margin-top:12px;">'.format(
                    space=html.escape(space),
                    traj=html.escape(traj.name),
                    wave=html.escape(wave.name),
                )
            )
        rows.append("\n".join(block))
    index = output_dir / "index.html"
    index.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>easy-mirro offline inference preview</title>"
        "<style>"
        "body{font-family:Inter,Arial,sans-serif;margin:0;background:#f6f7f9;color:#17202a;}"
        "main{max-width:1500px;margin:0 auto;padding:28px;}"
        ".episode{background:#fff;border:1px solid #d9dee7;border-radius:8px;padding:18px;margin:18px 0;"
        "box-shadow:0 1px 4px rgba(20,28,40,.06);}"
        "h1{margin:0 0 6px 0;font-size:28px;}h2{margin:0 0 14px 0;font-size:20px;}h3{margin:16px 0 8px 0;}"
        "p{line-height:1.45;color:#46515f;}.links a{margin-right:14px;color:#1f5fbf;text-decoration:none;}"
        "img{display:block;max-width:100%;border:1px solid #d8dde6;border-radius:6px;background:#fff;margin-top:10px;}"
        "</style></head>"
        "<body><main>"
        "<h1>easy-mirro offline inference preview</h1>"
        "<p>Plots are rendered in whichever action spaces are present in trajectory_pairs.pkl. "
        "Trajectory images include the HDF5 camera frames actually used by the model at the displayed timestep.</p>"
        + "\n".join(f"<section class='episode'>{row}</section>" for row in rows)
        + "</main></body></html>\n"
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Render static easy-mirro offline inference previews")
    parser.add_argument("-d", "--data-dir", "--data_dir", dest="data_dir", required=True)
    parser.add_argument("-o", "--output-dir", "--output_dir", dest="output_dir", default=None)
    parser.add_argument("-i", "--episode", default=None)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--position-scale", type=float, default=DEFAULT_POSITION_SCALE)
    parser.add_argument("--position-unit", default=DEFAULT_POSITION_UNIT)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "preview"
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    for episode_dir in episode_dirs(data_dir, args.episode):
        data = load_pickle(episode_dir / "trajectory_pairs.pkl")
        image_source = None if args.no_images else EpisodeImageSource(episode_dir, data)
        space_files = {}
        for space in available_spaces(data):
            traj_path = output_dir / f"{episode_dir.name}_{space}_trajectory.png"
            wave_path = output_dir / f"{episode_dir.name}_{space}_waveform.png"
            render_trajectory(
                data,
                image_source,
                episode_dir.name,
                traj_path,
                space,
                args.position_scale,
                args.position_unit,
            )
            render_waveform(
                data,
                episode_dir.name,
                wave_path,
                space,
                args.position_scale,
                args.position_unit,
                image_source,
            )
            space_files[space] = (traj_path, wave_path)
        print(f"Rendered {episode_dir.name}:")
        for space, (traj_path, wave_path) in space_files.items():
            print(f"  {space}: {traj_path}")
            print(f"  {space}: {wave_path}")
        rendered.append((episode_dir.name, space_files))
        if image_source is not None:
            image_source.close()

    index = write_index(output_dir, rendered)
    print(f"Index: {index}")


if __name__ == "__main__":
    main()
