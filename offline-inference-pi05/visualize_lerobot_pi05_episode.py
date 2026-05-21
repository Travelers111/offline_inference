#!/usr/bin/env python
"""Visualize single-arm pi0.5 offline inference trajectory pairs."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib

if "--no-show" in sys.argv or not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

from lerobot_frame_source import EpisodeImageSource


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


def available_spaces(trajectory_data: dict) -> list[str]:
    pair = trajectory_data["pairs"][0]
    spaces = []
    for space in ("delta", "absolute"):
        gt_key, pred_key = SPACE_KEYS[space]
        if gt_key in pair and pred_key in pair:
            spaces.append(space)
    return spaces


def resolve_space(trajectory_data: dict, requested: str) -> str:
    if requested == "auto":
        spaces = available_spaces(trajectory_data)
        prediction_space = str(trajectory_data.get("prediction_space", ""))
        if "delta" in prediction_space and "delta" in spaces:
            return "delta"
        if "absolute" in prediction_space and "absolute" in spaces:
            return "absolute"
        if spaces:
            return spaces[0]
        raise KeyError("Trajectory output does not contain supported action-space keys.")
    gt_key, pred_key = SPACE_KEYS[requested]
    pair = trajectory_data["pairs"][0]
    if gt_key in pair and pred_key in pair:
        return requested
    fallback = "absolute" if requested == "delta" else "delta"
    fallback_gt, fallback_pred = SPACE_KEYS[fallback]
    if fallback_gt in pair and fallback_pred in pair:
        print(f"[WARN] {requested} space is not present; falling back to {fallback}.")
        return fallback
    raise KeyError(f"Trajectory output does not contain {requested!r} space keys.")


def display_action(action: np.ndarray, action_dim: int, position_scale: float) -> np.ndarray:
    shown = np.asarray(action, dtype=np.float32).copy()
    for off in pose_block_offsets(action_dim):
        shown[..., off : off + 3] *= position_scale
    return shown


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


class PI05TrajectoryVisualizer:
    def __init__(
        self,
        trajectory_data: dict,
        image_source: EpisodeImageSource | None,
        data_dir: Path,
        episodes: list[str],
        episode_index: int,
        space: str,
        position_scale: float,
        position_unit: str,
    ):
        self.trajectory_data = trajectory_data
        self.image_source = image_source
        self.data_dir = data_dir
        self.episodes = episodes
        self.episode_index = episode_index
        self.pairs = trajectory_data["pairs"]
        self.action_dim = int(trajectory_data["action_dim"])
        self.space = resolve_space(trajectory_data, space)
        self.gt_key, self.pred_key = SPACE_KEYS[self.space]
        self.pose_blocks = pose_block_offsets(self.action_dim)
        self.position_scale = position_scale
        self.position_unit = position_unit
        self.current = 0
        self.fig = None
        self.ax_traj = None
        self.image_axes = []
        self.slider = None
        self._slider_updating = False
        self._axis_limits = None

    def axis_limits(self):
        if self._axis_limits is not None:
            return self._axis_limits
        xyz = []
        for pair in self.pairs:
            valid = pair["valid_length"]
            gt = display_action(pair[self.gt_key], self.action_dim, self.position_scale)
            pred = display_action(pair[self.pred_key], self.action_dim, self.position_scale)
            for off in self.pose_blocks:
                xyz.append(gt[:valid, off : off + 3])
                xyz.append(pred[:valid, off : off + 3])
        all_xyz = np.concatenate(xyz, axis=0)
        lo = np.nanmin(all_xyz, axis=0)
        hi = np.nanmax(all_xyz, axis=0)
        center = (lo + hi) / 2.0
        radius = max(float(np.nanmax(hi - lo)) / 2.0, 0.05 * self.position_scale)
        radius *= 1.15
        self._axis_limits = tuple(zip(center - radius, center + radius, strict=True))
        return self._axis_limits

    def load_episode(self, episode_name: str) -> bool:
        install_numpy_pickle_compat()
        episode_dir = self.data_dir / episode_name
        traj_path = episode_dir / "trajectory_pairs.pkl"
        if not traj_path.exists():
            print(f"Missing trajectory file: {traj_path}")
            return False
        with traj_path.open("rb") as f:
            self.trajectory_data = pickle.load(f)
        self.image_source = EpisodeImageSource.from_episode_dir(episode_dir, self.trajectory_data)
        self.pairs = self.trajectory_data["pairs"]
        self.action_dim = int(self.trajectory_data["action_dim"])
        self.space = resolve_space(self.trajectory_data, self.space)
        self.gt_key, self.pred_key = SPACE_KEYS[self.space]
        self.pose_blocks = pose_block_offsets(self.action_dim)
        self.current = 0
        self._axis_limits = None
        return True

    def next_pair(self):
        if self.current < len(self.pairs) - 1:
            self.current += 1
            self.update()

    def prev_pair(self):
        if self.current > 0:
            self.current -= 1
            self.update()

    def next_episode(self):
        if self.episode_index < len(self.episodes) - 1:
            self.episode_index += 1
            if self.load_episode(self.episodes[self.episode_index]):
                self.update()

    def prev_episode(self):
        if self.episode_index > 0:
            self.episode_index -= 1
            if self.load_episode(self.episodes[self.episode_index]):
                self.update()

    def on_key(self, event):
        if event.key in {" ", "right"}:
            self.next_pair()
        elif event.key == "left":
            self.prev_pair()
        elif event.key == "up":
            self.next_episode()
        elif event.key == "down":
            self.prev_episode()
        elif event.key == "q":
            plt.close(self.fig)

    def on_slider(self, value):
        if self._slider_updating:
            return
        self.current = int(value)
        self.update()

    def setup(self):
        image_count = len(self.image_source.available_cameras()) if self.image_source is not None else 0
        rows = max(1, image_count)
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.subplots_adjust(bottom=0.08)
        gs = GridSpec(rows, 2, width_ratios=[2, 1])
        self.ax_traj = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.image_axes = [self.fig.add_subplot(gs[i, 1]) for i in range(rows)]
        slider_ax = self.fig.add_axes([0.15, 0.015, 0.7, 0.025])
        self.slider = Slider(slider_ax, "Step", 0, max(len(self.pairs) - 1, 1), valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update()

    def update(self):
        pair = self.pairs[self.current]
        timestep = pair["timestep"]
        gt = display_action(pair[self.gt_key], self.action_dim, self.position_scale)
        pred = display_action(pair[self.pred_key], self.action_dim, self.position_scale)
        valid = pair["valid_length"]
        chunk_size = self.trajectory_data["chunk_size"]

        self.ax_traj.clear()
        (xlim, ylim, zlim) = self.axis_limits()
        self.ax_traj.set_xlim(*xlim)
        self.ax_traj.set_ylim(*ylim)
        self.ax_traj.set_zlim(*zlim)
        for block_idx, off in enumerate(self.pose_blocks):
            label = "Arm" if len(self.pose_blocks) == 1 else f"Block {block_idx}"
            self.ax_traj.plot(
                gt[:valid, off],
                gt[:valid, off + 1],
                gt[:valid, off + 2],
                color="tab:blue",
                label=f"GT {label}",
                linewidth=2,
            )
            self.ax_traj.scatter(gt[:valid, off], gt[:valid, off + 1], gt[:valid, off + 2], color="tab:blue", s=18)
            self.ax_traj.plot(
                pred[:valid, off],
                pred[:valid, off + 1],
                pred[:valid, off + 2],
                color="tab:red",
                linestyle="--",
                label=f"Pred {label}",
                linewidth=2,
            )
            self.ax_traj.scatter(pred[:valid, off], pred[:valid, off + 1], pred[:valid, off + 2], color="tab:red", s=18)
            self.ax_traj.scatter(*gt[0, off : off + 3], color="tab:blue", marker="^", s=90, edgecolors="black")
            self.ax_traj.scatter(*pred[0, off : off + 3], color="tab:red", marker="x", s=90)
            gripper_idx = off + POSE_BLOCK_SIZE - 1
            if gt.shape[1] > gripper_idx and pred.shape[1] > gripper_idx:
                print(
                    f"[{self.space} step {timestep:>3}] block={block_idx} "
                    f"Gripper: GT={gt[0, gripper_idx]:.4f} Pred={pred[0, gripper_idx]:.4f}"
                )
        self.ax_traj.set_xlabel(f"X ({self.position_unit})")
        self.ax_traj.set_ylabel(f"Y ({self.position_unit})")
        self.ax_traj.set_zlabel(f"Z ({self.position_unit})")
        episode_name = self.episodes[self.episode_index]
        self.ax_traj.set_title(
            f"{episode_name}  {self.space}  step {timestep}  pair {self.current + 1}/{len(self.pairs)}  "
            f"valid {valid}/{chunk_size}\n"
            "Left/Right: step  Up/Down: episode  Q: quit"
        )
        self.ax_traj.legend(loc="upper left")

        if self.image_source is not None:
            frame_items = self.image_source.get_frames(timestep)
            for ax, (camera, frame) in zip(self.image_axes, frame_items, strict=False):
                ax.clear()
                ax.imshow(frame)
                ax.set_title(f"{camera} | frame {timestep}")
                ax.axis("off")
            for ax in self.image_axes[len(frame_items) :]:
                ax.clear()
                ax.axis("off")
        else:
            for ax in self.image_axes:
                ax.clear()
                ax.axis("off")

        self._slider_updating = True
        self.slider.valmax = max(len(self.pairs) - 1, 1)
        self.slider.ax.set_xlim(0, self.slider.valmax)
        self.slider.set_val(self.current)
        self._slider_updating = False
        self.fig.canvas.draw_idle()

    def run(self):
        self.setup()
        plt.show()


def load_start(data_dir: Path, requested_episode: str | None):
    install_numpy_pickle_compat()
    episodes = sorted(
        path.name for path in data_dir.iterdir() if path.is_dir() and (path / "trajectory_pairs.pkl").exists()
    )
    if not episodes:
        raise FileNotFoundError(f"No episode directories with trajectory_pairs.pkl under {data_dir}")
    if requested_episode and requested_episode in episodes:
        episode_index = episodes.index(requested_episode)
    else:
        episode_index = 0
    episode_dir = data_dir / episodes[episode_index]
    with (episode_dir / "trajectory_pairs.pkl").open("rb") as f:
        trajectory_data = pickle.load(f)
    image_source = EpisodeImageSource.from_episode_dir(episode_dir, trajectory_data)
    return trajectory_data, image_source, episodes, episode_index


def main():
    parser = argparse.ArgumentParser(description="Visualize pi0.5 single-arm offline inference output")
    parser.add_argument(
        "-d",
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        type=str,
        default=str(Path(__file__).parent / "output"),
    )
    parser.add_argument("-i", "-e", "--episode", type=str, default=None)
    parser.add_argument("--space", choices=["auto", "delta", "absolute"], default="auto")
    parser.add_argument("--position-scale", type=float, default=DEFAULT_POSITION_SCALE)
    parser.add_argument("--position-unit", type=str, default=DEFAULT_POSITION_UNIT)
    parser.add_argument("--no-show", action="store_true", help="Load and render without opening an interactive window")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    trajectory_data, image_source, episodes, episode_index = load_start(data_dir, args.episode)
    visualizer = PI05TrajectoryVisualizer(
        trajectory_data,
        image_source,
        data_dir,
        episodes,
        episode_index,
        space=args.space,
        position_scale=args.position_scale,
        position_unit=args.position_unit,
    )
    if args.no_show:
        visualizer.setup()
        return
    visualizer.run()


if __name__ == "__main__":
    main()
