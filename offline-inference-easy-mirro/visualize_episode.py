#!/usr/bin/env python3
"""Interactive 3D visualizer for easy-mirro pose-block offline inference."""

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
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  Registers the 3D projection.

from image_frame_source import EpisodeImageSource, bgr_to_display_rgb


SPACE_KEYS = {
    "absolute": ("ground_truth_chunk", "predicted_chunk"),
    "delta": ("ground_truth_delta_chunk", "predicted_delta_chunk"),
}
DEFAULT_POSITION_SCALE = 10.0
DEFAULT_POSITION_UNIT = "dm"
DEFAULT_PLAY_INTERVAL_MS = 40
KEY_RELEASE_GRACE_MS = 60
POSE_BLOCK_SIZE = 10
BLOCK_COLORS = [
    ("tab:blue", "tab:red"),
    ("tab:cyan", "tab:orange"),
    ("tab:green", "tab:purple"),
    ("tab:brown", "tab:pink"),
]


def pose_block_offsets(action_dim: int) -> list[int]:
    if action_dim <= 0 or action_dim % POSE_BLOCK_SIZE != 0:
        raise ValueError(f"Action dim must be a positive multiple of 10, got {action_dim}")
    return list(range(0, action_dim, POSE_BLOCK_SIZE))


def display_action(action: np.ndarray, action_dim: int, position_scale: float) -> np.ndarray:
    shown = np.asarray(action, dtype=np.float32).copy()
    for off in pose_block_offsets(action_dim):
        shown[..., off : off + 3] *= position_scale
    return shown


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
        if "absolute" in prediction_space and "absolute" in spaces:
            return "absolute"
        if "delta" in prediction_space and "delta" in spaces:
            return "delta"
        if spaces:
            return spaces[0]
        raise KeyError("Trajectory output does not contain any supported action space keys.")
    pair = trajectory_data["pairs"][0]
    gt_key, pred_key = SPACE_KEYS[requested]
    if gt_key in pair and pred_key in pair:
        return requested
    if requested == "absolute":
        delta_gt, delta_pred = SPACE_KEYS["delta"]
        if delta_gt in pair and delta_pred in pair:
            print("[WARN] absolute space is not present in this output; falling back to delta.")
            return "delta"
    if requested == "delta":
        absolute_gt, absolute_pred = SPACE_KEYS["absolute"]
        if absolute_gt in pair and absolute_pred in pair:
            print("[WARN] delta space is not present in this output; falling back to absolute.")
            return "absolute"
    raise KeyError(f"Trajectory output does not contain {requested!r} space keys.")


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


class TrajectoryVisualizer:
    def __init__(
        self,
        trajectory_data: dict,
        image_source: EpisodeImageSource | None = None,
        data_dir: Path | None = None,
        all_episodes: list[str] | None = None,
        current_episode_idx: int = 0,
        space: str = "delta",
        position_scale: float = DEFAULT_POSITION_SCALE,
        position_unit: str = DEFAULT_POSITION_UNIT,
        play_interval_ms: int = DEFAULT_PLAY_INTERVAL_MS,
    ):
        self.pairs = trajectory_data["pairs"]
        self.chunk_size = trajectory_data["chunk_size"]
        self.action_dim = trajectory_data["action_dim"]
        self.episode_len = trajectory_data["episode_len"]
        self.current_index = 0
        self.image_source = image_source
        self.show_images = image_source is not None
        self.data_dir = data_dir
        self.all_episodes = all_episodes or []
        self.current_episode_idx = current_episode_idx
        self.space = space
        self.position_scale = position_scale
        self.position_unit = position_unit
        self.gt_key, self.pred_key = SPACE_KEYS[space]
        self.pose_blocks = pose_block_offsets(self.action_dim)
        self._axis_limits = None
        self._slider_updating = False
        self.play_interval_ms = play_interval_ms
        self._playing = False
        self._play_timer = None
        self._key_timer = None
        self._held_direction = 0
        self._key_release_timer = None
        self._pending_release_direction = 0

    def _available_cameras(self) -> list[str]:
        if self.image_source is None:
            return []
        return self.image_source.available_cameras()

    def _compute_axis_limits(self):
        if self._axis_limits is not None:
            return self._axis_limits
        xyz = []
        for pair in self.pairs:
            valid = pair["valid_length"]
            gt = display_action(pair[self.gt_key][:valid], self.action_dim, self.position_scale)
            pred = display_action(pair[self.pred_key][:valid], self.action_dim, self.position_scale)
            for off in self.pose_blocks:
                xyz_slice = slice(off, off + 3)
                xyz.extend([gt[:, xyz_slice], pred[:, xyz_slice]])
        all_xyz = np.concatenate(xyz, axis=0)
        lo = np.nanmin(all_xyz, axis=0)
        hi = np.nanmax(all_xyz, axis=0)
        center = (lo + hi) / 2.0
        radius = max(float(np.nanmax(hi - lo)) / 2.0, 0.05 * self.position_scale)
        radius *= 1.15
        self._axis_limits = tuple(zip(center - radius, center + radius))
        return self._axis_limits

    def setup_plot(self):
        camera_names = self._available_cameras()
        image_rows = max(len(camera_names), 1)
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.subplots_adjust(bottom=0.10)
        gs = GridSpec(image_rows, 2, width_ratios=[2.1, 1.0], figure=self.fig)
        self.ax = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.img_axes = [self.fig.add_subplot(gs[row, 1]) for row in range(image_rows)]

        slider_ax = self.fig.add_axes([0.14, 0.025, 0.60, 0.025])
        self.slider = Slider(
            slider_ax,
            "Step",
            0,
            max(len(self.pairs) - 1, 1),
            valinit=0,
            valstep=1,
            valfmt="%d",
        )
        self.slider.on_changed(self._on_slider_changed)
        play_ax = self.fig.add_axes([0.78, 0.017, 0.095, 0.042])
        self.play_button = Button(play_ax, "Play")
        self.play_button.on_clicked(self.toggle_playback)
        self._play_timer = self.fig.canvas.new_timer(interval=self.play_interval_ms)
        self._play_timer.add_callback(self._advance_playback)
        self._key_timer = self.fig.canvas.new_timer(interval=self.play_interval_ms)
        self._key_timer.add_callback(self._advance_held_key)
        self._key_release_timer = self.fig.canvas.new_timer(interval=KEY_RELEASE_GRACE_MS)
        self._key_release_timer.add_callback(self._finish_key_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)
        self.update_plot()

    def _plot_block(self, gt, pred, offset, gt_color, pred_color, label_suffix, linestyle):
        xyz_slice = slice(offset, offset + 3)
        gt_xyz = gt[:, xyz_slice]
        pred_xyz = pred[:, xyz_slice]
        self.ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color=gt_color, lw=2, label=f"GT {label_suffix}")
        self.ax.scatter(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color=gt_color, s=18, alpha=0.55)
        self.ax.plot(
            pred_xyz[:, 0],
            pred_xyz[:, 1],
            pred_xyz[:, 2],
            color=pred_color,
            lw=2,
            ls=linestyle,
            label=f"Pred {label_suffix}",
        )
        self.ax.scatter(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], color=pred_color, s=18, alpha=0.55)
        self.ax.scatter(*gt_xyz[0], color=gt_color, s=90, marker="^", edgecolors="black", linewidth=1.0)
        self.ax.scatter(*gt_xyz[-1], color=gt_color, s=90, marker="s", edgecolors="black", linewidth=1.0)
        self.ax.scatter(*pred_xyz[0], color=pred_color, s=90, marker="x", linewidth=2.0)
        self.ax.scatter(*pred_xyz[-1], color=pred_color, s=90, marker="D", edgecolors="black", linewidth=1.0)

    def _finish_draw(self, force_draw: bool = False) -> None:
        # Do not call flush_events() from key/timer callbacks. Tk processes
        # queued key events inside flush_events(), which can recursively reenter
        # on_key_press during long key holds and keep advancing after release.
        self.fig.canvas.draw_idle()

    def update_plot(self, force_draw: bool = False):
        self.ax.clear()
        pair = self.pairs[self.current_index]
        valid = pair["valid_length"]
        timestep = pair["timestep"]
        gt = display_action(pair[self.gt_key][:valid], self.action_dim, self.position_scale)
        pred = display_action(pair[self.pred_key][:valid], self.action_dim, self.position_scale)

        xlim, ylim, zlim = self._compute_axis_limits()
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_zlim(*zlim)
        for block_idx, off in enumerate(self.pose_blocks):
            gt_color, pred_color = BLOCK_COLORS[block_idx % len(BLOCK_COLORS)]
            label = "Arm" if len(self.pose_blocks) == 1 else f"Block{block_idx}"
            self._plot_block(gt, pred, off, gt_color, pred_color, label, "--")

        axis_len = 0.08 * self.position_scale
        self.ax.scatter(0, 0, 0, color="black", s=80, marker="*")
        self.ax.quiver(0, 0, 0, axis_len, 0, 0, color="red", arrow_length_ratio=0.2)
        self.ax.quiver(0, 0, 0, 0, axis_len, 0, color="green", arrow_length_ratio=0.2)
        self.ax.quiver(0, 0, 0, 0, 0, axis_len, color="blue", arrow_length_ratio=0.2)
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        self.ax.set_xlabel(f"X ({self.position_unit})")
        self.ax.set_ylabel(f"Y ({self.position_unit})")
        self.ax.set_zlabel(f"Z ({self.position_unit})")

        ep_info = ""
        if self.all_episodes:
            ep_name = self.all_episodes[self.current_episode_idx]
            ep_info = f"{ep_name} ({self.current_episode_idx + 1}/{len(self.all_episodes)}) - "
        self.ax.set_title(
            f"{ep_info}{self.space} space | xyz in {self.position_unit} | timestep {timestep}/{self.episode_len - 1} "
            f"| pair {self.current_index + 1}/{len(self.pairs)} | valid {valid}/{self.chunk_size}\n"
            "SPACE/RIGHT: next | LEFT: prev | P/Button: play-pause | UP/DOWN: episode | Q: quit",
            fontsize=10,
        )
        self.ax.legend(loc="upper left", fontsize=8)

        camera_names = self._available_cameras()
        for idx, ax in enumerate(self.img_axes):
            ax.clear()
            if idx < len(camera_names):
                cam_name = camera_names[idx]
                frame = self.image_source.get_frame(cam_name, timestep) if self.image_source is not None else None
                if frame is not None:
                    ax.imshow(bgr_to_display_rgb(frame))
                    ax.set_title(f"{cam_name} | HDF5 frame t={timestep}", fontsize=9)
                else:
                    ax.text(0.5, 0.5, f"{cam_name}\nmissing frame", ha="center", va="center")
                    ax.set_title(cam_name, fontsize=9)
            elif not camera_names:
                ax.text(0.5, 0.5, "No model camera frames found", ha="center", va="center")
            ax.axis("off")

        self._slider_updating = True
        self.slider.valmax = max(len(self.pairs) - 1, 1)
        self.slider.ax.set_xlim(0, self.slider.valmax)
        self.slider.set_val(self.current_index)
        self._slider_updating = False
        self._finish_draw(force_draw=force_draw)

    def _on_slider_changed(self, val):
        if self._slider_updating:
            return
        new_index = int(val)
        if new_index != self.current_index:
            self.current_index = new_index
            self.update_plot()

    def on_key_press(self, event):
        direction = self._direction_from_key(event.key)
        if direction:
            self.stop_playback()
            self._cancel_pending_key_release()
            if self._held_direction == direction:
                return
            self._held_direction = direction
            self.step_by(direction, force_draw=True)
            if self._key_timer is not None:
                self._key_timer.start()
        elif event.key == "up":
            self.stop_key_hold()
            self.stop_playback()
            self.next_episode()
        elif event.key == "down":
            self.stop_key_hold()
            self.stop_playback()
            self.prev_episode()
        elif event.key in ("p", "P"):
            self.stop_key_hold()
            self.toggle_playback()
        elif event.key == "q":
            self.stop_key_hold()
            self.stop_playback()
            plt.close(self.fig)

    def on_key_release(self, event):
        direction = self._direction_from_key(event.key)
        if direction and direction == self._held_direction:
            self._schedule_key_release(direction)

    @staticmethod
    def _direction_from_key(key) -> int:
        if key in (" ", "space", "right"):
            return 1
        if key == "left":
            return -1
        return 0

    def stop_key_hold(self):
        self._held_direction = 0
        self._pending_release_direction = 0
        if self._key_timer is not None:
            self._key_timer.stop()
        if self._key_release_timer is not None:
            self._key_release_timer.stop()

    def _cancel_pending_key_release(self):
        self._pending_release_direction = 0
        if self._key_release_timer is not None:
            self._key_release_timer.stop()

    def _schedule_key_release(self, direction: int):
        # Tk key auto-repeat may emit release/press pairs while the key is still
        # physically held. Stop after a short grace window unless another press
        # for the same direction arrives first.
        self._pending_release_direction = direction
        if self._key_release_timer is not None:
            self._key_release_timer.stop()
            self._key_release_timer.start()

    def _finish_key_release(self):
        if self._key_release_timer is not None:
            self._key_release_timer.stop()
        if self._pending_release_direction and self._pending_release_direction == self._held_direction:
            self.stop_key_hold()
        self._pending_release_direction = 0
        return True

    def _advance_held_key(self):
        if not self._held_direction:
            return True
        self.step_by(self._held_direction, force_draw=True)
        return True

    def step_by(self, direction: int, force_draw: bool = False):
        new_index = int(np.clip(self.current_index + direction, 0, len(self.pairs) - 1))
        if new_index == self.current_index:
            return
        self.current_index = new_index
        self.update_plot(force_draw=force_draw)

    def next_pair(self, force_draw: bool = False):
        self.step_by(1, force_draw=force_draw)

    def prev_pair(self, force_draw: bool = False):
        self.step_by(-1, force_draw=force_draw)

    def toggle_playback(self, _event=None):
        self.stop_key_hold()
        if self._playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if self._play_timer is None:
            return
        self._playing = True
        self.play_button.label.set_text("Pause")
        self._play_timer.start()

    def stop_playback(self):
        self._playing = False
        if self._play_timer is not None:
            self._play_timer.stop()
        if hasattr(self, "play_button"):
            self.play_button.label.set_text("Play")

    def _advance_playback(self):
        if not self._playing:
            return True
        self.current_index = 0 if self.current_index >= len(self.pairs) - 1 else self.current_index + 1
        self.update_plot(force_draw=True)
        return True

    def load_episode_data(self, episode_name: str) -> bool:
        episode_dir = self.data_dir / episode_name
        trajectory_path = episode_dir / "trajectory_pairs.pkl"
        if not trajectory_path.exists():
            print(f"Warning: trajectory file not found: {trajectory_path}")
            return False
        data = load_pickle(trajectory_path)
        self.pairs = data["pairs"]
        self.chunk_size = data["chunk_size"]
        self.action_dim = data["action_dim"]
        self.episode_len = data["episode_len"]
        self.pose_blocks = pose_block_offsets(self.action_dim)
        self.current_index = 0
        self._axis_limits = None
        if self.image_source is not None:
            self.image_source.close()
        self.image_source = EpisodeImageSource(episode_dir, data) if self.show_images else None
        return True

    def next_episode(self):
        if not self.all_episodes or self.current_episode_idx >= len(self.all_episodes) - 1:
            return
        self.current_episode_idx += 1
        if self.load_episode_data(self.all_episodes[self.current_episode_idx]):
            self.update_plot(force_draw=True)

    def prev_episode(self):
        if not self.all_episodes or self.current_episode_idx <= 0:
            return
        self.current_episode_idx -= 1
        if self.load_episode_data(self.all_episodes[self.current_episode_idx]):
            self.update_plot(force_draw=True)

    def run(self, no_show: bool = False):
        self.setup_plot()
        try:
            if no_show:
                self.fig.canvas.draw()
                plt.close(self.fig)
            else:
                plt.show()
        finally:
            self.stop_key_hold()
            self.stop_playback()
            if self.image_source is not None:
                self.image_source.close()


def load_pickle(path: Path):
    install_numpy_pickle_compat()
    with path.open("rb") as f:
        return pickle.load(f)


def find_episodes(data_dir: Path) -> list[str]:
    return sorted(d.name for d in data_dir.iterdir() if d.is_dir() and (d / "trajectory_pairs.pkl").exists())


def main():
    parser = argparse.ArgumentParser(description="Interactive easy-mirro 10D/20D pose-block 3D trajectory visualizer")
    parser.add_argument("-d", "--data-dir", "--data_dir", dest="data_dir", default=None)
    parser.add_argument("-i", "--episode", default=None)
    parser.add_argument("--space", choices=["auto", *sorted(SPACE_KEYS)], default="auto")
    parser.add_argument("--position-scale", type=float, default=DEFAULT_POSITION_SCALE)
    parser.add_argument("--position-unit", default=DEFAULT_POSITION_UNIT)
    parser.add_argument("--play-interval-ms", type=int, default=DEFAULT_PLAY_INTERVAL_MS)
    parser.add_argument("--no-images", action="store_true", help="Do not read/show HDF5 camera frames")
    parser.add_argument("--no-show", action="store_true", help="Render one frame and exit, for smoke tests")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else Path(__file__).parent / "output"
    episodes = find_episodes(data_dir)
    if not episodes:
        raise FileNotFoundError(f"No episode directories with trajectory_pairs.pkl under {data_dir}")
    current_idx = episodes.index(args.episode) if args.episode in episodes else 0
    episode_dir = data_dir / episodes[current_idx]
    data = load_pickle(episode_dir / "trajectory_pairs.pkl")
    image_source = None if args.no_images else EpisodeImageSource(episode_dir, data)

    print(f"Found {len(episodes)} episodes: {episodes}")
    print(f"Loading: {episode_dir / 'trajectory_pairs.pkl'}")
    print(f"Episode length: {data['episode_len']}, chunk size: {data['chunk_size']}, action dim: {data['action_dim']}")
    print("Controls: SPACE/RIGHT=next  LEFT=prev  P/Button=play-pause  UP/DOWN=episode  Q=quit")

    space = resolve_space(data, args.space)
    visualizer = TrajectoryVisualizer(
        data,
        image_source,
        data_dir=data_dir,
        all_episodes=episodes,
        current_episode_idx=current_idx,
        space=space,
        position_scale=args.position_scale,
        position_unit=args.position_unit,
        play_interval_ms=args.play_interval_ms,
    )
    visualizer.run(no_show=args.no_show)


if __name__ == "__main__":
    main()
