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
from matplotlib.widgets import Button, Slider

from lerobot_frame_source import EpisodeImageSource
from viz_run_info import load_episode_metadata, load_run_info, merge_episode_info, short_run_title


SPACE_KEYS = {
    "absolute": ("ground_truth_chunk", "predicted_chunk"),
    "delta": ("ground_truth_delta_chunk", "predicted_delta_chunk"),
}
POSE_BLOCK_SIZE = 10
DEFAULT_POSITION_SCALE = 10.0
DEFAULT_POSITION_UNIT = "dm"
DEFAULT_PLAY_INTERVAL_MS = 40
KEY_RELEASE_GRACE_MS = 60


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
        run_info: dict,
        space: str,
        position_scale: float,
        position_unit: str,
        play_interval_ms: int,
    ):
        self.trajectory_data = trajectory_data
        self.image_source = image_source
        self.data_dir = data_dir
        self.episodes = episodes
        self.episode_index = episode_index
        self.run_info = run_info
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
        self.play_interval_ms = play_interval_ms
        self._playing = False
        self._play_timer = None
        self._key_timer = None
        self._held_direction = 0
        self._key_release_timer = None
        self._pending_release_direction = 0

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
        self.run_info = merge_episode_info(self.run_info, load_episode_metadata(episode_dir))
        self.pairs = self.trajectory_data["pairs"]
        self.action_dim = int(self.trajectory_data["action_dim"])
        self.space = resolve_space(self.trajectory_data, self.space)
        self.gt_key, self.pred_key = SPACE_KEYS[self.space]
        self.pose_blocks = pose_block_offsets(self.action_dim)
        self.current = 0
        self._axis_limits = None
        return True

    def step_by(self, direction: int, force_draw: bool = False):
        new_index = int(np.clip(self.current + direction, 0, len(self.pairs) - 1))
        if new_index == self.current:
            return
        self.current = new_index
        self.update(force_draw=force_draw)

    def next_pair(self, force_draw: bool = False):
        self.step_by(1, force_draw=force_draw)

    def prev_pair(self, force_draw: bool = False):
        self.step_by(-1, force_draw=force_draw)

    def next_episode(self):
        if self.episode_index < len(self.episodes) - 1:
            self.episode_index += 1
            if self.load_episode(self.episodes[self.episode_index]):
                self.update(force_draw=True)

    def prev_episode(self):
        if self.episode_index > 0:
            self.episode_index -= 1
            if self.load_episode(self.episodes[self.episode_index]):
                self.update(force_draw=True)

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
        elif event.key in {"p", "P"}:
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
        if key in {" ", "space", "right"}:
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
        # Tk key auto-repeat can emit release/press pairs while the key is
        # still physically held. Stop after a short grace window unless another
        # press for the same direction arrives first.
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
        if hasattr(self, "play_button"):
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
        self.current = 0 if self.current >= len(self.pairs) - 1 else self.current + 1
        self.update(force_draw=True)
        return True

    def _capture_view(self):
        if self.ax_traj is None:
            return None
        return (
            getattr(self.ax_traj, "elev", None),
            getattr(self.ax_traj, "azim", None),
            getattr(self.ax_traj, "roll", None),
        )

    def _restore_view(self, view):
        if self.ax_traj is None or view is None:
            return
        elev, azim, roll = view
        if elev is None or azim is None:
            return
        if roll is not None:
            try:
                self.ax_traj.view_init(elev=elev, azim=azim, roll=roll)
                return
            except TypeError:
                pass
        self.ax_traj.view_init(elev=elev, azim=azim)

    def _finish_draw(self, force_draw: bool = False):
        # Do not call flush_events() from key/timer callbacks. Some GUI
        # backends recursively process queued key events there during long
        # holds, which can keep advancing after release.
        self.fig.canvas.draw_idle()

    def on_slider(self, value):
        if self._slider_updating:
            return
        self.current = int(value)
        self.stop_key_hold()
        self.stop_playback()
        self.update(force_draw=True)

    def setup(self):
        image_count = len(self.image_source.available_cameras()) if self.image_source is not None else 0
        rows = max(1, image_count)
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.subplots_adjust(bottom=0.08, top=0.9)
        gs = GridSpec(rows, 2, width_ratios=[2, 1])
        self.ax_traj = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.image_axes = [self.fig.add_subplot(gs[i, 1]) for i in range(rows)]
        slider_ax = self.fig.add_axes([0.15, 0.015, 0.7, 0.025])
        self.slider = Slider(slider_ax, "Step", 0, max(len(self.pairs) - 1, 1), valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider)
        play_ax = self.fig.add_axes([0.88, 0.01, 0.075, 0.04])
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
        self.update(force_draw=True)

    def update(self, force_draw: bool = False):
        pair = self.pairs[self.current]
        timestep = pair["timestep"]
        gt = display_action(pair[self.gt_key], self.action_dim, self.position_scale)
        pred = display_action(pair[self.pred_key], self.action_dim, self.position_scale)
        valid = pair["valid_length"]
        chunk_size = self.trajectory_data["chunk_size"]

        view = self._capture_view()
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
        self.ax_traj.set_xlabel(f"X ({self.position_unit})")
        self.ax_traj.set_ylabel(f"Y ({self.position_unit})")
        self.ax_traj.set_zlabel(f"Z ({self.position_unit})")
        try:
            self.ax_traj.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        self._restore_view(view)
        episode_name = self.episodes[self.episode_index]
        self.fig.suptitle(short_run_title(self.run_info), fontsize=12, y=0.98)
        self.ax_traj.set_title(
            f"{episode_name}  {self.space}  step {timestep}  pair {self.current + 1}/{len(self.pairs)}  "
            f"valid {valid}/{chunk_size}\n"
            "SPACE/RIGHT: next  LEFT: prev  P/Button: play-pause  UP/DOWN: episode  Q: quit"
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
        self._finish_draw(force_draw=force_draw)

    def run(self, no_show: bool = False):
        self.setup()
        try:
            if no_show:
                self.fig.canvas.draw()
                plt.close(self.fig)
            else:
                plt.show()
        finally:
            self.stop_key_hold()
            self.stop_playback()


def load_start(data_dir: Path, requested_episode: str | None):
    install_numpy_pickle_compat()
    run_info = load_run_info(data_dir)
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
    run_info = merge_episode_info(run_info, load_episode_metadata(episode_dir))
    return trajectory_data, image_source, episodes, episode_index, run_info


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
    parser.add_argument("--play-interval-ms", type=int, default=DEFAULT_PLAY_INTERVAL_MS)
    parser.add_argument("--no-show", action="store_true", help="Load and render without opening an interactive window")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    trajectory_data, image_source, episodes, episode_index, run_info = load_start(data_dir, args.episode)
    visualizer = PI05TrajectoryVisualizer(
        trajectory_data,
        image_source,
        data_dir,
        episodes,
        episode_index,
        run_info,
        space=args.space,
        position_scale=args.position_scale,
        position_unit=args.position_unit,
        play_interval_ms=args.play_interval_ms,
    )
    visualizer.run(no_show=args.no_show)


if __name__ == "__main__":
    main()
