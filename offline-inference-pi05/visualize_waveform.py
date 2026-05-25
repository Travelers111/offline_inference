#!/usr/bin/env python
"""Waveform visualizer for single-arm pi0.5 offline inference output.

This mirrors offline-inference/visualize_waveform.py, adapted from the old
20D dual-arm layout to the pi0.5 UMI 10D pose action.
"""

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
DEFAULT_POSITION_SCALE = 10.0
DEFAULT_POSITION_UNIT = "dm"
DEFAULT_PLAY_INTERVAL_MS = 40
KEY_RELEASE_GRACE_MS = 60


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


def display_action(action: np.ndarray, position_scale: float) -> np.ndarray:
    shown = np.asarray(action, dtype=np.float32).copy()
    shown[..., 0:3] *= position_scale
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


class WaveformVisualizer:
    def __init__(
        self,
        trajectory_data: dict,
        image_source: EpisodeImageSource | None = None,
        data_dir: Path | None = None,
        all_episodes: list[str] | None = None,
        current_episode_idx: int = 0,
        run_info: dict | None = None,
        space: str = "auto",
        position_scale: float = DEFAULT_POSITION_SCALE,
        position_unit: str = DEFAULT_POSITION_UNIT,
        play_interval_ms: int = DEFAULT_PLAY_INTERVAL_MS,
    ):
        self.pairs = trajectory_data["pairs"]
        self.chunk_size = trajectory_data["chunk_size"]
        self.action_dim = trajectory_data["action_dim"]
        self.episode_len = trajectory_data["episode_len"]
        self.space = resolve_space(trajectory_data, space)
        self.gt_key, self.pred_key = SPACE_KEYS[self.space]
        self.position_scale = position_scale
        self.position_unit = position_unit
        self.current_index = 0
        self.image_source = image_source
        self.data_dir = data_dir
        self.all_episodes = all_episodes or []
        self.current_episode_idx = current_episode_idx
        self.run_info = run_info or {}

        self.dims = {
            "xyz": slice(0, 3),
            "rot": slice(3, 9),
            "gripper": slice(9, 10),
        }
        self.labels = {
            "xyz": ["X", "Y", "Z"],
            "rot": ["R0", "R1", "R2", "R3", "R4", "R5"],
            "gripper": ["Gripper"],
        }

        self._gt_bg = None
        self._chunk_artists = []
        self._vlines = []
        self.image_axes = []
        self._slider_updating = False
        self.play_interval_ms = play_interval_ms
        self._playing = False
        self._play_timer = None
        self._key_timer = None
        self._held_direction = 0
        self._key_release_timer = None
        self._pending_release_direction = 0

    def _build_episode_arrays(self):
        gt = np.zeros((len(self.pairs), self.action_dim), dtype=np.float32)
        for idx, pair in enumerate(self.pairs):
            gt[idx] = pair[self.gt_key][0]
        gt = display_action(gt, self.position_scale)
        self._gt_bg = gt

    def _chunk_xaxis(self, pair_index: int):
        start = self.pairs[pair_index]["timestep"]
        valid = self.pairs[pair_index]["valid_length"]
        return np.arange(start, start + valid)

    def setup_plot(self):
        self._build_episode_arrays()
        image_count = len(self.image_source.available_cameras()) if self.image_source is not None else 0
        self.fig = plt.figure(figsize=(17 if image_count else 14, 18))
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.06, hspace=0.6, wspace=0.14)

        n_rows = 10
        if image_count:
            gs = GridSpec(
                n_rows + 1,
                2,
                figure=self.fig,
                width_ratios=[3.0, 1.2],
                height_ratios=[1] * n_rows + [0.25],
            )
        else:
            gs = GridSpec(n_rows + 1, 1, figure=self.fig, height_ratios=[1] * n_rows + [0.25])

        self.axes = []
        row = 0
        for kind in ["xyz", "rot", "gripper"]:
            for _ in self.labels[kind]:
                ax = self.fig.add_subplot(gs[row, 0] if image_count else gs[row])
                ax.tick_params(labelsize=7)
                self.axes.append((kind, row, ax))
                row += 1

        self.image_axes = []
        if image_count:
            for idx in range(image_count):
                start = int(idx * n_rows / image_count)
                stop = max(start + 1, int((idx + 1) * n_rows / image_count))
                self.image_axes.append(self.fig.add_subplot(gs[start:stop, 1]))

        slider_ax = self.fig.add_axes([0.1, 0.015, 0.8, 0.02])
        self.slider = Slider(
            slider_ax,
            "Timestep",
            0,
            max(len(self.pairs) - 1, 1),
            valinit=0,
            valstep=1,
            valfmt="%d",
        )
        self.slider.on_changed(self._on_slider_changed)
        play_ax = self.fig.add_axes([0.92, 0.008, 0.065, 0.035])
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

        self._draw_background()
        self._draw_chunk_overlay()
        self._draw_images()
        self._update_title()

    def _dim_for_axis(self, kind: str, row: int) -> int:
        if kind == "xyz":
            return self.dims[kind].start + row
        if kind == "rot":
            return self.dims[kind].start + (row - 3)
        return 9

    def _label_for_axis(self, kind: str, row: int) -> str:
        if kind == "xyz":
            return f"{self.labels[kind][row]} ({self.position_unit})"
        if kind == "rot":
            return f"Rot {self.labels[kind][row - 3]}"
        return "Gripper"

    def _draw_background(self):
        t = np.arange(len(self.pairs))
        for kind, row, ax in self.axes:
            ax.clear()
            dim = self._dim_for_axis(kind, row)
            ax.plot(t, self._gt_bg[:, dim], color="steelblue", lw=0.8, alpha=0.35, zorder=1)
            ax.set_title(self._label_for_axis(kind, row), fontsize=8, pad=2)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.plot([], [], color="steelblue", lw=2, label="GT chunk")
                ax.plot([], [], color="tomato", lw=2, ls="--", label="Pred chunk")
                ax.legend(fontsize=7, loc="upper right", framealpha=0.5)

    def _draw_chunk_overlay(self):
        for artist in self._chunk_artists + self._vlines:
            artist.remove()
        self._chunk_artists = []
        self._vlines = []

        pair = self.pairs[self.current_index]
        gt_chunk = display_action(pair[self.gt_key], self.position_scale)
        pred_chunk = display_action(pair[self.pred_key], self.position_scale)
        valid = pair["valid_length"]
        x = self._chunk_xaxis(self.current_index)

        for kind, row, ax in self.axes:
            dim = self._dim_for_axis(kind, row)
            gt_line = ax.plot(x, gt_chunk[:valid, dim], color="steelblue", lw=2.2, alpha=0.95, zorder=3)[0]
            pred_line = ax.plot(
                x,
                pred_chunk[:valid, dim],
                color="tomato",
                lw=2.2,
                ls="--",
                alpha=0.95,
                zorder=3,
            )[0]
            vline = ax.axvline(x=x[0], color="orange", lw=1.2, alpha=0.8, zorder=4)
            self._chunk_artists.extend([gt_line, pred_line])
            self._vlines.append(vline)

    def _draw_images(self):
        if self.image_source is None:
            return
        timestep = self.pairs[self.current_index]["timestep"]
        frame_items = self.image_source.get_frames(timestep)
        for ax, (camera, frame) in zip(self.image_axes, frame_items, strict=False):
            ax.clear()
            ax.imshow(frame)
            ax.set_title(f"{camera} | frame {timestep}", fontsize=9)
            ax.axis("off")
        for ax in self.image_axes[len(frame_items) :]:
            ax.clear()
            ax.axis("off")

    def _update_title(self):
        pair = self.pairs[self.current_index]
        timestep = pair["timestep"]
        valid = pair["valid_length"]
        ep_info = ""
        if self.all_episodes:
            ep_name = self.all_episodes[self.current_episode_idx]
            ep_info = f"{ep_name} ({self.current_episode_idx + 1}/{len(self.all_episodes)}) | "
        self.fig.suptitle(
            f"{short_run_title(self.run_info)}\n"
            f"{ep_info}{self.space} | Timestep {timestep}/{self.episode_len - 1}  "
            f"(Pair {self.current_index + 1}/{len(self.pairs)}, chunk={valid}/{self.chunk_size})   "
            "SPACE/RIGHT: next | LEFT: prev | P/Button: play-pause | UP/DOWN: episode | Q: quit",
            fontsize=10,
        )

    def _finish_draw(self, force_draw: bool = False):
        # Do not call flush_events() from key/timer callbacks. Some GUI
        # backends recursively process queued key events there during long
        # holds, which can keep advancing after release.
        self.fig.canvas.draw_idle()

    def update_plot(self, force_draw: bool = False):
        self._draw_chunk_overlay()
        self._draw_images()
        self._update_title()
        if hasattr(self, "slider"):
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
            self.stop_key_hold()
            self.stop_playback()
            self.current_index = new_index
            self.update_plot(force_draw=True)

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
        self.current_index = 0 if self.current_index >= len(self.pairs) - 1 else self.current_index + 1
        self.update_plot(force_draw=True)
        return True

    def load_episode_data(self, episode_name: str):
        trajectory_path = self.data_dir / episode_name / "trajectory_pairs.pkl"
        if not trajectory_path.exists():
            print(f"Warning: Not found: {trajectory_path}")
            return False
        print(f"\nLoading episode: {episode_name}")
        install_numpy_pickle_compat()
        with trajectory_path.open("rb") as f:
            trajectory_data = pickle.load(f)
        episode_dir = self.data_dir / episode_name
        self.image_source = EpisodeImageSource.from_episode_dir(episode_dir, trajectory_data)
        self.run_info = merge_episode_info(self.run_info, load_episode_metadata(episode_dir))
        self.pairs = trajectory_data["pairs"]
        self.chunk_size = trajectory_data["chunk_size"]
        self.action_dim = trajectory_data["action_dim"]
        self.episode_len = trajectory_data["episode_len"]
        self.space = resolve_space(trajectory_data, self.space)
        self.gt_key, self.pred_key = SPACE_KEYS[self.space]
        self.current_index = 0
        self._chunk_artists = []
        self._vlines = []
        self._build_episode_arrays()
        self._draw_background()
        self._draw_images()
        print(f"Episode length: {self.episode_len}, Pairs: {len(self.pairs)}")
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


def main():
    parser = argparse.ArgumentParser(
        description="Waveform visualizer: full-episode GT + per-step chunk GT vs Pred"
    )
    parser.add_argument("-i", "--episode", type=str, default=None)
    parser.add_argument(
        "-d",
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--space", choices=["auto", "delta", "absolute"], default="auto")
    parser.add_argument("--position-scale", type=float, default=DEFAULT_POSITION_SCALE)
    parser.add_argument("--position-unit", type=str, default=DEFAULT_POSITION_UNIT)
    parser.add_argument("--play-interval-ms", type=int, default=DEFAULT_PLAY_INTERVAL_MS)
    parser.add_argument("--no-show", action="store_true", help="Load and render without opening a window")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else Path(__file__).parent / "output"
    run_info = load_run_info(data_dir)
    all_episodes = sorted(
        d.name for d in data_dir.iterdir() if d.is_dir() and (d / "trajectory_pairs.pkl").exists()
    )
    if not all_episodes:
        print(f"Error: No episode directories with trajectory_pairs.pkl in {data_dir}")
        return

    print(f"Found {len(all_episodes)} episodes: {all_episodes}")
    current_episode_idx = all_episodes.index(args.episode) if args.episode in all_episodes else 0
    starting_episode = all_episodes[current_episode_idx]
    episode_dir = data_dir / starting_episode
    trajectory_path = data_dir / starting_episode / "trajectory_pairs.pkl"

    print(f"\nLoading: {trajectory_path}")
    install_numpy_pickle_compat()
    with trajectory_path.open("rb") as f:
        trajectory_data = pickle.load(f)
    run_info = merge_episode_info(run_info, load_episode_metadata(episode_dir))

    print(f"Episode length: {trajectory_data['episode_len']}")
    print(f"Chunk size:     {trajectory_data['chunk_size']}")
    print(f"Action dim:     {trajectory_data['action_dim']}")
    print(f"Pairs:          {len(trajectory_data['pairs'])}")
    print("\nControls: SPACE/RIGHT=next  LEFT=prev  P/Button=play-pause  UP/DOWN=episode  Q=quit")

    image_source = EpisodeImageSource.from_episode_dir(episode_dir, trajectory_data)
    visualizer = WaveformVisualizer(
        trajectory_data,
        image_source=image_source,
        data_dir=data_dir,
        all_episodes=all_episodes,
        current_episode_idx=current_episode_idx,
        run_info=run_info,
        space=args.space,
        position_scale=args.position_scale,
        position_unit=args.position_unit,
        play_interval_ms=args.play_interval_ms,
    )
    visualizer.run(no_show=args.no_show)


if __name__ == "__main__":
    main()
