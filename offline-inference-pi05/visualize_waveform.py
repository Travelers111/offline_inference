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
from matplotlib.widgets import Slider

from lerobot_frame_source import EpisodeImageSource


SPACE_KEYS = {
    "absolute": ("ground_truth_chunk", "predicted_chunk"),
    "delta": ("ground_truth_delta_chunk", "predicted_delta_chunk"),
}
DEFAULT_POSITION_SCALE = 10.0
DEFAULT_POSITION_UNIT = "dm"


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
        space: str = "auto",
        position_scale: float = DEFAULT_POSITION_SCALE,
        position_unit: str = DEFAULT_POSITION_UNIT,
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
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.06, hspace=0.6, wspace=0.14)

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
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

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
            f"{ep_info}{self.space} | Timestep {timestep}/{self.episode_len - 1}  "
            f"(Pair {self.current_index + 1}/{len(self.pairs)}, chunk={valid}/{self.chunk_size})   "
            "SPACE/RIGHT: next | LEFT: prev | UP/DOWN: episode | Q: quit",
            fontsize=10,
        )

    def update_plot(self):
        self._draw_chunk_overlay()
        self._draw_images()
        self._update_title()
        if hasattr(self, "slider"):
            self._slider_updating = True
            self.slider.valmax = max(len(self.pairs) - 1, 1)
            self.slider.ax.set_xlim(0, self.slider.valmax)
            self.slider.set_val(self.current_index)
            self._slider_updating = False
        self.fig.canvas.draw_idle()

    def _on_slider_changed(self, val):
        if self._slider_updating:
            return
        new_index = int(val)
        if new_index != self.current_index:
            self.current_index = new_index
            self.update_plot()

    def on_key_press(self, event):
        if event.key in (" ", "right"):
            self.next_pair()
        elif event.key == "left":
            self.prev_pair()
        elif event.key == "up":
            self.next_episode()
        elif event.key == "down":
            self.prev_episode()
        elif event.key == "q":
            plt.close(self.fig)

    def next_pair(self):
        if self.current_index < len(self.pairs) - 1:
            self.current_index += 1
            self.update_plot()

    def prev_pair(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_plot()

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
            self.update_plot()

    def prev_episode(self):
        if not self.all_episodes or self.current_episode_idx <= 0:
            return
        self.current_episode_idx -= 1
        if self.load_episode_data(self.all_episodes[self.current_episode_idx]):
            self.update_plot()

    def run(self):
        self.setup_plot()
        plt.show()


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
    parser.add_argument("--no-show", action="store_true", help="Load and render without opening a window")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else Path(__file__).parent / "output"
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

    print(f"Episode length: {trajectory_data['episode_len']}")
    print(f"Chunk size:     {trajectory_data['chunk_size']}")
    print(f"Action dim:     {trajectory_data['action_dim']}")
    print(f"Pairs:          {len(trajectory_data['pairs'])}")
    print("\nControls: SPACE/RIGHT=next  LEFT=prev  UP=next episode  DOWN=prev episode  Q=quit")

    image_source = EpisodeImageSource.from_episode_dir(episode_dir, trajectory_data)
    visualizer = WaveformVisualizer(
        trajectory_data,
        image_source=image_source,
        data_dir=data_dir,
        all_episodes=all_episodes,
        current_episode_idx=current_episode_idx,
        space=args.space,
        position_scale=args.position_scale,
        position_unit=args.position_unit,
    )
    if args.no_show:
        visualizer.setup_plot()
        return
    visualizer.run()


if __name__ == "__main__":
    main()
