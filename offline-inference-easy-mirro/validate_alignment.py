#!/usr/bin/env python3
"""Validate easy-mirro offline GT/action-space alignment.

Checks that offline ground-truth deltas are the same SE3 deltas used by
easy-mirro training, that absolute chunks match HDF5 /action directly, and that
delta->absolute reconstruction is the inverse used by online inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_EASY_MIRRO_ROOT = WORKSPACE_ROOT / "easy-mirro-dual-new_1"
TRAINING_LAYOUT_SOURCES = {"temp/qpos_action", "observations/qpos_action"}
RAW_FRAME0_SOURCES = {"state/6d_rot_frame0_next_action", "state/joint_position_frame0_next_action"}


def install_numpy_pickle_compat() -> None:
    if "numpy._core" not in sys.modules and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
    try:
        import numpy.core.multiarray as multiarray
        import numpy.core.numeric as numeric

        sys.modules.setdefault("numpy._core.multiarray", multiarray)
        sys.modules.setdefault("numpy._core.numeric", numeric)
    except Exception:
        pass


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_eval_module():
    return load_module("eval_easy_mirro", SCRIPT_DIR / "eval_easy_mirro.py")


def load_training_modules(easy_mirro_root: Path):
    root = easy_mirro_root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    dataset_mod = load_module("easy_mirro_dataset", root / "data_utils" / "dataset.py")
    stats_mod = load_module("easy_mirro_stats", root / "data_utils" / "get_norm_stats_by_task.py")
    return dataset_mod, stats_mod


def find_hdf5_files(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.rglob("*.hdf5") if not path.name.startswith("._"))


def selected_indices(length: int, max_frames: int | None) -> list[int]:
    if max_frames is None or max_frames >= length:
        return list(range(length))
    if max_frames <= 1:
        return [0]
    return sorted(set(np.linspace(0, length - 1, max_frames, dtype=int).tolist()))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def validate_hdf5(
    hdf5_path: Path,
    eval_mod,
    dataset_mod,
    stats_mod,
    chunk_size: int,
    max_frames: int | None,
    require_training_layout: bool,
    allow_raw_fallback: bool,
) -> dict:
    with h5py.File(hdf5_path, "r") as root:
        qpos, actions, source, _ = eval_mod.read_qpos_actions(
            root,
            hdf5_path,
            allow_raw_fallback=allow_raw_fallback,
        )
    qpos = np.asarray(qpos, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    action_dim = actions.shape[-1]
    if qpos.shape[-1] < action_dim:
        raise ValueError(f"{hdf5_path}: qpos dim {qpos.shape[-1]} < action dim {action_dim}")
    qpos = qpos[:, : max(qpos.shape[-1], action_dim)]
    actions = actions[:, :action_dim]
    episode_len = min(len(qpos), len(actions))
    qpos = qpos[:episode_len]
    actions = actions[:episode_len]

    if require_training_layout and source not in TRAINING_LAYOUT_SOURCES:
        raise ValueError(f"{hdf5_path} state_source={source} is not canonical training GT layout.")

    indices = selected_indices(episode_len, max_frames)
    stats_delta_all = stats_mod.compute_chunk_delta(qpos, actions, chunk_size)

    max_eval_vs_dataset = 0.0
    max_eval_vs_stats = 0.0
    max_roundtrip = 0.0
    for t in indices:
        valid = min(chunk_size, episode_len - t)
        action_seq = actions[t : t + valid]
        eval_delta = eval_mod.compute_action_delta(qpos[t], action_seq)
        dataset_delta = dataset_mod._compute_action_delta(qpos[t], action_seq)
        stats_delta = stats_delta_all[t, :valid]
        recon_abs = eval_mod.delta_chunk_to_absolute(qpos[t], eval_delta)

        max_eval_vs_dataset = max(max_eval_vs_dataset, max_abs(eval_delta, dataset_delta))
        max_eval_vs_stats = max(max_eval_vs_stats, max_abs(eval_delta, stats_delta))
        max_roundtrip = max(max_roundtrip, max_abs(recon_abs, action_seq))

    return {
        "episode": hdf5_path.stem,
        "path": str(hdf5_path),
        "episode_len": episode_len,
        "checked_frames": len(indices),
        "state_source": source,
        "uses_training_ground_truth_layout": source in TRAINING_LAYOUT_SOURCES,
        "max_eval_vs_training_dataset_delta": max_eval_vs_dataset,
        "max_eval_vs_norm_stats_delta": max_eval_vs_stats,
        "max_delta_to_absolute_roundtrip": max_roundtrip,
    }


def validate_output_dir(output_dir: Path, eval_mod, max_frames: int | None) -> list[dict]:
    if output_dir is None or not output_dir.exists():
        return []
    results = []
    for episode_dir in sorted(path for path in output_dir.iterdir() if (path / "trajectory_pairs.pkl").exists()):
        install_numpy_pickle_compat()
        with (episode_dir / "trajectory_pairs.pkl").open("rb") as f:
            data = pickle.load(f)
        source_hdf5_path = Path(data.get("source_hdf5_path", ""))
        qpos = None
        actions = None
        if source_hdf5_path.exists():
            with h5py.File(source_hdf5_path, "r") as root:
                qpos, actions, _, _ = eval_mod.read_qpos_actions(root, source_hdf5_path)
            qpos = np.asarray(qpos, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.float32)
            action_dim = int(data.get("action_dim", actions.shape[-1]))
            qpos = qpos[:, : max(qpos.shape[-1], action_dim)]
            actions = actions[:, :action_dim]
        pairs = data["pairs"]
        indices = selected_indices(len(pairs), max_frames)
        max_saved_gt_delta_consistency = 0.0
        max_saved_gt_absolute_consistency = 0.0
        for idx in indices:
            pair = pairs[idx]
            valid = pair["valid_length"]
            finite_arrays = []
            if "ground_truth_delta_chunk" in pair:
                finite_arrays.append(pair["ground_truth_delta_chunk"][:valid])
            if "predicted_delta_chunk" in pair:
                finite_arrays.append(pair["predicted_delta_chunk"][:valid])
            if "ground_truth_chunk" in pair:
                finite_arrays.append(pair["ground_truth_chunk"][:valid])
            if "predicted_chunk" in pair:
                finite_arrays.append(pair["predicted_chunk"][:valid])
            if not finite_arrays or not all(np.isfinite(arr).all() for arr in finite_arrays):
                raise ValueError(f"Non-finite values found in {episode_dir.name} pair {idx}")
            if qpos is not None and actions is not None:
                t = pair["timestep"]
                chunk_size = data["chunk_size"]
                valid_full = pair["valid_length"]
                end = min(t + chunk_size, len(actions))
                action_dim = int(data.get("action_dim", actions.shape[-1]))
                if "ground_truth_delta_chunk" in pair:
                    expected_gt_delta = np.zeros((chunk_size, action_dim), dtype=np.float32)
                    expected_gt_delta[: end - t] = eval_mod.compute_action_delta(qpos[t], actions[t:end, :action_dim])
                    max_saved_gt_delta_consistency = max(
                        max_saved_gt_delta_consistency,
                        max_abs(pair["ground_truth_delta_chunk"][:valid_full], expected_gt_delta[:valid_full]),
                    )
                if "ground_truth_chunk" in pair:
                    expected_gt_absolute = np.zeros((chunk_size, action_dim), dtype=np.float32)
                    expected_gt_absolute[: end - t] = actions[t:end, :action_dim]
                    max_saved_gt_absolute_consistency = max(
                        max_saved_gt_absolute_consistency,
                        max_abs(pair["ground_truth_chunk"][:valid_full], expected_gt_absolute[:valid_full]),
                    )
        results.append(
            {
                "episode": episode_dir.name,
                "pairs": len(pairs),
                "checked_pairs": len(indices),
                "chunk_size": data["chunk_size"],
                "action_dim": data["action_dim"],
                "finite_and_shape_ok": True,
                "source_hdf5_path": str(source_hdf5_path) if source_hdf5_path else None,
                "max_saved_gt_delta_consistency": max_saved_gt_delta_consistency,
                "max_saved_gt_absolute_consistency": max_saved_gt_absolute_consistency,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate easy-mirro offline/training delta alignment")
    parser.add_argument("--data-dir", required=True, help="HDF5 directory to validate")
    parser.add_argument("--output-dir", default=None, help="Optional offline inference output directory")
    parser.add_argument("--easy-mirro-root", default=str(DEFAULT_EASY_MIRRO_ROOT))
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=64, help="Sample up to N frames per episode")
    parser.add_argument("--require-training-layout", action="store_true")
    parser.add_argument("--allow-raw-fallback", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    eval_mod = load_eval_module()
    dataset_mod, stats_mod = load_training_modules(Path(args.easy_mirro_root))
    files = find_hdf5_files(Path(args.data_dir).expanduser().resolve())
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {args.data_dir}")

    print(f"Validating {len(files)} HDF5 files with chunk_size={args.chunk_size}")
    failures = []
    for hdf5_path in files:
        result = validate_hdf5(
            hdf5_path,
            eval_mod,
            dataset_mod,
            stats_mod,
            args.chunk_size,
            args.max_frames,
            args.require_training_layout,
            args.allow_raw_fallback,
        )
        print(
            f"{result['episode']}: source={result['state_source']}, "
            f"frames={result['checked_frames']}/{result['episode_len']}, "
            f"eval_vs_dataset={result['max_eval_vs_training_dataset_delta']:.3e}, "
            f"eval_vs_stats={result['max_eval_vs_norm_stats_delta']:.3e}, "
            f"roundtrip={result['max_delta_to_absolute_roundtrip']:.3e}"
        )
        for key in (
            "max_eval_vs_training_dataset_delta",
            "max_eval_vs_norm_stats_delta",
            "max_delta_to_absolute_roundtrip",
        ):
            if result[key] > args.tolerance:
                failures.append((result["episode"], key, result[key]))
        if not result["uses_training_ground_truth_layout"]:
            print(
                "  WARN: this file does not contain the canonical training GT layout "
                "(/observations/qpos + /action); validation uses the same frame-0 conversion/action derivation as eval_easy_mirro.py."
            )

    if args.output_dir:
        out_results = validate_output_dir(Path(args.output_dir).expanduser().resolve(), eval_mod, args.max_frames)
        for result in out_results:
            print(
                f"output {result['episode']}: pairs={result['pairs']}, "
                f"checked={result['checked_pairs']}, chunk={result['chunk_size']}, "
                f"action_dim={result['action_dim']}, finite_shape_ok={result['finite_and_shape_ok']}, "
                f"saved_gt_delta={result['max_saved_gt_delta_consistency']:.3e}, "
                f"saved_gt_absolute={result['max_saved_gt_absolute_consistency']:.3e}"
            )
            for key in ("max_saved_gt_delta_consistency", "max_saved_gt_absolute_consistency"):
                if result[key] > args.tolerance:
                    failures.append((f"output/{result['episode']}", key, result[key]))

    if failures:
        for episode, key, value in failures:
            print(f"FAIL: {episode} {key}={value:.6e} > tolerance={args.tolerance:.6e}")
        raise SystemExit(1)
    print("Alignment validation passed.")


if __name__ == "__main__":
    main()
