# easy-mirro Pi0 Offline Inference

This is the HDF5 offline inference/evaluation entry for `easy-mirro-dual-new_1` Pi0 EasyMirror checkpoints.

For the full Chinese project guide, see [`PROJECT_GUIDE_zh.md`](PROJECT_GUIDE_zh.md).

It follows the model contract from `easy-mirro-dual-new_1/README_PI0_CONFIG.md`:

- input state is 20D absolute frame-0 pose, padded to 32D
- delta checkpoints output chunk-wise 20D SE3 delta action
- absolute checkpoints output chunk-wise 20D HDF5 `/action`
- primary metrics follow checkpoint `action_mode`, read from config/stats by default
- normalization stats are loaded only from `dataset_stats.pkl` beside the checkpoint or its parent
- camera inputs default to `auto`, which reads `image_features` from the checkpoint config

## Run

```bash
export CHECKPOINT=/home/eai/debug/ckpt__fz/双相机相对输出/checkpoint-45000
export EVAL_DATA_DIR=/home/eai/debug/rawdata
export BATCH_SIZE=8

bash offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

For the three-camera SE3-delta checkpoint, no code change or manual `CAMERAS`
setting is needed:

```bash
export CHECKPOINT=/home/eai/debug/ckpt__fz/3cam_se3_delta_mixed_output_45000_inference
export EVAL_DATA_DIR=/home/eai/debug/rawdata

bash offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

For the three-camera absolute-action checkpoint, the evaluator will automatically
switch to absolute output comparison:

```bash
export CHECKPOINT=/home/eai/debug/ckpt__fz/3cam_absolute_mixed_output_45000_inference
export EVAL_DATA_DIR=/home/eai/debug/rawdata

bash offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

For the single-arm blackboard task, use the blackboard wrapper so the correct
model code root is used. The checkpoint must be a real trained checkpoint
directory with safetensors plus `config.json`/`dataset_stats.pkl`:

```bash
export CHECKPOINT=/path/to/pi0_wiping_blackboard/checkpoint-45000
export EVAL_DATA_DIR=/home/eai/debug/data/blackboard_testdata

bash offline-inference-easy-mirro/run_eval_blackboard.sh --strict
```

The blackboard task uses `state_dim=10`, `action_dim=10`, `cam_high,cam_fisheye`,
absolute state input, and SE3-delta action output.

The evaluator will read checkpoint `image_features`, for example:

```text
['observation.images.cam_high',
 'observation.images.cam_left_wrist',
 'observation.images.cam_right_wrist']
```

Manual override is still available:

```bash
export CAMERAS=cam_left_wrist,cam_right_wrist
```

List available HDF5 episodes:

```bash
bash offline-inference-easy-mirro/run_eval_easy_mirro.sh --list-episodes
```

Use a training output root instead of a concrete checkpoint:

```bash
export CHECKPOINT=/home/eai/debug/ckpt__fz
```

The script will pick the latest `checkpoint-*` directory and load `dataset_stats.pkl` from the checkpoint or its parent.

Evaluate both Fangzhou checkpoints on canonical `rawdata`, validate alignment, render previews, and write a checkpoint comparison:

```bash
bash offline-inference-easy-mirro/run_eval_fz_test2_all.sh
```

Default output for the all-run:

```text
offline_inference_output/fz_rawdata/
  checkpoint-35000/
  checkpoint-45000/
  comparison.json
```

## Required HDF5 Layout

Canonical training-layout HDF5 files must contain `/observations/qpos` and `/action`.

The evaluator now rejects raw collection files by default. This is intentional: raw files such as `test2` do not contain `/observations/qpos` and `/action`, so converting them inside eval can silently diverge from the preprocessing used during training.

```text
/observations/qpos  (N, 20)  absolute frame-0 state
/action             (N, 20)  absolute future action; action[t] is the first future target
```

Single-arm blackboard HDF5 files use the same paths but 10D vectors:

```text
/observations/qpos  (N, 10)
/action             (N, 10)
/observations/images/cam_high
/observations/images/cam_fisheye
```

For each timestep, ground-truth delta is computed exactly as in training:

```text
gt_delta[t, k] = inv(T_qpos[t]) @ T_action[t + k]
```

Rotations use SE3 matrix multiplication, not component-wise subtraction. Gripper dimensions stay absolute, matching training.

For absolute checkpoints, ground truth is direct HDF5 action:

```text
gt_absolute[t, k] = /action[t + k]
```

There is a legacy/debug escape hatch, `--allow-raw-fallback`, but it should not be used for strict checkpoint evaluation.

## Output

Default output is outside this project directory:

```text
../offline_inference_output/<checkpoint_name>__<data_dir_name>
```

For example, `CHECKPOINT=ckpt/blackboard_pi0_45000` and
`EVAL_DATA_DIR=data/clean_replayraw` writes to:

```text
../offline_inference_output/blackboard_pi0_45000__clean_replayraw
```

Set `OUTPUT_DIR=/path/to/result` or pass `--output-dir` only when you need a
manual override. The `offline-inference-easy-mirro/` directory is intended to
contain only portable code, wrappers, docs, and tools.

Per episode:

```text
collection_xxx/
  trajectory_pairs.pkl
  metadata.pkl
  metadata.json
```

Summary:

```text
summary.txt
metrics.json
```

The `trajectory_pairs.pkl` entries include:

```python
{
    "timestep": int,
    "observation_state": np.ndarray,         # (20,), absolute frame-0 qpos[t]
    "ground_truth_action_chunk": np.ndarray, # (chunk_size, 20), absolute /action values for debugging
    "ground_truth_delta_chunk": np.ndarray,  # (chunk_size, 20), chunk-wise SE3 delta
    "valid_length": int,
}
```

Delta outputs additionally contain:

```python
"predicted_delta_chunk": np.ndarray       # (chunk_size, 20), model delta after unnormalize
"ground_truth_chunk": np.ndarray          # absolute /action chunk
"predicted_chunk": np.ndarray             # T_qpos[t] @ T_pred_delta[t+k]
"predicted_action_chunk": np.ndarray      # explicit absolute-action alias
```

Absolute outputs additionally contain:

```python
"ground_truth_chunk": np.ndarray          # alias of absolute /action chunk for visualization
"predicted_chunk": np.ndarray             # model absolute action after unnormalize
"predicted_action_chunk": np.ndarray      # explicit absolute-action alias
```

Use `summary.txt` for a quick read. The primary aggregate is:

```text
Average valid-chunk delta MSE/MAE       # delta checkpoints
Average valid-chunk absolute MSE/MAE    # absolute checkpoints
```

For delta checkpoints, the evaluator also writes a secondary absolute
comparison:

```text
Average valid-chunk absolute-from-delta MSE/MAE
```

This is computed by composing each predicted SE3 delta with the input absolute
state:

```text
T_abs_pred[t+k] = T_qpos[t] @ T_delta_pred[t+k]
```

## Visualization

Interactive dual-arm 3D trajectory viewer:

```bash
python offline-inference-easy-mirro/visualize_episode.py \
  --data-dir ../offline_inference_output/blackboard_pi0_45000__clean_replayraw
```

Interactive 10D/20D waveform viewer:

```bash
python offline-inference-easy-mirro/visualize_waveform.py \
  --data-dir ../offline_inference_output/blackboard_pi0_45000__clean_replayraw
```

Both interactive viewers support:

```bash
--episode collection_xxx
--space auto       # default, follows output action space
--space delta
--space absolute
--position-scale 10     # default: show xyz in decimeters
--position-unit dm
--play-interval-ms 40    # playback timer interval
--no-images        # disable HDF5 frame panels
```

Both viewers show the current HDF5 camera frames used by the model. The camera
list comes from each episode's metadata, so unused cameras are not displayed.
The displayed frame is aligned to the current pair `timestep`.

Controls:

```text
SPACE/RIGHT: next timestep
LEFT: previous timestep
P or Play button: play/pause continuous playback
UP/DOWN: switch episode
Q: quit
```

Static HTML preview:

Generate dual-arm trajectory and 20D waveform PNGs plus an `index.html`. The preview renders whichever action spaces are present:

```bash
python offline-inference-easy-mirro/render_preview.py \
  --data-dir ../offline_inference_output/fz_rawdata/checkpoint-45000
```

Static previews render every action space present in `trajectory_pairs.pkl`.
For delta models this now includes both `delta` and reconstructed `absolute`
plots. XYZ coordinates are displayed in decimeters by default, while rotation
and gripper channels remain in their raw units. Trajectory preview images also
include the model-input HDF5 camera frames for the displayed snapshot timesteps.

## Alignment Check

Verify that offline GT deltas match the original training delta code:

```bash
python offline-inference-easy-mirro/validate_alignment.py \
  --data-dir /path/to/hdf5_eval_dir \
  --output-dir ../offline_inference_output/blackboard_pi0_45000__clean_replayraw
```

This checks:

- offline GT delta equals `easy-mirro-dual-new_1/data_utils/dataset.py::_compute_action_delta`
- offline GT delta equals `get_norm_stats_by_task.py::compute_chunk_delta`
- saved absolute GT chunks equal direct HDF5 `/action[t:t+chunk_size]`, when present
