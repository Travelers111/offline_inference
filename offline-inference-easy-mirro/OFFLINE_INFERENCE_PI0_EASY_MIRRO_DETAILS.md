# Pi0 EasyMirror Offline Inference Details

## Contract

The evaluator matches `easy-mirro-dual-new_1` Pi0 training:

- `observation.state`: 20D frame-0 absolute dual-arm pose, padded to 32D.
- model target/output: 20D chunk-wise SE3 delta, padded internally to 32D.
- chunk size: loaded from checkpoint config, expected `50`.
- normalization: `NormalizeMixed` with `dataset_stats.pkl` from the checkpoint directory or its parent.

## Action Label

For a frame `t`, the ground-truth action chunk is first built in absolute frame-0 coordinates:

```text
absolute_chunk = action[t : t + chunk_size]
```

Then it is converted exactly like training:

```text
T_delta[k] = inv(T_qpos[t]) @ T_action[t + k]
delta_xyz[k] = T_delta[k][:3, 3]
delta_rot6d[k] = concat(T_delta[k][0, :3], T_delta[k][1, :3])
delta_gripper[k] = action_gripper[t + k]
```

The comparison that matters is:

```text
predicted_delta_chunk <-> ground_truth_delta_chunk
```

No absolute-space reconstruction is used for scoring in the strict evaluator.

## Canonical Data Requirement

Strict evaluation requires the same processed HDF5 layout as training:

```text
/observations/qpos   (N, 20)
/action              (N, 20)
```

`rawdata` has this layout. The `test2` raw collection files do not, so they are
rejected by default instead of being silently converted with a different path.

`--allow-raw-fallback` exists only as a legacy/debug path and should not be used
for strict checkpoint evaluation.

## Validation

Run:

```bash
python3 offline-inference-easy-mirro/validate_alignment.py \
  --data-dir rawdata \
  --chunk-size 50 \
  --require-training-layout
```

Expected tolerances are around `1e-7` for:

- evaluator delta vs training dataset delta
- evaluator delta vs norm-stats delta
- delta-to-absolute roundtrip
- saved ground-truth delta vs recomputed delta, when `--output-dir` is provided

## Full Fangzhou Evaluation

```bash
bash offline-inference-easy-mirro/run_eval_fz_test2_all.sh
```

The script name is kept for compatibility, but its default data directory is now
`rawdata`. Outputs:

```text
offline_inference_output/fz_rawdata/checkpoint-35000/
offline_inference_output/fz_rawdata/checkpoint-45000/
offline_inference_output/fz_rawdata/comparison.json
```

Open:

```text
offline_inference_output/fz_rawdata/checkpoint-45000/preview/index.html
```

The preview renders every saved action space. Delta checkpoints now include both delta and reconstructed absolute plots, with xyz displayed in decimeters by default.
