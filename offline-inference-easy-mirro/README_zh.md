# easy-mirro Pi0 离线推理

这是用于 `easy-mirro-dual-new_1` Pi0 EasyMirror checkpoint 的 HDF5 离线推理/评估入口。

完整项目说明见 [`PROJECT_GUIDE_zh.md`](PROJECT_GUIDE_zh.md)。

它遵循 `easy-mirro-dual-new_1/README_PI0_CONFIG.md` 中定义的模型契约：

- 输入 state 是 20 维绝对 frame-0 位姿，填充到 32 维
- 模型输出是按 chunk 表示的 20 维 SE3 delta action
- 主要指标比较 `predicted_delta_chunk` 和 `ground_truth_delta_chunk`
- 归一化统计只从 checkpoint 旁边或其父目录中的 `dataset_stats.pkl` 加载

## 运行

```bash
export CHECKPOINT=/home/eai/debug/ckpt__fz/checkpoint-45000
export EVAL_DATA_DIR=/home/eai/debug/rawdata
export BATCH_SIZE=8

bash offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

列出可用的 HDF5 episode：

```bash
bash offline-inference-easy-mirro/run_eval_easy_mirro.sh --list-episodes
```

也可以使用训练输出根目录，而不是某个具体 checkpoint：

```bash
export CHECKPOINT=/home/eai/debug/ckpt__fz
```

脚本会选择最新的 `checkpoint-*` 目录，并从该 checkpoint 或其父目录加载 `dataset_stats.pkl`。

在标准 `rawdata` 上评估两个 Fangzhou checkpoint，验证对齐，渲染预览，并写出 checkpoint 对比：

```bash
bash offline-inference-easy-mirro/run_eval_fz_test2_all.sh
```

all-run 的默认输出：

```text
offline_inference_output/fz_rawdata/
  checkpoint-35000/
  checkpoint-45000/
  comparison.json
```

## 必需的 HDF5 布局

标准训练布局的 HDF5 文件必须包含 `/observations/qpos` 和 `/action`。

评估器现在默认拒绝原始采集文件。这是有意设计的：`test2` 这类原始文件不包含 `/observations/qpos` 和 `/action`，如果在 eval 内部转换，可能会静默偏离训练时使用的预处理流程。

```text
/observations/qpos  (N, 20)  absolute frame-0 state
/action             (N, 20)  absolute future action; action[t] is the first future target
```

对于每个 timestep，ground-truth delta 会按训练时完全相同的方式计算：

```text
gt_delta[t, k] = inv(T_qpos[t]) @ T_action[t + k]
```

旋转使用 SE3 矩阵乘法，而不是逐分量相减。夹爪维度保持绝对值，与训练一致。

存在一个旧版/调试用的逃生开关 `--allow-raw-fallback`，但不应在严格 checkpoint 评估中使用。

## 输出

默认输出在本项目目录外：

```text
../offline_inference_output/<checkpoint名字>__<数据目录名字>
```

例如 `CHECKPOINT=ckpt/blackboard_pi0_45000`、`EVAL_DATA_DIR=data/clean_replayraw` 会写到：

```text
../offline_inference_output/blackboard_pi0_45000__clean_replayraw
```

只有需要手动覆盖时才设置 `OUTPUT_DIR=/path/to/result` 或传 `--output-dir`。`offline-inference-easy-mirro/` 目录只保留可移植的代码、脚本、文档和工具。

每个 episode：

```text
collection_xxx/
  trajectory_pairs.pkl
  metadata.pkl
  metadata.json
```

汇总：

```text
summary.txt
metrics.json
```

`trajectory_pairs.pkl` 条目包括：

```python
{
    "timestep": int,
    "observation_state": np.ndarray,         # (20,), absolute frame-0 qpos[t]
    "ground_truth_action_chunk": np.ndarray, # (chunk_size, 20), absolute /action values for debugging
    "ground_truth_delta_chunk": np.ndarray,  # (chunk_size, 20), chunk-wise SE3 delta
    "predicted_delta_chunk": np.ndarray,     # (chunk_size, 20), raw model delta after unnormalize
    "ground_truth_chunk": np.ndarray,        # absolute /action chunk
    "predicted_chunk": np.ndarray,           # T_qpos[t] @ T_pred_delta[t+k]
    "valid_length": int,
}
```

可使用 `summary.txt` 快速查看结果。主要聚合指标是：

```text
Average valid-chunk delta MSE/MAE
Average valid-chunk absolute-from-delta MSE/MAE
```

`absolute-from-delta` 是把模型输出的 SE3 delta 接回当前绝对 state 得到的绝对 action：

```text
T_abs_pred[t+k] = T_qpos[t] @ T_delta_pred[t+k]
```

## 可视化

交互式双臂 3D 轨迹查看器：

```bash
python offline-inference-easy-mirro/visualize_episode.py \
  --data-dir ../offline_inference_output/blackboard_pi0_45000__clean_replayraw
```

交互式 20 维波形查看器：

```bash
python offline-inference-easy-mirro/visualize_waveform.py \
  --data-dir ../offline_inference_output/blackboard_pi0_45000__clean_replayraw
```

两个交互式查看器都支持：

```bash
--episode collection_xxx
--space delta
--space absolute   # delta 模型也会保存由 delta 重建出的 absolute action
--position-scale 10
--position-unit dm
```

控制：

```text
SPACE/RIGHT: next timestep
LEFT: previous timestep
UP/DOWN: switch episode
Q: quit
```

静态 HTML 预览：

生成轨迹和 10D/20D 波形 PNG，以及一个 `index.html`。预览会渲染输出里存在的所有空间；delta 模型现在会同时渲染 `delta` 和重建后的 `absolute`。xyz 默认按 dm 显示，旋转和夹爪保持原始单位：

```bash
python offline-inference-easy-mirro/render_preview.py \
  --data-dir ../offline_inference_output/fz_rawdata/checkpoint-45000
```

## 对齐检查

验证离线 GT delta 是否与原始训练 delta 代码一致：

```bash
python offline-inference-easy-mirro/validate_alignment.py \
  --data-dir /path/to/hdf5_eval_dir \
  --output-dir ../offline_inference_output/blackboard_pi0_45000__clean_replayraw
```

该检查会验证：

- offline GT delta 等于 `easy-mirro-dual-new_1/data_utils/dataset.py::_compute_action_delta`
- offline GT delta 等于 `get_norm_stats_by_task.py::compute_chunk_delta`
