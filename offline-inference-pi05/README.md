# LeRobot pi0.5 Offline Inference

完整设计和实现逻辑说明见：

```text
offline-inference-pi05/OFFLINE_INFERENCE_PI05_DETAILS.md
```

这个目录是新的 pi0.5 离线推理项目，面向 LeRobot v3 dataset。当前默认适配黑板擦除 pi0.5 checkpoint 和测试数据：

```text
checkpoint: ckpt/inference_pi05_45000/pretrained_model
tokenizer:   ckpt/inference_pi05_45000/paligemma-3b-pt-224-tokenizer
data:        data/lerobot_blackboard_testdata
```

它不会改动旧的 `offline-inference/eval_offline.py`。主脚本会走 LeRobot 的：

```text
policy preprocessor -> PI05Policy.predict_action_chunk -> policy postprocessor
```

所以 `se3_pose` 相对 action、`MIXED_QUANTILES` normalization、tokenizer、绝对 action 还原都和 pi0.5 训练链路一致。

当前黑板 checkpoint 的核心契约是：

```text
输入 state:  observation.state[t]，10D absolute pose
模型输出:    chunk-wise SE(3) relative action
主指标:      predicted_delta_chunk vs ground_truth_delta_chunk
absolute:    predicted_chunk vs ground_truth_chunk
可视化:      同时保存 delta 空间和 absolute 重建空间
```

## 快速运行

黑板测试数据直接运行：

```bash
bash offline-inference-pi05/run_eval_blackboard.sh
```

只检查数据 episode 范围：

```bash
bash offline-inference-pi05/run_eval_blackboard.sh --list-episodes
```

小规模调试：

```bash
bash offline-inference-pi05/run_eval_blackboard.sh \
  --episodes 0 \
  --limit-frames 8 \
  --batch-size 2
```

如果要换 checkpoint：

```bash
CHECKPOINT=/path/to/pretrained_model \
bash offline-inference-pi05/run_eval_blackboard.sh
```

如果直接传训练输出根目录也可以，主脚本会自动查找：

```text
<output>/checkpoints/last/pretrained_model
```

## 输出

默认输出到：

```text
offline_inference_output/inference_pi05_45000__lerobot_blackboard_testdata
```

每个 episode 会生成：

```text
episode_000000/
  trajectory_pairs.pkl
  metadata.pkl
  metadata.json
```

总览文件：

```text
summary.txt
metrics.json
```

`trajectory_pairs.pkl` 结构沿用旧离线推理脚本：

```python
{
    "pairs": [
        {
            "timestep": int,
            "observation_state": np.ndarray,           # absolute state[t], (10,)
            "ground_truth_delta_chunk": np.ndarray,    # GT relative action, (chunk_size, 10)
            "predicted_delta_chunk": np.ndarray,       # model relative output, (chunk_size, 10)
            "ground_truth_chunk": np.ndarray,          # GT absolute action, (chunk_size, 10)
            "predicted_chunk": np.ndarray,             # predicted absolute action, (chunk_size, 10)
            "ground_truth_action_chunk": np.ndarray,   # absolute alias
            "predicted_action_chunk": np.ndarray,      # absolute alias
            "valid_length": int,
        },
    ],
    "chunk_size": int,
    "state_dim": 10,
    "action_dim": 10,
    "episode_len": int,
    "action_mode": "delta",
    "state_mode": "absolute",
    "prediction_space": "chunkwise_se3_delta_10d",
}
```

动作维度是单手 UMI pose：

```text
xyz(3) + rot6d_first_two_rows(6) + gripper(1)
```

如果要保存图像用于可视化，加 `--save-images`。注意 LeRobot 视频帧展开成 `images.pkl` 会很大。

可视化现在不强制依赖 `images.pkl`。如果输出里有 `source_dataset_root` 和 `dataset_from_index`，可视化脚本会直接回到原始 LeRobot dataset 视频里按当前 timestep 解码图像帧，显示在轨迹图或波形图右侧。

查看单手 10D 轨迹，入口名字和旧项目保持一致：

```bash
python offline-inference-pi05/visualize_episode.py \
  --data_dir offline_inference_output/inference_pi05_45000__lerobot_blackboard_testdata \
  --space delta
```

查看 10 个动作维度的波形叠加图：

```bash
python offline-inference-pi05/visualize_waveform.py \
  --data_dir offline_inference_output/inference_pi05_45000__lerobot_blackboard_testdata \
  --space delta
```

这两个交互可视化都支持：

```text
Left/Right 或 Space: 切换 timestep
Up/Down: 切换 episode
--space delta: 看模型真实 relative action 输出空间
--space absolute: 看 absolute action 空间；delta 模型会显示 relative action 复合回 state 后的轨迹
```

如果远程桌面没有弹出交互窗口，可以先生成静态 PNG 预览：

```bash
python offline-inference-pi05/render_preview.py \
  --data_dir offline_inference_output/inference_pi05_45000__lerobot_blackboard_testdata \
  --space all
```

生成结果在输出目录的 `preview/index.html`。`--space all` 会同时渲染：

```text
episode_xxxxxx_delta_trajectory.png
episode_xxxxxx_delta_waveform.png
episode_xxxxxx_absolute_trajectory.png
episode_xxxxxx_absolute_waveform.png
```

## 关键参数

- `--checkpoint`: pi0.5 checkpoint、某个 step checkpoint，或训练输出根目录。
- `--dataset-root`: LeRobot dataset 根目录，默认 `data/lerobot_blackboard_testdata`。
- `--processor-source auto|checkpoint|dataset-config`: 默认 `auto`。有兼容的 checkpoint processor 就用 checkpoint，否则按当前 pi0.5 UMI 配置和 dataset stats 重建。
- `--use-relative-actions/--no-use-relative-actions`: 默认读取 checkpoint 配置；黑板脚本显式开启。
- `--relative-action-mode se3_pose`: 黑板 checkpoint 使用 SE(3) pose relative action。
- `--pose-arm-offsets '[0]' --pose-arm-stride 10`: 单手 10D pose block。
- `--metric-drop-last-frames`: 默认读取 checkpoint 的 `drop_n_last_frames`，黑板 checkpoint 是 `49`。
