# LeRobot pi0.5 Offline Inference

完整设计和实现逻辑说明见：

```text
offline_inference/offline-inference-pi05/OFFLINE_INFERENCE_PI05_DETAILS.md
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
bash offline_inference/offline-inference-pi05/run_eval_blackboard.sh
```

只检查数据 episode 范围：

```bash
bash offline_inference/offline-inference-pi05/run_eval_blackboard.sh --list-episodes
```

小规模调试：

```bash
bash offline_inference/offline-inference-pi05/run_eval_blackboard.sh \
  --episodes 0 \
  --limit-frames 8 \
  --batch-size 2
```

如果要换 checkpoint：

```bash
CHECKPOINT=/path/to/pretrained_model \
bash offline_inference/offline-inference-pi05/run_eval_blackboard.sh
```

如果直接传训练输出根目录也可以，主脚本会自动查找：

```text
<output>/checkpoints/last/pretrained_model
```

## 单相机 / 双相机 checkpoint

`run_eval_blackboard.sh` 默认使用：

```bash
--cameras checkpoint
```

也就是从 checkpoint 的 `config.json -> input_features` 读取模型实际需要的视觉输入。这样同一个离线推理脚本可以兼容两种 pi0.5：

```text
双相机 checkpoint: observation.images.cam_hand + observation.images.cam_top
单相机 checkpoint: observation.images.cam_hand
```

如果数据集里仍然保存了双相机视频，但 checkpoint 是单相机训练出来的，脚本只会把 `cam_hand` 喂给模型，不会把未训练过的 `cam_top` 加进 policy input features。

跑单相机 checkpoint 示例：

```bash
CHECKPOINT=/home/eai/PI/ckpt/pi05_ckpt/pi05_singlecam_55000/pretrained_model \
bash offline_inference/offline-inference-pi05/run_eval_blackboard.sh
```

如果要强制指定相机，也可以：

```bash
CAMERAS=cam_hand \
CHECKPOINT=/home/eai/PI/ckpt/pi05_ckpt/pi05_singlecam_55000/pretrained_model \
bash offline_inference/offline-inference-pi05/run_eval_blackboard.sh
```

如果你用 `prepare_pi05_umi_blackboard_cam_hand_dataset.sh` 生成了只包含 `cam_hand` 的 LeRobot 数据集，同时指定对应的数据根目录和 repo id：

```bash
DATASET_ROOT=/data/share/aloha_gr/human_data/lerobot_dataset/Lerobot_Blackboard_smoothed_cam_hand \
DATASET_REPO_ID=local/Lerobot_Blackboard_smoothed_cam_hand \
CHECKPOINT=/home/eai/PI/ckpt/pi05_ckpt/pi05_singlecam_55000/pretrained_model \
bash offline_inference/offline-inference-pi05/run_eval_blackboard.sh
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
python offline_inference/offline-inference-pi05/visualize_episode.py \
  --data_dir offline_inference_output/inference_pi05_45000__lerobot_blackboard_testdata \
  --space delta
```

查看 10 个动作维度的波形叠加图：

```bash
python offline_inference/offline-inference-pi05/visualize_waveform.py \
  --data_dir offline_inference_output/inference_pi05_45000__lerobot_blackboard_testdata \
  --space delta
```

这两个交互可视化都支持：

```text
长按 Right / Space: 连续向前播放 timestep，图像帧和动作同步刷新
长按 Left: 连续回退 timestep，松开就停止
P 或 Play 按钮: 自动播放 / 暂停
Up/Down: 切换 episode
--space delta: 看模型真实 relative action 输出空间
--space absolute: 看 absolute action 空间；delta 模型会显示 relative action 复合回 state 后的轨迹
```

`visualize_episode.py` 的 3D 轨迹视角不会在切换 timestep 时自动归位。你用鼠标旋转到任意角度后，可以继续长按键盘查看后续帧，脚本会保留当前 `elev/azim/roll` 视角。

播放速度可以用 `--play-interval-ms` 调整，默认 40ms。

如果远程桌面没有弹出交互窗口，可以先生成静态 PNG 预览：

```bash
python offline_inference/offline-inference-pi05/render_preview.py \
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
- `--cameras checkpoint|all|cam_hand,...`: 默认 `checkpoint`，按 checkpoint 的视觉输入自动选择相机；单相机 checkpoint 会只取 `observation.images.cam_hand`。
- `--processor-source auto|checkpoint|dataset-config`: 默认 `auto`。有兼容的 checkpoint processor 就用 checkpoint，否则按当前 pi0.5 UMI 配置和 dataset stats 重建。
- `--use-relative-actions/--no-use-relative-actions`: 默认读取 checkpoint 配置；黑板脚本显式开启。
- `--relative-action-mode se3_pose`: 黑板 checkpoint 使用 SE(3) pose relative action。
- `--pose-arm-offsets '[0]' --pose-arm-stride 10`: 单手 10D pose block。
- `--metric-drop-last-frames`: 默认读取 checkpoint 的 `drop_n_last_frames`，黑板 checkpoint 是 `49`。
