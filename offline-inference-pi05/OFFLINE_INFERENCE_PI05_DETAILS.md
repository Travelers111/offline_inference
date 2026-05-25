# offline-inference-pi05 详细说明

本文档说明 `offline-inference-pi05` 这个离线推理项目的完整逻辑。它的目标是：在不连接真实机器人、不启动在线控制系统的情况下，用 LeRobot 版本的 pi0.5 policy checkpoint 对本地 LeRobot v3 数据集逐帧推理，并把模型预测的未来 action chunk 与数据集里的 ground truth action chunk 对齐保存，方便后续评估和可视化。

当前默认适配黑板 pi0.5 checkpoint 和 LeRobot 黑板测试数据：

```text
模型: ckpt/inference_pi05_45000/pretrained_model
tokenizer: ckpt/inference_pi05_45000/paligemma-3b-pt-224-tokenizer
数据: data/lerobot_blackboard_testdata
输出: offline_inference_output/inference_pi05_45000__lerobot_blackboard_testdata
```

## 0. 当前黑板版本的关键更新

2026-05-21 后，`offline-inference-pi05` 的主逻辑已经按 `offline-inference-easy-mirro` 的 delta 评估思路调整：

```text
输入 state:
  observation.state[t]
  10D absolute pose

模型输出:
  predicted_delta_chunk
  10D chunk-wise SE(3) relative action

主评估空间:
  predicted_delta_chunk vs ground_truth_delta_chunk

辅助可视化空间:
  predicted_chunk / predicted_action_chunk
  由 predicted_delta_chunk 和 observation.state[t] 复合出来的 absolute action
```

也就是说，这个版本不是只把 postprocessor 还原后的 absolute action 和数据集 action 比较，而是和 Pi0 Easy-MIRRO 的 `delta` 模式一致：

```text
ground_truth_delta_chunk[t+k] = inv(T_state[t]) @ T_action[t+k]
predicted_delta_chunk         = model output after unnormalizer
predicted_chunk               = T_state[t] @ T_predicted_delta_chunk
```

黑板数据启动脚本是：

```text
offline-inference-pi05/run_eval_blackboard.sh
```

## 1. 整体逻辑

一句话概括：

```text
读取 LeRobot dataset 的每一帧 observation
  -> 用 pi0.5 preprocessor 构造模型输入
  -> 调用 PI05Policy.predict_action_chunk 预测未来 50 步
  -> 用 unnormalizer 得到模型的 relative SE(3) action chunk
  -> 用完整 postprocessor 额外还原 absolute action chunk 用于可视化
  -> 和数据集里构造出的 relative / absolute ground truth chunk 对齐保存
  -> 计算误差指标并生成可视化文件
```

主入口是：

```text
offline-inference-pi05/eval_lerobot_pi05.py
```

推荐启动脚本是：

```text
offline-inference-pi05/run_eval_blackboard.sh
```

推理输出默认保存到：

```text
offline_inference_output/inference_pi05_45000__lerobot_blackboard_testdata
```

### 1.1 为什么不能直接照搬旧 pi0 离线推理

旧项目 `offline-inference/eval_offline.py` 是给之前的 pi0/MIRRO 风格模型和 HDF5/pkl 数据流写的，核心逻辑是手动做：

```text
load hdf5/pkl
  -> 手写 image/state batch
  -> 手写 normalize/unnormalize
  -> 直接调模型内部 sample_actions
```

pi0.5 这里不能这样做，因为本项目训练链路里有几个关键语义必须和 LeRobot processor 保持一致：

- action 训练目标不是简单的 absolute action。
- action 会先按 `se3_pose` 转成相对当前 state 的 SE(3) pose action。
- state/action 使用 `MIXED_QUANTILES`，其中 xyz/gripper 做 quantile normalization，rot6d 保持原样。
- policy postprocessor 需要拿到 preprocessor 缓存的当前 state，才能把相对 pose action 还原成 absolute pose action。
- pi0.5 prompt 里包含 normalized state 离散化后的 token，必须走 `Pi05PrepareStateTokenizerProcessorStep` 和 tokenizer。

所以新脚本坚持走 LeRobot 官方 policy pipeline，并在输出阶段拆成两个视角：

```text
policy preprocessor
  -> PI05Policy.predict_action_chunk
  -> postprocessor[:1] 得到 relative action
  -> full policy postprocessor 得到 absolute-from-delta action
```

这样离线推理看到的输入输出语义与训练时一致。

## 2. 项目文件结构

```text
offline-inference-pi05/
  README.md
  OFFLINE_INFERENCE_PI05_DETAILS.md
  eval_lerobot_pi05.py
  run_eval_blackboard.sh
  run_eval_clean_test.sh
  visualize_episode.py
  visualize_lerobot_pi05_episode.py
  visualize_waveform.py
  render_preview.py
```

各文件职责：

```text
eval_lerobot_pi05.py
```

主推理脚本。负责加载 dataset、加载 policy、构造 pre/post processor、逐帧推理、保存 `trajectory_pairs.pkl`、写 summary 和 metrics。

```text
run_eval_blackboard.sh
```

黑板 pi0.5 checkpoint 的便捷启动脚本。默认指向 `ckpt/inference_pi05_45000/pretrained_model`、`data/lerobot_blackboard_testdata` 和本地 tokenizer，并显式启用 `se3_pose` relative action。

```text
run_eval_clean_test.sh
```

旧 clean test 数据的便捷启动脚本。保留它是为了兼容之前 `data/clean_test_lerobot_pi05` 的实验。

```text
visualize_episode.py
visualize_lerobot_pi05_episode.py
```

交互式 3D 轨迹可视化。`visualize_episode.py` 是为了和旧项目入口名字保持一致的 wrapper；真实实现放在 `visualize_lerobot_pi05_episode.py`。

```text
visualize_waveform.py
```

交互式波形可视化。显示 10 个动作维度的 full-episode GT 背景曲线，以及当前 timestep 的 GT chunk / Pred chunk 叠加。

```text
render_preview.py
```

静态 PNG/HTML 预览生成脚本。用于远程桌面没有弹窗、或者只想快速看结果的情况。

## 3. 数据输入逻辑

### 3.1 LeRobot dataset 格式

目标数据目录是标准 LeRobot v3 dataset。当前黑板默认数据是：

```text
data/lerobot_blackboard_testdata/
  data/chunk-000/file-000.parquet
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
  videos/observation.images.cam_hand/chunk-000/file-000.mp4
  videos/observation.images.cam_top/chunk-000/file-000.mp4
```

主脚本用 LeRobot 的 `LeRobotDataset` 读取：

```python
dataset = LeRobotDataset(
    repo_id,
    root=dataset_root,
    video_backend="pyav",
    download_videos=False,
    return_uint8=False,
)
```

`return_uint8=False` 表示图像会以 `float32`、范围 `[0, 1]` 的 tensor 形式返回，正好符合 LeRobot policy 图像预处理的预期。

### 3.2 当前数据的 feature

`data/lerobot_blackboard_testdata` 的核心 feature 是：

```text
observation.images.cam_hand
observation.images.cam_top
observation.state
action
language_instruction / task_index / timestamp / episode_index / frame_index / index
```

其中 state/action 都是 10 维：

```text
0: x
1: y
2: z
3: rot6d_row0_x
4: rot6d_row0_y
5: rot6d_row0_z
6: rot6d_row1_x
7: rot6d_row1_y
8: rot6d_row1_z
9: gripper
```

也就是：

```text
xyz(3) + rot6d_first_two_rows(6) + gripper(1)
```

### 3.3 episode 范围

LeRobot dataset 的 episode 边界来自 `dataset.meta.episodes`。脚本用：

```python
episode_rows(dataset, episode_index)
```

读取：

```text
dataset_from_index
dataset_to_index
length
```

推理时按 episode 单独处理，避免跨 episode 取 future chunk。

可以用下面命令只查看 episode 范围：

```bash
bash offline-inference-pi05/run_eval_blackboard.sh --list-episodes
```

当前黑板测试数据是 8 个 episode：

```text
episode_000000: rows=[0, 986), length=986
episode_000001: rows=[986, 1821), length=835
episode_000002: rows=[1821, 2520), length=699
episode_000003: rows=[2520, 3247), length=727
episode_000004: rows=[3247, 4100), length=853
episode_000005: rows=[4100, 4779), length=679
episode_000006: rows=[4779, 5284), length=505
episode_000007: rows=[5284, 5833), length=549
```

## 4. 模型加载逻辑

### 4.1 checkpoint 路径解析

`--checkpoint` 支持几种写法：

```text
1. 直接指向 pretrained_model 目录
2. 指向某个 checkpoint 目录，脚本会找 checkpoint/pretrained_model
3. 指向训练 output 根目录，脚本会找 output/checkpoints/last/pretrained_model
4. 指向 lerobot/models/pi05_base
```

实现函数是：

```python
resolve_model_dir(path)
```

它会依次检查这些候选路径：

```text
base
base/pretrained_model
base/checkpoints/last/pretrained_model
base/last/pretrained_model
```

只要目录里有：

```text
config.json
model.safetensors
```

就认为是可加载的 policy checkpoint。

### 4.2 重建 pi0.5 config

脚本先从 checkpoint 读取原始 policy config：

```python
cfg = PreTrainedConfig.from_pretrained(model_dir)
```

然后会覆盖一些运行时配置：

```text
cfg.pretrained_path = model_dir
cfg.device = cuda/cpu
cfg.dtype = bfloat16/float32
cfg.tokenizer_name = local tokenizer path
cfg.use_relative_actions = True
cfg.relative_action_mode = se3_pose
cfg.pose_arm_offsets = [0]
cfg.pose_arm_stride = 10
cfg.normalization_mapping = {
    VISUAL: IDENTITY,
    STATE: MIXED_QUANTILES,
    ACTION: MIXED_QUANTILES,
}
```

这里最关键的是：

```text
use_relative_actions=true
relative_action_mode=se3_pose
pose_arm_offsets=[0]
pose_arm_stride=10
normalization_mapping=...MIXED_QUANTILES...
```

这些配置必须和 pi0.5 UMI 训练时一致。

### 4.3 为什么要从 dataset 重新推断 input/output features

`pi05_base` 原始 config 里的图像 key 是通用预训练名字，例如：

```text
observation.images.base_0_rgb
observation.images.left_wrist_0_rgb
observation.images.right_wrist_0_rgb
```

但是当前 UMI 数据集使用：

```text
observation.images.cam_hand
observation.images.cam_top
```

所以脚本会用：

```python
dataset_to_policy_features(dataset.meta.features)
```

从目标 dataset 重新推断 state/action features：

```text
cfg.input_features
cfg.output_features
```

视觉输入不会无条件使用 dataset 的全部相机，而是先解析 `--cameras`。
当前默认是：

```bash
--cameras checkpoint
```

脚本会从 checkpoint 的 `config.json -> input_features` 读取模型实际训练时的视觉输入：

```text
双相机 pi0.5: observation.images.cam_hand + observation.images.cam_top
单相机 pi0.5: observation.images.cam_hand
```

然后只把这些相机写回 `cfg.input_features`。这样一个单相机 checkpoint 即使评估在仍然保存了 `cam_top` 的双相机数据集上，也只会消费 `cam_hand`，不会把未训练过的第二路相机加入模型输入。需要强制改相机时可以传：

```bash
--cameras cam_hand
--cameras all
```

这样 policy 的视觉输入 key、state shape 和 action shape 都与当前 checkpoint 和数据一致。

## 5. processor 构建逻辑

### 5.1 processor 的作用

主推理链路是：

```text
raw batch
  -> preprocessor
  -> policy.predict_action_chunk
  -> postprocessor
  -> absolute action chunk
```

preprocessor 负责：

```text
1. 增加 batch 维度 / 整理 batch
2. 缓存当前 observation.state
3. 归一化 state
4. 构造 pi0.5 文本 prompt
5. tokenizer 生成 language tokens / attention mask
6. 把 tensor 移到 device
```

postprocessor 负责：

```text
1. unnormalize action
2. 如果启用 relative action，把相对 action 还原成 absolute action
3. action 移回 CPU
```

### 5.2 processor-source 的三种模式

参数：

```bash
--processor-source auto|checkpoint|dataset-config
```

含义：

```text
auto
```

默认模式。优先判断 checkpoint 里有没有兼容的 processor。如果 checkpoint processor 里已经包含：

```text
relative_pose_actions_processor
absolute_pose_actions_processor
```

就使用 checkpoint 自带 processor；否则按当前 config 和 dataset stats 重建。

```text
checkpoint
```

强制使用 checkpoint 保存的 processor。如果 checkpoint processor 不兼容，会报错。

```text
dataset-config
```

强制按当前 pi0.5 UMI 配置和 dataset stats 重建 processor。

当前黑板 checkpoint 跑时，processor source 是：

```text
checkpoint
```

原因是 `ckpt/inference_pi05_45000/pretrained_model` 已经保存了训练时的 `policy_preprocessor.json`、`policy_postprocessor.json` 和 normalizer/unnormalizer state。离线推理应优先使用这些 checkpoint processor，避免用测试集 stats 重新构造 normalization。

### 5.3 preprocessor 实际步骤

当 `processor-source=dataset-config` 时，`make_pi05_pre_post_processors` 会构造类似下面的 pipeline：

```text
preprocessor:
  RenameObservationsProcessorStep
  AddBatchDimensionProcessorStep
  RelativePoseActionsProcessorStep
  NormalizerProcessorStep
  Pi05PrepareStateTokenizerProcessorStep
  TokenizerProcessorStep
  DeviceProcessorStep
```

离线推理时输入 batch 没有 action，`RelativePoseActionsProcessorStep` 不会改 action，但它会缓存当前 `observation.state`。这个缓存很重要，因为 postprocessor 要靠它把模型输出的相对 action 转回 absolute action。

### 5.4 postprocessor 实际步骤

postprocessor 类似：

```text
postprocessor:
  UnnormalizerProcessorStep
  AbsolutePoseActionsProcessorStep
  DeviceProcessorStep(cpu)
```

流程是：

```text
模型输出 normalized relative action
  -> unnormalize 到 raw relative action 空间
  -> 使用 preprocessor 缓存的当前 state 做 SE(3) 还原
  -> 得到 episode-local absolute pose action
```

最终保存的 `predicted_chunk` 是 absolute pose action，可以直接和 dataset 里的 `action` 比较。

## 6. SE(3) relative action 逻辑

### 6.1 数据保存语义

当前 LeRobot 数据集里的语义是：

```text
observation.state[t] = episode-local absolute qpos[t]
action[t]            = episode-local absolute qpos[t + 1]
```

也就是说 parquet 里保存的是 absolute pose，不是 relative pose。

### 6.2 训练和推理时的模型语义

训练 pi0.5 时启用了：

```text
use_relative_actions=true
relative_action_mode=se3_pose
pose_arm_offsets=[0]
pose_arm_stride=10
```

所以模型实际学习的是：

```text
relative_action[k] = inverse(T_state[t]) @ T_action[t+k]
relative_gripper[k] = absolute gripper[t+k]
```

其中：

```text
T_state[t]
```

由当前 9D pose：

```text
xyz + rot6d
```

转成 SE(3) 矩阵。

旋转不能用简单减法处理，必须通过矩阵乘法：

```text
relative transform = inverse(current transform) @ future transform
```

### 6.3 为什么输出要还原成 absolute

为了方便离线评估和可视化，脚本保存的预测结果不是 relative action，而是 postprocessor 还原后的 absolute action。

这样可以直接画：

```text
GT absolute trajectory
Pred absolute trajectory
```

也可以直接计算：

```text
predicted_chunk - ground_truth_chunk
```

## 7. 单帧推理和 batching

### 7.1 每个样本输入什么

每个 timestep 的 raw batch 包含：

```python
{
    "observation.images.cam_hand": Tensor[B, C, H, W],
    "observation.images.cam_top": Tensor[B, C, H, W],
    "observation.state": Tensor[B, 10],
    "task": list[str],
}
```

其中 `task` 默认来自 dataset 的 `task` 字段，也可以用：

```bash
--task "..."
```

强制覆盖。

### 7.2 policy 输出什么

pi0.5 policy 输出 action chunk：

```text
Tensor[B, chunk_size, action_dim]
```

当前配置：

```text
chunk_size = 50
action_dim = 10
```

也就是每一帧输入都会预测未来 50 步 10D action。

### 7.3 batch-size

`--batch-size` 控制一次送多少个 timestep 给模型推理。

我们在 4090 上验证：

```bash
--batch-size 4 --device cuda --dtype bfloat16
```

可以跑通黑板 checkpoint 的 smoke test。完整 8 个 episode 的耗时取决于 GPU 和 batch size。

如果显存不足，可以降到：

```bash
--batch-size 1
```

## 8. ground truth chunk 构造

模型对每一帧都会输出一个未来 50 步 chunk。黑板 checkpoint 的主评估空间是 relative SE(3) action，所以脚本会同时构造 absolute chunk 和 delta chunk。

absolute chunk 直接来自 LeRobot 数据里的 `action`：

```python
ground_truth_action_chunk = actions[t : t + chunk_size]
```

delta chunk 按当前 absolute state 和未来 absolute action 构造：

```python
ground_truth_delta_chunk = state_action_to_relative_pose_action_np(
    actions[t : t + chunk_size],
    states[t],
    arm_offsets=[0],
    arm_stride=10,
)
```

数学上就是：

```text
T_delta[t+k] = inv(T_state[t]) @ T_action[t+k]
```

如果接近 episode 末尾，未来步数不够 50，则：

```text
valid_length < chunk_size
```

不够的部分用最后一个 valid action padding：

```python
ground_truth_action_chunk[valid_length:] = ground_truth_action_chunk[valid_length - 1]
ground_truth_delta_chunk[valid_length:] = ground_truth_delta_chunk[valid_length - 1]
```

每个 pair 保存：

```python
{
    "timestep": t,
    "observation_state": states[t],                 # absolute input state, (10,)
    "ground_truth_delta_chunk": gt_delta_chunk,     # primary GT, (50, 10)
    "predicted_delta_chunk": pred_delta_chunk,      # primary prediction, (50, 10)
    "ground_truth_chunk": gt_absolute_chunk,        # absolute GT, (50, 10)
    "predicted_chunk": pred_absolute_chunk,         # absolute-from-delta prediction, (50, 10)
    "ground_truth_action_chunk": gt_absolute_chunk,
    "predicted_action_chunk": pred_absolute_chunk,
    "valid_length": valid_length,
}
```

其中 `valid_length` 告诉可视化和指标统计：这个 chunk 里真正有效的未来步数是多少。主指标只统计 valid 部分。

## 9. 输出文件格式

每个 episode 的输出目录：

```text
output_dir/
  episode_000000/
    trajectory_pairs.pkl
    metadata.pkl
    metadata.json
```

总览输出：

```text
output_dir/
  summary.txt
  metrics.json
```

如果启用：

```bash
--save-images
```

每个 episode 还会保存：

```text
images.pkl
```

注意：`images.pkl` 会把视频帧展开成 numpy array，数据集长的时候会明显变大。

### 9.1 trajectory_pairs.pkl

核心文件是：

```python
{
    "pairs": [
        {
            "timestep": int,
            "observation_state": np.ndarray,          # absolute state[t], (10,)
            "ground_truth_delta_chunk": np.ndarray,   # relative GT, (chunk_size, 10)
            "predicted_delta_chunk": np.ndarray,      # model relative output, (chunk_size, 10)
            "ground_truth_chunk": np.ndarray,         # absolute GT, (chunk_size, 10)
            "predicted_chunk": np.ndarray,            # absolute-from-delta pred, (chunk_size, 10)
            "ground_truth_action_chunk": np.ndarray,  # absolute alias
            "predicted_action_chunk": np.ndarray,     # absolute alias
            "valid_length": int,
        },
        ...
    ],
    "chunk_size": 50,
    "state_dim": 10,
    "action_dim": 10,
    "episode_len": int,
    "action_mode": "delta",
    "state_mode": "absolute",
    "prediction_space": "chunkwise_se3_delta_10d",
    "source_dataset_root": str,
    "episode_index": int,
    "dataset_from_index": int,
    "dataset_to_index": int,
}
```

这个结构刻意和旧 `offline-inference` 保持接近，方便复用可视化和分析习惯。

### 9.2 metadata.json / metadata.pkl

保存 episode 级别元信息：

```python
{
    "episode_name": "episode_000000",
    "episode_index": 0,
    "episode_len": 475,
    "task": "...",
    "dataset_tasks": [...],
    "cameras": [
        "observation.images.cam_hand",
        "observation.images.cam_top",
    ],
    "chunk_size": 50,
    "action_dim": 10,
    "action_names": [...],
    "metrics": {...},
}
```

`metadata.json` 方便直接查看；`metadata.pkl` 保持 Python 分析兼容。

## 10. 指标统计逻辑

脚本会计算两类指标，并按 action 空间分开命名：

```text
first_step_delta
valid_chunk_delta
first_step_absolute_from_delta
valid_chunk_absolute_from_delta
```

### 10.1 first_step

只比较每个 timestep 预测 chunk 的第 0 步：

```text
predicted_delta_chunk[t, 0] vs ground_truth_delta_chunk[t, 0]
```

这接近“当前帧预测的下一步 action”误差。

### 10.2 valid_chunk

比较每个 timestep 里所有 valid 的未来步：

```text
predicted_delta_chunk[:valid_length] vs ground_truth_delta_chunk[:valid_length]
```

接近“整段未来轨迹预测”误差。

absolute-from-delta 指标是辅助项：

```text
predicted_action_chunk[:valid_length] vs ground_truth_action_chunk[:valid_length]
```

它用于判断 relative 输出复合回 absolute 轨迹后，在可视化空间里和真实 absolute action 的偏差。

### 10.3 指标项

每类指标包含：

```text
mse
mae
per_dim_mse
max_error
smoothness
```

其中 `smoothness` 是预测序列相邻差分的方差均值，只是一个轨迹平滑程度参考。

### 10.4 pi05_base 的已跑结果

用 `lerobot/models/pi05_base` 跑当前数据得到：

```text
Average first-step MSE: 95.53471527
Average first-step MAE: 2.99648499
Average valid-chunk MSE: 105.20001984
Average valid-chunk MAE: 3.08195715
```

这个结果只表示离线推理链路跑通。`pi05_base` 不是针对铲胶布任务 fine-tune 的模型，所以误差大是正常的。

## 11. 可视化逻辑

### 11.1 交互式 3D 轨迹可视化

入口：

```bash
python offline-inference-pi05/visualize_episode.py \
  --data_dir offline-inference-pi05/output_pi05_base
```

默认打开：

```text
episode_000000
```

如果想直接打开某个 episode：

```bash
python offline-inference-pi05/visualize_episode.py \
  --data_dir offline-inference-pi05/output_pi05_base \
  --episode episode_000002
```

窗口显示的是当前 timestep 的一个 action chunk：

```text
蓝色: ground truth future chunk
红色: predicted future chunk
```

每个 chunk 默认 50 步。键盘控制：

```text
长按 Right / Space: 连续向前播放 timestep
长按 Left: 连续回退 timestep
P / Play 按钮: 自动播放 / 暂停
Up: 下一个 episode
Down: 上一个 episode
Q: 退出
```

`visualize_episode.py` 在每次重绘 3D 轨迹前会读取当前 Matplotlib 3D 轴的 `elev/azim/roll`，重绘后再恢复这个视角。因此用户用鼠标旋转到任意角度后，继续按键或长按播放时不会被强制归位。

右侧图像与当前 timestep 同步刷新。图像来源优先是输出目录中的 `images.pkl`；如果没有保存 `images.pkl`，可视化脚本会根据 `source_dataset_root` 和 `dataset_from_index` 回到原始 LeRobot dataset 按需解码当前帧。

### 11.2 交互式波形可视化

入口：

```bash
python offline-inference-pi05/visualize_waveform.py \
  --data_dir offline-inference-pi05/output_pi05_base
```

它会画 10 个动作维度：

```text
x, y, z, rot0, rot1, rot2, rot3, rot4, rot5, gripper
```

图中：

```text
浅色背景: full-episode GT first-step 曲线
蓝色粗线: 当前 chunk 的 GT
红色虚线: 当前 chunk 的 Pred
橙色竖线: 当前 timestep
```

这个视图适合看某个维度是否整体偏移、抖动、发散，尤其适合检查 gripper 和 xyz。

波形图也使用同一套键盘长按 timer：

```text
长按 Right / Space: 当前 chunk 和右侧图像连续前进
长按 Left: 当前 chunk 和右侧图像连续回退
P / Play 按钮: 自动播放 / 暂停
松开方向键: 停止 key timer
```

实现上不依赖操作系统的重复 keypress，而是使用 Matplotlib timer；松手时通过一个短 `release grace timer` 过滤 Tk 后端的自动重复抖动，避免松手后继续补播放。

### 11.3 静态 PNG/HTML 预览

如果服务器没有 GUI，或者 Matplotlib 窗口弹不出来，可以用：

```bash
python offline-inference-pi05/render_preview.py \
  --data_dir offline-inference-pi05/output_pi05_base
```

它会生成：

```text
output_pi05_base/preview/
  episode_000000_trajectory.png
  episode_000000_waveform.png
  ...
  index.html
```

打开：

```text
offline-inference-pi05/output_pi05_base/preview/index.html
```

可以一次看所有 episode 的静态预览。

## 12. 常用命令

### 12.1 跑 pi05_base 完整离线推理

```bash
conda activate lerobot-pi05

python offline-inference-pi05/eval_lerobot_pi05.py \
  --checkpoint lerobot/models/pi05_base \
  --dataset-root data/clean_test_lerobot_pi05 \
  --output-dir offline-inference-pi05/output_pi05_base \
  --batch-size 4 \
  --device cuda \
  --dtype bfloat16
```

### 12.2 跑训练后的 fine-tuned checkpoint

假设训练输出目录是：

```text
lerobot/outputs/pi05_wiping_tape_h20_fullft_b16x8
```

可以直接：

```bash
python offline-inference-pi05/eval_lerobot_pi05.py \
  --checkpoint lerobot/outputs/pi05_wiping_tape_h20_fullft_b16x8 \
  --dataset-root data/clean_test_lerobot_pi05 \
  --output-dir offline-inference-pi05/output_finetuned \
  --batch-size 4 \
  --device cuda \
  --dtype bfloat16
```

脚本会自动解析：

```text
lerobot/outputs/.../checkpoints/last/pretrained_model
```

### 12.3 只跑一个 episode 调试

```bash
python offline-inference-pi05/eval_lerobot_pi05.py \
  --checkpoint lerobot/models/pi05_base \
  --dataset-root data/clean_test_lerobot_pi05 \
  --output-dir offline-inference-pi05/output_debug \
  --episodes 0 \
  --limit-frames 16 \
  --batch-size 2 \
  --device cuda \
  --dtype bfloat16
```

### 12.4 保存相机帧用于可视化

```bash
python offline-inference-pi05/eval_lerobot_pi05.py \
  --checkpoint lerobot/models/pi05_base \
  --dataset-root data/clean_test_lerobot_pi05 \
  --output-dir offline-inference-pi05/output_with_images \
  --save-images \
  --batch-size 4 \
  --device cuda \
  --dtype bfloat16
```

注意：`--save-images` 会显著增大输出目录。

## 13. 和旧 offline-inference 的对应关系

旧项目：

```text
offline-inference/
  eval_offline.py
  visualize_episode.py
  visualize_waveform.py
```

新项目：

```text
offline-inference-pi05/
  eval_lerobot_pi05.py
  visualize_episode.py
  visualize_waveform.py
```

保持一致的地方：

- 都输出 `trajectory_pairs.pkl`。
- 每个 timestep 都保存 `ground_truth_chunk` 和 `predicted_chunk`。
- 都支持 3D chunk 可视化。
- 都支持 waveform 可视化。
- 都支持 episode/timestep 键盘导航。

不同点：

```text
旧项目:
  数据多为 HDF5/pkl。
  动作通常是双臂 20D。
  模型前处理多为脚本手写。

新项目:
  数据是 LeRobot v3 parquet/video/meta。
  动作是 UMI 单手 10D pose。
  推理必须走 LeRobot pi0.5 processor。
  支持 se3_pose relative action 和 mixed normalization。
```

## 14. 常见问题

### 14.1 ModuleNotFoundError: matplotlib

如果运行可视化报：

```text
ModuleNotFoundError: No module named 'matplotlib'
```

在环境里安装：

```bash
conda activate lerobot-pi05
python -m pip install matplotlib
```

### 14.2 交互窗口没弹出来

先确认：

```bash
echo $DISPLAY
```

如果没有图形显示，使用静态预览：

```bash
python offline-inference-pi05/render_preview.py \
  --data_dir offline-inference-pi05/output_pi05_base
```

然后打开：

```text
offline-inference-pi05/output_pi05_base/preview/index.html
```

### 14.3 为什么 pi05_base 预测很差

`lerobot/models/pi05_base` 是通用预训练权重，不是铲胶布任务 fine-tune 后的权重。它能验证：

```text
数据读取
processor
模型 forward
postprocess
输出保存
可视化
```

但不能代表任务性能。真正评估任务效果需要换成 fine-tuned checkpoint。

### 14.4 为什么输出 action 是 absolute，不是 relative

模型内部输出的是 relative action；postprocessor 会根据当前 state 还原成 absolute action。保存 absolute action 是为了：

- 和 dataset action 直接比较。
- 3D 轨迹可视化直观。
- 复用旧 `trajectory_pairs.pkl` 习惯。

### 14.5 为什么 processor source 是 dataset-config

base checkpoint 里的 processor 不知道当前 UMI 数据的：

```text
cam_hand/cam_top
10D single-arm pose
se3_pose relative action
MIXED_QUANTILES stats
```

所以对 `pi05_base` 必须根据 dataset 和当前配置重建 processor。训练后的 checkpoint 如果保存了正确 processor，也可以用 `--processor-source checkpoint`。
