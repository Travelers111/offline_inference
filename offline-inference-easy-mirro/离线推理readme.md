# offline-inference-easy-mirro 最终项目总览

更新时间：2026-05-20

本文档是 `offline-inference-easy-mirro/` 的项目说明。它面向后续接手、复现实验、检查离线推理逻辑、对比 checkpoint、生成可视化的对接者。

这个目录现在只保存离线推理需要的代码、脚本、文档和工具。评估结果默认不再写进本项目目录，而是统一写到项目父目录的：

```text
/home/eai/debug/offline_inference_output/
```

默认实验目录命名规则是：

```text
<checkpoint目录名>__<数据目录名>
```

例如：

```text
CHECKPOINT=/home/eai/debug/ckpt/blackboard_pi0_45000
EVAL_DATA_DIR=/home/eai/debug/data/blackboard_testdata
```

默认输出为：

```text
/home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata
```

如果显式设置 `OUTPUT_DIR` 或传入 `--output-dir`，则使用指定目录。

## 1. 这个项目做什么

`offline-inference-easy-mirro` 是一个 Pi0 / Easy-MIRRO checkpoint 的离线推理和离线评估项目。

它不做在线控制，不控制真实机械臂，也不重新训练模型。它做的是：

1. 读取已经预处理好的 HDF5 轨迹数据。
2. 读取 checkpoint 中的模型权重、`config.json` 和 `dataset_stats.pkl`。
3. 对 HDF5 里的每一个时间步 `t`，用当前帧的 state 和图像做一次模型推理。
4. 每次推理得到未来 `chunk_size=50` 步 action 预测。
5. 按训练时一致的方式构造 ground truth。
6. 比较模型输出和 ground truth。
7. 保存 `summary.txt`、`metrics.json`、逐帧 `trajectory_pairs.pkl` 和可视化图。

它主要用于确认：

- checkpoint 的离线预测是否合理。
- 离线推理逻辑是否和训练逻辑对齐。
- 归一化是否使用了 checkpoint 自己的统计量。
- 双相机/三相机输入是否和 checkpoint 配置一致。
- delta 输出和 absolute 输出有没有被正确区分。
- 单手 10D 和双手 20D 数据是否都能按同一套 pose-block 逻辑评估。
- 可视化中每个 state 对未来 50 步 action 的预测是否平滑、是否跟 ground truth 对齐。

## 2. 核心训练契约

当前项目支持两种 action 输出模式：

- `delta`：模型输出相对于当前输入 state 的未来 SE3 delta action。
- `absolute`：模型直接输出 HDF5 `/action` 中的未来绝对 action。

两种模式的输入 state 都是绝对位姿：

```text
输入 state = /observations/qpos[t]
```

区别只在模型输出和 ground truth 的构造方式。

## 3. 数据维度约定

本项目把 action/state 看作若干个 10 维 pose block 的拼接。

每个 10D pose block：

```text
[0:3] xyz
[3:9] rot6d
[9]   gripper
```

双手任务是 20D，也就是两个 pose block：

```text
Left arm:
  [0:3]   xyz
  [3:9]   rot6d
  [9]     gripper

Right arm:
  [10:13] xyz
  [13:19] rot6d
  [19]    gripper
```

单手黑板任务是 10D：

```text
Single arm:
  [0:3] xyz
  [3:9] rot6d
  [9]   gripper
```

代码不是硬编码只能 20D；只要 action dim 是 10 的整数倍，可视化和 delta/absolute 处理都按 pose block 循环执行。目前实际验证过的是：

- 双手 20D。
- 单手黑板 10D。

## 4. HDF5 数据格式要求

严格离线评估要求 HDF5 已经是训练布局，至少包含：

```text
/observations/qpos
/action
/observations/images/<camera_name>
```

双手任务典型格式：

```text
/observations/qpos              (N, 20)
/action                         (N, 20)
/observations/images/cam_high
/observations/images/cam_left_wrist
/observations/images/cam_right_wrist
```

双相机任务可能只有：

```text
/observations/images/cam_left_wrist
/observations/images/cam_right_wrist
```

单手黑板任务典型格式：

```text
/observations/qpos              (N, 10)
/action                         (N, 10)
/observations/images/cam_high
/observations/images/cam_fisheye
```

图像可以是普通数组，也可以是压缩图像；`eval_easy_mirro.py` 会按数据格式解码。

默认情况下，评估器要求 HDF5 中必须已经有 `/observations/qpos` 和 `/action`。这是为了避免离线评估阶段临时把 raw 数据转换成训练数据格式，从而引入和训练预处理不一致的问题。

只有调试旧数据时才使用：

```bash
--allow-raw-fallback
```

严格评估 checkpoint 时不建议使用这个参数。

## 5. 归一化逻辑

离线推理不会用评估数据重新计算统计量。

归一化和反归一化只使用 checkpoint 对应的：

```text
dataset_stats.pkl
```

查找顺序是：

```text
checkpoint/dataset_stats.pkl
checkpoint_parent/dataset_stats.pkl
```

推理时核心流程：

1. 读取 `/observations/qpos[t]` 作为原始 state。
2. 根据 checkpoint 的 `state_mode` 决定送入模型的 state。
3. state 会按模型需要补齐到模型内部维度，例如 32D。
4. 调用 checkpoint 模型自己的 `normalize_inputs()`。
5. 图像按 checkpoint 配置预处理。
6. 调用 `sample_actions()` 得到 normalized action。
7. 调用 checkpoint 模型自己的 `normalize_targets.unnormalize()` 反归一化。
8. 截取真实 action dim，例如 10D 或 20D。

这点很关键：如果错用了别的数据集统计量，或者用评估数据重新算统计量，离线指标会失真。

## 6. State Mode

当前代码支持：

```text
absolute
zero_pose_gripper
zero_pose_keep_gripper
```

常规双手和当前黑板单手任务使用的是：

```text
state_mode = absolute
```

也就是模型输入直接来自：

```text
/observations/qpos[t]
```

`zero_pose_gripper` 是 `easy-mirro-fz-zeros/scripts/train_pi0_zero_pose_gripper_delta_mixed.sh` 使用的模式。它不是把 20D state 全部置零，而是：

```text
每个 10D pose block:
  xyz      -> 0
  rot6d    -> 0
  gripper  -> 保留 /observations/qpos[t] 中的绝对夹爪值
```

双手 20D 时等价于：

```text
[0:9]   左手位姿清零
[9]     左手 gripper 保留
[10:19] 右手位姿清零
[19]    右手 gripper 保留
```

`zero_pose_keep_gripper` 是同一个工程逻辑的旧别名。评估器内部会把这两个名字视为同一种 state 变换，但会保留 checkpoint/config 中原始记录的字符串写入 `metadata` 和 `summary`，方便追溯训练来源。

这一点和 delta GT 构造要分开理解：

- 送入模型和参与 qpos 归一化的是 `state_for_model(qpos, state_mode)`。
- 构造 GT delta 仍然使用 HDF5 里的真实 `/observations/qpos[t]` 和 `/action[t+k]`。
- 所以 zero-pose checkpoint 的输出比较仍然是 `inv(T_qpos[t]) @ T_action[t+k]`，不是用全零 state 去构造 GT。

如果 checkpoint 的 `config.json` 和 `dataset_stats.pkl` 中保存了 `state_mode`，评估器会读取并检查一致性。如果两边冲突，程序会报错，不会继续推理。

## 7. Delta 输出模型的离线逻辑

常规 absolute-state delta 模型的训练契约是：

```text
输入：/observations/qpos[t] 的绝对 state
输出：未来 50 步相对于当前 state 的 SE3 delta action
```

zero-pose-gripper delta 模型的训练契约是：

```text
模型输入：位姿清零、gripper 保留的 state_for_model(qpos[t])
监督目标：未来 50 步相对于真实 qpos[t] 的 SE3 delta action
```

对每个输入时间步 `t`，ground truth delta 的构造方式是：

```text
GT_delta[t, k] = inv(T_qpos[t]) @ T_action[t + k]
```

其中：

- `T_qpos[t]` 来自 `/observations/qpos[t]`。
- `T_action[t + k]` 来自 `/action[t + k]`。
- `k = 0 ... 49`。
- 如果 episode 尾部不足 50 步，则只统计有效长度 `valid_length`。

注意：这里不是简单相减。xyz 和 rot6d 会先转成 SE3 变换矩阵，通过矩阵乘法计算相对位姿，再转回 pose 表达。

Gripper 维度不做 SE3 delta，保持训练时的逻辑：使用未来 action 中的绝对 gripper 值。

Delta checkpoint 的主比较对象是：

```text
predicted_delta_chunk <-> ground_truth_delta_chunk
```

也就是 summary 里的：

```text
Primary comparison: model predicted delta action vs GT chunk-wise SE3 delta action
Average first-step delta MSE/MAE
Average valid-chunk delta MSE/MAE
```

为了观察真实机械臂操作轨迹和平滑程度，当前代码还会把 delta 输出接回当前绝对 state，生成绝对 action 曲线：

```text
Pred_absolute[t, k] = T_qpos[t] @ T_pred_delta[t, k]
```

这个曲线保存为：

```text
predicted_chunk
predicted_action_chunk
```

它和 HDF5 `/action[t:t+50]` 对比，得到 secondary metric：

```text
Average first-step absolute-from-delta MSE/MAE
Average valid-chunk absolute-from-delta MSE/MAE
```

需要注意：这不是闭环 rollout。每个时间步 `t` 都使用真实 HDF5 的 `/observations/qpos[t]` 作为当前 state，然后独立预测未来 50 步。可视化显示的是“当前 state 下模型预测的未来 50 步 chunk”，不是把上一步模型预测再喂给下一步模型的在线闭环轨迹。

## 8. Absolute 输出模型的离线逻辑

Absolute 模型的训练契约是：

```text
输入：/observations/qpos[t] 的绝对 state
输出：HDF5 /action[t:t+50] 的未来绝对 action
```

对每个输入时间步 `t`，ground truth 直接取：

```text
GT_absolute[t, k] = /action[t + k]
```

Absolute checkpoint 的主比较对象是：

```text
predicted_chunk <-> ground_truth_chunk
```

summary 中对应：

```text
Primary comparison: model predicted absolute action vs GT HDF5 /action chunk
Average first-step absolute MSE/MAE
Average valid-chunk absolute MSE/MAE
```

Absolute 模型不需要把 delta 接回 state，因为模型输出本身已经是绝对 action。

## 9. 每个 state 预测未来 50 步的含义

假设一个 episode 长度是 `N=880`，`chunk_size=50`。

离线推理会对每个 `t` 做一次模型推理：

```text
t=0   -> 预测 action[0:50]
t=1   -> 预测 action[1:51]
t=2   -> 预测 action[2:52]
...
t=879 -> 预测 action[879:929]，但实际只有 1 步有效
```

所以 `trajectory_pairs.pkl` 中每一个 pair 对应一个输入 state：

```text
pair = 第 t 帧 observation 对未来 50 步 action 的预测结果
```

尾部不足 50 帧时：

```text
valid_length < 50
```

指标只比较有效部分，不比较 padding。

## 10. 自动识别相机

默认 `CAMERAS=auto`。

评估器会读取 checkpoint 的 `config.json` 中的 `image_features`，例如：

```json
[
  "observation.images.cam_high",
  "observation.images.cam_left_wrist",
  "observation.images.cam_right_wrist"
]
```

自动得到：

```text
cam_high, cam_left_wrist, cam_right_wrist
```

双相机 checkpoint 可能是：

```text
cam_left_wrist, cam_right_wrist
```

黑板任务可能是：

```text
cam_high, cam_fisheye
```

如果 HDF5 数据缺少 checkpoint 需要的相机，评估器会直接报错。这样能避免把三相机模型误用双相机数据，或者相机顺序错了还继续跑。

如果确实需要手动覆盖：

```bash
CAMERAS=cam_left_wrist,cam_right_wrist \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

一般不建议手动覆盖，除非你明确知道 checkpoint 训练时的相机配置。

## 11. 自动识别 Action Mode

默认：

```bash
--action-mode auto
```

评估器会读取 checkpoint 的：

```text
config.json
dataset_stats.pkl
```

自动判断：

```text
delta
absolute
```

如果 `config.json` 和 `dataset_stats.pkl` 中记录的 action mode 冲突，程序会拒绝继续运行。原因是这种情况通常说明 checkpoint 和 stats 不属于同一次训练。

可以手动指定：

```bash
--action-mode delta
--action-mode absolute
```

但严格评估时推荐使用默认自动识别。

## 12. 支持的模型版本

当前代码兼容以下模式。

| 模式                | 输入相机 | state         | 输出             | action dim | 推荐入口                   |
| ------------------- | -------- | ------------- | ---------------- | ---------- | -------------------------- |
| 双相机双手 delta    | 2 cam    | absolute qpos | SE3 delta        | 20D        | `run_eval_easy_mirro.sh` |
| 三相机双手 delta    | 3 cam    | absolute qpos | SE3 delta        | 20D        | `run_eval_easy_mirro.sh` |
| 三相机双手 absolute | 3 cam    | absolute qpos | HDF5 `/action` | 20D        | `run_eval_easy_mirro.sh` |
| 方舟 zero-pose delta | 2 cam    | zero pose + absolute gripper | SE3 delta | 20D | `run_eval_easy_mirro.sh` |
| 单手黑板 delta      | 2 cam    | absolute qpos | SE3 delta        | 10D        | `run_eval_blackboard.sh` |

黑板任务推荐用 `run_eval_blackboard.sh`，因为它默认把 `EASY_MIRRO_ROOT` 指到：

```text
/home/eai/debug/easy-mirro-blackboard
```

双手任务默认用：

```text
/home/eai/debug/easy-mirro-dual-new_1
```

当前工作区如果没有 `easy-mirro-dual-new_1`，`run_eval_easy_mirro.sh` 会自动回退到：

```text
/home/eai/debug/easy-mirro-fz-zeros
```

如果两个训练工程都存在，而你要评估 `fz_zerostate_45000` 这类 zero-pose-gripper checkpoint，建议显式指定：

```bash
EASY_MIRRO_ROOT=/home/eai/debug/easy-mirro-fz-zeros
```

如果模型代码根目录不同，可以设置：

```bash
EASY_MIRRO_ROOT=/path/to/easy-mirro-root
```

## 13. 主要文件结构

```text
offline-inference-easy-mirro/
  README.md
  README_zh.md
  PROJECT_GUIDE_zh.md
  FINAL_PROJECT_OVERVIEW_zh.md
  OFFLINE_INFERENCE_PI0_EASY_MIRRO_DETAILS.md
  OFFLINE_INFERENCE_PI0_EASY_MIRRO_DETAILS_zh.md

  run_eval_easy_mirro.sh
  run_eval_blackboard.sh
  run_eval_fz_test2_all.sh

  eval_easy_mirro.py
  image_frame_source.py
  validate_alignment.py
  render_preview.py
  visualize_episode.py
  visualize_waveform.py
  compare_checkpoints.py
```

职责说明：

```text
run_eval_easy_mirro.sh
  双手/通用评估入口。负责设置 PYTHONPATH、环境变量，然后调用 eval_easy_mirro.py。

run_eval_blackboard.sh
  单手黑板任务入口。默认使用 easy-mirro-blackboard 的模型代码根目录。

run_eval_fz_test2_all.sh
  批量评估已有 Fangzhou checkpoint 的脚本，会运行推理、alignment 校验、静态预览和 checkpoint 对比。

eval_easy_mirro.py
  核心离线推理脚本。负责加载 checkpoint、读取 HDF5、推理、构造 GT、算指标、保存结果。

image_frame_source.py
  可视化图像读取工具。根据 metadata/trajectory_pairs 中记录的 source_hdf5_path 和 cameras，
  从原始 HDF5 里按当前 timestep 读取模型实际用到的相机帧。

validate_alignment.py
  验证离线 GT delta 是否和训练代码中的 delta 构造一致。

render_preview.py
  生成静态 PNG 和 index.html 预览，适合服务器无 GUI 环境。

visualize_episode.py
  交互式 3D 轨迹可视化。

visualize_waveform.py
  交互式 action 维度波形可视化。

compare_checkpoints.py
  从多个输出目录读取 metrics.json，生成 checkpoint 对比 JSON。
```

## 14. 输出目录结构

一次推理完成后，默认输出目录类似：

```text
/home/eai/debug/offline_inference_output/<checkpoint名字>__<数据目录名字>/
  summary.txt
  metrics.json
  collection_xxx/
    trajectory_pairs.pkl
    metadata.pkl
    metadata.json
  collection_yyy/
    trajectory_pairs.pkl
    metadata.pkl
    metadata.json
  preview/
    index.html
    *_delta_trajectory.png
    *_delta_waveform.png
    *_absolute_trajectory.png
    *_absolute_waveform.png
```

如果运行时使用 `--save-images`，episode 目录里还会保存：

```text
images.pkl
```

默认不保存图像，是为了避免输出目录过大。当前可视化不依赖 `images.pkl`，而是优先从每个 episode 的：

```text
metadata.json / metadata.pkl
trajectory_pairs.pkl
```

读取：

```text
source_hdf5_path   原始 HDF5 文件路径
cameras            当前 checkpoint 推理实际使用的相机列表
```

然后按当前 pair 的 `timestep` 到原始 HDF5 里读取对应图像帧。这样可视化展示的就是模型推理实际输入的相机画面，没被模型使用的相机不会显示。

`images.pkl` 只是兼容旧输出或显式 `--save-images` 的缓存路径。一般不需要为了看图像重新推理并加 `--save-images`。

## 15. `summary.txt` 怎么看

`summary.txt` 是最快速的人类可读结果。

Delta checkpoint 重点看：

```text
Primary comparison: model predicted delta action vs GT chunk-wise SE3 delta action
Average first-step delta MSE
Average first-step delta MAE
Average valid-chunk delta MSE
Average valid-chunk delta MAE
```

同时会有 secondary absolute-from-delta：

```text
Secondary comparison: delta prediction composed with input state into absolute action
Average first-step absolute-from-delta MSE
Average valid-chunk absolute-from-delta MSE
```

Absolute checkpoint 重点看：

```text
Primary comparison: model predicted absolute action vs GT HDF5 /action chunk
Average first-step absolute MSE
Average first-step absolute MAE
Average valid-chunk absolute MSE
Average valid-chunk absolute MAE
```

概念解释：

```text
first-step:
  每个 state 预测出来的 50 步 chunk 中第 1 步。

valid-chunk:
  每个 state 预测出来的未来最多 50 步，episode 尾部不足 50 步时只统计 valid_length。
```

## 16. `trajectory_pairs.pkl` 字段

`trajectory_pairs.pkl` 是最重要的逐帧结果文件。它保存一个 dict，里面的 `pairs` 是逐帧结果列表。

公共字段：

```python
{
    "timestep": int,
    "observation_state": np.ndarray,         # qpos[t]
    "ground_truth_action_chunk": np.ndarray, # HDF5 /action[t:t+50]
    "ground_truth_delta_chunk": np.ndarray,  # inv(T_qpos[t]) @ T_action[t+k]
    "valid_length": int,
}
```

Delta 输出额外字段：

```python
{
    "predicted_delta_chunk": np.ndarray,     # 模型输出的 delta action
    "ground_truth_chunk": np.ndarray,        # absolute GT alias
    "predicted_chunk": np.ndarray,           # T_qpos[t] @ predicted_delta_chunk
    "predicted_action_chunk": np.ndarray,    # absolute pred alias
}
```

Absolute 输出额外字段：

```python
{
    "ground_truth_chunk": np.ndarray,        # /action[t:t+50]
    "predicted_chunk": np.ndarray,           # 模型输出的 absolute action
    "predicted_action_chunk": np.ndarray,    # absolute pred alias
}
```

可视化脚本正是读取这些字段画图。

## 17. 基础运行命令

以下命令默认把结果写到：

```text
/home/eai/debug/offline_inference_output/<checkpoint_name>__<data_name>
```

### 17.1 双相机双手 delta

```bash
CHECKPOINT=/home/eai/debug/ckpt__fz/checkpoint-45000 \
EVAL_DATA_DIR=/home/eai/debug/rawdata \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

如果 checkpoint 在中文命名目录或旧目录中，也可以直接指定真实路径：

```bash
CHECKPOINT=/home/eai/debug/ckpt__fz/双相机相对输出/checkpoint-45000 \
EVAL_DATA_DIR=/home/eai/debug/rawdata \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

### 17.2 三相机双手 delta

```bash
CHECKPOINT=/home/eai/debug/ckpt__fz/3cam_se3_delta_mixed_output_45000_inference \
EVAL_DATA_DIR=/home/eai/debug/rawdata \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

### 17.3 三相机双手 absolute

```bash
CHECKPOINT=/home/eai/debug/ckpt__fz/3cam_absolute_mixed_output_45000_inference \
EVAL_DATA_DIR=/home/eai/debug/rawdata \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

### 17.4 单手黑板 delta

```bash
CHECKPOINT=/home/eai/debug/ckpt/blackboard_pi0_45000 \
EVAL_DATA_DIR=/home/eai/debug/data/blackboard_testdata \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_blackboard.sh --strict
```

### 17.5 方舟 zero-pose-gripper 双手 delta

这个 checkpoint 对应的训练脚本是：

```text
easy-mirro-fz-zeros/scripts/train_pi0_zero_pose_gripper_delta_mixed.sh
```

训练配置是：

```text
state_mode = zero_pose_gripper
action_mode = delta
norm_mode = mixed
chunk_size = 50
state_dim/action_dim = 20/20
```

离线评估命令：

```bash
EASY_MIRRO_ROOT=/home/eai/debug/easy-mirro-fz-zeros \
CHECKPOINT=/home/eai/debug/ckpt/fz_zerostate_45000 \
EVAL_DATA_DIR=/home/eai/debug/rawdata \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

如果要先做很小的 smoke test：

```bash
EASY_MIRRO_ROOT=/home/eai/debug/easy-mirro-fz-zeros \
CHECKPOINT=/home/eai/debug/ckpt/fz_zerostate_45000 \
EVAL_DATA_DIR=/home/eai/debug/rawdata \
BATCH_SIZE=2 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh \
  --strict \
  --max-episodes 1 \
  --limit-frames 4
```

注意：这里的 `EVAL_DATA_DIR` 只是示例，实际要替换成要评估的 processed HDF5 数据目录。

### 17.6 指定输出目录

```bash
CHECKPOINT=/home/eai/debug/ckpt/blackboard_pi0_45000 \
EVAL_DATA_DIR=/home/eai/debug/data/clean_replayraw \
OUTPUT_DIR=/home/eai/debug/offline_inference_output/my_blackboard_test \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_blackboard.sh --strict
```

### 17.7 只跑少量帧做 smoke test

```bash
CHECKPOINT=/home/eai/debug/ckpt/blackboard_pi0_45000 \
EVAL_DATA_DIR=/home/eai/debug/data/clean_replayraw \
BATCH_SIZE=2 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_blackboard.sh \
  --strict \
  --max-episodes 1 \
  --limit-frames 4
```

### 17.8 只跑指定 episode

按文件名或 stem 指定：

```bash
CHECKPOINT=/home/eai/debug/ckpt__fz/3cam_absolute_mixed_output_45000_inference \
EVAL_DATA_DIR=/home/eai/debug/data/推理0515_sz_compressed \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh \
  --strict \
  --files collection_1778826190
```

按排序后的 episode index 指定：

```bash
CHECKPOINT=/home/eai/debug/ckpt__fz/3cam_absolute_mixed_output_45000_inference \
EVAL_DATA_DIR=/home/eai/debug/data/推理0515_sz_compressed \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh \
  --strict \
  --episodes 0-2
```

### 17.9 列出数据目录中的 episode

```bash
CHECKPOINT=/home/eai/debug/ckpt__fz/3cam_absolute_mixed_output_45000_inference \
EVAL_DATA_DIR=/home/eai/debug/data/推理0515_sz_compressed \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --list-episodes
```

### 17.10 覆盖 task 文本

默认 `TASK=auto` 时会优先读取 HDF5 的：

```text
attrs["language_instruction"]
```

也可以手动指定：

```bash
TASK="wipe the blackboard" \
CHECKPOINT=/home/eai/debug/ckpt/blackboard_pi0_45000 \
EVAL_DATA_DIR=/home/eai/debug/data/blackboard_testdata \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_blackboard.sh --strict
```

## 18. 可视化总览

当前有三种可视化方式：

1. `render_preview.py`：静态 HTML 和 PNG，最适合服务器无 GUI 环境。
2. `visualize_episode.py`：交互式 3D 轨迹查看。
3. `visualize_waveform.py`：交互式 10D/20D action 波形查看。

三种可视化都支持显示当前 HDF5 图像帧：

- 只显示模型推理实际使用的相机。
- 相机列表来自输出里的 `metadata.json` / `metadata.pkl` 的 `cameras` 字段。
- 原始图像路径来自 `source_hdf5_path`。
- 当前显示的图像帧号来自当前 pair 的 `timestep`。
- 切换 timestep、拖动进度条、长按键盘、点击播放按钮、切换 episode 时，图像和动作曲线/轨迹会同步更新。
- 如果不想读取或显示图像，可以加 `--no-images`。

图像读取由 `image_frame_source.py` 统一处理。它会优先读原始 HDF5，不要求输出目录里有 `images.pkl`。如果 episode 目录里存在 `images.pkl`，也可以兼容读取这个缓存。

这一点很重要：可视化展示的不是任意相机，也不是所有 HDF5 相机，而是当前 checkpoint 实际送入模型的图像输入。因此三相机模型会显示三路相机，双相机模型只显示双相机，黑板单手模型显示它训练/推理使用的相机。

可视化支持：

```text
--space auto
--space delta
--space absolute
--no-images
--play-interval-ms 40
```

对 delta checkpoint：

- `--space delta` 看模型输出 delta 和 GT delta。
- `--space absolute` 看 delta 接回当前 state 后的 absolute action 曲线。

对 absolute checkpoint：

- `--space absolute` 看模型直接输出的 absolute action 和 HDF5 `/action`。
- 如果请求不存在的空间，脚本会尽量 fallback 到可用空间，并打印 warning。

xyz 坐标默认以分米显示：

```text
--position-scale 10
--position-unit dm
```

这样更容易观察机械臂实际运动的平滑程度。rot6d 和 gripper 不做单位缩放。

`--play-interval-ms` 只用于交互式 3D 和波形图，默认是 `40ms`，也就是比早期版本快一倍。数值越小播放越快，例如：

```bash
--play-interval-ms 25
```

如果图像分辨率很高或机器绘图慢，可以调大一些，例如：

```bash
--play-interval-ms 80
```

## 19. 生成静态 HTML 预览

推荐先运行：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/render_preview.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata
```

输出：

```text
preview/index.html
preview/*_trajectory.png
preview/*_waveform.png
```

如果只想渲染某个 episode：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/render_preview.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  -i collection_1775201188
```

如果不想在 HTML 中嵌入或复制图像：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/render_preview.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  --no-images
```

静态预览会自动渲染 `trajectory_pairs.pkl` 中存在的所有 action space。对于 delta 输出，现在通常会同时生成：

```text
*_delta_trajectory.png
*_delta_waveform.png
*_absolute_trajectory.png
*_absolute_waveform.png
```

其中 absolute 是把模型 delta 输出接回 `/observations/qpos[t]` 后得到的曲线。

静态预览中的图像逻辑：

- `*_trajectory.png` 里会在轨迹图下方显示若干 snapshot 的 HDF5 当前帧图像。
- `*_waveform.png` 顶部也会放入对应 snapshot 的 HDF5 图像条，下面是各维度波形。
- snapshot 通常取 episode 开始、中间和最后一个完整 50 步 chunk 附近。
- 图像标题会标出 `t=<timestep>` 和相机名。
- 图像只包含模型使用的相机，不包含 HDF5 中额外存在但模型没有用的相机。

因此静态 HTML 适合快速检查：

- 当前帧机器人或人手到底在做什么。
- 模型预测的未来 50 步动作和图像中任务阶段是否对应。
- 预测曲线跳变是否发生在某个具体视觉状态附近。

## 20. 交互式 3D 轨迹可视化

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_episode.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  -i collection_1775201188 \
  --space absolute
```

查看 delta 空间：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_episode.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  -i collection_1775201188 \
  --space delta
```

使用自动空间选择：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_episode.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  --space auto
```

设置播放速度：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_episode.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  -i collection_1775201188 \
  --space absolute \
  --play-interval-ms 40
```

如果不想显示 HDF5 图像：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_episode.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  -i collection_1775201188 \
  --space absolute \
  --no-images
```

3D 界面布局：

- 左侧是当前 state 对未来 50 步 action chunk 的 3D 轨迹。
- 右侧是当前 `timestep` 对应的 HDF5 图像帧。
- 右侧只显示模型使用过的相机。
- 标题里会显示 episode 名、当前 action space、当前 timestep、pair index、valid chunk 长度。
- 底部有进度条和 `Play/Pause` 按钮。

交互控制：

```text
SPACE / RIGHT        按住向前连续播放，松开停止
LEFT                 按住向后连续回退，松开停止
P                    播放/暂停自动播放
Play / Pause 按钮    播放/暂停自动播放
拖动底部进度条       跳转到指定 pair/timestep
UP                   下一个 episode
DOWN                 上一个 episode
Q                    退出
```

键盘长按逻辑：

- 长按 `SPACE` 或 `RIGHT` 时，内部 timer 按 `--play-interval-ms` 连续向前播放。
- 长按 `LEFT` 时，内部 timer 连续向后回退。
- 松开按键后会停止，不会继续处理一串积压的 keypress。
- 方向键长按和 `Play/Pause` 自动播放互斥；按方向键会停止自动播放。
- `P` 或点击 `Play/Pause` 会停止方向键长按状态，然后切换自动播放。

这里特意不在键盘回调里调用 `flush_events()`。原因是 Tk/Matplotlib 后端会在 `flush_events()` 中继续处理排队的键盘事件，长按时可能递归进入 `on_key_press`，导致松手后继续播放甚至栈溢出。当前实现只用 timer 和 `draw_idle()` 更新，避免 keypress 队列失控。

在没有 GUI 的服务器上做 smoke test：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_episode.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  -i collection_1775201188 \
  --space absolute \
  --no-show
```

## 21. 交互式波形可视化

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_waveform.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  -i collection_1775201188 \
  --space absolute
```

查看 delta：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_waveform.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  -i collection_1775201188 \
  --space delta
```

设置播放速度：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_waveform.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  -i collection_1775201188 \
  --space absolute \
  --play-interval-ms 40
```

如果不想显示 HDF5 图像：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_waveform.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  -i collection_1775201188 \
  --space absolute \
  --no-images
```

波形图用于看每个 action 维度的预测曲线是否平滑、是否偏移、是否存在突然跳变。

波形界面布局：

- 左侧是 10D 或 20D action 的维度波形。
- 背景淡色曲线是整个 episode 每个 pair 的 first-step 预测/GT。
- 加粗曲线是当前 pair 预测的未来 50 步 chunk。
- 橙色竖线是当前 chunk 起点，也就是当前输入 state 的 `timestep`。
- 右侧是当前 `timestep` 对应的 HDF5 图像帧。
- 底部有进度条和 `Play/Pause` 按钮。

波形图的交互控制和 3D 图一致：

```text
SPACE / RIGHT        按住向前连续播放，松开停止
LEFT                 按住向后连续回退，松开停止
P                    播放/暂停自动播放
Play / Pause 按钮    播放/暂停自动播放
拖动底部进度条       跳转到指定 pair/timestep
UP                   下一个 episode
DOWN                 上一个 episode
Q                    退出
```

波形图也使用同一套键盘长按 timer 逻辑，不依赖系统重复 keypress。这样图像帧和波形 chunk 会同步刷新，松开键盘后不会继续补播放。

如果看到：

```text
FigureCanvasAgg is non-interactive
Maximum number of clients reached
```

说明当前环境没有可交互图形后端，或者 X server 客户端数量已满。这时推荐使用 `render_preview.py` 生成静态 HTML，而不是 `plt.show()` 交互窗口。

## 22. 对齐验证

`validate_alignment.py` 用来确认离线评估里的 GT 构造是否和训练代码一致。

基础命令：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/validate_alignment.py \
  --data-dir /home/eai/debug/data/blackboard_testdata \
  --output-dir /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  --easy-mirro-root /home/eai/debug/easy-mirro-blackboard \
  --chunk-size 50 \
  --require-training-layout
```

它会检查：

1. 离线评估中计算的 SE3 delta 是否等于训练 dataset 里的 `_compute_action_delta()`。
2. 离线评估中计算的 SE3 delta 是否等于统计脚本里的 delta 计算方式。
3. delta 转回 absolute 是否可以还原 HDF5 `/action`。
4. 输出目录中的 `trajectory_pairs.pkl` 保存的 GT delta 是否一致。
5. absolute 输出中的 `ground_truth_chunk` 是否等于 HDF5 `/action[t:t+50]`。

正常误差应该接近：

```text
0 到 1e-7
```

如果 alignment 不通过，不应该相信当前离线推理指标。

## 23. Checkpoint 对比

多个输出目录可以用 `compare_checkpoints.py` 生成对比 JSON：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/compare_checkpoints.py \
  --outputs \
  /home/eai/debug/offline_inference_output/checkpoint-35000__rawdata \
  /home/eai/debug/offline_inference_output/checkpoint-45000__rawdata \
  --names checkpoint-35000 checkpoint-45000 \
  --output /home/eai/debug/offline_inference_output/comparison.json
```

已有批量脚本：

```bash
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_fz_test2_all.sh
```

它会做：

1. 评估 checkpoint。
2. 跑 alignment 检查。
3. 生成 preview。
4. 写 checkpoint comparison。

默认输出：

```text
/home/eai/debug/offline_inference_output/fz_rawdata/
```

## 24. 典型工作流

### 24.1 新 checkpoint 快速检查

1. 先列出数据：

```bash
CHECKPOINT=/path/to/checkpoint \
EVAL_DATA_DIR=/path/to/hdf5_data \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --list-episodes
```

2. 跑少量帧 smoke test：

```bash
CHECKPOINT=/path/to/checkpoint \
EVAL_DATA_DIR=/path/to/hdf5_data \
BATCH_SIZE=2 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh \
  --strict \
  --max-episodes 1 \
  --limit-frames 4
```

3. 跑完整评估：

```bash
CHECKPOINT=/path/to/checkpoint \
EVAL_DATA_DIR=/path/to/hdf5_data \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

4. 生成静态 preview：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/render_preview.py \
  -d /home/eai/debug/offline_inference_output/<checkpoint_name>__<data_name>
```

5. 检查 `summary.txt` 和 `preview/index.html`。

### 24.2 单手黑板 checkpoint 工作流

```bash
CHECKPOINT=/home/eai/debug/ckpt/blackboard_pi0_45000 \
EVAL_DATA_DIR=/home/eai/debug/data/blackboard_testdata \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_blackboard.sh --strict
```

然后：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/render_preview.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata
```

需要看 delta：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_waveform.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  --space delta
```

需要看接回绝对 state 后的 action 曲线：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_waveform.py \
  -d /home/eai/debug/offline_inference_output/blackboard_pi0_45000__blackboard_testdata \
  --space absolute
```

## 25. 指标解读注意事项

### 25.1 Delta 和 absolute 的 MSE 不能直接横向比较

Delta 模型的主 MSE 在 SE3 delta 空间。

Absolute 模型的主 MSE 在 HDF5 `/action` 绝对空间。

这两个数值尺度不一样，所以不能简单说：

```text
absolute MSE 小于 delta MSE，因此 absolute 一定更好
```

同一 action mode 的 checkpoint 可以直接比较。不同 action mode 需要结合：

- absolute-from-delta 曲线。
- 3D 轨迹图。
- 波形图。
- 按维度误差。
- 真实上机成功率。

### 25.2 Gripper 可能主导总 MSE

夹爪通常是 0 到 80 左右的连续值。xyz 和 rot6d 的数值尺度可能更小。

因此 10D/20D 平均 MSE 可能被 gripper 维度主导。看细节时不要只看总 MSE，最好结合波形图按维度看：

```text
xyz
rot6d
gripper
```

### 25.3 离线推理不是在线闭环成功率

离线推理使用已有 HDF5 的图像和 state。对于 UMI 采集数据，可能出现：

```text
训练/离线数据顶部相机看到人手拿夹爪
真实上机顶部相机看到机械臂本体
```

这种 vision gap 不一定完全体现在离线 MSE 中。

离线评估能确认：

- 推理代码是否对齐训练。
- checkpoint 在数据分布内预测是否合理。
- 不同 checkpoint 的离线相对趋势。
- 预测曲线是否平滑、是否明显偏离。

最终上机成功率仍需要真实机器人闭环测试确认。

## 26. 常见错误和排查

### 26.1 找不到 `/observations/qpos` 或 `/action`

说明数据不是训练布局。需要先用训练时相同的预处理脚本生成 processed HDF5。

严格评估不建议用 raw fallback。

### 26.2 缺少相机

如果 checkpoint 需要 `cam_high, cam_left_wrist, cam_right_wrist`，但 HDF5 只有两个相机，会直接报错。

检查：

```bash
CHECKPOINT=... EVAL_DATA_DIR=... \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --list-episodes
```

以及 HDF5 中：

```text
/observations/images/<camera_name>
```

是否存在。

### 26.3 Action mode 冲突

如果 `config.json` 和 `dataset_stats.pkl` 一个写 delta，一个写 absolute，程序会拒绝推理。

处理方式：

- 检查 checkpoint 和 `dataset_stats.pkl` 是否来自同一次训练。
- 不要手动复制别的训练目录里的 stats。

### 26.4 State mode 冲突或 zero-pose 模型效果异常

如果 checkpoint 是 `fz_zerostate_45000` 这类模型，`config.json` 和 `dataset_stats.pkl` 应该记录：

```text
state_mode = zero_pose_gripper
action_mode = delta
norm_mode = mixed
```

如果程序报 `Unsupported state_mode` 或 `state mode disagree`，优先检查：

- checkpoint 和 `dataset_stats.pkl` 是否来自同一次训练。
- 是否显式设置了错误的 `EASY_MIRRO_ROOT`。
- 是否把 zero-pose-gripper checkpoint 当成普通 absolute-state checkpoint 解释。

如果程序能跑但曲线明显不对，重点确认模型输入 state 是否按训练逻辑变换：位姿清零、gripper 保留。不能把 20D state 全部置零，因为训练时夹爪维度不是零。

### 26.5 离线结果突然很差

优先排查：

1. 是否用错 checkpoint 或 stats。
2. 是否把 delta 输出当 absolute 比较。
3. 是否把 absolute 输出当 delta 比较。
4. 是否用简单减法构造 delta，而不是 `inv(T_qpos) @ T_action`。
5. 是否相机数量和顺序不一致。
6. 是否评估数据不是训练布局。
7. 是否 HDF5 `/action` 的定义和训练时不一致。
8. 是否 task/language instruction 和训练时差异太大。
9. zero-pose-gripper 模型是否保留了 gripper 输入，而不是整段 state 全零。

### 26.6 可视化无法弹窗

如果出现：

```text
FigureCanvasAgg is non-interactive
Maximum number of clients reached
```

优先使用：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/render_preview.py -d <output_dir>
```

不要在无 GUI 的服务器上依赖交互式 `plt.show()`。

### 26.7 长按键盘后仍继续播放

当前版本已经针对 Tk/Matplotlib 的键盘自动重复做了处理：

- 长按方向键不会直接处理系统重复 keypress 队列。
- 按下方向键后启动内部 timer。
- 松开方向键后停止 timer。
- 自动重复中的 release/press 抖动会被一个很短的 grace window 过滤。
- 键盘回调里不会调用 `flush_events()`，避免递归处理 keypress。

如果仍然遇到松手后继续播放，优先确认运行的是最新脚本：

```bash
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_episode.py --help
python3 /home/eai/debug/offline-inference-easy-mirro/visualize_waveform.py --help
```

帮助信息里应该能看到：

```text
--play-interval-ms PLAY_INTERVAL_MS
--no-images
```

也可以先把播放间隔调大，降低 GUI 绘图压力：

```bash
--play-interval-ms 80
```

如果使用远程 X11、VNC 或服务器图形转发，交互刷新可能比本地桌面慢。此时建议优先用 `render_preview.py` 生成静态 HTML 检查整体结果，再用交互窗口查看少量 episode。

## 27. 当前版本最重要的保证

当前项目的关键保证是：

1. 归一化只使用 checkpoint 对应的 `dataset_stats.pkl`。
2. 输入 state 默认是 HDF5 `/observations/qpos[t]` 的绝对位姿。
3. 每个 state 独立预测后面 50 步 action。
4. Delta 输出的 GT 使用 SE3 矩阵：`inv(T_qpos[t]) @ T_action[t+k]`。
5. Delta 输出会额外转成 absolute action 曲线，便于观察真实机械臂轨迹和平滑程度。
6. Absolute 输出直接和 HDF5 `/action[t:t+50]` 比较。
7. 双手 20D 和单手 10D 使用同一套 pose-block 逻辑。
8. 相机默认从 checkpoint `image_features` 自动读取。
9. action mode 默认从 checkpoint config/stats 自动读取。
10. 缺失字段、缺失相机、mode 冲突会显式报错。
11. `validate_alignment.py` 可以检查离线 GT 和训练 GT 是否一致。
12. 输出默认写到 `/home/eai/debug/offline_inference_output/`，项目目录只保留代码和工具。

这些保证是为了避免之前那类问题：模型看起来离线效果很差，但实际原因是离线推理/评估逻辑和训练逻辑不一致。
