# easy-mirro Pi0 离线推理项目完整说明

本文档说明 `offline-inference-easy-mirro` 这个目录里的离线推理/离线评估项目做了什么、整体流程是什么、推理逻辑如何与训练对齐、每个脚本负责什么、结果文件怎么看，以及使用时需要注意哪些问题。

## 1. 项目目标

这个项目用于评估 `easy-mirro-dual-new_1` 训练出来的 Pi0 / MIRRO checkpoint。

它不是在线控制机器人，也不是重新训练模型。它做的是：

1. 读取已经处理好的 HDF5 轨迹数据。
2. 读取 checkpoint 的模型权重、模型配置和 `dataset_stats.pkl` 归一化统计量。
3. 对每一帧 observation 做一次模型推理，得到未来 `chunk_size=50` 步动作预测。
4. 按照训练时完全相同的 action 表达方式构造 ground truth。
5. 比较模型输出和 ground truth，保存数值指标、逐帧预测结果和可视化结果。

这个项目主要用来回答：

- 这个 checkpoint 在已有数据上预测得准不准？
- 训练和推理的数据格式有没有对齐？
- 模型输出是 delta action 还是 absolute action？
- 三相机/双相机输入是否正确送进模型？
- 之前离线推理效果差，到底是模型问题，还是评估逻辑写错？

## 2. 支持的模型类型

当前离线推理支持两类 Pi0 checkpoint。

### 2.1 SE3 delta 输出模型

训练脚本示例：

```text
easy-mirro-dual-new_1/scripts/train_pi0_absolute_delta_optimized.sh
easy-mirro-dual-new_1/scripts/train_pi0_3cam_se3_delta_mixed.sh
```

这类模型的训练契约是：

```text
输入 state：/observations/qpos[t]，20 维绝对位姿
输出 action：相对于当前 state 的未来 50 步 SE3 delta
```

ground truth delta 的构造方式是：

```text
T_delta[t, k] = inv(T_qpos[t]) @ T_action[t + k]
```

注意这里不是简单相减。平移和旋转都在 SE3 矩阵空间中计算。夹爪维度不做 delta，仍然使用未来 action 里的绝对夹爪值。

这类模型的主要比较对象是：

```text
predicted_delta_chunk  <->  ground_truth_delta_chunk
```

单手黑板任务也属于这类模型，只是维度从双手 20D 变成单手 10D：

```text
easy-mirro-blackboard/scripts/train_pi0_wiping_blackboard_delta_mixed.sh
state_dim=10
action_dim=10
camera_names=cam_high,cam_fisheye
```

### 2.2 absolute action 输出模型

训练脚本示例：

```text
easy-mirro-dual-new_1/scripts/train_pi0_3cam_absolute_mixed.sh
```

这类模型的训练契约是：

```text
输入 state：/observations/qpos[t]，20 维绝对位姿
输出 action：HDF5 里的 /action[t:t+50] 绝对未来动作
```

ground truth 不再计算 SE3 delta，而是直接取 HDF5 中的 action chunk：

```text
GT_absolute[t, k] = /action[t + k]
```

这类模型的主要比较对象是：

```text
predicted_chunk  <->  ground_truth_chunk
```

## 3. 自动识别逻辑

用户通常不需要手动指定双相机/三相机，也不需要手动指定 delta/absolute。评估器会优先读取 checkpoint 自己的配置。

### 3.1 自动识别相机

`eval_easy_mirro.py` 会读取 checkpoint 的 `config.json`：

```json
"image_features": [
  "observation.images.cam_high",
  "observation.images.cam_left_wrist",
  "observation.images.cam_right_wrist"
]
```

然后自动得到：

```text
cam_high, cam_left_wrist, cam_right_wrist
```

双相机 checkpoint 则通常是：

```text
cam_left_wrist, cam_right_wrist
```

如果评估数据中缺少模型需要的相机，评估器会直接报错，而不是静默少传一个相机继续跑。这样可以避免误评估。

### 3.2 自动识别 action mode

评估器会读取：

```text
checkpoint/config.json
dataset_stats.pkl
```

并解析：

```text
action_mode = delta
action_mode = absolute
```

如果 `config.json` 和 `dataset_stats.pkl` 对 action mode 的记录不一致，程序会拒绝继续运行，因为这说明 checkpoint 和归一化统计不匹配。

如果是旧 checkpoint 没有 `action_mode` 字段，评估器会根据 `dataset_stats.pkl` 里的 `use_delta` 兼容判断。

## 4. 数据格式要求

严格离线评估要求 HDF5 已经是训练布局，也就是包含：

```text
/observations/qpos              (N, 20)
/action                         (N, 20)
/observations/images/<camera>   (N, H, W, 3) 或压缩图像
```

20 维 state/action 的含义是：

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

单手黑板任务是 10 维：

```text
Single arm:
  [0:3] xyz
  [3:9] rot6d
  [9]   gripper
```

这里的 `/observations/qpos` 是 frame-0 坐标系下的绝对双臂位姿。`/action` 是未来动作位姿，absolute 模型直接学它，delta 模型会基于它和当前 qpos 计算 SE3 delta。

默认情况下，评估器会拒绝缺少 `/observations/qpos` 和 `/action` 的原始采集文件。原因是：如果在离线评估代码里临时把 raw state 转成 qpos/action，很容易和训练时的数据预处理不完全一致，最终导致离线指标错误。

只有调试旧数据时才建议使用：

```bash
--allow-raw-fallback
```

严格评估 checkpoint 时不建议使用这个参数。

## 5. 归一化和反归一化

离线推理不会用评估数据重新计算统计量。它只使用 checkpoint 对应的 `dataset_stats.pkl`。

加载位置的查找顺序是：

```text
checkpoint/dataset_stats.pkl
checkpoint_parent/dataset_stats.pkl
CHECKPOINT路径/dataset_stats.pkl
```

推理时流程是：

1. 原始 20 维 state 填充到 32 维。
2. 用 checkpoint 的 `normalize_inputs` 归一化 state。
3. 图像按模型配置处理并送入模型。
4. 模型输出 normalized action。
5. 用 checkpoint 的 `normalize_targets.unnormalize` 反归一化。
6. 截取前 20 维作为真实 action 输出。

这点非常重要：离线评估必须和训练时使用同一份 `dataset_stats.pkl`，否则模型输出会被错误缩放，指标会完全不可信。

## 6. 完整推理流程

入口脚本是：

```text
run_eval_easy_mirro.sh
```

它主要负责：

- 设置 `PYTHONPATH`
- 激活必要环境
- 读取环境变量中的 `CHECKPOINT`、`EVAL_DATA_DIR`，以及可选的 `OUTPUT_DIR`
- 调用 `eval_easy_mirro.py`

如果没有显式设置 `OUTPUT_DIR`，结果不会写进 `offline-inference-easy-mirro/` 项目目录，而是写到父目录：

```text
offline_inference_output/<checkpoint名字>__<数据目录名字>
```

核心逻辑在：

```text
eval_easy_mirro.py
```

整体代码流程如下。

### 6.1 解析 checkpoint

函数：

```text
resolve_checkpoint()
load_checkpoint_config()
read_stats()
```

功能：

- 找到真实 checkpoint 目录。
- 找到 `dataset_stats.pkl`。
- 读取 `config.json`。
- 验证归一化统计量里没有 NaN/Inf。

### 6.2 解析相机和 action mode

函数：

```text
resolve_camera_names()
resolve_action_mode()
```

功能：

- 从 `image_features` 自动得到相机列表。
- 从 `action_mode` 或 `use_delta` 自动判断输出空间。

### 6.3 加载模型

函数：

```text
build_model()
```

功能：

- 把 `easy-mirro-dual-new_1` 加入 Python path。
- 注册 MIRRO 模型。
- 调用训练工程中的 `load_vla_model()`。
- 把 checkpoint 权重和 `dataset_stats.pkl` 加载进模型。
- 设置 `NormalizeMixed` 或 checkpoint 中指定的归一化方式。

### 6.4 读取评估数据

函数：

```text
find_hdf5_files()
load_episode()
read_qpos_actions()
decode_image()
```

功能：

- 找到评估目录下所有 `.hdf5` 文件。
- 读取 `/observations/qpos` 和 `/action`。
- 读取 checkpoint 要求的每个相机图像。
- 如果图像是压缩格式，先解码。

### 6.5 批量推理

函数：

```text
run_batched_inference()
```

对每个 episode 的每一帧 `t`：

1. 取 `qpos[t]` 作为输入 state。
2. 取第 `t` 帧的多相机图像。
3. 构造 batch：

```python
{
    "observation.state": state,
    "observation.images.cam_xxx": image,
    "task": language_instruction,
    "reasoning": None,
}
```

4. 调用：

```text
model.normalize_inputs()
model.prepare_images()
model.prepare_state()
model.prepare_language()
model.model.sample_actions()
model.normalize_targets.unnormalize()
```

5. 得到形状为：

```text
(episode_len, chunk_size, 20)
```

的模型预测。

### 6.6 构造 ground truth

函数：

```text
build_ground_truth_delta_chunks()
compute_action_delta()
build_trajectory_pairs()
```

对于每个时间步 `t`，都会构造一个 50 步 chunk。

如果 episode 后面不足 50 帧，则：

```text
valid_length < 50
```

指标计算时只比较有效部分，不比较填充的 0。

delta checkpoint：

```text
GT = inv(T_qpos[t]) @ T_action[t+k]
Pred = model output after unnormalize
Metric = Pred_delta vs GT_delta
Secondary absolute pred = T_qpos[t] @ Pred_delta
Secondary metric = Pred_absolute_from_delta vs /action[t:t+50]
```

absolute checkpoint：

```text
GT = /action[t:t+50]
Pred = model output after unnormalize
Metric = Pred_absolute vs GT_absolute
```

### 6.7 保存结果

函数：

```text
save_episode_output()
write_summary()
```

默认输出目录在项目目录外：

```text
../offline_inference_output/<checkpoint名字>__<数据目录名字>
```

这样 `offline-inference-easy-mirro/` 只保留代码、脚本、文档和工具，便于直接拷贝移植。

每个 episode 会保存：

```text
collection_xxx/
  trajectory_pairs.pkl
  metadata.pkl
  metadata.json
```

整个输出目录会保存：

```text
summary.txt
metrics.json
```

## 7. 输出文件解释

### 7.1 summary.txt

这是最适合快速查看的文件。

delta checkpoint 会显示：

```text
Primary comparison: model predicted delta action vs GT chunk-wise SE3 delta action
Average first-step delta MSE
Average valid-chunk delta MSE
Secondary comparison: delta prediction composed with input state into absolute action
Average valid-chunk absolute-from-delta MSE
```

absolute checkpoint 会显示：

```text
Primary comparison: model predicted absolute action vs GT HDF5 /action chunk
Average first-step absolute MSE
Average valid-chunk absolute MSE
```

`first-step` 指每个 state 预测出来的 chunk 中第 1 步。

`valid-chunk` 指所有有效未来步，也就是最多 50 步，但 episode 尾部不足 50 步的部分不参与统计。

### 7.2 metrics.json

这是机器可读的完整指标文件，适合写脚本做多 checkpoint 对比。

里面包含：

- checkpoint 路径
- dataset stats 路径
- camera 列表
- action mode
- 每条 episode 的指标
- 平均指标

### 7.3 trajectory_pairs.pkl

这是最重要的逐帧结果文件。每一项 pair 对应一个输入时间步 `t`。

公共字段：

```python
{
    "timestep": int,
    "observation_state": np.ndarray,         # qpos[t], shape (20,)
    "ground_truth_action_chunk": np.ndarray, # /action[t:t+50], shape (50, 20)
    "ground_truth_delta_chunk": np.ndarray,  # SE3 delta GT, shape (50, 20)
    "valid_length": int,
}
```

delta checkpoint 额外字段：

```python
"predicted_delta_chunk": np.ndarray
"ground_truth_chunk": np.ndarray
"predicted_chunk": np.ndarray             # T_qpos[t] @ predicted_delta_chunk
"predicted_action_chunk": np.ndarray
```

absolute checkpoint 额外字段：

```python
"ground_truth_chunk": np.ndarray
"predicted_chunk": np.ndarray
"predicted_action_chunk": np.ndarray
```

可视化脚本就是读取这些字段画图。

## 8. 常用运行命令

### 8.1 双相机 delta checkpoint

```bash
CHECKPOINT=/home/eai/debug/ckpt__fz/双相机相对输出/checkpoint-45000 \
EVAL_DATA_DIR=/home/eai/debug/rawdata \
OUTPUT_DIR=/home/eai/debug/offline_inference_output/2cam_delta_rawdata \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

### 8.2 三相机 delta checkpoint

```bash
CHECKPOINT=/home/eai/debug/ckpt__fz/3cam_se3_delta_mixed_output_45000_inference \
EVAL_DATA_DIR=/home/eai/debug/rawdata \
OUTPUT_DIR=/home/eai/debug/offline_inference_output/3cam_delta_rawdata \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

### 8.3 三相机 absolute checkpoint

```bash
CHECKPOINT=/home/eai/debug/ckpt__fz/3cam_absolute_mixed_output_45000_inference \
EVAL_DATA_DIR=/home/eai/debug/rawdata \
OUTPUT_DIR=/home/eai/debug/offline_inference_output/3cam_absolute_rawdata \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --strict
```

### 8.4 只跑少量帧做 smoke test

```bash
CHECKPOINT=/home/eai/debug/ckpt__fz/3cam_absolute_mixed_output_45000_inference \
EVAL_DATA_DIR=/home/eai/debug/rawdata \
OUTPUT_DIR=/home/eai/debug/offline_inference_output/smoke \
BATCH_SIZE=2 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh \
  --strict \
  --max-episodes 1 \
  --limit-frames 2
```

### 8.5 单手黑板任务

黑板任务使用 `easy-mirro-blackboard` 里的模型代码，因此建议用专门的 wrapper：

```bash
CHECKPOINT=/path/to/pi0_wiping_blackboard/checkpoint-45000 \
EVAL_DATA_DIR=/home/eai/debug/data/blackboard_testdata \
OUTPUT_DIR=/home/eai/debug/offline_inference_output/blackboard_ckpt45000 \
BATCH_SIZE=4 \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_blackboard.sh --strict
```

这个 wrapper 默认：

```text
EASY_MIRRO_ROOT=/home/eai/debug/easy-mirro-blackboard
TASK=auto
CAMERAS=auto
```

`TASK=auto` 会优先读取 HDF5 里的 `attrs["language_instruction"]`，黑板测试数据里已经有这个字段。

### 8.6 列出数据集里的 episode

```bash
CHECKPOINT=/home/eai/debug/ckpt__fz/3cam_absolute_mixed_output_45000_inference \
EVAL_DATA_DIR=/home/eai/debug/rawdata \
bash /home/eai/debug/offline-inference-easy-mirro/run_eval_easy_mirro.sh --list-episodes
```

## 9. 可视化

三种可视化都会默认显示当前帧的 HDF5 图像。图像来源不是重新随便找相机，而是根据输出目录中每个 episode 的 `metadata.json` / `trajectory_pairs.pkl`：

```text
source_hdf5_path   原始 HDF5 文件
cameras            checkpoint 推理实际使用的相机列表
```

所以可视化只显示模型实际输入过的相机，并且图像帧号和当前 `timestep` 对齐。切换 timestep 或 episode 时，3D 和波形图里的图像面板会同步刷新。如果只想看曲线不读取图像，可以加 `--no-images`。

交互式 3D 和波形图支持连续播放：

```text
P 或 Play 按钮   播放/暂停
--play-interval-ms 40  设置播放间隔，单位毫秒
```

长按 `SPACE`、`RIGHT`、`LEFT` 时，动作曲线和图像帧都会同步连续刷新。

### 9.1 静态 HTML 预览

```bash
python3 offline-inference-easy-mirro/render_preview.py \
  -d offline_inference_output/3cam_absolute_rawdata
```

输出：

```text
output_xxx/preview/index.html
output_xxx/preview/*_trajectory.png
output_xxx/preview/*_waveform.png
```

这个方式适合在服务器或没有 GUI 的环境中查看结果。trajectory PNG 下方会显示当前 snapshot 对应的 HDF5 当前帧图像，例如 `t=120` 的轨迹图会同时显示第 120 帧的模型输入相机画面。

### 9.2 交互式 3D 轨迹查看

```bash
python3 offline-inference-easy-mirro/visualize_episode.py \
  -d offline_inference_output/3cam_absolute_rawdata \
  --space absolute
```

右侧会显示当前 timestep 的模型输入相机帧。

delta 输出可以使用：

```bash
--space delta
```

默认也可以：

```bash
--space auto
```

它会根据 `trajectory_pairs.pkl` 里的字段自动选择。

### 9.3 交互式 20 维波形查看

```bash
python3 offline-inference-easy-mirro/visualize_waveform.py \
  -d offline_inference_output/3cam_absolute_rawdata \
  --space absolute
```

右侧图像面板会随当前 waveform chunk 的 timestep 同步更新。

快捷键：

```text
SPACE/RIGHT  下一帧
LEFT         上一帧
P/Play按钮   播放/暂停
UP/DOWN      切换 episode
Q            退出
```

如果在服务器上看到：

```text
FigureCanvasAgg is non-interactive
Maximum number of clients reached
```

说明当前环境没有可交互图形后端，或者 X server 客户端数量满了。这时优先用 `render_preview.py` 生成静态 HTML。

## 10. 对齐验证

脚本：

```text
validate_alignment.py
```

用途是确认离线评估的 ground truth 构造和训练代码完全一致。

运行：

```bash
python3 offline-inference-easy-mirro/validate_alignment.py \
  --data-dir rawdata \
  --output-dir offline_inference_output/3cam_absolute_mixed_output_45000_inference__rawdata \
  --chunk-size 50 \
  --require-training-layout
```

它会检查：

1. 离线评估中的 SE3 delta 是否等于训练 dataset 里的 `_compute_action_delta()`。
2. 离线评估中的 SE3 delta 是否等于归一化统计脚本里的 `compute_chunk_delta()`。
3. delta 转回 absolute 是否能还原 `/action`。
4. 如果输出目录里有 `trajectory_pairs.pkl`，检查保存的 GT delta 是否一致。
5. 如果是 absolute 输出，检查保存的 `ground_truth_chunk` 是否严格等于 HDF5 `/action[t:t+50]`。

正常情况下误差应该接近：

```text
0.000e+00 到 1e-7
```

如果这里不通过，说明离线评估逻辑和训练逻辑没有对齐，不应该相信推理指标。

## 11. 文件结构说明

```text
offline-inference-easy-mirro/
  README.md
  README_zh.md
  PROJECT_GUIDE_zh.md
  OFFLINE_INFERENCE_PI0_EASY_MIRRO_DETAILS.md
  OFFLINE_INFERENCE_PI0_EASY_MIRRO_DETAILS_zh.md

  run_eval_easy_mirro.sh
  eval_easy_mirro.py

  validate_alignment.py
  render_preview.py
  visualize_episode.py
  visualize_waveform.py
  compare_checkpoints.py
  run_eval_fz_test2_all.sh
```

推理结果默认放在项目父目录：

```text
offline_inference_output/
  <checkpoint名字>__<数据目录名字>/
    summary.txt
    metrics.json
    collection_xxx/
      trajectory_pairs.pkl
      metadata.pkl
      metadata.json
    preview/
      index.html
      *.png
```

主要文件职责：

```text
run_eval_easy_mirro.sh
  Shell 入口。设置环境变量和 PYTHONPATH，然后调用 eval_easy_mirro.py。

eval_easy_mirro.py
  核心离线推理和指标计算脚本。

validate_alignment.py
  检查离线 GT 构造是否与训练代码一致。

render_preview.py
  生成静态 PNG 和 index.html。

visualize_episode.py
  交互式 3D 轨迹查看器。

visualize_waveform.py
  交互式 20 维动作波形查看器。

compare_checkpoints.py
  对多个 checkpoint 输出目录做指标对比。

run_eval_fz_test2_all.sh
  批量评估脚本，保留用于已有 Fangzhou checkpoint 对比流程。
```

## 12. 指标解读注意事项

### 12.1 first-step 与 valid-chunk

`first-step`：

```text
每个 state 预测出的未来 50 步 chunk 中第 1 步
```

它更接近“当前帧下一个动作是否准确”。

`valid-chunk`：

```text
每个 state 预测出的未来最多 50 步动作整体
```

它更能反映模型是否能预测较长 horizon 的动作趋势。

### 12.2 gripper 会主导总 MSE

夹爪通常是 0 到 80 左右的连续值，而 xyz 和旋转 delta 的数值尺度小很多。因此直接看 20 维平均 MSE 时，gripper 维度可能占主导。

如果要细分析，需要按维度拆开看：

```text
L_xyz, L_rot6d, L_gripper
R_xyz, R_rot6d, R_gripper
```

不要只凭一个总 MSE 判断位姿预测一定好或不好。

### 12.3 delta 和 absolute 指标不能直接横向比较

delta 模型的 MSE 是在 SE3 delta 空间里算的。

absolute 模型的 MSE 是在绝对 `/action` 空间里算的。

两者数值尺度不同，不能简单说：

```text
absolute MSE 小于 delta MSE，所以 absolute 一定更好
```

正确比较方式是：

- 同 action mode 的 checkpoint 之间可以直接比较。
- 不同 action mode 需要结合轨迹图、波形图、上机成功率，以及按维度误差分析。

### 12.4 离线指标不完全等于真实上机成功率

离线推理使用的是已有 HDF5 中的图像和 state。对于 UMI 采集数据，顶部相机可能看到人手拿夹爪；真实上机时顶部相机看到机械臂。这种视觉 gap 不一定能在离线指标里完全体现。

因此离线评估能确认：

- 推理代码是否对齐训练。
- 模型在数据分布内的预测是否合理。
- 不同 checkpoint 的相对趋势。

但最终成功率仍然需要真实机器人闭环测试确认。

## 13. 常见错误和排查

### 13.1 HDF5 缺少 `/observations/qpos` 或 `/action`

报错含义：

```text
Expected /observations/qpos and /action
```

说明数据不是训练布局。需要先用训练时相同的预处理脚本转成 processed HDF5。

### 13.2 缺少某个相机

报错含义：

```text
Episode is missing camera images required by the model
```

说明 checkpoint 需要的 camera 在 HDF5 中不存在。要么数据不匹配，要么手动 `CAMERAS=` 指错了。

### 13.3 action mode 不一致

如果 `config.json` 和 `dataset_stats.pkl` 一个说 delta，一个说 absolute，程序会拒绝推理。这种情况下需要检查 checkpoint 和 stats 是否来自同一次训练。

### 13.4 离线指标异常差

优先检查：

1. 是否用错 checkpoint 的 `dataset_stats.pkl`。
2. 是否把 delta 输出当 absolute 比较，或把 absolute 输出当 delta 比较。
3. 是否用简单减法构造了 SE3 delta。
4. 是否使用了 raw fallback，而不是训练布局数据。
5. 相机数量和顺序是否与 checkpoint `image_features` 一致。
6. HDF5 的 `/action` 是否就是训练时用的 future action。

## 14. 这个项目当前最重要的保证

当前版本的离线推理有几个关键保证：

1. 归一化只使用 checkpoint 对应的 `dataset_stats.pkl`。
2. state 输入是 `/observations/qpos[t]` 的绝对 20 维位姿。
3. delta checkpoint 使用 SE3 矩阵乘法构造 GT delta。
4. absolute checkpoint 直接使用 HDF5 `/action[t:t+50]` 构造 GT。
5. 相机默认从 checkpoint config 自动读取。
6. 缺失相机、action mode 冲突、数据布局不匹配都会显式报错。
7. `validate_alignment.py` 可以验证离线 GT 与训练代码是否一致。

这些保证的目的，是避免离线评估再次出现“模型看起来很差，其实是评估逻辑错了”的情况。
