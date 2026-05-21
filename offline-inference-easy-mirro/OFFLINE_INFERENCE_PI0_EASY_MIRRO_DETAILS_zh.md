# Pi0 EasyMirror 离线推理细节

## 契约

评估器与 `easy-mirro-dual-new_1` Pi0 训练保持一致：

- `observation.state`：20 维 frame-0 绝对双臂位姿，填充到 32 维。
- 模型目标/输出：20 维按 chunk 表示的 SE3 delta，内部填充到 32 维。
- chunk size：从 checkpoint 配置加载，预期为 `50`。
- 归一化：使用 `NormalizeMixed`，并从 checkpoint 目录或其父目录读取 `dataset_stats.pkl`。

## 动作标签

对于帧 `t`，ground-truth action chunk 会先在绝对 frame-0 坐标中构造：

```text
absolute_chunk = action[t : t + chunk_size]
```

然后按训练时完全相同的方式转换：

```text
T_delta[k] = inv(T_qpos[t]) @ T_action[t + k]
delta_xyz[k] = T_delta[k][:3, 3]
delta_rot6d[k] = concat(T_delta[k][0, :3], T_delta[k][1, :3])
delta_gripper[k] = action_gripper[t + k]
```

真正用于比较的是：

```text
predicted_delta_chunk <-> ground_truth_delta_chunk
```

严格评估器打分时不会使用绝对空间重建结果。

## 标准数据要求

严格评估要求使用与训练相同的已处理 HDF5 布局：

```text
/observations/qpos   (N, 20)
/action              (N, 20)
```

`rawdata` 具备该布局。`test2` 原始采集文件不具备该布局，因此默认会被拒绝，而不是用另一条路径静默转换。

`--allow-raw-fallback` 仅作为旧版/调试路径存在，不应在严格 checkpoint 评估中使用。

## 验证

运行：

```bash
python3 offline-inference-easy-mirro/validate_alignment.py \
  --data-dir rawdata \
  --chunk-size 50 \
  --require-training-layout
```

以下项目的预期容差约为 `1e-7`：

- 评估器 delta 与训练数据集 delta
- 评估器 delta 与归一化统计 delta
- delta 到 absolute 的往返转换
- 提供 `--output-dir` 时，已保存 ground-truth delta 与重新计算 delta

## 完整 Fangzhou 评估

```bash
bash offline-inference-easy-mirro/run_eval_fz_test2_all.sh
```

脚本名称为了兼容性保留，但默认数据目录现在是 `rawdata`。输出：

```text
offline_inference_output/fz_rawdata/checkpoint-35000/
offline_inference_output/fz_rawdata/checkpoint-45000/
offline_inference_output/fz_rawdata/comparison.json
```

打开：

```text
offline_inference_output/fz_rawdata/checkpoint-45000/preview/index.html
```

预览会渲染所有已保存的动作空间。delta checkpoint 现在同时包含 delta 和重建后的 absolute 图，xyz 默认以 dm 显示。
