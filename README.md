# SocTest

`SocTest` 是一个 Android 端 SoC benchmark 应用，用来快速评估平板或手机芯片是否能支撑按摩机器人项目需要的离线视觉任务。

当前 `TFLite` 分支重点验证：

- 人体关键点检测
- 人体分割
- 不同 SoC 之间的耗时对比
- Android 上 TensorFlow Lite 的 CPU / GPU / NNAPI 路径

这个应用主要用于在 Qualcomm、MediaTek Dimensity、MediaTek G 系列等平台上做真实设备测试。

## 功能

应用支持：

- 选择后端：`CPU`、`GPU`、`NPU / NNAPI`
- 选择任务预设
- 选择单张图片或整个图片文件夹
- 对单张图片或批量图片运行推理
- 在界面中查看渲染后的结果图
- 记录分阶段耗时：
  - 预处理
  - 推理
  - 后处理
  - 结果叠加渲染
  - 总耗时
- 导出批量测试 CSV
- 为每张输入图保存渲染结果
- 在导出结果中记录设备和 SoC 元数据

## 当前模型

当前内置模型：

- 人体关键点：`RTMPose` 的 TFLite 转换版，文件名为 `rtmpose_t_body7_256x192_float32.tflite`
- 人体分割：`MediaPipe Selfie Segmentation` 的 ONNX 资产转换版，文件名为 `mediapipe_selfie_segmentation_float32.tflite`

说明：

- 关键点路径是真实的 TensorFlow Lite 推理。
- 分割路径是真实的 TensorFlow Lite 推理，模型来源与 `ncnn` / `onnx` 分支保持一致，只是这里额外经过 `ONNX -> TFLite` 转换后接入 Android benchmark。
- 当前接入的是无额外自定义算子的 plain TFLite 文件，可继续用于 `CPU`、`GPU`、`NNAPI` 路径验证。

## 后端路径

当前分支使用 `TensorFlow Lite` 作为推理运行时。

后端映射：

- `CPU`：标准 TFLite interpreter
- `GPU`：TFLite GPU delegate
- `NPU / NNAPI`：TFLite NNAPI 路径

注意：

- `NNAPI` 不保证一定是真正的 NPU 执行。根据设备驱动支持和模型算子覆盖情况，Android 可能会部分加速，也可能静默回退到 CPU。
- GPU 兼容性会受芯片厂商、驱动和 Android 版本影响。

## 技术栈

- Android
- Kotlin
- Jetpack Compose
- TensorFlow Lite
- TFLite GPU delegate
- NNAPI via TFLite

## 环境要求

- Android `minSdk 28`
- Android Studio 或命令行 Android SDK 环境
- JDK 17
- Gradle 8.7

## 构建

在 SocTest 仓库根目录执行：

```bash
./gradlew :app:assembleDebug
```

Debug APK 输出路径：

```text
app/build/outputs/apk/debug/app-debug.apk
```

## 安装到设备

使用 `adb`：

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

也可以手动把 APK 传到设备上安装。

## 使用方式

1. 打开应用。
2. 选择后端：
   - `CPU`
   - `GPU`
   - `NPU / NNAPI`
3. 选择任务预设：
   - `Human Keypoints`
   - `Human Segmentation`
   - `Point Cloud CPU`
4. 选择输入：
   - 单张图片
   - 图片文件夹
5. 点击 `Run Benchmark`。

运行结束后：

- 界面会展示第一张结果图
- 批量结果图会自动保存
- CSV 会自动导出
- 应用内会展示运行信息和批量统计结果

## 输出文件

每次批量测试会在应用 external files 目录下创建一个结果文件夹。

文件夹名称包含：

- 任务
- 后端
- 时间戳

示例：

```text
human_keypoints_gpu_20260409_123456/
```

结果文件夹中包含：

- 每张输入图对应的渲染结果图
- 一个包含耗时和设备元数据的 CSV 文件

渲染图命名：

```text
original_name_rendered.png
```

CSV 命名：

```text
human_keypoints_gpu_20260409_123456.csv
```

典型 Android 设备路径：

```text
/storage/emulated/0/Android/data/com.rockenmini.socbenchmark/files/renders/<batch-folder>/
```

## CSV 内容

导出的 CSV 包含：

1. 设备元数据
   - manufacturer
   - brand
   - model
   - device
   - product
   - hardware
   - board
   - Android version
   - SDK level
   - SoC manufacturer
   - SoC model

2. benchmark 元数据
   - task
   - backend
   - input mode
   - source count

3. 单图结果
   - file name
   - resolution
   - total time
   - preprocess time
   - inference time
   - postprocess time
   - overlay render time

4. 末尾统计行
   - `avg`
   - `max`
   - `min`

统计行会分别统计：

- total
- preprocess
- inference
- postprocess
- overlay render

## 耗时解释

应用会把耗时拆成多个阶段：

- `Preprocess`：resize、tensor packing、输入准备。
- `Inference`：纯 TFLite interpreter 执行耗时。
- `Postprocess`：输出 tensor 解码到绘制前结果。
- `Overlay Render`：把 mask、关键点和标签绘制到可显示 bitmap 的耗时。
- `Total`：benchmark pipeline 内预处理、推理、后处理的端到端耗时。

`Overlay Render` 单独统计，方便区分模型推理慢还是可视化绘制慢。

## 当前限制

- `NNAPI` 可能根据设备情况回退到 CPU。
- 当前分割模型不是针对人像边界优化的高质量模型。
- 应用当前聚焦图片 benchmark。
- 点云处理仍是 preview / placeholder，还不是完整的生产级点云 benchmark。

## 建议测试流程

对每台目标设备：

1. 使用同一批测试图片。
2. 分别运行：
   - keypoints on `CPU`
   - keypoints on `GPU`
   - keypoints on `NNAPI`
   - segmentation on `CPU`
   - segmentation on `GPU`
   - segmentation on `NNAPI`
3. 收集导出的 batch 文件夹。
4. 对比：
   - 平均推理耗时
   - 最慢单张耗时
   - 预处理开销
   - 叠加渲染开销
   - GPU / NNAPI 是否真的改善延迟

## 当前测试结果汇总

当前已整理 `data/soc_test_result/TFLite` 下的 12 份 CSV 结果，覆盖：

- 3 个平台：`6300`、`680`、`G100-ultra`
- 2 个任务：`Human Keypoints`、`Human Segmentation`
- 2 个后端：`CPU`、`GPU`
- 每组测试样本数：`3031`

整体结论：

- `6300` 在四组对比里全部排名第 1，整体性能最好。
- `680` 在四组对比里全部排名最后，整体性能最弱。
- `G100-ultra` 整体居中。
- 这批 TFLite 测试中，`GPU` 没有明显优于 `CPU`，多数场景下反而更慢；只有 `6300` 的分割任务上，`GPU` 与 `CPU` 基本持平。

### Human Keypoints - CPU

| Rank | Platform | Device | Avg Total (ms) | Inference (ms) | FPS |
|---|---|---|---:|---:|---:|
| 1 | `6300` | Lenovo TB335FC | 46.12 | 30.24 | 21.68 |
| 2 | `G100-ultra` | Redmi 25040RP0AC | 66.74 | 47.74 | 14.98 |
| 3 | `680` | HONOR NDL2-W09 | 102.55 | 71.87 | 9.75 |

### Human Keypoints - GPU

| Rank | Platform | Device | Avg Total (ms) | Inference (ms) | FPS |
|---|---|---|---:|---:|---:|
| 1 | `6300` | Lenovo TB335FC | 51.42 | 35.85 | 19.45 |
| 2 | `G100-ultra` | Redmi 25040RP0AC | 78.85 | 59.52 | 12.68 |
| 3 | `680` | HONOR NDL2-W09 | 106.56 | 75.51 | 9.38 |

### Human Segmentation - CPU

| Rank | Platform | Device | Avg Total (ms) | Inference (ms) | FPS |
|---|---|---|---:|---:|---:|
| 1 | `6300` | Lenovo TB335FC | 52.95 | 36.60 | 18.89 |
| 2 | `G100-ultra` | Redmi 25040RP0AC | 62.99 | 43.94 | 15.88 |
| 3 | `680` | HONOR NDL2-W09 | 69.48 | 34.78 | 14.39 |

### Human Segmentation - GPU

| Rank | Platform | Device | Avg Total (ms) | Inference (ms) | FPS |
|---|---|---|---:|---:|---:|
| 1 | `6300` | Lenovo TB335FC | 52.60 | 36.48 | 19.01 |
| 2 | `G100-ultra` | Redmi 25040RP0AC | 66.73 | 47.82 | 14.98 |
| 3 | `680` | HONOR NDL2-W09 | 77.05 | 39.82 | 12.98 |

### CPU vs GPU

| Platform | Task | CPU Avg Total (ms) | GPU Avg Total (ms) | GPU vs CPU |
|---|---|---:|---:|---|
| `6300` | Human Keypoints | 46.12 | 51.42 | GPU slower by 11.5% |
| `6300` | Human Segmentation | 52.95 | 52.60 | GPU faster by 0.7% |
| `680` | Human Keypoints | 102.55 | 106.56 | GPU slower by 3.9% |
| `680` | Human Segmentation | 69.48 | 77.05 | GPU slower by 10.9% |
| `G100-ultra` | Human Keypoints | 66.74 | 78.85 | GPU slower by 18.1% |
| `G100-ultra` | Human Segmentation | 62.99 | 66.73 | GPU slower by 5.9% |

从阶段耗时上看，总耗时差异主要由 `inference_ms` 决定；`Human Segmentation` 的 `overlay_render_ms` 也明显高于 `Human Keypoints`。

## 6300 稳定性分析

`6300` 的 `CPU/GPU` 稳定性可以分成两类来看：`Human Keypoints` 两个后端都比较稳，`Human Segmentation` 中 GPU 更稳，而 CPU 存在明显长尾。

### 波动统计

| Task | Backend | Mean (ms) | Std (ms) | CV | P50 (ms) | P90 (ms) | P99 (ms) | Max (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Human Keypoints | CPU | 46.12 | 5.83 | 12.64% | 43.60 | 53.93 | 63.44 | 99.42 |
| Human Keypoints | GPU | 51.42 | 6.34 | 12.33% | 48.00 | 62.42 | 70.13 | 89.94 |
| Human Segmentation | CPU | 52.95 | 11.73 | 22.14% | 50.81 | 56.78 | 61.59 | 330.81 |
| Human Segmentation | GPU | 52.60 | 3.58 | 6.80% | 51.14 | 57.63 | 63.18 | 91.71 |

可以直接据此判断：

- `Human Keypoints` 上，CPU 和 GPU 的波动水平接近，但 CPU 更快。
- `Human Segmentation` 上，GPU 的耗时分布明显更紧，稳定性优于 CPU。
- `Human Segmentation CPU` 的平均值并不差，但存在异常长尾样本，导致最大耗时远高于常规区间。

### 长尾占比

| Task | Backend | >1.5x Mean | >2.0x Mean | >Mean+3σ |
|---|---|---:|---:|---:|
| Human Keypoints | CPU | 4 (`0.13%`) | 1 (`0.03%`) | 27 (`0.89%`) |
| Human Keypoints | GPU | 4 (`0.13%`) | 0 | 24 (`0.79%`) |
| Human Segmentation | CPU | 12 (`0.40%`) | 12 (`0.40%`) | 12 (`0.40%`) |
| Human Segmentation | GPU | 1 (`0.03%`) | 0 | 23 (`0.76%`) |

补充说明：

- `Human Keypoints` 的慢样本主要和大分辨率图片、`preprocess_ms` 突增有关。
- `Human Segmentation CPU` 的异常长尾主要来自 `inference_ms` 和 `overlay_render_ms` 在个别样本上的明显抖动。
- 如果关注端到端时延一致性，`6300` 上的分割任务优先考虑 `GPU`；如果关注纯速度，关键点任务优先选择 `CPU`。

## 分支说明

- `TFLite` 分支用于 TensorFlow Lite 路线实验。
- `onnx` 分支用于 ONNX Runtime 路线实验，更贴近 PyTorch -> ONNX 的算法开发和部署流程。

## 仓库状态

这个仓库仍处于快速演进阶段，目标是做实际硬件评估工具，还不是正式产品版本。

后续可能继续补充：

- 更好的人像分割模型
- 更可靠的 NNAPI 诊断信息
- warmup / repeat count 控制
- 更完整的点云 benchmark
- 更完善的报告和结果聚合工具
