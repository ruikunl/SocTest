# SocTest

`SocTest` 是一个 Android 端 SoC benchmark 应用，用来快速评估平板或手机芯片是否能支撑按摩机器人项目需要的离线视觉任务。

当前 `onnx` 分支重点验证：

- 人体关键点检测
- 人体分割
- 不同 SoC 之间的耗时对比
- Android 上 ONNX Runtime 的 CPU / NNAPI 路径

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

当前分支内置模型：

- 人体关键点：`RTMPose-t` ONNX，文件名为 `rtmpose_t_body7_256x192.onnx`
- 人体分割：`MediaPipe Selfie Segmentation` ONNX
  - `mediapipe_selfie_segmentation.onnx`
  - `mediapipe_selfie.data`

说明：

- 关键点路径是真实的 ONNX Runtime 推理。
- 当前关键点路径是偏 benchmark 的 top-down 简化方案：先按模型比例对输入图做中心裁剪，再 resize 到模型输入尺寸。它适合当前 SoC 验证阶段中“人物通常居中”的测试场景，但还不是最终量产算法 pipeline。
- 分割路径是真实的 ONNX Runtime 推理，使用轻量级 selfie segmentation 模型。
- 分割 ONNX 模型依赖外部 `.data` 文件，所以应用会先把 ONNX 和 `.data` 文件都释放到本地存储，再创建 ONNX Runtime session。

## 后端路径

当前分支使用 `ONNX Runtime` 作为推理运行时。

后端映射：

- `CPU`：ONNX Runtime + `XNNPACK`
- `GPU`：界面保留这个选项，方便和其他分支保持相同测试流程；但当前使用的 baseline `onnxruntime-android` 包不包含通用 Android GPU execution provider，所以目前该路径会回退到 `XNNPACK CPU`
- `NPU / NNAPI`：请求 ONNX Runtime 的 `NNAPI` execution provider

注意：

- `NNAPI` 不保证一定是真正的 NPU 执行。根据设备驱动、图切分和算子覆盖情况，Android 可能会部分加速，也可能静默回退到 CPU。
- 当前 ORT baseline 更适合验证 `CPU vs NNAPI`，不适合作为通用 Android GPU benchmark 路线。

## 技术栈

- Android
- Kotlin
- Jetpack Compose
- ONNX Runtime Android
- XNNPACK via ONNX Runtime
- NNAPI via ONNX Runtime

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

- `Preprocess`：resize、crop 或 letterbox、tensor packing、输入准备。
- `Inference`：纯 ONNX Runtime session 执行耗时。
- `Postprocess`：输出 tensor 解码到绘制前结果。
- `Overlay Render`：把 mask、关键点和标签绘制到可显示 bitmap 的耗时。
- `Total`：benchmark pipeline 内预处理、推理、后处理的端到端耗时。

`Overlay Render` 单独统计，方便区分模型推理慢还是可视化绘制慢。

## 当前限制

- 当前分支的 `GPU` 路径会回退到 CPU，因为 baseline ORT 包不包含通用 Android GPU EP。
- `NNAPI` 可能根据设备情况回退到 CPU，或只加速图中的一部分。
- 当前关键点路径假设人物居中，尚未接入人体检测器。
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
   - `NNAPI` 是否真的改善延迟
   - 当前 `GPU` 选择回退到 CPU 后与 CPU 路径的差异

## 分支说明

- `main` 和 `TFLite` 分支：TensorFlow Lite 路线，用于 CPU / GPU delegate / NNAPI 实验。
- `onnx` 分支：ONNX Runtime 路线，更贴近 PyTorch -> ONNX 的算法开发和部署流程。

## 仓库状态

这个仓库仍处于快速演进阶段，目标是做实际硬件评估工具，还不是正式产品版本。

后续可能继续补充：

- 更适合 ONNX 路线的人体分割模型
- 更可靠的 NNAPI 诊断信息
- warmup / repeat count 控制
- 更完整的点云 benchmark
- 更完善的报告和结果聚合工具
