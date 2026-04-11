# SocTest

`SocTest` is an Android benchmark app for quickly evaluating whether a tablet or phone SoC can support offline vision workloads needed by a robotics project.

This `onnx` branch focuses on:
- human keypoint detection
- person segmentation
- cross-device timing comparison on different SoCs
- ONNX Runtime CPU / NNAPI path validation on Android

The app is intended for practical device-side testing on Qualcomm, MediaTek Dimensity, and MediaTek G-series platforms.

## What It Does

The app lets you:
- choose a backend: `CPU`, `GPU`, or `NPU / NNAPI`
- choose a task preset
- select a single image or a whole image folder
- run inference on one image or batch-process a folder
- visualize the rendered result image
- record timing breakdowns for:
  - preprocess
  - inference
  - postprocess
  - overlay render
  - total
- export batch results as CSV
- save rendered result images for every input
- include device and SoC metadata in exported results

## Current Models

Current built-in models on this branch:
- Human keypoints: `RTMPose-t` ONNX (`rtmpose_t_body7_256x192.onnx`)
- Person segmentation: `MediaPipe Selfie Segmentation` ONNX
  - `mediapipe_selfie_segmentation.onnx`
  - `mediapipe_selfie.data`

Notes:
- The keypoint path is real ONNX Runtime inference.
- The current keypoint path uses a benchmark-oriented top-down approximation:
  the input image is center-cropped to the model aspect ratio and then resized.
  This is suitable for current SoC validation where the subject is typically centered,
  but it is not the final production algorithm pipeline.
- The segmentation path is real ONNX Runtime inference with a lightweight selfie segmentation model.
- The segmentation ONNX asset uses an external `.data` file, so the app extracts both model files to local storage before session creation.

## Backend Paths

This branch uses `ONNX Runtime` as the inference runtime.

Current backend mapping:
- `CPU`: ONNX Runtime + `XNNPACK`
- `GPU`: UI option is kept for comparison flow, but the baseline `onnxruntime-android` package used here does not bundle a generic Android GPU execution provider, so this path currently falls back to `XNNPACK CPU`
- `NPU / NNAPI`: ONNX Runtime `NNAPI` execution provider request path

Important:
- `NNAPI` does not guarantee real NPU execution. Depending on device driver support, graph partitioning, and op coverage, Android may partially accelerate or silently fall back to CPU.
- The current ORT baseline build is useful for `CPU vs NNAPI` validation, but it is not a general Android GPU benchmark path.

## Tech Stack

- Android
- Kotlin
- Jetpack Compose
- ONNX Runtime Android
- XNNPACK via ONNX Runtime
- NNAPI via ONNX Runtime

## Requirements

- Android `minSdk 28`
- Android Studio or command-line Android SDK setup
- JDK 17
- Gradle 8.7

## Build

From the SocTest repository root:

```bash
./gradlew :app:assembleDebug
```

Debug APK output:

```text
app/build/outputs/apk/debug/app-debug.apk
```

## Install On Device

With `adb`:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Or transfer the APK manually and install it from the device.

## How To Use

1. Open the app.
2. Select a backend:
   - `CPU`
   - `GPU`
   - `NPU / NNAPI`
3. Select a task preset:
   - `Human Keypoints`
   - `Human Segmentation`
   - `Point Cloud CPU`
4. Select either:
   - a single image
   - an image folder
5. Press `Run Benchmark`.

After a run:
- the first result image is shown in the UI
- all batch result images are saved automatically
- the CSV is exported automatically
- runtime information and batch statistics are shown in the app

## Output Files

Each batch creates one folder under the app's external files directory.

The folder name includes:
- task
- backend
- timestamp

Example:

```text
human_keypoints_gpu_20260409_123456/
```

Inside that folder you will find:
- rendered images for each input
- one CSV file with timing and device metadata

Rendered image naming:

```text
original_name_rendered.png
```

CSV naming:

```text
human_keypoints_gpu_20260409_123456.csv
```

Typical Android device path:

```text
/storage/emulated/0/Android/data/com.rockenmini.socbenchmark/files/renders/<batch-folder>/
```

## CSV Contents

The exported CSV includes:

1. device metadata
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

2. benchmark metadata
   - task
   - backend
   - input mode
   - source count

3. per-image results
   - file name
   - resolution
   - total time
   - preprocess time
   - inference time
   - postprocess time
   - overlay render time

4. summary rows at the end
   - `avg`
   - `max`
   - `min`

These summary rows are provided for:
- total
- preprocess
- inference
- postprocess
- overlay render

## Timing Interpretation

The app splits timing into separate stages:

- `Preprocess`
  Resize / crop or letterbox / tensor packing / input preparation.

- `Inference`
  Raw ONNX Runtime session execution only.

- `Postprocess`
  Output tensor decoding before drawing.

- `Overlay Render`
  Time spent drawing masks, keypoints, and labels onto a displayable bitmap.

- `Total`
  End-to-end time across preprocess, inference, and postprocess inside the benchmark pipeline.

Note:
- Overlay rendering is tracked separately so you can tell whether slow results come from the model itself or from visualization work.

## Current Limitations

- `GPU` currently falls back to CPU on this branch because no generic Android GPU EP is bundled in the baseline ORT package used here.
- `NNAPI` may fall back to CPU or partition the graph depending on the device.
- The current keypoint path assumes a centered person and does not yet include a detector.
- The app is focused on image-based benchmarking for now.
- Point cloud processing is still a preview / placeholder path rather than a full production point-cloud benchmark.

## Suggested Workflow For SoC Evaluation

For each target device:

1. Use the same test image folder.
2. Run:
   - keypoints on `CPU`
   - keypoints on `GPU`
   - keypoints on `NNAPI`
   - segmentation on `CPU`
   - segmentation on `GPU`
   - segmentation on `NNAPI`
3. Export and collect the batch folders.
4. Compare:
   - average inference time
   - worst-case time
   - preprocess overhead
   - overlay cost
   - whether `NNAPI` actually improves latency
   - how much the current `GPU` selection falls back compared with CPU

## Branch Notes

- `main` and `TFLite` branch:
  TensorFlow Lite based implementation for CPU / GPU delegate / NNAPI experiments.
- `onnx` branch:
  ONNX Runtime based implementation intended to align more closely with a PyTorch to ONNX workflow.

## Repository Status

This repository is an actively evolving benchmark tool intended for practical hardware evaluation, not yet a polished release product.

Planned next steps may include:
- better ONNX-native segmentation models
- more robust NNAPI diagnostics
- warmup / repeat count controls
- richer point cloud benchmarks
- improved reporting and result aggregation
