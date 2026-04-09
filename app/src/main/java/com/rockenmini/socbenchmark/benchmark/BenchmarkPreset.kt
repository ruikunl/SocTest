package com.rockenmini.socbenchmark.benchmark

enum class BenchmarkPreset(
    val title: String,
    val summary: String
) {
    SEGMENTATION(
        title = "Human Segmentation",
        summary = "Benchmark CPU / GPU(selection) / NNAPI paths for the current ONNX Runtime person segmentation model."
    ),
    POSE(
        title = "Human Keypoints",
        summary = "Benchmark CPU / GPU(selection) / NNAPI paths for the current ONNX Runtime human keypoint model."
    ),
    POINT_CLOUD(
        title = "Point Cloud CPU",
        summary = "Reserve a small CPU-only benchmark for point cloud preprocessing and geometry ops."
    )
}
