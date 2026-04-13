package com.rockenmini.socbenchmark.benchmark

enum class BenchmarkPreset(
    val title: String,
    val summary: String
) {
    SEGMENTATION(
        title = "Human Segmentation",
        summary = "Benchmark CPU / Vulkan GPU / NNAPI-fallback paths for the current ncnn person segmentation model."
    ),
    POSE(
        title = "Human Keypoints",
        summary = "Benchmark CPU / Vulkan GPU / NNAPI-fallback paths for the current ncnn human keypoint model."
    ),
    POINT_CLOUD(
        title = "Point Cloud CPU",
        summary = "Reserve a small CPU-only benchmark for point cloud preprocessing and geometry ops."
    )
}
