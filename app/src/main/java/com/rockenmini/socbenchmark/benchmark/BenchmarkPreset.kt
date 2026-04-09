package com.rockenmini.socbenchmark.benchmark

enum class BenchmarkPreset(
    val title: String,
    val summary: String
) {
    SEGMENTATION(
        title = "Human Segmentation",
        summary = "Prepare CPU/GPU/NNAPI timing for a lightweight body segmentation model."
    ),
    POSE(
        title = "Human Keypoints",
        summary = "Prepare CPU/GPU/NNAPI timing for a lightweight pose estimation model."
    ),
    POINT_CLOUD(
        title = "Point Cloud CPU",
        summary = "Reserve a small CPU-only benchmark for point cloud preprocessing and geometry ops."
    )
}

