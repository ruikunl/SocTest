package com.rockenmini.socbenchmark.benchmark

enum class ComputeBackend {
    CPU,
    GPU,
    NNAPI;

    val displayName: String
        get() = when (this) {
            CPU -> "CPU"
            GPU -> "GPU"
            NNAPI -> "NPU / NNAPI"
        }
}

data class BenchmarkConfig(
    val backend: ComputeBackend,
    val warmupRuns: Int = 3,
    val measuredRuns: Int = 20,
    val threads: Int = 4
)

data class BenchmarkResult(
    val backend: ComputeBackend,
    val warmupRuns: Int,
    val measuredRuns: Int,
    val averageMs: Double,
    val minMs: Double,
    val maxMs: Double,
    val allRunsMs: List<Double>,
    val note: String? = null
)

fun interface TimedWork {
    fun runOnce()
}

data class TaskModelSpec(
    val displayName: String,
    val assetPath: String
)

enum class TensorLayout {
    NCHW,
    NHWC
}

data class PoseModelSpec(
    val displayName: String,
    val assetPath: String,
    val inputWidth: Int,
    val inputHeight: Int,
    val inputLayout: TensorLayout,
    val keypointCount: Int,
    val simccXWidth: Int,
    val simccYWidth: Int,
    val simccSplitRatio: Float,
    val scoreThreshold: Float,
    val meanRgb: FloatArray,
    val stdRgb: FloatArray
)

object BenchmarkModels {
    // This spec intentionally captures the semantic parameters that should stay aligned across
    // ONNX, ncnn and TFLite. The only expected runtime-specific difference is tensor layout:
    // TFLite here uses NHWC, while the ncnn bridge feeds the same normalized RGB values as NCHW.
    val pose = PoseModelSpec(
        displayName = "RTMPose",
        assetPath = "models/rtmpose_t_body7_256x192_float32.tflite",
        inputWidth = 192,
        inputHeight = 256,
        inputLayout = TensorLayout.NHWC,
        keypointCount = 17,
        simccXWidth = 384,
        simccYWidth = 512,
        simccSplitRatio = 2.0f,
        scoreThreshold = 0.35f,
        meanRgb = floatArrayOf(123.675f, 116.28f, 103.53f),
        stdRgb = floatArrayOf(58.395f, 57.12f, 57.375f)
    )

    val segmentation = TaskModelSpec(
        displayName = "MediaPipe Selfie Segmentation",
        assetPath = "models/mediapipe_selfie_segmentation_float32.tflite"
    )
}
