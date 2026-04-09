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
