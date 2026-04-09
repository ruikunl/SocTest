package com.rockenmini.socbenchmark.benchmark

import kotlin.math.round
import kotlin.system.measureNanoTime

object BenchmarkTimer {
    fun measure(config: BenchmarkConfig, work: TimedWork, note: String? = null): BenchmarkResult {
        repeat(config.warmupRuns) {
            work.runOnce()
        }

        val samples = buildList {
            repeat(config.measuredRuns) {
                val elapsedNs = measureNanoTime {
                    work.runOnce()
                }
                add(elapsedNs / 1_000_000.0)
            }
        }

        val avg = samples.average().rounded(3)
        val min = samples.minOrNull()?.rounded(3) ?: 0.0
        val max = samples.maxOrNull()?.rounded(3) ?: 0.0

        return BenchmarkResult(
            backend = config.backend,
            warmupRuns = config.warmupRuns,
            measuredRuns = config.measuredRuns,
            averageMs = avg,
            minMs = min,
            maxMs = max,
            allRunsMs = samples.map { it.rounded(3) },
            note = note
        )
    }
}

private fun Double.rounded(scale: Int): Double {
    val factor = buildFactor(scale)
    return round(this * factor) / factor
}

private fun buildFactor(scale: Int): Double {
    var factor = 1.0
    repeat(scale) {
        factor *= 10.0
    }
    return factor
}
