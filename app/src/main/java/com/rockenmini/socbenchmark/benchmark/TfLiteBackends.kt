package com.rockenmini.socbenchmark.benchmark

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.io.Closeable

data class TfLiteSession(
    val interpreter: Interpreter,
    val backend: ComputeBackend,
    val note: String,
    private val closeables: List<Closeable> = emptyList()
) : Closeable {
    override fun close() {
        interpreter.close()
        closeables.forEach { it.close() }
    }
}

object TfLiteBackends {
    fun createSession(
        context: Context,
        modelAssetPath: String,
        config: BenchmarkConfig
    ): TfLiteSession {
        val model = FileUtil.loadMappedFile(context, modelAssetPath)
        val options = Interpreter.Options().apply {
            setNumThreads(config.threads)
        }

        return when (config.backend) {
            ComputeBackend.CPU -> {
                options.setUseNNAPI(false)
                TfLiteSession(
                    interpreter = Interpreter(model, options),
                    backend = config.backend,
                    note = "CPU delegate active with ${config.threads} threads."
                )
            }

            ComputeBackend.GPU -> {
                val delegate = GpuDelegate()
                options.addDelegate(delegate)
                TfLiteSession(
                    interpreter = Interpreter(model, options),
                    backend = config.backend,
                    note = "GPU delegate active."
                )
            }

            ComputeBackend.NNAPI -> {
                options.setUseNNAPI(true)
                TfLiteSession(
                    interpreter = Interpreter(model, options),
                    backend = config.backend,
                    note = "NNAPI requested. Actual acceleration depends on device driver support."
                )
            }
        }
    }
}
