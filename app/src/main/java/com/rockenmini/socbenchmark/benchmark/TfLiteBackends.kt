package com.rockenmini.socbenchmark.benchmark

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
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
        // Models are loaded as memory-mapped files so startup stays cheap even for larger assets.
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
                    note = "TFLite CPU path active with ${config.threads} threads."
                )
            }

            ComputeBackend.GPU -> {
                // We intentionally use a conservative GPU configuration here. RTMPose keypoint
                // outputs are more sensitive to delegate precision than the current segmentation
                // model, so we prefer stable results over the most aggressive defaults.
                val compatibilityList = CompatibilityList()
                val delegateOptions = compatibilityList.bestOptionsForThisDevice.apply {
                    setPrecisionLossAllowed(false)
                    setQuantizedModelsAllowed(false)
                    setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER)
                    setSerializationParams(context.codeCacheDir.absolutePath, modelAssetPath.replace('/', '_'))
                }
                val delegate = GpuDelegate(delegateOptions)
                options.addDelegate(delegate)
                TfLiteSession(
                    interpreter = Interpreter(model, options),
                    backend = config.backend,
                    note = buildString {
                        append("TFLite GPU delegate active")
                        if (compatibilityList.isDelegateSupportedOnThisDevice) {
                            append(" with compatibility-tuned conservative options.")
                        } else {
                            append(" with manually forced delegate options; device compatibility list reported unsupported state.")
                        }
                    },
                    closeables = listOf(delegate, compatibilityList)
                )
            }

            ComputeBackend.NNAPI -> {
                // NNAPI is only a request path. The actual device may execute on NPU, DSP, GPU,
                // or silently fall back to CPU depending on driver and op coverage.
                options.setUseNNAPI(true)
                TfLiteSession(
                    interpreter = Interpreter(model, options),
                    backend = config.backend,
                    note = "TFLite NNAPI path requested. Actual acceleration depends on device driver support."
                )
            }
        }
    }
}
