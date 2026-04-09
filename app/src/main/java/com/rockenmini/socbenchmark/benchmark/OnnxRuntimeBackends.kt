package com.rockenmini.socbenchmark.benchmark

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import java.io.File
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.EnumSet

data class OnnxRuntimeSession(
    val session: OrtSession,
    val backend: ComputeBackend,
    val note: String,
    private val closeables: List<AutoCloseable> = emptyList()
) : Closeable {
    override fun close() {
        session.close()
        closeables.forEach { it.close() }
    }
}

object OnnxRuntimeBackends {
    // ORT environments are process-level singletons in normal usage. Keeping one shared instance
    // avoids repeated native runtime setup and matches how the cached sessions are used elsewhere.
    val environment: OrtEnvironment by lazy {
        OrtEnvironment.getEnvironment()
    }

    fun createSession(
        context: Context,
        modelAssetPath: String,
        config: BenchmarkConfig
    ): OnnxRuntimeSession {
        val modelBuffer = loadAssetToDirectBuffer(context, modelAssetPath)
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            setInterOpNumThreads(1)
        }

        return when (config.backend) {
            ComputeBackend.CPU -> {
                // XNNPACK is the main ORT mobile CPU acceleration path on Android. We keep the ORT
                // threadpool small and hand the requested parallelism to XNNPACK instead.
                options.addConfigEntry("session.intra_op.allow_spinning", "0")
                options.addXnnpack(
                    mapOf("intra_op_num_threads" to config.threads.toString())
                )
                OnnxRuntimeSession(
                    session = environment.createSession(modelBuffer, options),
                    backend = config.backend,
                    note = "ONNX Runtime CPU path with XNNPACK (${config.threads} threads).",
                    closeables = listOf(options)
                )
            }

            ComputeBackend.GPU -> {
                // The baseline onnxruntime-android package does not provide a generic Android GPU EP.
                // To keep the UI flow intact for now we fall back to the same XNNPACK CPU path and
                // surface that clearly in runtime info.
                options.addConfigEntry("session.intra_op.allow_spinning", "0")
                options.addXnnpack(
                    mapOf("intra_op_num_threads" to config.threads.toString())
                )
                OnnxRuntimeSession(
                    session = environment.createSession(modelBuffer, options),
                    backend = config.backend,
                    note = "GPU EP is not enabled in this ORT baseline build. Falling back to XNNPACK CPU.",
                    closeables = listOf(options)
                )
            }

            ComputeBackend.NNAPI -> {
                // NNAPI is the official Android hardware acceleration path in ORT. We request FP16
                // where possible, but the device may still partition the graph or fall back.
                options.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                OnnxRuntimeSession(
                    session = environment.createSession(modelBuffer, options),
                    backend = config.backend,
                    note = "ONNX Runtime NNAPI path requested. Actual acceleration depends on driver and graph partitioning.",
                    closeables = listOf(options)
                )
            }
        }
    }

    fun createSessionFromFile(
        modelFile: File,
        config: BenchmarkConfig
    ): OnnxRuntimeSession {
        // Some ONNX models reference external weight files via sidecar .data assets. Those models
        // cannot be created from an in-memory ByteBuffer because ORT resolves the extra files from
        // the model path on disk, so segmentation uses this file-based session path.
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            setInterOpNumThreads(1)
        }

        return when (config.backend) {
            ComputeBackend.CPU -> {
                options.addConfigEntry("session.intra_op.allow_spinning", "0")
                options.addXnnpack(
                    mapOf("intra_op_num_threads" to config.threads.toString())
                )
                OnnxRuntimeSession(
                    session = environment.createSession(modelFile.absolutePath, options),
                    backend = config.backend,
                    note = "ONNX Runtime CPU path with XNNPACK (${config.threads} threads).",
                    closeables = listOf(options)
                )
            }

            ComputeBackend.GPU -> {
                options.addConfigEntry("session.intra_op.allow_spinning", "0")
                options.addXnnpack(
                    mapOf("intra_op_num_threads" to config.threads.toString())
                )
                OnnxRuntimeSession(
                    session = environment.createSession(modelFile.absolutePath, options),
                    backend = config.backend,
                    note = "GPU EP is not enabled in this ORT baseline build. Falling back to XNNPACK CPU.",
                    closeables = listOf(options)
                )
            }

            ComputeBackend.NNAPI -> {
                options.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                OnnxRuntimeSession(
                    session = environment.createSession(modelFile.absolutePath, options),
                    backend = config.backend,
                    note = "ONNX Runtime NNAPI path requested. Actual acceleration depends on driver and graph partitioning.",
                    closeables = listOf(options)
                )
            }
        }
    }

    private fun loadAssetToDirectBuffer(context: Context, assetPath: String): ByteBuffer {
        // Keypoint models on this branch are single-file ONNX assets, so loading them straight into
        // a direct buffer keeps startup simple and avoids temporary file management.
        val bytes = context.assets.open(assetPath).use { it.readBytes() }
        return ByteBuffer.allocateDirect(bytes.size).apply {
            order(ByteOrder.nativeOrder())
            put(bytes)
            rewind()
        }
    }
}
