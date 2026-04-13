package com.rockenmini.socbenchmark.benchmark

import android.content.Context
import java.io.Closeable
import java.io.File

class NcnnSession private constructor(
    private val handle: Long,
    val backend: ComputeBackend,
    val note: String
) : Closeable {
    fun runPose(input: FloatArray, width: Int, height: Int, channels: Int): Pair<FloatArray, FloatArray> {
        val outputs = NcnnBridge.nativeRunPose(handle, input, width, height, channels)
        return outputs[0] to outputs[1]
    }

    fun runSegmentation(input: FloatArray, width: Int, height: Int, channels: Int): FloatArray {
        return NcnnBridge.nativeRunSegmentation(handle, input, width, height, channels)
    }

    override fun close() {
        NcnnBridge.nativeClose(handle)
    }

    companion object {
        fun create(
            context: Context,
            paramAssetPath: String,
            binAssetPath: String,
            config: BenchmarkConfig
        ): NcnnSession {
            val modelDir = File(context.filesDir, "ncnn_models").apply { mkdirs() }
            val paramFile = copyAssetIfNeeded(context, paramAssetPath, File(modelDir, File(paramAssetPath).name))
            val binFile = copyAssetIfNeeded(context, binAssetPath, File(modelDir, File(binAssetPath).name))
            val handle = NcnnBridge.nativeCreate(
                paramFile.absolutePath,
                binFile.absolutePath,
                config.backend.ordinal,
                config.threads
            )
            return NcnnSession(
                handle = handle,
                backend = config.backend,
                note = buildNote(config)
            )
        }

        private fun copyAssetIfNeeded(context: Context, assetPath: String, destination: File): File {
            if (!destination.exists()) {
                destination.parentFile?.mkdirs()
                context.assets.open(assetPath).use { input ->
                    destination.outputStream().use { output -> input.copyTo(output) }
                }
            }
            return destination
        }

        private fun buildNote(config: BenchmarkConfig): String {
            return when (config.backend) {
                ComputeBackend.CPU ->
                    "ncnn CPU path (${config.threads} threads)."
                ComputeBackend.GPU ->
                    if (NcnnBridge.gpuCount() > 0) {
                        "ncnn Vulkan GPU path requested."
                    } else {
                        "ncnn Vulkan GPU requested, but no Vulkan GPU was reported. Falling back to CPU."
                    }
                ComputeBackend.NNAPI ->
                    "ncnn does not provide an NNAPI path in this build. Falling back to CPU."
            }
        }
    }
}

object NcnnBridge {
    init {
        System.loadLibrary("soctest_ncnn")
        initializeGpu()
    }

    external fun initializeGpu()
    external fun releaseGpu()
    external fun gpuCount(): Int
    external fun nativeCreate(paramPath: String, binPath: String, backend: Int, threads: Int): Long
    external fun nativeRunPose(
        handle: Long,
        input: FloatArray,
        width: Int,
        height: Int,
        channels: Int
    ): Array<FloatArray>
    external fun nativeRunSegmentation(
        handle: Long,
        input: FloatArray,
        width: Int,
        height: Int,
        channels: Int
    ): FloatArray
    external fun nativeClose(handle: Long)
}
