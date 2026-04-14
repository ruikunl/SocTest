package com.rockenmini.socbenchmark.segmentation

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import androidx.core.graphics.scale
import com.rockenmini.socbenchmark.benchmark.BenchmarkConfig
import com.rockenmini.socbenchmark.benchmark.BenchmarkPreset
import com.rockenmini.socbenchmark.benchmark.ComputeBackend
import com.rockenmini.socbenchmark.benchmark.OnnxRuntimeBackends
import com.rockenmini.socbenchmark.benchmark.OnnxRuntimeSession
import com.rockenmini.socbenchmark.preview.PreviewMetrics
import com.rockenmini.socbenchmark.preview.PreviewRenderResult
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.min
import kotlin.math.round
import kotlin.system.measureNanoTime

class SegmentationDetector(private val context: Context) : Closeable {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val dispatcher: ExecutorCoroutineDispatcher = executor.asCoroutineDispatcher()
    private val sessions = ConcurrentHashMap<ComputeBackend, OnnxRuntimeSession>()
    private val extractedModelFile by lazy { prepareModelFiles() }
    private val inputPixels = IntArray(MODEL_SIZE * MODEL_SIZE)
    private val inputByteBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(MODEL_SIZE * MODEL_SIZE * 3 * 4)
        .order(ByteOrder.nativeOrder())
    private val inputFloatBuffer: FloatBuffer = inputByteBuffer.asFloatBuffer()
    private val maskColors = IntArray(MODEL_SIZE * MODEL_SIZE)

    suspend fun run(source: Bitmap, backend: ComputeBackend, sourceCount: Int): PreviewRenderResult =
        withContext(dispatcher) {
            // The segmentation model uses an external .data file, so we cache file-backed sessions
            // per backend after the first extraction to local storage.
            val session = sessions.getOrPut(backend) {
                OnnxRuntimeBackends.createSessionFromFile(
                    modelFile = extractedModelFile,
                    config = BenchmarkConfig(
                        backend = backend,
                        warmupRuns = 0,
                        measuredRuns = 1,
                        threads = 4
                    )
                )
            }

            var inputTensor: OnnxTensor? = null
            var maskBitmap: Bitmap? = null
            var ortResult: OrtSession.Result? = null
            var preprocessMs = 0.0
            var inferenceMs = 0.0
            var postprocessMs = 0.0
            var overlayRenderMs = 0.0

            val totalMs = measureNanoTime {
                // totalMs excludes the final visualization pass. This lets exported timings answer
                // whether latency comes from model execution or from bitmap compositing.
                val preprocessOutput = PreprocessOutput(source.width, source.height)
                preprocessMs = measureStepMs {
                    val tensorBuffer = preprocessBitmap(source, preprocessOutput)
                    inputTensor = OnnxTensor.createTensor(
                        OnnxRuntimeBackends.environment,
                        tensorBuffer,
                        longArrayOf(1, 3, MODEL_SIZE.toLong(), MODEL_SIZE.toLong())
                    )
                }

                inferenceMs = measureStepMs {
                    ortResult = session.session.run(mapOf(INPUT_NAME to checkNotNull(inputTensor)))
                }

                postprocessMs = measureStepMs {
                    checkNotNull(ortResult).use { result ->
                        val mask = (result.get(OUTPUT_NAME).orElseThrow() as OnnxTensor).getFloatBuffer()
                        maskBitmap = buildMaskBitmap(mask, preprocessOutput)
                    }
                }
            } / 1_000_000.0

            inputTensor?.close()

            var overlay: Bitmap? = null
            overlayRenderMs = measureStepMs {
                overlay = buildOverlayBitmap(
                    source = source,
                    maskBitmap = checkNotNull(maskBitmap),
                    backend = backend
                )
            }

            PreviewRenderResult(
                bitmap = checkNotNull(overlay),
                metrics = PreviewMetrics(
                    totalMs = totalMs.rounded(2),
                    preprocessMs = preprocessMs.rounded(2),
                    inferenceMs = inferenceMs.rounded(2),
                    postprocessMs = postprocessMs.rounded(2),
                    overlayRenderMs = overlayRenderMs.rounded(2),
                    resolutionText = "${source.width} x ${source.height}",
                    backendText = backend.displayName,
                    taskText = BenchmarkPreset.SEGMENTATION.title,
                    sourceCount = sourceCount,
                    note = session.note
                )
            )
        }

    private fun preprocessBitmap(source: Bitmap, metadata: PreprocessOutput): FloatBuffer {
        // The selfie model expects a square 256x256 input. We letterbox instead of center-cropping
        // so the whole subject stays visible in portrait photos and the output mask can be mapped
        // back onto the original frame without losing the top/bottom of the body.
        val canvasBitmap = Bitmap.createBitmap(MODEL_SIZE, MODEL_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBitmap)
        canvas.drawColor(Color.BLACK)

        val scale = min(
            MODEL_SIZE.toFloat() / source.width.toFloat(),
            MODEL_SIZE.toFloat() / source.height.toFloat()
        )
        val scaledWidth = source.width * scale
        val scaledHeight = source.height * scale
        val offsetX = (MODEL_SIZE - scaledWidth) / 2f
        val offsetY = (MODEL_SIZE - scaledHeight) / 2f
        val destRect = RectF(offsetX, offsetY, offsetX + scaledWidth, offsetY + scaledHeight)
        canvas.drawBitmap(source, null, destRect, null)

        metadata.scale = scale
        metadata.offsetX = offsetX
        metadata.offsetY = offsetY
        metadata.scaledWidth = scaledWidth
        metadata.scaledHeight = scaledHeight

        canvasBitmap.getPixels(inputPixels, 0, MODEL_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE)
        inputFloatBuffer.rewind()

        // Model input is NCHW float32. Qualcomm's packaged ONNX asset reports plain float input,
        // and using [0, 1] normalized RGB is the safest baseline here.
        repeat(3) { channel ->
            for (y in 0 until MODEL_SIZE) {
                for (x in 0 until MODEL_SIZE) {
                    val pixel = inputPixels[y * MODEL_SIZE + x]
                    val value = when (channel) {
                        0 -> Color.red(pixel) / 255f
                        1 -> Color.green(pixel) / 255f
                        else -> Color.blue(pixel) / 255f
                    }
                    inputFloatBuffer.put(value)
                }
            }
        }

        canvasBitmap.recycle()
        inputFloatBuffer.rewind()
        return inputFloatBuffer
    }

    private fun buildMaskBitmap(mask: FloatBuffer, metadata: PreprocessOutput): Bitmap {
        // The model outputs a square mask in letterboxed model space. We first colorize that mask,
        // then remove the padded border area, and finally scale the active region back to the
        // original image resolution.
        val maskValues = FloatArray(MODEL_SIZE * MODEL_SIZE)
        mask.rewind()
        mask.get(maskValues)

        val modelMaskBitmap = Bitmap.createBitmap(MODEL_SIZE, MODEL_SIZE, Bitmap.Config.ARGB_8888)
        for (y in 0 until MODEL_SIZE) {
            for (x in 0 until MODEL_SIZE) {
                val confidence = maskValues[y * MODEL_SIZE + x].coerceIn(0f, 1f)
                val alpha = (confidence * 180f).toInt().coerceIn(0, 180)
                val color = if (confidence >= PERSON_THRESHOLD) {
                    Color.argb(alpha, 34, 197, 94)
                } else {
                    Color.TRANSPARENT
                }
                maskColors[y * MODEL_SIZE + x] = color
            }
        }
        modelMaskBitmap.setPixels(maskColors, 0, MODEL_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE)

        val croppedLeft = metadata.offsetX.toInt().coerceAtLeast(0)
        val croppedTop = metadata.offsetY.toInt().coerceAtLeast(0)
        val croppedWidth = metadata.scaledWidth.toInt().coerceAtMost(MODEL_SIZE - croppedLeft).coerceAtLeast(1)
        val croppedHeight = metadata.scaledHeight.toInt().coerceAtMost(MODEL_SIZE - croppedTop).coerceAtLeast(1)
        val activeMask = Bitmap.createBitmap(
            modelMaskBitmap,
            croppedLeft,
            croppedTop,
            croppedWidth,
            croppedHeight
        )
        val sourceMask = activeMask.scale(metadata.sourceWidth, metadata.sourceHeight, true)

        modelMaskBitmap.recycle()
        if (activeMask !== sourceMask) {
            activeMask.recycle()
        }
        return sourceMask
    }

    private fun buildOverlayBitmap(
        source: Bitmap,
        maskBitmap: Bitmap,
        backend: ComputeBackend
    ): Bitmap {
        val overlay = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(overlay)
        canvas.drawBitmap(maskBitmap, 0f, 0f, Paint(Paint.ANTI_ALIAS_FLAG))

        val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = source.width * 0.04f
            setShadowLayer(8f, 0f, 0f, Color.BLACK)
        }
        canvas.drawText(
            "Selfie Segmentation ONNX Runtime (${backend.displayName})",
            source.width * 0.05f,
            source.height * 0.08f,
            labelPaint
        )
        return overlay
    }

    override fun close() {
        sessions.values.forEach { it.close() }
        sessions.clear()
        dispatcher.close()
        executor.shutdown()
    }

    private fun measureStepMs(block: () -> Unit): Double {
        return measureNanoTime(block) / 1_000_000.0
    }

    private fun prepareModelFiles(): File {
        // ORT resolves the sidecar weight file relative to the .onnx path, so both assets must live
        // together in the app's local filesystem before session creation.
        val modelDir = File(context.filesDir, "onnx_models/mediapipe_selfie_segmentation").apply {
            mkdirs()
        }
        val modelFile = File(modelDir, "mediapipe_selfie_segmentation.onnx")
        val dataFile = File(modelDir, "mediapipe_selfie.data")

        if (!modelFile.exists()) {
            context.assets.open(MODEL_ASSET_NAME).use { input ->
                modelFile.outputStream().use { output -> input.copyTo(output) }
            }
        }
        if (!dataFile.exists()) {
            context.assets.open(MODEL_DATA_ASSET_NAME).use { input ->
                dataFile.outputStream().use { output -> input.copyTo(output) }
            }
        }

        return modelFile
    }

    companion object {
        private const val MODEL_ASSET_NAME = "models/mediapipe_selfie_segmentation.onnx"
        private const val MODEL_DATA_ASSET_NAME = "models/mediapipe_selfie.data"
        private const val INPUT_NAME = "image"
        private const val OUTPUT_NAME = "mask"
        private const val MODEL_SIZE = 256
        private const val PERSON_THRESHOLD = 0.35f
    }
}

private data class PreprocessOutput(
    val sourceWidth: Int,
    val sourceHeight: Int,
    var scale: Float = 1f,
    var offsetX: Float = 0f,
    var offsetY: Float = 0f,
    var scaledWidth: Float = 0f,
    var scaledHeight: Float = 0f
)

private fun Double.rounded(scale: Int): Double {
    var factor = 1.0
    repeat(scale) {
        factor *= 10.0
    }
    return round(this * factor) / factor
}
