package com.rockenmini.socbenchmark.segmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.core.graphics.scale
import com.rockenmini.socbenchmark.benchmark.BenchmarkConfig
import com.rockenmini.socbenchmark.benchmark.BenchmarkPreset
import com.rockenmini.socbenchmark.benchmark.ComputeBackend
import com.rockenmini.socbenchmark.benchmark.TfLiteBackends
import com.rockenmini.socbenchmark.benchmark.TfLiteSession
import com.rockenmini.socbenchmark.preview.PreviewMetrics
import com.rockenmini.socbenchmark.preview.PreviewRenderResult
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.round
import kotlin.system.measureNanoTime

class SelfieSegmentationDetector(private val context: Context) : Closeable {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val dispatcher: ExecutorCoroutineDispatcher = executor.asCoroutineDispatcher()
    private val sessions = ConcurrentHashMap<ComputeBackend, TfLiteSession>()

    suspend fun run(source: Bitmap, backend: ComputeBackend, sourceCount: Int): PreviewRenderResult =
        withContext(dispatcher) {
            // Just like pose, we cache one interpreter per backend so per-image timing measures
            // preprocess/inference/postprocess rather than repeated model initialization.
            val session = sessions.getOrPut(backend) {
                TfLiteBackends.createSession(
                    context = context,
                    modelAssetPath = MODEL_ASSET_NAME,
                    config = BenchmarkConfig(
                        backend = backend,
                        warmupRuns = 0,
                        measuredRuns = 1,
                        threads = 4
                    )
                )
            }

            val inputTensor = session.interpreter.getInputTensor(0)
            val outputTensor = session.interpreter.getOutputTensor(0)
            val inputShape = inputTensor.shape()
            val outputShape = outputTensor.shape()
            val inputHeight = inputShape[1]
            val inputWidth = inputShape[2]
            val inputType = inputTensor.dataType()
            val outputType = outputTensor.dataType()

            var inputBuffer: ByteBuffer? = null
            var maskBitmap: Bitmap? = null
            var preprocessMs = 0.0
            var inferenceMs = 0.0
            var postprocessMs = 0.0
            var overlayRenderMs = 0.0

            val totalMs = measureNanoTime {
                // Segmentation preprocess includes resize plus packing into the tensor's expected
                // numeric format. The model input type is read from the tensor so this path stays
                // valid if we later swap in another TFLite segmentation model.
                preprocessMs = measureStepMs {
                    val resizedBitmap = source.scale(inputWidth, inputHeight, false)
                    inputBuffer = bitmapToInputBuffer(resizedBitmap, inputType)
                }

                val outputBuffer = createOutputBuffer(outputShape, outputType)
                // Inference time is only the raw TFLite execution.
                inferenceMs = measureStepMs {
                    session.interpreter.run(checkNotNull(inputBuffer), outputBuffer)
                }

                // Postprocess covers decoding the raw segmentation output into a displayable mask.
                postprocessMs = measureStepMs {
                    maskBitmap = buildMaskBitmap(
                        output = outputBuffer,
                        outputShape = outputShape,
                        outputType = outputType
                    )
                }
            } / 1_000_000.0

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

    private fun bitmapToInputBuffer(bitmap: Bitmap, dataType: DataType): ByteBuffer {
        // Some models accept UINT8 and others FLOAT32. We convert according to the tensor metadata
        // instead of hard-coding one format, which makes benchmarking model swaps safer.
        val bytesPerChannel = if (dataType == DataType.FLOAT32) 4 else 1
        val buffer = ByteBuffer.allocateDirect(bitmap.width * bitmap.height * 3 * bytesPerChannel)
        buffer.order(ByteOrder.nativeOrder())

        for (y in 0 until bitmap.height) {
            for (x in 0 until bitmap.width) {
                val pixel = bitmap.getPixel(x, y)
                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)

                if (dataType == DataType.FLOAT32) {
                    buffer.putFloat(r / 255f)
                    buffer.putFloat(g / 255f)
                    buffer.putFloat(b / 255f)
                } else {
                    buffer.put(r.toByte())
                    buffer.put(g.toByte())
                    buffer.put(b.toByte())
                }
            }
        }

        buffer.rewind()
        return buffer
    }

    private fun createOutputBuffer(outputShape: IntArray, dataType: DataType): Any {
        // Output rank and channel count can differ between segmentation models, so we allocate
        // against the tensor shape reported by the interpreter rather than assuming fixed dims.
        val height = outputShape[1]
        val width = outputShape[2]
        val channels = outputShape.getOrElse(3) { 1 }
        return if (dataType == DataType.FLOAT32) {
            Array(1) { Array(height) { Array(width) { FloatArray(channels) } } }
        } else {
            Array(1) { Array(height) { Array(width) { ByteArray(channels) } } }
        }
    }

    private fun buildMaskBitmap(
        output: Any,
        outputShape: IntArray,
        outputType: DataType
    ): Bitmap {
        val maskHeight = outputShape[1]
        val maskWidth = outputShape[2]
        val maskBitmap = Bitmap.createBitmap(maskWidth, maskHeight, Bitmap.Config.ARGB_8888)

        for (y in 0 until maskHeight) {
            for (x in 0 until maskWidth) {
                val confidence = if (outputType == DataType.FLOAT32) {
                    val channels = (output as Array<Array<Array<FloatArray>>>)[0][y][x]
                    confidenceFromChannels(channels)
                } else {
                    val channels = (output as Array<Array<Array<ByteArray>>>)[0][y][x]
                    confidenceFromChannels(channels)
                }

                val alpha = (confidence * 170).toInt().coerceIn(0, 170)
                val color = if (confidence >= 0.5f) {
                    Color.argb(alpha, 34, 197, 94)
                } else {
                    Color.argb((alpha * 0.35f).toInt(), 30, 41, 59)
                }
                maskBitmap.setPixel(x, y, color)
            }
        }

        return maskBitmap
    }

    private fun buildOverlayBitmap(
        source: Bitmap,
        maskBitmap: Bitmap,
        backend: ComputeBackend
    ): Bitmap {
        // Rendering the colored mask on top of the source image is tracked separately so it does
        // not get mixed into model postprocess time.
        val overlay = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(overlay)
        val scaledMask = maskBitmap.scale(source.width, source.height, true)
        canvas.drawBitmap(scaledMask, 0f, 0f, Paint(Paint.ANTI_ALIAS_FLAG))

        val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = source.width * 0.04f
            setShadowLayer(8f, 0f, 0f, Color.BLACK)
        }
        canvas.drawText("Person Segmentation (${backend.displayName})", source.width * 0.05f, source.height * 0.08f, labelPaint)
        return overlay
    }

    private fun confidenceFromChannels(channels: FloatArray): Float {
        // For multi-class output we approximate "personness" by comparing the person class against
        // background. This keeps the benchmark pipeline simple while still giving a usable overlay.
        return when {
            channels.isEmpty() -> 0f
            channels.size == 1 -> channels[0].coerceIn(0f, 1f)
            else -> {
                val person = channels.getOrElse(PERSON_CLASS_INDEX) { channels.maxOrNull() ?: 0f }
                val background = channels.firstOrNull() ?: 0f
                (0.5f + (person - background) / 2f).coerceIn(0f, 1f)
            }
        }
    }

    private fun confidenceFromChannels(channels: ByteArray): Float {
        return when {
            channels.isEmpty() -> 0f
            channels.size == 1 -> (channels[0].toInt() and 0xFF) / 255f
            else -> {
                val person = (channels.getOrElse(PERSON_CLASS_INDEX) { 0 }.toInt() and 0xFF) / 255f
                val background = ((channels.firstOrNull()?.toInt() ?: 0) and 0xFF) / 255f
                (0.5f + (person - background) / 2f).coerceIn(0f, 1f)
            }
        }
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

    companion object {
        // This branch now uses the ONNX-exported MediaPipe selfie segmentation network converted
        // to a plain TFLite model so it stays runnable on stock TFLite backends.
        private const val MODEL_ASSET_NAME = "models/mediapipe_selfie_segmentation_float32.tflite"
        private const val PERSON_CLASS_INDEX = 15
    }
}

private fun Double.rounded(scale: Int): Double {
    var factor = 1.0
    repeat(scale) {
        factor *= 10.0
    }
    return round(this * factor) / factor
}
