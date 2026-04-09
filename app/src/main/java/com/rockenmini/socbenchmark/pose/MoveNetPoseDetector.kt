package com.rockenmini.socbenchmark.pose

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
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.round
import kotlin.system.measureNanoTime

class MoveNetPoseDetector(private val context: Context) : Closeable {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val dispatcher: ExecutorCoroutineDispatcher = executor.asCoroutineDispatcher()
    private val sessions = ConcurrentHashMap<ComputeBackend, TfLiteSession>()

    suspend fun run(source: Bitmap, backend: ComputeBackend, sourceCount: Int): PreviewRenderResult =
        withContext(dispatcher) {
            // Each backend keeps its own interpreter instance. This avoids paying model/delegate
            // initialization cost on every image and makes repeated benchmark runs more stable.
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

            var resizedBitmap: Bitmap? = null
            var inferenceOutput: Array<Array<FloatArray>>? = null
            var preprocessMs = 0.0
            var inferenceMs = 0.0
            var postprocessMs = 0.0
            var overlayRenderMs = 0.0

            val totalMs = measureNanoTime {
                // Preprocess only covers image resize here. It intentionally excludes model loading
                // because interpreter creation is cached in the session above.
                preprocessMs = measureStepMs {
                    resizedBitmap = source.scale(INPUT_SIZE, INPUT_SIZE, false)
                }

                // MoveNet Lightning expects a 192x192 RGB UINT8 tensor. Packing into a direct
                // ByteBuffer keeps the Java side representation aligned with the TFLite input tensor.
                val inputTensor = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3).apply {
                    order(ByteOrder.nativeOrder())
                }
                val scaled = checkNotNull(resizedBitmap)
                for (y in 0 until INPUT_SIZE) {
                    for (x in 0 until INPUT_SIZE) {
                        val pixel = scaled.getPixel(x, y)
                        inputTensor.put(Color.red(pixel).toByte())
                        inputTensor.put(Color.green(pixel).toByte())
                        inputTensor.put(Color.blue(pixel).toByte())
                    }
                }
                inputTensor.rewind()

                // Inference time is only the interpreter invocation itself.
                val outputTensor = Array(1) { Array(1) { Array(KEYPOINT_COUNT) { FloatArray(3) } } }
                inferenceMs = measureStepMs {
                    session.interpreter.run(inputTensor, outputTensor)
                }

                // Postprocess here only extracts the raw keypoint tensor from the model output.
                // Drawing the overlay is performed afterwards and is not mixed into postprocessMs.
                postprocessMs = measureStepMs {
                    inferenceOutput = outputTensor[0]
                }
            } / 1_000_000.0

            var overlay: Bitmap? = null
            overlayRenderMs = measureStepMs {
                overlay = drawPose(source, checkNotNull(inferenceOutput), backend)
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
                    taskText = BenchmarkPreset.POSE.title,
                    sourceCount = sourceCount,
                    note = session.note
                )
            )
        }

    private fun drawPose(
        source: Bitmap,
        keypointsWithScores: Array<Array<FloatArray>>,
        backend: ComputeBackend
    ): Bitmap {
        // Rendering is deliberately kept outside inference timing so the displayed numbers reflect
        // model-side cost rather than UI-side drawing cost.
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(255, 140, 0)
            strokeWidth = result.width * 0.008f
            style = Paint.Style.STROKE
        }
        val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(34, 211, 238)
            style = Paint.Style.FILL
        }
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = result.width * 0.04f
            setShadowLayer(8f, 0f, 0f, Color.BLACK)
        }

        val points = keypointsWithScores[0]
        BONES.forEach { (start, end) ->
            val startPoint = points[start]
            val endPoint = points[end]
            if (startPoint[2] >= SCORE_THRESHOLD && endPoint[2] >= SCORE_THRESHOLD) {
                canvas.drawLine(
                    startPoint[1] * result.width,
                    startPoint[0] * result.height,
                    endPoint[1] * result.width,
                    endPoint[0] * result.height,
                    linePaint
                )
            }
        }

        points.forEachIndexed { index, point ->
            if (point[2] >= SCORE_THRESHOLD) {
                val x = point[1] * result.width
                val y = point[0] * result.height
                canvas.drawCircle(x, y, result.width * 0.014f, pointPaint)
                canvas.drawText(KEYPOINT_NAMES[index], x + 10f, y - 10f, textPaint)
            }
        }

        canvas.drawText("MoveNet Pose (${backend.displayName})", result.width * 0.05f, result.height * 0.08f, textPaint)
        return result
    }

    override fun close() {
        // GPU / NNAPI delegates are tied to interpreter lifecycle, so closing the cached sessions is
        // the point where accelerator resources are actually released.
        sessions.values.forEach { it.close() }
        sessions.clear()
        dispatcher.close()
        executor.shutdown()
    }

    private fun measureStepMs(block: () -> Unit): Double {
        return measureNanoTime(block) / 1_000_000.0
    }

    companion object {
        // This is the single-pose MoveNet Lightning model bundled in app assets.
        private const val MODEL_ASSET_NAME = "models/movenet_singlepose_lightning_f16.tflite"
        private const val INPUT_SIZE = 192
        private const val KEYPOINT_COUNT = 17
        private const val SCORE_THRESHOLD = 0.2f

        private val KEYPOINT_NAMES = listOf(
            "nose", "l_eye", "r_eye", "l_ear", "r_ear",
            "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
            "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"
        )

        private val BONES = listOf(
            0 to 1, 0 to 2, 1 to 3, 2 to 4,
            0 to 5, 0 to 6, 5 to 7, 7 to 9,
            6 to 8, 8 to 10, 5 to 6,
            5 to 11, 6 to 12, 11 to 12,
            11 to 13, 13 to 15, 12 to 14, 14 to 16
        )
    }
}

private fun Double.rounded(scale: Int): Double {
    var factor = 1.0
    repeat(scale) {
        factor *= 10.0
    }
    return round(this * factor) / factor
}
