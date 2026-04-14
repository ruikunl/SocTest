package com.rockenmini.socbenchmark.pose

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import com.rockenmini.socbenchmark.benchmark.BenchmarkConfig
import com.rockenmini.socbenchmark.benchmark.BenchmarkPreset
import com.rockenmini.socbenchmark.benchmark.ComputeBackend
import com.rockenmini.socbenchmark.benchmark.NcnnSession
import com.rockenmini.socbenchmark.preview.PreviewMetrics
import com.rockenmini.socbenchmark.preview.PreviewRenderResult
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.io.Closeable
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
import kotlin.system.measureNanoTime

class PoseDetector(private val context: Context) : Closeable {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val dispatcher: ExecutorCoroutineDispatcher = executor.asCoroutineDispatcher()
    private val sessions = ConcurrentHashMap<ComputeBackend, NcnnSession>()
    private val inputPixels = IntArray(MODEL_WIDTH * MODEL_HEIGHT)
    private val inputValues = FloatArray(MODEL_WIDTH * MODEL_HEIGHT * 3)

    suspend fun run(source: Bitmap, backend: ComputeBackend, sourceCount: Int): PreviewRenderResult =
        withContext(dispatcher) {
            // Each backend owns one long-lived ncnn session so benchmark numbers do not include
            // repeated model loading or backend initialization cost.
            val session = sessions.getOrPut(backend) {
                NcnnSession.create(
                    context = context,
                    paramAssetPath = MODEL_PARAM_ASSET_NAME,
                    binAssetPath = MODEL_BIN_ASSET_NAME,
                    config = BenchmarkConfig(
                        backend = backend,
                        warmupRuns = 0,
                        measuredRuns = 1,
                        threads = if (backend == ComputeBackend.GPU) 4 else 1
                    )
                )
            }

            var inputTensor: FloatArray? = null
            var decodedPose: List<Keypoint>? = null
            var simccX: FloatArray? = null
            var simccY: FloatArray? = null
            var preprocessMs = 0.0
            var inferenceMs = 0.0
            var postprocessMs = 0.0
            var overlayRenderMs = 0.0

            val totalMs = measureNanoTime {
                // totalMs intentionally covers only preprocess + inference + postprocess.
                // The final overlay drawing is timed separately so UI rendering cost does not get
                // mixed into model execution numbers.
                val preprocessOutput = PreprocessOutput(source.width, source.height)
                preprocessMs = measureStepMs {
                    inputTensor = preprocessBitmap(source, preprocessOutput)
                }

                inferenceMs = measureStepMs {
                    val outputs = session.runPose(
                        input = checkNotNull(inputTensor),
                        width = MODEL_WIDTH,
                        height = MODEL_HEIGHT,
                        channels = 3
                    )
                    simccX = outputs.first
                    simccY = outputs.second
                }

                postprocessMs = measureStepMs {
                    decodedPose = decodeKeypoints(
                        simccX = checkNotNull(simccX),
                        simccY = checkNotNull(simccY),
                        metadata = preprocessOutput
                    )
                }
            } / 1_000_000.0

            var overlay: Bitmap? = null
            overlayRenderMs = measureStepMs {
                overlay = drawPose(source, checkNotNull(decodedPose), backend)
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

    private fun preprocessBitmap(source: Bitmap, metadata: PreprocessOutput): FloatArray {
        // RTMPose is a top-down model. Since this benchmark app currently has no person detector,
        // we approximate the person box with a centered crop that matches the model aspect ratio.
        // This works well for the current evaluation setup where the subject is usually centered
        // and occupies a large part of the frame.
        val cropRect = buildCenteredCropRect(source.width, source.height)
        val croppedBitmap = Bitmap.createBitmap(
            source,
            cropRect.left,
            cropRect.top,
            cropRect.width(),
            cropRect.height()
        )
        val resizedBitmap = Bitmap.createScaledBitmap(
            croppedBitmap,
            MODEL_WIDTH,
            MODEL_HEIGHT,
            true
        )

        metadata.cropLeft = cropRect.left.toFloat()
        metadata.cropTop = cropRect.top.toFloat()
        metadata.cropWidth = cropRect.width().toFloat()
        metadata.cropHeight = cropRect.height().toFloat()

        resizedBitmap.getPixels(inputPixels, 0, MODEL_WIDTH, 0, 0, MODEL_WIDTH, MODEL_HEIGHT)
        var offset = 0

        // ncnn receives the same NCHW layout as the ONNX baseline: red plane, green plane, blue plane.
        repeat(3) { channel ->
            for (y in 0 until MODEL_HEIGHT) {
                for (x in 0 until MODEL_WIDTH) {
                    val pixel = inputPixels[y * MODEL_WIDTH + x]
                    val normalized = when (channel) {
                        0 -> ((Color.red(pixel) - MEAN[0]) / STD[0]).toFloat()
                        1 -> ((Color.green(pixel) - MEAN[1]) / STD[1]).toFloat()
                        else -> ((Color.blue(pixel) - MEAN[2]) / STD[2]).toFloat()
                    }
                    inputValues[offset++] = normalized
                }
            }
        }

        // createBitmap/createScaledBitmap may legally return the original bitmap instance when no
        // real copy is needed. We only recycle owned temporaries so the Compose UI never receives
        // a bitmap that has already been freed.
        if (croppedBitmap !== source && croppedBitmap !== resizedBitmap) {
            croppedBitmap.recycle()
        }
        if (resizedBitmap !== source && resizedBitmap !== croppedBitmap) {
            resizedBitmap.recycle()
        }
        return inputValues
    }

    private fun decodeKeypoints(
        simccX: FloatArray,
        simccY: FloatArray,
        metadata: PreprocessOutput
    ): List<Keypoint> {
        // RTMPose uses SimCC outputs, so postprocess here is "argmax over x logits + argmax over y
        // logits" for each joint, followed by mapping the coordinates back from model space into
        // the original image crop.
        val keypoints = ArrayList<Keypoint>(KEYPOINT_COUNT)
        repeat(KEYPOINT_COUNT) { jointIndex ->
            val xOffset = jointIndex * SIMCC_X_DIM
            val yOffset = jointIndex * SIMCC_Y_DIM

            var bestXIndex = 0
            var bestYIndex = 0
            var bestXValue = Float.NEGATIVE_INFINITY
            var bestYValue = Float.NEGATIVE_INFINITY

            for (i in 0 until SIMCC_X_DIM) {
                val value = simccX[xOffset + i]
                if (value > bestXValue) {
                    bestXValue = value
                    bestXIndex = i
                }
            }
            for (i in 0 until SIMCC_Y_DIM) {
                val value = simccY[yOffset + i]
                if (value > bestYValue) {
                    bestYValue = value
                    bestYIndex = i
                }
            }

            val modelX = bestXIndex / SIMCC_SPLIT_RATIO
            val modelY = bestYIndex / SIMCC_SPLIT_RATIO
            val sourceX = (
                metadata.cropLeft + modelX * (metadata.cropWidth / MODEL_WIDTH.toFloat())
            ).coerceIn(0f, metadata.sourceWidth - 1f)
            val sourceY = (
                metadata.cropTop + modelY * (metadata.cropHeight / MODEL_HEIGHT.toFloat())
            ).coerceIn(0f, metadata.sourceHeight - 1f)

            keypoints += Keypoint(
                name = KEYPOINT_NAMES[jointIndex],
                x = sourceX,
                y = sourceY,
                score = ((sigmoid(bestXValue) + sigmoid(bestYValue)) / 2f)
            )
        }

        return keypoints
    }

    private fun drawPose(
        source: Bitmap,
        keypoints: List<Keypoint>,
        backend: ComputeBackend
    ): Bitmap {
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

        BONES.forEach { (start, end) ->
            val startPoint = keypoints[start]
            val endPoint = keypoints[end]
            if (startPoint.score >= SCORE_THRESHOLD && endPoint.score >= SCORE_THRESHOLD) {
                canvas.drawLine(startPoint.x, startPoint.y, endPoint.x, endPoint.y, linePaint)
            }
        }

        keypoints.forEach { point ->
            if (point.score >= SCORE_THRESHOLD) {
                canvas.drawCircle(point.x, point.y, result.width * 0.014f, pointPaint)
                canvas.drawText(point.name, point.x + 10f, point.y - 10f, textPaint)
            }
        }

        canvas.drawText(
            "RTMPose ncnn (${backend.displayName})",
            result.width * 0.05f,
            result.height * 0.08f,
            textPaint
        )
        return result
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

    private fun sigmoid(value: Float): Float {
        return (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()
    }

    private fun buildCenteredCropRect(sourceWidth: Int, sourceHeight: Int): android.graphics.Rect {
        // This keeps the crop aspect ratio aligned with the model input so coordinate remapping stays
        // linear. It is a benchmarking shortcut for mostly-centered single-person images, not a
        // replacement for a real person detector in a production pipeline.
        val targetAspect = MODEL_WIDTH.toFloat() / MODEL_HEIGHT.toFloat()
        val sourceAspect = sourceWidth.toFloat() / sourceHeight.toFloat()

        val cropWidth: Int
        val cropHeight: Int
        if (sourceAspect > targetAspect) {
            cropHeight = sourceHeight
            cropWidth = (cropHeight * targetAspect).toInt()
        } else {
            cropWidth = sourceWidth
            cropHeight = (cropWidth / targetAspect).toInt()
        }

        val left = max(0, (sourceWidth - cropWidth) / 2)
        val top = max(0, (sourceHeight - cropHeight) / 2)
        return android.graphics.Rect(left, top, left + cropWidth, top + cropHeight)
    }

    companion object {
        private const val MODEL_PARAM_ASSET_NAME = "models/ncnn/rtmpose_t_body7_256x192.param"
        private const val MODEL_BIN_ASSET_NAME = "models/ncnn/rtmpose_t_body7_256x192.bin"
        private const val MODEL_WIDTH = 192
        private const val MODEL_HEIGHT = 256
        private const val KEYPOINT_COUNT = 17
        private const val SIMCC_X_DIM = 384
        private const val SIMCC_Y_DIM = 512
        private const val SIMCC_SPLIT_RATIO = 2f
        private const val SCORE_THRESHOLD = 0.35f

        private val MEAN = doubleArrayOf(123.675, 116.28, 103.53)
        private val STD = doubleArrayOf(58.395, 57.12, 57.375)

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

private data class PreprocessOutput(
    val sourceWidth: Int,
    val sourceHeight: Int,
    var cropLeft: Float = 0f,
    var cropTop: Float = 0f,
    var cropWidth: Float = 0f,
    var cropHeight: Float = 0f
)

private data class Keypoint(
    val name: String,
    val x: Float,
    val y: Float,
    val score: Float
)

private fun Double.rounded(scale: Int): Double {
    var factor = 1.0
    repeat(scale) {
        factor *= 10.0
    }
    return round(this * factor) / factor
}
