package com.rockenmini.socbenchmark.pose

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import com.rockenmini.socbenchmark.benchmark.BenchmarkConfig
import com.rockenmini.socbenchmark.benchmark.BenchmarkModels
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
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.round
import kotlin.system.measureNanoTime

class PoseDetector(private val context: Context) : Closeable {
    private val spec = BenchmarkModels.pose
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val dispatcher: ExecutorCoroutineDispatcher = executor.asCoroutineDispatcher()
    private val sessions = ConcurrentHashMap<ComputeBackend, TfLiteSession>()

    suspend fun run(source: Bitmap, backend: ComputeBackend, sourceCount: Int): PreviewRenderResult =
        withContext(dispatcher) {
            val session = sessions.getOrPut(backend) {
                TfLiteBackends.createSession(
                    context = context,
                    modelAssetPath = BenchmarkModels.pose.assetPath,
                    config = BenchmarkConfig(
                        backend = backend,
                        warmupRuns = 0,
                        measuredRuns = 1,
                        threads = 4
                    )
                )
            }

            var inputTensor: ByteBuffer? = null
            var decodedPose: List<Keypoint>? = null
            var simccX: FloatArray? = null
            var simccY: FloatArray? = null
            var preprocessMs = 0.0
            var inferenceMs = 0.0
            var postprocessMs = 0.0
            var overlayRenderMs = 0.0
            var note = session.note
            val preprocessInfo = PreprocessInfo(source.width, source.height)

            val totalMs = measureNanoTime {
                preprocessMs = measureStepMs {
                    inputTensor = preprocessBitmap(source, preprocessInfo)
                }

                val simccXTensor = Array(1) { Array(spec.keypointCount) { FloatArray(spec.simccXWidth) } }
                val simccYTensor = Array(1) { Array(spec.keypointCount) { FloatArray(spec.simccYWidth) } }
                val outputMap = linkedMapOf<Int, Any>()
                val output0Width = session.interpreter.getOutputTensor(0).shape().last()
                if (output0Width == spec.simccXWidth) {
                    outputMap[0] = simccXTensor
                    outputMap[1] = simccYTensor
                } else {
                    outputMap[0] = simccYTensor
                    outputMap[1] = simccXTensor
                }

                inferenceMs = measureStepMs {
                    session.interpreter.runForMultipleInputsOutputs(arrayOf(checkNotNull(inputTensor)), outputMap)
                }

                postprocessMs = measureStepMs {
                    simccX = flattenSimcc(simccXTensor[0])
                    simccY = flattenSimcc(simccYTensor[0])
                    decodedPose = decodeKeypoints(
                        simccX = checkNotNull(simccX),
                        simccY = checkNotNull(simccY),
                        metadata = preprocessInfo
                    )
                    note = appendPoseDiagnostic(
                        baseNote = session.note,
                        backend = backend,
                        keypoints = checkNotNull(decodedPose)
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
                    note = note
                )
            )
        }

    private fun preprocessBitmap(source: Bitmap, metadata: PreprocessInfo): ByteBuffer {
        val cropRect = buildCenteredCropRect(source.width, source.height)
        val croppedBitmap = Bitmap.createBitmap(
            source,
            cropRect.left,
            cropRect.top,
            cropRect.width(),
            cropRect.height()
        )
        val resizedBitmap = Bitmap.createScaledBitmap(croppedBitmap, spec.inputWidth, spec.inputHeight, true)

        metadata.cropLeft = cropRect.left.toFloat()
        metadata.cropTop = cropRect.top.toFloat()
        metadata.cropWidth = cropRect.width().toFloat()
        metadata.cropHeight = cropRect.height().toFloat()

        val inputTensor = ByteBuffer.allocateDirect(spec.inputWidth * spec.inputHeight * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        for (y in 0 until spec.inputHeight) {
            for (x in 0 until spec.inputWidth) {
                val pixel = resizedBitmap.getPixel(x, y)
                inputTensor.putFloat((Color.red(pixel) - spec.meanRgb[0]) / spec.stdRgb[0])
                inputTensor.putFloat((Color.green(pixel) - spec.meanRgb[1]) / spec.stdRgb[1])
                inputTensor.putFloat((Color.blue(pixel) - spec.meanRgb[2]) / spec.stdRgb[2])
            }
        }
        inputTensor.rewind()

        if (croppedBitmap !== source && croppedBitmap !== resizedBitmap) {
            croppedBitmap.recycle()
        }
        if (resizedBitmap !== source && resizedBitmap !== croppedBitmap) {
            resizedBitmap.recycle()
        }
        return inputTensor
    }

    private fun flattenSimcc(simcc: Array<FloatArray>): FloatArray {
        val flattened = FloatArray(simcc.size * simcc[0].size)
        var offset = 0
        simcc.forEach { logits ->
            logits.copyInto(flattened, destinationOffset = offset)
            offset += logits.size
        }
        return flattened
    }

    private fun decodeKeypoints(
        simccX: FloatArray,
        simccY: FloatArray,
        metadata: PreprocessInfo
    ): List<Keypoint> {
        val keypoints = ArrayList<Keypoint>(spec.keypointCount)
        repeat(spec.keypointCount) { jointIndex ->
            val xOffset = jointIndex * spec.simccXWidth
            val yOffset = jointIndex * spec.simccYWidth

            var bestXIndex = 0
            var bestYIndex = 0
            var bestXValue = Float.NEGATIVE_INFINITY
            var bestYValue = Float.NEGATIVE_INFINITY

            for (i in 0 until spec.simccXWidth) {
                val value = simccX[xOffset + i]
                if (value > bestXValue) {
                    bestXValue = value
                    bestXIndex = i
                }
            }
            for (i in 0 until spec.simccYWidth) {
                val value = simccY[yOffset + i]
                if (value > bestYValue) {
                    bestYValue = value
                    bestYIndex = i
                }
            }

            val modelX = bestXIndex / spec.simccSplitRatio
            val modelY = bestYIndex / spec.simccSplitRatio
            val sourceX = (
                metadata.cropLeft + modelX * (metadata.cropWidth / spec.inputWidth.toFloat())
            ).coerceIn(0f, metadata.sourceWidth - 1f)
            val sourceY = (
                metadata.cropTop + modelY * (metadata.cropHeight / spec.inputHeight.toFloat())
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
            if (startPoint.score >= spec.scoreThreshold && endPoint.score >= spec.scoreThreshold) {
                canvas.drawLine(startPoint.x, startPoint.y, endPoint.x, endPoint.y, linePaint)
            }
        }

        keypoints.forEach { point ->
            if (point.score >= spec.scoreThreshold) {
                canvas.drawCircle(point.x, point.y, result.width * 0.014f, pointPaint)
                canvas.drawText(point.name, point.x + 10f, point.y - 10f, textPaint)
            }
        }

        canvas.drawText("${BenchmarkModels.pose.displayName} Pose (${backend.displayName})", result.width * 0.05f, result.height * 0.08f, textPaint)
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

    private fun buildCenteredCropRect(sourceWidth: Int, sourceHeight: Int): Rect {
        val targetAspect = spec.inputWidth.toFloat() / spec.inputHeight.toFloat()
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
        return Rect(left, top, left + cropWidth, top + cropHeight)
    }

    private fun appendPoseDiagnostic(
        baseNote: String,
        backend: ComputeBackend,
        keypoints: List<Keypoint>
    ): String {
        if (backend != ComputeBackend.GPU) {
            return baseNote
        }
        val visibleCount = keypoints.count { it.score >= spec.scoreThreshold }
        val maxScore = keypoints.maxOfOrNull { it.score } ?: 0f
        return if (visibleCount == 0) {
            "$baseNote GPU pose output had no visible keypoints at threshold ${spec.scoreThreshold}; max score=${"%.3f".format(maxScore)}."
        } else {
            "$baseNote GPU visible keypoints=$visibleCount/${spec.keypointCount}; max score=${"%.3f".format(maxScore)}."
        }
    }

    companion object {
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

    private data class PreprocessInfo(
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
}

private fun Double.rounded(scale: Int): Double {
    var factor = 1.0
    repeat(scale) {
        factor *= 10.0
    }
    return round(this * factor) / factor
}
