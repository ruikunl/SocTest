package com.rockenmini.socbenchmark.preview

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import com.rockenmini.socbenchmark.benchmark.BenchmarkPreset
import com.rockenmini.socbenchmark.benchmark.ComputeBackend
import kotlin.math.round
import kotlin.random.Random
import kotlin.system.measureNanoTime

object PreviewPipeline {
    fun render(
        source: Bitmap,
        preset: BenchmarkPreset,
        backend: ComputeBackend,
        sourceCount: Int
    ): PreviewRenderResult {
        var resultBitmap: Bitmap? = null
        var preprocessMs = 0.0
        var inferenceMs = 0.0
        var postprocessMs = 0.0
        var overlayRenderMs = 0.0

        val totalMs = measureNanoTime {
            preprocessMs = measureStepMs {
                // Keep a software copy for stable preview rendering on all devices.
                source.copy(Bitmap.Config.ARGB_8888, false)
            }

            // Preview mode has no real model, so "inference" stays at zero.
            inferenceMs = 0.0

            overlayRenderMs = measureStepMs {
                resultBitmap = when (preset) {
                    BenchmarkPreset.SEGMENTATION -> drawSegmentationPreview(source)
                    BenchmarkPreset.POSE -> drawPosePreview(source)
                    BenchmarkPreset.POINT_CLOUD -> drawPointCloudPreview(source)
                }
            }

            postprocessMs = measureStepMs {
                // No-op hook reserved for real output decoding later.
                resultBitmap = resultBitmap?.copy(Bitmap.Config.ARGB_8888, false)
            }
        } / 1_000_000.0

        val finalBitmap = checkNotNull(resultBitmap)
        val resolution = "${source.width} x ${source.height}"
        val note = when (backend) {
            ComputeBackend.CPU -> "Current preview path runs on CPU. Real model timing will reuse this panel."
            ComputeBackend.GPU -> "GPU mode is selected in UI. Real delegate inference will be wired in next."
            ComputeBackend.NNAPI -> "NPU / NNAPI mode is selected in UI. Real delegate inference will be wired in next."
        }

        return PreviewRenderResult(
            bitmap = finalBitmap,
            metrics = PreviewMetrics(
                totalMs = totalMs.rounded(2),
                preprocessMs = preprocessMs.rounded(2),
                inferenceMs = inferenceMs.rounded(2),
                postprocessMs = postprocessMs.rounded(2),
                overlayRenderMs = overlayRenderMs.rounded(2),
                resolutionText = resolution,
                backendText = backend.displayName,
                taskText = preset.title,
                sourceCount = sourceCount,
                note = note
            )
        )
    }

    private fun drawSegmentationPreview(source: Bitmap): Bitmap {
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        val overlayPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(120, 40, 218, 111)
        }
        val edgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(14, 110, 57)
            style = Paint.Style.STROKE
            strokeWidth = result.width * 0.012f
        }
        val textPaint = labelPaint(result)

        val left = result.width * 0.24f
        val top = result.height * 0.08f
        val right = result.width * 0.76f
        val bottom = result.height * 0.92f

        canvas.drawOval(left, top, right, bottom, overlayPaint)
        canvas.drawOval(left, top, right, bottom, edgePaint)
        canvas.drawText("Segmentation Preview", result.width * 0.06f, result.height * 0.1f, textPaint)
        return result
    }

    private fun drawPosePreview(source: Bitmap): Bitmap {
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(255, 140, 0)
            strokeWidth = result.width * 0.01f
            style = Paint.Style.STROKE
        }
        val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(28, 196, 255)
            style = Paint.Style.FILL
        }
        val textPaint = labelPaint(result)

        val points = listOf(
            0.5f to 0.14f,
            0.42f to 0.25f,
            0.58f to 0.25f,
            0.36f to 0.42f,
            0.64f to 0.42f,
            0.46f to 0.42f,
            0.54f to 0.42f,
            0.44f to 0.66f,
            0.56f to 0.66f,
            0.40f to 0.88f,
            0.60f to 0.88f
        ).map { (x, y) -> x * result.width to y * result.height }

        val skeleton = listOf(
            0 to 1, 0 to 2, 1 to 3, 2 to 4,
            1 to 5, 2 to 6, 5 to 6,
            5 to 7, 6 to 8, 7 to 9, 8 to 10
        )

        skeleton.forEach { (start, end) ->
            val p1 = points[start]
            val p2 = points[end]
            canvas.drawLine(p1.first, p1.second, p2.first, p2.second, linePaint)
        }
        points.forEach { (x, y) ->
            canvas.drawCircle(x, y, result.width * 0.018f, jointPaint)
        }
        canvas.drawText("Pose Preview", result.width * 0.06f, result.height * 0.1f, textPaint)
        return result
    }

    private fun drawPointCloudPreview(source: Bitmap): Bitmap {
        val result = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawColor(Color.rgb(7, 15, 24))

        val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
        }
        val textPaint = labelPaint(result)
        val random = Random(42)

        val sampleStep = maxOf(12, source.width / 48)
        for (y in 0 until source.height step sampleStep) {
            for (x in 0 until source.width step sampleStep) {
                val pixel = source.getPixel(x, y)
                val brightness = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
                val radius = 1.5f + (brightness / 255f) * 6.0f
                val depthOffset = ((brightness - 128) / 128f) * 40f
                pointPaint.color = Color.rgb(
                    80 + brightness / 3,
                    120 + brightness / 4,
                    180 + brightness / 6
                )
                val jitterX = random.nextFloat() * 10f - 5f
                val jitterY = random.nextFloat() * 10f - 5f
                canvas.drawCircle(
                    x + jitterX,
                    y + jitterY + depthOffset,
                    radius,
                    pointPaint
                )
            }
        }

        val fadePaint = Paint().apply {
            xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER)
            color = Color.argb(36, 255, 255, 255)
        }
        canvas.drawRect(0f, 0f, result.width.toFloat(), result.height.toFloat(), fadePaint)
        canvas.drawText("Point Cloud Preview", result.width * 0.06f, result.height * 0.1f, textPaint)
        return result
    }

    private fun labelPaint(bitmap: Bitmap): Paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = bitmap.width * 0.045f
        setShadowLayer(12f, 0f, 0f, Color.BLACK)
    }

    private fun measureStepMs(block: () -> Unit): Double {
        return measureNanoTime(block) / 1_000_000.0
    }
}

private fun Double.rounded(scale: Int): Double {
    var factor = 1.0
    repeat(scale) {
        factor *= 10.0
    }
    return round(this * factor) / factor
}
