package com.rockenmini.socbenchmark.preview

import android.graphics.Bitmap
import android.net.Uri
import androidx.compose.runtime.Immutable
import com.rockenmini.socbenchmark.benchmark.BenchmarkPreset
import com.rockenmini.socbenchmark.benchmark.ComputeBackend

enum class InputSourceMode(val title: String) {
    IMAGE("Single Image"),
    FOLDER("Image Folder")
}

@Immutable
data class PreviewMetrics(
    val totalMs: Double,
    val preprocessMs: Double,
    val renderMs: Double,
    val postprocessMs: Double,
    val resolutionText: String,
    val backendText: String,
    val taskText: String,
    val sourceCount: Int,
    val note: String
)

data class BatchRecord(
    val fileName: String,
    val resolutionText: String,
    val totalMs: Double,
    val preprocessMs: Double,
    val renderMs: Double,
    val postprocessMs: Double
)

data class BatchSummary(
    val count: Int,
    val averageMs: Double,
    val minMs: Double,
    val maxMs: Double
)

data class DeviceInfo(
    val manufacturer: String,
    val brand: String,
    val model: String,
    val device: String,
    val product: String,
    val hardware: String,
    val board: String,
    val androidVersion: String,
    val sdkInt: Int,
    val socManufacturer: String,
    val socModel: String
)

data class PreviewRenderResult(
    val bitmap: Bitmap,
    val metrics: PreviewMetrics
)

data class SourceSelection(
    val mode: InputSourceMode,
    val displayName: String,
    val detailText: String,
    val sourceCount: Int,
    val previewBitmap: Bitmap,
    val imageUris: List<Uri>,
    val imageNames: List<String>
)

data class MainUiState(
    val selectedPreset: BenchmarkPreset = BenchmarkPreset.SEGMENTATION,
    val selectedBackend: ComputeBackend = ComputeBackend.CPU,
    val selectedInputMode: InputSourceMode = InputSourceMode.IMAGE,
    val deviceInfo: DeviceInfo,
    val isBusy: Boolean = false,
    val sourceSelection: SourceSelection? = null,
    val resultBitmap: Bitmap? = null,
    val metrics: PreviewMetrics? = null,
    val batchSummary: BatchSummary? = null,
    val latestRecords: List<BatchRecord> = emptyList(),
    val exportFilePath: String? = null,
    val renderedImageDirPath: String? = null,
    val message: String = "UI preview mode is ready. Choose backend, task, and an image or folder."
)
