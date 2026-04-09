package com.rockenmini.socbenchmark

import android.app.Application
import android.os.Build
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.provider.OpenableColumns
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.rockenmini.socbenchmark.benchmark.BenchmarkPreset
import com.rockenmini.socbenchmark.benchmark.ComputeBackend
import com.rockenmini.socbenchmark.pose.RtmposeOnnxPoseDetector
import com.rockenmini.socbenchmark.preview.BatchRecord
import com.rockenmini.socbenchmark.preview.BatchSummary
import com.rockenmini.socbenchmark.preview.DeviceInfo
import com.rockenmini.socbenchmark.preview.InputSourceMode
import com.rockenmini.socbenchmark.preview.MainUiState
import com.rockenmini.socbenchmark.preview.MetricStats
import com.rockenmini.socbenchmark.preview.PreviewPipeline
import com.rockenmini.socbenchmark.preview.PreviewRenderResult
import com.rockenmini.socbenchmark.preview.SourceSelection
import com.rockenmini.socbenchmark.segmentation.SelfieSegmentationDetector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import kotlin.math.round

class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val _uiState = MutableStateFlow(
        MainUiState(
            deviceInfo = collectDeviceInfo()
        )
    )
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()
    private val rtmposeOnnxPoseDetector by lazy { RtmposeOnnxPoseDetector(application) }
    private val selfieSegmentationDetector by lazy { SelfieSegmentationDetector(application) }
    private var lastBatchTimestamp: String? = null
    private var lastBatchNamePrefix: String? = null
    private var lastBatchDirPath: String? = null

    fun selectPreset(preset: BenchmarkPreset) {
        _uiState.value = _uiState.value.copy(selectedPreset = preset)
    }

    fun selectBackend(backend: ComputeBackend) {
        _uiState.value = _uiState.value.copy(selectedBackend = backend)
    }

    fun selectInputMode(mode: InputSourceMode) {
        _uiState.value = _uiState.value.copy(selectedInputMode = mode)
    }

    fun handleImagePicked(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val bitmap = decodeBitmap(uri)
            if (bitmap == null) {
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    message = "Unable to open the selected image."
                )
                return@launch
            }

            val name = queryDisplayName(uri) ?: "Selected image"
            _uiState.value = _uiState.value.copy(
                isBusy = false,
                selectedInputMode = InputSourceMode.IMAGE,
                sourceSelection = SourceSelection(
                    mode = InputSourceMode.IMAGE,
                    displayName = name,
                    detailText = "${bitmap.width} x ${bitmap.height}",
                    sourceCount = 1,
                    previewBitmap = bitmap,
                    imageUris = listOf(uri),
                    imageNames = listOf(name)
                ),
                resultBitmap = null,
                metrics = null,
                batchSummary = null,
                latestRecords = emptyList(),
                exportFilePath = null,
                renderedImageDirPath = null,
                message = "Image loaded. You can now run benchmark."
            )
        }
    }

    fun handleFolderPicked(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val tree = DocumentFile.fromTreeUri(getApplication(), uri)
            val imageFiles = tree
                ?.listFiles()
                ?.filter { file -> file.isFile && (file.type?.startsWith("image/") == true) }
                ?.sortedBy { it.name ?: "" }
                .orEmpty()

            if (imageFiles.isEmpty()) {
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    message = "The selected folder does not contain readable image files."
                )
                return@launch
            }

            val previewFile = imageFiles.first()
            val bitmap = decodeBitmap(previewFile.uri)
            if (bitmap == null) {
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    message = "The folder was selected, but the preview image could not be opened."
                )
                return@launch
            }

            val folderName = tree?.name ?: "Image folder"
            val previewName = previewFile.name ?: "preview image"
            _uiState.value = _uiState.value.copy(
                isBusy = false,
                selectedInputMode = InputSourceMode.FOLDER,
                sourceSelection = SourceSelection(
                    mode = InputSourceMode.FOLDER,
                    displayName = folderName,
                    detailText = "${imageFiles.size} images, preview: $previewName",
                    sourceCount = imageFiles.size,
                    previewBitmap = bitmap,
                    imageUris = imageFiles.map { it.uri },
                    imageNames = imageFiles.map { it.name ?: "image" }
                ),
                resultBitmap = null,
                metrics = null,
                batchSummary = null,
                latestRecords = emptyList(),
                exportFilePath = null,
                renderedImageDirPath = null,
                message = "Folder loaded. Batch benchmark is ready."
            )
        }
    }

    fun runPreview() {
        val current = _uiState.value
        val source = current.sourceSelection ?: run {
            _uiState.value = current.copy(message = "Please select an image or a folder first.")
            return
        }
        if (current.isBusy) return

        _uiState.value = current.copy(
            isBusy = true,
            exportFilePath = null,
            renderedImageDirPath = null,
            message = "Running ${current.selectedPreset.title} on ${current.selectedBackend.displayName} for ${source.sourceCount} image(s)..."
        )

        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val records = mutableListOf<BatchRecord>()
                var firstRender: PreviewRenderResult? = null
                val batchTimestamp = buildBatchTimestamp()
                val batchNamePrefix = buildBatchNamePrefix(
                    preset = current.selectedPreset,
                    backend = current.selectedBackend
                )
                val renderDir = createRenderedImageDir(batchNamePrefix, batchTimestamp)
                lastBatchTimestamp = batchTimestamp
                lastBatchNamePrefix = batchNamePrefix
                lastBatchDirPath = renderDir.absolutePath

                source.imageUris.forEachIndexed { index, uri ->
                    // The first file reuses the preview bitmap already decoded during file selection.
                    // Later files are decoded lazily here so batch mode does not pre-load the whole folder.
                    val bitmap = if (index == 0) {
                        source.previewBitmap
                    } else {
                        decodeBitmap(uri)
                    } ?: return@forEachIndexed

                    // The detector itself returns both the rendered bitmap and the timing breakdown
                    // for preprocess / inference / postprocess.
                    val renderResult = runBenchmarkForBitmap(
                        bitmap = bitmap,
                        preset = current.selectedPreset,
                        backend = current.selectedBackend,
                        sourceCount = source.sourceCount
                    )

                    if (firstRender == null) {
                        firstRender = renderResult
                    }

                    saveRenderedBitmap(
                        bitmap = renderResult.bitmap,
                        outputDir = renderDir,
                        fileName = buildRenderedFileName(
                            originalName = source.imageNames.getOrElse(index) { "image_${index + 1}" }
                        )
                    )

                    val metrics = renderResult.metrics
                    records += BatchRecord(
                        fileName = source.imageNames.getOrElse(index) { "image_${index + 1}" },
                        resolutionText = metrics.resolutionText,
                        totalMs = metrics.totalMs,
                        preprocessMs = metrics.preprocessMs,
                        inferenceMs = metrics.inferenceMs,
                        postprocessMs = metrics.postprocessMs,
                        overlayRenderMs = metrics.overlayRenderMs
                    )
                }

                val summary = buildBatchSummary(records)
                val first = checkNotNull(firstRender)
                val exportPath = exportResults(
                    selectedPreset = current.selectedPreset,
                    selectedBackend = current.selectedBackend,
                    selectedInputMode = current.selectedInputMode,
                    device = current.deviceInfo,
                    records = records,
                    summary = summary,
                    outputDirPath = renderDir.absolutePath
                )
                BatchExecutionResult(first, records, summary, renderDir.absolutePath, exportPath)
            }.onSuccess { result ->
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    resultBitmap = result.firstRender.bitmap,
                    metrics = result.firstRender.metrics.copy(sourceCount = result.records.size),
                    batchSummary = result.summary,
                    latestRecords = result.records,
                    exportFilePath = result.exportFilePath,
                    renderedImageDirPath = result.renderedImageDirPath,
                    message = if (result.records.size > 1) {
                        "Batch benchmark complete. ${result.records.size} images processed and CSV exported."
                    } else {
                        "Single image benchmark complete and CSV exported."
                    }
                )
            }.onFailure { error ->
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    message = error.message ?: "Benchmark failed."
                )
            }
        }
    }

    override fun onCleared() {
        rtmposeOnnxPoseDetector.close()
        selfieSegmentationDetector.close()
        super.onCleared()
    }

    private suspend fun runBenchmarkForBitmap(
        bitmap: Bitmap,
        preset: BenchmarkPreset,
        backend: ComputeBackend,
        sourceCount: Int
    ): PreviewRenderResult {
        return when (preset) {
            BenchmarkPreset.POSE -> rtmposeOnnxPoseDetector.run(bitmap, backend, sourceCount)
            BenchmarkPreset.SEGMENTATION -> selfieSegmentationDetector.run(bitmap, backend, sourceCount)
            BenchmarkPreset.POINT_CLOUD -> PreviewPipeline.render(bitmap, preset, backend, sourceCount)
        }
    }

    private fun buildBatchSummary(records: List<BatchRecord>): BatchSummary {
        return BatchSummary(
            count = records.size,
            total = buildMetricStats(records.map { it.totalMs }),
            preprocess = buildMetricStats(records.map { it.preprocessMs }),
            inference = buildMetricStats(records.map { it.inferenceMs }),
            postprocess = buildMetricStats(records.map { it.postprocessMs }),
            overlayRender = buildMetricStats(records.map { it.overlayRenderMs })
        )
    }

    private fun buildMetricStats(values: List<Double>): MetricStats {
        return MetricStats(
            averageMs = values.average().rounded(2),
            minMs = (values.minOrNull() ?: 0.0).rounded(2),
            maxMs = (values.maxOrNull() ?: 0.0).rounded(2)
        )
    }

    private fun exportResults(
        selectedPreset: BenchmarkPreset,
        selectedBackend: ComputeBackend,
        selectedInputMode: InputSourceMode,
        device: DeviceInfo,
        records: List<BatchRecord>,
        summary: BatchSummary,
        outputDirPath: String
    ): String {
        val timestamp = lastBatchTimestamp ?: buildBatchTimestamp()
        val filePrefix = lastBatchNamePrefix ?: buildBatchNamePrefix(
            preset = selectedPreset,
            backend = selectedBackend
        )
        val exportDir = File(outputDirPath).apply { mkdirs() }
        val file = File(exportDir, "${filePrefix}_${timestamp}.csv")

        buildString {
            appendLine("device_key,device_value")
            appendLine("manufacturer,${csvCell(device.manufacturer)}")
            appendLine("brand,${csvCell(device.brand)}")
            appendLine("model,${csvCell(device.model)}")
            appendLine("device,${csvCell(device.device)}")
            appendLine("product,${csvCell(device.product)}")
            appendLine("hardware,${csvCell(device.hardware)}")
            appendLine("board,${csvCell(device.board)}")
            appendLine("android_version,${csvCell(device.androidVersion)}")
            appendLine("sdk_int,${device.sdkInt}")
            appendLine("soc_manufacturer,${csvCell(device.socManufacturer)}")
            appendLine("soc_model,${csvCell(device.socModel)}")
            appendLine()
            appendLine("task,backend,input_mode,source_count")
            appendLine(
                listOf(
                    csvCell(selectedPreset.title),
                    csvCell(selectedBackend.displayName),
                    csvCell(selectedInputMode.title),
                    records.size.toString()
                ).joinToString(",")
            )
            appendLine()
            appendLine("file_name,resolution,total_ms,preprocess_ms,inference_ms,postprocess_ms,overlay_render_ms")
            records.forEach { record ->
                appendLine(
                    listOf(
                        csvCell(record.fileName),
                        csvCell(record.resolutionText),
                        record.totalMs.toString(),
                        record.preprocessMs.toString(),
                        record.inferenceMs.toString(),
                        record.postprocessMs.toString(),
                        record.overlayRenderMs.toString()
                    ).joinToString(",")
                )
            }
            appendLine()
            appendLine("summary_type,total_ms,preprocess_ms,inference_ms,postprocess_ms,overlay_render_ms")
            appendLine(
                listOf(
                    "avg",
                    summary.total.averageMs.toString(),
                    summary.preprocess.averageMs.toString(),
                    summary.inference.averageMs.toString(),
                    summary.postprocess.averageMs.toString(),
                    summary.overlayRender.averageMs.toString()
                ).joinToString(",")
            )
            appendLine(
                listOf(
                    "max",
                    summary.total.maxMs.toString(),
                    summary.preprocess.maxMs.toString(),
                    summary.inference.maxMs.toString(),
                    summary.postprocess.maxMs.toString(),
                    summary.overlayRender.maxMs.toString()
                ).joinToString(",")
            )
            appendLine(
                listOf(
                    "min",
                    summary.total.minMs.toString(),
                    summary.preprocess.minMs.toString(),
                    summary.inference.minMs.toString(),
                    summary.postprocess.minMs.toString(),
                    summary.overlayRender.minMs.toString()
                ).joinToString(",")
            )
        }.also { content ->
            file.writeText(content)
        }

        return file.absolutePath
    }

    private fun decodeBitmap(uri: Uri): Bitmap? {
        val resolver = getApplication<Application>().contentResolver
        return runCatching {
            val source = ImageDecoder.createSource(resolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                val maxEdge = 1600
                val width = info.size.width
                val height = info.size.height
                val scale = maxOf(width, height).toFloat() / maxEdge.toFloat()
                if (scale > 1f) {
                    decoder.setTargetSize(
                        (width / scale).toInt().coerceAtLeast(1),
                        (height / scale).toInt().coerceAtLeast(1)
                    )
                }
            }
        }.getOrNull()
    }

    private fun buildBatchTimestamp(): String {
        return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
    }

    private fun buildBatchNamePrefix(
        preset: BenchmarkPreset,
        backend: ComputeBackend
    ): String {
        val task = preset.title.lowercase()
            .replace(Regex("[^a-z0-9]+"), "_")
            .trim('_')
        val backendName = backend.displayName.lowercase()
            .replace(Regex("[^a-z0-9]+"), "_")
            .trim('_')
        return "${task}_${backendName}"
    }

    private fun createRenderedImageDir(prefix: String, timestamp: String): File {
        return File(
            getApplication<Application>().getExternalFilesDir("renders"),
            "${prefix}_$timestamp"
        ).apply { mkdirs() }
    }

    private fun saveRenderedBitmap(bitmap: Bitmap, outputDir: File, fileName: String) {
        val outputFile = File(outputDir, fileName)
        FileOutputStream(outputFile).use { stream ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
            stream.flush()
        }
    }

    private fun buildRenderedFileName(originalName: String): String {
        val base = originalName.substringBeforeLast(".")
            .replace(Regex("[^A-Za-z0-9._-]"), "_")
            .ifBlank { "image" }
        return "${base}_rendered.png"
    }

    private fun queryDisplayName(uri: Uri): String? {
        val resolver = getApplication<Application>().contentResolver
        return resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor ->
                val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (index >= 0 && cursor.moveToFirst()) cursor.getString(index) else null
            }
    }

    private fun csvCell(value: String): String {
        return "\"${value.replace("\"", "\"\"")}\""
    }

    private fun collectDeviceInfo(): DeviceInfo {
        val socManufacturer = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MANUFACTURER ?: "unknown"
        } else {
            Build.HARDWARE ?: "unknown"
        }
        val socModel = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL ?: Build.HARDWARE ?: "unknown"
        } else {
            Build.HARDWARE ?: "unknown"
        }

        return DeviceInfo(
            manufacturer = Build.MANUFACTURER.orUnknown(),
            brand = Build.BRAND.orUnknown(),
            model = Build.MODEL.orUnknown(),
            device = Build.DEVICE.orUnknown(),
            product = Build.PRODUCT.orUnknown(),
            hardware = Build.HARDWARE.orUnknown(),
            board = Build.BOARD.orUnknown(),
            androidVersion = Build.VERSION.RELEASE.orUnknown(),
            sdkInt = Build.VERSION.SDK_INT,
            socManufacturer = socManufacturer.orUnknown(),
            socModel = socModel.orUnknown()
        )
    }
}

private data class BatchExecutionResult(
    val firstRender: PreviewRenderResult,
    val records: List<BatchRecord>,
    val summary: BatchSummary,
    val renderedImageDirPath: String,
    val exportFilePath: String
)

private fun Double.rounded(scale: Int): Double {
    var factor = 1.0
    repeat(scale) {
        factor *= 10.0
    }
    return round(this * factor) / factor
}

private fun String?.orUnknown(): String = if (this.isNullOrBlank()) "unknown" else this
