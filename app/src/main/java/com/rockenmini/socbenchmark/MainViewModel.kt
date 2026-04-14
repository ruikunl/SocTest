package com.rockenmini.socbenchmark

import android.app.Application
import android.os.Build
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.rockenmini.socbenchmark.benchmark.BenchmarkPreset
import com.rockenmini.socbenchmark.benchmark.ComputeBackend
import com.rockenmini.socbenchmark.pose.PoseDetector
import com.rockenmini.socbenchmark.preview.BatchRecord
import com.rockenmini.socbenchmark.preview.BatchSummary
import com.rockenmini.socbenchmark.preview.DeviceInfo
import com.rockenmini.socbenchmark.preview.InputSourceMode
import com.rockenmini.socbenchmark.preview.MainUiState
import com.rockenmini.socbenchmark.preview.MetricStats
import com.rockenmini.socbenchmark.preview.PreviewPipeline
import com.rockenmini.socbenchmark.preview.PreviewRenderResult
import com.rockenmini.socbenchmark.preview.SourceSelection
import com.rockenmini.socbenchmark.segmentation.SegmentationDetector
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
    private val poseDetector by lazy { PoseDetector(application) }
    private val segmentationDetector by lazy { SegmentationDetector(application) }
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
                progressCurrent = 0,
                progressTotal = 0,
                progressLabel = null,
                selectedInputMode = InputSourceMode.IMAGE,
                sourceSelection = SourceSelection(
                    mode = InputSourceMode.IMAGE,
                    displayName = name,
                    detailText = "${bitmap.width} x ${bitmap.height}",
                    sourceCount = 1,
                    previewBitmap = bitmap,
                    firstImageUri = uri,
                    firstImageName = name
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
        _uiState.value = _uiState.value.copy(
            isBusy = true,
            progressCurrent = 0,
            progressTotal = 0,
            progressLabel = "Loading folder preview...",
            message = "Loading folder preview..."
        )
        viewModelScope.launch(Dispatchers.IO) {
            val inspection = inspectImageFolder(uri)
            if (inspection == null) {
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    progressCurrent = 0,
                    progressTotal = 0,
                    progressLabel = null,
                    message = "The selected folder does not contain readable image files."
                )
                return@launch
            }

            val bitmap = decodeBitmap(inspection.previewEntry.uri)
            if (bitmap == null) {
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    progressCurrent = 0,
                    progressTotal = 0,
                    progressLabel = null,
                    message = "The folder was selected, but the preview image could not be opened."
                )
                return@launch
            }

            _uiState.value = _uiState.value.copy(
                isBusy = false,
                progressCurrent = 0,
                progressTotal = 0,
                progressLabel = null,
                selectedInputMode = InputSourceMode.FOLDER,
                sourceSelection = SourceSelection(
                    mode = InputSourceMode.FOLDER,
                    displayName = inspection.folderName,
                    detailText = "${inspection.imageCount} images, preview: ${inspection.previewEntry.name}",
                    sourceCount = inspection.imageCount,
                    previewBitmap = bitmap,
                    firstImageUri = inspection.previewEntry.uri,
                    firstImageName = inspection.previewEntry.name,
                    folderUri = uri
                ),
                resultBitmap = null,
                metrics = null,
                batchSummary = null,
                latestRecords = emptyList(),
                exportFilePath = null,
                renderedImageDirPath = null,
                message = "Folder loaded. ncnn batch benchmark is ready."
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
            progressCurrent = 0,
            progressTotal = source.sourceCount,
            progressLabel = if (source.sourceCount > 1) {
                "Preparing batch run 0 / ${source.sourceCount}"
            } else {
                "Preparing single-image run..."
            },
            exportFilePath = null,
            renderedImageDirPath = null,
            message = "Running ${current.selectedPreset.title} on ${current.selectedBackend.displayName} for ${source.sourceCount} image(s)..."
        )

        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val imageEntries = when (source.mode) {
                    InputSourceMode.IMAGE -> listOf(
                        ImageEntry(
                            uri = source.firstImageUri,
                            name = source.firstImageName
                        )
                    )
                    InputSourceMode.FOLDER -> enumerateImageFolder(checkNotNull(source.folderUri))
                }
                check(imageEntries.isNotEmpty()) { "No readable image files found for batch execution." }

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

                imageEntries.forEachIndexed { index, entry ->
                    _uiState.value = _uiState.value.copy(
                        progressCurrent = index,
                        progressTotal = imageEntries.size,
                        progressLabel = "Processing ${index + 1} / ${imageEntries.size}: ${entry.name}"
                    )

                    // The first file reuses the preview bitmap already decoded during file selection.
                    // Later files are decoded lazily here so batch mode does not pre-load the whole folder.
                    val bitmap = if (index == 0) {
                        source.previewBitmap
                    } else {
                        decodeBitmap(entry.uri)
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
                            originalName = entry.name
                        )
                    )

                    val metrics = renderResult.metrics
                    records += BatchRecord(
                        fileName = entry.name,
                        resolutionText = metrics.resolutionText,
                        totalMs = metrics.totalMs,
                        preprocessMs = metrics.preprocessMs,
                        inferenceMs = metrics.inferenceMs,
                        postprocessMs = metrics.postprocessMs,
                        overlayRenderMs = metrics.overlayRenderMs
                    )

                    _uiState.value = _uiState.value.copy(
                        progressCurrent = records.size,
                        progressTotal = imageEntries.size,
                        progressLabel = "Processed ${records.size} / ${imageEntries.size}"
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
                    progressCurrent = 0,
                    progressTotal = 0,
                    progressLabel = null,
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
                    progressCurrent = 0,
                    progressTotal = 0,
                    progressLabel = null,
                    message = error.message ?: "Benchmark failed."
                )
            }
        }
    }

    override fun onCleared() {
        poseDetector.close()
        segmentationDetector.close()
        super.onCleared()
    }

    private suspend fun runBenchmarkForBitmap(
        bitmap: Bitmap,
        preset: BenchmarkPreset,
        backend: ComputeBackend,
        sourceCount: Int
    ): PreviewRenderResult {
        return when (preset) {
            BenchmarkPreset.POSE -> poseDetector.run(bitmap, backend, sourceCount)
            BenchmarkPreset.SEGMENTATION -> segmentationDetector.run(bitmap, backend, sourceCount)
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

    private fun inspectImageFolder(treeUri: Uri): FolderInspection? {
        val folderName = DocumentFile.fromTreeUri(getApplication(), treeUri)?.name ?: "Image folder"
        val previewEntry = enumerateImageFolder(treeUri, limit = 1).firstOrNull() ?: return null
        var imageCount = 0
        queryImageFolder(treeUri) { _, _, _ ->
            imageCount += 1
        }
        if (imageCount == 0) return null

        return FolderInspection(
            folderName = folderName,
            previewEntry = previewEntry,
            imageCount = imageCount
        )
    }

    private fun enumerateImageFolder(treeUri: Uri, limit: Int = Int.MAX_VALUE): List<ImageEntry> {
        val imageEntries = mutableListOf<ImageEntry>()
        queryImageFolder(treeUri) { documentUri, displayName, _ ->
            if (imageEntries.size < limit) {
                imageEntries += ImageEntry(
                    uri = documentUri,
                    name = displayName
                )
            }
        }
        return imageEntries
    }

    private fun queryImageFolder(
        treeUri: Uri,
        onImage: (documentUri: Uri, displayName: String, index: Int) -> Unit
    ) {
        val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(
            treeUri,
            DocumentsContract.getTreeDocumentId(treeUri)
        )
        val projection = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE
        )
        val resolver = getApplication<Application>().contentResolver
        resolver.query(childrenUri, projection, null, null, null)?.use { cursor ->
            val documentIdColumn = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
            val displayNameColumn = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_DISPLAY_NAME)
            val mimeTypeColumn = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_MIME_TYPE)
            var imageIndex = 0
            while (cursor.moveToNext()) {
                val mimeType = cursor.getString(mimeTypeColumn) ?: continue
                if (!mimeType.startsWith("image/")) continue
                val documentId = cursor.getString(documentIdColumn) ?: continue
                val displayName = cursor.getString(displayNameColumn)?.takeIf { it.isNotBlank() }
                    ?: "image_${imageIndex + 1}"
                val documentUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, documentId)
                onImage(documentUri, displayName, imageIndex)
                imageIndex += 1
            }
        }
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

private data class ImageEntry(
    val uri: Uri,
    val name: String
)

private data class FolderInspection(
    val folderName: String,
    val previewEntry: ImageEntry,
    val imageCount: Int
)

private fun Double.rounded(scale: Int): Double {
    var factor = 1.0
    repeat(scale) {
        factor *= 10.0
    }
    return round(this * factor) / factor
}

private fun String?.orUnknown(): String = if (this.isNullOrBlank()) "unknown" else this
