package com.rockenmini.socbenchmark

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts.OpenDocument
import androidx.activity.result.contract.ActivityResultContracts.OpenDocumentTree
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.rockenmini.socbenchmark.benchmark.BenchmarkPreset
import com.rockenmini.socbenchmark.benchmark.ComputeBackend
import com.rockenmini.socbenchmark.preview.InputSourceMode

class MainActivity : ComponentActivity() {
    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    BenchmarkScreen(
                        viewModel = viewModel,
                        onPersistableUri = ::takeReadPermission
                    )
                }
            }
        }
    }

    private fun takeReadPermission(uri: Uri) {
        runCatching {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        }
    }
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun BenchmarkScreen(
    viewModel: MainViewModel,
    onPersistableUri: (Uri) -> Unit
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    val imagePicker = rememberLauncherForActivityResult(OpenDocument()) { uri ->
        uri?.let {
            onPersistableUri(it)
            viewModel.handleImagePicked(it)
        }
    }

    val folderPicker = rememberLauncherForActivityResult(OpenDocumentTree()) { uri ->
        uri?.let {
            onPersistableUri(it)
            viewModel.handleFolderPicked(it)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "SoC Benchmark",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "选择后端、任务、图片或文件夹，查看结果图、批处理统计和导出结果。",
            style = MaterialTheme.typography.bodyLarge
        )

        SelectorCard(
            title = "Compute Backend",
            description = "当前 ONNX Runtime 版本支持 CPU 与 NNAPI 路径。GPU 选项会保留在界面中，但会明确提示是否回退到 CPU。"
        ) {
            ComputeBackend.entries.forEach { backend ->
                FilterChip(
                    selected = uiState.selectedBackend == backend,
                    onClick = { viewModel.selectBackend(backend) },
                    label = { Text(backend.displayName) }
                )
            }
        }

        SelectorCard(
            title = "Task Preset",
            description = uiState.selectedPreset.summary
        ) {
            BenchmarkPreset.entries.forEach { preset ->
                FilterChip(
                    selected = uiState.selectedPreset == preset,
                    onClick = { viewModel.selectPreset(preset) },
                    label = { Text(preset.title) }
                )
            }
        }

        SelectorCard(
            title = "Input Source",
            description = "可选择单张图片或整个图片文件夹。文件夹模式会先预览第一张图片。"
        ) {
            InputSourceMode.entries.forEach { mode ->
                FilterChip(
                    selected = uiState.selectedInputMode == mode,
                    onClick = { viewModel.selectInputMode(mode) },
                    label = { Text(mode.title) }
                )
            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text("Device Info", style = MaterialTheme.typography.titleMedium)
                InfoLine("Manufacturer", uiState.deviceInfo.manufacturer)
                InfoLine("Brand", uiState.deviceInfo.brand)
                InfoLine("Model", uiState.deviceInfo.model)
                InfoLine("Android", "${uiState.deviceInfo.androidVersion} (SDK ${uiState.deviceInfo.sdkInt})")
                InfoLine("SoC Vendor", uiState.deviceInfo.socManufacturer)
                InfoLine("SoC Model", uiState.deviceInfo.socModel)
                InfoLine("Hardware", uiState.deviceInfo.hardware)
            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text("Select Test Data", style = MaterialTheme.typography.titleMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(
                        onClick = { imagePicker.launch(arrayOf("image/*")) },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Choose Image")
                    }
                    OutlinedButton(
                        onClick = { folderPicker.launch(null) },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Choose Folder")
                    }
                }

                uiState.sourceSelection?.let { selection ->
                    Text(
                        text = selection.displayName,
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = selection.detailText,
                        style = MaterialTheme.typography.bodyMedium
                    )
                } ?: Text(
                    text = "No image or folder selected yet.",
                    style = MaterialTheme.typography.bodyMedium
                )

                Button(
                    onClick = viewModel::runPreview,
                    enabled = !uiState.isBusy && uiState.sourceSelection != null,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(if (uiState.isBusy) "Processing..." else "Run Benchmark")
                }

            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text("Image Preview", style = MaterialTheme.typography.titleMedium)
                ImagePanel(
                    title = "Selected Image",
                    bitmap = uiState.sourceSelection?.previewBitmap
                )
                ImagePanel(
                    title = "Preview Result",
                    bitmap = uiState.resultBitmap
                )
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text("Runtime Info", style = MaterialTheme.typography.titleMedium)
                Text(uiState.message, style = MaterialTheme.typography.bodyMedium)
                Text(
                    text = "This branch uses ONNX Runtime. CPU runs through XNNPACK. NNAPI is a request path and may partially accelerate or fall back. GPU may fall back when no generic ORT Android GPU EP is bundled.",
                    style = MaterialTheme.typography.bodySmall
                )
                uiState.metrics?.let { metrics ->
                    InfoLine("Backend", metrics.backendText)
                    InfoLine("Task", metrics.taskText)
                    InfoLine("Resolution", metrics.resolutionText)
                    InfoLine("Source Count", metrics.sourceCount.toString())
                    InfoLine("Total Time", "${metrics.totalMs} ms")
                    InfoLine("Preprocess", "${metrics.preprocessMs} ms")
                    InfoLine("Inference", "${metrics.inferenceMs} ms")
                    InfoLine("Postprocess", "${metrics.postprocessMs} ms")
                    InfoLine("Overlay Render", "${metrics.overlayRenderMs} ms")
                    Text(
                        text = metrics.note,
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                uiState.batchSummary?.let { summary ->
                    InfoLine("Batch Count", summary.count.toString())
                    InfoLine("Total Avg", "${summary.total.averageMs} ms")
                    InfoLine("Total Min", "${summary.total.minMs} ms")
                    InfoLine("Total Max", "${summary.total.maxMs} ms")
                    InfoLine("Pre Avg", "${summary.preprocess.averageMs} ms")
                    InfoLine("Pre Min", "${summary.preprocess.minMs} ms")
                    InfoLine("Pre Max", "${summary.preprocess.maxMs} ms")
                    InfoLine("Inf Avg", "${summary.inference.averageMs} ms")
                    InfoLine("Inf Min", "${summary.inference.minMs} ms")
                    InfoLine("Inf Max", "${summary.inference.maxMs} ms")
                    InfoLine("Post Avg", "${summary.postprocess.averageMs} ms")
                    InfoLine("Post Min", "${summary.postprocess.minMs} ms")
                    InfoLine("Post Max", "${summary.postprocess.maxMs} ms")
                    InfoLine("Overlay Avg", "${summary.overlayRender.averageMs} ms")
                    InfoLine("Overlay Min", "${summary.overlayRender.minMs} ms")
                    InfoLine("Overlay Max", "${summary.overlayRender.maxMs} ms")
                }
                uiState.exportFilePath?.let { path ->
                    Text(
                        text = "Exported CSV: $path",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                uiState.renderedImageDirPath?.let { path ->
                    Text(
                        text = "Rendered images: $path",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }
    }
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun SelectorCard(
    title: String,
    description: String,
    content: @Composable () -> Unit
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(description, style = MaterialTheme.typography.bodyMedium)
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                content()
            }
        }
    }
}

@Composable
private fun ImagePanel(
    title: String,
    bitmap: Bitmap?
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(title, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.SemiBold)
        if (bitmap == null) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(220.dp),
                shape = RoundedCornerShape(18.dp)
            ) {
                Column(
                    modifier = Modifier.fillMaxSize().padding(16.dp),
                    verticalArrangement = Arrangement.Center
                ) {
                    Text("No preview yet", style = MaterialTheme.typography.bodyMedium)
                }
            }
        } else {
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = title,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(220.dp),
                contentScale = ContentScale.Fit
            )
        }
    }
}

@Composable
private fun InfoLine(label: String, value: String) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = "$label:",
            modifier = Modifier.width(100.dp),
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold
        )
        Text(text = value, style = MaterialTheme.typography.bodyMedium)
    }
}
