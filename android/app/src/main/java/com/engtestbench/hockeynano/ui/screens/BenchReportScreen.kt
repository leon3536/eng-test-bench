package com.engtestbench.hockeynano.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.Send
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.Http
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.engtestbench.hockeynano.core.bench.BenchSeedReport
import com.engtestbench.hockeynano.core.bench.HarnessCell
import com.engtestbench.hockeynano.data.BenchSeedRepository
import com.engtestbench.hockeynano.ui.theme.BenchGreen
import com.engtestbench.hockeynano.ui.theme.BorderSubtle
import com.engtestbench.hockeynano.ui.theme.DangerRed
import com.engtestbench.hockeynano.ui.theme.ObsidianBg
import com.engtestbench.hockeynano.ui.theme.PanelBg
import com.engtestbench.hockeynano.ui.theme.PanelBgElevated
import com.engtestbench.hockeynano.ui.theme.TargetGold
import com.engtestbench.hockeynano.ui.theme.TeamACyan
import com.engtestbench.hockeynano.ui.theme.TeamBOrange
import com.engtestbench.hockeynano.ui.theme.TextMuted
import com.engtestbench.hockeynano.ui.theme.TextPrimary
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
import java.util.Locale

@Composable
fun BenchReportScreen(
    benchSeedRepository: BenchSeedRepository,
    nanoStatus: String,
    onAskAssistant: suspend (question: String, reportContext: String) -> String,
    modifier: Modifier = Modifier
) {
    val scope = rememberCoroutineScope()
    var report by remember { mutableStateOf<BenchSeedReport?>(null) }
    var loadError by remember { mutableStateOf<String?>(null) }
    var selectedCell by remember { mutableStateOf<HarnessCell?>(null) }

    var questionText by remember { mutableStateOf("") }
    var assistantAnswer by remember { mutableStateOf<String?>(null) }
    var isAskingNano by remember { mutableStateOf(false) }

    var serverUrl by remember { mutableStateOf("http://10.0.2.2:8787") }
    var probeOutput by remember { mutableStateOf<String?>(null) }
    var isProbing by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        runCatching { benchSeedRepository.loadSeedReport() }
            .onSuccess { loaded ->
                report = loaded
                selectedCell = loaded.cells.firstOrNull()
            }
            .onFailure {
                if (it is CancellationException) throw it
                loadError = it.message ?: "Failed to load bench_summary_seed.json"
            }
    }

    val currentReport = report

    fun submitQuestion(q: String) {
        val trimmed = q.trim()
        val rep = currentReport ?: return
        if (trimmed.isEmpty() || isAskingNano) return
        scope.launch {
            isAskingNano = true
            try {
                val ctx = buildCompactReportContext(rep)
                assistantAnswer = runCatching { onAskAssistant(trimmed, ctx) }
                    .onFailure { if (it is CancellationException) throw it }
                    .getOrElse { "Assistant error: ${it.message ?: "unavailable"}" }
            } finally {
                isAskingNano = false
            }
        }
    }

    LazyColumn(
        modifier = modifier
            .fillMaxSize()
            .background(ObsidianBg)
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = PaddingValues(bottom = 16.dp)
    ) {
        if (currentReport == null) {
            item(key = "loading") {
                Surface(
                    color = PanelBg,
                    shape = RoundedCornerShape(10.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
                ) {
                    Text(
                        text = loadError ?: "Loading eng-test-bench seed report…",
                        color = if (loadError != null) DangerRed else TextMuted,
                        fontSize = 13.sp,
                        modifier = Modifier.padding(14.dp)
                    )
                }
            }
        } else {
            // 1. Header & KPI strip
            item(key = "header") {
                BenchHeaderPanel(report = currentReport)
            }

            // 2. Ask Gemini Nano Bench Assistant
            item(key = "assistant") {
                NanoAssistantPanel(
                    nanoStatus = nanoStatus,
                    question = questionText,
                    onQuestionChange = { questionText = it },
                    isAsking = isAskingNano,
                    answer = assistantAnswer,
                    onQuickPrompt = { prompt ->
                        questionText = prompt
                        submitQuestion(prompt)
                    },
                    onSubmit = { submitQuestion(questionText) }
                )
            }

            // 3. 11x9 Matrix Table
            item(key = "matrix") {
                BenchMatrixCard(
                    report = currentReport,
                    selectedCell = selectedCell,
                    onSelectCell = { selectedCell = it }
                )
            }

            // 4. Optional Live Server Probe
            item(key = "server_probe") {
                LiveServerProbePanel(
                    serverUrl = serverUrl,
                    onUrlChange = { serverUrl = it },
                    isProbing = isProbing,
                    probeOutput = probeOutput,
                    onProbe = {
                        if (!isProbing && serverUrl.isNotBlank()) {
                            scope.launch {
                                isProbing = true
                                try {
                                    val res = benchSeedRepository.probeLiveBenchServer(serverUrl.trim())
                                    probeOutput = res.fold(
                                        onSuccess = { "OK: $it" },
                                        onFailure = { "Offline: ${it.message ?: "connection failed"}" }
                                    )
                                } finally {
                                    isProbing = false
                                }
                            }
                        }
                    }
                )
            }

            // 5. Verified Findings
            item(key = "findings_header") {
                Text(
                    text = "Verified Benchmark Findings (${currentReport.findings.size})",
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary,
                    fontSize = 14.sp
                )
            }

            itemsIndexed(
                items = currentReport.findings,
                key = { idx, _ -> "finding_$idx" }
            ) { idx, finding ->
                FindingCard(index = idx + 1, text = finding)
            }
        }
    }
}

@Composable
private fun BenchHeaderPanel(report: BenchSeedReport) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(PanelBg, RoundedCornerShape(10.dp))
            .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = report.title,
            style = MaterialTheme.typography.titleMedium,
            fontSize = 17.sp,
            color = BenchGreen,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = report.subtitle,
            color = TextMuted,
            fontSize = 12.sp
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            KpiTile("TRIALS", "${report.totalTrials}", TeamACyan, Modifier.weight(1f))
            KpiTile("SPEND", String.format(Locale.US, "$%.2f", report.totalSpendUsd), TargetGold, Modifier.weight(1f))
            KpiTile("TASKS", "${report.taskCount}", BenchGreen, Modifier.weight(1f))
            KpiTile("DESIGNS", "${report.designsCount}", TextPrimary, Modifier.weight(1f))
        }
        Text(
            text = "${report.harnesses.size} Harnesses × ${report.models.size} Models • ${report.cells.size} cells",
            color = TextMuted,
            fontFamily = FontFamily.Monospace,
            fontSize = 11.sp
        )
    }
}

@Composable
private fun KpiTile(label: String, value: String, accent: Color, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .background(PanelBgElevated, RoundedCornerShape(8.dp))
            .border(1.dp, BorderSubtle, RoundedCornerShape(8.dp))
            .padding(horizontal = 8.dp, vertical = 6.dp)
    ) {
        Text(text = label, color = TextMuted, fontSize = 9.5.sp, fontFamily = FontFamily.Monospace)
        Text(text = value, color = accent, fontSize = 14.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
    }
}

@Composable
private fun BenchMatrixCard(
    report: BenchSeedReport,
    selectedCell: HarnessCell?,
    onSelectCell: (HarnessCell) -> Unit
) {
    val cellLookup = remember(report.cells) {
        report.cells.associateBy { it.harness to it.model }
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(PanelBg, RoundedCornerShape(10.dp))
            .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
            .padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "Harness × Model Matrix (${report.harnesses.size}×${report.models.size})",
            color = TextPrimary,
            fontSize = 14.sp,
            fontWeight = FontWeight.SemiBold
        )
        Text(
            text = "Scroll horizontally to inspect all 9 models • Tap any cell for metrics",
            color = TextMuted,
            fontSize = 11.sp
        )

        val hScroll = rememberScrollState()
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(hScroll),
            verticalArrangement = Arrangement.spacedBy(3.dp)
        ) {
            // Header row
            Row(horizontalArrangement = Arrangement.spacedBy(3.dp)) {
                Box(
                    modifier = Modifier
                        .width(116.dp)
                        .background(PanelBgElevated, RoundedCornerShape(5.dp))
                        .padding(6.dp)
                ) {
                    Text("Harness \\ Model", color = TextMuted, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                }
                report.models.forEach { modelName ->
                    Box(
                        modifier = Modifier
                            .width(86.dp)
                            .background(PanelBgElevated, RoundedCornerShape(5.dp))
                            .padding(horizontal = 5.dp, vertical = 6.dp)
                    ) {
                        Text(
                            text = modelName,
                            color = TextPrimary,
                            fontSize = 9.5.sp,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }

            // Matrix rows
            report.harnesses.forEach { harness ->
                Row(
                    horizontalArrangement = Arrangement.spacedBy(3.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .width(116.dp)
                            .height(44.dp)
                            .background(PanelBgElevated, RoundedCornerShape(5.dp))
                            .padding(horizontal = 6.dp, vertical = 4.dp),
                        contentAlignment = Alignment.CenterStart
                    ) {
                        Text(
                            text = harness,
                            color = TextPrimary,
                            fontSize = 10.5.sp,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }

                    report.models.forEach { model ->
                        val cell = cellLookup[harness to model]
                        MatrixCellView(
                            cell = cell,
                            isSelected = cell != null && cell == selectedCell,
                            onClick = { if (cell != null) onSelectCell(cell) }
                        )
                    }
                }
            }
        }

        selectedCell?.let { c ->
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(PanelBgElevated, RoundedCornerShape(8.dp))
                    .border(1.dp, BenchGreen.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
                    .padding(9.dp),
                verticalArrangement = Arrangement.spacedBy(3.dp)
            ) {
                Text(
                    text = "${c.harness} × ${c.model}",
                    color = BenchGreen,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold
                )
                val costStr = c.costUsd?.let { String.format(Locale.US, "$%.4f/task", it) } ?: "N/A"
                val secStr = c.medianSec?.let { String.format(Locale.US, "%.1fs median", it) } ?: "N/A"
                Text(
                    text = "Passed ${c.passed}/${c.total} • $costStr • $secStr • Lookups=${c.lookupFail}" +
                        (c.badge?.let { " • [$it]" } ?: ""),
                    color = TextPrimary,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 11.sp
                )
            }
        }
    }
}

@Composable
private fun MatrixCellView(
    cell: HarnessCell?,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    if (cell == null) {
        Box(
            modifier = Modifier
                .width(86.dp)
                .height(44.dp)
                .background(ObsidianBg.copy(alpha = 0.7f), RoundedCornerShape(5.dp))
                .border(1.dp, BorderSubtle.copy(alpha = 0.4f), RoundedCornerShape(5.dp)),
            contentAlignment = Alignment.Center
        ) {
            Text("—", color = TextMuted, fontSize = 11.sp)
        }
        return
    }

    val ratio = cell.passed.toFloat() / cell.total.coerceAtLeast(1).toFloat()
    val scoreColor = when {
        ratio >= 0.96f -> BenchGreen
        ratio >= 0.85f -> TeamACyan
        ratio >= 0.70f -> TargetGold
        else -> TeamBOrange
    }
    val hasWinner = !cell.badge.isNullOrBlank()
    val borderColor = when {
        isSelected -> TargetGold
        hasWinner -> BenchGreen.copy(alpha = 0.8f)
        else -> BorderSubtle
    }

    Column(
        modifier = Modifier
            .width(86.dp)
            .height(44.dp)
            .background(scoreColor.copy(alpha = 0.11f), RoundedCornerShape(5.dp))
            .border(if (isSelected) 1.5.dp else 1.dp, borderColor, RoundedCornerShape(5.dp))
            .clickable { onClick() }
            .padding(horizontal = 5.dp, vertical = 3.dp),
        verticalArrangement = Arrangement.Center
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "${cell.passed}/${cell.total}",
                color = scoreColor,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold,
                fontSize = 11.sp
            )
            when {
                cell.lookupFail > 0 -> Text("⚠${cell.lookupFail}", color = DangerRed, fontSize = 9.sp)
                hasWinner -> Text("★", color = TargetGold, fontSize = 10.sp)
            }
        }
        val costText = cell.costUsd?.let { String.format(Locale.US, "$%.3f", it) } ?: "—"
        val secText = cell.medianSec?.let { String.format(Locale.US, "%.0fs", it) } ?: ""
        Text(
            text = "$costText $secText",
            color = TextMuted,
            fontFamily = FontFamily.Monospace,
            fontSize = 9.sp,
            maxLines = 1
        )
    }
}

@Composable
private fun NanoAssistantPanel(
    nanoStatus: String,
    question: String,
    onQuestionChange: (String) -> Unit,
    isAsking: Boolean,
    answer: String?,
    onQuickPrompt: (String) -> Unit,
    onSubmit: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(PanelBg, RoundedCornerShape(10.dp))
            .border(1.dp, TeamACyan.copy(alpha = 0.45f), RoundedCornerShape(10.dp))
            .padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Icon(Icons.Rounded.AutoAwesome, contentDescription = null, tint = TeamACyan, modifier = Modifier.size(16.dp))
                Text(
                    text = "Ask Gemini Nano Bench Assistant",
                    color = TextPrimary,
                    fontSize = 13.5.sp,
                    fontWeight = FontWeight.Bold
                )
            }
            Text(
                text = nanoStatus.uppercase(Locale.US),
                color = BenchGreen,
                fontFamily = FontFamily.Monospace,
                fontSize = 10.sp
            )
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            listOf(
                "Which harness + model is cheapest with >=26/27 pass?",
                "Why did Droid fail 3 server tasks?",
                "Best harness for open models?"
            ).forEach { chipPrompt ->
                AssistChip(
                    onClick = { onQuickPrompt(chipPrompt) },
                    label = { Text(chipPrompt, fontSize = 10.5.sp, maxLines = 1) }
                )
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedTextField(
                value = question,
                onValueChange = onQuestionChange,
                placeholder = { Text("Ask about harnesses, cost, latency…", fontSize = 12.sp) },
                singleLine = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { onSubmit() }),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = TeamACyan,
                    unfocusedBorderColor = BorderSubtle,
                    focusedTextColor = TextPrimary,
                    unfocusedTextColor = TextPrimary
                ),
                modifier = Modifier.weight(1f)
            )
            Button(
                onClick = onSubmit,
                enabled = question.isNotBlank() && !isAsking,
                colors = ButtonDefaults.buttonColors(
                    containerColor = TeamACyan,
                    contentColor = ObsidianBg
                ),
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 10.dp)
            ) {
                if (isAsking) {
                    CircularProgressIndicator(
                        color = ObsidianBg,
                        strokeWidth = 2.dp,
                        modifier = Modifier.size(16.dp)
                    )
                } else {
                    Icon(Icons.AutoMirrored.Rounded.Send, contentDescription = "Ask", modifier = Modifier.size(16.dp))
                }
            }
        }

        if (!answer.isNullOrBlank()) {
            Surface(
                color = PanelBgElevated,
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .border(1.dp, BorderSubtle, RoundedCornerShape(8.dp))
            ) {
                Text(
                    text = answer,
                    color = TextPrimary,
                    fontSize = 12.sp,
                    modifier = Modifier.padding(10.dp)
                )
            }
        }
    }
}

@Composable
private fun LiveServerProbePanel(
    serverUrl: String,
    onUrlChange: (String) -> Unit,
    isProbing: Boolean,
    probeOutput: String?,
    onProbe: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(PanelBg, RoundedCornerShape(10.dp))
            .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
            .padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Icon(Icons.Rounded.Http, contentDescription = null, tint = BenchGreen, modifier = Modifier.size(16.dp))
            Text("Optional Live eng-test-bench Server Probe", color = TextPrimary, fontSize = 12.5.sp, fontWeight = FontWeight.SemiBold)
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedTextField(
                value = serverUrl,
                onValueChange = onUrlChange,
                singleLine = true,
                label = { Text("Server URL", fontSize = 10.sp) },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = BenchGreen,
                    unfocusedBorderColor = BorderSubtle,
                    focusedTextColor = TextPrimary,
                    unfocusedTextColor = TextPrimary
                ),
                modifier = Modifier.weight(1f)
            )
            OutlinedButton(
                onClick = onProbe,
                enabled = !isProbing && serverUrl.isNotBlank()
            ) {
                Text(if (isProbing) "…" else "Probe", color = BenchGreen, fontSize = 12.sp)
            }
        }
        val probe = probeOutput
        if (!probe.isNullOrBlank()) {
            Text(
                text = probe,
                color = if (probe.startsWith("OK")) BenchGreen else TextMuted,
                fontFamily = FontFamily.Monospace,
                fontSize = 11.sp
            )
        }
    }
}

@Composable
private fun FindingCard(index: Int, text: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(PanelBg, RoundedCornerShape(8.dp))
            .border(1.dp, BorderSubtle, RoundedCornerShape(8.dp))
            .padding(10.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Surface(
            color = BenchGreen.copy(alpha = 0.2f),
            shape = RoundedCornerShape(4.dp)
        ) {
            Text(
                text = "#$index",
                color = BenchGreen,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold,
                fontSize = 11.sp,
                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
            )
        }
        Text(
            text = text,
            color = TextPrimary,
            fontSize = 12.sp,
            modifier = Modifier.weight(1f)
        )
    }
}

private fun buildCompactReportContext(report: BenchSeedReport): String {
    val topPassCheap = report.cells
        .filter { it.passed >= 26 && it.costUsd != null }
        .sortedBy { it.costUsd }
        .take(5)
        .joinToString("; ") { c ->
            "${c.harness}+${c.model}: ${c.passed}/${c.total}, $${String.format(Locale.US, "%.4f", c.costUsd ?: 0.0)}, ${c.medianSec ?: 0.0}s"
        }
    val winners = report.cells
        .filter { !it.badge.isNullOrBlank() }
        .take(8)
        .joinToString("; ") { "${it.harness}+${it.model}=${it.passed}/${it.total}(${it.badge})" }
    val findingsStr = report.findings.take(7).joinToString(" | ")
    return "Title=${report.title}; Trials=${report.totalTrials}; Spend=$${report.totalSpendUsd}; Tasks=${report.taskCount}; " +
        "TopLowCost26Plus=[$topPassCheap]; ColumnWinners=[$winners]; Findings=[$findingsStr]"
}
