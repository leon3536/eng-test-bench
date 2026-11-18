package com.engtestbench.hockeynano.ui.screens

import android.content.Intent
import android.graphics.Bitmap
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
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
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.ContentCut
import androidx.compose.material.icons.rounded.FolderOpen
import androidx.compose.material.icons.rounded.GridView
import androidx.compose.material.icons.rounded.Layers
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.Speed
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.media3.exoplayer.ExoPlayer
import com.engtestbench.hockeynano.core.model.ClipWindow30s
import com.engtestbench.hockeynano.core.model.DiscoveredPlayerSummary
import com.engtestbench.hockeynano.core.model.PipelineState
import com.engtestbench.hockeynano.core.model.TeamId
import com.engtestbench.hockeynano.core.model.TeamProfile
import com.engtestbench.hockeynano.ui.components.PlayerVideoOverlay
import com.engtestbench.hockeynano.ui.theme.BenchGreen
import com.engtestbench.hockeynano.ui.theme.BorderSubtle
import com.engtestbench.hockeynano.ui.theme.DangerRed
import com.engtestbench.hockeynano.ui.theme.ObsidianBg
import com.engtestbench.hockeynano.ui.theme.PanelBg
import com.engtestbench.hockeynano.ui.theme.PanelBgElevated
import com.engtestbench.hockeynano.ui.theme.RefGray
import com.engtestbench.hockeynano.ui.theme.TargetGold
import com.engtestbench.hockeynano.ui.theme.TeamACyan
import com.engtestbench.hockeynano.ui.theme.TeamBOrange
import com.engtestbench.hockeynano.ui.theme.TextMuted
import com.engtestbench.hockeynano.ui.theme.TextPrimary
import com.engtestbench.hockeynano.ui.theme.formatClock
import com.engtestbench.hockeynano.ui.theme.teamColor
import com.engtestbench.hockeynano.ui.theme.teamTag
import java.util.Locale

data class StudioExportUiState(
    val exportingClipId: String? = null,
    val exportLabel: String? = null,
    val progress: Float = 0f,
    val isBatchRunning: Boolean = false,
    val statusMessage: String? = null
)

@Composable
fun NanoHockeyStudioScreen(
    player: ExoPlayer,
    pipelineState: PipelineState,
    discoveredVideos: List<Pair<String, String>>,
    selectedVideoTitle: String,
    selectedVideoSource: String,
    playbackSpeed: Float,
    clipPlayRange: Pair<Double, Double>?,
    exportState: StudioExportUiState,
    hasMediaPermission: Boolean,
    onRequestMediaPermission: () -> Unit,
    onRefreshVideos: () -> Unit,
    onSelectVideo: (title: String, source: String) -> Unit,
    onToggleSpeed: (Float) -> Unit,
    onTogglePanoramic: (Boolean) -> Unit,
    onVideoDimensionsKnown: (durationSec: Double, width: Int, height: Int) -> Unit,
    onVideoFrame: suspend (Bitmap, Double) -> Unit,
    onSelectPlayer: (jerseyNumber: String?, teamFilter: TeamId?) -> Unit,
    onAssignTrack: (trackId: Int, jerseyNumber: String, team: TeamId?) -> Unit,
    onPlayClipWindow: (ClipWindow30s) -> Unit,
    onClearClipRange: () -> Unit,
    onFastSplitWindow: (ClipWindow30s) -> Unit,
    onBurnMarksWindow: (ClipWindow30s) -> Unit,
    onSplitAllWindows: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    var manualNumber by remember(pipelineState.selectedPlayerNumber) {
        mutableStateOf(pipelineState.selectedPlayerNumber.orEmpty())
    }

    val openDocLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            var displayName = "Selected Video"
            runCatching {
                context.contentResolver.query(
                    uri,
                    arrayOf(OpenableColumns.DISPLAY_NAME),
                    null,
                    null,
                    null
                )?.use { c ->
                    val idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (idx >= 0 && c.moveToFirst()) {
                        displayName = c.getString(idx) ?: displayName
                    }
                }
            }
            onSelectVideo(displayName, uri.toString())
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(ObsidianBg)
            .padding(horizontal = 8.dp, vertical = 4.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        // 1. Video selector + speed + panoramic tiling controls
        VideoSelectorRow(
            discoveredVideos = discoveredVideos,
            selectedVideoSource = selectedVideoSource,
            playbackSpeed = playbackSpeed,
            panoramicEnabled = pipelineState.telemetry.panoramicTilingEnabled,
            onRefreshVideos = onRefreshVideos,
            onPickVideo = { openDocLauncher.launch(arrayOf("video/*")) },
            onSelectVideo = onSelectVideo,
            onToggleSpeed = onToggleSpeed,
            onTogglePanoramic = onTogglePanoramic
        )

        if (!hasMediaPermission) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(PanelBgElevated, RoundedCornerShape(8.dp))
                    .border(1.dp, TargetGold.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
                    .padding(horizontal = 10.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Grant video permission to scan /sdcard/Download/HockeyClips",
                    color = TextPrimary,
                    fontSize = 11.sp,
                    modifier = Modifier.weight(1f)
                )
                TextButton(
                    onClick = onRequestMediaPermission,
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp)
                ) {
                    Text("Allow", color = TargetGold, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
            }
        }

        // 2. Synchronized TextureView + Canvas overlay player
        PlayerVideoOverlay(
            player = player,
            videoSourceKey = selectedVideoSource.ifBlank { selectedVideoTitle },
            history = pipelineState.history,
            currentFrame = pipelineState.currentFrame,
            selectedPlayerNumber = pipelineState.selectedPlayerNumber,
            selectedTeamFilter = pipelineState.selectedTeamFilter,
            clipWindows = pipelineState.selectedPlayerClips,
            clipPlayRange = clipPlayRange,
            onClearClipRange = onClearClipRange,
            onVideoDimensionsKnown = onVideoDimensionsKnown,
            onVideoFrame = onVideoFrame,
            onSelectPlayer = onSelectPlayer,
            onAssignTrack = onAssignTrack
        )

        // 3. Scrollable Telemetry + Player Jersey Picker + 30s Clip Windows
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            verticalArrangement = Arrangement.spacedBy(8.dp),
            contentPadding = PaddingValues(bottom = 12.dp)
        ) {
            item(key = "telemetry") {
                GeminiNanoTelemetryStrip(pipelineState = pipelineState)
            }

            item(key = "player_picker") {
                PlayerJerseyPickerPanel(
                    pipelineState = pipelineState,
                    manualNumber = manualNumber,
                    onManualNumberChange = { manualNumber = it.filter(Char::isDigit).take(3) },
                    onSubmitManualNumber = {
                        val clean = manualNumber.trim().trimStart('#')
                        onSelectPlayer(clean.ifBlank { null }, pipelineState.selectedTeamFilter)
                    },
                    onSelectChip = { playerSummary ->
                        val nextNum = if (pipelineState.selectedPlayerNumber == playerSummary.jerseyNumber) {
                            null
                        } else {
                            playerSummary.jerseyNumber
                        }
                        onSelectPlayer(nextNum, pipelineState.selectedTeamFilter)
                    },
                    onSelectTeamFilter = { filter ->
                        onSelectPlayer(pipelineState.selectedPlayerNumber, filter)
                    }
                )
            }

            item(key = "export_status") {
                AnimatedVisibility(
                    visible = exportState.exportingClipId != null || !exportState.statusMessage.isNullOrBlank()
                ) {
                    ExportStatusBanner(exportState = exportState)
                }
            }

            item(key = "clips_header") {
                ClipsHeaderRow(
                    selectedPlayerNumber = pipelineState.selectedPlayerNumber,
                    clipCount = pipelineState.selectedPlayerClips.size,
                    isBusy = exportState.exportingClipId != null || exportState.isBatchRunning,
                    onSplitAll = onSplitAllWindows
                )
            }

            if (pipelineState.selectedPlayerNumber.isNullOrBlank()) {
                item(key = "no_player_selected") {
                    EmptyHintCard(
                        text = "Tap a jersey # chip above, type a number, or tap any skater box on the video to find 30s clip windows."
                    )
                }
            } else if (pipelineState.selectedPlayerClips.isEmpty()) {
                item(key = "no_windows_yet") {
                    EmptyHintCard(
                        text = "Scanning video for #${pipelineState.selectedPlayerNumber}… Windows appear once ≥1.0s of visibility is tracked."
                    )
                }
            } else {
                items(
                    items = pipelineState.selectedPlayerClips,
                    key = { it.id }
                ) { clip ->
                    ClipWindow30sCard(
                        clip = clip,
                        isExportingThis = exportState.exportingClipId == clip.id,
                        exportProgress = if (exportState.exportingClipId == clip.id) exportState.progress else 0f,
                        anyExportBusy = exportState.exportingClipId != null || exportState.isBatchRunning,
                        onPlay = { onPlayClipWindow(clip) },
                        onFastSplit = { onFastSplitWindow(clip) },
                        onBurnMarks = { onBurnMarksWindow(clip) }
                    )
                }
            }
        }
    }
}

@Composable
private fun VideoSelectorRow(
    discoveredVideos: List<Pair<String, String>>,
    selectedVideoSource: String,
    playbackSpeed: Float,
    panoramicEnabled: Boolean,
    onRefreshVideos: () -> Unit,
    onPickVideo: () -> Unit,
    onSelectVideo: (String, String) -> Unit,
    onToggleSpeed: (Float) -> Unit,
    onTogglePanoramic: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        OutlinedButton(
            onClick = onPickVideo,
            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
            shape = RoundedCornerShape(8.dp),
            modifier = Modifier.height(32.dp)
        ) {
            Icon(Icons.Rounded.FolderOpen, contentDescription = null, modifier = Modifier.size(15.dp), tint = TeamACyan)
            Spacer(Modifier.width(4.dp))
            Text("Pick MP4", fontSize = 11.sp, color = TextPrimary)
        }

        IconButton(
            onClick = onRefreshVideos,
            modifier = Modifier.size(30.dp)
        ) {
            Icon(Icons.Rounded.Refresh, contentDescription = "Refresh videos", tint = TextMuted, modifier = Modifier.size(17.dp))
        }

        val is2x = playbackSpeed >= 1.5f
        FilterChip(
            selected = is2x,
            onClick = { onToggleSpeed(if (is2x) 1.0f else 2.0f) },
            label = { Text(if (is2x) "2x" else "1x", fontSize = 11.sp, fontWeight = FontWeight.Bold) },
            leadingIcon = { Icon(Icons.Rounded.Speed, contentDescription = null, modifier = Modifier.size(14.dp)) },
            colors = FilterChipDefaults.filterChipColors(
                selectedContainerColor = TargetGold.copy(alpha = 0.22f),
                selectedLabelColor = TargetGold,
                selectedLeadingIconColor = TargetGold
            ),
            modifier = Modifier.height(32.dp)
        )

        FilterChip(
            selected = panoramicEnabled,
            onClick = { onTogglePanoramic(!panoramicEnabled) },
            label = { Text("Tiling", fontSize = 11.sp) },
            leadingIcon = { Icon(Icons.Rounded.GridView, contentDescription = null, modifier = Modifier.size(14.dp)) },
            colors = FilterChipDefaults.filterChipColors(
                selectedContainerColor = BenchGreen.copy(alpha = 0.22f),
                selectedLabelColor = BenchGreen,
                selectedLeadingIconColor = BenchGreen
            ),
            modifier = Modifier.height(32.dp)
        )

        discoveredVideos.forEach { (title, src) ->
            val selected = src == selectedVideoSource
            FilterChip(
                selected = selected,
                onClick = { onSelectVideo(title, src) },
                label = {
                    Text(
                        text = title,
                        fontSize = 11.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor = BenchGreen.copy(alpha = 0.22f),
                    selectedLabelColor = BenchGreen
                ),
                modifier = Modifier.height(32.dp)
            )
        }
    }
}

@Composable
private fun GeminiNanoTelemetryStrip(pipelineState: PipelineState) {
    val tel = pipelineState.telemetry
    val statusUpper = tel.nanoStatus.uppercase(Locale.US)
    val statusColor = when {
        statusUpper.contains("READY") || statusUpper.contains("AVAIL") -> BenchGreen
        statusUpper.contains("DOWNLOAD") || statusUpper.contains("INIT") -> TargetGold
        else -> DangerRed
    }

    val currentPlayers = pipelineState.currentFrame?.players.orEmpty()
    val countA = currentPlayers.count { it.team == TeamId.TEAM_A }
    val countB = currentPlayers.count { it.team == TeamId.TEAM_B }
    val teamA = pipelineState.teams[TeamId.TEAM_A]
    val teamB = pipelineState.teams[TeamId.TEAM_B]

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(PanelBg, RoundedCornerShape(10.dp))
            .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(5.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(statusColor)
                )
                Text(
                    text = "Nano: $statusUpper",
                    color = statusColor,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold,
                    fontSize = 10.5.sp,
                    maxLines = 1,
                    softWrap = false
                )
                if (tel.lastNanoLatencyMs > 0L) {
                    Text(
                        text = "(${tel.lastNanoLatencyMs}ms)",
                        color = TextMuted,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 9.5.sp,
                        maxLines = 1,
                        softWrap = false
                    )
                }
            }

            Spacer(Modifier.weight(1f))

            Text(
                text = String.format(
                    Locale.US,
                    "FPS %.1f • YOLO %.0fms • OCR %d • Nano %d",
                    tel.fps,
                    tel.detectorMs,
                    tel.ocrReadsTotal,
                    tel.nanoCallsTotal
                ),
                color = TextMuted,
                fontFamily = FontFamily.Monospace,
                fontSize = 9.5.sp,
                maxLines = 1,
                softWrap = false,
                overflow = TextOverflow.Ellipsis
            )
        }

        // Team A vs Team B color swatches + Nano kit names + live skater counts
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            TeamSwatchBadge(
                profile = teamA,
                fallbackLabel = "Team A",
                accent = TeamACyan,
                liveCount = countA,
                modifier = Modifier.weight(1f)
            )
            Text(
                text = "vs",
                color = TextMuted,
                fontSize = 10.sp,
                modifier = Modifier.padding(horizontal = 6.dp)
            )
            TeamSwatchBadge(
                profile = teamB,
                fallbackLabel = "Team B",
                accent = TeamBOrange,
                liveCount = countB,
                modifier = Modifier.weight(1f)
            )
        }

        if (tel.lastNanoMessage.isNotBlank()) {
            Text(
                text = tel.lastNanoMessage,
                color = TextMuted,
                fontSize = 11.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
private fun TeamSwatchBadge(
    profile: TeamProfile?,
    fallbackLabel: String,
    accent: Color,
    liveCount: Int,
    modifier: Modifier = Modifier
) {
    val swatchColor = profile?.meanRgb?.takeIf { it.size >= 3 }?.let { rgb ->
        Color(
            red = rgb[0].coerceIn(0, 255),
            green = rgb[1].coerceIn(0, 255),
            blue = rgb[2].coerceIn(0, 255)
        )
    } ?: accent

    val name = profile?.displayName?.ifBlank { fallbackLabel } ?: fallbackLabel
    val star = if (profile?.nanoVerified == true) " ★" else ""

    Row(
        modifier = modifier
            .background(PanelBgElevated, RoundedCornerShape(6.dp))
            .border(1.dp, accent.copy(alpha = 0.45f), RoundedCornerShape(6.dp))
            .padding(horizontal = 7.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Box(
            modifier = Modifier
                .size(12.dp)
                .clip(RoundedCornerShape(3.dp))
                .background(swatchColor)
                .border(1.dp, Color.White.copy(alpha = 0.4f), RoundedCornerShape(3.dp))
        )
        Text(
            text = "$name$star",
            color = TextPrimary,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f)
        )
        Text(
            text = "$liveCount",
            color = accent,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Bold,
            fontSize = 11.sp
        )
    }
}

@Composable
private fun PlayerJerseyPickerPanel(
    pipelineState: PipelineState,
    manualNumber: String,
    onManualNumberChange: (String) -> Unit,
    onSubmitManualNumber: () -> Unit,
    onSelectChip: (DiscoveredPlayerSummary) -> Unit,
    onSelectTeamFilter: (TeamId?) -> Unit
) {
    val teamFilter = pipelineState.selectedTeamFilter
    val filteredPlayers = remember(pipelineState.discoveredPlayers, teamFilter) {
        if (teamFilter == null) {
            pipelineState.discoveredPlayers
        } else {
            pipelineState.discoveredPlayers.filter { it.team == teamFilter }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(PanelBg, RoundedCornerShape(10.dp))
            .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
            .padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedTextField(
                value = manualNumber,
                onValueChange = onManualNumberChange,
                placeholder = { Text("#97", fontSize = 12.sp) },
                label = { Text("Jersey #", fontSize = 11.sp) },
                singleLine = true,
                keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Number,
                    imeAction = ImeAction.Done
                ),
                keyboardActions = KeyboardActions(onDone = { onSubmitManualNumber() }),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = BenchGreen,
                    unfocusedBorderColor = BorderSubtle,
                    focusedTextColor = TextPrimary,
                    unfocusedTextColor = TextPrimary
                ),
                modifier = Modifier.width(108.dp)
            )

            Button(
                onClick = onSubmitManualNumber,
                colors = ButtonDefaults.buttonColors(
                    containerColor = BenchGreen,
                    contentColor = ObsidianBg
                ),
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                modifier = Modifier.height(40.dp)
            ) {
                Text("Track #", fontSize = 12.sp, fontWeight = FontWeight.Bold)
            }

            Row(
                modifier = Modifier
                    .weight(1f)
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                listOf(
                    null to "All",
                    TeamId.TEAM_A to "A",
                    TeamId.TEAM_B to "B"
                ).forEach { (team, label) ->
                    val active = teamFilter == team
                    val chipAccent = team?.let(::teamColor) ?: BenchGreen
                    FilterChip(
                        selected = active,
                        onClick = { onSelectTeamFilter(team) },
                        label = { Text(label, fontSize = 11.sp) },
                        colors = FilterChipDefaults.filterChipColors(
                            selectedContainerColor = chipAccent.copy(alpha = 0.22f),
                            selectedLabelColor = chipAccent
                        ),
                        modifier = Modifier.height(32.dp)
                    )
                }
            }
        }

        if (filteredPlayers.isEmpty()) {
            Text(
                text = "Auto-discovering jersey numbers via YOLO + ML Kit OCR + Gemini Nano…",
                color = TextMuted,
                fontSize = 11.sp
            )
        } else {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                filteredPlayers.forEach { p ->
                    val selected = pipelineState.selectedPlayerNumber == p.jerseyNumber &&
                        (teamFilter == null || p.team == teamFilter)
                    val accent = teamColor(p.team)
                    val star = if (p.nanoVerified) " ★" else ""
                    val seenLabel = String.format(Locale.US, "%.1fs", p.totalSeenSec)
                    Surface(
                        color = if (selected) TargetGold else PanelBgElevated,
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier
                            .border(
                                width = if (selected) 1.5.dp else 1.dp,
                                color = if (selected) TargetGold else accent.copy(alpha = 0.65f),
                                shape = RoundedCornerShape(8.dp)
                            )
                            .clickable { onSelectChip(p) }
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 9.dp, vertical = 6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(5.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(7.dp)
                                    .clip(CircleShape)
                                    .background(if (selected) ObsidianBg else accent)
                            )
                            Text(
                                text = "[${teamTag(p.team)}] #${p.jerseyNumber}$star",
                                color = if (selected) ObsidianBg else TextPrimary,
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.sp
                            )
                            Text(
                                text = "$seenLabel • ${p.clipCount30s}×30s",
                                color = if (selected) ObsidianBg.copy(alpha = 0.8f) else TextMuted,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 10.sp
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ExportStatusBanner(exportState: StudioExportUiState) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(PanelBgElevated, RoundedCornerShape(8.dp))
            .border(1.dp, BenchGreen.copy(alpha = 0.6f), RoundedCornerShape(8.dp))
            .padding(horizontal = 10.dp, vertical = 7.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        val label = exportState.exportLabel ?: exportState.statusMessage.orEmpty()
        Text(
            text = label,
            color = TextPrimary,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis
        )
        if (exportState.exportingClipId != null || exportState.isBatchRunning) {
            LinearProgressIndicator(
                progress = { exportState.progress.coerceIn(0f, 1f) },
                color = TargetGold,
                trackColor = BorderSubtle,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(5.dp)
                    .clip(RoundedCornerShape(3.dp))
            )
        }
    }
}

@Composable
private fun ClipsHeaderRow(
    selectedPlayerNumber: String?,
    clipCount: Int,
    isBusy: Boolean,
    onSplitAll: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        val heading = if (selectedPlayerNumber.isNullOrBlank()) {
            "30-Second Clip Windows"
        } else {
            "30s Windows • #$selectedPlayerNumber ($clipCount)"
        }
        Text(
            text = heading,
            style = MaterialTheme.typography.titleMedium,
            fontSize = 14.sp,
            color = TextPrimary
        )
        if (clipCount > 0) {
            OutlinedButton(
                onClick = onSplitAll,
                enabled = !isBusy,
                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.height(30.dp)
            ) {
                Icon(Icons.Rounded.Layers, contentDescription = null, modifier = Modifier.size(14.dp), tint = BenchGreen)
                Spacer(Modifier.width(4.dp))
                Text("Split All 30s", fontSize = 11.sp, color = BenchGreen, fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

@Composable
private fun ClipWindow30sCard(
    clip: ClipWindow30s,
    isExportingThis: Boolean,
    exportProgress: Float,
    anyExportBusy: Boolean,
    onPlay: () -> Unit,
    onFastSplit: () -> Unit,
    onBurnMarks: () -> Unit
) {
    val accent = teamColor(clip.team)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(PanelBg, RoundedCornerShape(10.dp))
            .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
            .padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Surface(
                    color = accent.copy(alpha = 0.18f),
                    shape = RoundedCornerShape(5.dp),
                    modifier = Modifier.border(1.dp, accent.copy(alpha = 0.6f), RoundedCornerShape(5.dp))
                ) {
                    Text(
                        text = "[${teamTag(clip.team)}] #${clip.playerNumber}",
                        color = accent,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                    )
                }
                Text(
                    text = "${formatClock(clip.startSec)} – ${formatClock(clip.endSec)}",
                    color = TextPrimary,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp
                )
            }

            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                if (!clip.fastClipPath.isNullOrBlank()) {
                    StatusBadge("FAST ✓", BenchGreen)
                }
                if (!clip.markedClipPath.isNullOrBlank()) {
                    StatusBadge("MARKED ✓", TargetGold)
                }
            }
        }

        Text(
            text = String.format(
                Locale.US,
                "Duration %.0fs • Visible %.1fs (%d hits) • Live %.0f%%",
                clip.durationSec,
                clip.visibilitySec,
                clip.sampleHits,
                clip.avgLiveProb * 100f
            ),
            color = TextMuted,
            fontFamily = FontFamily.Monospace,
            fontSize = 11.sp
        )

        if (!clip.nanoSummary.isNullOrBlank()) {
            Text(
                text = "Nano: ${clip.nanoSummary}",
                color = TextPrimary,
                fontSize = 12.sp
            )
        }

        if (isExportingThis) {
            LinearProgressIndicator(
                progress = { exportProgress.coerceIn(0f, 1f) },
                color = TargetGold,
                trackColor = BorderSubtle,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(5.dp)
                    .clip(RoundedCornerShape(3.dp))
            )
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Button(
                onClick = onPlay,
                colors = ButtonDefaults.buttonColors(
                    containerColor = TeamACyan.copy(alpha = 0.2f),
                    contentColor = TeamACyan
                ),
                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
                modifier = Modifier.height(32.dp)
            ) {
                Icon(Icons.Rounded.PlayArrow, contentDescription = null, modifier = Modifier.size(15.dp))
                Spacer(Modifier.width(4.dp))
                Text("Play 30s", fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
            }

            Button(
                onClick = onFastSplit,
                enabled = !anyExportBusy,
                colors = ButtonDefaults.buttonColors(
                    containerColor = BenchGreen,
                    contentColor = ObsidianBg
                ),
                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
                modifier = Modifier.height(32.dp)
            ) {
                Icon(Icons.Rounded.ContentCut, contentDescription = null, modifier = Modifier.size(14.dp))
                Spacer(Modifier.width(4.dp))
                Text("Split 30s MP4", fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }

            Button(
                onClick = onBurnMarks,
                enabled = !anyExportBusy,
                colors = ButtonDefaults.buttonColors(
                    containerColor = TargetGold,
                    contentColor = ObsidianBg
                ),
                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
                modifier = Modifier.height(32.dp)
            ) {
                Icon(Icons.Rounded.AutoAwesome, contentDescription = null, modifier = Modifier.size(14.dp))
                Spacer(Modifier.width(4.dp))
                Text("Burn Marks 30s MP4", fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun StatusBadge(label: String, color: Color) {
    Surface(
        color = color.copy(alpha = 0.16f),
        shape = RoundedCornerShape(4.dp),
        modifier = Modifier.border(1.dp, color.copy(alpha = 0.6f), RoundedCornerShape(4.dp))
    ) {
        Text(
            text = label,
            color = color,
            fontSize = 9.5.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(horizontal = 5.dp, vertical = 2.dp)
        )
    }
}

@Composable
private fun EmptyHintCard(text: String) {
    Surface(
        color = PanelBg,
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
    ) {
        Text(
            text = text,
            color = TextMuted,
            fontSize = 12.sp,
            modifier = Modifier.padding(12.dp)
        )
    }
}
