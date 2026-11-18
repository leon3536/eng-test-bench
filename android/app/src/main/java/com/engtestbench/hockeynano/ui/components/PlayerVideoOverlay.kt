package com.engtestbench.hockeynano.ui.components

import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.view.TextureView
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Forward10
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Replay10
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.Player
import androidx.media3.common.VideoSize
import androidx.media3.exoplayer.ExoPlayer
import com.engtestbench.hockeynano.core.model.ClipWindow30s
import com.engtestbench.hockeynano.core.model.FrameMarkSample
import com.engtestbench.hockeynano.core.model.PipelineTelemetry
import com.engtestbench.hockeynano.core.model.RectNorm
import com.engtestbench.hockeynano.core.model.TeamId
import com.engtestbench.hockeynano.core.model.TrackedPlayer
import com.engtestbench.hockeynano.ui.theme.BenchGreen
import com.engtestbench.hockeynano.ui.theme.BorderSubtle
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
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import java.util.Locale
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.roundToInt

private val AmberHudColor = Color(0xFFFFB300)

/**
 * Synchronized TextureView + Compose Canvas overlay player.
 * Captures high-resolution frames (~5x/sec) during playback, feeds them to RealtimeHockeyProcessor,
 * interpolates tracked player positions at display refresh rate, and supports tap-to-select/assign.
 */
@Composable
fun PlayerVideoOverlay(
    player: ExoPlayer,
    videoSourceKey: String,
    history: List<FrameMarkSample>,
    currentFrame: FrameMarkSample?,
    selectedPlayerNumber: String?,
    selectedTeamFilter: TeamId?,
    clipWindows: List<ClipWindow30s>,
    clipPlayRange: Pair<Double, Double>?,
    telemetry: PipelineTelemetry = PipelineTelemetry(),
    onIsPlayingChanged: (Boolean) -> Unit = {},
    onClearClipRange: () -> Unit,
    onVideoDimensionsKnown: (durationSec: Double, width: Int, height: Int) -> Unit,
    onVideoFrame: suspend (highResBitmap: Bitmap, timestampSec: Double) -> Unit,
    onSelectPlayer: (jerseyNumber: String?, teamFilter: TeamId?) -> Unit,
    onAssignTrack: (trackId: Int, jerseyNumber: String, team: TeamId?) -> Unit,
    modifier: Modifier = Modifier
) {
    var videoWidth by remember(videoSourceKey) { mutableIntStateOf(1920) }
    var videoHeight by remember(videoSourceKey) { mutableIntStateOf(1080) }
    var durationSec by remember(videoSourceKey) { mutableDoubleStateOf(0.0) }
    var currentPosSec by remember(videoSourceKey) { mutableDoubleStateOf(0.0) }
    var isPlaying by remember(player) { mutableStateOf(player.isPlaying) }
    var renderedFirstFrameForSource by remember(videoSourceKey) { mutableStateOf(false) }
    var textureViewRef by remember { mutableStateOf<TextureView?>(null) }
    var tappedPlayer by remember { mutableStateOf<TrackedPlayer?>(null) }

    val latestHistory by rememberUpdatedState(history)
    val latestCurrentFrame by rememberUpdatedState(currentFrame)
    val latestSelectedNumber by rememberUpdatedState(selectedPlayerNumber)
    val latestSelectedTeam by rememberUpdatedState(selectedTeamFilter)
    val latestClipRange by rememberUpdatedState(clipPlayRange)
    val latestOnVideoFrame by rememberUpdatedState(onVideoFrame)
    val latestOnDimensions by rememberUpdatedState(onVideoDimensionsKnown)
    val latestOnIsPlayingChanged by rememberUpdatedState(onIsPlayingChanged)

    // Listen to ExoPlayer metadata (dimensions & duration).
    DisposableEffect(player, videoSourceKey) {
        var reportedReady = false
        fun maybeReport() {
            val durMs = player.duration
            val durS = if (durMs > 0 && durMs != Long.MIN_VALUE) durMs / 1000.0 else 0.0
            if (durS > 0.0) durationSec = durS
            if (!reportedReady && durS > 0.0 && videoWidth > 0 && videoHeight > 0) {
                reportedReady = true
                latestOnDimensions(durS, videoWidth, videoHeight)
            }
        }

        val listener = object : Player.Listener {
            override fun onVideoSizeChanged(videoSize: VideoSize) {
                if (videoSize.width > 0 && videoSize.height > 0) {
                    videoWidth = videoSize.width
                    videoHeight = videoSize.height
                    maybeReport()
                }
            }

            override fun onPlaybackStateChanged(playbackState: Int) {
                if (playbackState == Player.STATE_READY) {
                    maybeReport()
                }
            }

            override fun onRenderedFirstFrame() {
                renderedFirstFrameForSource = true
            }

            override fun onIsPlayingChanged(playing: Boolean) {
                isPlaying = playing
                latestOnIsPlayingChanged(playing)
            }
        }
        player.addListener(listener)
        val initialPlaying = player.isPlaying
        isPlaying = initialPlaying
        latestOnIsPlayingChanged(initialPlaying)
        val initialSize = player.videoSize
        if (initialSize.width > 0 && initialSize.height > 0) {
            videoWidth = initialSize.width
            videoHeight = initialSize.height
        }
        maybeReport()
        onDispose {
            player.removeListener(listener)
        }
    }

    // Read playhead smoothly at display refresh rate and enforce optional 30s range stop.
    LaunchedEffect(player, videoSourceKey) {
        while (isActive) {
            withFrameNanos { }
            val posMs = player.currentPosition
            val posSec = (posMs.coerceAtLeast(0L)) / 1000.0
            currentPosSec = posSec
            val nowPlaying = player.isPlaying
            if (nowPlaying != isPlaying) {
                isPlaying = nowPlaying
                latestOnIsPlayingChanged(nowPlaying)
            }
            val durMs = player.duration
            if (durMs > 0 && durMs != Long.MIN_VALUE) {
                durationSec = durMs / 1000.0
            }
            val vs = player.videoSize
            if (vs.width > 0 && vs.height > 0 && (vs.width != videoWidth || vs.height != videoHeight)) {
                videoWidth = vs.width
                videoHeight = vs.height
            }
            val range = latestClipRange
            if (range != null && player.isPlaying && posSec >= range.second) {
                player.pause()
                onClearClipRange()
            }
        }
    }

    // High-res TextureView bitmap sampler (~5x/sec during playback + initial frame).
    LaunchedEffect(player, videoSourceKey) {
        var lastCapturedPos = -1.0
        while (isActive) {
            val tv = textureViewRef
            val vw = videoWidth
            val vh = videoHeight
            if (tv != null &&
                tv.isAvailable &&
                vw > 0 &&
                vh > 0 &&
                (renderedFirstFrameForSource || (player.isPlaying && player.playbackState == Player.STATE_READY))
            ) {
                val playingNow = player.isPlaying
                val posSec = player.currentPosition.coerceAtLeast(0L) / 1000.0
                val shouldCaptureInitial = lastCapturedPos < 0.0
                val shouldCaptureLive = playingNow && (abs(posSec - lastCapturedPos) >= 0.04 || lastCapturedPos < 0.0)
                if (shouldCaptureInitial || shouldCaptureLive) {
                    val tStart = SystemClock.elapsedRealtime()
                    val capW = min(vw, 1920).coerceAtLeast(320)
                    var capH = ((capW.toFloat() * vh) / vw.toFloat()).roundToInt().coerceAtLeast(180)
                    if (capH % 2 != 0) capH++
                    val bitmap = runCatching { tv.getBitmap(capW, capH) }.getOrNull()
                    if (bitmap != null) {
                        lastCapturedPos = posSec
                        runCatching { latestOnVideoFrame(bitmap, posSec) }
                    }
                    val elapsed = SystemClock.elapsedRealtime() - tStart
                    delay((180L - elapsed).coerceAtLeast(25L))
                } else {
                    delay(90L)
                }
            } else {
                delay(100L)
            }
        }
    }

    val aspectRatio = remember(videoWidth, videoHeight) {
        (videoWidth.toFloat() / videoHeight.coerceAtLeast(1).toFloat()).coerceIn(1.1f, 2.8f)
    }

    val displayPlayers = remember(latestHistory, latestCurrentFrame, currentPosSec) {
        interpolatePlayersForDisplay(latestHistory, latestCurrentFrame, currentPosSec)
    }
    val textMeasurer = rememberTextMeasurer()

    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(PanelBg)
            .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
            .clip(RoundedCornerShape(10.dp))
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(aspectRatio)
                .background(Color.Black),
            contentAlignment = Alignment.Center
        ) {
            AndroidView(
                factory = { ctx ->
                    TextureView(ctx).apply {
                        isOpaque = true
                        textureViewRef = this
                        player.setVideoTextureView(this)
                    }
                },
                update = { tv ->
                    if (textureViewRef !== tv) {
                        textureViewRef = tv
                        player.setVideoTextureView(tv)
                    }
                },
                onRelease = { tv ->
                    if (textureViewRef === tv) {
                        textureViewRef = null
                    }
                    player.clearVideoTextureView(tv)
                },
                modifier = Modifier.fillMaxSize()
            )

            val currentDisplayPlayers by rememberUpdatedState(displayPlayers)
            Canvas(
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInput(videoSourceKey) {
                        detectTapGestures { tapOffset ->
                            val w = size.width.toFloat().coerceAtLeast(1f)
                            val h = size.height.toFloat().coerceAtLeast(1f)
                            val nx = tapOffset.x / w
                            val ny = tapOffset.y / h
                            val hit = currentDisplayPlayers
                                .filter { p ->
                                    nx in (p.box.x1 - 0.03f)..(p.box.x2 + 0.03f) &&
                                        ny in (p.box.y1 - 0.03f)..(p.box.y2 + 0.03f)
                                }
                                .minByOrNull { p ->
                                    hypot(p.box.centerX - nx, p.box.centerY - ny)
                                }
                            if (hit != null) {
                                if (!hit.jerseyNumber.isNullOrBlank()) {
                                    onSelectPlayer(hit.jerseyNumber, null)
                                }
                                tappedPlayer = hit
                            } else {
                                if (player.isPlaying) {
                                    player.pause()
                                } else {
                                    if (player.playbackState == Player.STATE_ENDED) {
                                        player.seekTo(0L)
                                    }
                                    player.play()
                                }
                            }
                        }
                    }
            ) {
                drawPlayersOverlay(
                    players = displayPlayers,
                    selectedNumber = latestSelectedNumber,
                    selectedTeam = latestSelectedTeam,
                    textMeasurer = textMeasurer
                )
            }

            // Live overlay HUD corner pills
            if (clipPlayRange != null) {
                Surface(
                    color = ObsidianBg.copy(alpha = 0.85f),
                    shape = RoundedCornerShape(6.dp),
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(8.dp)
                        .border(1.dp, TargetGold, RoundedCornerShape(6.dp))
                ) {
                    Text(
                        text = "30S WINDOW ${formatClock(clipPlayRange.first)}–${formatClock(clipPlayRange.second)}",
                        color = TargetGold,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }

            LiveAiHudCornerPill(
                isPlaying = isPlaying,
                telemetry = telemetry,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(8.dp)
            )
        }

        // Timeline bar with 30s window highlights
        val safeDur = durationSec.takeIf { it > 0.0 } ?: 1.0
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 2.dp)
                .height(30.dp)
        ) {
            Canvas(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(5.dp)
                    .align(Alignment.TopCenter)
            ) {
                drawRect(BorderSubtle, Offset.Zero, size)
                for (w in clipWindows) {
                    val x0 = ((w.startSec / safeDur).toFloat().coerceIn(0f, 1f)) * size.width
                    val x1 = ((w.endSec / safeDur).toFloat().coerceIn(0f, 1f)) * size.width
                    drawRect(
                        color = TargetGold.copy(alpha = 0.85f),
                        topLeft = Offset(x0, 0f),
                        size = Size((x1 - x0).coerceAtLeast(3f), size.height)
                    )
                }
            }
            Slider(
                value = (currentPosSec / safeDur).toFloat().coerceIn(0f, 1f),
                onValueChange = { frac ->
                    onClearClipRange()
                    player.seekTo((frac * safeDur * 1000.0).toLong())
                },
                colors = SliderDefaults.colors(
                    thumbColor = BenchGreen,
                    activeTrackColor = BenchGreen,
                    inactiveTrackColor = PanelBgElevated
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .align(Alignment.BottomCenter)
                    .height(22.dp)
            )
        }

        // Transport bar
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 6.dp, vertical = 2.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(
                    onClick = {
                        if (player.isPlaying) {
                            player.pause()
                        } else {
                            if (player.playbackState == Player.STATE_ENDED) {
                                player.seekTo(0L)
                            }
                            player.play()
                        }
                    },
                    modifier = Modifier.size(36.dp)
                ) {
                    Icon(
                        imageVector = if (isPlaying) Icons.Rounded.Pause else Icons.Rounded.PlayArrow,
                        contentDescription = if (isPlaying) "Pause" else "Play",
                        tint = BenchGreen
                    )
                }
                IconButton(
                    onClick = {
                        onClearClipRange()
                        player.seekTo(((currentPosSec - 10.0).coerceAtLeast(0.0) * 1000.0).toLong())
                    },
                    modifier = Modifier.size(34.dp)
                ) {
                    Icon(Icons.Rounded.Replay10, contentDescription = "-10s", tint = TextPrimary)
                }
                IconButton(
                    onClick = {
                        onClearClipRange()
                        val target = (currentPosSec + 10.0).coerceAtMost(safeDur)
                        player.seekTo((target * 1000.0).toLong())
                    },
                    modifier = Modifier.size(34.dp)
                ) {
                    Icon(Icons.Rounded.Forward10, contentDescription = "+10s", tint = TextPrimary)
                }
                Spacer(Modifier.width(4.dp))
                Text(
                    text = "${formatClock(currentPosSec)} / ${formatClock(durationSec)}",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextPrimary
                )
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                if (clipPlayRange != null) {
                    TextButton(
                        onClick = onClearClipRange,
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp)
                    ) {
                        Text("Full Video", color = TargetGold, fontSize = 11.sp)
                    }
                }
                Text(
                    text = "${displayPlayers.size} skaters",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextMuted,
                    modifier = Modifier.padding(end = 6.dp)
                )
            }
        }
    }

    tappedPlayer?.let { playerHit ->
        TapAssignPlayerDialog(
            player = playerHit,
            onDismiss = { tappedPlayer = null },
            onSelectOnly = { num, team ->
                onSelectPlayer(num, team)
                tappedPlayer = null
            },
            onAssignAndSelect = { num, team ->
                onAssignTrack(playerHit.trackId, num, team)
                onSelectPlayer(num, team)
                tappedPlayer = null
            }
        )
    }
}

@Composable
private fun LiveAiHudCornerPill(
    isPlaying: Boolean,
    telemetry: PipelineTelemetry,
    modifier: Modifier = Modifier
) {
    val showLive = isPlaying && !telemetry.analysisBudgetReached
    val showPaused = !isPlaying && !telemetry.analysisBudgetReached && telemetry.framesAnalyzed > 0
    val showDone = telemetry.analysisBudgetReached
    if (!showLive && !showPaused && !showDone) return

    val pulseTransition = rememberInfiniteTransition(label = "liveHudDot")
    val pulseAlpha by pulseTransition.animateFloat(
        initialValue = 0.32f,
        targetValue = 1.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 650),
            repeatMode = RepeatMode.Reverse
        ),
        label = "liveHudDotAlpha"
    )

    val accentColor = when {
        showDone -> BenchGreen
        showLive -> BenchGreen
        else -> AmberHudColor
    }
    val labelText = when {
        showDone -> "AI DONE ✓ • #${telemetry.framesAnalyzed}"
        showLive -> "LIVE AI • #${telemetry.framesAnalyzed} • ${String.format(Locale.US, "%.1f", telemetry.fps)} fps"
        else -> "AI PAUSED • #${telemetry.framesAnalyzed}"
    }

    Row(
        modifier = modifier
            .background(ObsidianBg.copy(alpha = 0.78f), RoundedCornerShape(12.dp))
            .border(1.dp, accentColor.copy(alpha = 0.85f), RoundedCornerShape(12.dp))
            .padding(horizontal = 8.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp)
    ) {
        if (!showDone) {
            Box(
                modifier = Modifier
                    .size(7.dp)
                    .graphicsLayer { alpha = if (showLive) pulseAlpha else 1f }
                    .clip(CircleShape)
                    .background(accentColor)
            )
        }
        Text(
            text = labelText,
            color = accentColor,
            fontFamily = FontFamily.Monospace,
            fontSize = 10.sp,
            fontWeight = FontWeight.Bold,
            maxLines = 1,
            softWrap = false
        )
    }
}

@Composable
private fun TapAssignPlayerDialog(
    player: TrackedPlayer,
    onDismiss: () -> Unit,
    onSelectOnly: (String, TeamId?) -> Unit,
    onAssignAndSelect: (String, TeamId) -> Unit
) {
    var jerseyInput by remember(player.trackId) {
        mutableStateOf(player.jerseyNumber.orEmpty())
    }
    var chosenTeam by remember(player.trackId) {
        mutableStateOf(
            if (player.team == TeamId.UNKNOWN) TeamId.TEAM_A else player.team
        )
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = PanelBg,
        titleContentColor = TextPrimary,
        textContentColor = TextPrimary,
        title = {
            Text(
                text = "Track T${player.trackId} • Team ${teamTag(player.team)}",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedTextField(
                        value = jerseyInput,
                        onValueChange = { jerseyInput = it.filter(Char::isDigit).take(3) },
                        label = { Text("Jersey #") },
                        singleLine = true,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.width(120.dp)
                    )
                    listOf(
                        TeamId.TEAM_A to "A",
                        TeamId.TEAM_B to "B",
                        TeamId.REFEREE to "Ref"
                    ).forEach { (team, label) ->
                        FilterChip(
                            selected = chosenTeam == team,
                            onClick = { chosenTeam = team },
                            label = { Text(label, fontSize = 12.sp) },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = teamColor(team).copy(alpha = 0.25f),
                                selectedLabelColor = teamColor(team)
                            )
                        )
                    }
                }
                if (!player.jerseyNumber.isNullOrBlank()) {
                    Text(
                        text = "Currently tagged #${player.jerseyNumber} (${(player.numberConfidence * 100).roundToInt()}% conf)",
                        color = TextMuted,
                        fontSize = 12.sp
                    )
                }
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    val clean = jerseyInput.trim().trimStart('#')
                    if (clean.isNotEmpty()) {
                        onAssignAndSelect(clean, chosenTeam)
                    }
                },
                enabled = jerseyInput.isNotBlank(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = BenchGreen,
                    contentColor = ObsidianBg
                )
            ) {
                Text("Assign & Track")
            }
        },
        dismissButton = {
            Row {
                if (!player.jerseyNumber.isNullOrBlank()) {
                    TextButton(onClick = { onSelectOnly(player.jerseyNumber!!, null) }) {
                        Text("Select #${player.jerseyNumber}", color = TargetGold)
                    }
                }
                TextButton(onClick = onDismiss) {
                    Text("Close", color = TextMuted)
                }
            }
        }
    )
}

internal fun interpolatePlayersForDisplay(
    history: List<FrameMarkSample>,
    currentFrame: FrameMarkSample?,
    targetSec: Double
): List<TrackedPlayer> {
    if (history.isEmpty()) {
        val cf = currentFrame ?: return emptyList()
        return if (abs(targetSec - cf.timestampSec) <= 1.25) cf.players else emptyList()
    }

    var before: FrameMarkSample? = null
    var after: FrameMarkSample? = null
    for (sample in history) {
        val ts = sample.timestampSec
        if (ts <= targetSec) {
            if (before == null || ts > before.timestampSec) {
                before = sample
            }
        }
        if (ts >= targetSec) {
            if (after == null || ts < after.timestampSec) {
                after = sample
            }
        }
    }

    if (before != null && after != null) {
        val span = after.timestampSec - before.timestampSec
        if (span in 0.001..1.5) {
            val alpha = ((targetSec - before.timestampSec) / span).toFloat().coerceIn(0f, 1f)
            val beforeById = before.players.associateBy { it.trackId }
            val seenIds = HashSet<Int>(after.players.size)
            val out = ArrayList<TrackedPlayer>(after.players.size + 4)
            for (ap in after.players) {
                seenIds.add(ap.trackId)
                val bp = beforeById[ap.trackId]
                if (bp != null) {
                    val lerpedBox = RectNorm(
                        x1 = bp.box.x1 + (ap.box.x1 - bp.box.x1) * alpha,
                        y1 = bp.box.y1 + (ap.box.y1 - bp.box.y1) * alpha,
                        x2 = bp.box.x2 + (ap.box.x2 - bp.box.x2) * alpha,
                        y2 = bp.box.y2 + (ap.box.y2 - bp.box.y2) * alpha
                    )
                    out.add(
                        ap.copy(
                            box = lerpedBox,
                            jerseyNumber = ap.jerseyNumber ?: bp.jerseyNumber,
                            nanoVerified = ap.nanoVerified || bp.nanoVerified
                        )
                    )
                } else if (after.timestampSec - targetSec <= 0.55) {
                    out.add(ap)
                }
            }
            if (targetSec - before.timestampSec <= 0.55) {
                for (bp in before.players) {
                    if (bp.trackId !in seenIds) {
                        out.add(bp)
                    }
                }
            }
            return out
        }
    }

    if (before != null && targetSec - before.timestampSec <= 1.35) {
        return before.players
    }
    if (after != null && after.timestampSec - targetSec <= 1.35) {
        return after.players
    }
    val cf = currentFrame
    if (cf != null && abs(targetSec - cf.timestampSec) <= 1.35) {
        return cf.players
    }
    return emptyList()
}

private fun isPlayerSelected(
    player: TrackedPlayer,
    selectedNumber: String?,
    selectedTeam: TeamId?
): Boolean {
    if (selectedNumber.isNullOrBlank()) return false
    if (player.jerseyNumber != selectedNumber) return false
    return selectedTeam == null || player.team == selectedTeam
}

private fun DrawScope.drawPlayersOverlay(
    players: List<TrackedPlayer>,
    selectedNumber: String?,
    selectedTeam: TeamId?,
    textMeasurer: TextMeasurer
) {
    if (players.isEmpty()) return
    val w = size.width
    val h = size.height
    val placedPills = ArrayList<RectF>(players.size)

    val selectedList = ArrayList<TrackedPlayer>()
    val othersList = ArrayList<TrackedPlayer>()
    for (p in players) {
        if (isPlayerSelected(p, selectedNumber, selectedTeam)) {
            selectedList.add(p)
        } else {
            othersList.add(p)
        }
    }

    // Reserve pill rects for selected players first so their badges never collide,
    // then draw unselected first and selected last (on top).
    val orderedForDrawing = othersList + selectedList
    val pillPositions = HashMap<Int, Pair<RectF, androidx.compose.ui.text.TextLayoutResult>>(players.size)

    // Measure & stagger selected first, then others sorted left-to-right
    val orderedForStagger = selectedList + othersList.sortedBy { it.box.centerX }
    for (p in orderedForStagger) {
        val sel = isPlayerSelected(p, selectedNumber, selectedTeam)
        val tag = teamTag(p.team)
        val idPart = if (!p.jerseyNumber.isNullOrBlank()) "#${p.jerseyNumber}" else "T${p.trackId}"
        val star = if (p.nanoVerified) " ★" else ""
        val label = "[$tag] $idPart$star"
        val textLayout = textMeasurer.measure(
            text = label,
            style = TextStyle(
                color = if (sel) ObsidianBg else TextPrimary,
                fontSize = if (sel) 11.sp else 9.5.sp,
                fontWeight = FontWeight.Bold
            )
        )
        val padH = 8f
        val padV = 4f
        val pillW = textLayout.size.width + padH * 2f
        val pillH = textLayout.size.height + padV * 2f
        val boxLeft = p.box.x1 * w
        val boxRight = p.box.x2 * w
        val boxTop = p.box.y1 * h
        val centerX = (boxLeft + boxRight) * 0.5f

        val pillLeft = (centerX - pillW * 0.5f).coerceIn(2f, (w - pillW - 2f).coerceAtLeast(2f))
        var pillTop = (boxTop - pillH - 4f).coerceAtLeast(2f)

        var rect = RectF(pillLeft, pillTop, pillLeft + pillW, pillTop + pillH)
        var attempts = 0
        while (attempts < 5 && placedPills.any { RectF.intersects(it, rect) }) {
            val upCandidate = pillTop - (pillH + 3f)
            pillTop = if (upCandidate >= 2f) {
                upCandidate
            } else {
                (boxTop + 4f + attempts * (pillH + 3f)).coerceAtMost((h - pillH - 2f).coerceAtLeast(2f))
            }
            rect = RectF(pillLeft, pillTop, pillLeft + pillW, pillTop + pillH)
            attempts++
        }
        placedPills.add(rect)
        pillPositions[p.trackId] = rect to textLayout
    }

    for (p in orderedForDrawing) {
        val sel = isPlayerSelected(p, selectedNumber, selectedTeam)
        val color = if (sel) TargetGold else teamColor(p.team)

        // 1. Motion trail
        if (p.trail.size >= 2) {
            val trailPath = Path()
            p.trail.forEachIndexed { idx, pt ->
                val px = pt.first * w
                val py = pt.second * h
                if (idx == 0) trailPath.moveTo(px, py) else trailPath.lineTo(px, py)
            }
            drawPath(
                path = trailPath,
                color = color.copy(alpha = if (sel) 0.85f else 0.40f),
                style = Stroke(width = if (sel) 4.2f else 2f, cap = StrokeCap.Round)
            )
        }

        // 2. Bounding box coordinates
        val x1 = p.box.x1 * w
        val y1 = p.box.y1 * h
        val x2 = p.box.x2 * w
        val y2 = p.box.y2 * h
        val bw = (x2 - x1).coerceAtLeast(10f)
        val bh = (y2 - y1).coerceAtLeast(16f)

        // 3. Team-colored skate ring at footY (double halo for selected player)
        val footCenterX = p.box.centerX * w
        val footCenterY = p.box.footY * h
        val ringW = (bw * 1.18f).coerceAtLeast(24f)
        val ringH = (bw * 0.36f).coerceAtLeast(9f)
        drawOval(
            color = color.copy(alpha = if (sel) 0.25f else 0.14f),
            topLeft = Offset(footCenterX - ringW * 0.5f, footCenterY - ringH * 0.5f),
            size = Size(ringW, ringH)
        )
        drawOval(
            color = color,
            topLeft = Offset(footCenterX - ringW * 0.5f, footCenterY - ringH * 0.5f),
            size = Size(ringW, ringH),
            style = Stroke(width = if (sel) 3.4f else 2.0f)
        )
        if (sel) {
            val outerW = ringW * 1.42f
            val outerH = ringH * 1.45f
            drawOval(
                color = TargetGold.copy(alpha = 0.65f),
                topLeft = Offset(footCenterX - outerW * 0.5f, footCenterY - outerH * 0.5f),
                size = Size(outerW, outerH),
                style = Stroke(width = 2.4f)
            )
        }

        // 4. Player box (or target corners for selected player)
        if (sel) {
            drawRoundRect(
                color = TargetGold.copy(alpha = 0.35f),
                topLeft = Offset(x1, y1),
                size = Size(bw, bh),
                cornerRadius = CornerRadius(4f, 4f),
                style = Stroke(width = 1.6f)
            )
            val cornerLenX = (bw * 0.28f).coerceIn(8f, 22f)
            val cornerLenY = (bh * 0.22f).coerceIn(8f, 22f)
            val strokeW = 4.2f
            // Top-left
            drawLine(TargetGold, Offset(x1, y1), Offset(x1 + cornerLenX, y1), strokeW)
            drawLine(TargetGold, Offset(x1, y1), Offset(x1, y1 + cornerLenY), strokeW)
            // Top-right
            drawLine(TargetGold, Offset(x2, y1), Offset(x2 - cornerLenX, y1), strokeW)
            drawLine(TargetGold, Offset(x2, y1), Offset(x2, y1 + cornerLenY), strokeW)
            // Bottom-left
            drawLine(TargetGold, Offset(x1, y2), Offset(x1 + cornerLenX, y2), strokeW)
            drawLine(TargetGold, Offset(x1, y2), Offset(x1, y2 - cornerLenY), strokeW)
            // Bottom-right
            drawLine(TargetGold, Offset(x2, y2), Offset(x2 - cornerLenX, y2), strokeW)
            drawLine(TargetGold, Offset(x2, y2), Offset(x2, y2 - cornerLenY), strokeW)
        } else {
            drawRoundRect(
                color = color.copy(alpha = 0.85f),
                topLeft = Offset(x1, y1),
                size = Size(bw, bh),
                cornerRadius = CornerRadius(5f, 5f),
                style = Stroke(width = 2.0f)
            )
        }

        // 5. Anti-overlap pill badge
        val (pillRect, textLayout) = pillPositions[p.trackId] ?: continue
        val pillBg = if (sel) TargetGold else ObsidianBg.copy(alpha = 0.88f)
        val pillBorder = if (sel) TargetGold else color
        drawRoundRect(
            color = pillBg,
            topLeft = Offset(pillRect.left, pillRect.top),
            size = Size(pillRect.width(), pillRect.height()),
            cornerRadius = CornerRadius(6f, 6f)
        )
        drawRoundRect(
            color = pillBorder,
            topLeft = Offset(pillRect.left, pillRect.top),
            size = Size(pillRect.width(), pillRect.height()),
            cornerRadius = CornerRadius(6f, 6f),
            style = Stroke(width = if (sel) 2.2f else 1.4f)
        )
        drawText(
            textLayoutResult = textLayout,
            topLeft = Offset(pillRect.left + 8f, pillRect.top + 4f)
        )
    }
}
