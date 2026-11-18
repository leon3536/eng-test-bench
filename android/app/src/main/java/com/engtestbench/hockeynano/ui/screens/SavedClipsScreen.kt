package com.engtestbench.hockeynano.ui.screens

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.ContentCut
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.Share
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
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
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.FileProvider
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import com.engtestbench.hockeynano.data.ClipLibraryRepository
import com.engtestbench.hockeynano.ui.theme.BenchGreen
import com.engtestbench.hockeynano.ui.theme.BorderSubtle
import com.engtestbench.hockeynano.ui.theme.ObsidianBg
import com.engtestbench.hockeynano.ui.theme.PanelBg
import com.engtestbench.hockeynano.ui.theme.PanelBgElevated
import com.engtestbench.hockeynano.ui.theme.TargetGold
import com.engtestbench.hockeynano.ui.theme.TeamACyan
import com.engtestbench.hockeynano.ui.theme.TextMuted
import com.engtestbench.hockeynano.ui.theme.TextPrimary
import com.engtestbench.hockeynano.ui.theme.formatBytes
import com.engtestbench.hockeynano.ui.theme.formatClock
import com.engtestbench.hockeynano.ui.theme.teamColor
import com.engtestbench.hockeynano.ui.theme.teamTag
import com.engtestbench.hockeynano.video.ExportedClipInfo
import java.io.File
import java.util.Locale

private enum class ClipFilter { ALL, MARKED, FAST }

@Composable
fun SavedClipsScreen(
    clipLibraryRepository: ClipLibraryRepository,
    refreshNonce: Int,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    var clips by remember { mutableStateOf<List<ExportedClipInfo>>(emptyList()) }
    var selectedClip by remember { mutableStateOf<ExportedClipInfo?>(null) }
    var filter by remember { mutableStateOf(ClipFilter.ALL) }
    var localRefresh by remember { mutableIntStateOf(0) }

    LaunchedEffect(refreshNonce, localRefresh) {
        val loaded = runCatching { clipLibraryRepository.listExportedClips() }.getOrDefault(emptyList())
        clips = loaded
        if (selectedClip == null || loaded.none { it.id == selectedClip?.id }) {
            selectedClip = loaded.firstOrNull()
        }
    }

    val filteredClips = remember(clips, filter) {
        when (filter) {
            ClipFilter.ALL -> clips
            ClipFilter.MARKED -> clips.filter { it.burnedMarks }
            ClipFilter.FAST -> clips.filter { !it.burnedMarks }
        }
    }

    val inlinePlayer = remember { ExoPlayer.Builder(context).build() }
    DisposableEffect(inlinePlayer) {
        onDispose {
            inlinePlayer.release()
        }
    }

    LaunchedEffect(selectedClip?.filePath) {
        val path = selectedClip?.filePath
        if (!path.isNullOrBlank() && File(path).exists()) {
            inlinePlayer.setMediaItem(MediaItem.fromUri(Uri.fromFile(File(path))))
            inlinePlayer.prepare()
            inlinePlayer.playWhenReady = true
        } else {
            inlinePlayer.stop()
            inlinePlayer.clearMediaItems()
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(ObsidianBg)
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Header + filters
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column {
                Text(
                    text = "30s Exported Hockey Clips",
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary
                )
                val totalSize = clips.sumOf { it.sizeBytes }
                Text(
                    text = "${clips.size} MP4 clips • ${formatBytes(totalSize)} total",
                    color = TextMuted,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 11.sp
                )
            }
            IconButton(onClick = { localRefresh++ }) {
                Icon(Icons.Rounded.Refresh, contentDescription = "Reload clips", tint = BenchGreen)
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf(
                ClipFilter.ALL to "All (${clips.size})",
                ClipFilter.MARKED to "Marked (${clips.count { it.burnedMarks }})",
                ClipFilter.FAST to "Fast (${clips.count { !it.burnedMarks }})"
            ).forEach { (f, label) ->
                FilterChip(
                    selected = filter == f,
                    onClick = { filter = f },
                    label = { Text(label, fontSize = 11.sp) },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = BenchGreen.copy(alpha = 0.22f),
                        selectedLabelColor = BenchGreen
                    ),
                    modifier = Modifier.height(32.dp)
                )
            }
        }

        // Inline ExoPlayer preview for selected clip
        selectedClip?.let { active ->
            InlineClipPlayerCard(
                clip = active,
                player = inlinePlayer,
                onShare = { shareExportedClip(context, active) }
            )
        }

        if (filteredClips.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .background(PanelBg, RoundedCornerShape(10.dp))
                    .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
                    .padding(18.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "No exported 30s MP4 clips yet. Run Split 30s MP4 or Burn Marks 30s MP4 in the NANO HOCKEY tab.",
                    color = TextMuted,
                    fontSize = 13.sp
                )
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                contentPadding = PaddingValues(bottom = 12.dp)
            ) {
                items(
                    items = filteredClips,
                    key = { it.id }
                ) { clip ->
                    SavedClipCard(
                        clip = clip,
                        isSelected = clip.id == selectedClip?.id,
                        onPlay = { selectedClip = clip },
                        onShare = { shareExportedClip(context, clip) }
                    )
                }
            }
        }
    }
}

@Composable
private fun InlineClipPlayerCard(
    clip: ExportedClipInfo,
    player: ExoPlayer,
    onShare: () -> Unit
) {
    val fileExists = remember(clip.filePath) { File(clip.filePath).exists() }
    val accent = if (clip.burnedMarks) TargetGold else BenchGreen

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(PanelBg, RoundedCornerShape(10.dp))
            .border(1.dp, accent.copy(alpha = 0.6f), RoundedCornerShape(10.dp))
            .clip(RoundedCornerShape(10.dp))
    ) {
        if (fileExists) {
            AndroidView(
                factory = { ctx ->
                    PlayerView(ctx).apply {
                        this.player = player
                        useController = true
                        setShowBuffering(PlayerView.SHOW_BUFFERING_WHEN_PLAYING)
                    }
                },
                update = { view ->
                    if (view.player !== player) {
                        view.player = player
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(16f / 9f)
                    .background(Color.Black)
            )
        } else {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(120.dp)
                    .background(Color.Black),
                contentAlignment = Alignment.Center
            ) {
                Text("File missing on disk: ${clip.filePath}", color = TextMuted, fontSize = 12.sp)
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "#${clip.playerNumber} • ${if (clip.burnedMarks) "BURNED MARKS 30S" else "FAST LOSSLESS 30S"}",
                    color = accent,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "${formatClock(clip.startSec)}–${formatClock(clip.endSec)} (${String.format(Locale.US, "%.1fs", clip.durationSec)}) • ${formatBytes(clip.sizeBytes)}",
                    color = TextMuted,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 11.sp
                )
                val summary = clip.nanoSummary
                if (!summary.isNullOrBlank()) {
                    Text(
                        text = summary,
                        color = TextPrimary,
                        fontSize = 11.sp,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.padding(top = 3.dp)
                    )
                }
            }
            Spacer(Modifier.width(8.dp))
            OutlinedButton(
                onClick = onShare,
                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                shape = RoundedCornerShape(8.dp)
            ) {
                Icon(Icons.Rounded.Share, contentDescription = null, modifier = Modifier.size(15.dp), tint = TeamACyan)
                Spacer(Modifier.width(4.dp))
                Text("Share", fontSize = 11.sp, color = TeamACyan)
            }
        }
    }
}

@Composable
private fun SavedClipCard(
    clip: ExportedClipInfo,
    isSelected: Boolean,
    onPlay: () -> Unit,
    onShare: () -> Unit
) {
    val teamAccent = teamColor(clip.team)
    val typeAccent = if (clip.burnedMarks) TargetGold else BenchGreen

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(if (isSelected) PanelBgElevated else PanelBg, RoundedCornerShape(10.dp))
            .border(
                width = if (isSelected) 1.5.dp else 1.dp,
                color = if (isSelected) typeAccent else BorderSubtle,
                shape = RoundedCornerShape(10.dp)
            )
            .clickable { onPlay() }
            .padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(5.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Surface(
                    color = teamAccent.copy(alpha = 0.18f),
                    shape = RoundedCornerShape(5.dp),
                    modifier = Modifier.border(1.dp, teamAccent.copy(alpha = 0.6f), RoundedCornerShape(5.dp))
                ) {
                    Text(
                        text = "[${teamTag(clip.team)}] #${clip.playerNumber}",
                        color = teamAccent,
                        fontWeight = FontWeight.Bold,
                        fontSize = 11.sp,
                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                    )
                }

                Surface(
                    color = typeAccent.copy(alpha = 0.16f),
                    shape = RoundedCornerShape(5.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(3.dp)
                    ) {
                        Icon(
                            imageVector = if (clip.burnedMarks) Icons.Rounded.AutoAwesome else Icons.Rounded.ContentCut,
                            contentDescription = null,
                            tint = typeAccent,
                            modifier = Modifier.size(12.dp)
                        )
                        Text(
                            text = if (clip.burnedMarks) "MARKED" else "FAST",
                            color = typeAccent,
                            fontWeight = FontWeight.Bold,
                            fontSize = 10.sp
                        )
                    }
                }
            }

            Text(
                text = "${String.format(Locale.US, "%.1fs", clip.durationSec)} • ${formatBytes(clip.sizeBytes)}",
                color = TextMuted,
                fontFamily = FontFamily.Monospace,
                fontSize = 11.sp
            )
        }

        Text(
            text = "${formatClock(clip.startSec)} – ${formatClock(clip.endSec)} • ${File(clip.filePath).name}",
            color = TextPrimary,
            fontSize = 12.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )

        if (!clip.nanoSummary.isNullOrBlank()) {
            Text(
                text = "Nano: ${clip.nanoSummary}",
                color = TextMuted,
                fontSize = 11.sp,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = onPlay,
                colors = ButtonDefaults.buttonColors(
                    containerColor = BenchGreen,
                    contentColor = ObsidianBg
                ),
                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
                modifier = Modifier.height(30.dp)
            ) {
                Icon(Icons.Rounded.PlayArrow, contentDescription = null, modifier = Modifier.size(15.dp))
                Spacer(Modifier.width(4.dp))
                Text(if (isSelected) "Playing" else "Play", fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }

            OutlinedButton(
                onClick = onShare,
                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
                modifier = Modifier.height(30.dp)
            ) {
                Icon(Icons.Rounded.Share, contentDescription = null, modifier = Modifier.size(14.dp), tint = TeamACyan)
                Spacer(Modifier.width(4.dp))
                Text("Share", fontSize = 11.sp, color = TeamACyan)
            }
        }
    }
}

private fun shareExportedClip(context: Context, clip: ExportedClipInfo) {
    val file = File(clip.filePath)
    val authority = "${context.packageName}.fileprovider"
    val hasFileProvider = runCatching {
        context.packageManager.resolveContentProvider(authority, 0) != null
    }.getOrDefault(false)

    val summaryText = buildString {
        append("HockeyNano #${clip.playerNumber} (${formatClock(clip.startSec)}-${formatClock(clip.endSec)})")
        if (!clip.nanoSummary.isNullOrBlank()) {
            append("\n").append(clip.nanoSummary)
        }
        append("\nFile: ").append(clip.filePath)
    }

    if (hasFileProvider && file.exists()) {
        val uriResult = runCatching { FileProvider.getUriForFile(context, authority, file) }
        uriResult.getOrNull()?.let { contentUri ->
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "video/mp4"
                putExtra(Intent.EXTRA_STREAM, contentUri)
                putExtra(Intent.EXTRA_TEXT, summaryText)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            val chooser = Intent.createChooser(intent, "Share 30s Hockey Clip")
            chooser.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            runCatching { context.startActivity(chooser) }.onSuccess { return }
        }
    }

    val textIntent = Intent(Intent.ACTION_SEND).apply {
        type = "text/plain"
        putExtra(Intent.EXTRA_SUBJECT, "HockeyNano 30s Clip #${clip.playerNumber}")
        putExtra(Intent.EXTRA_TEXT, summaryText)
    }
    val chooser = Intent.createChooser(textIntent, "Share Clip Summary & Path")
    chooser.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    runCatching {
        context.startActivity(chooser)
    }.onFailure {
        Toast.makeText(context, "Saved at ${clip.filePath}", Toast.LENGTH_LONG).show()
    }
}
