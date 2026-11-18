package com.engtestbench.hockeynano

import android.Manifest
import android.app.Application
import android.content.ContentUris
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Assessment
import androidx.compose.material.icons.rounded.MovieFilter
import androidx.compose.material.icons.rounded.SportsHockey
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.PlaybackParameters
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.datasource.DefaultDataSource
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.exoplayer.DefaultLoadControl
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory
import androidx.media3.exoplayer.source.MergingMediaSource
import androidx.media3.exoplayer.source.ProgressiveMediaSource
import com.engtestbench.hockeynano.core.model.AnalysisDurationPreset
import com.engtestbench.hockeynano.core.model.ClipWindow30s
import com.engtestbench.hockeynano.core.model.PipelineState
import com.engtestbench.hockeynano.core.model.TeamId
import com.engtestbench.hockeynano.data.BenchSeedRepository
import com.engtestbench.hockeynano.data.ClipLibraryRepository
import com.engtestbench.hockeynano.ui.screens.BenchReportScreen
import com.engtestbench.hockeynano.ui.screens.NanoHockeyStudioScreen
import com.engtestbench.hockeynano.ui.screens.SavedClipsScreen
import com.engtestbench.hockeynano.ui.screens.StudioExportUiState
import com.engtestbench.hockeynano.ui.theme.BenchGreen
import com.engtestbench.hockeynano.ui.theme.BorderSubtle
import com.engtestbench.hockeynano.ui.theme.HockeyNanoTheme
import com.engtestbench.hockeynano.ui.theme.ObsidianBg
import com.engtestbench.hockeynano.ui.theme.PanelBg
import com.engtestbench.hockeynano.ui.theme.TextMuted
import com.engtestbench.hockeynano.ui.theme.TextPrimary
import com.engtestbench.hockeynano.video.FastMp4ClipSplitter
import com.engtestbench.hockeynano.video.MarkedMp4ClipExporter
import com.engtestbench.hockeynano.vision.RealtimeHockeyProcessor
import com.engtestbench.hockeynano.youtube.ResolvedYouTubeVideo
import com.engtestbench.hockeynano.youtube.YouTubeHockeyResolver
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

private const val TAG = "HockeyNanoBench"

enum class AppTab(val label: String, val icon: ImageVector) {
    STUDIO("NANO HOCKEY", Icons.Rounded.SportsHockey),
    CLIPS("30S CLIPS", Icons.Rounded.MovieFilter),
    BENCH("TEST BENCH", Icons.Rounded.Assessment)
}

data class AutomationCommand(
    val seq: Int,
    val videoPath: String?,
    val youtubeUrl: String?,
    val analysisDuration: String?,
    val startOffsetSec: Double?,
    val selectPlayer: String?,
    val autoSplit: Boolean,
    val burnMarks: Boolean,
    val playbackSpeed: Float
)

@OptIn(UnstableApi::class)
class HockeyStudioViewModel(application: Application) : AndroidViewModel(application) {
    private val appContext: Context = application.applicationContext

    val processor = RealtimeHockeyProcessor(appContext)
    val clipLibraryRepository = ClipLibraryRepository(appContext)
    val benchSeedRepository = BenchSeedRepository(appContext)

    private val httpDataSourceFactory: DefaultHttpDataSource.Factory =
        DefaultHttpDataSource.Factory()
            .setUserAgent(YouTubeHockeyResolver.USER_AGENT)
            .setAllowCrossProtocolRedirects(true)

    private val dataSourceFactory: DefaultDataSource.Factory =
        DefaultDataSource.Factory(appContext, httpDataSourceFactory)

    val player: ExoPlayer = ExoPlayer.Builder(appContext)
        .setMediaSourceFactory(DefaultMediaSourceFactory(dataSourceFactory))
        .setLoadControl(
            DefaultLoadControl.Builder()
                .setBufferDurationsMs(15_000, 45_000, 1_200, 2_000)
                .build()
        )
        .build()

    val pipelineState: StateFlow<PipelineState> = processor.state

    var discoveredVideos by mutableStateOf<List<Pair<String, String>>>(emptyList())
        private set
    var selectedVideoTitle by mutableStateOf("")
        private set
    var selectedVideoSource by mutableStateOf("")
        private set
    var playbackSpeed by mutableStateOf(1.0f)
        private set
    var clipPlayRange by mutableStateOf<Pair<Double, Double>?>(null)
        private set
    var analysisLimitSec by mutableStateOf<Double?>(AnalysisDurationPreset.MIN_5.seconds)
        private set
    var isResolvingYouTube by mutableStateOf(false)
        private set
    var youtubeError by mutableStateOf<String?>(null)
        private set
    var resolvedYoutube by mutableStateOf<ResolvedYouTubeVideo?>(null)
        private set

    private val _exportState = MutableStateFlow(StudioExportUiState())
    val exportState: StateFlow<StudioExportUiState> = _exportState.asStateFlow()
    var clipsRefreshNonce by mutableIntStateOf(0)
        private set

    private var videoLoadJob: Job? = null
    private var automationJob: Job? = null
    private var initialVideoLoaded = false
    private var activeYouTubeFallbackUrl: String? = null
    private var activeYouTubeStartSec: Double = 0.0
    private var didMuxedFallbackForLoad: Boolean = false
    private var lastBudgetReached: Boolean = false

    private val playerListener = object : Player.Listener {
        override fun onPlayerError(error: PlaybackException) {
            val yt = resolvedYoutube
            val fallbackUrl = activeYouTubeFallbackUrl
            if (yt != null && !didMuxedFallbackForLoad && !fallbackUrl.isNullOrBlank()) {
                didMuxedFallbackForLoad = true
                val curSec = player.currentPosition.coerceAtLeast(0L) / 1000.0
                val resumeSec = if (curSec > activeYouTubeStartSec + 1.0) curSec else activeYouTubeStartSec
                val resumeMs = (resumeSec * 1000.0).toLong()
                val resumeAutoPlay = player.playWhenReady
                Log.w(
                    TAG,
                    "YOUTUBE_FALLBACK_MUXED id=${yt.videoId} err=${error.errorCodeName} " +
                        "resumeSec=${String.format(Locale.US, "%.1f", resumeSec)}"
                )
                player.stop()
                player.clearMediaItems()
                player.setMediaItem(MediaItem.fromUri(fallbackUrl), resumeMs)
                player.playbackParameters = PlaybackParameters(playbackSpeed)
                player.prepare()
                player.playWhenReady = resumeAutoPlay
            } else if (yt != null) {
                val msg = error.cause?.message ?: error.message ?: error.errorCodeName
                youtubeError = "Stream playback error: $msg"
                Log.e(TAG, "YOUTUBE_PLAYBACK_ERROR id=${yt.videoId} err=$msg", error)
            } else {
                Log.e(TAG, "PLAYER_ERROR source='$selectedVideoSource' err=${error.errorCodeName}", error)
            }
        }
    }

    init {
        player.addListener(playerListener)
        observeAnalysisBudget()
        refreshDiscoveredVideos(autoSelectFirst = true)
    }

    private fun observeAnalysisBudget() {
        viewModelScope.launch {
            processor.state.collect { st ->
                val reached = st.telemetry.analysisBudgetReached
                if (!lastBudgetReached && reached && clipPlayRange == null) {
                    player.pause()
                    _exportState.update { cur ->
                        cur.copy(statusMessage = "Analysis duration complete — select #player to split 30s clips")
                    }
                    Log.i(
                        TAG,
                        "ANALYSIS_AUTO_PAUSED analyzedSpan=${String.format(Locale.US, "%.1f", st.telemetry.analyzedSpanSec)}s " +
                            "limitSec=${st.telemetry.analysisLimitSec} players=${st.discoveredPlayers.size} history=${st.history.size}"
                    )
                }
                lastBudgetReached = reached
            }
        }
    }

    fun setAnalysisDurationLimit(limitSec: Double?) {
        val normalized = limitSec?.takeIf { it > 0.0 }
        analysisLimitSec = normalized
        processor.setAnalysisDurationLimit(normalized)
        if (normalized == null || processor.state.value.telemetry.analyzedSpanSec < normalized) {
            _exportState.update { cur ->
                if (cur.statusMessage?.startsWith("Analysis duration complete") == true) {
                    cur.copy(statusMessage = null)
                } else {
                    cur
                }
            }
        }
        Log.i(TAG, "ANALYSIS_LIMIT_SET limitSec=$normalized")
    }

    fun analyzeYouTubeUrl(rawUrl: String) {
        val clean = rawUrl.trim()
        if (clean.isEmpty()) {
            youtubeError = "Enter a YouTube link to analyze"
            return
        }
        if (!YouTubeHockeyResolver.isYouTubeUrl(clean)) {
            youtubeError = "Unsupported URL — paste a youtube.com or youtu.be link"
            return
        }
        loadVideo(title = "YouTube Hockey", source = clean, autoPlay = true)
    }

    fun refreshDiscoveredVideos(autoSelectFirst: Boolean = false) {
        viewModelScope.launch {
            val bundledUri = "android.resource://${appContext.packageName}/raw/hockey_sample_30s"
            val found = withContext(Dispatchers.IO) {
                runCatching { clipLibraryRepository.discoverLocalHockeyVideos() }
                    .getOrDefault(emptyList())
            }.let { list ->
                if (list.isEmpty()) listOf("Bundled 30s Demo" to bundledUri) else list
            }
            discoveredVideos = found
            if (autoSelectFirst && !initialVideoLoaded && found.isNotEmpty()) {
                val first = found.firstOrNull { !YouTubeHockeyResolver.isYouTubeUrl(it.second) } ?: found.first()
                loadVideo(first.first, first.second, autoPlay = false)
            }
        }
    }

    fun loadVideo(
        title: String,
        source: String,
        autoPlay: Boolean,
        startOffsetSecOverride: Double? = null
    ) {
        val cleanSource = source.trim()
        if (cleanSource.isEmpty()) return
        initialVideoLoaded = true
        videoLoadJob?.cancel()
        videoLoadJob = viewModelScope.launch {
            selectedVideoTitle = title.ifBlank {
                if (YouTubeHockeyResolver.isYouTubeUrl(cleanSource)) "Resolving YouTube…" else "Hockey Video"
            }
            selectedVideoSource = cleanSource
            clipPlayRange = null
            youtubeError = null
            resolvedYoutube = null
            activeYouTubeFallbackUrl = null
            activeYouTubeStartSec = 0.0
            didMuxedFallbackForLoad = false
            lastBudgetReached = false

            player.stop()
            player.clearMediaItems()

            if (YouTubeHockeyResolver.isYouTubeUrl(cleanSource)) {
                isResolvingYouTube = true
                val yt = try {
                    withContext(Dispatchers.IO) {
                        YouTubeHockeyResolver.resolve(cleanSource, preferMaxHeight = 1080)
                    }
                } catch (ce: CancellationException) {
                    isResolvingYouTube = false
                    throw ce
                } catch (err: Throwable) {
                    isResolvingYouTube = false
                    val msg = err.message?.takeIf { it.isNotBlank() } ?: "Unable to resolve YouTube stream"
                    youtubeError = msg
                    Log.e(TAG, "YOUTUBE_RESOLVE_ERROR url='$cleanSource' err=$msg", err)
                    return@launch
                }
                isResolvingYouTube = false
                resolvedYoutube = yt
                val displayTitle = yt.title.ifBlank { title }.ifBlank { "YouTube (${yt.videoId})" }
                selectedVideoTitle = displayTitle

                withContext(Dispatchers.IO) {
                    runCatching { clipLibraryRepository.saveRecentYouTubeVideo(displayTitle, cleanSource) }
                }
                refreshDiscoveredVideos(autoSelectFirst = false)

                val rawStart = startOffsetSecOverride ?: yt.startOffsetSec ?: 0.0
                val startSec = if (yt.durationSec > 1.0) {
                    rawStart.coerceIn(0.0, (yt.durationSec - 1.0).coerceAtLeast(0.0))
                } else {
                    rawStart.coerceAtLeast(0.0)
                }
                val startMs = (startSec * 1000.0).toLong()
                val sessionDuration = if (yt.durationSec > 0.5) yt.durationSec else 120.0
                val panoramic = processor.state.value.telemetry.panoramicTilingEnabled

                activeYouTubeFallbackUrl = yt.muxedFallbackUrl?.takeIf { it.isNotBlank() && it != yt.videoStreamUrl }
                activeYouTubeStartSec = startSec

                processor.startSession(
                    videoTitle = displayTitle,
                    videoSource = cleanSource,
                    durationSec = sessionDuration,
                    panoramicTiling = panoramic,
                    analysisLimitSec = analysisLimitSec,
                    analysisStartSec = startSec
                )

                val audioUrl = yt.audioStreamUrl?.takeIf { it.isNotBlank() }
                if (audioUrl != null && !yt.hasEmbeddedAudio) {
                    val progFactory = ProgressiveMediaSource.Factory(dataSourceFactory)
                    val videoSrc = progFactory.createMediaSource(MediaItem.fromUri(yt.videoStreamUrl))
                    val audioSrc = progFactory.createMediaSource(MediaItem.fromUri(audioUrl))
                    val merged = MergingMediaSource(videoSrc, audioSrc)
                    player.setMediaSource(merged, startMs)
                } else {
                    player.setMediaItem(MediaItem.fromUri(yt.videoStreamUrl), startMs)
                }
                player.playbackParameters = PlaybackParameters(playbackSpeed)
                player.prepare()
                player.playWhenReady = autoPlay

                Log.i(
                    TAG,
                    "YOUTUBE_STREAM_LOADED id=${yt.videoId} title='$displayTitle' size=${yt.width}x${yt.height} " +
                        "dur=${String.format(Locale.US, "%.1f", yt.durationSec)}s startSec=${String.format(Locale.US, "%.1f", startSec)}s " +
                        "limitSec=$analysisLimitSec mergedAudio=${audioUrl != null && !yt.hasEmbeddedAudio} " +
                        "muxedFallback=${activeYouTubeFallbackUrl != null} speed=$playbackSpeed autoPlay=$autoPlay"
                )
                return@launch
            }

            isResolvingYouTube = false
            val playableUri = withContext(Dispatchers.IO) { resolvePlayableUri(appContext, cleanSource) }
            val isRemoteHttp = playableUri.scheme?.lowercase(Locale.US) in setOf("http", "https")
            val probedDuration = if (isRemoteHttp) {
                0.0
            } else {
                withContext(Dispatchers.IO) { probeDurationSeconds(appContext, playableUri) }
            }
            val startDuration = if (probedDuration > 0.5) probedDuration else 120.0
            val startSec = if (startOffsetSecOverride != null) {
                if (probedDuration > 1.0) {
                    startOffsetSecOverride.coerceIn(0.0, (probedDuration - 1.0).coerceAtLeast(0.0))
                } else {
                    startOffsetSecOverride.coerceAtLeast(0.0)
                }
            } else {
                0.0
            }
            val startMs = (startSec * 1000.0).toLong()
            val panoramic = processor.state.value.telemetry.panoramicTilingEnabled

            processor.startSession(
                videoTitle = selectedVideoTitle,
                videoSource = cleanSource,
                durationSec = startDuration,
                panoramicTiling = panoramic,
                analysisLimitSec = analysisLimitSec,
                analysisStartSec = startSec
            )

            player.setMediaItem(MediaItem.fromUri(playableUri), startMs)
            player.playbackParameters = PlaybackParameters(playbackSpeed)
            player.prepare()
            player.playWhenReady = autoPlay

            Log.i(
                TAG,
                "VIDEO_LOADED title='$selectedVideoTitle' source='$cleanSource' uri='$playableUri' probedDur=${
                    String.format(Locale.US, "%.1f", probedDuration)
                }s startSec=${String.format(Locale.US, "%.1f", startSec)}s limitSec=$analysisLimitSec speed=$playbackSpeed autoPlay=$autoPlay"
            )
        }
    }

    fun onVideoDimensionsKnown(durationSec: Double, width: Int, height: Int) {
        Log.i(
            TAG,
            "VIDEO_READY title='$selectedVideoTitle' dur=${String.format(Locale.US, "%.1f", durationSec)}s size=${width}x${height}"
        )
        val cur = processor.state.value
        if (durationSec > 0.5 && kotlin.math.abs(cur.telemetry.videoDurationSec - durationSec) > 1.0 && cur.history.isEmpty()) {
            processor.startSession(
                videoTitle = selectedVideoTitle,
                videoSource = selectedVideoSource,
                durationSec = durationSec,
                panoramicTiling = cur.telemetry.panoramicTilingEnabled,
                analysisLimitSec = analysisLimitSec,
                analysisStartSec = cur.telemetry.analysisStartSec
            )
        }
    }

    fun setSpeed(speed: Float) {
        val clamped = speed.coerceIn(0.25f, 4.0f)
        playbackSpeed = clamped
        player.playbackParameters = PlaybackParameters(clamped)
    }

    fun setPanoramic(enabled: Boolean) {
        processor.setPanoramicTiling(enabled)
    }

    suspend fun onVideoFrame(bitmap: Bitmap, timestampSec: Double) {
        processor.onVideoFrame(bitmap, timestampSec)
    }

    fun selectPlayer(jerseyNumber: String?, teamFilter: TeamId?) {
        processor.selectPlayerNumber(jerseyNumber, teamFilter)
    }

    fun assignTrack(trackId: Int, jerseyNumber: String, team: TeamId?) {
        processor.assignTrackJerseyNumber(trackId, jerseyNumber, team)
    }

    fun playClipWindow(clip: ClipWindow30s) {
        clipPlayRange = clip.startSec to clip.endSec
        player.seekTo((clip.startSec * 1000.0).toLong())
        player.playWhenReady = true
        Log.i(TAG, "PLAY_30S_WINDOW id=${clip.id} #${clip.playerNumber} ${clip.startSec}..${clip.endSec}")
    }

    fun clearClipRange() {
        clipPlayRange = null
    }

    fun launchFastSplit(clip: ClipWindow30s) {
        val cur = _exportState.value
        if (cur.exportingClipId != null || cur.isBatchRunning) return
        viewModelScope.launch {
            runFastSplitSuspend(clip)
        }
    }

    fun launchBurnMarks(clip: ClipWindow30s) {
        val cur = _exportState.value
        if (cur.exportingClipId != null || cur.isBatchRunning) return
        viewModelScope.launch {
            runBurnMarksSuspend(clip)
        }
    }

    fun launchSplitAll() {
        val clips = processor.state.value.selectedPlayerClips
        val cur = _exportState.value
        if (clips.isEmpty() || cur.exportingClipId != null || cur.isBatchRunning) return
        viewModelScope.launch {
            _exportState.value = StudioExportUiState(
                isBatchRunning = true,
                exportLabel = "Batch splitting ${clips.size} × 30s MP4 clips…",
                progress = 0f
            )
            var okCount = 0
            clips.forEachIndexed { idx, clip ->
                _exportState.update {
                    it.copy(
                        exportingClipId = clip.id,
                        exportLabel = "Batch fast split ${idx + 1}/${clips.size}: #${clip.playerNumber}",
                        progress = (idx.toFloat() / clips.size.toFloat())
                    )
                }
                if (runFastSplitInternal(clip)) okCount++
            }
            _exportState.value = StudioExportUiState(
                statusMessage = "Batch complete: $okCount/${clips.size} fast 30s MP4 clips saved"
            )
            clipsRefreshNonce++
        }
    }

    private suspend fun runFastSplitSuspend(clip: ClipWindow30s): Boolean {
        _exportState.value = StudioExportUiState(
            exportingClipId = clip.id,
            exportLabel = "Splitting lossless 30s MP4 for #${clip.playerNumber}…",
            progress = 0.5f
        )
        val ok = runFastSplitInternal(clip)
        if (!ok && _exportState.value.statusMessage == null) {
            _exportState.value = StudioExportUiState(statusMessage = "Fast split failed for #${clip.playerNumber}")
        }
        clipsRefreshNonce++
        return ok
    }

    private suspend fun runFastSplitInternal(clip: ClipWindow30s): Boolean {
        val activeYt = resolvedYoutube?.takeIf { it.originalUrl == selectedVideoSource }
        val primarySource = activeYt?.videoStreamUrl
            ?: withContext(Dispatchers.IO) { resolveExtractorSource(appContext, selectedVideoSource) }
        val primaryAudio = activeYt?.audioStreamUrl
        val outFile = clipLibraryRepository.outputFileFor(clip.playerNumber, clip.startSec, clip.endSec, burnedMarks = false)
        var result = withContext(Dispatchers.IO) {
            FastMp4ClipSplitter.split30sClip(
                context = appContext,
                sourcePathOrUri = primarySource,
                startSec = clip.startSec,
                durationSec = clip.durationSec,
                playerNumber = clip.playerNumber,
                team = clip.team,
                outputFile = outFile,
                nanoSummary = clip.nanoSummary,
                audioSourcePathOrUri = primaryAudio
            )
        }
        val muxedFallback = activeYt?.muxedFallbackUrl?.takeIf { it.isNotBlank() && it != primarySource }
        if (result.isFailure && muxedFallback != null) {
            Log.w(
                TAG,
                "YOUTUBE_EXPORT_MUXED_FALLBACK kind=fast clip=${clip.id} err=${result.exceptionOrNull()?.message}"
            )
            result = withContext(Dispatchers.IO) {
                FastMp4ClipSplitter.split30sClip(
                    context = appContext,
                    sourcePathOrUri = muxedFallback,
                    startSec = clip.startSec,
                    durationSec = clip.durationSec,
                    playerNumber = clip.playerNumber,
                    team = clip.team,
                    outputFile = outFile,
                    nanoSummary = clip.nanoSummary,
                    audioSourcePathOrUri = null
                )
            }
        }
        return result.fold(
            onSuccess = { info ->
                withContext(Dispatchers.IO) { clipLibraryRepository.recordExport(info) }
                processor.markClipExported(clip.id, fastPath = info.filePath)
                _exportState.value = StudioExportUiState(
                    statusMessage = "Saved fast 30s MP4: ${File(info.filePath).name}"
                )
                Log.i(TAG, "AUTO_FAST_SPLIT_DONE clip=${clip.id} path=${info.filePath} size=${info.sizeBytes}")
                true
            },
            onFailure = { err ->
                Log.e(TAG, "Fast split error for ${clip.id}: ${err.message}", err)
                _exportState.value = StudioExportUiState(
                    statusMessage = "Fast split error: ${err.message ?: "failed"}"
                )
                false
            }
        )
    }

    private suspend fun runBurnMarksSuspend(clip: ClipWindow30s): Boolean {
        _exportState.value = StudioExportUiState(
            exportingClipId = clip.id,
            exportLabel = "Burning AI marks into 30s MP4 for #${clip.playerNumber} (0%)…",
            progress = 0f
        )
        val activeYt = resolvedYoutube?.takeIf { it.originalUrl == selectedVideoSource }
        val primarySource = activeYt?.videoStreamUrl
            ?: withContext(Dispatchers.IO) { resolveExtractorSource(appContext, selectedVideoSource) }
        val primaryAudio = activeYt?.audioStreamUrl
        val outFile = clipLibraryRepository.outputFileFor(clip.playerNumber, clip.startSec, clip.endSec, burnedMarks = true)
        val snapshotState = processor.state.value
        val windowSamples = snapshotState.history.filter {
            it.timestampSec in (clip.startSec - 1.0)..(clip.endSec + 1.0)
        }
        var result = withContext(Dispatchers.IO) {
            MarkedMp4ClipExporter.exportMarked30sClip(
                context = appContext,
                sourcePathOrUri = primarySource,
                startSec = clip.startSec,
                durationSec = clip.durationSec,
                samples = windowSamples,
                teams = snapshotState.teams,
                highlightPlayerNumber = clip.playerNumber,
                teamId = clip.team,
                outputFile = outFile,
                nanoSummary = clip.nanoSummary,
                audioSourcePathOrUri = primaryAudio,
                onProgress = { p ->
                    val pct = (p * 100f).toInt().coerceIn(0, 100)
                    _exportState.update { cur ->
                        cur.copy(
                            progress = p,
                            exportLabel = "Burning AI marks #${clip.playerNumber} ($pct%)…"
                        )
                    }
                }
            )
        }
        val muxedFallback = activeYt?.muxedFallbackUrl?.takeIf { it.isNotBlank() && it != primarySource }
        if (result.isFailure && muxedFallback != null) {
            Log.w(
                TAG,
                "YOUTUBE_EXPORT_MUXED_FALLBACK kind=marked clip=${clip.id} err=${result.exceptionOrNull()?.message}"
            )
            _exportState.update { cur ->
                cur.copy(
                    progress = 0f,
                    exportLabel = "Retrying AI marks on muxed stream for #${clip.playerNumber} (0%)…"
                )
            }
            result = withContext(Dispatchers.IO) {
                MarkedMp4ClipExporter.exportMarked30sClip(
                    context = appContext,
                    sourcePathOrUri = muxedFallback,
                    startSec = clip.startSec,
                    durationSec = clip.durationSec,
                    samples = windowSamples,
                    teams = snapshotState.teams,
                    highlightPlayerNumber = clip.playerNumber,
                    teamId = clip.team,
                    outputFile = outFile,
                    nanoSummary = clip.nanoSummary,
                    audioSourcePathOrUri = null,
                    onProgress = { p ->
                        val pct = (p * 100f).toInt().coerceIn(0, 100)
                        _exportState.update { cur ->
                            cur.copy(
                                progress = p,
                                exportLabel = "Retrying AI marks #${clip.playerNumber} ($pct%)…"
                            )
                        }
                    }
                )
            }
        }
        clipsRefreshNonce++
        return result.fold(
            onSuccess = { info ->
                withContext(Dispatchers.IO) { clipLibraryRepository.recordExport(info) }
                processor.markClipExported(clip.id, markedPath = info.filePath)
                _exportState.value = StudioExportUiState(
                    statusMessage = "Saved marked 30s MP4: ${File(info.filePath).name}"
                )
                Log.i(TAG, "AUTO_BURN_MARKS_DONE clip=${clip.id} path=${info.filePath} size=${info.sizeBytes}")
                true
            },
            onFailure = { err ->
                Log.e(TAG, "Marked export error for ${clip.id}: ${err.message}", err)
                _exportState.value = StudioExportUiState(
                    statusMessage = "Marked export error: ${err.message ?: "failed"}"
                )
                false
            }
        )
    }

    fun executeAutomation(cmd: AutomationCommand) {
        automationJob?.cancel()
        automationJob = viewModelScope.launch {
            Log.i(
                TAG,
                "AUTOMATION_START seq=${cmd.seq} video=${cmd.videoPath} youtube=${cmd.youtubeUrl} " +
                    "duration=${cmd.analysisDuration} startOffset=${cmd.startOffsetSec} select=${cmd.selectPlayer} " +
                    "autoSplit=${cmd.autoSplit} burnMarks=${cmd.burnMarks} speed=${cmd.playbackSpeed}"
            )
            setSpeed(cmd.playbackSpeed)

            if (!cmd.analysisDuration.isNullOrBlank()) {
                val (preset, parsedLimit) = AnalysisDurationPreset.fromToken(cmd.analysisDuration)
                if (preset != null || parsedLimit != null) {
                    setAnalysisDurationLimit(parsedLimit)
                }
            }

            val targetSource = cmd.youtubeUrl?.takeIf { it.isNotBlank() }
                ?: cmd.videoPath?.takeIf { it.isNotBlank() }
            if (!targetSource.isNullOrBlank()) {
                val isYt = YouTubeHockeyResolver.isYouTubeUrl(targetSource)
                val title = if (isYt) "YouTube Hockey" else File(targetSource).name.ifBlank { "Auto Hockey Clip" }
                loadVideo(
                    title = title,
                    source = targetSource,
                    autoPlay = true,
                    startOffsetSecOverride = cmd.startOffsetSec
                )
                videoLoadJob?.join()
                if (isYt && youtubeError != null) {
                    Log.e(TAG, "AUTOMATION_ABORT_YT_ERROR err=$youtubeError")
                    return@launch
                }
            }

            if (cmd.selectPlayer.isNullOrBlank() && !cmd.autoSplit && !cmd.burnMarks) {
                return@launch
            }

            // Wait for real-time scan window so YOLO + OCR + Gemini Nano build player tracks & windows
            val scanStartMs = SystemClock.elapsedRealtime()
            val explicitTarget = cmd.selectPlayer?.trim()?.trimStart('#')
            if (!explicitTarget.isNullOrBlank() && !explicitTarget.equals("auto", ignoreCase = true)) {
                processor.selectPlayerNumber(explicitTarget, null)
            }

            var chosenNumber: String? = null
            var chosenTeam: TeamId? = null
            val limit = analysisLimitSec
            val isShortBoundedRun = limit != null && limit <= 60.0
            val maxWaitMs = if (limit != null && limit <= 60.0) {
                (limit * 2_000.0).toLong().coerceIn(35_000L, 140_000L)
            } else {
                26_000L
            }

            while (SystemClock.elapsedRealtime() - scanStartMs < maxWaitMs) {
                delay(650L)
                val st = processor.state.value
                val dur = st.telemetry.videoDurationSec.takeIf { it > 1.0 } ?: 120.0
                val availableSpan = (dur - st.telemetry.analysisStartSec).coerceAtLeast(1.0)
                val targetScanSec = kotlin.math.min(availableSpan * 0.85, 31.5)
                val spanSec = st.telemetry.analyzedSpanSec
                val elapsed = SystemClock.elapsedRealtime() - scanStartMs

                if (st.telemetry.analysisBudgetReached) {
                    if (!explicitTarget.isNullOrBlank() && !explicitTarget.equals("auto", ignoreCase = true)) {
                        chosenNumber = explicitTarget
                        chosenTeam = st.selectedTeamFilter
                    } else {
                        val best = st.discoveredPlayers.maxByOrNull { it.totalSeenSec + it.clipCount30s * 3.0 }
                        if (best != null) {
                            chosenNumber = best.jerseyNumber
                            chosenTeam = best.team
                            processor.selectPlayerNumber(best.jerseyNumber, null)
                        }
                    }
                    break
                }

                if (player.playbackState == Player.STATE_ENDED && elapsed > 3_000L) {
                    break
                }

                if (!isShortBoundedRun) {
                    if (!explicitTarget.isNullOrBlank() && !explicitTarget.equals("auto", ignoreCase = true)) {
                        if (st.selectedPlayerClips.isNotEmpty() && spanSec >= targetScanSec) {
                            chosenNumber = explicitTarget
                            chosenTeam = st.selectedTeamFilter
                            break
                        }
                    } else {
                        val readyPlayers = st.discoveredPlayers
                        val enoughCoverage = spanSec >= targetScanSec || elapsed >= 18_000L
                        if (enoughCoverage && readyPlayers.isNotEmpty()) {
                            val best = readyPlayers.maxByOrNull { it.totalSeenSec + it.clipCount30s * 3.0 }
                            if (best != null) {
                                chosenNumber = best.jerseyNumber
                                chosenTeam = best.team
                                processor.selectPlayerNumber(best.jerseyNumber, null)
                                break
                            }
                        }
                    }
                }
            }

            if (chosenNumber == null && !explicitTarget.isNullOrBlank() && !explicitTarget.equals("auto", ignoreCase = true)) {
                chosenNumber = explicitTarget
                chosenTeam = processor.state.value.selectedTeamFilter
            }

            // Fallback if "auto" had no OCR/Nano jersey number yet: pick top tracked skater ID
            if (chosenNumber == null && (explicitTarget == null || explicitTarget.equals("auto", ignoreCase = true))) {
                val st = processor.state.value
                val fallbackTop = st.discoveredPlayers.maxByOrNull { it.totalSeenSec }
                if (fallbackTop != null) {
                    chosenNumber = fallbackTop.jerseyNumber
                    chosenTeam = fallbackTop.team
                    processor.selectPlayerNumber(fallbackTop.jerseyNumber, null)
                } else {
                    val trackCounts = HashMap<Int, Pair<Int, TeamId>>()
                    for (sample in st.history) {
                        for (p in sample.players) {
                            val prev = trackCounts[p.trackId]
                            trackCounts[p.trackId] = ((prev?.first ?: 0) + 1) to p.team
                        }
                    }
                    val topEntry = trackCounts.maxByOrNull { it.value.first }
                    if (topEntry != null) {
                        val num = topEntry.key.toString()
                        chosenNumber = num
                        chosenTeam = topEntry.value.second
                        processor.assignTrackJerseyNumber(topEntry.key, num, topEntry.value.second)
                        processor.selectPlayerNumber(num, null)
                    }
                }
            }

            delay(400L)
            val afterSelect = processor.state.value
            Log.i(
                TAG,
                "AUTO_PLAYER_SELECTED player=#$chosenNumber team=$chosenTeam " +
                    "discovered=${afterSelect.discoveredPlayers.size} clips=${afterSelect.selectedPlayerClips.size} " +
                    "processedSec=${String.format(Locale.US, "%.1f", afterSelect.telemetry.processedSec)} " +
                    "analyzedSpanSec=${String.format(Locale.US, "%.1f", afterSelect.telemetry.analyzedSpanSec)}"
            )

            val topClip = afterSelect.selectedPlayerClips.maxByOrNull {
                it.visibilitySec * (0.7 + 0.3 * it.avgLiveProb)
            } ?: afterSelect.selectedPlayerClips.firstOrNull()

            if (topClip == null) {
                Log.w(TAG, "AUTOMATION_NO_CLIP_WINDOW player=#$chosenNumber history=${afterSelect.history.size}")
                return@launch
            }

            // Ensure video scan has reached near the end of topClip before burning marks
            if (cmd.burnMarks) {
                val waitWindowStart = SystemClock.elapsedRealtime()
                while (SystemClock.elapsedRealtime() - waitWindowStart < 10_000L) {
                    val tel = processor.state.value.telemetry
                    if (tel.processedSec >= topClip.endSec - 1.5 ||
                        tel.analysisBudgetReached ||
                        player.playbackState == Player.STATE_ENDED
                    ) {
                        break
                    }
                    delay(500L)
                }
            }

            var fastOk = !cmd.autoSplit
            var markedOk = !cmd.burnMarks

            if (cmd.autoSplit) {
                Log.i(TAG, "AUTO_FAST_SPLIT_START clip=${topClip.id} range=${topClip.startSec}..${topClip.endSec}")
                fastOk = runFastSplitSuspend(topClip)
            }
            if (cmd.burnMarks) {
                val sampleCount = processor.state.value.history.count {
                    it.timestampSec in (topClip.startSec - 1.0)..(topClip.endSec + 1.0)
                }
                Log.i(TAG, "AUTO_BURN_MARKS_START clip=${topClip.id} samples=$sampleCount")
                markedOk = runBurnMarksSuspend(topClip)
            }

            Log.i(TAG, "AUTOMATION_COMPLETE fastOk=$fastOk markedOk=$markedOk clip=${topClip.id}")
        }
    }

    override fun onCleared() {
        videoLoadJob?.cancel()
        automationJob?.cancel()
        player.removeListener(playerListener)
        player.release()
        processor.close()
        super.onCleared()
    }
}

class MainActivity : ComponentActivity() {
    private var cmdSeq = 0
    private var pendingAutomation by mutableStateOf<AutomationCommand?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        parseAutomationIntent(intent)?.let { pendingAutomation = it }

        setContent {
            HockeyNanoTheme {
                HockeyNanoBenchApp(automationCommand = pendingAutomation)
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        parseAutomationIntent(intent)?.let { pendingAutomation = it }
    }

    private fun parseAutomationIntent(intent: Intent?): AutomationCommand? {
        if (intent == null) return null
        val videoPath = intent.getStringExtra("video_path")
        val youtubeUrl = intent.getStringExtra("youtube_url")
        val analysisDuration = intent.getStringExtra("analysis_duration")
        val startOffsetSec = readDoubleExtraCompat(intent, "start_offset_sec")
        val selectPlayer = intent.getStringExtra("select_player")
        val autoSplit = intent.getBooleanExtra("auto_split", false)
        val burnMarks = intent.getBooleanExtra("burn_marks", false)
        val speed = readFloatExtraCompat(intent, "playback_speed", 1.0f)
        val hasAnyExtra = !youtubeUrl.isNullOrBlank() ||
            !videoPath.isNullOrBlank() ||
            !analysisDuration.isNullOrBlank() ||
            startOffsetSec != null ||
            !selectPlayer.isNullOrBlank() ||
            autoSplit ||
            burnMarks ||
            intent.hasExtra("playback_speed")
        if (!hasAnyExtra) return null

        val cmd = AutomationCommand(
            seq = ++cmdSeq,
            videoPath = videoPath,
            youtubeUrl = youtubeUrl,
            analysisDuration = analysisDuration,
            startOffsetSec = startOffsetSec,
            selectPlayer = selectPlayer,
            autoSplit = autoSplit,
            burnMarks = burnMarks,
            playbackSpeed = speed
        )
        Log.i(
            TAG,
            "INTENT_RECEIVED seq=${cmd.seq} video=${cmd.videoPath} youtube=${cmd.youtubeUrl} " +
                "duration=${cmd.analysisDuration} startOffset=${cmd.startOffsetSec} select=${cmd.selectPlayer} " +
                "autoSplit=${cmd.autoSplit} burnMarks=${cmd.burnMarks} speed=${cmd.playbackSpeed}"
        )
        return cmd
    }

    @Suppress("DEPRECATION")
    private fun readFloatExtraCompat(intent: Intent, key: String, defaultVal: Float): Float {
        val raw = intent.extras?.get(key) ?: return defaultVal
        return when (raw) {
            is Number -> raw.toFloat()
            is String -> raw.toFloatOrNull() ?: defaultVal
            else -> defaultVal
        }
    }

    @Suppress("DEPRECATION")
    private fun readDoubleExtraCompat(intent: Intent, key: String): Double? {
        val raw = intent.extras?.get(key) ?: return null
        return when (raw) {
            is Number -> raw.toDouble()
            is String -> raw.toDoubleOrNull()
            else -> null
        }
    }
}

@Composable
private fun HockeyNanoBenchApp(
    automationCommand: AutomationCommand?,
    vm: HockeyStudioViewModel = viewModel()
) {
    val context = LocalContext.current
    val pipelineState by vm.pipelineState.collectAsStateWithLifecycle()
    val exportState by vm.exportState.collectAsStateWithLifecycle()
    var activeTab by rememberTabState()
    var hasMediaPermission by rememberMediaPermissionState(context)

    val permissionName = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        Manifest.permission.READ_MEDIA_VIDEO
    } else {
        Manifest.permission.READ_EXTERNAL_STORAGE
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasMediaPermission = granted
        Log.i(TAG, "MEDIA_PERMISSION_RESULT granted=$granted")
        vm.refreshDiscoveredVideos(autoSelectFirst = false)
    }

    LaunchedEffect(Unit) {
        if (!hasMediaPermission) {
            permissionLauncher.launch(permissionName)
        }
    }

    LaunchedEffect(automationCommand?.seq) {
        val cmd = automationCommand ?: return@LaunchedEffect
        activeTab = AppTab.STUDIO
        vm.executeAutomation(cmd)
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = ObsidianBg,
        bottomBar = {
            NavigationBar(
                containerColor = PanelBg,
                contentColor = TextPrimary,
                tonalElevation = 0.dp
            ) {
                AppTab.entries.forEach { tab ->
                    val selected = activeTab == tab
                    NavigationBarItem(
                        selected = selected,
                        onClick = { activeTab = tab },
                        icon = { Icon(tab.icon, contentDescription = tab.label) },
                        label = {
                            Text(
                                text = tab.label,
                                fontSize = 10.sp,
                                fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium
                            )
                        },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = BenchGreen,
                            selectedTextColor = BenchGreen,
                            unselectedIconColor = TextMuted,
                            unselectedTextColor = TextMuted,
                            indicatorColor = BorderSubtle
                        )
                    )
                }
            }
        }
    ) { innerPadding ->
        val contentModifier = Modifier
            .fillMaxSize()
            .padding(innerPadding)

        when (activeTab) {
            AppTab.STUDIO -> NanoHockeyStudioScreen(
                player = vm.player,
                pipelineState = pipelineState,
                discoveredVideos = vm.discoveredVideos,
                selectedVideoTitle = vm.selectedVideoTitle,
                selectedVideoSource = vm.selectedVideoSource,
                playbackSpeed = vm.playbackSpeed,
                clipPlayRange = vm.clipPlayRange,
                exportState = exportState,
                hasMediaPermission = hasMediaPermission,
                analysisLimitSec = vm.analysisLimitSec,
                isResolvingYouTube = vm.isResolvingYouTube,
                youtubeError = vm.youtubeError,
                resolvedYoutube = vm.resolvedYoutube,
                onRequestMediaPermission = { permissionLauncher.launch(permissionName) },
                onRefreshVideos = { vm.refreshDiscoveredVideos(autoSelectFirst = false) },
                onSelectVideo = { title, src -> vm.loadVideo(title, src, autoPlay = true) },
                onAnalyzeYouTubeUrl = { url -> vm.analyzeYouTubeUrl(url) },
                onSetAnalysisDurationLimit = { limitSec -> vm.setAnalysisDurationLimit(limitSec) },
                onToggleSpeed = { vm.setSpeed(it) },
                onTogglePanoramic = { vm.setPanoramic(it) },
                onVideoDimensionsKnown = { dur, w, h -> vm.onVideoDimensionsKnown(dur, w, h) },
                onVideoFrame = { bmp, ts -> vm.onVideoFrame(bmp, ts) },
                onSelectPlayer = { num, team -> vm.selectPlayer(num, team) },
                onAssignTrack = { trackId, num, team -> vm.assignTrack(trackId, num, team) },
                onPlayClipWindow = { vm.playClipWindow(it) },
                onClearClipRange = { vm.clearClipRange() },
                onFastSplitWindow = { vm.launchFastSplit(it) },
                onBurnMarksWindow = { vm.launchBurnMarks(it) },
                onSplitAllWindows = { vm.launchSplitAll() },
                modifier = contentModifier
            )

            AppTab.CLIPS -> SavedClipsScreen(
                clipLibraryRepository = vm.clipLibraryRepository,
                refreshNonce = vm.clipsRefreshNonce,
                modifier = contentModifier
            )

            AppTab.BENCH -> BenchReportScreen(
                benchSeedRepository = vm.benchSeedRepository,
                nanoStatus = pipelineState.telemetry.nanoStatus,
                onAskAssistant = { q, ctx -> vm.processor.askNanoBenchAssistant(q, ctx) },
                modifier = contentModifier
            )
        }
    }
}

@Composable
private fun rememberTabState() = androidx.compose.runtime.remember {
    mutableStateOf(AppTab.STUDIO)
}

@Composable
private fun rememberMediaPermissionState(context: Context) = androidx.compose.runtime.remember {
    mutableStateOf(checkVideoReadPermission(context))
}

private fun checkVideoReadPermission(context: Context): Boolean {
    val perm = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        Manifest.permission.READ_MEDIA_VIDEO
    } else {
        Manifest.permission.READ_EXTERNAL_STORAGE
    }
    return ContextCompat.checkSelfPermission(context, perm) == PackageManager.PERMISSION_GRANTED
}

private fun resolvePlayableUri(context: Context, source: String): Uri {
    if (source.contains("://")) return Uri.parse(source)
    val file = File(source)
    if (file.canRead()) return Uri.fromFile(file)
    return queryMediaStoreVideoUri(context, source) ?: Uri.fromFile(file)
}

private fun resolveExtractorSource(context: Context, source: String): String {
    if (source.contains("://")) return source
    val file = File(source)
    if (file.canRead()) return source
    return queryMediaStoreVideoUri(context, source)?.toString() ?: source
}

private fun queryMediaStoreVideoUri(context: Context, rawPath: String): Uri? {
    val name = File(rawPath).name
    if (name.isBlank()) return null
    return runCatching {
        context.contentResolver.query(
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
            arrayOf(MediaStore.Video.Media._ID, MediaStore.Video.Media.DISPLAY_NAME),
            "${MediaStore.Video.Media.DISPLAY_NAME} = ?",
            arrayOf(name),
            null
        )?.use { cursor ->
            if (cursor.moveToFirst()) {
                val id = cursor.getLong(0)
                ContentUris.withAppendedId(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, id)
            } else {
                null
            }
        }
    }.getOrNull()
}

private fun probeDurationSeconds(context: Context, uri: Uri): Double {
    val scheme = uri.scheme?.lowercase(Locale.US)
    if (scheme == "http" || scheme == "https") return 0.0
    val retriever = MediaMetadataRetriever()
    return try {
        when (scheme) {
            "file", null -> {
                val p = uri.path ?: return 0.0
                retriever.setDataSource(p)
            }
            else -> retriever.setDataSource(context, uri)
        }
        val msStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
        (msStr?.toLongOrNull() ?: 0L) / 1000.0
    } catch (_: Throwable) {
        0.0
    } finally {
        runCatching { retriever.release() }
    }
}
