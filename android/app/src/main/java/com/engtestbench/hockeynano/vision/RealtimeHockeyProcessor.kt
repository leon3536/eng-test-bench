package com.engtestbench.hockeynano.vision

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.engtestbench.hockeynano.core.clip.PlayerClipSegmenter
import com.engtestbench.hockeynano.core.model.ClipWindow30s
import com.engtestbench.hockeynano.core.model.DiscoveredPlayerSummary
import com.engtestbench.hockeynano.core.model.FrameMarkSample
import com.engtestbench.hockeynano.core.model.PipelineState
import com.engtestbench.hockeynano.core.model.PipelineTelemetry
import com.engtestbench.hockeynano.core.model.RectNorm
import com.engtestbench.hockeynano.core.model.TeamId
import com.engtestbench.hockeynano.core.model.TrackedPlayer
import com.engtestbench.hockeynano.core.tracking.MultiTeamPlayerTracker
import com.engtestbench.hockeynano.core.tracking.OnIceFilter
import com.engtestbench.hockeynano.core.tracking.RawDetection
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.Closeable
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Coordinates real-time on-device hockey tracking, dual-pass ML Kit jersey OCR (~400ms cadence),
 * and single-flight Pixel 11 Pro Gemini Nano multimodal VLM tasks (team kit labeling, 2x2 jersey grids,
 * 30s shift summaries, and eng-test-bench assistant) while publishing [PipelineState].
 */
class RealtimeHockeyProcessor(private val context: Context) : Closeable {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val detector: YoloOnnxDetector? = YoloOnnxDetector.load(context, threads = 4)
    private val jerseyRecognizer = JerseyNumberRecognizer()
    private val nanoEngine = GeminiNanoHockeyEngine(context)
    private val tracker = MultiTeamPlayerTracker()

    private val stateLock = Any()
    private val cropLock = Any()
    private val nanoLaunchLock = Any()
    private val yoloBusy = AtomicBoolean(false)
    private val nanoReschedule = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)
    private val sessionGen = AtomicLong(0L)

    private val _state = MutableStateFlow(PipelineState())
    val state: StateFlow<PipelineState> = _state.asStateFlow()

    private var videoTitle: String = ""
    private var videoSource: String = ""
    private var videoDurationSec: Double = 0.0
    private var panoramicTiling: Boolean = true
    private var selectedPlayerNumber: String? = null
    private var selectedTeamFilter: TeamId? = null
    private var analysisLimitSec: Double? = null
    private var analysisStartSecConfigured: Double = 0.0
    private var analysisAnchorSec: Double = 0.0
    private var hasProcessedFirstFrame: Boolean = false
    private var analysisBudgetReached: Boolean = false
    private var lastFullRecomputeWallMs: Long = 0L
    private var nextCompactionThreshold: Int = COMPACT_HISTORY_THRESHOLD + 1
    private var cachedDiscoveredPlayers: List<DiscoveredPlayerSummary> = emptyList()
    private var cachedRawClips: List<ClipWindow30s> = emptyList()

    private val historySamples = ArrayList<FrameMarkSample>(512)
    private var currentSample: FrameMarkSample? = null
    private var lastTrackerTimestampSec: Double = Double.NaN
    private var maxHistoryTimeSec: Double = 0.0

    private val clipSummariesById = ConcurrentHashMap<String, String>()
    private val exportedFastPaths = ConcurrentHashMap<String, String>()
    private val exportedMarkedPaths = ConcurrentHashMap<String, String>()
    private val ocrAttempts = ConcurrentHashMap<Int, Int>()
    private val nanoGridAttempts = ConcurrentHashMap<Int, Int>()
    private val trackHitCounts = ConcurrentHashMap<Int, Int>()

    private val ocrReadsTotal = AtomicInteger(0)
    private val nanoCallsTotal = AtomicInteger(0)

    private var fpsEma: Float = 0f
    private var lastFrameWallMs: Long = 0L
    private var lastDetectorMs: Float = 0f
    private var lastNanoLatencyMs: Long = 0L
    private var lastNanoMessage: String = ""

    private var ocrJob: Job? = null
    private var nanoJob: Job? = null
    private var lastOcrWallMs: Long = 0L
    private var lastNanoGridWallMs: Long = 0L
    private var lastNanoTeamWallMs: Long = 0L
    private var teamsNanoVerified: Boolean = false
    private var lastVerifiedTeamALabel: String = ""
    private var lastVerifiedTeamBLabel: String = ""
    private var teamLabelAttempts: Int = 0
    private var lastPreviewCaptureWallMs: Long = 0L

    private val pendingTeamACrops = ArrayList<Pair<Int, Bitmap>>(3)
    private val pendingTeamBCrops = ArrayList<Pair<Int, Bitmap>>(3)
    private val pendingNanoGridItems = ArrayList<GridCropItem>(4)
    private var latestPreviewBitmap: Bitmap? = null

    private val statusWatcherJob: Job

    init {
        statusWatcherJob = scope.launch {
            nanoEngine.status.collect { statusText ->
                synchronized(stateLock) {
                    emitStateLocked(nanoStatusOverride = statusText)
                }
            }
        }
        scope.launch {
            nanoEngine.refreshStatus()
        }
    }

    fun startSession(
        videoTitle: String,
        videoSource: String,
        durationSec: Double,
        panoramicTiling: Boolean = true,
        analysisLimitSec: Double? = null,
        analysisStartSec: Double = 0.0,
    ) {
        if (closed.get()) return
        val gen = sessionGen.incrementAndGet()
        ocrJob?.cancel()
        nanoJob?.cancel()

        detector?.resetTemporalState()
        clearAllCropBuffers()

        val cleanLimit = analysisLimitSec?.takeIf { it.isFinite() && it >= 0.0 }
        val safeStart = analysisStartSec.takeIf { it.isFinite() && it >= 0.0 } ?: 0.0

        synchronized(stateLock) {
            tracker.reset()
            this.videoTitle = videoTitle
            this.videoSource = videoSource
            this.videoDurationSec = durationSec.coerceAtLeast(0.0)
            this.panoramicTiling = panoramicTiling
            this.selectedPlayerNumber = null
            this.selectedTeamFilter = null
            this.analysisLimitSec = cleanLimit
            this.analysisStartSecConfigured = safeStart
            this.analysisAnchorSec = safeStart
            this.hasProcessedFirstFrame = false
            this.analysisBudgetReached = false
            this.lastFullRecomputeWallMs = 0L
            this.nextCompactionThreshold = COMPACT_HISTORY_THRESHOLD + 1
            this.cachedDiscoveredPlayers = emptyList()
            this.cachedRawClips = emptyList()

            historySamples.clear()
            currentSample = null
            lastTrackerTimestampSec = Double.NaN
            maxHistoryTimeSec = 0.0

            clipSummariesById.clear()
            exportedFastPaths.clear()
            exportedMarkedPaths.clear()
            ocrAttempts.clear()
            nanoGridAttempts.clear()
            trackHitCounts.clear()

            ocrReadsTotal.set(0)
            nanoCallsTotal.set(0)
            fpsEma = 0f
            lastFrameWallMs = 0L
            lastDetectorMs = 0f
            lastNanoLatencyMs = 0L
            lastNanoMessage = ""
            lastOcrWallMs = 0L
            lastNanoGridWallMs = 0L
            lastNanoTeamWallMs = 0L
            teamsNanoVerified = false
            lastVerifiedTeamALabel = ""
            lastVerifiedTeamBLabel = ""
            teamLabelAttempts = 0
            lastPreviewCaptureWallMs = 0L

            emitStateLocked(forceFullRecompute = false)
        }

        Log.i(
            TAG,
            "SESSION_START title=\"$videoTitle\" duration=${"%.1f".format(Locale.US, durationSec)}s " +
                "panoramic=$panoramicTiling limitSec=$cleanLimit startSec=${"%.1f".format(Locale.US, safeStart)} " +
                "detectorReady=${detector != null}",
        )

        scope.launch {
            if (sessionGen.get() == gen) {
                nanoEngine.refreshStatus()
            }
        }
    }

    fun setAnalysisDurationLimit(limitSec: Double?) {
        val cleanLimit = limitSec?.takeIf { it.isFinite() && it >= 0.0 }
        synchronized(stateLock) {
            analysisLimitSec = cleanLimit
            val analyzedSpanSec = computeAnalyzedSpanSecLocked()
            val wasReached = analysisBudgetReached
            val nowReached = cleanLimit != null && analyzedSpanSec >= cleanLimit
            analysisBudgetReached = nowReached
            if (!wasReached && nowReached) {
                Log.i(
                    TAG,
                    "ANALYSIS_LIMIT_REACHED limitSec=$cleanLimit analyzedSpanSec=${"%.1f".format(Locale.US, analyzedSpanSec)} samples=${historySamples.size}",
                )
            }
            emitStateLocked(forceFullRecompute = true)
        }
    }

    suspend fun onVideoFrame(highResBitmap: Bitmap, timestampSec: Double) {
        if (closed.get() || highResBitmap.isRecycled) return
        if (synchronized(stateLock) { analysisBudgetReached }) return
        if (!yoloBusy.compareAndSet(false, true)) return

        try {
            withContext(Dispatchers.Default) {
                if (closed.get() || highResBitmap.isRecycled) return@withContext
                val detEngine = detector ?: return@withContext
                val gen = sessionGen.get()

                val (useTiling, budgetReachedBeforeDetect) = synchronized(stateLock) {
                    panoramicTiling to analysisBudgetReached
                }
                if (budgetReachedBeforeDetect || sessionGen.get() != gen) return@withContext

                val yoloResult = detEngine.detect(highResBitmap, timestampSec, useTiling)
                if (sessionGen.get() != gen) return@withContext

                val aspectRatio = highResBitmap.width.toFloat() / highResBitmap.height.coerceAtLeast(1).toFloat()
                val rawDetections = yoloResult.detections
                    .filter { OnIceFilter.isOnIce(it.box, aspectRatio, useTiling) }
                    .map { RawDetection(it.box, it.chestRgb) }

                val nowMs = SystemClock.elapsedRealtime()
                val frameResult: Pair<FrameMarkSample, Boolean>? = synchronized(stateLock) {
                    if (sessionGen.get() != gen || analysisBudgetReached) return@synchronized null

                    if (!hasProcessedFirstFrame) {
                        hasProcessedFirstFrame = true
                        if (analysisStartSecConfigured <= 0.0 && analysisAnchorSec <= 0.0 && timestampSec > 5.0) {
                            analysisAnchorSec = timestampSec
                        }
                    }

                    if (lastFrameWallMs > 0L) {
                        val dtSec = (nowMs - lastFrameWallMs) / 1000f
                        if (dtSec in 0.02f..3.0f) {
                            val instFps = 1f / dtSec
                            fpsEma = if (fpsEma <= 0f) instFps else 0.7f * fpsEma + 0.3f * instFps
                        }
                    }
                    lastFrameWallMs = nowMs
                    lastDetectorMs = yoloResult.detectorMs

                    val isBackwardSeek = historySamples.isNotEmpty() &&
                        !lastTrackerTimestampSec.isNaN() &&
                        timestampSec < lastTrackerTimestampSec - 1.0

                    val sampleForOverlay: FrameMarkSample = if (isBackwardSeek) {
                        val nearby = historySamples.minByOrNull { abs(it.timestampSec - timestampSec) }
                            ?.takeIf { abs(it.timestampSec - timestampSec) <= 0.45 }
                        if (nearby != null) {
                            nearby
                        } else {
                            val s = tracker.updateFrame(timestampSec, rawDetections, yoloResult.livePlayProb)
                            insertHistorySampleSortedLocked(s)
                            lastTrackerTimestampSec = timestampSec
                            maxHistoryTimeSec = max(maxHistoryTimeSec, timestampSec)
                            s
                        }
                    } else {
                        val s = tracker.updateFrame(timestampSec, rawDetections, yoloResult.livePlayProb)
                        if (historySamples.isEmpty() || timestampSec - historySamples.last().timestampSec >= 0.08) {
                            historySamples.add(s)
                        }
                        maxHistoryTimeSec = max(maxHistoryTimeSec, timestampSec)
                        lastTrackerTimestampSec = timestampSec
                        s
                    }

                    compactOlderHalfIfNeededLocked()
                    currentSample = sampleForOverlay

                    val analyzedSpanSec = computeAnalyzedSpanSecLocked()
                    val limit = analysisLimitSec
                    val reachedNow = limit != null && analyzedSpanSec >= limit
                    val transitionedToReached = !analysisBudgetReached && reachedNow
                    if (reachedNow) {
                        analysisBudgetReached = true
                    }
                    if (transitionedToReached) {
                        Log.i(
                            TAG,
                            "ANALYSIS_LIMIT_REACHED limitSec=$limit analyzedSpanSec=${"%.1f".format(Locale.US, analyzedSpanSec)} samples=${historySamples.size}",
                        )
                    }

                    val shouldFullRecompute = transitionedToReached ||
                        lastFullRecomputeWallMs == 0L ||
                        (nowMs - lastFullRecomputeWallMs >= FULL_RECOMPUTE_CADENCE_MS)

                    emitStateLocked(forceFullRecompute = shouldFullRecompute)
                    sampleForOverlay to reachedNow
                }

                if (frameResult == null || sessionGen.get() != gen) return@withContext
                val (frameSample, budgetReachedNow) = frameResult
                frameSample.players.forEach { p ->
                    trackHitCounts[p.trackId] = (trackHitCounts[p.trackId] ?: 0) + 1
                }

                Log.i(
                    TAG,
                    "FRAME_STEP t=${"%.2f".format(Locale.US, timestampSec)}s detMs=${"%.1f".format(Locale.US, yoloResult.detectorMs)} " +
                        "tiled=${yoloResult.tiledPassUsed} det=${yoloResult.detections.size} onIce=${rawDetections.size} " +
                        "live=${"%.2f".format(Locale.US, yoloResult.livePlayProb)} ocr=${ocrReadsTotal.get()} nano=${nanoEngine.status.value}",
                )

                if (!budgetReachedNow) {
                    scheduleAsyncOcrIfDue(highResBitmap, frameSample, nowMs, gen)
                    captureNanoCrops(highResBitmap, frameSample, nowMs)
                    maybeLaunchNanoWorker(gen)
                }
            }
        } finally {
            yoloBusy.set(false)
        }
    }

    fun selectPlayerNumber(jerseyNumber: String?, teamFilter: TeamId? = null) {
        val clean = jerseyNumber?.trim()?.removePrefix("#")?.takeIf { it.isNotEmpty() }
        val gen = sessionGen.get()
        synchronized(stateLock) {
            selectedPlayerNumber = clean
            selectedTeamFilter = teamFilter
            emitStateLocked(forceFullRecompute = true)
        }
        if (clean != null) {
            maybeLaunchNanoWorker(gen)
        }
    }

    fun assignTrackJerseyNumber(trackId: Int, jerseyNumber: String, team: TeamId? = null) {
        val clean = jerseyNumber.trim().removePrefix("#")
        if (clean.isEmpty()) return
        val gen = sessionGen.get()
        synchronized(stateLock) {
            tracker.assignTrackJerseyNumber(trackId, clean, team)
            emitStateLocked(forceFullRecompute = true)
        }
        maybeLaunchNanoWorker(gen)
    }

    fun setPanoramicTiling(enabled: Boolean) {
        synchronized(stateLock) {
            if (panoramicTiling == enabled) return
            panoramicTiling = enabled
            emitStateLocked()
        }
    }

    fun markClipExported(clipId: String, fastPath: String? = null, markedPath: String? = null) {
        if (fastPath != null) exportedFastPaths[clipId] = fastPath
        if (markedPath != null) exportedMarkedPaths[clipId] = markedPath
        synchronized(stateLock) {
            emitStateLocked()
        }
    }

    suspend fun askNanoBenchAssistant(question: String, reportContext: String): String {
        val response = nanoEngine.askBenchAssistant(question, reportContext)
        synchronized(stateLock) {
            if (response.fromNano) nanoCallsTotal.incrementAndGet()
            if (response.latencyMs > 0L) lastNanoLatencyMs = response.latencyMs
            lastNanoMessage = "Bench Q&A (${response.latencyMs}ms)"
            emitStateLocked()
        }
        return response.text
    }

    private fun scheduleAsyncOcrIfDue(
        highResBitmap: Bitmap,
        sample: FrameMarkSample,
        nowMs: Long,
        gen: Long,
    ) {
        if (nowMs - lastOcrWallMs < OCR_CADENCE_MS || ocrJob?.isActive == true) return

        val priorityPlayers = sample.players
            .filter { it.box.height >= MIN_OCR_BOX_HEIGHT && it.box.width >= 0.015f }
            .sortedWith(
                compareBy<TrackedPlayer>(
                    {
                        when {
                            it.jerseyNumber == null -> 0
                            !it.nanoVerified -> 1
                            else -> 2
                        }
                    },
                    { ocrAttempts[it.trackId] ?: 0 },
                    { -(it.box.width * it.box.height) },
                ),
            )
            .take(MAX_OCR_TRACKS_PER_STEP)

        if (priorityPlayers.isEmpty()) return

        val tasks = ArrayList<Pair<Int, Bitmap>>(priorityPlayers.size)
        for (player in priorityPlayers) {
            val crop = jerseyRecognizer.cropUpperTorso(highResBitmap, player.box) ?: continue
            ocrAttempts[player.trackId] = (ocrAttempts[player.trackId] ?: 0) + 1
            tasks.add(player.trackId to crop)
        }
        if (tasks.isEmpty()) return

        lastOcrWallMs = nowMs
        ocrJob = scope.launch(Dispatchers.Default) {
            try {
                var anyRead = false
                for ((trackId, crop) in tasks) {
                    try {
                        if (!isActive || sessionGen.get() != gen) continue
                        val res = jerseyRecognizer.recognizeFromTorsoCrop(crop)
                        if (res != null && sessionGen.get() == gen) {
                            ocrReadsTotal.incrementAndGet()
                            anyRead = true
                            synchronized(stateLock) {
                                if (sessionGen.get() == gen) {
                                    tracker.recordJerseyRead(trackId, res.jerseyNumber, res.confidence, false)
                                }
                            }
                        }
                    } finally {
                        if (!crop.isRecycled) crop.recycle()
                    }
                }
                if (sessionGen.get() == gen) {
                    synchronized(stateLock) {
                        if (sessionGen.get() == gen) {
                            emitStateLocked(forceFullRecompute = anyRead)
                        }
                    }
                }
            } catch (t: Throwable) {
                tasks.forEach { (_, c) -> if (!c.isRecycled) c.recycle() }
                Log.w(TAG, "Async OCR step failed: ${t.message}")
            }
        }
    }

    private fun captureNanoCrops(highResBitmap: Bitmap, sample: FrameMarkSample, nowMs: Long) {
        val teamProfiles = synchronized(stateLock) { tracker.getTeamProfiles() }
        val needsTeamCrops = synchronized(stateLock) { shouldRefreshTeamLabelsLocked(nowMs) }
        synchronized(cropLock) {
            if (needsTeamCrops) {
                collectTeamCropsLocked(highResBitmap, sample, TeamId.TEAM_A, teamProfiles, pendingTeamACrops)
                collectTeamCropsLocked(highResBitmap, sample, TeamId.TEAM_B, teamProfiles, pendingTeamBCrops)
            }

            // Refresh 4 largest unconfirmed skater crops for the 2x2 Nano jersey grid.
            val gridCandidates = sample.players
                .filter {
                    it.box.height >= MIN_OCR_BOX_HEIGHT &&
                        it.box.width >= 0.015f &&
                        (it.jerseyNumber == null || !it.nanoVerified)
                }
                .sortedWith(
                    compareBy<TrackedPlayer>(
                        { if (it.jerseyNumber == null) 0 else 1 },
                        { nanoGridAttempts[it.trackId] ?: 0 },
                        { -(it.box.width * it.box.height) },
                    ),
                )
                .take(4)

            if (gridCandidates.isNotEmpty()) {
                recycleGridItemsLocked(pendingNanoGridItems)
                for (p in gridCandidates) {
                    val crop = cropScaledTorso(highResBitmap, p.box, maxHeight = 256) ?: continue
                    pendingNanoGridItems.add(GridCropItem(p.trackId, p.team, crop))
                }
            }

            // Capture a periodic downscaled frame preview for grounded Nano clip summaries.
            if (nowMs - lastPreviewCaptureWallMs >= 3000L && !highResBitmap.isRecycled) {
                lastPreviewCaptureWallMs = nowMs
                val pw = 640
                val ph = ((highResBitmap.height.toFloat() * pw) / highResBitmap.width.coerceAtLeast(1))
                    .toInt().coerceIn(180, 480)
                val scaled = runCatching { Bitmap.createScaledBitmap(highResBitmap, pw, ph, true) }.getOrNull()
                if (scaled != null) {
                    latestPreviewBitmap?.let { if (!it.isRecycled) it.recycle() }
                    latestPreviewBitmap = scaled
                }
            }
        }
    }

    private fun maybeLaunchNanoWorker(gen: Long) {
        if (closed.get()) return
        synchronized(nanoLaunchLock) {
            if (nanoJob?.isActive == true) {
                nanoReschedule.set(true)
                return
            }
            nanoReschedule.set(false)
            nanoJob = scope.launch(Dispatchers.IO) {
                try {
                    val status = nanoEngine.refreshStatus()
                    if (status != "ready" || sessionGen.get() != gen) return@launch
                    do {
                        nanoReschedule.set(false)
                        runNanoWorkCycle(gen)
                    } while (isActive && sessionGen.get() == gen && nanoReschedule.get())
                } finally {
                    val relaunch = synchronized(nanoLaunchLock) {
                        val pending = nanoReschedule.getAndSet(false)
                        nanoJob = null
                        pending
                    }
                    if (relaunch && !closed.get() && sessionGen.get() == gen) {
                        maybeLaunchNanoWorker(gen)
                    }
                }
            }
        }
    }

    private suspend fun runNanoWorkCycle(gen: Long) {
        var iterations = 0
        while (sessionGen.get() == gen && iterations < 8) {
            iterations++
            var didWork = false

            // 1. Prioritize summarizing any selected player's 30s clips that lack a Nano summary.
            val clipToSummarize = synchronized(stateLock) {
                if (sessionGen.get() != gen) null
                else _state.value.selectedPlayerClips.firstOrNull { !clipSummariesById.containsKey(it.id) }
            }
            if (clipToSummarize != null) {
                didWork = true
                val teamName = synchronized(stateLock) {
                    _state.value.teams[clipToSummarize.team]?.displayName ?: clipToSummarize.team.name
                }
                val avgSpeed = computeWindowAvgSpeed(clipToSummarize)
                val previewCopy = synchronized(cropLock) {
                    latestPreviewBitmap?.takeIf { !it.isRecycled }
                        ?.let { runCatching { it.copy(Bitmap.Config.ARGB_8888, false) }.getOrNull() }
                }
                val summaryResp = try {
                    nanoEngine.summarizeClip(
                        playerNumber = clipToSummarize.playerNumber,
                        teamName = teamName,
                        startSec = clipToSummarize.startSec,
                        endSec = clipToSummarize.endSec,
                        visibilitySec = clipToSummarize.visibilitySec,
                        avgLiveProb = clipToSummarize.avgLiveProb,
                        avgSpeedNorm = avgSpeed,
                        keyframeComposite = previewCopy,
                    )
                } finally {
                    previewCopy?.let { if (!it.isRecycled) it.recycle() }
                }

                if (sessionGen.get() == gen) {
                    clipSummariesById[clipToSummarize.id] = summaryResp.text
                    synchronized(stateLock) {
                        if (sessionGen.get() == gen) {
                            if (summaryResp.fromNano) nanoCallsTotal.incrementAndGet()
                            if (summaryResp.latencyMs > 0L) lastNanoLatencyMs = summaryResp.latencyMs
                            lastNanoMessage = "Clip #${clipToSummarize.playerNumber} summarized"
                            emitStateLocked()
                        }
                    }
                }
                continue
            }

            // 2. Multimodal team kit identification (side-by-side 768x384 composite).
            val nowMs = SystemClock.elapsedRealtime()
            val shouldLabelTeams = synchronized(stateLock) {
                shouldRefreshTeamLabelsLocked(nowMs)
            }
            if (shouldLabelTeams) {
                val (cropsA, cropsB) = snapshotTeamCrops()
                if (cropsA.size >= 2 && cropsB.size >= 2) {
                    didWork = true
                    val profiles = synchronized(stateLock) { tracker.getTeamProfiles() }
                    val rgbA = profiles[TeamId.TEAM_A]?.meanRgb ?: listOf(40, 60, 120)
                    val rgbB = profiles[TeamId.TEAM_B]?.meanRgb ?: listOf(225, 225, 225)
                    val labels = try {
                        nanoEngine.identifyTeamsMultimodal(cropsA, cropsB, rgbA, rgbB)
                    } finally {
                        cropsA.forEach { if (!it.isRecycled) it.recycle() }
                        cropsB.forEach { if (!it.isRecycled) it.recycle() }
                    }
                    if (sessionGen.get() == gen) {
                        synchronized(stateLock) {
                            if (sessionGen.get() == gen) {
                                lastNanoTeamWallMs = SystemClock.elapsedRealtime()
                                teamLabelAttempts++
                                if (labels.nanoVerified) {
                                    val validNonStriped = !isStripedOrRefereeLabel(labels.teamALabel) &&
                                        !isStripedOrRefereeLabel(labels.teamBLabel)
                                    val hadValidLabels = teamsNanoVerified &&
                                        !isStripedOrRefereeLabel(lastVerifiedTeamALabel) &&
                                        !isStripedOrRefereeLabel(lastVerifiedTeamBLabel)
                                    if (!hadValidLabels || validNonStriped) {
                                        teamsNanoVerified = true
                                        lastVerifiedTeamALabel = labels.teamALabel
                                        lastVerifiedTeamBLabel = labels.teamBLabel
                                        nanoCallsTotal.incrementAndGet()
                                        tracker.updateTeamProfile(TeamId.TEAM_A, labels.teamALabel, labels.teamALabel, true)
                                        tracker.updateTeamProfile(TeamId.TEAM_B, labels.teamBLabel, labels.teamBLabel, true)
                                    }
                                }
                                if (labels.latencyMs > 0L) lastNanoLatencyMs = labels.latencyMs
                                lastNanoMessage = "${labels.teamALabel} vs ${labels.teamBLabel}"
                                emitStateLocked()
                            }
                        }
                    }
                    continue
                } else {
                    cropsA.forEach { if (!it.isRecycled) it.recycle() }
                    cropsB.forEach { if (!it.isRecycled) it.recycle() }
                }
            }

            // 3. Multimodal 2x2 jersey number grid every ~2.8s on priority unconfirmed crops.
            if (nowMs - lastNanoGridWallMs >= NANO_GRID_CADENCE_MS) {
                val gridSnapshot = snapshotNanoGridItems()
                if (gridSnapshot.isNotEmpty()) {
                    didWork = true
                    lastNanoGridWallMs = nowMs
                    gridSnapshot.forEach {
                        nanoGridAttempts[it.trackId] = (nanoGridAttempts[it.trackId] ?: 0) + 1
                    }
                    val (reads, latencyMs) = try {
                        nanoEngine.readJerseyGridMultimodal(gridSnapshot)
                    } finally {
                        gridSnapshot.forEach { if (!it.crop.isRecycled) it.crop.recycle() }
                    }
                    if (sessionGen.get() == gen) {
                        synchronized(stateLock) {
                            if (sessionGen.get() == gen) {
                                nanoCallsTotal.incrementAndGet()
                                if (latencyMs > 0L) lastNanoLatencyMs = latencyMs
                                if (reads.isNotEmpty()) {
                                    reads.forEach { r ->
                                        tracker.recordJerseyRead(r.trackId, r.jerseyNumber, r.confidence, true)
                                    }
                                    lastNanoMessage = "Nano read ${reads.joinToString { "#${it.jerseyNumber}" }}"
                                    emitStateLocked(forceFullRecompute = true)
                                } else {
                                    lastNanoMessage = "Nano 2x2 grid (${latencyMs}ms)"
                                    emitStateLocked(forceFullRecompute = false)
                                }
                            }
                        }
                    }
                    continue
                }
            }

            if (!didWork) break
        }
    }

    private fun computeWindowAvgSpeed(clip: ClipWindow30s): Float {
        val speeds = synchronized(stateLock) {
            historySamples
                .filter { it.timestampSec in clip.startSec..clip.endSec }
                .flatMap { s ->
                    s.players.filter { p ->
                        p.jerseyNumber == clip.playerNumber &&
                            (clip.team == TeamId.UNKNOWN || p.team == clip.team)
                    }
                }
                .map { it.speedNormPerSec }
        }
        return if (speeds.isEmpty()) 0f else speeds.average().toFloat()
    }

    private fun insertHistorySampleSortedLocked(sample: FrameMarkSample) {
        val idx = historySamples.indexOfFirst { it.timestampSec > sample.timestampSec }
        if (idx < 0) historySamples.add(sample) else historySamples.add(idx, sample)
    }

    private fun computeAnalyzedSpanSecLocked(): Double =
        (maxHistoryTimeSec - analysisAnchorSec).coerceAtLeast(0.0)

    private fun compactOlderHalfIfNeededLocked() {
        if (historySamples.size <= COMPACT_HISTORY_THRESHOLD) return
        if (historySamples.size < nextCompactionThreshold) return

        val beforeSize = historySamples.size
        val halfIdx = beforeSize / 2
        val compacted = ArrayList<FrameMarkSample>(beforeSize)
        var lastKeptTime = Double.NEGATIVE_INFINITY

        for (i in 0 until halfIdx) {
            val sample = historySamples[i]
            val hasJersey = sample.players.any { !it.jerseyNumber.isNullOrBlank() }
            if (hasJersey || sample.timestampSec - lastKeptTime >= COMPACT_MIN_SPACING_SEC) {
                compacted.add(sample)
                lastKeptTime = sample.timestampSec
            }
        }
        for (i in halfIdx until beforeSize) {
            compacted.add(historySamples[i])
        }

        historySamples.clear()
        historySamples.addAll(compacted)
        nextCompactionThreshold = if (historySamples.size > COMPACT_HISTORY_THRESHOLD) {
            historySamples.size + 1_000
        } else {
            COMPACT_HISTORY_THRESHOLD + 1
        }
        Log.i(TAG, "HISTORY_COMPACTED before=$beforeSize after=${historySamples.size}")
    }

    private fun emitStateLocked(
        forceFullRecompute: Boolean = false,
        nanoStatusOverride: String? = null,
    ) {
        val effectiveDuration = max(videoDurationSec, maxHistoryTimeSec)
        val selNum = selectedPlayerNumber

        if (forceFullRecompute) {
            if (historySamples.isNotEmpty()) {
                val annotated = tracker.annotateHistory(historySamples.toList())
                historySamples.clear()
                historySamples.addAll(annotated)
            }
            currentSample = currentSample?.let { tracker.annotateHistory(listOf(it)).firstOrNull() }
            val snapshotForCompute = historySamples.toList()
            cachedDiscoveredPlayers = tracker.getDiscoveredPlayers(snapshotForCompute, effectiveDuration)
            cachedRawClips = if (!selNum.isNullOrBlank()) {
                PlayerClipSegmenter.compute30sWindows(
                    snapshotForCompute,
                    selNum,
                    selectedTeamFilter,
                    effectiveDuration,
                )
            } else {
                emptyList()
            }
            lastFullRecomputeWallMs = SystemClock.elapsedRealtime()
        }

        val historySnapshot = historySamples.toList()
        val enrichedClips = cachedRawClips.map { c ->
            c.copy(
                nanoSummary = clipSummariesById[c.id] ?: c.nanoSummary,
                fastClipPath = exportedFastPaths[c.id] ?: c.fastClipPath,
                markedClipPath = exportedMarkedPaths[c.id] ?: c.markedClipPath,
            )
        }
        val analyzedSpanSec = computeAnalyzedSpanSecLocked()
        val telemetry = PipelineTelemetry(
            nanoStatus = nanoStatusOverride ?: nanoEngine.status.value,
            fps = fpsEma,
            detectorMs = lastDetectorMs,
            ocrReadsTotal = ocrReadsTotal.get(),
            nanoCallsTotal = nanoCallsTotal.get(),
            lastNanoLatencyMs = lastNanoLatencyMs,
            lastNanoMessage = lastNanoMessage,
            panoramicTilingEnabled = panoramicTiling,
            processedSec = maxHistoryTimeSec,
            videoDurationSec = effectiveDuration,
            analysisStartSec = analysisAnchorSec,
            analyzedSpanSec = analyzedSpanSec,
            analysisLimitSec = analysisLimitSec,
            analysisBudgetReached = analysisBudgetReached,
        )
        _state.value = PipelineState(
            videoTitle = videoTitle,
            videoSource = videoSource,
            teams = tracker.getTeamProfiles(),
            currentFrame = currentSample,
            history = historySnapshot,
            discoveredPlayers = cachedDiscoveredPlayers,
            selectedPlayerNumber = selNum,
            selectedTeamFilter = selectedTeamFilter,
            selectedPlayerClips = enrichedClips,
            telemetry = telemetry,
        )
    }

    private fun cropScaledTorso(highResBitmap: Bitmap, box: RectNorm, maxHeight: Int): Bitmap? {
        val raw = jerseyRecognizer.cropUpperTorso(highResBitmap, box) ?: return null
        if (raw.height <= maxHeight) return raw
        val scale = maxHeight.toFloat() / raw.height.toFloat()
        val tw = max(12, (raw.width * scale).toInt())
        val scaled = Bitmap.createScaledBitmap(raw, tw, maxHeight, true)
        if (scaled !== raw && !raw.isRecycled) raw.recycle()
        return scaled
    }

    private fun shouldRefreshTeamLabelsLocked(nowMs: Long): Boolean {
        if (!teamsNanoVerified) return true
        val existingInvalid = isStripedOrRefereeLabel(lastVerifiedTeamALabel) ||
            isStripedOrRefereeLabel(lastVerifiedTeamBLabel)
        return existingInvalid && teamLabelAttempts < 3 && nowMs - lastNanoTeamWallMs >= 8_000L
    }

    private fun isStripedOrRefereeLabel(label: String): Boolean {
        val low = label.lowercase()
        return low.contains("striped") || low.contains("stripe") || low.contains("referee") ||
            low.contains("official") || low.contains("zebra") || low.contains("unknown")
    }

    private fun collectTeamCropsLocked(
        highResBitmap: Bitmap,
        sample: FrameMarkSample,
        team: TeamId,
        teamProfiles: Map<TeamId, com.engtestbench.hockeynano.core.model.TeamProfile>,
        targetBuffer: MutableList<Pair<Int, Bitmap>>,
    ) {
        val centroid = teamProfiles[team]?.meanRgb
        val hasTwoTeams = teamProfiles.containsKey(TeamId.TEAM_A) && teamProfiles.containsKey(TeamId.TEAM_B)
        // Drop any buffered crop whose track later re-clustered onto a different team/referee.
        val visibleById = sample.players.associateBy { it.trackId }
        val iter = targetBuffer.iterator()
        while (iter.hasNext()) {
            val (tid, bmp) = iter.next()
            val cur = visibleById[tid]
            if (cur != null && cur.team != team) {
                if (!bmp.isRecycled) bmp.recycle()
                iter.remove()
            }
        }
        // Prefer mid-ice stable skater tracks rather than foreground-edge referee outliers.
        val candidates = sample.players
            .filter { p ->
                p.team == team &&
                    p.box.height in 0.055f..0.28f &&
                    p.box.centerX in 0.08f..0.92f &&
                    p.box.footY in 0.22f..0.88f &&
                    (!hasTwoTeams || centroid == null || rgbDist(p.chestRgb, centroid) <= 78f)
            }
            .ifEmpty {
                sample.players.filter { p -> p.team == team && p.box.height in 0.05f..0.32f }
            }
            .sortedWith(
                compareByDescending<TrackedPlayer> { trackHitCounts[it.trackId] ?: 1 }
                    .thenBy { p -> if (centroid != null) rgbDist(p.chestRgb, centroid) else 0f }
                    .thenByDescending { it.box.width * it.box.height },
            )
            .take(2)

        for (p in candidates) {
            val crop = cropScaledTorso(highResBitmap, p.box, maxHeight = 192) ?: continue
            val existingIdx = targetBuffer.indexOfFirst { it.first == p.trackId }
            if (existingIdx >= 0) {
                val old = targetBuffer.removeAt(existingIdx).second
                if (!old.isRecycled) old.recycle()
                targetBuffer.add(p.trackId to crop)
            } else {
                if (targetBuffer.size >= 3) {
                    val old = targetBuffer.removeAt(0).second
                    if (!old.isRecycled) old.recycle()
                }
                targetBuffer.add(p.trackId to crop)
            }
        }
    }

    private fun rgbDist(a: List<Int>, b: List<Int>): Float {
        if (a.size < 3 || b.size < 3) return 0f
        val dr = (a[0] - b[0]).toFloat()
        val dg = (a[1] - b[1]).toFloat()
        val db = (a[2] - b[2]).toFloat()
        return kotlin.math.sqrt(dr * dr + dg * dg + db * db)
    }

    private fun snapshotTeamCrops(): Pair<List<Bitmap>, List<Bitmap>> = synchronized(cropLock) {
        val a = pendingTeamACrops.mapNotNull { (_, bmp) ->
            if (bmp.isRecycled) null else runCatching { bmp.copy(Bitmap.Config.ARGB_8888, false) }.getOrNull()
        }
        val b = pendingTeamBCrops.mapNotNull { (_, bmp) ->
            if (bmp.isRecycled) null else runCatching { bmp.copy(Bitmap.Config.ARGB_8888, false) }.getOrNull()
        }
        a to b
    }

    private fun snapshotNanoGridItems(): List<GridCropItem> = synchronized(cropLock) {
        val copy = pendingNanoGridItems.mapNotNull { item ->
            if (item.crop.isRecycled) null
            else runCatching {
                GridCropItem(item.trackId, item.team, item.crop.copy(Bitmap.Config.ARGB_8888, false))
            }.getOrNull()
        }
        recycleGridItemsLocked(pendingNanoGridItems)
        copy
    }

    private fun recycleGridItemsLocked(list: MutableList<GridCropItem>) {
        list.forEach { if (!it.crop.isRecycled) it.crop.recycle() }
        list.clear()
    }

    private fun clearAllCropBuffers() = synchronized(cropLock) {
        pendingTeamACrops.forEach { (_, bmp) -> if (!bmp.isRecycled) bmp.recycle() }
        pendingTeamACrops.clear()
        pendingTeamBCrops.forEach { (_, bmp) -> if (!bmp.isRecycled) bmp.recycle() }
        pendingTeamBCrops.clear()
        recycleGridItemsLocked(pendingNanoGridItems)
        latestPreviewBitmap?.let { if (!it.isRecycled) it.recycle() }
        latestPreviewBitmap = null
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        sessionGen.incrementAndGet()
        statusWatcherJob.cancel()
        ocrJob?.cancel()
        nanoJob?.cancel()
        scope.cancel()
        clearAllCropBuffers()
        runCatching { detector?.close() }
        runCatching { jerseyRecognizer.close() }
        runCatching { nanoEngine.close() }
    }

    companion object {
        private const val TAG = "HockeyNanoBench"
        private const val OCR_CADENCE_MS = 400L
        private const val MAX_OCR_TRACKS_PER_STEP = 4
        private const val NANO_GRID_CADENCE_MS = 2000L
        private const val MIN_OCR_BOX_HEIGHT = 0.045f
        private const val FULL_RECOMPUTE_CADENCE_MS = 1_200L
        private const val COMPACT_HISTORY_THRESHOLD = 18_000
        private const val COMPACT_MIN_SPACING_SEC = 1.0
    }
}
