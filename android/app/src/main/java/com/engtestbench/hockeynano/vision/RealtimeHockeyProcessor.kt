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
import com.engtestbench.hockeynano.core.tracking.JerseyCropGeometry
import com.engtestbench.hockeynano.core.tracking.JerseyReadKind
import com.engtestbench.hockeynano.core.tracking.JerseyReadSource
import com.engtestbench.hockeynano.core.tracking.KitLabFeatures
import com.engtestbench.hockeynano.core.tracking.MultiTeamPlayerTracker
import com.engtestbench.hockeynano.core.tracking.OnIceFilter
import com.engtestbench.hockeynano.core.tracking.RawDetection
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
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
import kotlin.math.roundToInt

/**
 * Coordinates real-time on-device hockey tracking, ONNX STR (~220ms cadence),
 * dual-pass ML Kit jersey OCR (~400ms cadence), and single-flight Pixel 11 Pro
 * Gemini Nano multimodal VLM tasks (white-balanced team kit labeling, multi-view
 * 2x2 jersey sheets, 30s shift summaries, and eng-test-bench assistant) while
 * publishing [PipelineState].
 */
class RealtimeHockeyProcessor(private val context: Context) : Closeable {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val detector: YoloOnnxDetector? = YoloOnnxDetector.load(context, threads = 4)
    private val strRecognizer = JerseyStrOnnxRecognizer(context)
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
    private val strAttempts = ConcurrentHashMap<Int, Int>()
    private val ocrAttempts = ConcurrentHashMap<Int, Int>()
    private val nanoGridAttempts = ConcurrentHashMap<Int, Int>()
    private val trackHitCounts = ConcurrentHashMap<Int, Int>()
    private val lastStrByTrack = ConcurrentHashMap<Int, TrackStrScore>()
    val lastOrientationByTrack = ConcurrentHashMap<Int, Pair<Float, Double>>()
    private val strongStrTwoDigitByTrack = ConcurrentHashMap<Int, Pair<String, Float>>()
    private val trackOfficialHints = ConcurrentHashMap<Int, OfficialCueEma>()
    private val trackStrSignatures = ConcurrentHashMap<Int, TrackCropSignature>()
    private val trackOcrSignatures = ConcurrentHashMap<Int, TrackCropSignature>()

    private data class TrackCropSignature(
        val lumaGrid4x4: FloatArray,
        val anchorTimestampSec: Double,
    )

    private data class StrCropTask(
        val trackId: Int,
        val boxHeight: Float,
        val crop: Bitmap,
    )

    private data class OfficialCueEma(
        val emaPants: Float,
        val emaChroma: Float,
    )

    private data class TrackStrScore(
        val legibilityProb: Float,
        val backProb: Float,
        val sharpness: Float,
        val timestampSec: Double,
    )

    private data class TrackViewSnapshot(
        val timestampSec: Double,
        val score: Float,
        val bitmap: Bitmap,
        val orientationDeg: Float = 0f,
    ) {
        fun recycle() {
            if (!bitmap.isRecycled) bitmap.recycle()
        }
    }

    private val ocrReadsTotal = AtomicInteger(0)
    private val nanoCallsTotal = AtomicInteger(0)
    private val framesAnalyzedCount = AtomicInteger(0)

    private var fpsEma: Float = 0f
    private var lastFrameWallMs: Long = 0L
    private var lastDetectorMs: Float = 0f
    private var lastNanoLatencyMs: Long = 0L
    private var lastNanoMessage: String = ""
    private var lastLoggedDiscoveredSummary: String = ""

    private var strJob: Job? = null
    private var ocrJob: Job? = null
    private var nanoJob: Job? = null
    private var lastStrWallMs: Long = 0L
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
    private val trackViewPool = LinkedHashMap<Int, MutableList<TrackViewSnapshot>>()
    private val trackLastSeenSec = HashMap<Int, Double>()
    private val pendingGridTrackQueue = ArrayList<Pair<Int, TeamId>>(4)
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
        strJob?.cancel()
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
            this.lastLoggedDiscoveredSummary = ""

            historySamples.clear()
            currentSample = null
            lastTrackerTimestampSec = Double.NaN
            maxHistoryTimeSec = 0.0

            clipSummariesById.clear()
            exportedFastPaths.clear()
            exportedMarkedPaths.clear()
            strAttempts.clear()
            ocrAttempts.clear()
            nanoGridAttempts.clear()
            trackHitCounts.clear()
            lastStrByTrack.clear()
            lastOrientationByTrack.clear()
            strongStrTwoDigitByTrack.clear()
            trackOfficialHints.clear()
            trackStrSignatures.clear()
            trackOcrSignatures.clear()

            ocrReadsTotal.set(0)
            nanoCallsTotal.set(0)
            framesAnalyzedCount.set(0)
            fpsEma = 0f
            lastFrameWallMs = 0L
            lastDetectorMs = 0f
            lastNanoLatencyMs = 0L
            lastNanoMessage = ""
            lastStrWallMs = 0L
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
                val iceWb = detEngine.currentIceWhiteBalance()

                val aspectRatio = highResBitmap.width.toFloat() / highResBitmap.height.coerceAtLeast(1).toFloat()
                val rawDetections = yoloResult.detections
                    .filter { OnIceFilter.isOnIce(it.box, aspectRatio, useTiling, it.iceContext) }
                    .map {
                        RawDetection(
                            box = it.box,
                            chestRgb = it.chestRgb,
                            stripeScore = it.stripeScore,
                            kitSummaryLab7 = it.kitSummaryLab7,
                            lowerBellyLuma = it.lowerBellyLuma,
                            pantsDarkness = it.pantsDarkness,
                            armbandScore = it.armbandScore,
                            torsoUpperLabL = it.torsoUpperLabL,
                        )
                    }

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

                    framesAnalyzedCount.incrementAndGet()
                    emitStateLocked(forceFullRecompute = shouldFullRecompute)
                    sampleForOverlay to reachedNow
                }

                if (frameResult == null || sessionGen.get() != gen) return@withContext
                val (frameSample, budgetReachedNow) = frameResult
                val detByBox = rawDetections.associateBy { it.box }
                frameSample.players.forEach { p ->
                    trackHitCounts[p.trackId] = (trackHitCounts[p.trackId] ?: 0) + 1
                    val matchedDet = detByBox[p.box]
                    if (matchedDet != null && matchedDet.pantsDarkness.isFinite()) {
                        val lab = KitLabFeatures.rgbToLab(matchedDet.chestRgb)
                        val chroma = KitLabFeatures.labChroma(lab[1], lab[2])
                        val prev = trackOfficialHints[p.trackId]
                        trackOfficialHints[p.trackId] = if (prev == null) {
                            OfficialCueEma(matchedDet.pantsDarkness, chroma)
                        } else {
                            OfficialCueEma(
                                emaPants = 0.8f * prev.emaPants + 0.2f * matchedDet.pantsDarkness,
                                emaChroma = 0.8f * prev.emaChroma + 0.2f * chroma,
                            )
                        }
                    }
                }

                Log.i(
                    TAG,
                    "FRAME_STEP frame=#${framesAnalyzedCount.get()} t=${"%.2f".format(Locale.US, timestampSec)}s detMs=${"%.1f".format(Locale.US, yoloResult.detectorMs)} " +
                        "tiled=${yoloResult.tiledPassUsed} det=${yoloResult.detections.size} onIce=${rawDetections.size} " +
                        "live=${"%.2f".format(Locale.US, yoloResult.livePlayProb)} ocr=${ocrReadsTotal.get()} nano=${nanoEngine.status.value}",
                )

                if (!budgetReachedNow) {
                    scheduleAsyncStrIfDue(highResBitmap, frameSample, iceWb, timestampSec, nowMs, gen)
                    scheduleAsyncOcrIfDue(highResBitmap, frameSample, iceWb, timestampSec, nowMs, gen)
                    captureNanoCrops(highResBitmap, frameSample, iceWb, timestampSec, nowMs)
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

    fun recomputeStateNow() {
        synchronized(stateLock) {
            emitStateLocked(forceFullRecompute = true)
        }
    }

    suspend fun awaitProcessingDrain(maxWaitMs: Long = 1_800L) {
        val deadline = SystemClock.elapsedRealtime() + maxWaitMs.coerceAtLeast(100L)
        while (SystemClock.elapsedRealtime() < deadline) {
            val busy = yoloBusy.get() ||
                strJob?.isActive == true ||
                ocrJob?.isActive == true ||
                nanoJob?.isActive == true
            if (!busy) break
            delay(50L)
        }
        recomputeStateNow()
    }

    private fun isLikelyOfficialTrack(player: TrackedPlayer): Boolean =
        player.team == TeamId.REFEREE || isLikelyOfficialTrackId(player.trackId)

    private fun isLikelyOfficialTrackId(trackId: Int): Boolean {
        val hint = trackOfficialHints[trackId]
        if (hint != null && hint.emaPants >= 0.60f && hint.emaChroma < 15f) return true
        val curTeam = synchronized(stateLock) {
            currentSample?.players?.firstOrNull { it.trackId == trackId }?.team
        }
        return curTeam == TeamId.REFEREE
    }

    private fun selectPrioritySkaters(
        sample: FrameMarkSample,
        limit: Int,
        attempts: Map<Int, Int>,
        unconfirmedOnly: Boolean,
        minHeight: Float = MIN_OCR_BOX_HEIGHT,
    ): List<TrackedPlayer> = sample.players
        .filter {
            !isLikelyOfficialTrack(it) &&
                it.box.height >= minHeight &&
                it.box.width >= 0.015f &&
                (!unconfirmedOnly || it.jerseyNumber == null || !it.nanoVerified)
        }
        .sortedWith(
            compareBy<TrackedPlayer>(
                { if (it.nanoVerified) 1 else 0 },
                { p ->
                    val area = p.box.width * p.box.height
                    val str = lastStrByTrack[p.trackId]
                    val sharpness = str?.sharpness ?: 100f
                    val backProb = str?.backProb ?: 0.5f
                    val legProb = str?.legibilityProb ?: 0.5f
                    val att = attempts[p.trackId] ?: 0
                    val viewScore = (0.3f + 0.7f * legProb) * (0.4f + 0.6f * backProb) * sharpness * p.box.height
                    -((viewScore * area) / (1f + 0.5f * att))
                },
            ),
        )
        .take(limit)

    private fun computeCropLumaGrid4x4(crop: Bitmap): FloatArray {
        val w = crop.width.coerceAtLeast(1)
        val h = crop.height.coerceAtLeast(1)
        val pixels = IntArray(w * h)
        crop.getPixels(pixels, 0, w, 0, 0, w, h)
        val grid = FloatArray(16)
        for (cy in 0 until 4) {
            val y0 = (cy * h) / 4
            val y1 = (((cy + 1) * h) / 4).coerceAtLeast(y0 + 1).coerceAtMost(h)
            for (cx in 0 until 4) {
                val x0 = (cx * w) / 4
                val x1 = (((cx + 1) * w) / 4).coerceAtLeast(x0 + 1).coerceAtMost(w)
                var sum = 0.0
                var count = 0
                for (y in y0 until y1) {
                    val rowOff = y * w
                    for (x in x0 until x1) {
                        val p = pixels[rowOff + x]
                        sum += 0.299 * ((p ushr 16) and 0xFF) +
                            0.587 * ((p ushr 8) and 0xFF) +
                            0.114 * (p and 0xFF)
                        count++
                    }
                }
                grid[cy * 4 + cx] = if (count > 0) (sum / count).toFloat() else 0f
            }
        }
        return grid
    }

    private fun resolveEffectiveReadTimestamp(
        trackId: Int,
        crop: Bitmap,
        timestampSec: Double,
        signatures: ConcurrentHashMap<Int, TrackCropSignature>,
    ): Double {
        if (crop.isRecycled || crop.width <= 0 || crop.height <= 0) return timestampSec
        val grid = computeCropLumaGrid4x4(crop)
        val prev = signatures[trackId]
        if (prev != null && abs(timestampSec - prev.anchorTimestampSec) <= UNCHANGED_CROP_ANCHOR_WINDOW_SEC) {
            var absDiffSum = 0f
            for (i in 0 until 16) {
                absDiffSum += abs(grid[i] - prev.lumaGrid4x4[i])
            }
            if (absDiffSum / 16f < UNCHANGED_CROP_LUMA_DIFF_MAX) {
                return prev.anchorTimestampSec
            }
        }
        signatures[trackId] = TrackCropSignature(grid, timestampSec)
        return timestampSec
    }

    private fun scheduleAsyncStrIfDue(
        highResBitmap: Bitmap,
        sample: FrameMarkSample,
        iceWb: FloatArray,
        timestampSec: Double,
        nowMs: Long,
        gen: Long,
    ) {
        if (!strRecognizer.isAvailable || nowMs - lastStrWallMs < STR_CADENCE_MS || strJob?.isActive == true) return
        val priorityPlayers = selectPrioritySkaters(sample, MAX_STR_TRACKS_PER_STEP, strAttempts, unconfirmedOnly = false)
        if (priorityPlayers.isEmpty()) return

        val tasks = ArrayList<StrCropTask>(priorityPlayers.size)
        for (player in priorityPlayers) {
            val crop = strRecognizer.cropUpperTorso(highResBitmap, player.box, iceWb) ?: continue
            strAttempts[player.trackId] = (strAttempts[player.trackId] ?: 0) + 1
            tasks.add(StrCropTask(player.trackId, player.box.height, crop))
        }
        if (tasks.isEmpty()) return

        val wbCopy = iceWb.copyOf()
        lastStrWallMs = nowMs
        strJob = scope.launch(Dispatchers.Default) {
            try {
                var anyStrongRead = false
                val strLogParts = ArrayList<String>(tasks.size)
                for ((trackId, boxHeight, crop) in tasks) {
                    try {
                        if (!isActive || sessionGen.get() != gen || isLikelyOfficialTrackId(trackId)) continue
                        val res = strRecognizer.recognizeFromTorsoCrop(crop, wbCopy) ?: continue
                        val effectiveTs = resolveEffectiveReadTimestamp(trackId, crop, timestampSec, trackStrSignatures)
                        lastStrByTrack[trackId] = TrackStrScore(
                            legibilityProb = res.legibilityProb,
                            backProb = res.backProb,
                            sharpness = res.sharpness,
                            timestampSec = timestampSec,
                        )
                        val minOriSharpness = if (strRecognizer.modelInputSize >= 128) {
                            MIN_ORIENTATION_SHARPNESS_V5
                        } else {
                            MIN_ORIENTATION_SHARPNESS_V4
                        }
                        if (res.sharpness >= minOriSharpness &&
                            (res.legibilityProb >= 0.18f || res.backProb >= 0.35f)
                        ) {
                            lastOrientationByTrack[trackId] = res.orientationDeg to timestampSec
                        }
                        val rawCandidates = if (res.legibilityProb >= JerseyStrOnnxRecognizer.MIN_CANDIDATE_LEGIBILITY_PROB &&
                            res.backProb >= JerseyStrOnnxRecognizer.MIN_CANDIDATE_BACK_PROB
                        ) {
                            res.topCandidates
                        } else {
                            emptyList()
                        }
                        val rawTopProb = rawCandidates.firstOrNull()?.second ?: 0f
                        val distMinSharpness = if (strRecognizer.modelInputSize >= 128) {
                            DISTANT_STR_MIN_SHARPNESS_V5
                        } else {
                            DISTANT_STR_MIN_SHARPNESS_V4
                        }
                        val candidates = if (boxHeight < DISTANT_STR_FULL_BOX_HEIGHT &&
                            (rawTopProb < DISTANT_STR_MIN_TOP_PROB ||
                                res.legibilityProb < DISTANT_STR_MIN_LEGIBILITY_PROB ||
                                res.sharpness < distMinSharpness)
                        ) {
                            emptyList()
                        } else {
                            rawCandidates
                        }
                        val partialLeft = if (boxHeight >= DISTANT_STR_PARTIAL_BOX_HEIGHT) res.partialLeftDigit else null
                        val partialRight = if (boxHeight >= DISTANT_STR_PARTIAL_BOX_HEIGHT) res.partialRightDigit else null

                        if (candidates.isNotEmpty() || partialLeft != null || partialRight != null) {
                            candidates.firstOrNull()?.let { (num, p) ->
                                if (num.length == 2 &&
                                    p >= 0.60f &&
                                    res.legibilityProb >= JerseyStrOnnxRecognizer.MIN_CANDIDATE_LEGIBILITY_PROB &&
                                    res.backProb >= JerseyStrOnnxRecognizer.MIN_CANDIDATE_BACK_PROB
                                ) {
                                    val prev = strongStrTwoDigitByTrack[trackId]
                                    if (prev == null || p > prev.second) {
                                        strongStrTwoDigitByTrack[trackId] = num to p
                                    }
                                }
                            }
                            val topProb = candidates.firstOrNull()?.second ?: 0f
                            val qualityWeight = if (candidates.isNotEmpty()) {
                                if (res.legibilityProb >= 0.94f && res.backProb >= 0.94f && topProb >= 0.86f) {
                                    1.25f
                                } else {
                                    ((0.35f + 0.65f * res.legibilityProb) * (0.50f + 0.50f * res.backProb)).coerceIn(0.05f, 1.25f)
                                }
                            } else {
                                (0.40f + 0.60f * res.backProb).coerceIn(0.10f, 1.0f)
                            }
                            if (topProb >= 0.30f) {
                                ocrReadsTotal.incrementAndGet()
                                anyStrongRead = true
                            }
                            val topTag = candidates.firstOrNull()?.let { (num, p) ->
                                "#$num:${"%.2f".format(Locale.US, p)}"
                            } ?: listOfNotNull(
                                partialLeft?.let { "${it.first}?:${"%.2f".format(Locale.US, it.second)}" },
                                partialRight?.let { "?${it.first}:${"%.2f".format(Locale.US, it.second)}" },
                            ).joinToString("+")
                            val angTag = if (abs(res.orientationDeg) >= 8f) {
                                val p2Tag = if (abs(res.orientationDeg) >= JerseyCropGeometry.DEROTATE_MIN_ANGLE_DEG) {
                                    if (res.usedDerotationPass2) ",p2=1" else ",p2=0"
                                } else {
                                    ""
                                }
                                ",ang=${res.orientationDeg.roundToInt()}$p2Tag"
                            } else {
                                ""
                            }
                            strLogParts.add(
                                "T$trackId:$topTag(leg=${"%.2f".format(Locale.US, res.legibilityProb)},bk=${"%.2f".format(Locale.US, res.backProb)}$angTag)",
                            )
                            synchronized(stateLock) {
                                if (sessionGen.get() == gen) {
                                    tracker.recordJerseyDistribution(
                                        trackId = trackId,
                                        timestampSec = effectiveTs,
                                        candidates = candidates,
                                        qualityWeight = qualityWeight,
                                        source = JerseyReadSource.ONNX_STR,
                                        partialLeftDigit = partialLeft,
                                        partialRightDigit = partialRight,
                                    )
                                }
                            }
                        }
                    } finally {
                        if (!crop.isRecycled) crop.recycle()
                    }
                }
                if (strLogParts.isNotEmpty() && sessionGen.get() == gen) {
                    Log.i(TAG, "ONNX_STR t=${"%.2f".format(Locale.US, timestampSec)}s reads=\"${strLogParts.joinToString(",")}\"")
                }
                if (sessionGen.get() == gen && anyStrongRead) {
                    synchronized(stateLock) {
                        if (sessionGen.get() == gen) {
                            emitStateLocked(forceFullRecompute = true)
                        }
                    }
                }
            } catch (t: Throwable) {
                tasks.forEach { if (!it.crop.isRecycled) it.crop.recycle() }
                Log.w(TAG, "Async STR step failed: ${t.message}")
            }
        }
    }

    private fun scheduleAsyncOcrIfDue(
        highResBitmap: Bitmap,
        sample: FrameMarkSample,
        iceWb: FloatArray,
        timestampSec: Double,
        nowMs: Long,
        gen: Long,
    ) {
        if (nowMs - lastOcrWallMs < OCR_CADENCE_MS || ocrJob?.isActive == true) return
        val priorityPlayers = selectPrioritySkaters(sample, MAX_OCR_TRACKS_PER_STEP, ocrAttempts, unconfirmedOnly = false)
        if (priorityPlayers.isEmpty()) return

        val tasks = ArrayList<Pair<Int, Bitmap>>(priorityPlayers.size)
        for (player in priorityPlayers) {
            val crop = jerseyRecognizer.cropUpperTorso(highResBitmap, player.box, iceWb) ?: continue
            ocrAttempts[player.trackId] = (ocrAttempts[player.trackId] ?: 0) + 1
            tasks.add(player.trackId to crop)
        }
        if (tasks.isEmpty()) return

        val wbCopy = iceWb.copyOf()
        lastOcrWallMs = nowMs
        ocrJob = scope.launch(Dispatchers.Default) {
            try {
                var anyRead = false
                val ocrLogParts = ArrayList<String>(tasks.size)
                for ((trackId, crop) in tasks) {
                    try {
                        if (!isActive || sessionGen.get() != gen || isLikelyOfficialTrackId(trackId)) continue
                        val res = jerseyRecognizer.recognizeFromTorsoCrop(crop, wbCopy)
                        if (res != null && sessionGen.get() == gen) {
                            val effectiveTs = resolveEffectiveReadTimestamp(trackId, crop, timestampSec, trackOcrSignatures)
                            ocrReadsTotal.incrementAndGet()
                            anyRead = true
                            val kind = when {
                                res.jerseyNumber.length >= 2 -> JerseyReadKind.FULL
                                res.xRel < -0.10f -> JerseyReadKind.LEFT_PARTIAL
                                res.xRel > 0.10f -> JerseyReadKind.RIGHT_PARTIAL
                                else -> JerseyReadKind.SINGLE
                            }
                            ocrLogParts.add(
                                "T$trackId:#${res.jerseyNumber}(${"%.2f".format(Locale.US, res.confidence)},${kind.name},x=${"%.2f".format(Locale.US, res.xRel)})",
                            )
                            synchronized(stateLock) {
                                if (sessionGen.get() == gen) {
                                    tracker.recordJerseyRead(
                                        trackId = trackId,
                                        jerseyNumber = res.jerseyNumber,
                                        confidence = res.confidence,
                                        timestampSec = effectiveTs,
                                        kind = kind,
                                        qualityWeight = 1f,
                                        source = JerseyReadSource.MLKIT_OCR,
                                    )
                                }
                            }
                        }
                    } finally {
                        if (!crop.isRecycled) crop.recycle()
                    }
                }
                if (ocrLogParts.isNotEmpty() && sessionGen.get() == gen) {
                    Log.i(TAG, "MLKIT_OCR t=${"%.2f".format(Locale.US, timestampSec)}s reads=\"${ocrLogParts.joinToString(",")}\"")
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

    private fun captureNanoCrops(
        highResBitmap: Bitmap,
        sample: FrameMarkSample,
        iceWb: FloatArray,
        timestampSec: Double,
        nowMs: Long,
    ) {
        val teamProfiles = synchronized(stateLock) { tracker.getTeamProfiles() }
        val needsTeamCrops = synchronized(stateLock) { shouldCollectTeamCropsLocked() }
        val confirmedIds = sample.players.filter { it.nanoVerified }.map { it.trackId }.toSet()

        // Accumulate crisp upper-skater context (x: -12%..112%, y: -6%..86%) + orientation metadata;
        // upright derotation & unsharp sharpening happen inside buildJerseyGridComposite only for selected grid slots.
        val unconfirmed = selectPrioritySkaters(
            sample = sample,
            limit = MAX_VIEW_CANDIDATES_PER_FRAME,
            attempts = nanoGridAttempts,
            unconfirmedOnly = true,
            minHeight = MIN_NANO_BOX_HEIGHT,
        )
        val capturedViews = ArrayList<Pair<TrackedPlayer, TrackViewSnapshot>>(unconfirmed.size)
        for (p in unconfirmed) {
            val crop = cropWhiteBalancedRegion(
                highResBitmap = highResBitmap,
                box = p.box,
                iceWb = iceWb,
                xStartFrac = -0.12f,
                xEndFrac = 1.12f,
                yStartFrac = -0.06f,
                yEndFrac = 0.86f,
                maxHeight = 260,
                contrastAlpha = 1.15f,
            ) ?: continue
            val orientDeg = lastOrientationByTrack[p.trackId]
                ?.takeIf { abs(timestampSec - it.second) <= 1.0 }
                ?.first ?: 0f
            val str = lastStrByTrack[p.trackId]?.takeIf { abs(timestampSec - it.timestampSec) <= 0.6 }
            val strLeg = str?.legibilityProb ?: 0.5f
            val strBack = str?.backProb ?: 0.5f
            val sharpness = str?.sharpness ?: JerseyStrOnnxRecognizer.computeLaplacianSharpness(crop)
            val score = (0.3f + 0.7f * strLeg) * (0.5f + 0.5f * strBack) * sharpness * p.box.height
            capturedViews.add(p to TrackViewSnapshot(timestampSec, score, crop, orientDeg))
        }

        synchronized(cropLock) {
            if (needsTeamCrops) {
                collectTeamCropsLocked(highResBitmap, sample, iceWb, TeamId.TEAM_A, teamProfiles, pendingTeamACrops)
                collectTeamCropsLocked(highResBitmap, sample, iceWb, TeamId.TEAM_B, teamProfiles, pendingTeamBCrops)
            }

            // Prune stale or already-confirmed tracks from the multi-view pool.
            val poolIter = trackViewPool.entries.iterator()
            while (poolIter.hasNext()) {
                val (tid, snaps) = poolIter.next()
                val lastSeen = trackLastSeenSec[tid] ?: timestampSec
                if (tid in confirmedIds || abs(timestampSec - lastSeen) > VIEW_POOL_TTL_SEC) {
                    snaps.forEach { it.recycle() }
                    poolIter.remove()
                    trackLastSeenSec.remove(tid)
                }
            }

            for ((p, snap) in capturedViews) {
                trackLastSeenSec[p.trackId] = timestampSec
                val list = trackViewPool.getOrPut(p.trackId) { ArrayList(MAX_VIEWS_PER_TRACK) }
                val nearbyIdx = list.indexOfFirst { abs(it.timestampSec - timestampSec) < 0.22 }
                if (nearbyIdx >= 0) {
                    if (snap.score > list[nearbyIdx].score) {
                        list[nearbyIdx].recycle()
                        list[nearbyIdx] = snap
                    } else {
                        snap.recycle()
                    }
                } else {
                    list.add(snap)
                    if (list.size > MAX_VIEWS_PER_TRACK) {
                        var minIdx = 0
                        for (i in 1 until list.size) {
                            if (list[i].score < list[minIdx].score) minIdx = i
                        }
                        list.removeAt(minIdx).recycle()
                    }
                }
            }

            while (trackViewPool.size > MAX_VIEW_POOL_TRACKS) {
                val oldestKey = trackLastSeenSec.minByOrNull { it.value }?.key ?: break
                trackViewPool.remove(oldestKey)?.forEach { it.recycle() }
                trackLastSeenSec.remove(oldestKey)
            }

            // Refresh top 4 unconfirmed priority tracks ranked strictly by bestViewScore * boxArea
            // (moderated lightly by attempt count) without demoting tentative ONNX_STR reads.
            pendingGridTrackQueue.clear()
            val rankedQueue = unconfirmed
                .mapNotNull { p ->
                    val bestViewScore = trackViewPool[p.trackId]
                        ?.filter { !it.bitmap.isRecycled }
                        ?.maxOfOrNull { it.score } ?: return@mapNotNull null
                    val boxArea = p.box.width * p.box.height
                    val att = nanoGridAttempts[p.trackId] ?: 0
                    val priority = (bestViewScore * boxArea) / (1f + 0.5f * att)
                    Triple(p.trackId, p.team, priority)
                }
                .sortedByDescending { it.third }
                .take(4)
            for ((tid, team, _) in rankedQueue) {
                pendingGridTrackQueue.add(tid to team)
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
                                    nanoCallsTotal.incrementAndGet()
                                    val distinctLabels = !labels.teamALabel.trim().equals(
                                        labels.teamBLabel.trim(),
                                        ignoreCase = true,
                                    )
                                    val validNonStriped = distinctLabels &&
                                        !isStripedOrRefereeLabel(labels.teamALabel) &&
                                        !isStripedOrRefereeLabel(labels.teamBLabel)
                                    val hadValidLabels = teamsNanoVerified &&
                                        !lastVerifiedTeamALabel.trim().equals(
                                            lastVerifiedTeamBLabel.trim(),
                                            ignoreCase = true,
                                        ) &&
                                        !isStripedOrRefereeLabel(lastVerifiedTeamALabel) &&
                                        !isStripedOrRefereeLabel(lastVerifiedTeamBLabel)
                                    if (distinctLabels && (!hadValidLabels || validNonStriped)) {
                                        teamsNanoVerified = true
                                        lastVerifiedTeamALabel = labels.teamALabel
                                        lastVerifiedTeamBLabel = labels.teamBLabel
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

            // 3. Multimodal 2x2 multi-view jersey number grid every 1600ms while unconfirmed tracks exist.
            if (nowMs - lastNanoGridWallMs >= NANO_GRID_CADENCE_MS) {
                val (gridSnapshot, trackTimestamps) = snapshotNanoGridItems()
                if (gridSnapshot.isNotEmpty()) {
                    didWork = true
                    lastNanoGridWallMs = nowMs
                    gridSnapshot.forEach {
                        nanoGridAttempts[it.trackId] = (nanoGridAttempts[it.trackId] ?: 0) + 1
                    }
                    val (reads, latencyMs) = try {
                        nanoEngine.readMultiViewJerseyGridMultimodal(gridSnapshot)
                    } finally {
                        gridSnapshot.forEach { item ->
                            item.views.forEach { if (!it.isRecycled) it.recycle() }
                        }
                    }
                    if (sessionGen.get() == gen) {
                        synchronized(stateLock) {
                            if (sessionGen.get() == gen) {
                                nanoCallsTotal.incrementAndGet()
                                if (latencyMs > 0L) lastNanoLatencyMs = latencyMs
                                if (reads.isNotEmpty()) {
                                    reads.forEach { r ->
                                        if (isLikelyOfficialTrackId(r.trackId)) return@forEach
                                        val strong2 = strongStrTwoDigitByTrack[r.trackId]?.first
                                        val cleanNumRaw = r.jerseyNumber.trim('?')
                                        val (cleanNum, kind) = when {
                                            r.isPartialRight -> cleanNumRaw to JerseyReadKind.RIGHT_PARTIAL
                                            r.isPartialLeft -> cleanNumRaw to JerseyReadKind.LEFT_PARTIAL
                                            cleanNumRaw.length == 1 && strong2 != null && strong2.length == 2 && cleanNumRaw[0] in strong2 -> {
                                                // Convert partial single-digit Nano occlusion onto strong 2-digit STR partner.
                                                val k = if (strong2[1] == cleanNumRaw[0]) {
                                                    JerseyReadKind.RIGHT_PARTIAL
                                                } else {
                                                    JerseyReadKind.LEFT_PARTIAL
                                                }
                                                cleanNumRaw to k
                                            }
                                            cleanNumRaw.length == 1 -> cleanNumRaw to JerseyReadKind.SINGLE
                                            else -> cleanNumRaw to JerseyReadKind.FULL
                                        }
                                        val readTs = trackTimestamps[r.trackId] ?: maxHistoryTimeSec
                                        tracker.recordJerseyRead(
                                            trackId = r.trackId,
                                            jerseyNumber = cleanNum,
                                            confidence = r.confidence,
                                            timestampSec = readTs,
                                            kind = kind,
                                            qualityWeight = 1f,
                                            source = JerseyReadSource.GEMINI_NANO,
                                        )
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
            val discoveredSummary = cachedDiscoveredPlayers.joinToString(",") {
                val teamTag = when (it.team) {
                    TeamId.TEAM_A -> "A"
                    TeamId.TEAM_B -> "B"
                    TeamId.REFEREE -> "R"
                    TeamId.UNKNOWN -> "?"
                }
                "$teamTag:#${it.jerseyNumber}(${"%.1f".format(Locale.US, it.totalSeenSec)}s${if (it.nanoVerified) "*" else ""})"
            }
            if (discoveredSummary != lastLoggedDiscoveredSummary) {
                lastLoggedDiscoveredSummary = discoveredSummary
                Log.i(TAG, "DISCOVERED_PLAYERS count=${cachedDiscoveredPlayers.size} players=\"$discoveredSummary\"")
            }
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
            framesAnalyzed = framesAnalyzedCount.get(),
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

    private fun cropWhiteBalancedRegion(
        highResBitmap: Bitmap,
        box: RectNorm,
        iceWb: FloatArray,
        xStartFrac: Float,
        xEndFrac: Float,
        yStartFrac: Float,
        yEndFrac: Float,
        maxHeight: Int,
        contrastAlpha: Float,
    ): Bitmap? {
        if (highResBitmap.isRecycled) return null
        val w = highResBitmap.width
        val h = highResBitmap.height
        if (w <= 1 || h <= 1) return null

        val bw = box.width
        val bh = box.height
        val x0 = ((box.x1 + xStartFrac * bw) * w).toInt().coerceIn(0, w - 1)
        val x1 = ((box.x1 + xEndFrac * bw) * w).toInt().coerceIn(x0 + 1, w)
        val y0 = ((box.y1 + yStartFrac * bh) * h).toInt().coerceIn(0, h - 1)
        val y1 = ((box.y1 + yEndFrac * bh) * h).toInt().coerceIn(y0 + 1, h)
        val cw = x1 - x0
        val ch = y1 - y0
        if (cw < 10 || ch < 14) return null

        val rawCrop = Bitmap.createBitmap(highResBitmap, x0, y0, cw, ch)
        val scaled = if (ch > maxHeight) {
            val scale = maxHeight.toFloat() / ch.toFloat()
            val tw = max(12, (cw * scale).roundToInt())
            val s = Bitmap.createScaledBitmap(rawCrop, tw, maxHeight, true)
            if (s !== rawCrop && !rawCrop.isRecycled) rawCrop.recycle()
            s
        } else {
            rawCrop
        }

        val wbR = iceWb.getOrElse(0) { 1f }.coerceIn(0.6f, 1.6f)
        val wbG = iceWb.getOrElse(1) { 1f }.coerceIn(0.6f, 1.6f)
        val wbB = iceWb.getOrElse(2) { 1f }.coerceIn(0.6f, 1.6f)
        val sw = scaled.width
        val sh = scaled.height
        val px = IntArray(sw * sh)
        scaled.getPixels(px, 0, sw, 0, 0, sw, sh)
        if (!scaled.isRecycled) scaled.recycle()

        for (i in px.indices) {
            val p = px[i]
            val rWb = ((p ushr 16) and 0xFF) * wbR
            val gWb = ((p ushr 8) and 0xFF) * wbG
            val bWb = (p and 0xFF) * wbB
            val r = ((rWb - 128f) * contrastAlpha + 128f).roundToInt().coerceIn(0, 255)
            val g = ((gWb - 128f) * contrastAlpha + 128f).roundToInt().coerceIn(0, 255)
            val b = ((bWb - 128f) * contrastAlpha + 128f).roundToInt().coerceIn(0, 255)
            px[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(px, sw, sh, Bitmap.Config.ARGB_8888)
    }

    private fun shouldCollectTeamCropsLocked(): Boolean {
        if (teamLabelAttempts >= MAX_TEAM_LABEL_ATTEMPTS) return false
        if (!teamsNanoVerified) return true
        return lastVerifiedTeamALabel.trim().equals(lastVerifiedTeamBLabel.trim(), ignoreCase = true) ||
            isStripedOrRefereeLabel(lastVerifiedTeamALabel) ||
            isStripedOrRefereeLabel(lastVerifiedTeamBLabel)
    }

    private fun shouldRefreshTeamLabelsLocked(nowMs: Long): Boolean {
        if (!shouldCollectTeamCropsLocked()) return false
        return teamLabelAttempts == 0 || (nowMs - lastNanoTeamWallMs >= TEAM_LABEL_RETRY_MS)
    }

    private fun isStripedOrRefereeLabel(label: String): Boolean {
        val low = label.lowercase()
        return low.contains("striped") || low.contains("stripe") || low.contains("referee") ||
            low.contains("official") || low.contains("zebra") || low.contains("unknown")
    }

    private fun collectTeamCropsLocked(
        highResBitmap: Bitmap,
        sample: FrameMarkSample,
        iceWb: FloatArray,
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
            // White-balanced full upper-skater context (y: -4%..80%) prevents ice-tint kit misnaming & policy blocks.
            val crop = cropWhiteBalancedRegion(
                highResBitmap = highResBitmap,
                box = p.box,
                iceWb = iceWb,
                xStartFrac = -0.10f,
                xEndFrac = 1.10f,
                yStartFrac = -0.04f,
                yEndFrac = 0.80f,
                maxHeight = 200,
                contrastAlpha = 1.12f,
            ) ?: continue
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

    private fun snapshotNanoGridItems(): Pair<List<MultiViewGridCropItem>, Map<Int, Double>> = synchronized(cropLock) {
        val items = ArrayList<MultiViewGridCropItem>(4)
        val timestamps = HashMap<Int, Double>(4)
        for ((trackId, team) in pendingGridTrackQueue) {
            val snaps = trackViewPool[trackId]?.filter { !it.bitmap.isRecycled }.orEmpty()
            if (snaps.isEmpty()) continue
            val best = snaps.maxByOrNull { it.score } ?: continue
            val copy = runCatching { best.bitmap.copy(Bitmap.Config.ARGB_8888, false) }.getOrNull() ?: continue
            items.add(MultiViewGridCropItem(trackId, team, listOf(copy), best.orientationDeg))
            timestamps[trackId] = best.timestampSec
        }
        pendingGridTrackQueue.clear()
        items to timestamps
    }

    private fun clearAllCropBuffers() = synchronized(cropLock) {
        pendingTeamACrops.forEach { (_, bmp) -> if (!bmp.isRecycled) bmp.recycle() }
        pendingTeamACrops.clear()
        pendingTeamBCrops.forEach { (_, bmp) -> if (!bmp.isRecycled) bmp.recycle() }
        pendingTeamBCrops.clear()
        trackViewPool.values.forEach { list ->
            list.forEach { it.recycle() }
        }
        trackViewPool.clear()
        trackLastSeenSec.clear()
        pendingGridTrackQueue.clear()
        latestPreviewBitmap?.let { if (!it.isRecycled) it.recycle() }
        latestPreviewBitmap = null
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        sessionGen.incrementAndGet()
        statusWatcherJob.cancel()
        strJob?.cancel()
        ocrJob?.cancel()
        nanoJob?.cancel()
        scope.cancel()
        clearAllCropBuffers()
        runCatching { detector?.close() }
        runCatching { strRecognizer.close() }
        runCatching { jerseyRecognizer.close() }
        runCatching { nanoEngine.close() }
    }

    companion object {
        private const val TAG = "HockeyNanoBench"
        private const val STR_CADENCE_MS = 220L
        private const val MAX_STR_TRACKS_PER_STEP = 6
        private const val OCR_CADENCE_MS = 400L
        private const val MAX_OCR_TRACKS_PER_STEP = 4
        private const val NANO_GRID_CADENCE_MS = 1600L
        private const val MIN_OCR_BOX_HEIGHT = 0.085f
        private const val MIN_NANO_BOX_HEIGHT = 0.052f
        private const val DISTANT_STR_FULL_BOX_HEIGHT = 0.115f
        private const val DISTANT_STR_PARTIAL_BOX_HEIGHT = 0.105f
        private const val DISTANT_STR_MIN_TOP_PROB = 0.58f
        private const val DISTANT_STR_MIN_LEGIBILITY_PROB = 0.75f
        private const val DISTANT_STR_MIN_SHARPNESS_V5 = 12.0f
        private const val DISTANT_STR_MIN_SHARPNESS_V4 = 28.0f
        private const val MIN_ORIENTATION_SHARPNESS_V5 = 5.5f
        private const val MIN_ORIENTATION_SHARPNESS_V4 = 18.0f
        private const val UNCHANGED_CROP_ANCHOR_WINDOW_SEC = 2.5
        private const val UNCHANGED_CROP_LUMA_DIFF_MAX = 3.5f
        private const val MAX_VIEW_CANDIDATES_PER_FRAME = 10
        private const val MAX_VIEWS_PER_TRACK = 5
        private const val MAX_VIEW_POOL_TRACKS = 20
        private const val VIEW_POOL_TTL_SEC = 8.0
        private const val FULL_RECOMPUTE_CADENCE_MS = 1_200L
        private const val MAX_TEAM_LABEL_ATTEMPTS = 4
        private const val TEAM_LABEL_RETRY_MS = 5_000L
        private const val COMPACT_HISTORY_THRESHOLD = 18_000
        private const val COMPACT_MIN_SPACING_SEC = 1.0
    }
}
