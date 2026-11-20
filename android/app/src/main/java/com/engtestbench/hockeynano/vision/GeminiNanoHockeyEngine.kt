package com.engtestbench.hockeynano.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.util.Log
import com.engtestbench.hockeynano.core.model.TeamId
import com.engtestbench.hockeynano.core.tracking.JerseyCropGeometry
import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.common.GenAiException
import com.google.mlkit.genai.prompt.Generation
import com.google.mlkit.genai.prompt.GenerativeModel
import com.google.mlkit.genai.prompt.ImagePart
import com.google.mlkit.genai.prompt.TextPart
import com.google.mlkit.genai.prompt.generateContentRequest
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.takeWhile
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.Closeable
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class TeamKitLabels(
    val teamALabel: String,
    val teamBLabel: String,
    val nanoVerified: Boolean,
    val latencyMs: Long,
    val rawOutput: String = "",
)

data class GridCropItem(
    val trackId: Int,
    val team: TeamId,
    val crop: Bitmap,
)

data class MultiViewGridCropItem(
    val trackId: Int,
    val team: TeamId,
    val views: List<Bitmap>,
    val orientationDeg: Float = 0f,
)

data class NanoJerseyRead(
    val trackId: Int,
    val jerseyNumber: String,
    val confidence: Float,
    val isPartialLeft: Boolean = false,
    val isPartialRight: Boolean = false,
)

data class NanoTextResponse(
    val text: String,
    val fromNano: Boolean,
    val latencyMs: Long,
)

/**
 * Pixel 11 Pro on-device Gemini Nano multimodal VLM engine via ML Kit GenAI Prompt API (`AICore`).
 * Handles lifecycle/download cleanly and executes single-flight multimodal team labeling,
 * 2x2 jersey number grids, 30s shift summaries, and eng-test-bench Q&A.
 */
class GeminiNanoHockeyEngine(@Suppress("unused") private val context: Context) : Closeable {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val inferenceMutex = Mutex()

    private val _status = MutableStateFlow("initializing")
    val status: StateFlow<String> = _status.asStateFlow()

    @Volatile
    private var model: GenerativeModel? = null

    @Volatile
    private var downloadStarted = false

    @Volatile
    private var multimodalBroken = false

    suspend fun refreshStatus(): String = withContext(Dispatchers.IO) {
        val prev = _status.value
        val next = try {
            val m = model ?: Generation.getClient().also { model = it }
            when (m.checkStatus()) {
                FeatureStatus.AVAILABLE -> "ready"
                FeatureStatus.DOWNLOADABLE, FeatureStatus.DOWNLOADING -> {
                    startDownloadIfNeeded(m)
                    "downloading"
                }
                else -> "unavailable"
            }
        } catch (ce: CancellationException) {
            throw ce
        } catch (e: Throwable) {
            Log.w(TAG, "NANO_STATUS check failed: ${e.javaClass.simpleName}: ${e.message}")
            "unavailable"
        }
        _status.value = next
        if (next != prev || prev == "initializing") {
            Log.i(TAG, "NANO_STATUS status=$next")
        }
        next
    }

    private fun startDownloadIfNeeded(m: GenerativeModel) {
        if (downloadStarted) return
        downloadStarted = true
        scope.launch {
            try {
                m.download()
                    .takeWhile { it !is DownloadStatus.DownloadCompleted && it !is DownloadStatus.DownloadFailed }
                    .collect { }
                val after = if (m.checkStatus() == FeatureStatus.AVAILABLE) "ready" else "unavailable"
                _status.value = after
                Log.i(TAG, "NANO_STATUS status=$after after=download")
            } catch (ce: CancellationException) {
                throw ce
            } catch (e: Throwable) {
                _status.value = "unavailable"
                Log.w(TAG, "NANO_STATUS download_failed=${e.message}")
            }
        }
    }

    suspend fun identifyTeamsMultimodal(
        teamACrops: List<Bitmap>,
        teamBCrops: List<Bitmap>,
        fallbackTeamARgb: List<Int>,
        fallbackTeamBRgb: List<Int>,
    ): TeamKitLabels = withContext(Dispatchers.IO) {
        val fallbackA = describeRgbKit(fallbackTeamARgb, "Dark")
        val fallbackB = describeRgbKit(fallbackTeamBRgb, "Light")
        if (teamACrops.isEmpty() || teamBCrops.isEmpty() || multimodalBroken) {
            return@withContext TeamKitLabels(fallbackA, fallbackB, nanoVerified = false, latencyMs = 0L)
        }
        if (refreshStatus() != "ready") {
            return@withContext TeamKitLabels(fallbackA, fallbackB, nanoVerified = false, latencyMs = 0L)
        }

        val composite = buildTeamComparisonComposite(teamACrops.take(3), teamBCrops.take(3))
        val prompt =
            "This 768x384 image shows ice hockey players split into two sides: left half labeled A " +
                "(Team A crops) and right half labeled B (Team B crops). " +
                "Left half TEAM A wears the DARK base jersey and right half TEAM B wears the WHITE/LIGHT base jersey " +
                "— ignore the white ice background and name each side's base + trim jersey color in 1-3 words " +
                "(for example: Dark Navy/Red | White/Blue). " +
                "Reply ONLY in this exact format:\nTEAM_A: <kit> | TEAM_B: <kit>"

        val startMs = System.currentTimeMillis()
        try {
            val raw = inferenceMutex.withLock {
                val m = model ?: Generation.getClient().also { model = it }
                val req = generateContentRequest(ImagePart(composite), TextPart(prompt)) {
                    temperature = 0.1f
                    topK = 8
                    maxOutputTokens = 48
                }
                m.generateContent(req).candidates.firstOrNull()?.text?.trim().orEmpty()
            }
            val latencyMs = max(1L, System.currentTimeMillis() - startMs)
            val parsed = parseTeamKitLabels(raw)
            if (parsed != null) {
                Log.i(
                    TAG,
                    "NANO_TEAM_LABEL latencyMs=${latencyMs}ms teamA=\"${parsed.first}\" teamB=\"${parsed.second}\" raw=\"${raw.replace('\n', ' ').take(120)}\"",
                )
                TeamKitLabels(parsed.first, parsed.second, nanoVerified = true, latencyMs = latencyMs, rawOutput = raw)
            } else {
                Log.i(
                    TAG,
                    "NANO_TEAM_LABEL latencyMs=${latencyMs}ms fallback=true teamA=\"$fallbackA\" teamB=\"$fallbackB\" raw=\"${raw.replace('\n', ' ').take(120)}\"",
                )
                TeamKitLabels(fallbackA, fallbackB, nanoVerified = false, latencyMs = latencyMs, rawOutput = raw)
            }
        } catch (ce: CancellationException) {
            throw ce
        } catch (e: Throwable) {
            val latencyMs = max(1L, System.currentTimeMillis() - startMs)
            if (isMultimodalUnsupported(e)) multimodalBroken = true
            Log.w(TAG, "NANO_TEAM_LABEL error latencyMs=${latencyMs}ms err=${e.javaClass.simpleName}:${e.message}")
            TeamKitLabels(fallbackA, fallbackB, nanoVerified = false, latencyMs = latencyMs)
        } finally {
            if (!composite.isRecycled) composite.recycle()
        }
    }

    suspend fun readJerseyGridMultimodal(items: List<GridCropItem>): Pair<List<NanoJerseyRead>, Long> {
        val multi = items.map { MultiViewGridCropItem(it.trackId, it.team, listOf(it.crop)) }
        return readMultiViewJerseyGridMultimodal(multi)
    }

    suspend fun readMultiViewJerseyGridMultimodal(
        items: List<MultiViewGridCropItem>,
    ): Pair<List<NanoJerseyRead>, Long> = withContext(Dispatchers.IO) {
        val batch = items.take(4)
        if (batch.isEmpty() || multimodalBroken) return@withContext emptyList<NanoJerseyRead>() to 0L
        if (refreshStatus() != "ready") return@withContext emptyList<NanoJerseyRead>() to 0L
        runJerseyGridBatch(batch)
    }

    private suspend fun runJerseyGridBatch(
        batch: List<MultiViewGridCropItem>,
        allowEchoRetry: Boolean = true,
    ): Pair<List<NanoJerseyRead>, Long> {
        if (batch.isEmpty() || multimodalBroken) return emptyList<NanoJerseyRead>() to 0L
        val grid = buildJerseyGridComposite(batch)
        val gridDesc = when (batch.size) {
            1 -> "384x384 1x1"
            2 -> "768x384 1x2"
            else -> "768x768 2x2"
        }
        val slotLetters = SLOT_LETTERS.take(batch.size)
        val slotList = slotLetters.joinToString(", ") { "[$it]" }
        val slotTemplate = slotLetters.joinToString(" ") { "$it:#<num or NONE>" }
        val prompt =
            "This $gridDesc ice hockey analytics sheet shows ${batch.size} skater track cells labeled $slotList. " +
                "Each cell shows one upright ice hockey player. " +
                "Read any clearly visible 1 or 2 digit jersey number (1-99) printed on the jersey in each cell. " +
                "If only one digit of a two-digit number is legible, use ? for the hidden digit (for example #9? or #?7). " +
                "If a number is blurry, tiny, or unreadable, output NONE. " +
                "Reply ONLY in one line:\n$slotTemplate"

        val startMs = System.currentTimeMillis()
        try {
            val raw = inferenceMutex.withLock {
                val m = model ?: Generation.getClient().also { model = it }
                val req = generateContentRequest(ImagePart(grid), TextPart(prompt)) {
                    temperature = 0.0f
                    topK = 1
                    maxOutputTokens = 48
                }
                m.generateContent(req).candidates.firstOrNull()?.text?.trim().orEmpty()
            }
            val latencyMs = max(1L, System.currentTimeMillis() - startMs)
            val parsed = parseJerseyGridResponse(raw, batch)
            if (isCrossSlotEcho(parsed, batch)) {
                Log.w(
                    TAG,
                    "NANO_JERSEY_READ echo_rejected latencyMs=${latencyMs}ms n=${batch.size} raw=\"${raw.replace('\n', ' ').take(120)}\"",
                )
                if (!grid.isRecycled) grid.recycle()
                if (allowEchoRetry && batch.isNotEmpty()) {
                    val (retryReads, retryMs) = runJerseyGridBatch(listOf(batch.first()), allowEchoRetry = false)
                    return retryReads to (latencyMs + retryMs)
                }
                return emptyList<NanoJerseyRead>() to latencyMs
            }
            val summary = parsed.joinToString(",") { "T${it.trackId}:#${it.jerseyNumber}" }.ifEmpty { "none" }
            Log.i(
                TAG,
                "NANO_JERSEY_READ latencyMs=${latencyMs}ms n=${batch.size} count=${parsed.size} reads=\"$summary\" raw=\"${raw.replace('\n', ' ').take(120)}\"",
            )
            return parsed to latencyMs
        } catch (ce: CancellationException) {
            throw ce
        } catch (e: Throwable) {
            val latencyMs = max(1L, System.currentTimeMillis() - startMs)
            if (isPolicyCheckBlocked(e)) {
                Log.w(TAG, "NANO_JERSEY_READ policy_blocked batch=${batch.size} latencyMs=${latencyMs}ms err=${e.message}")
                if (!grid.isRecycled) grid.recycle()
                if (batch.size > 2) {
                    val (leftReads, leftMs) = runJerseyGridBatch(batch.take(2), allowEchoRetry)
                    val (rightReads, rightMs) = runJerseyGridBatch(batch.drop(2), allowEchoRetry)
                    return (leftReads + rightReads) to (latencyMs + leftMs + rightMs)
                }
                if (batch.size == 2) {
                    val (firstReads, firstMs) = runJerseyGridBatch(listOf(batch[0]), allowEchoRetry = false)
                    val (secondReads, secondMs) = runJerseyGridBatch(listOf(batch[1]), allowEchoRetry = false)
                    return (firstReads + secondReads) to (latencyMs + firstMs + secondMs)
                }
                return emptyList<NanoJerseyRead>() to latencyMs
            }
            if (isMultimodalUnsupported(e)) multimodalBroken = true
            Log.w(TAG, "NANO_JERSEY_READ error latencyMs=${latencyMs}ms err=${e.javaClass.simpleName}:${e.message}")
            return emptyList<NanoJerseyRead>() to latencyMs
        } finally {
            if (!grid.isRecycled) grid.recycle()
        }
    }

    suspend fun summarizeClip(
        playerNumber: String,
        teamName: String,
        startSec: Double,
        endSec: Double,
        visibilitySec: Double,
        avgLiveProb: Float,
        avgSpeedNorm: Float,
        keyframeComposite: Bitmap? = null,
    ): NanoTextResponse = withContext(Dispatchers.IO) {
        val livePct = (avgLiveProb * 100f).toInt().coerceIn(0, 100)
        val paceLabel = when {
            avgSpeedNorm >= 0.09f -> "high-tempo"
            avgSpeedNorm >= 0.04f -> "steady"
            else -> "positional"
        }
        val fallback = "#$playerNumber ($teamName): ${"%.1f".format(visibilitySec)}s tracked in " +
            "${formatClock(startSec)}-${formatClock(endSec)} ($livePct% live play, $paceLabel skating)."

        if (refreshStatus() != "ready") {
            return@withContext NanoTextResponse(fallback, fromNano = false, latencyMs = 0L)
        }

        val prompt =
            "You are an ice hockey video scout on Pixel 11 Pro. Summarize this 30s shift in 1-2 concise sentences (<= 32 words, plain text, no bullet list).\n" +
                "Player: #$playerNumber ($teamName)\n" +
                "Window: ${formatClock(startSec)} to ${formatClock(endSec)} (${"%.1f".format(visibilitySec)}s tracked on ice)\n" +
                "Live play: $livePct% | Skating pace: $paceLabel"

        val startMs = System.currentTimeMillis()
        val imageForPrompt = if (!multimodalBroken && keyframeComposite != null && !keyframeComposite.isRecycled) {
            keyframeComposite
        } else {
            null
        }
        try {
            val raw = inferenceMutex.withLock {
                val m = model ?: Generation.getClient().also { model = it }
                val req = if (imageForPrompt != null) {
                    generateContentRequest(ImagePart(imageForPrompt), TextPart(prompt)) {
                        temperature = 0.25f
                        topK = 16
                        maxOutputTokens = 96
                    }
                } else {
                    generateContentRequest(TextPart(prompt)) {
                        temperature = 0.25f
                        topK = 16
                        maxOutputTokens = 96
                    }
                }
                m.generateContent(req).candidates.firstOrNull()?.text?.trim().orEmpty()
            }
            val latencyMs = max(1L, System.currentTimeMillis() - startMs)
            val cleaned = raw.replace("**", "").replace('\n', ' ').trim()
            if (cleaned.isNotEmpty()) {
                Log.i(
                    TAG,
                    "NANO_CLIP_SUMMARY player=#$playerNumber window=${startSec.toInt()}-${endSec.toInt()}s latencyMs=${latencyMs}ms summary=\"${cleaned.take(140)}\"",
                )
                NanoTextResponse(cleaned, fromNano = true, latencyMs = latencyMs)
            } else {
                NanoTextResponse(fallback, fromNano = false, latencyMs = latencyMs)
            }
        } catch (ce: CancellationException) {
            throw ce
        } catch (e: Throwable) {
            val latencyMs = max(1L, System.currentTimeMillis() - startMs)
            if (imageForPrompt != null && (isMultimodalUnsupported(e) || isPolicyCheckBlocked(e))) {
                if (isMultimodalUnsupported(e)) multimodalBroken = true
                return@withContext summarizeClip(
                    playerNumber, teamName, startSec, endSec, visibilitySec, avgLiveProb, avgSpeedNorm, null,
                )
            }
            Log.w(TAG, "NANO_CLIP_SUMMARY error latencyMs=${latencyMs}ms err=${e.javaClass.simpleName}:${e.message}")
            NanoTextResponse(fallback, fromNano = false, latencyMs = latencyMs)
        }
    }

    suspend fun askBenchAssistant(question: String, reportContext: String): NanoTextResponse =
        withContext(Dispatchers.IO) {
            val statusNow = refreshStatus()
            if (statusNow != "ready") {
                val note = "Gemini Nano is currently $statusNow. Based on the local eng-test-bench report: " +
                    reportContext.lines().take(4).joinToString(" ")
                return@withContext NanoTextResponse(note, fromNano = false, latencyMs = 0L)
            }

            val prompt =
                "You are an engineering benchmark assistant running locally via Gemini Nano on Pixel 11 Pro.\n" +
                    "Answer the question concisely (2-4 sentences) grounded strictly in this benchmark context:\n" +
                    reportContext.take(2400) + "\n\nQuestion: $question"

            val startMs = System.currentTimeMillis()
            try {
                val raw = inferenceMutex.withLock {
                    val m = model ?: Generation.getClient().also { model = it }
                    val req = generateContentRequest(TextPart(prompt)) {
                        temperature = 0.25f
                        topK = 16
                        maxOutputTokens = 256
                    }
                    m.generateContent(req).candidates.firstOrNull()?.text?.trim().orEmpty()
                }
                val latencyMs = max(1L, System.currentTimeMillis() - startMs)
                val cleaned = raw.ifEmpty { "No response returned by Gemini Nano." }
                Log.i(TAG, "NANO_BENCH_ASSISTANT latencyMs=${latencyMs}ms q=\"${question.take(60)}\"")
                NanoTextResponse(cleaned, fromNano = raw.isNotEmpty(), latencyMs = latencyMs)
            } catch (ce: CancellationException) {
                throw ce
            } catch (e: Throwable) {
                val latencyMs = max(1L, System.currentTimeMillis() - startMs)
                Log.w(TAG, "NANO_BENCH_ASSISTANT error latencyMs=${latencyMs}ms err=${e.message}")
                NanoTextResponse(
                    "Gemini Nano error (${e.javaClass.simpleName}). Summary context: ${reportContext.take(240)}",
                    fromNano = false,
                    latencyMs = latencyMs,
                )
            }
        }

    private fun buildTeamComparisonComposite(teamACrops: List<Bitmap>, teamBCrops: List<Bitmap>): Bitmap {
        val out = Bitmap.createBitmap(768, 384, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.rgb(22, 27, 34))

        val bmpPaint = Paint(Paint.FILTER_BITMAP_FLAG)
        val dividerPaint = Paint().apply { color = Color.rgb(180, 190, 200); strokeWidth = 4f }
        canvas.drawLine(384f, 40f, 384f, 384f, dividerPaint)

        val bannerBg = Paint().apply { color = Color.rgb(14, 19, 27) }
        canvas.drawRect(0f, 0f, 768f, 40f, bannerBg)
        val bannerText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(200, 214, 229)
            textSize = 18f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        canvas.drawText("ICE HOCKEY PLAYER • TEAM KIT COMPARISON", 16f, 26f, bannerText)

        drawSideHeader(canvas, "TEAM A", 12f, 48f, Color.rgb(0, 229, 255))
        drawSideHeader(canvas, "TEAM B", 396f, 48f, Color.rgb(255, 109, 0))

        drawCropsInHalf(canvas, teamACrops, halfX = 0, Color.rgb(0, 229, 255), bmpPaint)
        drawCropsInHalf(canvas, teamBCrops, halfX = 384, Color.rgb(255, 109, 0), bmpPaint)
        return out
    }

    private fun drawSideHeader(canvas: Canvas, label: String, x: Float, y: Float, accentColor: Int) {
        val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(220, 12, 16, 22) }
        val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = accentColor
            style = Paint.Style.STROKE
            strokeWidth = 2.5f
        }
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = accentColor
            textSize = 22f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        val rect = RectF(x, y, x + 116f, y + 34f)
        canvas.drawRoundRect(rect, 8f, 8f, bgPaint)
        canvas.drawRoundRect(rect, 8f, 8f, borderPaint)
        canvas.drawText(label, x + 14f, y + 25f, textPaint)
    }

    private fun drawCropsInHalf(
        canvas: Canvas,
        crops: List<Bitmap>,
        halfX: Int,
        accentColor: Int,
        bmpPaint: Paint,
    ) {
        if (crops.isEmpty()) return
        val slotW = 112f
        val slotH = 282f
        val gap = 10f
        val startX = halfX + 14f
        val topY = 90f
        val framePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = accentColor
            style = Paint.Style.STROKE
            strokeWidth = 2f
        }
        crops.take(3).forEachIndexed { idx, crop ->
            if (crop.isRecycled) return@forEachIndexed
            val sx = startX + idx * (slotW + gap)
            val box = RectF(sx, topY, sx + slotW, topY + slotH)
            drawAspectFitCrop(canvas, crop, box, bmpPaint)
            canvas.drawRect(box, framePaint)
        }
    }

    private fun buildJerseyGridComposite(items: List<MultiViewGridCropItem>): Bitmap {
        val cols = if (items.size == 1) 1 else 2
        val rows = if (items.size <= 2) 1 else 2
        val out = Bitmap.createBitmap(cols * 384, rows * 384, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.rgb(18, 24, 34))

        val bmpPaint = Paint(Paint.FILTER_BITMAP_FLAG)
        val borderPaint = Paint().apply {
            color = Color.rgb(80, 98, 120)
            style = Paint.Style.STROKE
            strokeWidth = 3f
        }
        val headerBg = Paint().apply { color = Color.rgb(13, 18, 27) }
        val pillBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(235, 10, 12, 16) }
        val pillBorder = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(255, 214, 0)
            style = Paint.Style.STROKE
            strokeWidth = 2.5f
        }
        val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(255, 214, 0)
            textSize = 24f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        val cardTagPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(184, 204, 228)
            textSize = 15f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }

        items.take(4).forEachIndexed { idx, item ->
            val col = idx % 2
            val row = idx / 2
            val cellX = col * 384f
            val cellY = row * 384f
            canvas.drawRect(cellX, cellY, cellX + 384f, cellY + 384f, borderPaint)
            canvas.drawRect(cellX + 2f, cellY + 2f, cellX + 382f, cellY + 42f, headerBg)

            val view = item.views.firstOrNull { !it.isRecycled }
            if (view != null) {
                val target = RectF(cellX + 8f, cellY + 44f, cellX + 376f, cellY + 380f)
                val derotated = if (abs(item.orientationDeg) >= JerseyCropGeometry.DEROTATE_MIN_ANGLE_DEG) {
                    TorsoCropHelper.derotateBitmapUpright(
                        src = view,
                        ccwOrientationDeg = item.orientationDeg,
                        expandBounds = true,
                    )
                } else {
                    null
                }
                val working = derotated ?: view
                try {
                    drawCrispAspectFitCrop(canvas, working, target, bmpPaint)
                } finally {
                    if (derotated != null && !derotated.isRecycled) derotated.recycle()
                }
            }

            val badgeRect = RectF(cellX + 10f, cellY + 6f, cellX + 66f, cellY + 38f)
            canvas.drawRoundRect(badgeRect, 7f, 7f, pillBg)
            canvas.drawRoundRect(badgeRect, 7f, 7f, pillBorder)
            val slotLetter = SLOT_LETTERS[idx]
            canvas.drawText("[$slotLetter]", cellX + 16f, cellY + 31f, labelPaint)
            canvas.drawText("ICE HOCKEY PLAYER", cellX + 76f, cellY + 28f, cardTagPaint)
        }
        return out
    }

    private fun drawCrispAspectFitCrop(canvas: Canvas, crop: Bitmap, bounds: RectF, paint: Paint) {
        val cw = crop.width.coerceAtLeast(1)
        val ch = crop.height.coerceAtLeast(1)
        val scale = min(bounds.width() / cw.toFloat(), bounds.height() / ch.toFloat())
        val dw = (cw * scale).roundToInt().coerceAtLeast(1)
        val dh = (ch * scale).roundToInt().coerceAtLeast(1)
        val left = bounds.left + (bounds.width() - dw) / 2f
        val top = bounds.top + (bounds.height() - dh) / 2f

        if (dw > cw || dh > ch) {
            val srcPixels = IntArray(cw * ch)
            crop.getPixels(srcPixels, 0, cw, 0, 0, cw, ch)
            val sharpPixels = JerseyCropGeometry.upscaleBicubicAndUnsharp(
                srcPixels = srcPixels,
                srcW = cw,
                srcH = ch,
                dstW = dw,
                dstH = dh,
                wbR = 1f,
                wbG = 1f,
                wbB = 1f,
                contrastAlpha = 1f,
                unsharpAmount = 0.55f,
            )
            if (sharpPixels.size >= dw * dh) {
                val sharpBmp = Bitmap.createBitmap(sharpPixels, dw, dh, Bitmap.Config.ARGB_8888)
                try {
                    canvas.drawBitmap(sharpBmp, left, top, paint)
                } finally {
                    if (!sharpBmp.isRecycled) sharpBmp.recycle()
                }
                return
            }
        }
        canvas.drawBitmap(crop, Rect(0, 0, cw, ch), RectF(left, top, left + dw, top + dh), paint)
    }

    private fun drawAspectFitCrop(canvas: Canvas, crop: Bitmap, bounds: RectF, paint: Paint) {
        val cw = crop.width.coerceAtLeast(1).toFloat()
        val ch = crop.height.coerceAtLeast(1).toFloat()
        val scale = min(bounds.width() / cw, bounds.height() / ch)
        val dw = cw * scale
        val dh = ch * scale
        val left = bounds.left + (bounds.width() - dw) / 2f
        val top = bounds.top + (bounds.height() - dh) / 2f
        canvas.drawBitmap(crop, Rect(0, 0, crop.width, crop.height), RectF(left, top, left + dw, top + dh), paint)
    }

    private fun parseTeamKitLabels(raw: String): Pair<String, String>? {
        val clean = raw.replace("**", "").trim()
        val matchA = TEAM_A_REGEX.find(clean)?.groupValues?.get(1)?.trim()
        val matchB = TEAM_B_REGEX.find(clean)?.groupValues?.get(1)?.trim()
        if (!matchA.isNullOrBlank() && !matchB.isNullOrBlank()) {
            val a = sanitizeKitLabel(matchA)
            val b = sanitizeKitLabel(matchB)
            if (a.isNotEmpty() && b.isNotEmpty()) return a to b
        }
        return null
    }

    private fun sanitizeKitLabel(s: String): String =
        s.trim().trim('"', '\'', '.', ',', ':', ';').take(28).trim()

    private fun parseJerseyGridResponse(raw: String, items: List<MultiViewGridCropItem>): List<NanoJerseyRead> {
        val clean = raw.replace("**", "").replace('\n', ' ').trim()
        val results = ArrayList<NanoJerseyRead>(items.size)
        val seenTracks = HashSet<Int>()
        for (match in GRID_READ_REGEX.findAll(clean)) {
            val slotChar = match.groupValues[1].uppercase().firstOrNull() ?: continue
            val slotIdx = slotChar - 'A'
            val token = match.groupValues[2].uppercase()
            if (slotIdx !in items.indices) continue
            val item = items[slotIdx]
            if (token.startsWith("?") && token.length == 2 && token[1] in '0'..'9') {
                if (seenTracks.add(item.trackId)) {
                    results.add(
                        NanoJerseyRead(
                            trackId = item.trackId,
                            jerseyNumber = token,
                            confidence = 0.80f,
                            isPartialRight = true,
                        ),
                    )
                }
                continue
            }
            if (token.endsWith("?") && token.length == 2 && token[0] in '1'..'9') {
                if (seenTracks.add(item.trackId)) {
                    results.add(
                        NanoJerseyRead(
                            trackId = item.trackId,
                            jerseyNumber = token,
                            confidence = 0.80f,
                            isPartialLeft = true,
                        ),
                    )
                }
                continue
            }
            val value = token.toIntOrNull() ?: continue
            if (value !in 1..99) continue
            if (seenTracks.add(item.trackId)) {
                results.add(NanoJerseyRead(item.trackId, value.toString(), confidence = 0.88f))
            }
        }
        return results
    }

    private fun isCrossSlotEcho(
        reads: List<NanoJerseyRead>,
        batch: List<MultiViewGridCropItem>,
    ): Boolean {
        if (batch.size < 2 || reads.size < 2) return false
        // Reject any multi-slot batch where Nano echoes the same number >= 2 times, even across teams.
        val seenNumbers = HashSet<String>(reads.size)
        for (r in reads) {
            if (!seenNumbers.add(r.jerseyNumber)) return true
        }
        return false
    }

    private fun isPolicyCheckBlocked(t: Throwable): Boolean {
        if (t is GenAiException && t.errorCode == 4) return true
        val msg = t.message?.lowercase().orEmpty()
        return msg.contains("errorcode 4") ||
            msg.contains("error code 4") ||
            msg.contains("policy check") ||
            msg.contains("policy")
    }

    private fun isMultimodalUnsupported(t: Throwable): Boolean {
        if (isPolicyCheckBlocked(t)) return false
        if (t is GenAiException && t.errorCode == GenAiException.ErrorCode.NOT_SUPPORTED) return true
        val msg = t.message?.lowercase().orEmpty()
        return msg.contains("unsupported") || (msg.contains("image") && msg.contains("not support"))
    }

    private fun describeRgbKit(rgb: List<Int>, fallback: String): String {
        if (rgb.size < 3) return fallback
        val r = rgb[0]
        val g = rgb[1]
        val b = rgb[2]
        val lum = 0.299f * r + 0.587f * g + 0.114f * b
        val maxC = max(r, max(g, b))
        val minC = min(r, min(g, b))
        val sat = if (maxC > 0) (maxC - minC).toFloat() / maxC else 0f
        return when {
            lum >= 175f && sat < 0.25f -> "White Kit"
            lum <= 68f && sat < 0.35f -> "Black/Dark"
            b > r + 25 && b >= g -> if (lum < 110f) "Navy Blue" else "Royal Blue"
            r > b + 35 && r > g + 25 -> "Red Kit"
            g > r + 25 && g > b + 20 -> "Green Kit"
            r > 170 && g > 140 && b < 90 -> "Gold/Yellow"
            lum < 120f -> "Dark Kit"
            else -> "Light Kit"
        }
    }

    private fun formatClock(sec: Double): String {
        val total = sec.toInt().coerceAtLeast(0)
        return "%d:%02d".format(total / 60, total % 60)
    }

    override fun close() {
        runCatching { model?.close() }
        scope.cancel()
    }

    companion object {
        private const val TAG = "HockeyNanoBench"
        private val SLOT_LETTERS = listOf('A', 'B', 'C', 'D')
        private val TEAM_A_REGEX = Regex("TEAM_A\\s*[:=-]\\s*([^|\\n]+)", RegexOption.IGNORE_CASE)
        private val TEAM_B_REGEX = Regex("TEAM_B\\s*[:=-]\\s*([^|\\n]+)", RegexOption.IGNORE_CASE)
        private val GRID_READ_REGEX = Regex(
            "(?:^|\\s)(?:Slot\\s*)?\\[?([A-D])]?\\s*[:=-]\\s*(?:#\\s*)?(\\?\\d|\\d\\?|\\d{1,2}|NONE|N/A|NA|UNKNOWN)(?=\\s|$|[,;.|])",
            RegexOption.IGNORE_CASE,
        )
    }
}
