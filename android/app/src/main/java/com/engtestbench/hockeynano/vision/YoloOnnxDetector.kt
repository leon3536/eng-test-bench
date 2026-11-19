package com.engtestbench.hockeynano.vision

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import com.engtestbench.hockeynano.core.model.RectNorm
import com.engtestbench.hockeynano.core.tracking.OnIceFilter
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cbrt
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

@Serializable
data class LiveHeadConfig(
    val window: Int = 16,
    val fps: Float = 2.0f,
    val features: List<String> = emptyList(),
    val norm_prior: Int = 8,
    val threshold: Float = 0.57243f,
    val det_w: Int = 640,
    val det_h: Int = 360,
    val img_w: Int = 640,
    val img_h: Int = 384,
    val min_conf: Float = 0.3f,
    val grey_w: Int = 160,
    val grey_h: Int = 90,
    val coarse: Int = 10,
    val match_heights: Float = 3.0f,
    val moving_heights: Float = 0.3f,
)

data class YoloDetection(
    val box: RectNorm,
    val confidence: Float,
    val chestRgb: List<Int>,
    val stripeScore: Float = 0f,
    val kitSummaryLab7: FloatArray = FloatArray(7),
    val lowerBellyLuma: Float = Float.NaN,
    val pantsDarkness: Float = Float.NaN,
    val armbandScore: Float = 0f,
    val torsoUpperLabL: Float = Float.NaN,
    val iceContext: OnIceFilter.IceContext? = null,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is YoloDetection) return false
        return box == other.box &&
            confidence == other.confidence &&
            chestRgb == other.chestRgb &&
            stripeScore == other.stripeScore &&
            kitSummaryLab7.contentEquals(other.kitSummaryLab7) &&
            lowerBellyLuma.toBits() == other.lowerBellyLuma.toBits() &&
            pantsDarkness.toBits() == other.pantsDarkness.toBits() &&
            armbandScore == other.armbandScore &&
            torsoUpperLabL.toBits() == other.torsoUpperLabL.toBits() &&
            iceContext == other.iceContext
    }

    override fun hashCode(): Int {
        var result = box.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + chestRgb.hashCode()
        result = 31 * result + stripeScore.hashCode()
        result = 31 * result + kitSummaryLab7.contentHashCode()
        result = 31 * result + lowerBellyLuma.toBits()
        result = 31 * result + pantsDarkness.toBits()
        result = 31 * result + armbandScore.hashCode()
        result = 31 * result + torsoUpperLabL.toBits()
        result = 31 * result + (iceContext?.hashCode() ?: 0)
        return result
    }
}

data class YoloFrameResult(
    val detections: List<YoloDetection>,
    val livePlayProb: Float,
    val detectorMs: Float,
    val tiledPassUsed: Boolean,
)

/**
 * On-device YOLOv8n person detector + 2-fps causal live-play motion head (`coach_person_detector.onnx` &
 * `coach_live_head.onnx`). Runs a full-frame 640x360 letterboxed pass plus, when [panoramicTiling] is enabled
 * on wide rinks (aspectRatio >= 1.9), two overlapping far-ice high-res crops fused via class-agnostic NMS (IoU > 0.48).
 */
class YoloOnnxDetector private constructor(
    context: Context,
    val config: LiveHeadConfig,
    threads: Int,
) : Closeable {

    private val env = OrtEnvironment.getEnvironment()
    private val detectorOptions = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        runCatching { addConfigEntry("session.intra_op.allow_spinning", "0") }
        runCatching { addConfigEntry("session.inter_op.allow_spinning", "0") }
    }
    private val xnnpackEnabled = runCatching {
        detectorOptions.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
        true
    }.getOrDefault(false)

    private val liveHeadOptions = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(min(2, threads))
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        runCatching { addConfigEntry("session.intra_op.allow_spinning", "0") }
    }

    private val detectorSession = env.createSession(
        context.assets.open("coach_person_detector.onnx").use { it.readBytes() },
        detectorOptions,
    )
    private val liveHeadSession = env.createSession(
        context.assets.open("coach_live_head.onnx").use { it.readBytes() },
        liveHeadOptions,
    )

    init {
        Log.i("HockeyNanoBench", "YOLO_INIT threads=$threads xnnpack=$xnnpackEnabled")
    }

    private val detectLock = Any()
    private val detW = config.det_w
    private val detH = config.det_h
    private val imgW = config.img_w
    private val imgH = config.img_h
    private val detPadY = (imgH - detH) / 2

    private val detBitmap = Bitmap.createBitmap(detW, detH, Bitmap.Config.ARGB_8888)
    private val detCanvas = Canvas(detBitmap)
    private val tileBitmap = Bitmap.createBitmap(imgW, imgH, Bitmap.Config.ARGB_8888)
    private val tileCanvas = Canvas(tileBitmap)
    private val featW = CANONICAL_FEAT_W
    private val featH = CANONICAL_FEAT_H
    private val featBitmap = Bitmap.createBitmap(featW, featH, Bitmap.Config.ARGB_8888)
    private val featCanvas = Canvas(featBitmap)
    private val filterPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    private val detPixels = IntArray(detW * detH)
    private val tilePixels = IntArray(imgW * imgH)
    private val inputTensorBuf = FloatArray(3 * imgW * imgH)
    private val directInputBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(4 * 3 * imgW * imgH)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
    private val wbLock = Any()
    private var boxPixelBuf = IntArray(8192)
    private var chestR = IntArray(4096)
    private var chestG = IntArray(4096)
    private var chestB = IntArray(4096)
    private var borderL = FloatArray(2048)
    private var borderA = FloatArray(2048)
    private var borderB = FloatArray(2048)
    private var torsoL = FloatArray(4096)
    private var torsoA = FloatArray(4096)
    private var torsoB = FloatArray(4096)
    private var torsoKeep = BooleanArray(4096)
    private var torsoAssign = IntArray(4096)
    private var bellyLumaBuf = FloatArray(2048)
    private var upperLabLBuf = FloatArray(4096)
    private var lastExtractedUpperLabL: Float = Float.NaN
    private val iceStatsBuf = IntArray(2)
    private val tmpIceLab = FloatArray(3)
    private val dftLuma = FloatArray(DFT_W * DFT_H)
    private val dftLumaSorted = FloatArray(DFT_W * DFT_H)
    private val dftWinRow = FloatArray(DFT_W)
    private val dftBandPower = FloatArray(DFT_H)
    private val dftLowPower = FloatArray(DFT_H)
    private val dftAcEnergy = FloatArray(DFT_H)
    private val dftRowAmp = FloatArray(DFT_H)
    private val dftRowAmpSorted = FloatArray(DFT_H)
    private val dftGroupScores = FloatArray(STRIPE_ROW_GROUPS)
    private val iceR = IntArray(ICE_WB_COLS * ICE_WB_ROWS)
    private val iceG = IntArray(ICE_WB_COLS * ICE_WB_ROWS)
    private val iceB = IntArray(ICE_WB_COLS * ICE_WB_ROWS)
    private val iceWb = floatArrayOf(1f, 1f, 1f)

    private data class TorsoFeatures(
        val chestRgb: List<Int>,
        val stripeScore: Float,
        val kitSummaryLab7: FloatArray,
        val lowerBellyLuma: Float,
        val pantsDarkness: Float,
        val armbandScore: Float,
        val torsoUpperLabL: Float,
    )

    fun currentIceWhiteBalance(): FloatArray = synchronized(wbLock) {
        iceWb.copyOf()
    }

    private val motionExtractor = MotionFeatureExtractor(config)
    private val featureRing = CausalFeatureRing(config.window, MotionFeatureExtractor.N_FEATURES)
    private val standardizer = RunningStandardizer(MotionFeatureExtractor.N_FEATURES, config.norm_prior)

    private var lastMotionTimeSec = Double.NaN
    private var lastLivePlayProb = 0.85f

    private data class CandidateBox(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val confidence: Float,
        val clippedEdge: Boolean,
    )

    fun resetTemporalState() {
        synchronized(detectLock) {
            motionExtractor.reset()
            featureRing.reset()
            standardizer.reset()
            lastMotionTimeSec = Double.NaN
            lastLivePlayProb = 0.85f
        }
    }

    fun detect(
        highResBitmap: Bitmap,
        timestampSec: Double,
        panoramicTiling: Boolean,
    ): YoloFrameResult = synchronized(detectLock) {
        val tStartNs = System.nanoTime()
        val srcW = highResBitmap.width
        val srcH = highResBitmap.height
        if (srcW <= 1 || srcH <= 1) {
            return YoloFrameResult(emptyList(), lastLivePlayProb, 0f, false)
        }

        val softMinConf = min(config.min_conf, SOFT_PLAYER_MIN_CONF)

        // 1. Full-frame 640x360 pass letterboxed with 12px 114/255 grey bars to 640x384.
        detCanvas.drawBitmap(highResBitmap, Rect(0, 0, srcW, srcH), Rect(0, 0, detW, detH), filterPaint)
        detBitmap.getPixels(detPixels, 0, detW, 0, 0, detW, detH)
        val fullCandidates = runFullFramePass(softMinConf)

        // 2. Causal 2-fps live-play head update strictly on >= config.min_conf full-frame layout & 160x90 grey shift.
        if (!lastMotionTimeSec.isNaN() && timestampSec < lastMotionTimeSec - 1.5) {
            motionExtractor.reset()
            featureRing.reset()
            standardizer.reset()
            lastMotionTimeSec = Double.NaN
        }
        if (lastMotionTimeSec.isNaN() || timestampSec - lastMotionTimeSec >= 0.45) {
            val liveHeadCandidates = fullCandidates.filter { it.confidence >= config.min_conf }
            lastLivePlayProb = stepLiveHead(liveHeadCandidates)
            lastMotionTimeSec = timestampSec
        }

        // 3. Optional panoramic far-ice overlapping tiles on wide aspect ratio (>= 1.65, covers 16:9 & 2.25:1).
        val aspectRatio = srcW.toFloat() / srcH.toFloat()
        val useTiles = panoramicTiling && aspectRatio >= 1.65f
        val allCandidates = ArrayList<CandidateBox>(fullCandidates.size + 32)
        allCandidates.addAll(fullCandidates)
        if (useTiles) {
            allCandidates.addAll(runTilePass(highResBitmap, 0.08f, 0.20f, 0.56f, 0.72f, softMinConf))
            allCandidates.addAll(runTilePass(highResBitmap, 0.44f, 0.20f, 0.92f, 0.72f, softMinConf))
        }

        // 4. Class-agnostic NMS (IoU > 0.42 or containment > 0.60), preferring unclipped boxes on tile seams.
        val kept = nmsMerge(allCandidates, iouThreshold = 0.42f)

        // 5. Canonicalize feature/ice-context scale at <= 960x540 for clean WB/kit clustering while
        //    retaining highResBitmap for crisp DFT stripe & dark-pants official separation.
        val featureBitmap = if (srcW > featW || srcH > featH) {
            featCanvas.drawBitmap(highResBitmap, Rect(0, 0, srcW, srcH), Rect(0, 0, featW, featH), filterPaint)
            featBitmap
        } else {
            highResBitmap
        }
        updateIceWhiteBalance(featureBitmap)
        val wb = currentIceWhiteBalance()
        val detections = ArrayList<YoloDetection>(kept.size)
        for (cand in kept) {
            val rect = RectNorm(cand.x1, cand.y1, cand.x2, cand.y2)
            val iceCtx = computeIceContext(featureBitmap, rect, wb[0], wb[1], wb[2])
            if (cand.confidence < config.min_conf) {
                val loOrFoot = iceCtx.footIceFraction >= SOFT_MIN_FOOT_ICE ||
                    iceCtx.lowerSideIceFraction >= SOFT_MIN_LOWER_SIDE_ICE
                val admitSoft = rect.height >= SOFT_MIN_BOX_HEIGHT &&
                    iceCtx.surroundingIceFraction >= SOFT_MIN_SURROUND_ICE &&
                    loOrFoot
                if (!admitSoft) continue
            }
            val torso = extractTorsoFeatures(featureBitmap, highResBitmap, rect, wb[0], wb[1], wb[2])
            detections.add(
                YoloDetection(
                    box = rect,
                    confidence = cand.confidence,
                    chestRgb = torso.chestRgb,
                    stripeScore = torso.stripeScore,
                    kitSummaryLab7 = torso.kitSummaryLab7,
                    lowerBellyLuma = torso.lowerBellyLuma,
                    pantsDarkness = torso.pantsDarkness,
                    armbandScore = torso.armbandScore,
                    torsoUpperLabL = torso.torsoUpperLabL,
                    iceContext = iceCtx,
                )
            )
        }

        val elapsedMs = (System.nanoTime() - tStartNs) / 1_000_000f
        YoloFrameResult(
            detections = detections,
            livePlayProb = lastLivePlayProb,
            detectorMs = elapsedMs,
            tiledPassUsed = useTiles,
        )
    }

    private fun runFullFramePass(minConf: Float): List<CandidateBox> {
        val plane = imgW * imgH
        inputTensorBuf.fill(114f / 255f)
        for (y in 0 until detH) {
            val rowOff = y * detW
            val dstRowOff = (y + detPadY) * imgW
            for (x in 0 until detW) {
                val p = detPixels[rowOff + x]
                val idx = dstRowOff + x
                inputTensorBuf[idx] = ((p ushr 16) and 0xFF) / 255f
                inputTensorBuf[plane + idx] = ((p ushr 8) and 0xFF) / 255f
                inputTensorBuf[2 * plane + idx] = (p and 0xFF) / 255f
            }
        }
        val rows = runDetectorOnBuffer()
        val out = ArrayList<CandidateBox>(24)
        for (r in rows) {
            if (r.size < 6 || r[4] < minConf || r[5] != 0f) continue
            val x1 = (r[0] / detW).coerceIn(0f, 1f)
            val y1 = ((r[1] - detPadY) / detH).coerceIn(0f, 1f)
            val x2 = (r[2] / detW).coerceIn(0f, 1f)
            val y2 = ((r[3] - detPadY) / detH).coerceIn(0f, 1f)
            if (x2 - x1 < 0.004f || y2 - y1 < 0.008f) continue
            out.add(CandidateBox(x1, y1, x2, y2, r[4], clippedEdge = false))
        }
        return out
    }

    private fun runTilePass(
        highResBitmap: Bitmap,
        cropX0: Float,
        cropY0: Float,
        cropX1: Float,
        cropY1: Float,
        minConf: Float,
    ): List<CandidateBox> {
        val srcW = highResBitmap.width
        val srcH = highResBitmap.height
        val sx0 = (cropX0 * srcW).toInt().coerceIn(0, srcW - 2)
        val sy0 = (cropY0 * srcH).toInt().coerceIn(0, srcH - 2)
        val sx1 = (cropX1 * srcW).toInt().coerceIn(sx0 + 2, srcW)
        val sy1 = (cropY1 * srcH).toInt().coerceIn(sy0 + 2, srcH)
        val cw = sx1 - sx0
        val ch = sy1 - sy0

        val scale = min(imgW.toFloat() / cw, imgH.toFloat() / ch)
        val dw = (cw * scale).toInt().coerceIn(1, imgW)
        val dh = (ch * scale).toInt().coerceIn(1, imgH)
        val padX = (imgW - dw) / 2
        val padY = (imgH - dh) / 2

        tileCanvas.drawColor(Color.rgb(114, 114, 114))
        tileCanvas.drawBitmap(
            highResBitmap,
            Rect(sx0, sy0, sx1, sy1),
            Rect(padX, padY, padX + dw, padY + dh),
            filterPaint,
        )
        tileBitmap.getPixels(tilePixels, 0, imgW, 0, 0, imgW, imgH)

        val plane = imgW * imgH
        for (i in 0 until plane) {
            val p = tilePixels[i]
            inputTensorBuf[i] = ((p ushr 16) and 0xFF) / 255f
            inputTensorBuf[plane + i] = ((p ushr 8) and 0xFF) / 255f
            inputTensorBuf[2 * plane + i] = (p and 0xFF) / 255f
        }

        val rows = runDetectorOnBuffer()
        val cropW = cropX1 - cropX0
        val cropH = cropY1 - cropY0
        val out = ArrayList<CandidateBox>(20)
        for (r in rows) {
            if (r.size < 6 || r[4] < minConf || r[5] != 0f) continue
            val nx1 = ((r[0] - padX) / dw).coerceIn(0f, 1f)
            val ny1 = ((r[1] - padY) / dh).coerceIn(0f, 1f)
            val nx2 = ((r[2] - padX) / dw).coerceIn(0f, 1f)
            val ny2 = ((r[3] - padY) / dh).coerceIn(0f, 1f)
            if (nx2 - nx1 < 0.008f || ny2 - ny1 < 0.015f) continue

            val fx1 = (cropX0 + nx1 * cropW).coerceIn(0f, 1f)
            val fy1 = (cropY0 + ny1 * cropH).coerceIn(0f, 1f)
            val fx2 = (cropX0 + nx2 * cropW).coerceIn(0f, 1f)
            val fy2 = (cropY0 + ny2 * cropH).coerceIn(0f, 1f)
            if (fx2 - fx1 < 0.004f || fy2 - fy1 < 0.008f) continue

            val onEdge = nx1 < 0.008f || nx2 > 0.992f || ny1 < 0.008f || ny2 > 0.992f
            out.add(CandidateBox(fx1, fy1, fx2, fy2, r[4], clippedEdge = onEdge))
        }
        return out
    }

    private fun runDetectorOnBuffer(): Array<FloatArray> {
        directInputBuffer.clear()
        directInputBuffer.put(inputTensorBuf)
        directInputBuffer.rewind()
        return OnnxTensor.createTensor(
            env,
            directInputBuffer,
            longArrayOf(1L, 3L, imgH.toLong(), imgW.toLong()),
        ).use { tensor ->
            detectorSession.run(mapOf("images" to tensor)).use { out ->
                @Suppress("UNCHECKED_CAST")
                (out[0].value as Array<Array<FloatArray>>)[0]
            }
        }
    }

    private fun stepLiveHead(fullCandidates: List<CandidateBox>): Float {
        val flatBoxes = FloatArray(4 * fullCandidates.size)
        fullCandidates.forEachIndexed { idx, c ->
            flatBoxes[4 * idx] = c.x1
            flatBoxes[4 * idx + 1] = c.y1
            flatBoxes[4 * idx + 2] = c.x2
            flatBoxes[4 * idx + 3] = c.y2
        }
        val grey = motionExtractor.grey(detPixels)
        val features = motionExtractor.push(flatBoxes, grey)
        featureRing.push(standardizer.apply(features))
        val window = featureRing.windowFlat()
        val logits = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(window),
            longArrayOf(1L, config.window.toLong(), MotionFeatureExtractor.N_FEATURES.toLong()),
        ).use { tensor ->
            liveHeadSession.run(mapOf("window" to tensor)).use { out ->
                @Suppress("UNCHECKED_CAST")
                (out[0].value as Array<FloatArray>)[0]
            }
        }
        return 1f / (1f + exp(-logits[0]))
    }

    private fun nmsMerge(candidates: List<CandidateBox>, iouThreshold: Float): List<CandidateBox> {
        if (candidates.isEmpty()) return emptyList()
        val sorted = candidates.sortedByDescending {
            it.confidence - if (it.clippedEdge) 0.06f else 0f
        }
        val kept = ArrayList<CandidateBox>(sorted.size)
        for (cand in sorted) {
            var suppressed = false
            for (k in kept) {
                val iou = boxIou(cand, k)
                if (iou > iouThreshold || intersectionOverMinArea(cand, k) > 0.60f) {
                    suppressed = true
                    break
                }
            }
            if (!suppressed) kept.add(cand)
        }
        return kept
    }

    private fun updateIceWhiteBalance(bitmap: Bitmap) {
        val w = bitmap.width
        val h = bitmap.height
        if (w < 8 || h < 8) {
            synchronized(wbLock) {
                iceWb[0] = 1f
                iceWb[1] = 1f
                iceWb[2] = 1f
            }
            return
        }
        var idx = 0
        for (ry in 0 until ICE_WB_ROWS) {
            val fy = 0.60f + (0.25f * ry) / (ICE_WB_ROWS - 1)
            val sy = (fy * h).toInt().coerceIn(0, h - 1)
            for (rx in 0 until ICE_WB_COLS) {
                val fx = 0.20f + (0.60f * rx) / (ICE_WB_COLS - 1)
                val sx = (fx * w).toInt().coerceIn(0, w - 1)
                val p = bitmap.getPixel(sx, sy)
                iceR[idx] = (p ushr 16) and 0xFF
                iceG[idx] = (p ushr 8) and 0xFF
                iceB[idx] = p and 0xFF
                idx++
            }
        }
        iceR.sort(0, idx)
        iceG.sort(0, idx)
        iceB.sort(0, idx)
        val mid = idx / 2
        val medR = iceR[mid].toFloat()
        val medG = iceG[mid].toFloat()
        val medB = iceB[mid].toFloat()
        val iceLuma = 0.299f * medR + 0.587f * medG + 0.114f * medB
        if (iceLuma < 100f) {
            synchronized(wbLock) {
                iceWb[0] = 1f
                iceWb[1] = 1f
                iceWb[2] = 1f
            }
            return
        }
        val meanIce = (medR + medG + medB) / 3f
        val gR = (meanIce / max(1f, medR)).coerceIn(0.70f, 1.45f)
        val gG = (meanIce / max(1f, medG)).coerceIn(0.70f, 1.45f)
        val gB = (meanIce / max(1f, medB)).coerceIn(0.70f, 1.45f)
        synchronized(wbLock) {
            iceWb[0] = gR
            iceWb[1] = gG
            iceWb[2] = gB
        }
    }

    private fun loadBoxPixels(bitmap: Bitmap, box: RectNorm): Pair<Int, Int>? {
        val w = bitmap.width
        val h = bitmap.height
        val bx0 = (box.x1 * w).toInt().coerceIn(0, w - 1)
        val by0 = (box.y1 * h).toInt().coerceIn(0, h - 1)
        val bx1 = (box.x2 * w).toInt().coerceIn(bx0 + 1, w)
        val by1 = (box.y2 * h).toInt().coerceIn(by0 + 1, h)
        val bwPx = bx1 - bx0
        val bhPx = by1 - by0
        val boxCount = bwPx * bhPx
        if (bwPx < 3 || bhPx < 6 || boxCount <= 0) return null
        if (boxPixelBuf.size < boxCount) {
            boxPixelBuf = IntArray(boxCount)
        }
        bitmap.getPixels(boxPixelBuf, 0, bwPx, bx0, by0, bwPx, bhPx)
        return bwPx to bhPx
    }

    private fun extractTorsoFeatures(
        featureBitmap: Bitmap,
        highResBitmap: Bitmap,
        box: RectNorm,
        wbR: Float,
        wbG: Float,
        wbB: Float,
    ): TorsoFeatures {
        val featDims = loadBoxPixels(featureBitmap, box)
            ?: return TorsoFeatures(
                chestRgb = listOf(128, 128, 128),
                stripeScore = 0f,
                kitSummaryLab7 = floatArrayOf(50f, 0f, 0f, 50f, 0f, 0f, 0f),
                lowerBellyLuma = Float.NaN,
                pantsDarkness = Float.NaN,
                armbandScore = 0f,
                torsoUpperLabL = Float.NaN,
            )
        val featBwPx = featDims.first
        val featBhPx = featDims.second
        val featTorsoCenterFrac = findTorsoCenterFrac(featBwPx, featBhPx, wbR, wbG, wbB)
        val chestRgb = extractCoreChestRgb(featBwPx, featBhPx, wbR, wbG, wbB, featTorsoCenterFrac)
        val kitSummaryLab7 = extractKitSummaryLab7(featBwPx, featBhPx, wbR, wbG, wbB, featTorsoCenterFrac)
        val torsoUpperLabL = lastExtractedUpperLabL
        val lowerBellyLuma = extractLowerBellyLuma(featBwPx, featBhPx, wbR, wbG, wbB, featTorsoCenterFrac)

        val hiDims = if (featureBitmap === highResBitmap) {
            featDims
        } else {
            loadBoxPixels(highResBitmap, box)
        }
        val pantsDarkness: Float
        val armbandScore: Float
        val stripeScore: Float
        if (hiDims != null) {
            val hiBwPx = hiDims.first
            val hiBhPx = hiDims.second
            val hiTorsoCenterFrac = if (featureBitmap === highResBitmap) {
                featTorsoCenterFrac
            } else {
                findTorsoCenterFrac(hiBwPx, hiBhPx, wbR, wbG, wbB)
            }
            pantsDarkness = extractPantsDarkness(hiBwPx, hiBhPx, wbR, wbG, wbB)
            armbandScore = extractArmbandScore(hiBwPx, hiBhPx, wbR, wbG, wbB)
            stripeScore = computeDftStripeScore(hiBwPx, hiBhPx, wbR, wbG, wbB, hiTorsoCenterFrac)
        } else {
            pantsDarkness = Float.NaN
            armbandScore = 0f
            stripeScore = 0f
        }

        return TorsoFeatures(
            chestRgb = chestRgb,
            stripeScore = stripeScore,
            kitSummaryLab7 = kitSummaryLab7,
            lowerBellyLuma = lowerBellyLuma,
            pantsDarkness = pantsDarkness,
            armbandScore = armbandScore,
            torsoUpperLabL = torsoUpperLabL,
        )
    }

    private fun findTorsoCenterFrac(
        bwPx: Int,
        bhPx: Int,
        wbR: Float,
        wbG: Float,
        wbB: Float,
    ): Float {
        // Wide stick-extension / crouched boxes (e.g., 82x70) shift the skater off-center;
        // locate the darkest 42%-wide torso strip across y: 20%..56% so blank ice is avoided.
        if (bwPx <= 0.65f * bhPx || bwPx < 10) return 0.50f
        val y0 = (0.20f * bhPx).toInt().coerceIn(0, bhPx - 1)
        val y1 = (0.56f * bhPx).toInt().coerceIn(y0 + 1, bhPx)
        val rows = max(1, y1 - y0)
        val winW = (0.42f * bwPx).roundToInt().coerceIn(3, bwPx)
        val minX = (0.04f * bwPx).toInt().coerceIn(0, max(0, bwPx - winW))
        val maxX = (bwPx - winW - (0.04f * bwPx).toInt()).coerceIn(minX, bwPx - winW)
        if (maxX <= minX) return 0.50f

        val colLuma = FloatArray(bwPx)
        for (x in 0 until bwPx) {
            var sum = 0f
            for (y in y0 until y1) {
                val p = boxPixelBuf[y * bwPx + x]
                val r = (((p ushr 16) and 0xFF) * wbR).coerceIn(0f, 255f)
                val g = (((p ushr 8) and 0xFF) * wbG).coerceIn(0f, 255f)
                val b = ((p and 0xFF) * wbB).coerceIn(0f, 255f)
                sum += 0.299f * r + 0.587f * g + 0.114f * b
            }
            colLuma[x] = sum / rows
        }

        var bestScore = Float.MAX_VALUE
        var bestCenter = 0.50f
        for (sx0 in minX..maxX) {
            var winSum = 0f
            for (x in sx0 until sx0 + winW) winSum += colLuma[x]
            val meanLum = winSum / winW
            val centerFrac = (sx0 + winW * 0.5f) / bwPx.toFloat()
            val score = meanLum + 20f * abs(centerFrac - 0.50f)
            if (score < bestScore) {
                bestScore = score
                bestCenter = centerFrac
            }
        }
        return bestCenter.coerceIn(0.24f, 0.76f)
    }

    private fun extractCoreChestRgb(
        bwPx: Int,
        bhPx: Int,
        wbR: Float,
        wbG: Float,
        wbB: Float,
        torsoCenterFrac: Float,
    ): List<Int> {
        val x0 = ((torsoCenterFrac - 0.14f) * bwPx).toInt().coerceIn(0, bwPx - 1)
        val x1 = ((torsoCenterFrac + 0.14f) * bwPx).toInt().coerceIn(x0 + 1, bwPx)
        val y0 = (0.30f * bhPx).toInt().coerceIn(0, bhPx - 1)
        val y1 = (0.54f * bhPx).toInt().coerceIn(y0 + 1, bhPx)
        val count = (x1 - x0) * (y1 - y0)
        if (count <= 0) return listOf(128, 128, 128)
        if (chestR.size < count) {
            chestR = IntArray(count)
            chestG = IntArray(count)
            chestB = IntArray(count)
        }
        var idx = 0
        for (y in y0 until y1) {
            val rowOff = y * bwPx
            for (x in x0 until x1) {
                val p = boxPixelBuf[rowOff + x]
                chestR[idx] = (((p ushr 16) and 0xFF) * wbR).roundToInt().coerceIn(0, 255)
                chestG[idx] = (((p ushr 8) and 0xFF) * wbG).roundToInt().coerceIn(0, 255)
                chestB[idx] = ((p and 0xFF) * wbB).roundToInt().coerceIn(0, 255)
                idx++
            }
        }
        chestR.sort(0, idx)
        chestG.sort(0, idx)
        chestB.sort(0, idx)
        val lo = if (idx >= 5) (idx * 2) / 5 else 0
        val hi = if (idx >= 5) max(lo + 1, (idx * 4) / 5) else idx
        val span = max(1, hi - lo)
        var sumR = 0L
        var sumG = 0L
        var sumB = 0L
        for (i in lo until hi) {
            sumR += chestR[i]
            sumG += chestG[i]
            sumB += chestB[i]
        }
        return listOf(
            (sumR / span).toInt().coerceIn(0, 255),
            (sumG / span).toInt().coerceIn(0, 255),
            (sumB / span).toInt().coerceIn(0, 255),
        )
    }

    private fun extractKitSummaryLab7(
        bwPx: Int,
        bhPx: Int,
        wbR: Float,
        wbG: Float,
        wbB: Float,
        torsoCenterFrac: Float,
    ): FloatArray {
        lastExtractedUpperLabL = Float.NaN
        // 1. Border ice background estimation on x: 0..10% & 90..100%, y: 15..55%.
        val by0 = (0.15f * bhPx).toInt().coerceIn(0, bhPx - 1)
        val by1 = (0.55f * bhPx).toInt().coerceIn(by0 + 1, bhPx)
        val leftX1 = (0.10f * bwPx).toInt().coerceIn(1, max(1, bwPx / 3))
        val rightX0 = (0.90f * bwPx).toInt().coerceIn(leftX1, bwPx - 1)
        val maxBorder = (by1 - by0) * (leftX1 + (bwPx - rightX0))
        if (borderL.size < maxBorder) {
            borderL = FloatArray(maxBorder)
            borderA = FloatArray(maxBorder)
            borderB = FloatArray(maxBorder)
        }
        var totalBorder = 0
        var iceBorderCount = 0
        val labTmp = FloatArray(3)
        for (y in by0 until by1) {
            val rowOff = y * bwPx
            for (x in 0 until leftX1) {
                totalBorder++
                pixelToLab(boxPixelBuf[rowOff + x], wbR, wbG, wbB, labTmp)
                if (labTmp[0] >= 48f && hypot(labTmp[1], labTmp[2]) < 22f) {
                    borderL[iceBorderCount] = labTmp[0]
                    borderA[iceBorderCount] = labTmp[1]
                    borderB[iceBorderCount] = labTmp[2]
                    iceBorderCount++
                }
            }
            for (x in rightX0 until bwPx) {
                totalBorder++
                pixelToLab(boxPixelBuf[rowOff + x], wbR, wbG, wbB, labTmp)
                if (labTmp[0] >= 48f && hypot(labTmp[1], labTmp[2]) < 22f) {
                    borderL[iceBorderCount] = labTmp[0]
                    borderA[iceBorderCount] = labTmp[1]
                    borderB[iceBorderCount] = labTmp[2]
                    iceBorderCount++
                }
            }
        }
        val hasIceRef = iceBorderCount >= max(4, totalBorder / 4)
        var iceBgL = 85f
        var iceBgA = 0f
        var iceBgB = 0f
        if (hasIceRef) {
            borderL.sort(0, iceBorderCount)
            borderA.sort(0, iceBorderCount)
            borderB.sort(0, iceBorderCount)
            val mid = iceBorderCount / 2
            iceBgL = borderL[mid]
            iceBgA = borderA[mid]
            iceBgB = borderB[mid]
        }

        // 2. Torso pixels (centered on torsoCenterFrac; x: 26%..74% for normal boxes, slid for wide stick boxes).
        val halfW = if (bwPx > 0.65f * bhPx) 0.21f else 0.24f
        val tx0 = ((torsoCenterFrac - halfW) * bwPx).toInt().coerceIn(0, bwPx - 1)
        val tx1 = ((torsoCenterFrac + halfW) * bwPx).toInt().coerceIn(tx0 + 1, bwPx)
        val ty0 = (0.22f * bhPx).toInt().coerceIn(0, bhPx - 1)
        val ty1 = (0.56f * bhPx).toInt().coerceIn(ty0 + 1, bhPx)
        val torsoCount = (tx1 - tx0) * (ty1 - ty0)
        if (torsoCount <= 0) return floatArrayOf(50f, 0f, 0f, 50f, 0f, 0f, 0f)
        if (torsoL.size < torsoCount) {
            torsoL = FloatArray(torsoCount)
            torsoA = FloatArray(torsoCount)
            torsoB = FloatArray(torsoCount)
            torsoKeep = BooleanArray(torsoCount)
            torsoAssign = IntArray(torsoCount)
        }

        var idx = 0
        var iceLikeCount = 0
        for (y in ty0 until ty1) {
            val rowOff = y * bwPx
            for (x in tx0 until tx1) {
                pixelToLab(boxPixelBuf[rowOff + x], wbR, wbG, wbB, labTmp)
                val l = labTmp[0]
                val a = labTmp[1]
                val b = labTmp[2]
                torsoL[idx] = l
                torsoA[idx] = a
                torsoB[idx] = b
                var isIceLike = false
                if (hasIceRef) {
                    val dl = l - iceBgL
                    val da = a - iceBgA
                    val db = b - iceBgB
                    if (dl * dl + da * da + db * db < ICE_DELTA_E_SQ_THRESHOLD) {
                        isIceLike = true
                        iceLikeCount++
                    }
                }
                torsoKeep[idx] = !isIceLike
                idx++
            }
        }

        // Discount pixels within deltaE < 11 of surrounding ice unless >40% ice-like (white jersey),
        // or up to 75% on wide stick-boxes when >= 10 dark non-ice pixels remain.
        val wideStickBox = bwPx > 0.65f * bhPx
        val maxDropFrac = if (wideStickBox) 0.75f else 0.40f
        var nonIceDarkCount = 0
        var nonIceSumL = 0f
        if (wideStickBox && hasIceRef && iceLikeCount > torsoCount * 0.40f && iceLikeCount <= torsoCount * 0.75f) {
            for (i in 0 until torsoCount) {
                if (torsoKeep[i]) {
                    nonIceDarkCount++
                    nonIceSumL += torsoL[i]
                }
            }
        }
        val allowWideDarkDrop = !wideStickBox ||
            iceLikeCount <= torsoCount * 0.40f ||
            (nonIceDarkCount >= 10 && (nonIceSumL / max(1, nonIceDarkCount)) < 45f)
        val dropIce = hasIceRef &&
            allowWideDarkDrop &&
            iceLikeCount > 0 &&
            iceLikeCount <= (torsoCount * maxDropFrac) &&
            (torsoCount - iceLikeCount) >= if (wideStickBox) 10 else 6
        if (!dropIce) {
            for (i in 0 until torsoCount) torsoKeep[i] = true
        }

        // Deterministic 2-means CIELAB seeded at min-L and max-L pixels.
        var minIdx = -1
        var maxIdx = -1
        var minL = Float.MAX_VALUE
        var maxL = -Float.MAX_VALUE
        var keptCount = 0
        var sumAllL = 0.0
        var sumAllA = 0.0
        var sumAllB = 0.0
        for (i in 0 until torsoCount) {
            if (!torsoKeep[i]) continue
            keptCount++
            val l = torsoL[i]
            sumAllL += l
            sumAllA += torsoA[i]
            sumAllB += torsoB[i]
            if (l < minL) {
                minL = l
                minIdx = i
            }
            if (l > maxL) {
                maxL = l
                maxIdx = i
            }
        }
        if (keptCount <= 0 || minIdx < 0 || maxIdx < 0) {
            lastExtractedUpperLabL = Float.NaN
            return floatArrayOf(50f, 0f, 0f, 50f, 0f, 0f, 0f)
        }
        if (upperLabLBuf.size < keptCount) upperLabLBuf = FloatArray(keptCount)
        var upperIdx = 0
        for (i in 0 until torsoCount) {
            if (torsoKeep[i]) upperLabLBuf[upperIdx++] = torsoL[i]
        }
        upperLabLBuf.sort(0, upperIdx)
        val p80Pos = 0.80f * (upperIdx - 1)
        val p80Lo = p80Pos.toInt().coerceIn(0, upperIdx - 1)
        val p80Hi = min(upperIdx - 1, p80Lo + 1)
        val p80Frac = p80Pos - p80Lo
        lastExtractedUpperLabL = upperLabLBuf[p80Lo] * (1f - p80Frac) + upperLabLBuf[p80Hi] * p80Frac
        if (maxL - minL < 2.0f || keptCount < 4) {
            val mL = (sumAllL / keptCount).toFloat()
            val mA = (sumAllA / keptCount).toFloat()
            val mB = (sumAllB / keptCount).toFloat()
            val share = if (mL >= 62f) 100f else 0f
            return floatArrayOf(mL, mA, mB, mL, mA, mB, share)
        }

        var c0L = torsoL[minIdx]
        var c0A = torsoA[minIdx]
        var c0B = torsoB[minIdx]
        var c1L = torsoL[maxIdx]
        var c1A = torsoA[maxIdx]
        var c1B = torsoB[maxIdx]

        var n0 = 0
        var n1 = 0
        repeat(6) {
            var s0L = 0.0; var s0A = 0.0; var s0B = 0.0; var c0N = 0
            var s1L = 0.0; var s1A = 0.0; var s1B = 0.0; var c1N = 0
            for (i in 0 until torsoCount) {
                if (!torsoKeep[i]) continue
                val l = torsoL[i]; val a = torsoA[i]; val b = torsoB[i]
                val d0 = (l - c0L) * (l - c0L) + (a - c0A) * (a - c0A) + (b - c0B) * (b - c0B)
                val d1 = (l - c1L) * (l - c1L) + (a - c1A) * (a - c1A) + (b - c1B) * (b - c1B)
                if (d0 <= d1) {
                    torsoAssign[i] = 0
                    s0L += l; s0A += a; s0B += b; c0N++
                } else {
                    torsoAssign[i] = 1
                    s1L += l; s1A += a; s1B += b; c1N++
                }
            }
            n0 = c0N
            n1 = c1N
            if (n0 == 0 || n1 == 0) return@repeat
            val n0L = (s0L / n0).toFloat(); val n0A = (s0A / n0).toFloat(); val n0B = (s0B / n0).toFloat()
            val n1L = (s1L / n1).toFloat(); val n1A = (s1A / n1).toFloat(); val n1B = (s1B / n1).toFloat()
            val maxShift = maxOf(
                abs(n0L - c0L), abs(n0A - c0A), abs(n0B - c0B),
                abs(n1L - c1L), abs(n1A - c1A), abs(n1B - c1B),
            )
            c0L = n0L; c0A = n0A; c0B = n0B
            c1L = n1L; c1A = n1A; c1B = n1B
            if (maxShift < 0.05f) return@repeat
        }

        if (n0 == 0 || n1 == 0) {
            val mL = (sumAllL / keptCount).toFloat()
            val mA = (sumAllA / keptCount).toFloat()
            val mB = (sumAllB / keptCount).toFloat()
            val share = if (mL >= 62f) 100f else 0f
            return floatArrayOf(mL, mA, mB, mL, mA, mB, share)
        }

        return if (c0L <= c1L) {
            val lightShare = (n1.toFloat() / keptCount.toFloat()) * 100f
            floatArrayOf(c0L, c0A, c0B, c1L, c1A, c1B, lightShare)
        } else {
            val lightShare = (n0.toFloat() / keptCount.toFloat()) * 100f
            floatArrayOf(c1L, c1A, c1B, c0L, c0A, c0B, lightShare)
        }
    }

    private fun extractLowerBellyLuma(
        bwPx: Int,
        bhPx: Int,
        wbR: Float,
        wbG: Float,
        wbB: Float,
        torsoCenterFrac: Float,
    ): Float {
        // x: 32%..68% (slid to torsoCenterFrac on wide stick-boxes), y: 42%..56%.
        val x0 = ((torsoCenterFrac - 0.18f) * bwPx).toInt().coerceIn(0, bwPx - 1)
        val x1 = ((torsoCenterFrac + 0.18f) * bwPx).toInt().coerceIn(x0 + 1, bwPx)
        val y0 = (0.42f * bhPx).toInt().coerceIn(0, bhPx - 1)
        val y1 = (0.56f * bhPx).toInt().coerceIn(y0 + 1, bhPx)
        val count = (x1 - x0) * (y1 - y0)
        if (count < 4) return Float.NaN
        if (bellyLumaBuf.size < count) bellyLumaBuf = FloatArray(count)
        var idx = 0
        for (y in y0 until y1) {
            val rowOff = y * bwPx
            for (x in x0 until x1) {
                val p = boxPixelBuf[rowOff + x]
                val r = (((p ushr 16) and 0xFF) * wbR).coerceIn(0f, 255f)
                val g = (((p ushr 8) and 0xFF) * wbG).coerceIn(0f, 255f)
                val b = ((p and 0xFF) * wbB).coerceIn(0f, 255f)
                bellyLumaBuf[idx++] = 0.299f * r + 0.587f * g + 0.114f * b
            }
        }
        bellyLumaBuf.sort(0, idx)
        val lo = if (idx >= 5) (idx * 2) / 5 else 0
        val hi = if (idx >= 5) max(lo + 1, (idx * 4) / 5) else idx
        var sum = 0.0
        for (i in lo until hi) sum += bellyLumaBuf[i]
        return (sum / max(1, hi - lo)).toFloat()
    }

    private fun extractPantsDarkness(
        bwPx: Int,
        bhPx: Int,
        wbR: Float,
        wbG: Float,
        wbB: Float,
    ): Float {
        // Fraction of achromatic shadow-tolerant dark pixels (luma < 88 && chromaSpread < 36) across x: 28%..72%, y: 56%..84%.
        val x0 = (0.28f * bwPx).toInt().coerceIn(0, bwPx - 1)
        val x1 = (0.72f * bwPx).toInt().coerceIn(x0 + 1, bwPx)
        val y0 = (0.56f * bhPx).toInt().coerceIn(0, bhPx - 1)
        val y1 = (0.84f * bhPx).toInt().coerceIn(y0 + 1, bhPx)
        val count = (x1 - x0) * (y1 - y0)
        if (count < 4) return Float.NaN
        var darkCount = 0
        for (y in y0 until y1) {
            val rowOff = y * bwPx
            for (x in x0 until x1) {
                val p = boxPixelBuf[rowOff + x]
                val r = (((p ushr 16) and 0xFF) * wbR).coerceIn(0f, 255f)
                val g = (((p ushr 8) and 0xFF) * wbG).coerceIn(0f, 255f)
                val b = ((p and 0xFF) * wbB).coerceIn(0f, 255f)
                val lum = 0.299f * r + 0.587f * g + 0.114f * b
                val spread = maxOf(r, g, b) - minOf(r, g, b)
                if (lum < 88f && spread < 36f) darkCount++
            }
        }
        return darkCount.toFloat() / count.toFloat()
    }

    private fun extractArmbandScore(
        bwPx: Int,
        bhPx: Int,
        wbR: Float,
        wbG: Float,
        wbB: Float,
    ): Float {
        // HSV orange fraction (H 4..26 deg, S >= 0.45, V >= 0.40) on sleeves x: 0..25% & 75..100%, y: 15%..45%.
        val y0 = (0.15f * bhPx).toInt().coerceIn(0, bhPx - 1)
        val y1 = (0.45f * bhPx).toInt().coerceIn(y0 + 1, bhPx)
        val leftX1 = (0.25f * bwPx).toInt().coerceIn(1, max(1, bwPx / 2))
        val rightX0 = (0.75f * bwPx).toInt().coerceIn(leftX1, bwPx - 1)
        var total = 0
        var orange = 0
        for (y in y0 until y1) {
            val rowOff = y * bwPx
            for (x in 0 until leftX1) {
                total++
                if (isOrangeArmbandPixel(boxPixelBuf[rowOff + x], wbR, wbG, wbB)) orange++
            }
            for (x in rightX0 until bwPx) {
                total++
                if (isOrangeArmbandPixel(boxPixelBuf[rowOff + x], wbR, wbG, wbB)) orange++
            }
        }
        return if (total < 4) 0f else orange.toFloat() / total.toFloat()
    }

    private fun computeDftStripeScore(
        bwPx: Int,
        bhPx: Int,
        wbR: Float,
        wbG: Float,
        wbB: Float,
        torsoCenterFrac: Float,
    ): Float {
        // Torso-centered bilinear 64x36 1D Hann-DFT on x: torsoCenterFrac ± halfW, y: 22%..58%.
        val wideStickBox = bwPx > 0.65f * bhPx
        val halfW = if (wideStickBox) 0.21f else 0.24f
        val sx0 = (torsoCenterFrac - halfW).coerceIn(0f, 1f) * bwPx
        val sx1 = (torsoCenterFrac + halfW).coerceIn(0f, 1f) * bwPx
        val sy0 = 0.22f * bhPx
        val sy1 = 0.58f * bhPx
        val spanW = sx1 - sx0
        val spanH = sy1 - sy0
        if (spanW < 4f || spanH < 4f) return 0f

        val maxX = (bwPx - 1).toFloat()
        val maxY = (bhPx - 1).toFloat()
        var achromaticCount = 0
        var sumSat = 0f

        for (row in 0 until DFT_H) {
            val fy = (sy0 + ((row + 0.5f) / DFT_H) * spanH - 0.5f).coerceIn(0f, maxY)
            val y0 = fy.toInt().coerceIn(0, bhPx - 1)
            val y1 = min(y0 + 1, bhPx - 1)
            val ty = fy - y0
            val ity = 1f - ty
            val rowOff0 = y0 * bwPx
            val rowOff1 = y1 * bwPx
            val dstRowOff = row * DFT_W

            for (col in 0 until DFT_W) {
                val fx = (sx0 + ((col + 0.5f) / DFT_W) * spanW - 0.5f).coerceIn(0f, maxX)
                val x0 = fx.toInt().coerceIn(0, bwPx - 1)
                val x1 = min(x0 + 1, bwPx - 1)
                val tx = fx - x0
                val itx = 1f - tx

                val p00 = boxPixelBuf[rowOff0 + x0]
                val p10 = boxPixelBuf[rowOff0 + x1]
                val p01 = boxPixelBuf[rowOff1 + x0]
                val p11 = boxPixelBuf[rowOff1 + x1]

                val w00 = itx * ity
                val w10 = tx * ity
                val w01 = itx * ty
                val w11 = tx * ty

                val rRaw = w00 * ((p00 ushr 16) and 0xFF) +
                    w10 * ((p10 ushr 16) and 0xFF) +
                    w01 * ((p01 ushr 16) and 0xFF) +
                    w11 * ((p11 ushr 16) and 0xFF)
                val gRaw = w00 * ((p00 ushr 8) and 0xFF) +
                    w10 * ((p10 ushr 8) and 0xFF) +
                    w01 * ((p01 ushr 8) and 0xFF) +
                    w11 * ((p11 ushr 8) and 0xFF)
                val bRaw = w00 * (p00 and 0xFF) +
                    w10 * (p10 and 0xFF) +
                    w01 * (p01 and 0xFF) +
                    w11 * (p11 and 0xFF)

                val r = (rRaw * wbR).coerceIn(0f, 255f)
                val g = (gRaw * wbG).coerceIn(0f, 255f)
                val b = (bRaw * wbB).coerceIn(0f, 255f)

                val lum = 0.299f * r + 0.587f * g + 0.114f * b
                val maxC = maxOf(r, g, b)
                val minC = minOf(r, g, b)
                val spread = maxC - minC
                val sat = if (maxC <= 1f) 0f else spread / maxC
                if (spread < 36f || sat < 0.24f) achromaticCount++
                sumSat += sat

                val dIdx = dstRowOff + col
                dftLuma[dIdx] = lum
                dftLumaSorted[dIdx] = lum
            }
        }

        val totalPx = DFT_W * DFT_H
        dftLumaSorted.sort(0, totalPx)
        val p10 = dftLumaSorted[(totalPx * 0.10f).toInt()]
        val p90 = dftLumaSorted[(totalPx * 0.90f).toInt()]
        // White-stripe p90 luma guard >= 122f prevents solid dark navy (#14) false positives.
        if (p90 < 122f || (p10 + p90) * 0.5f < 78f || (p90 - p10) < 22f) return 0f

        val achromaticFrac = achromaticCount.toFloat() / totalPx.toFloat()
        val meanSat = sumSat / totalPx.toFloat()
        val chromaGate = ((achromaticFrac - 0.45f) / 0.35f).coerceIn(0f, 1f) *
            ((0.40f - meanSat) / 0.18f).coerceIn(0f, 1f)
        if (chromaGate <= 0f) return 0f

        val midLuma = 0.5f * (p10 + p90)
        var darkPx = 0
        for (i in 0 until totalPx) {
            if (dftLuma[i] < midLuma) darkPx++
        }
        val darkFrac = darkPx.toFloat() / totalPx.toFloat()
        val pxBalance = (min(darkFrac, 1f - darkFrac) / 0.25f).coerceIn(0f, 1f)
        if (pxBalance <= 0f) return 0f

        // Per-row 1D Hann-DFT across 64 columns in frequency bins k = 3..15 vs low bins k = 1..2.
        for (row in 0 until DFT_H) {
            val rowOff = row * DFT_W
            var mean = 0f
            for (c in 0 until DFT_W) mean += dftLuma[rowOff + c]
            mean /= DFT_W.toFloat()

            var acSumSq = 0f
            for (c in 0 until DFT_W) {
                val v = (dftLuma[rowOff + c] - mean) * HANN_WINDOW[c]
                dftWinRow[c] = v
                acSumSq += v * v
            }
            dftAcEnergy[row] = acSumSq / HANN_SUM_SQ

            var lowPow = 0f
            var bandPow = 0f
            for (k in 1..DFT_K_MAX) {
                val cosRow = DFT_COS[k]
                val sinRow = DFT_SIN[k]
                var re = 0f
                var im = 0f
                for (c in 0 until DFT_W) {
                    val v = dftWinRow[c]
                    re += v * cosRow[c]
                    im -= v * sinRow[c]
                }
                val p = 2f * (re * re + im * im) / (DFT_W.toFloat() * HANN_SUM_SQ)
                if (k <= 2) lowPow += p else bandPow += p
            }
            dftLowPower[row] = lowPow
            dftBandPower[row] = bandPow
            dftRowAmp[row] = sqrt(bandPow)
            dftRowAmpSorted[row] = dftRowAmp[row]
        }

        // 6-row-band consistency (6 bands of 6 rows each).
        val rowsPerBand = DFT_H / STRIPE_ROW_GROUPS
        for (g in 0 until STRIPE_ROW_GROUPS) {
            val rStart = g * rowsPerBand
            val rEnd = rStart + rowsPerBand
            var eb = 0f
            var el = 0f
            var etot = 0f
            for (r in rStart until rEnd) {
                eb += dftBandPower[r]
                el += dftLowPower[r]
                etot += dftAcEnergy[r]
            }
            eb /= rowsPerBand
            el /= rowsPerBand
            etot = max(1e-6f, etot / rowsPerBand)
            val ratioBand = eb / max(1e-6f, eb + el)
            val ratioTotal = eb / etot
            val amp = sqrt(eb)
            dftGroupScores[g] = ((ratioBand - 0.22f) / 0.36f).coerceIn(0f, 1f) *
                ((ratioTotal - 0.20f) / 0.35f).coerceIn(0f, 1f) *
                (amp / 6.5f).coerceIn(0f, 1f)
        }
        dftGroupScores.sort()
        val bandConsistency = 0.55f * dftGroupScores[2] + 0.45f * dftGroupScores[3]
        if (bandConsistency <= 0f) return 0f

        // Vertical row extent: referee stripes span almost every torso row; jersey numbers do not.
        dftRowAmpSorted.sort()
        val refAmp = dftRowAmpSorted[(DFT_H * 0.90f).toInt().coerceIn(0, DFT_H - 1)]
        if (refAmp <= 1e-4f) return 0f
        var activeRows = 0
        val rowThresh = 0.40f * refAmp
        for (r in 0 until DFT_H) {
            if (dftRowAmp[r] >= rowThresh) activeRows++
        }
        val extent = activeRows.toFloat() / DFT_H.toFloat()
        val extentScore = ((extent - 0.50f) / 0.30f).coerceIn(0f, 1f)
        if (extentScore <= 0f) return 0f

        return (sqrt(bandConsistency * extentScore * pxBalance) * chromaGate).coerceIn(0f, 1f)
    }

    private fun computeIceContext(
        bitmap: Bitmap,
        box: RectNorm,
        wbR: Float,
        wbG: Float,
        wbB: Float,
    ): OnIceFilter.IceContext {
        val w = bitmap.width
        val h = bitmap.height
        val x1 = box.x1
        val y1 = box.y1
        val x2 = box.x2
        val y2 = box.y2
        val bw = box.width
        val bh = box.height

        val footX0 = x1 - 0.20f * bw
        val footY0 = y2 - 0.02f * bh
        val footX1 = x2 + 0.20f * bw
        val footY1 = min(1f, y2 + 0.14f * bh)

        iceStatsBuf[0] = 0; iceStatsBuf[1] = 0
        accumulateIceSamples(bitmap, w, h, footX0, footY0, footX1, footY1, wbR, wbG, wbB, iceStatsBuf)
        val footIce = if (iceStatsBuf[1] > 0) iceStatsBuf[0].toFloat() / iceStatsBuf[1].toFloat() else 0f

        iceStatsBuf[0] = 0; iceStatsBuf[1] = 0
        accumulateIceSamples(bitmap, w, h, x1 - 0.25f * bw, y1 + 0.50f * bh, x1, y1 + 0.90f * bh, wbR, wbG, wbB, iceStatsBuf)
        val loL = if (iceStatsBuf[1] > 0) iceStatsBuf[0].toFloat() / iceStatsBuf[1].toFloat() else 0f

        iceStatsBuf[0] = 0; iceStatsBuf[1] = 0
        accumulateIceSamples(bitmap, w, h, x2, y1 + 0.50f * bh, x2 + 0.25f * bw, y1 + 0.90f * bh, wbR, wbG, wbB, iceStatsBuf)
        val loR = if (iceStatsBuf[1] > 0) iceStatsBuf[0].toFloat() / iceStatsBuf[1].toFloat() else 0f

        iceStatsBuf[0] = 0; iceStatsBuf[1] = 0
        accumulateIceSamples(bitmap, w, h, x1 - 0.25f * bw, y1 + 0.10f * bh, x1, y1 + 0.50f * bh, wbR, wbG, wbB, iceStatsBuf)
        val upL = if (iceStatsBuf[1] > 0) iceStatsBuf[0].toFloat() / iceStatsBuf[1].toFloat() else 0f

        iceStatsBuf[0] = 0; iceStatsBuf[1] = 0
        accumulateIceSamples(bitmap, w, h, x2, y1 + 0.10f * bh, x2 + 0.25f * bw, y1 + 0.50f * bh, wbR, wbG, wbB, iceStatsBuf)
        val upR = if (iceStatsBuf[1] > 0) iceStatsBuf[0].toFloat() / iceStatsBuf[1].toFloat() else 0f

        iceStatsBuf[0] = 0; iceStatsBuf[1] = 0
        accumulateIceSamples(bitmap, w, h, footX0, footY0, footX1, footY1, wbR, wbG, wbB, iceStatsBuf)
        accumulateIceSamples(bitmap, w, h, x1 - 0.25f * bw, y1 + 0.30f * bh, x1, y1 + 0.90f * bh, wbR, wbG, wbB, iceStatsBuf)
        accumulateIceSamples(bitmap, w, h, x2, y1 + 0.30f * bh, x2 + 0.25f * bw, y1 + 0.90f * bh, wbR, wbG, wbB, iceStatsBuf)
        val surrIce = if (iceStatsBuf[1] > 0) iceStatsBuf[0].toFloat() / iceStatsBuf[1].toFloat() else 0f

        return OnIceFilter.IceContext(
            footIceFraction = footIce,
            surroundingIceFraction = surrIce,
            lowerSideIceFraction = 0.5f * (loL + loR),
            upperSideIceFraction = 0.5f * (upL + upR),
        )
    }

    private fun accumulateIceSamples(
        bitmap: Bitmap,
        w: Int,
        h: Int,
        rx0: Float,
        ry0: Float,
        rx1: Float,
        ry1: Float,
        wbR: Float,
        wbG: Float,
        wbB: Float,
        statsOut: IntArray,
    ) {
        val px0 = (rx0 * w).toInt().coerceIn(0, w)
        val px1 = (rx1 * w).toInt().coerceIn(0, w)
        val py0 = (ry0 * h).toInt().coerceIn(0, h)
        val py1 = (ry1 * h).toInt().coerceIn(0, h)
        val spanW = px1 - px0
        val spanH = py1 - py0
        if (spanW <= 0 || spanH <= 0) return

        val ny = min(8, spanH)
        val nx = min(8, spanW)
        for (iy in 0 until ny) {
            val y = if (ny == 1) py0 else py0 + (iy * (spanH - 1)) / (ny - 1)
            for (ix in 0 until nx) {
                val x = if (nx == 1) px0 else px0 + (ix * (spanW - 1)) / (nx - 1)
                pixelToLab(bitmap.getPixel(x, y), wbR, wbG, wbB, tmpIceLab)
                val chroma = hypot(tmpIceLab[1], tmpIceLab[2])
                if (tmpIceLab[0] >= ICE_CTX_MIN_L && chroma < ICE_CTX_MAX_CHROMA) {
                    statsOut[0]++
                }
                statsOut[1]++
            }
        }
    }

    override fun close() {
        runCatching { detectorSession.close() }
        runCatching { liveHeadSession.close() }
        runCatching { detectorOptions.close() }
        runCatching { liveHeadOptions.close() }
        runCatching { detBitmap.recycle() }
        runCatching { tileBitmap.recycle() }
        runCatching { featBitmap.recycle() }
    }

    private class CausalFeatureRing(private val window: Int, private val dim: Int) {
        private val buf = Array(window) { FloatArray(dim) }
        private var count = 0

        fun push(features: FloatArray) {
            features.copyInto(buf[count % window])
            count++
        }

        fun windowFlat(out: FloatArray = FloatArray(window * dim)): FloatArray {
            val have = min(count, window)
            val oldest = count - have
            for (i in 0 until window) {
                val src = if (i < window - have) {
                    buf[oldest % window]
                } else {
                    buf[(oldest + i - (window - have)) % window]
                }
                src.copyInto(out, i * dim)
            }
            return out
        }

        fun reset() {
            count = 0
        }
    }

    private class RunningStandardizer(private val dim: Int, private val prior: Int) {
        private val s1 = DoubleArray(dim)
        private val s2 = DoubleArray(dim)
        private var n = 0

        fun apply(x: FloatArray): FloatArray {
            if (n == 0) {
                for (i in 0 until dim) {
                    s1[i] = prior * x[i].toDouble()
                    s2[i] = prior * x[i].toDouble() * x[i]
                }
            }
            n++
            val count = (n + prior).toDouble()
            return FloatArray(dim) { i ->
                s1[i] += x[i]
                s2[i] += x[i].toDouble() * x[i]
                val mean = s1[i] / count
                val std = sqrt(max(s2[i] / count - mean * mean, 0.0) + 1e-2)
                ((x[i] - mean) / std).toFloat()
            }
        }

        fun reset() {
            n = 0
        }
    }

    private class MotionFeatureExtractor(private val cfg: LiveHeadConfig) {
        private var prevGrey: FloatArray? = null
        private var prevFeet: FloatArray? = null

        fun grey(argb: IntArray): FloatArray {
            val w = cfg.det_w
            val gw = cfg.grey_w
            val gh = cfg.grey_h
            val fx = w / gw
            val fy = cfg.det_h / gh
            val out = FloatArray(gw * gh)
            val area = fx * fy
            for (gy in 0 until gh) {
                for (gx in 0 until gw) {
                    var sum = 0
                    for (y in gy * fy until (gy + 1) * fy) {
                        for (x in gx * fx until (gx + 1) * fx) {
                            val p = argb[y * w + x]
                            sum += ((p ushr 16 and 0xFF) * 4899 +
                                (p ushr 8 and 0xFF) * 9617 +
                                (p and 0xFF) * 1868 + 8192) ushr 14
                        }
                    }
                    val q = sum / area
                    val r = sum % area
                    out[gy * gw + gx] = (if (2 * r > area || (2 * r == area && q % 2 == 1)) q + 1 else q).toFloat()
                }
            }
            return out
        }

        fun push(boxes: FloatArray, grey: FloatArray): FloatArray {
            val pg = prevGrey
            val cam = if (pg != null) cameraShift(pg, grey, cfg.grey_w, cfg.grey_h, cfg.coarse) else 0f to 0f
            val (features, feet) = stepFeatures(boxes, prevFeet, cam, cfg.match_heights, cfg.moving_heights)
            prevGrey = grey
            prevFeet = feet
            return features
        }

        fun reset() {
            prevGrey = null
            prevFeet = null
        }

        companion object {
            const val N_FEATURES = 13

            private fun bestShift(a: FloatArray, b: FloatArray, w: Int, h: Int, cx: Int, cy: Int, r: Int): Pair<Int, Int> {
                var best = Double.MAX_VALUE
                var bx = cx
                var by = cy
                for (dy in cy - r..cy + r) {
                    for (dx in cx - r..cx + r) {
                        val ax0 = max(0, -dx)
                        val ax1 = min(w, w - dx)
                        val ay0 = max(0, -dy)
                        val ay1 = min(h, h - dy)
                        if (ax1 - ax0 < w / 2 || ay1 - ay0 < h / 2) continue
                        var s = 0.0
                        for (y in ay0 until ay1) {
                            val ra = y * w
                            val rb = (y + dy) * w + dx
                            for (x in ax0 until ax1) s += abs(a[ra + x] - b[rb + x])
                        }
                        val e = s / ((ax1 - ax0) * (ay1 - ay0))
                        if (e < best) {
                            best = e
                            bx = dx
                            by = dy
                        }
                    }
                }
                return bx to by
            }

            private fun half(a: FloatArray, w: Int, h: Int): Triple<FloatArray, Int, Int> {
                val hw = w / 2
                val hh = h / 2
                val out = FloatArray(hw * hh)
                for (y in 0 until hh) {
                    for (x in 0 until hw) {
                        val i = 2 * y * w + 2 * x
                        out[y * hw + x] = (a[i] + a[i + 1] + a[i + w] + a[i + w + 1]) / 4f
                    }
                }
                return Triple(out, hw, hh)
            }

            private fun cameraShift(prev: FloatArray, cur: FloatArray, w: Int, h: Int, coarse: Int): Pair<Float, Float> {
                val (p2, w2, h2) = half(prev, w, h)
                val (c2) = half(cur, w, h)
                val (p4, w4, h4) = half(p2, w2, h2)
                val (c4) = half(c2, w2, h2)
                var (dx, dy) = bestShift(p4, c4, w4, h4, 0, 0, coarse)
                bestShift(p2, c2, w2, h2, 2 * dx, 2 * dy, 2).let { dx = it.first; dy = it.second }
                bestShift(prev, cur, w, h, 2 * dx, 2 * dy, 2).let { dx = it.first; dy = it.second }
                return dx.toFloat() / w to dy.toFloat() / h
            }

            private fun quantile(v: DoubleArray, q: Double): Double {
                val s = v.sortedArray()
                val pos = (s.size - 1) * q
                val lo = pos.toInt()
                val hi = min(lo + 1, s.size - 1)
                return s[lo] + (s[hi] - s[lo]) * (pos - lo)
            }

            private fun stepFeatures(
                boxes: FloatArray,
                prevFeet: FloatArray?,
                cam: Pair<Float, Float>,
                matchHeights: Float,
                movingHeights: Float,
            ): Pair<FloatArray, FloatArray> {
                val n = boxes.size / 4
                val out = FloatArray(N_FEATURES)
                out[0] = ln(1.0 + n).toFloat()
                out[8] = hypot(cam.first, cam.second)
                val feet = FloatArray(2 * n) { i ->
                    if (i % 2 == 0) (boxes[4 * (i / 2)] + boxes[4 * (i / 2) + 2]) / 2f else boxes[4 * (i / 2) + 3]
                }
                if (n == 0) return out to feet
                val h = quantile(DoubleArray(n) { (boxes[4 * it + 3] - boxes[4 * it + 1]).toDouble() }, 0.5)
                out[1] = h.toFloat()
                val scale = max(h, 1e-3)
                var mx = 0.0
                var my = 0.0
                for (i in 0 until n) {
                    mx += feet[2 * i]
                    my += feet[2 * i + 1]
                }
                mx /= n
                my /= n
                var vx = 0.0
                var vy = 0.0
                for (i in 0 until n) {
                    val dx = feet[2 * i] - mx
                    val dy = feet[2 * i + 1] - my
                    vx += dx * dx
                    vy += dy * dy
                }
                out[2] = (sqrt(vx / n + vy / n) / scale).toFloat()
                val d = Array(n) { i ->
                    DoubleArray(n) { j ->
                        hypot((feet[2 * i] - feet[2 * j]).toDouble(), (feet[2 * i + 1] - feet[2 * j + 1]).toDouble()) / scale
                    }
                }
                if (n > 1) {
                    for (i in 0 until n) d[i][i] = Double.POSITIVE_INFINITY
                    out[3] = quantile(DoubleArray(n) { i -> d[i].min() }, 0.5).toFloat()
                }
                var near = 0
                for (i in 0 until n) {
                    if (hypot(feet[2 * i] - mx, feet[2 * i + 1] - my) / scale < 2.0) near++
                }
                out[4] = near.toFloat() / n
                val seen = BooleanArray(n)
                var best = 0
                for (s in 0 until n) {
                    if (seen[s]) continue
                    val stack = ArrayDeque(listOf(s))
                    seen[s] = true
                    var size = 0
                    while (stack.isNotEmpty()) {
                        val k = stack.removeLast()
                        size++
                        for (m in 0 until n) {
                            if (!seen[m] && d[k][m] < 1.5) {
                                seen[m] = true
                                stack.addLast(m)
                            }
                        }
                    }
                    best = max(best, size)
                }
                out[5] = best.toFloat() / n
                out[6] = (0 until n).count { feet[2 * it] < 1f / 3f }.toFloat() / n
                out[7] = (0 until n).count { feet[2 * it] > 2f / 3f }.toFloat() / n
                if (prevFeet != null && prevFeet.isNotEmpty() && h > 0) {
                    val m = prevFeet.size / 2
                    val dist = DoubleArray(n) { i ->
                        var b = Double.MAX_VALUE
                        for (j in 0 until m) {
                            val e = hypot(
                                (feet[2 * i] - (prevFeet[2 * j] + cam.first)).toDouble(),
                                (feet[2 * i + 1] - (prevFeet[2 * j + 1] + cam.second)).toDouble(),
                            )
                            if (e < b) b = e
                        }
                        b / h
                    }
                    val ok = dist.filter { it < matchHeights }.toDoubleArray()
                    if (ok.size >= 2) {
                        out[9] = quantile(ok, 0.5).toFloat()
                        out[10] = quantile(ok, 0.75).toFloat()
                        out[11] = ok.count { it > movingHeights }.toFloat() / ok.size
                        out[12] = ok.size.toFloat() / n
                    }
                }
                return out to feet
            }
        }
    }

    companion object {
        private const val CANONICAL_FEAT_W = 960
        private const val CANONICAL_FEAT_H = 540
        private const val ICE_WB_COLS = 16
        private const val ICE_WB_ROWS = 12
        private const val ICE_DELTA_E_SQ_THRESHOLD = 121f // deltaE < 11
        private const val ICE_CTX_MIN_L = 55f
        private const val ICE_CTX_MAX_CHROMA = 22f
        private const val SOFT_PLAYER_MIN_CONF = 0.23f
        private const val SOFT_MIN_BOX_HEIGHT = 0.052f
        private const val SOFT_MIN_SURROUND_ICE = 0.56f
        private const val SOFT_MIN_FOOT_ICE = 0.60f
        private const val SOFT_MIN_LOWER_SIDE_ICE = 0.48f
        private const val DFT_W = 64
        private const val DFT_H = 36
        private const val DFT_K_MAX = 15
        private const val STRIPE_ROW_GROUPS = 6
        private val configJson = Json { ignoreUnknownKeys = true }

        private val SRGB_TO_LINEAR = FloatArray(256) { idx ->
            val c = idx / 255f
            if (c <= 0.04045f) c / 12.92f else ((c + 0.055f) / 1.055f).pow(2.4f)
        }

        private val HANN_WINDOW = FloatArray(DFT_W) { c ->
            (0.5 - 0.5 * cos((2.0 * PI * c) / (DFT_W - 1))).toFloat()
        }

        private val HANN_SUM_SQ: Float = run {
            var sum = 0f
            for (v in HANN_WINDOW) sum += v * v
            max(1e-6f, sum)
        }

        private val DFT_COS = Array(DFT_K_MAX + 1) { k ->
            FloatArray(DFT_W) { c -> cos((2.0 * PI * k * c) / DFT_W).toFloat() }
        }

        private val DFT_SIN = Array(DFT_K_MAX + 1) { k ->
            FloatArray(DFT_W) { c -> sin((2.0 * PI * k * c) / DFT_W).toFloat() }
        }

        private fun labF(t: Float): Float =
            if (t > 0.0088564516f) cbrt(t) else 7.787037f * t + 16f / 116f

        private fun pixelToLab(
            argb: Int,
            wbR: Float,
            wbG: Float,
            wbB: Float,
            outLab: FloatArray,
        ) {
            val r = (((argb ushr 16) and 0xFF) * wbR).roundToInt().coerceIn(0, 255)
            val g = (((argb ushr 8) and 0xFF) * wbG).roundToInt().coerceIn(0, 255)
            val b = ((argb and 0xFF) * wbB).roundToInt().coerceIn(0, 255)
            val rl = SRGB_TO_LINEAR[r]
            val gl = SRGB_TO_LINEAR[g]
            val bl = SRGB_TO_LINEAR[b]
            val x = (0.4124564f * rl + 0.3575761f * gl + 0.1804375f * bl) / 0.95047f
            val y = 0.2126729f * rl + 0.7151522f * gl + 0.0721750f * bl
            val z = (0.0193339f * rl + 0.1191920f * gl + 0.9503041f * bl) / 1.08883f
            val fx = labF(x)
            val fy = labF(y)
            val fz = labF(z)
            outLab[0] = 116f * fy - 16f
            outLab[1] = 500f * (fx - fy)
            outLab[2] = 200f * (fy - fz)
        }

        private fun isOrangeArmbandPixel(argb: Int, wbR: Float, wbG: Float, wbB: Float): Boolean {
            val r = (((argb ushr 16) and 0xFF) * wbR).coerceIn(0f, 255f) / 255f
            val g = (((argb ushr 8) and 0xFF) * wbG).coerceIn(0f, 255f) / 255f
            val b = ((argb and 0xFF) * wbB).coerceIn(0f, 255f) / 255f
            val maxC = maxOf(r, g, b)
            if (maxC < 0.40f) return false
            val minC = minOf(r, g, b)
            val delta = maxC - minC
            if (delta <= 1e-5f) return false
            val sat = delta / maxC
            if (sat < 0.45f) return false
            var hue = when (maxC) {
                r -> 60f * (((g - b) / delta) % 6f)
                g -> 60f * (((b - r) / delta) + 2f)
                else -> 60f * (((r - g) / delta) + 4f)
            }
            if (hue < 0f) hue += 360f
            return hue in 4f..26f
        }

        private fun boxIou(a: CandidateBox, b: CandidateBox): Float {
            val iw = min(a.x2, b.x2) - max(a.x1, b.x1)
            val ih = min(a.y2, b.y2) - max(a.y1, b.y1)
            if (iw <= 0f || ih <= 0f) return 0f
            val inter = iw * ih
            val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
            val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
            return inter / max(1e-6f, areaA + areaB - inter)
        }

        private fun intersectionOverMinArea(a: CandidateBox, b: CandidateBox): Float {
            val iw = min(a.x2, b.x2) - max(a.x1, b.x1)
            val ih = min(a.y2, b.y2) - max(a.y1, b.y1)
            if (iw <= 0f || ih <= 0f) return 0f
            val inter = iw * ih
            val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
            val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
            return inter / max(1e-6f, min(areaA, areaB))
        }

        fun load(context: Context, threads: Int = 4): YoloOnnxDetector? = runCatching {
            val cfg = configJson.decodeFromString<LiveHeadConfig>(
                context.assets.open("coach_live.json").bufferedReader().use { it.readText() },
            )
            if (cfg.features.size != MotionFeatureExtractor.N_FEATURES) null
            else YoloOnnxDetector(context, cfg, threads)
        }.getOrNull()
    }
}
