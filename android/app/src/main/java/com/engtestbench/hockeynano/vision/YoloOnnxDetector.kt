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
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
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
)

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
    private val filterPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    private val detPixels = IntArray(detW * detH)
    private val tilePixels = IntArray(imgW * imgH)
    private val inputTensorBuf = FloatArray(3 * imgW * imgH)
    private val directInputBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(4 * 3 * imgW * imgH)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
    private var chestPixelBuf = IntArray(4096)
    private var chestR = IntArray(4096)
    private var chestG = IntArray(4096)
    private var chestB = IntArray(4096)

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

        // 1. Full-frame 640x360 pass letterboxed with 12px 114/255 grey bars to 640x384.
        detCanvas.drawBitmap(highResBitmap, Rect(0, 0, srcW, srcH), Rect(0, 0, detW, detH), filterPaint)
        detBitmap.getPixels(detPixels, 0, detW, 0, 0, detW, detH)
        val fullCandidates = runFullFramePass()

        // 2. Causal 2-fps live-play head update on full-frame layout & 160x90 grey shift.
        if (!lastMotionTimeSec.isNaN() && timestampSec < lastMotionTimeSec - 1.5) {
            motionExtractor.reset()
            featureRing.reset()
            standardizer.reset()
            lastMotionTimeSec = Double.NaN
        }
        if (lastMotionTimeSec.isNaN() || timestampSec - lastMotionTimeSec >= 0.45) {
            lastLivePlayProb = stepLiveHead(fullCandidates)
            lastMotionTimeSec = timestampSec
        }

        // 3. Optional panoramic far-ice overlapping tiles on wide aspect ratio (>= 1.65, covers 16:9 & 2.25:1).
        val aspectRatio = srcW.toFloat() / srcH.toFloat()
        val useTiles = panoramicTiling && aspectRatio >= 1.65f
        val allCandidates = ArrayList<CandidateBox>(fullCandidates.size + 32)
        allCandidates.addAll(fullCandidates)
        if (useTiles) {
            allCandidates.addAll(runTilePass(highResBitmap, 0.08f, 0.20f, 0.56f, 0.72f))
            allCandidates.addAll(runTilePass(highResBitmap, 0.44f, 0.20f, 0.92f, 0.72f))
        }

        // 4. Class-agnostic NMS (IoU > 0.48), preferring unclipped boxes on tile seams.
        val kept = nmsMerge(allCandidates, iouThreshold = 0.48f)

        // 5. Extract upper-chest (x: 25%..75%, y: 20%..52%) mean RGB from the high-res bitmap.
        val detections = kept.map { cand ->
            val rect = RectNorm(cand.x1, cand.y1, cand.x2, cand.y2)
            val rgb = extractChestRgb(highResBitmap, rect)
            YoloDetection(box = rect, confidence = cand.confidence, chestRgb = rgb)
        }

        val elapsedMs = (System.nanoTime() - tStartNs) / 1_000_000f
        YoloFrameResult(
            detections = detections,
            livePlayProb = lastLivePlayProb,
            detectorMs = elapsedMs,
            tiledPassUsed = useTiles,
        )
    }

    private fun runFullFramePass(): List<CandidateBox> {
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
            if (r.size < 6 || r[4] < config.min_conf || r[5] != 0f) continue
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
            if (r.size < 6 || r[4] < config.min_conf || r[5] != 0f) continue
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
                if (iou > iouThreshold) {
                    suppressed = true
                    break
                }
                if (cand.clippedEdge && intersectionOverMinArea(cand, k) > 0.65f) {
                    suppressed = true
                    break
                }
            }
            if (!suppressed) kept.add(cand)
        }
        return kept
    }

    private fun extractChestRgb(bitmap: Bitmap, box: RectNorm): List<Int> {
        val w = bitmap.width
        val h = bitmap.height
        // Core inner torso (34%..66% width, 26%..48% height) avoids background ice/glare edges.
        val x0 = ((box.x1 + box.width * 0.34f) * w).toInt().coerceIn(0, w - 1)
        val x1 = ((box.x1 + box.width * 0.66f) * w).toInt().coerceIn(x0 + 1, w)
        val y0 = ((box.y1 + box.height * 0.26f) * h).toInt().coerceIn(0, h - 1)
        val y1 = ((box.y1 + box.height * 0.48f) * h).toInt().coerceIn(y0 + 1, h)
        val cw = x1 - x0
        val ch = y1 - y0
        val count = cw * ch
        if (count <= 0) return listOf(128, 128, 128)
        if (chestPixelBuf.size < count) {
            chestPixelBuf = IntArray(count)
            chestR = IntArray(count)
            chestG = IntArray(count)
            chestB = IntArray(count)
        }
        bitmap.getPixels(chestPixelBuf, 0, cw, x0, y0, cw, ch)
        for (i in 0 until count) {
            val p = chestPixelBuf[i]
            chestR[i] = (p ushr 16) and 0xFF
            chestG[i] = (p ushr 8) and 0xFF
            chestB[i] = p and 0xFF
        }
        chestR.sort(0, count)
        chestG.sort(0, count)
        chestB.sort(0, count)

        val lo = if (count >= 4) count / 4 else 0
        val hi = if (count >= 4) max(lo + 1, (count * 3) / 4) else count
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

    override fun close() {
        runCatching { detectorSession.close() }
        runCatching { liveHeadSession.close() }
        runCatching { detectorOptions.close() }
        runCatching { liveHeadOptions.close() }
        runCatching { detBitmap.recycle() }
        runCatching { tileBitmap.recycle() }
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
        private val configJson = Json { ignoreUnknownKeys = true }

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
