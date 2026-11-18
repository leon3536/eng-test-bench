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
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class StrInferenceResult(
    val legibilityProb: Float,
    val backProb: Float,
    val sharpness: Float,
    val topCandidates: List<Pair<String, Float>>,
    val partialLeftDigit: Pair<Char, Float>? = null,
    val partialRightDigit: Pair<Char, Float>? = null,
)

/**
 * On-device Scene Text Recognition (STR) ONNX inference engine (`coach_jersey_str.onnx`).
 * Expects static `[1, 3, 96, 96]` RGB `[0..1]` input (`images`) with ImageNet normalization
 * embedded in the ONNX graph, returning calibrated `legibility_logit`, `back_logit`,
 * `number_logits[1,101]`, `left_digit_logits[1,11]`, and `right_digit_logits[1,11]`.
 *
 * Gracefully degrades (`isAvailable == false`, `recognize*` returns `null`) when the
 * `coach_jersey_str.onnx` asset is not yet bundled.
 */
class JerseyStrOnnxRecognizer(context: Context) : Closeable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessionOptions: OrtSession.SessionOptions?
    private val session: OrtSession?
    private val inputName: String

    private val inferLock = Any()
    private val letterboxBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
    private val letterboxCanvas = Canvas(letterboxBitmap)
    private val filterPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val letterboxPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val contentLuma = FloatArray(INPUT_SIZE * INPUT_SIZE)
    private val lumaHist = IntArray(256)
    private val inputFloatBuf = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
    private val directInputBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(4 * 3 * INPUT_SIZE * INPUT_SIZE)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()

    val isAvailable: Boolean
        get() = session != null

    init {
        var opts: OrtSession.SessionOptions? = null
        var sess: OrtSession? = null
        var xnnpack = false
        try {
            val modelBytes = context.assets.open(MODEL_ASSET).use { it.readBytes() }
            opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                runCatching { addConfigEntry("session.intra_op.allow_spinning", "0") }
                runCatching { addConfigEntry("session.inter_op.allow_spinning", "0") }
                xnnpack = runCatching {
                    addXnnpack(mapOf("intra_op_num_threads" to "2"))
                    true
                }.getOrDefault(false)
            }
            sess = env.createSession(modelBytes, opts)
            Log.i(TAG, "STR_INIT loaded=$MODEL_ASSET xnnpack=$xnnpack inputs=${sess.inputNames} outputs=${sess.outputNames}")
        } catch (e: Throwable) {
            runCatching { sess?.close() }
            runCatching { opts?.close() }
            sess = null
            opts = null
            Log.i(TAG, "STR_INIT unavailable ($MODEL_ASSET): ${e.javaClass.simpleName}: ${e.message}")
        }
        sessionOptions = opts
        session = sess
        inputName = sess?.inputNames?.firstOrNull() ?: "images"
    }

    fun cropUpperTorso(highResBitmap: Bitmap, box: RectNorm): Bitmap? {
        if (highResBitmap.isRecycled) return null
        val w = highResBitmap.width
        val h = highResBitmap.height
        if (w <= 1 || h <= 1) return null

        val bw = box.width
        val bh = box.height
        // Upper torso: x -6%..106%, y 10%..62%
        val x0 = ((box.x1 - 0.06f * bw) * w).toInt().coerceIn(0, w - 1)
        val x1 = ((box.x2 + 0.06f * bw) * w).toInt().coerceIn(x0 + 1, w)
        val y0 = ((box.y1 + 0.10f * bh) * h).toInt().coerceIn(0, h - 1)
        val y1 = ((box.y1 + 0.62f * bh) * h).toInt().coerceIn(y0 + 1, h)
        val cw = x1 - x0
        val ch = y1 - y0
        if (cw < 8 || ch < 10) return null

        return Bitmap.createBitmap(highResBitmap, x0, y0, cw, ch)
    }

    fun recognize(
        highResBitmap: Bitmap,
        box: RectNorm,
        iceWb: FloatArray = DEFAULT_WB,
    ): StrInferenceResult? {
        if (session == null) return null
        val crop = cropUpperTorso(highResBitmap, box) ?: return null
        return try {
            recognizeFromTorsoCrop(crop, iceWb)
        } finally {
            if (!crop.isRecycled) crop.recycle()
        }
    }

    fun recognizeFromTorsoCrop(
        torsoCrop: Bitmap,
        iceWb: FloatArray = DEFAULT_WB,
    ): StrInferenceResult? {
        val activeSession = session ?: return null
        if (torsoCrop.isRecycled || torsoCrop.width < 8 || torsoCrop.height < 10) return null

        val wbR = iceWb.getOrElse(0) { 1f }.coerceIn(0.6f, 1.6f)
        val wbG = iceWb.getOrElse(1) { 1f }.coerceIn(0.6f, 1.6f)
        val wbB = iceWb.getOrElse(2) { 1f }.coerceIn(0.6f, 1.6f)

        return synchronized(inferLock) {
            val cw = torsoCrop.width
            val ch = torsoCrop.height
            val scale = min(INPUT_SIZE.toFloat() / cw, INPUT_SIZE.toFloat() / ch)
            val dw = (cw * scale).roundToInt().coerceIn(1, INPUT_SIZE)
            val dh = (ch * scale).roundToInt().coerceIn(1, INPUT_SIZE)
            val padX = (INPUT_SIZE - dw) / 2
            val padY = (INPUT_SIZE - dh) / 2

            letterboxCanvas.drawColor(Color.rgb(PAD_GREY, PAD_GREY, PAD_GREY))
            letterboxCanvas.drawBitmap(
                torsoCrop,
                Rect(0, 0, cw, ch),
                Rect(padX, padY, padX + dw, padY + dh),
                filterPaint,
            )
            letterboxBitmap.getPixels(letterboxPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

            // First pass over content region: compute WB luma histogram for gentle contrast stretch & sharpness.
            lumaHist.fill(0)
            var contentCount = 0
            val xEnd = padX + dw
            val yEnd = padY + dh
            for (y in padY until yEnd) {
                val rowOff = y * INPUT_SIZE
                for (x in padX until xEnd) {
                    val idx = rowOff + x
                    val p = letterboxPixels[idx]
                    val r = (((p ushr 16) and 0xFF) * wbR).coerceIn(0f, 255f)
                    val g = (((p ushr 8) and 0xFF) * wbG).coerceIn(0f, 255f)
                    val b = ((p and 0xFF) * wbB).coerceIn(0f, 255f)
                    val lum = (0.299f * r + 0.587f * g + 0.114f * b).coerceIn(0f, 255f)
                    contentLuma[idx] = lum
                    lumaHist[lum.roundToInt().coerceIn(0, 255)]++
                    contentCount++
                }
            }

            // Mild local contrast stretch (~1.12x..1.24x) around content midpoint without distorting padding.
            var pLow = 0
            var pHigh = 255
            if (contentCount > 0) {
                val lowCut = max(1, (contentCount * 0.02f).toInt())
                val highCut = max(lowCut + 1, (contentCount * 0.98f).toInt())
                var acc = 0
                var gotLow = false
                for (bin in 0..255) {
                    acc += lumaHist[bin]
                    if (!gotLow && acc >= lowCut) {
                        pLow = bin
                        gotLow = true
                    }
                    if (acc >= highCut) {
                        pHigh = bin
                        break
                    }
                }
            }
            val rawSpan = max(32f, (pHigh - pLow).toFloat())
            val contrastGain = (1f + 0.12f * (220f / rawSpan).coerceIn(0.8f, 2.0f)).coerceIn(1.05f, 1.25f)
            val contentMid = 0.5f * (pLow + pHigh)

            val sharpness = computeContentLaplacianVariance(padX, padY, dw, dh)

            // Pack NCHW float32 [1, 3, 96, 96] in [0, 1].
            val plane = INPUT_SIZE * INPUT_SIZE
            inputFloatBuf.fill(PAD_GREY / 255f)
            for (y in padY until yEnd) {
                val rowOff = y * INPUT_SIZE
                for (x in padX until xEnd) {
                    val idx = rowOff + x
                    val p = letterboxPixels[idx]
                    val rWb = (((p ushr 16) and 0xFF) * wbR).coerceIn(0f, 255f)
                    val gWb = (((p ushr 8) and 0xFF) * wbG).coerceIn(0f, 255f)
                    val bWb = ((p and 0xFF) * wbB).coerceIn(0f, 255f)

                    val rAdj = ((rWb - contentMid) * contrastGain + contentMid).coerceIn(0f, 255f) / 255f
                    val gAdj = ((gWb - contentMid) * contrastGain + contentMid).coerceIn(0f, 255f) / 255f
                    val bAdj = ((bWb - contentMid) * contrastGain + contentMid).coerceIn(0f, 255f) / 255f

                    inputFloatBuf[idx] = rAdj
                    inputFloatBuf[plane + idx] = gAdj
                    inputFloatBuf[2 * plane + idx] = bAdj
                }
            }

            directInputBuffer.clear()
            directInputBuffer.put(inputFloatBuf)
            directInputBuffer.rewind()

            try {
                OnnxTensor.createTensor(
                    env,
                    directInputBuffer,
                    longArrayOf(1L, 3L, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
                ).use { tensor ->
                    activeSession.run(mapOf(inputName to tensor)).use { result ->
                        parseOutputs(result, sharpness)
                    }
                }
            } catch (e: Throwable) {
                Log.w(TAG, "STR_INFER failed: ${e.javaClass.simpleName}: ${e.message}")
                null
            }
        }
    }

    private fun computeContentLaplacianVariance(padX: Int, padY: Int, dw: Int, dh: Int): Float {
        if (dw < 4 || dh < 4) return 1f
        var sum = 0.0
        var sumSq = 0.0
        var count = 0
        val xStart = padX + 1
        val xEnd = padX + dw - 1
        val yStart = padY + 1
        val yEnd = padY + dh - 1
        for (y in yStart until yEnd) {
            val rowOff = y * INPUT_SIZE
            for (x in xStart until xEnd) {
                val idx = rowOff + x
                val lap = 4.0 * contentLuma[idx] -
                    contentLuma[idx - 1] -
                    contentLuma[idx + 1] -
                    contentLuma[idx - INPUT_SIZE] -
                    contentLuma[idx + INPUT_SIZE]
                sum += lap
                sumSq += lap * lap
                count++
            }
        }
        if (count < 4) return 1f
        val mean = sum / count
        val variance = (sumSq / count) - mean * mean
        return variance.toFloat().coerceIn(1f, 2500f)
    }

    private fun parseOutputs(result: OrtSession.Result, sharpness: Float): StrInferenceResult? {
        val legibilityLogit = extractScalarLogit(result, "legibility_logit") ?: return null
        val backLogit = extractScalarLogit(result, "back_logit") ?: 0f
        val numLogits = extractVectorLogits(result, "number_logits", 101) ?: return null
        val leftLogits = extractVectorLogits(result, "left_digit_logits", 11)
        val rightLogits = extractVectorLogits(result, "right_digit_logits", 11)

        val legibilityProb = sigmoid(legibilityLogit)
        val backProb = sigmoid(backLogit)
        val numProbs = softmax(numLogits)
        val leftProbs = leftLogits?.let { softmax(it) }
        val rightProbs = rightLogits?.let { softmax(it) }

        // Uniform prior 1..99: extract top legible 1..99 candidates only when not illegible-dominant.
        val topCandidates = ArrayList<Pair<String, Float>>(5)
        if (numProbs[0] < MAX_ILLEGIBLE_PROB && legibilityProb >= MIN_CANDIDATE_LEGIBILITY_PROB) {
            val indexed = ArrayList<Pair<Int, Float>>(24)
            for (n in 1..99) {
                val p = numProbs[n]
                if (p >= MIN_CANDIDATE_PROB) {
                    indexed.add(n to p)
                }
            }
            indexed.sortByDescending { it.second }
            for (i in 0 until min(5, indexed.size)) {
                val (n, p) = indexed[i]
                topCandidates.add(n.toString() to p)
            }
        }

        var partialLeft: Pair<Char, Float>? = null
        if (leftProbs != null && legibilityProb >= MIN_PARTIAL_LEGIBILITY_PROB) {
            val leftBlankProb = leftProbs[10]
            val rightBlankProb = rightProbs?.get(10) ?: 0f
            var bestD = -1
            var bestP = 0f
            // Tens/only digit in 1..9 for valid hockey numbers 1..99.
            for (d in 1..9) {
                if (leftProbs[d] > bestP) {
                    bestP = leftProbs[d]
                    bestD = d
                }
            }
            if (bestD in 1..9 &&
                bestP >= MIN_PARTIAL_DIGIT_PROB &&
                bestP > leftBlankProb * PARTIAL_BLANK_MARGIN &&
                (rightBlankProb >= MIN_BLANK_COMPANION_PROB || legibilityProb < 0.55f) &&
                numProbs[bestD] < 0.25f
            ) {
                partialLeft = ('0' + bestD) to bestP
            }
        }

        var partialRight: Pair<Char, Float>? = null
        if (rightProbs != null && legibilityProb >= MIN_PARTIAL_LEGIBILITY_PROB) {
            val rightBlankProb = rightProbs[10]
            val leftBlankProb = leftProbs?.get(10) ?: 0f
            var bestD = -1
            var bestP = 0f
            // Ones digit can be 0..9 (e.g., ?0 for #10, #20, #30).
            for (d in 0..9) {
                if (rightProbs[d] > bestP) {
                    bestP = rightProbs[d]
                    bestD = d
                }
            }
            if (bestD in 0..9 &&
                bestP >= MIN_PARTIAL_DIGIT_PROB &&
                bestP > rightBlankProb * PARTIAL_BLANK_MARGIN &&
                (leftBlankProb >= MIN_BLANK_COMPANION_PROB || legibilityProb < 0.60f)
            ) {
                partialRight = ('0' + bestD) to bestP
            }
        }

        return StrInferenceResult(
            legibilityProb = legibilityProb,
            backProb = backProb,
            sharpness = sharpness,
            topCandidates = topCandidates,
            partialLeftDigit = partialLeft,
            partialRightDigit = partialRight,
        )
    }

    private fun extractScalarLogit(result: OrtSession.Result, name: String): Float? {
        val rawVal = result.get(name).orElse(null)?.value ?: return null
        return when (rawVal) {
            is Array<*> -> (rawVal.firstOrNull() as? FloatArray)?.firstOrNull()
            is FloatArray -> rawVal.firstOrNull()
            is Float -> rawVal
            else -> null
        }
    }

    private fun extractVectorLogits(result: OrtSession.Result, name: String, expectedLen: Int): FloatArray? {
        val rawVal = result.get(name).orElse(null)?.value ?: return null
        val arr = when (rawVal) {
            is Array<*> -> rawVal.firstOrNull() as? FloatArray
            is FloatArray -> rawVal
            else -> null
        } ?: return null
        return if (arr.size >= expectedLen) arr else null
    }

    override fun close() {
        runCatching { session?.close() }
        runCatching { sessionOptions?.close() }
        runCatching { letterboxBitmap.recycle() }
    }

    companion object {
        private const val TAG = "HockeyNanoBench"
        private const val MODEL_ASSET = "coach_jersey_str.onnx"
        private const val INPUT_SIZE = 96
        private const val PAD_GREY = 114
        private const val MIN_CANDIDATE_PROB = 0.02f
        private const val MAX_ILLEGIBLE_PROB = 0.60f
        private const val MIN_CANDIDATE_LEGIBILITY_PROB = 0.55f
        private const val MIN_PARTIAL_LEGIBILITY_PROB = 0.30f
        private const val MIN_PARTIAL_DIGIT_PROB = 0.45f
        private const val PARTIAL_BLANK_MARGIN = 1.15f
        private const val MIN_BLANK_COMPANION_PROB = 0.28f
        private val DEFAULT_WB = floatArrayOf(1f, 1f, 1f)

        fun load(context: Context): JerseyStrOnnxRecognizer? =
            JerseyStrOnnxRecognizer(context).takeIf { it.isAvailable }

        fun computeLaplacianSharpness(bitmap: Bitmap): Float {
            if (bitmap.isRecycled || bitmap.width < 4 || bitmap.height < 4) return 1f
            val w = min(bitmap.width, 96)
            val h = min(bitmap.height, 96)
            val bmp = if (bitmap.width == w && bitmap.height == h) {
                bitmap
            } else {
                Bitmap.createScaledBitmap(bitmap, w, h, true)
            }
            val px = IntArray(w * h)
            bmp.getPixels(px, 0, w, 0, 0, w, h)
            if (bmp !== bitmap && !bmp.isRecycled) bmp.recycle()

            val lum = FloatArray(w * h)
            for (i in px.indices) {
                val p = px[i]
                lum[i] = 0.299f * ((p ushr 16) and 0xFF) +
                    0.587f * ((p ushr 8) and 0xFF) +
                    0.114f * (p and 0xFF)
            }
            var sum = 0.0
            var sumSq = 0.0
            var count = 0
            for (y in 1 until h - 1) {
                val rowOff = y * w
                for (x in 1 until w - 1) {
                    val idx = rowOff + x
                    val lap = 4.0 * lum[idx] - lum[idx - 1] - lum[idx + 1] - lum[idx - w] - lum[idx + w]
                    sum += lap
                    sumSq += lap * lap
                    count++
                }
            }
            if (count < 4) return 1f
            val mean = sum / count
            return ((sumSq / count) - mean * mean).toFloat().coerceIn(1f, 2500f)
        }

        private fun sigmoid(x: Float): Float {
            val xc = x.coerceIn(-20f, 20f)
            return 1f / (1f + exp(-xc))
        }

        private fun softmax(logits: FloatArray): FloatArray {
            var maxVal = Float.NEGATIVE_INFINITY
            for (v in logits) {
                if (v > maxVal) maxVal = v
            }
            val out = FloatArray(logits.size)
            var sum = 0.0
            for (i in logits.indices) {
                val e = exp((logits[i] - maxVal).coerceIn(-40f, 0f).toDouble())
                out[i] = e.toFloat()
                sum += e
            }
            val inv = if (sum > 0.0) (1.0 / sum).toFloat() else 0f
            for (i in out.indices) {
                out[i] *= inv
            }
            return out
        }
    }
}
