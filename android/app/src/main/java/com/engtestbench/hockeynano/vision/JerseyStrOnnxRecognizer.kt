package com.engtestbench.hockeynano.vision

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import com.engtestbench.hockeynano.core.model.RectNorm
import com.engtestbench.hockeynano.core.tracking.JerseyCropGeometry
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs
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
    val orientationDeg: Float = 0f,
    val usedDerotationPass2: Boolean = false,
)

/**
 * On-device Scene Text Recognition (STR) ONNX inference engine (`coach_jersey_str.onnx`).
 * Supports both `v4` (`[1, 3, 96, 96]`, 5 outputs) and `v5` (`[1, 3, 128, 128]`, 6 outputs with
 * `orientation_deg` in `[-60, +60]`, `+theta > 0` = OpenCV CCW rotation), with stick-immune
 * torso-centered cropping and conditional Android-CW upright derotation 2nd pass.
 */
class JerseyStrOnnxRecognizer(context: Context) : Closeable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessionOptions: OrtSession.SessionOptions?
    private val session: OrtSession?
    private val inputName: String
    private val inputSize: Int
    private val hasOrientationOutput: Boolean
    private val minLaplacianSharpness: Float

    private val inferLock = Any()
    private val letterboxBitmap: Bitmap
    private val letterboxCanvas: Canvas
    private val filterPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val letterboxPixels: IntArray
    private val contentLuma: FloatArray
    private val lumaHist = IntArray(256)
    private val inputFloatBuf: FloatArray
    private val directInputBuffer: FloatBuffer

    val isAvailable: Boolean
        get() = session != null

    val modelInputSize: Int
        get() = inputSize

    val supportsOrientation: Boolean
        get() = hasOrientationOutput

    private data class RawStrPass(
        val legibilityProb: Float,
        val backProb: Float,
        val sharpness: Float,
        val orientationDeg: Float,
        val numProbs: FloatArray,
        val leftProbs: FloatArray?,
        val rightProbs: FloatArray?,
    )

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
        inputSize = detectInputSize(sess, inputName)
        hasOrientationOutput = sess?.outputNames?.contains(ORIENTATION_OUTPUT_NAME) == true
        minLaplacianSharpness = if (inputSize >= 128) MIN_LAPLACIAN_SHARPNESS_V5 else MIN_LAPLACIAN_SHARPNESS_V4

        letterboxBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        letterboxCanvas = Canvas(letterboxBitmap)
        letterboxPixels = IntArray(inputSize * inputSize)
        contentLuma = FloatArray(inputSize * inputSize)
        inputFloatBuf = FloatArray(3 * inputSize * inputSize)
        directInputBuffer = ByteBuffer
            .allocateDirect(4 * 3 * inputSize * inputSize)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        if (sess != null) {
            Log.i(
                TAG,
                "STR_INIT loaded=$MODEL_ASSET inputSize=$inputSize hasOrient=$hasOrientationOutput " +
                    "xnnpack=$xnnpack inputs=${sess.inputNames} outputs=${sess.outputNames}",
            )
        }
    }

    fun cropUpperTorso(
        highResBitmap: Bitmap,
        box: RectNorm,
        iceWb: FloatArray = DEFAULT_WB,
    ): Bitmap? = TorsoCropHelper.cropUpperTorso(
        highResBitmap = highResBitmap,
        box = box,
        iceWb = iceWb,
        yStartFrac = JerseyCropGeometry.DEFAULT_TORSO_Y_START,
        yEndFrac = JerseyCropGeometry.DEFAULT_TORSO_Y_END,
        normalXPadFrac = JerseyCropGeometry.DEFAULT_TORSO_X_PAD_FRAC,
        minWidthPx = MIN_CROP_WIDTH_PX,
        minHeightPx = MIN_CROP_HEIGHT_PX,
    )

    fun recognize(
        highResBitmap: Bitmap,
        box: RectNorm,
        iceWb: FloatArray = DEFAULT_WB,
    ): StrInferenceResult? {
        if (session == null) return null
        val crop = cropUpperTorso(highResBitmap, box, iceWb) ?: return null
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
        if (torsoCrop.isRecycled || torsoCrop.width < MIN_CROP_WIDTH_PX || torsoCrop.height < MIN_CROP_HEIGHT_PX) {
            return null
        }

        val wbR = iceWb.getOrElse(0) { 1f }.coerceIn(0.6f, 1.6f)
        val wbG = iceWb.getOrElse(1) { 1f }.coerceIn(0.6f, 1.6f)
        val wbB = iceWb.getOrElse(2) { 1f }.coerceIn(0.6f, 1.6f)

        val pass1 = synchronized(inferLock) {
            runSingleStrPassLocked(activeSession, torsoCrop, wbR, wbG, wbB)
        } ?: return null

        if (pass1.sharpness < minLaplacianSharpness) {
            return StrInferenceResult(
                legibilityProb = 0f,
                backProb = 0f,
                sharpness = pass1.sharpness,
                topCandidates = emptyList(),
                partialLeftDigit = null,
                partialRightDigit = null,
                orientationDeg = 0f,
            )
        }

        val shouldDerotate = hasOrientationOutput &&
            abs(pass1.orientationDeg) >= JerseyCropGeometry.DEROTATE_MIN_ANGLE_DEG &&
            (pass1.legibilityProb >= DEROTATE_MIN_LEGIBILITY_PROB || pass1.backProb >= DEROTATE_MIN_BACK_PROB)

        if (!shouldDerotate) {
            return buildInferenceResult(pass1)
        }

        val uprightCrop = TorsoCropHelper.derotateBitmapUpright(
            src = torsoCrop,
            ccwOrientationDeg = pass1.orientationDeg,
            expandBounds = true,
        )
        val pass2 = try {
            synchronized(inferLock) {
                runSingleStrPassLocked(activeSession, uprightCrop, wbR, wbG, wbB)
            }
        } finally {
            if (!uprightCrop.isRecycled) uprightCrop.recycle()
        }

        val usedPass2 = pass2 != null && pass2.sharpness >= minLaplacianSharpness
        val fused = if (usedPass2) {
            fuseStrPasses(pass1, pass2!!)
        } else {
            pass1
        }
        return buildInferenceResult(fused, usedDerotationPass2 = usedPass2)
    }

    private fun runSingleStrPassLocked(
        activeSession: OrtSession,
        crop: Bitmap,
        wbR: Float,
        wbG: Float,
        wbB: Float,
    ): RawStrPass? {
        val cw = crop.width
        val ch = crop.height
        val size = inputSize
        val scale = min(size.toFloat() / cw, size.toFloat() / ch)
        val dw = (cw * scale).roundToInt().coerceIn(1, size)
        val dh = (ch * scale).roundToInt().coerceIn(1, size)
        val padX = (size - dw) / 2
        val padY = (size - dh) / 2

        if (size >= 128 && (cw < dw || ch < dh)) {
            val srcPixels = IntArray(cw * ch)
            crop.getPixels(srcPixels, 0, cw, 0, 0, cw, ch)
            val resizedPixels = JerseyCropGeometry.resizeBicubic(srcPixels, cw, ch, dw, dh)
            letterboxPixels.fill(Color.rgb(PAD_GREY, PAD_GREY, PAD_GREY))
            if (resizedPixels.size >= dw * dh) {
                for (y in 0 until dh) {
                    System.arraycopy(resizedPixels, y * dw, letterboxPixels, (padY + y) * size + padX, dw)
                }
            }
        } else {
            letterboxCanvas.drawColor(Color.rgb(PAD_GREY, PAD_GREY, PAD_GREY))
            letterboxCanvas.drawBitmap(
                crop,
                Rect(0, 0, cw, ch),
                Rect(padX, padY, padX + dw, padY + dh),
                filterPaint,
            )
            letterboxBitmap.getPixels(letterboxPixels, 0, size, 0, 0, size, size)
        }

        // First pass over content region: compute WB luma histogram for chromatic-preserving stretch & sharpness.
        lumaHist.fill(0)
        var contentCount = 0
        val xEnd = padX + dw
        val yEnd = padY + dh
        for (y in padY until yEnd) {
            val rowOff = y * size
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

        val sharpness = computeContentLaplacianVariance(padX, padY, dw, dh)
        if (sharpness < minLaplacianSharpness) {
            return RawStrPass(
                legibilityProb = 0f,
                backProb = 0f,
                sharpness = sharpness,
                orientationDeg = 0f,
                numProbs = FloatArray(101).apply { this[0] = 1f },
                leftProbs = null,
                rightProbs = null,
            )
        }

        // Mild chromatic-preserving linear stretch (~1.05x..1.25x) around content midpoint.
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

        // Pack NCHW float32 [1, 3, inputSize, inputSize] in [0, 1].
        val plane = size * size
        inputFloatBuf.fill(PAD_GREY / 255f)
        for (y in padY until yEnd) {
            val rowOff = y * size
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

        return try {
            OnnxTensor.createTensor(
                env,
                directInputBuffer,
                longArrayOf(1L, 3L, size.toLong(), size.toLong()),
            ).use { tensor ->
                activeSession.run(mapOf(inputName to tensor)).use { result ->
                    parseRawOutputs(result, sharpness)
                }
            }
        } catch (e: Throwable) {
            Log.w(TAG, "STR_INFER failed: ${e.javaClass.simpleName}: ${e.message}")
            null
        }
    }

    private fun computeContentLaplacianVariance(padX: Int, padY: Int, dw: Int, dh: Int): Float {
        if (dw < 4 || dh < 4) return 1f
        val size = inputSize
        var sum = 0.0
        var sumSq = 0.0
        var count = 0
        val xStart = padX + 1
        val xEnd = padX + dw - 1
        val yStart = padY + 1
        val yEnd = padY + dh - 1
        for (y in yStart until yEnd) {
            val rowOff = y * size
            for (x in xStart until xEnd) {
                val idx = rowOff + x
                val lap = 4.0 * contentLuma[idx] -
                    contentLuma[idx - 1] -
                    contentLuma[idx + 1] -
                    contentLuma[idx - size] -
                    contentLuma[idx + size]
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

    private fun parseRawOutputs(result: OrtSession.Result, sharpness: Float): RawStrPass? {
        val legibilityLogit = extractScalarLogit(result, "legibility_logit") ?: return null
        val backLogit = extractScalarLogit(result, "back_logit") ?: 0f
        val numLogits = extractVectorLogits(result, "number_logits", 101) ?: return null
        val leftLogits = extractVectorLogits(result, "left_digit_logits", 11)
        val rightLogits = extractVectorLogits(result, "right_digit_logits", 11)
        val orientationDeg = if (hasOrientationOutput) {
            (extractScalarLogit(result, ORIENTATION_OUTPUT_NAME)?.takeIf { it.isFinite() } ?: 0f)
                .coerceIn(-60f, 60f)
        } else {
            0f
        }

        return RawStrPass(
            legibilityProb = sigmoid(legibilityLogit),
            backProb = sigmoid(backLogit),
            sharpness = sharpness,
            orientationDeg = orientationDeg,
            numProbs = softmax(numLogits),
            leftProbs = leftLogits?.let { softmax(it) },
            rightProbs = rightLogits?.let { softmax(it) },
        )
    }

    private fun fuseStrPasses(p1: RawStrPass, p2: RawStrPass): RawStrPass {
        val w1 = max(0.02f, p1.legibilityProb * p1.legibilityProb)
        val w2 = max(0.02f, p2.legibilityProb * p2.legibilityProb)
        val wSum = w1 + w2
        val a1 = w1 / wSum
        val a2 = w2 / wSum

        val numProbs = FloatArray(p1.numProbs.size) { i ->
            a1 * p1.numProbs[i] + a2 * p2.numProbs[i]
        }
        val leftProbs = if (p1.leftProbs != null && p2.leftProbs != null) {
            FloatArray(p1.leftProbs.size) { i -> a1 * p1.leftProbs[i] + a2 * p2.leftProbs[i] }
        } else {
            p2.leftProbs ?: p1.leftProbs
        }
        val rightProbs = if (p1.rightProbs != null && p2.rightProbs != null) {
            FloatArray(p1.rightProbs.size) { i -> a1 * p1.rightProbs[i] + a2 * p2.rightProbs[i] }
        } else {
            p2.rightProbs ?: p1.rightProbs
        }

        return RawStrPass(
            legibilityProb = (a1 * p1.legibilityProb + a2 * p2.legibilityProb).coerceIn(0f, 1f),
            backProb = (a1 * p1.backProb + a2 * p2.backProb).coerceIn(0f, 1f),
            sharpness = p1.sharpness,
            orientationDeg = p1.orientationDeg,
            numProbs = numProbs,
            leftProbs = leftProbs,
            rightProbs = rightProbs,
        )
    }

    private fun buildInferenceResult(
        pass: RawStrPass,
        usedDerotationPass2: Boolean = false,
    ): StrInferenceResult {
        val legibilityProb = pass.legibilityProb
        val backProb = pass.backProb
        val numProbs = pass.numProbs
        val leftProbs = pass.leftProbs
        val rightProbs = pass.rightProbs

        // Uniform prior 1..99: extract top legible 1..99 back-view candidates only above crisp confidence floor.
        val topCandidates = ArrayList<Pair<String, Float>>(5)
        if (numProbs[0] < MAX_ILLEGIBLE_PROB &&
            legibilityProb >= MIN_CANDIDATE_LEGIBILITY_PROB &&
            backProb >= MIN_CANDIDATE_BACK_PROB
        ) {
            val indexed = ArrayList<Pair<Int, Float>>(24)
            for (n in 1..99) {
                val p = numProbs[n]
                if (p >= MIN_CANDIDATE_PROB) {
                    indexed.add(n to p)
                }
            }
            indexed.sortByDescending { it.second }
            if (indexed.isNotEmpty() && indexed[0].second >= MIN_TOP_CANDIDATE_PROB) {
                for (i in 0 until min(5, indexed.size)) {
                    val (n, p) = indexed[i]
                    topCandidates.add(n.toString() to p)
                }
            }
        }

        var partialLeft: Pair<Char, Float>? = null
        if (leftProbs != null &&
            legibilityProb >= MIN_PARTIAL_LEGIBILITY_PROB &&
            backProb >= MIN_PARTIAL_BACK_PROB
        ) {
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
                (rightBlankProb >= MIN_BLANK_COMPANION_PROB || legibilityProb < 0.65f) &&
                numProbs[bestD] < 0.25f
            ) {
                partialLeft = ('0' + bestD) to bestP
            }
        }

        var partialRight: Pair<Char, Float>? = null
        if (rightProbs != null &&
            legibilityProb >= MIN_PARTIAL_LEGIBILITY_PROB &&
            backProb >= MIN_PARTIAL_BACK_PROB
        ) {
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
                (leftBlankProb >= MIN_BLANK_COMPANION_PROB || legibilityProb < 0.65f)
            ) {
                partialRight = ('0' + bestD) to bestP
            }
        }

        return StrInferenceResult(
            legibilityProb = legibilityProb,
            backProb = backProb,
            sharpness = pass.sharpness,
            topCandidates = topCandidates,
            partialLeftDigit = partialLeft,
            partialRightDigit = partialRight,
            orientationDeg = pass.orientationDeg,
            usedDerotationPass2 = usedDerotationPass2,
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
        private const val ORIENTATION_OUTPUT_NAME = "orientation_deg"
        private const val DEFAULT_INPUT_SIZE = 96
        private const val PAD_GREY = 114
        private const val MIN_CROP_WIDTH_PX = 12
        private const val MIN_CROP_HEIGHT_PX = 18
        private const val MIN_LAPLACIAN_SHARPNESS_V4 = 18.0f
        private const val MIN_LAPLACIAN_SHARPNESS_V5 = 5.5f
        private const val DEROTATE_MIN_LEGIBILITY_PROB = 0.25f
        private const val DEROTATE_MIN_BACK_PROB = 0.45f
        private const val MIN_CANDIDATE_PROB = 0.02f
        private const val MIN_TOP_CANDIDATE_PROB = 0.32f
        private const val MAX_ILLEGIBLE_PROB = 0.48f
        internal const val MIN_CANDIDATE_LEGIBILITY_PROB = 0.64f
        internal const val MIN_CANDIDATE_BACK_PROB = 0.56f
        private const val MIN_PARTIAL_LEGIBILITY_PROB = 0.55f
        private const val MIN_PARTIAL_BACK_PROB = 0.60f
        private const val MIN_PARTIAL_DIGIT_PROB = 0.65f
        private const val PARTIAL_BLANK_MARGIN = 1.25f
        private const val MIN_BLANK_COMPANION_PROB = 0.32f
        private val DEFAULT_WB = floatArrayOf(1f, 1f, 1f)

        private fun detectInputSize(sess: OrtSession?, inputName: String): Int {
            if (sess == null) return DEFAULT_INPUT_SIZE
            return runCatching {
                val tensorInfo = sess.inputInfo[inputName]?.info as? TensorInfo ?: return DEFAULT_INPUT_SIZE
                val shape = tensorInfo.shape
                if (shape.size >= 4) {
                    val h = shape[2].toInt()
                    val w = shape[3].toInt()
                    if (h == w && h in 64..256) h else DEFAULT_INPUT_SIZE
                } else {
                    DEFAULT_INPUT_SIZE
                }
            }.getOrDefault(DEFAULT_INPUT_SIZE)
        }

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
