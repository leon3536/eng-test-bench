package com.engtestbench.hockeynano.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import com.engtestbench.hockeynano.core.model.RectNorm
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import java.io.Closeable
import kotlin.coroutines.resume
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class JerseyOcrResult(
    val jerseyNumber: String,
    val confidence: Float,
    val passLabel: String,
    val xRel: Float = 0f,
)

/**
 * Dual-pass on-device jersey number recognizer using ML Kit TextRecognition (16.0.1).
 * Crops upper torso (10%..68% height, -10%..+10% width) from the high-resolution frame,
 * applies ice white balance, upscales to >= 220px height, normalizes sports-font
 * digit look-alikes (never captain A/C), and returns normalized horizontal position [xRel].
 */
class JerseyNumberRecognizer : Closeable {

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    fun cropUpperTorso(highResBitmap: Bitmap, box: RectNorm): Bitmap? {
        if (highResBitmap.isRecycled) return null
        val w = highResBitmap.width
        val h = highResBitmap.height
        if (w <= 1 || h <= 1) return null

        val bw = box.width
        val bh = box.height
        val x0 = ((box.x1 - 0.10f * bw) * w).toInt().coerceIn(0, w - 1)
        val x1 = ((box.x2 + 0.10f * bw) * w).toInt().coerceIn(x0 + 1, w)
        val y0 = ((box.y1 + 0.10f * bh) * h).toInt().coerceIn(0, h - 1)
        val y1 = ((box.y1 + 0.68f * bh) * h).toInt().coerceIn(y0 + 1, h)
        val cw = x1 - x0
        val ch = y1 - y0
        if (cw < 10 || ch < 14) return null

        return Bitmap.createBitmap(highResBitmap, x0, y0, cw, ch)
    }

    suspend fun recognize(
        highResBitmap: Bitmap,
        box: RectNorm,
        iceWb: FloatArray = DEFAULT_WB,
    ): JerseyOcrResult? {
        val crop = cropUpperTorso(highResBitmap, box) ?: return null
        return try {
            recognizeFromTorsoCrop(crop, iceWb)
        } finally {
            if (!crop.isRecycled) crop.recycle()
        }
    }

    suspend fun recognizeFromTorsoCrop(
        torsoCrop: Bitmap,
        iceWb: FloatArray = DEFAULT_WB,
    ): JerseyOcrResult? {
        if (torsoCrop.isRecycled || torsoCrop.width < 10 || torsoCrop.height < 14) return null

        val targetH = max(torsoCrop.height, MIN_UPSCALED_HEIGHT).coerceAtMost(340)
        val scale = targetH.toFloat() / torsoCrop.height.toFloat()
        val targetW = (torsoCrop.width * scale).toInt().coerceIn(20, 380)

        val upscaled = if (torsoCrop.width == targetW && torsoCrop.height == targetH) {
            torsoCrop.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            Bitmap.createScaledBitmap(torsoCrop, targetW, targetH, true)
        }

        val srcPixels = IntArray(targetW * targetH)
        upscaled.getPixels(srcPixels, 0, targetW, 0, 0, targetW, targetH)
        if (upscaled !== torsoCrop && !upscaled.isRecycled) {
            upscaled.recycle()
        }

        val wbR = iceWb.getOrElse(0) { 1f }.coerceIn(0.6f, 1.6f)
        val wbG = iceWb.getOrElse(1) { 1f }.coerceIn(0.6f, 1.6f)
        val wbB = iceWb.getOrElse(2) { 1f }.coerceIn(0.6f, 1.6f)
        if (abs(wbR - 1f) > 1e-3f || abs(wbG - 1f) > 1e-3f || abs(wbB - 1f) > 1e-3f) {
            for (i in srcPixels.indices) {
                val p = srcPixels[i]
                val r = (((p ushr 16) and 0xFF) * wbR).roundToInt().coerceIn(0, 255)
                val g = (((p ushr 8) and 0xFF) * wbG).roundToInt().coerceIn(0, 255)
                val b = ((p and 0xFF) * wbB).roundToInt().coerceIn(0, 255)
                srcPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        val pass1Bmp = addQuietBorder(buildSmoothRgbBitmap(srcPixels, targetW, targetH))
        val pass2Bmp = addQuietBorder(buildContrastStretchedBitmap(srcPixels, targetW, targetH))

        return try {
            val cand1 = runMlKitPass(pass1Bmp, targetW, targetH)
            val cand2 = runMlKitPass(pass2Bmp, targetW, targetH)

            when {
                cand1 != null && cand2 != null && cand1.jerseyNumber == cand2.jerseyNumber -> {
                    JerseyOcrResult(
                        jerseyNumber = cand1.jerseyNumber,
                        confidence = min(0.98f, max(cand1.confidence, cand2.confidence) + 0.10f),
                        passLabel = "dual_agree",
                        xRel = 0.5f * (cand1.xRel + cand2.xRel),
                    )
                }
                cand1 != null && cand2 != null -> {
                    if (cand2.confidence >= cand1.confidence) {
                        JerseyOcrResult(cand2.jerseyNumber, cand2.confidence, "contrast", cand2.xRel)
                    } else {
                        JerseyOcrResult(cand1.jerseyNumber, cand1.confidence, "smooth_rgb", cand1.xRel)
                    }
                }
                cand2 != null -> JerseyOcrResult(cand2.jerseyNumber, cand2.confidence, "contrast", cand2.xRel)
                cand1 != null -> JerseyOcrResult(cand1.jerseyNumber, cand1.confidence, "smooth_rgb", cand1.xRel)
                else -> null
            }
        } finally {
            if (!pass1Bmp.isRecycled) pass1Bmp.recycle()
            if (!pass2Bmp.isRecycled) pass2Bmp.recycle()
        }
    }

    private data class Candidate(
        val jerseyNumber: String,
        val confidence: Float,
        val boxHeight: Int,
        val xRel: Float,
    )

    private suspend fun runMlKitPass(bitmap: Bitmap, cropWidth: Int, cropHeight: Int): Candidate? {
        val textResult: Text? = suspendCancellableCoroutine { cont ->
            recognizer.process(InputImage.fromBitmap(bitmap, 0))
                .addOnSuccessListener { result ->
                    if (cont.isActive) cont.resume(result)
                }
                .addOnFailureListener {
                    if (cont.isActive) cont.resume(null)
                }
        }
        if (textResult == null) return null

        var best: Candidate? = null
        for (block in textResult.textBlocks) {
            for (line in block.lines) {
                val letterCount = line.text.count { it.isLetter() }
                if (letterCount > 3) continue
                var matchedElement = false
                for (elem in line.elements) {
                    val normalized = normalizeSportsFontToken(elem.text) ?: continue
                    val value = normalized.toIntOrNull() ?: continue
                    if (value !in 1..99) continue

                    val box = elem.boundingBox
                    val boxH = box?.height() ?: 0
                    val rawConf = elem.confidence
                    val baseConf = if (!rawConf.isNaN() && rawConf > 0f) {
                        rawConf.coerceIn(0f, 1f)
                    } else {
                        (0.50f + 0.28f * (boxH.toFloat() / max(1, cropHeight)).coerceIn(0f, 1f))
                    }
                    // Slight discount if token required sports-font letter->digit normalization.
                    val hadConfusion = elem.text.any { it in CONFUSION_CHARS }
                    val conf = if (hadConfusion) baseConf * 0.90f else baseConf
                    if (conf < MIN_CONFIDENCE) continue

                    val elemCenterX = if (box != null) {
                        box.exactCenterX() - QUIET_BORDER_PX
                    } else {
                        cropWidth * 0.5f
                    }
                    val xRel = ((elemCenterX - cropWidth / 2f) / max(1f, cropWidth.toFloat())).coerceIn(-0.5f, 0.5f)

                    matchedElement = true
                    val candidate = Candidate(value.toString(), conf, boxH, xRel)
                    if (best == null ||
                        candidate.confidence > best.confidence + 0.04f ||
                        (abs(candidate.confidence - best.confidence) <= 0.04f && candidate.boxHeight > best.boxHeight)
                    ) {
                        best = candidate
                    }
                }
                // Fallback: extract standalone 1..99 tokens from line.text if element tokenization split digits/symbols.
                if (!matchedElement && !line.text.contains(':') && !line.text.contains('/')) {
                    val lineBox = line.boundingBox
                    val lineBoxH = lineBox?.height() ?: 0
                    val lineCenterX = if (lineBox != null) {
                        lineBox.exactCenterX() - QUIET_BORDER_PX
                    } else {
                        cropWidth * 0.5f
                    }
                    val lineXRel = ((lineCenterX - cropWidth / 2f) / max(1f, cropWidth.toFloat())).coerceIn(-0.5f, 0.5f)
                    for (m in LINE_FALLBACK_REGEX.findAll(line.text)) {
                        val value = m.groupValues[1].toIntOrNull() ?: continue
                        if (value !in 1..99) continue
                        val lineConf = (0.44f + 0.25f * (lineBoxH.toFloat() / max(1, cropHeight)).coerceIn(0f, 1f))
                        if (lineConf < MIN_CONFIDENCE) continue
                        val candidate = Candidate(value.toString(), lineConf, lineBoxH, lineXRel)
                        if (best == null ||
                            candidate.confidence > best.confidence + 0.04f ||
                            (abs(candidate.confidence - best.confidence) <= 0.04f && candidate.boxHeight > best.boxHeight)
                        ) {
                            best = candidate
                        }
                    }
                }
            }
        }
        return best
    }

    private fun buildSmoothRgbBitmap(pixels: IntArray, w: Int, h: Int): Bitmap =
        Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)

    private fun buildContrastStretchedBitmap(pixels: IntArray, w: Int, h: Int): Bitmap {
        val lum = IntArray(pixels.size)
        val hist = IntArray(256)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            val l = ((77 * r + 150 * g + 29 * b) ushr 8).coerceIn(0, 255)
            lum[i] = l
            hist[l]++
        }

        val total = pixels.size
        val lowCut = max(1, (total * 0.02f).toInt())
        val highCut = max(lowCut + 1, (total * 0.98f).toInt())
        var acc = 0
        var pLow = 0
        var pHigh = 255
        var foundLow = false
        for (bin in 0..255) {
            acc += hist[bin]
            if (!foundLow && acc >= lowCut) {
                pLow = bin
                foundLow = true
            }
            if (acc >= highCut) {
                pHigh = bin
                break
            }
        }
        val span = max(20, pHigh - pLow)

        val out = IntArray(pixels.size)
        for (i in lum.indices) {
            val v = (((lum[i] - pLow) * 255) / span).coerceIn(0, 255)
            out[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun addQuietBorder(inner: Bitmap): Bitmap {
        val out = Bitmap.createBitmap(
            inner.width + QUIET_BORDER_PX * 2,
            inner.height + QUIET_BORDER_PX * 2,
            Bitmap.Config.ARGB_8888,
        )
        val canvas = Canvas(out)
        canvas.drawColor(Color.rgb(122, 126, 132))
        canvas.drawBitmap(inner, QUIET_BORDER_PX.toFloat(), QUIET_BORDER_PX.toFloat(), null)
        if (!inner.isRecycled) inner.recycle()
        return out
    }

    override fun close() {
        runCatching { recognizer.close() }
    }

    companion object {
        private const val MIN_UPSCALED_HEIGHT = 220
        private const val QUIET_BORDER_PX = 24
        private const val MIN_CONFIDENCE = 0.36f
        private val DEFAULT_WB = floatArrayOf(1f, 1f, 1f)
        private val LINE_FALLBACK_REGEX = Regex("(?<![\\d.])#?0?([1-9]\\d?)(?![\\d.])")
        private val CONFUSION_CHARS = setOf('O', 'o', 'I', 'l', '|', 'Z', 'z', 'S', 's', 'B')
        private val STRONG_LOOKALIKES = setOf('I', 'l', '|')

        /**
         * Normalizes sports-font OCR confusions (`O->0, I/l/|->1, Z->2, S->5, B->8`) into `1..99`
         * while never mapping captain letters `A`/`C`/`K` or multi-letter words without real digits.
         */
        internal fun normalizeSportsFontToken(raw: String): String? {
            var s = raw.trim()
            if (s.startsWith("#")) s = s.drop(1).trim()
            if (s.isEmpty() || s.length > 3) return null

            var hasRealDigit = false
            val out = StringBuilder(2)
            val nonDigitChars = ArrayList<Char>(2)
            for (ch in s) {
                when {
                    ch in '0'..'9' -> {
                        hasRealDigit = true
                        out.append(ch)
                    }
                    ch == 'O' || ch == 'o' -> {
                        nonDigitChars.add(ch)
                        out.append('0')
                    }
                    ch == 'I' || ch == 'l' || ch == '|' -> {
                        nonDigitChars.add(ch)
                        out.append('1')
                    }
                    ch == 'Z' || ch == 'z' -> {
                        nonDigitChars.add(ch)
                        out.append('2')
                    }
                    ch == 'S' || ch == 's' -> {
                        nonDigitChars.add(ch)
                        out.append('5')
                    }
                    ch == 'B' -> {
                        nonDigitChars.add(ch)
                        out.append('8')
                    }
                    ch == '.' || ch == ',' || ch == '-' -> continue
                    else -> return null
                }
            }

            if (out.isEmpty() || out.length > 2) return null
            if (!hasRealDigit) {
                if (nonDigitChars.size != 1 || nonDigitChars[0] !in STRONG_LOOKALIKES) {
                    return null
                }
            }
            val value = out.toString().toIntOrNull() ?: return null
            return if (value in 1..99) value.toString() else null
        }
    }
}
