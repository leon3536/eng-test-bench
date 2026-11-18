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
import kotlin.math.max
import kotlin.math.min

data class JerseyOcrResult(
    val jerseyNumber: String,
    val confidence: Float,
    val passLabel: String,
)

/**
 * Dual-pass on-device jersey number recognizer using ML Kit TextRecognition (16.0.1).
 * Crops upper torso (10%..68% height, -10%..+10% width) from the high-resolution frame,
 * upscales to >= 180px height, and evaluates both sharpened RGB and contrast-stretched luminance.
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

    suspend fun recognize(highResBitmap: Bitmap, box: RectNorm): JerseyOcrResult? {
        val crop = cropUpperTorso(highResBitmap, box) ?: return null
        return try {
            recognizeFromTorsoCrop(crop)
        } finally {
            if (!crop.isRecycled) crop.recycle()
        }
    }

    suspend fun recognizeFromTorsoCrop(torsoCrop: Bitmap): JerseyOcrResult? {
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

        val pass1Bmp = addQuietBorder(buildSmoothRgbBitmap(srcPixels, targetW, targetH))
        val pass2Bmp = addQuietBorder(buildContrastStretchedBitmap(srcPixels, targetW, targetH))

        return try {
            val cand1 = runMlKitPass(pass1Bmp, targetH)
            val cand2 = runMlKitPass(pass2Bmp, targetH)

            when {
                cand1 != null && cand2 != null && cand1.jerseyNumber == cand2.jerseyNumber -> {
                    JerseyOcrResult(
                        jerseyNumber = cand1.jerseyNumber,
                        confidence = min(0.98f, max(cand1.confidence, cand2.confidence) + 0.10f),
                        passLabel = "dual_agree",
                    )
                }
                cand1 != null && cand2 != null -> {
                    if (cand2.confidence >= cand1.confidence) {
                        JerseyOcrResult(cand2.jerseyNumber, cand2.confidence, "contrast")
                    } else {
                        JerseyOcrResult(cand1.jerseyNumber, cand1.confidence, "smooth_rgb")
                    }
                }
                cand2 != null -> JerseyOcrResult(cand2.jerseyNumber, cand2.confidence, "contrast")
                cand1 != null -> JerseyOcrResult(cand1.jerseyNumber, cand1.confidence, "smooth_rgb")
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
    )

    private suspend fun runMlKitPass(bitmap: Bitmap, cropHeight: Int): Candidate? {
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
                    val token = elem.text.trim().removePrefix("#")
                    if (!TOKEN_REGEX.matches(token)) continue
                    val value = token.toIntOrNull() ?: continue
                    if (value !in 1..99) continue

                    val boxH = elem.boundingBox?.height() ?: 0
                    val rawConf = elem.confidence
                    val conf = if (!rawConf.isNaN() && rawConf > 0f) {
                        rawConf.coerceIn(0f, 1f)
                    } else {
                        (0.50f + 0.28f * (boxH.toFloat() / max(1, cropHeight)).coerceIn(0f, 1f))
                    }
                    if (conf < MIN_CONFIDENCE) continue

                    matchedElement = true
                    val candidate = Candidate(value.toString(), conf, boxH)
                    if (best == null ||
                        candidate.confidence > best.confidence + 0.04f ||
                        (kotlin.math.abs(candidate.confidence - best.confidence) <= 0.04f && candidate.boxHeight > best.boxHeight)
                    ) {
                        best = candidate
                    }
                }
                // Fallback: extract standalone 1..99 tokens from line.text if element tokenization split digits/symbols.
                if (!matchedElement && !line.text.contains(':') && !line.text.contains('/')) {
                    val lineBoxH = line.boundingBox?.height() ?: 0
                    for (m in LINE_FALLBACK_REGEX.findAll(line.text)) {
                        val value = m.groupValues[1].toIntOrNull() ?: continue
                        if (value !in 1..99) continue
                        val lineConf = (0.44f + 0.25f * (lineBoxH.toFloat() / max(1, cropHeight)).coerceIn(0f, 1f))
                        if (lineConf < MIN_CONFIDENCE) continue
                        val candidate = Candidate(value.toString(), lineConf, lineBoxH)
                        if (best == null ||
                            candidate.confidence > best.confidence + 0.04f ||
                            (kotlin.math.abs(candidate.confidence - best.confidence) <= 0.04f && candidate.boxHeight > best.boxHeight)
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
        private val TOKEN_REGEX = Regex("^\\d{1,2}$")
        private val LINE_FALLBACK_REGEX = Regex("(?<![\\d.])#?0?([1-9]\\d?)(?![\\d.])")
    }
}
