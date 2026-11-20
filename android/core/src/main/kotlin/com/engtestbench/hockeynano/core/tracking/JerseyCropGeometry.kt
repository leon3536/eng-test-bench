package com.engtestbench.hockeynano.core.tracking

import com.engtestbench.hockeynano.core.model.RectNorm
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

data class CropPixelBounds(
    val x0: Int,
    val y0: Int,
    val x1: Int,
    val y1: Int
) {
    val width: Int get() = x1 - x0
    val height: Int get() = y1 - y0
}

/**
 * Pure-Kotlin stick-immune torso crop bounds, Catmull-Rom bicubic + unsharp-mask upsampling,
 * and canonical OpenCV-CCW <-> Android-CW derotation math shared across STR, OCR, Nano ZOOM,
 * and `:core` unit tests.
 */
object JerseyCropGeometry {

    const val DEFAULT_TORSO_Y_START = 0.04f
    const val DEFAULT_TORSO_Y_END = 0.58f
    const val NANO_ZOOM_Y_START = 0.04f
    const val NANO_ZOOM_Y_END = 0.56f
    const val DEFAULT_TORSO_X_PAD_FRAC = 0.06f

    const val STICK_BOX_ASPECT_RATIO = 0.58f
    const val TORSO_SCAN_Y_START = 0.16f
    const val TORSO_SCAN_Y_END = 0.56f
    const val NON_ICE_MIN_LAB_L = 62f
    const val NON_ICE_MAX_CHROMA = 18f
    const val MIN_TORSO_WIDTH_HEIGHT_RATIO = 0.46f
    const val NON_ICE_SPAN_EXPANSION = 1.28f
    const val MAX_BOX_WIDTH_EXPANSION = 1.08f

    const val DEROTATE_MIN_ANGLE_DEG = 16f

    /**
     * Canonical Android `Matrix.postRotate(deg, cx, cy)` derotation angle for an ONNX
     * `orientation_deg` output (`+theta > 0` = OpenCV CCW rotation in image space).
     * Android `postRotate(+deg)` rotates CW in y-down image coordinates, directly undoing `+theta` CCW.
     */
    fun androidDerotationAngleDeg(ccwOrientationDeg: Float): Float =
        if (ccwOrientationDeg.isFinite()) ccwOrientationDeg.coerceIn(-60f, 60f) else 0f

    /** Computes pixel-clamped vertical bounds `[y0, y1)` for a normalized detection box. */
    fun computeVerticalBounds(
        imageHeight: Int,
        box: RectNorm,
        yStartFrac: Float = DEFAULT_TORSO_Y_START,
        yEndFrac: Float = DEFAULT_TORSO_Y_END
    ): Pair<Int, Int> {
        val bh = box.height
        val y0 = ((box.y1 + yStartFrac * bh) * imageHeight).toInt().coerceIn(0, imageHeight - 1)
        val y1 = ((box.y1 + yEndFrac * bh) * imageHeight).toInt().coerceIn(y0 + 1, imageHeight)
        return y0 to y1
    }

    /**
     * Returns the `[boxX0..boxX1, 16%..56% y]` pixel scan band when `bwPx > 0.58 * bhPx`
     * (wide stick-extension / crouched box), or `null` when standard horizontal padding suffices.
     */
    fun computeTorsoScanBand(
        imageWidth: Int,
        imageHeight: Int,
        box: RectNorm
    ): CropPixelBounds? {
        if (imageWidth <= 2 || imageHeight <= 2) return null
        val bx0 = (box.x1 * imageWidth).toInt().coerceIn(0, imageWidth - 1)
        val bx1 = (box.x2 * imageWidth).toInt().coerceIn(bx0 + 1, imageWidth)
        val by0 = (box.y1 * imageHeight).toInt().coerceIn(0, imageHeight - 1)
        val by1 = (box.y2 * imageHeight).toInt().coerceIn(by0 + 1, imageHeight)
        val bwPx = (bx1 - bx0).toFloat()
        val bhPx = (by1 - by0).toFloat()
        if (bwPx <= STICK_BOX_ASPECT_RATIO * bhPx || bx1 - bx0 < 4) return null

        val bh = box.height
        val sy0 = ((box.y1 + TORSO_SCAN_Y_START * bh) * imageHeight).toInt().coerceIn(0, imageHeight - 1)
        val sy1 = ((box.y1 + TORSO_SCAN_Y_END * bh) * imageHeight).toInt().coerceIn(sy0 + 1, imageHeight)
        if (sy1 - sy0 < 2) return null
        return CropPixelBounds(bx0, sy0, bx1, sy1)
    }

    /**
     * Computes stick-immune torso-centered horizontal pixel bounds `[x0, x1)` given an optional
     * `y: 16%..56%` scan strip of `ARGB_8888` pixels inside the detection box.
     */
    fun computeStickImmuneHorizontalBounds(
        imageWidth: Int,
        box: RectNorm,
        boxPxH: Float,
        scanPixels: IntArray?,
        scanWidth: Int,
        scanHeight: Int,
        wbR: Float = 1f,
        wbG: Float = 1f,
        wbB: Float = 1f,
        normalPadFrac: Float = DEFAULT_TORSO_X_PAD_FRAC
    ): Pair<Int, Int> {
        val bw = box.width
        val defaultX0 = ((box.x1 - normalPadFrac * bw) * imageWidth).toInt().coerceIn(0, imageWidth - 1)
        val defaultX1 = ((box.x2 + normalPadFrac * bw) * imageWidth).toInt().coerceIn(defaultX0 + 1, imageWidth)

        val bx0 = (box.x1 * imageWidth).toInt().coerceIn(0, imageWidth - 1)
        val bx1 = (box.x2 * imageWidth).toInt().coerceIn(bx0 + 1, imageWidth)
        val bwPx = (bx1 - bx0).toFloat()
        if (bwPx <= STICK_BOX_ASPECT_RATIO * boxPxH ||
            scanPixels == null ||
            scanWidth < 4 ||
            scanHeight < 2 ||
            scanPixels.size < scanWidth * scanHeight
        ) {
            return defaultX0 to defaultX1
        }

        val colFrac = FloatArray(scanWidth)
        for (c in 0 until scanWidth) {
            var nonIce = 0
            for (r in 0 until scanHeight) {
                val p = scanPixels[r * scanWidth + c]
                val rWb = (((p ushr 16) and 0xFF) * wbR).coerceIn(0f, 255f)
                val gWb = (((p ushr 8) and 0xFF) * wbG).coerceIn(0f, 255f)
                val bWb = ((p and 0xFF) * wbB).coerceIn(0f, 255f)
                val lab = KitLabFeatures.rgbToLab(rWb, gWb, bWb)
                val chroma = KitLabFeatures.labChroma(lab[1], lab[2])
                if (!(lab[0] >= NON_ICE_MIN_LAB_L && chroma < NON_ICE_MAX_CHROMA)) {
                    nonIce++
                }
            }
            colFrac[c] = nonIce.toFloat() / scanHeight.toFloat()
        }

        // Locate the torso core window (with slight center tie-breaker) so thin stick shafts
        // or far-edge board artifacts cannot pull the crop span outward.
        val winW = (0.36f * boxPxH).roundToInt().coerceIn(2, scanWidth)
        var bestStart = 0
        var bestWinScore = -Float.MAX_VALUE
        var windowSum = 0f
        for (c in 0 until winW) windowSum += colFrac[c]
        for (start in 0..(scanWidth - winW)) {
            if (start > 0) {
                windowSum += colFrac[start + winW - 1] - colFrac[start - 1]
            }
            val centerFrac = (start + winW * 0.5f) / scanWidth.toFloat()
            val score = windowSum - 0.15f * winW * abs(centerFrac - 0.5f)
            if (score > bestWinScore) {
                bestWinScore = score
                bestStart = start
            }
        }

        var peakCol = bestStart
        var localPeakFrac = -1f
        val midWin = bestStart + (winW - 1) * 0.5f
        for (c in bestStart until (bestStart + winW)) {
            val f = colFrac[c]
            if (f > localPeakFrac + 1e-4f ||
                (abs(f - localPeakFrac) <= 1e-4f && abs(c - midWin) < abs(peakCol - midWin))
            ) {
                localPeakFrac = f
                peakCol = c
            }
        }
        if (localPeakFrac < 0.14f) {
            return defaultX0 to defaultX1
        }

        val activeThresh = max(0.14f, localPeakFrac * 0.28f)
        val maxGap = max(2, (0.06f * boxPxH).roundToInt())

        var spanLeft = peakCol
        var gap = 0
        for (c in (peakCol - 1) downTo 0) {
            if (colFrac[c] >= activeThresh) {
                spanLeft = c
                gap = 0
            } else {
                gap++
                if (gap > maxGap) break
            }
        }

        var spanRight = peakCol
        gap = 0
        for (c in (peakCol + 1) until scanWidth) {
            if (colFrac[c] >= activeThresh) {
                spanRight = c
                gap = 0
            } else {
                gap++
                if (gap > maxGap) break
            }
        }

        val nonIceSpan = (spanRight - spanLeft + 1).toFloat()
        val torsoCenterAbs = bx0 + ((spanLeft + spanRight + 1) * 0.5f) * (bwPx / scanWidth.toFloat())
        val targetWidth = min(
            bwPx * MAX_BOX_WIDTH_EXPANSION,
            max(MIN_TORSO_WIDTH_HEIGHT_RATIO * boxPxH, nonIceSpan * (bwPx / scanWidth.toFloat()) * NON_ICE_SPAN_EXPANSION)
        )
        val x0 = (torsoCenterAbs - targetWidth * 0.5f).roundToInt().coerceIn(0, imageWidth - 1)
        val x1 = (torsoCenterAbs + targetWidth * 0.5f).roundToInt().coerceIn(x0 + 1, imageWidth)
        return x0 to x1
    }

    /**
     * Catmull-Rom bicubic resampler on `ARGB_8888` pixels (`unsharpAmount == 0`, `contrastAlpha == 1`,
     * neutral WB) matching `cv2.INTER_CUBIC` for telephoto `128x128` ONNX letterboxing.
     */
    fun resizeBicubic(
        srcPixels: IntArray,
        srcW: Int,
        srcH: Int,
        dstW: Int,
        dstH: Int
    ): IntArray = upscaleBicubicAndUnsharp(
        srcPixels = srcPixels,
        srcW = srcW,
        srcH = srcH,
        dstW = dstW,
        dstH = dstH,
        wbR = 1f,
        wbG = 1f,
        wbB = 1f,
        contrastAlpha = 1f,
        unsharpAmount = 0f
    )

    /**
     * Catmull-Rom bicubic upscaling with ice white balance, gentle contrast stretch around
     * midpoint `128`, and mild 3x3 unsharp-mask sharpening.
     */
    fun upscaleBicubicAndUnsharp(
        srcPixels: IntArray,
        srcW: Int,
        srcH: Int,
        dstW: Int,
        dstH: Int,
        wbR: Float = 1f,
        wbG: Float = 1f,
        wbB: Float = 1f,
        contrastAlpha: Float = 1.15f,
        unsharpAmount: Float = 0.55f
    ): IntArray {
        if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0 || srcPixels.size < srcW * srcH) {
            return IntArray(0)
        }
        val srcN = srcW * srcH
        val srcR = FloatArray(srcN)
        val srcG = FloatArray(srcN)
        val srcB = FloatArray(srcN)
        val applyStretch = wbR != 1f || wbG != 1f || wbB != 1f || contrastAlpha != 1f
        for (i in 0 until srcN) {
            val p = srcPixels[i]
            val r0 = ((p ushr 16) and 0xFF).toFloat()
            val g0 = ((p ushr 8) and 0xFF).toFloat()
            val b0 = (p and 0xFF).toFloat()
            if (applyStretch) {
                srcR[i] = ((r0 * wbR - 128f) * contrastAlpha + 128f).coerceIn(0f, 255f)
                srcG[i] = ((g0 * wbG - 128f) * contrastAlpha + 128f).coerceIn(0f, 255f)
                srcB[i] = ((b0 * wbB - 128f) * contrastAlpha + 128f).coerceIn(0f, 255f)
            } else {
                srcR[i] = r0
                srcG[i] = g0
                srcB[i] = b0
            }
        }

        val dstN = dstW * dstH
        val dstR = FloatArray(dstN)
        val dstG = FloatArray(dstN)
        val dstB = FloatArray(dstN)

        val scaleX = srcW.toFloat() / dstW.toFloat()
        val scaleY = srcH.toFloat() / dstH.toFloat()

        val wy = FloatArray(4)
        val wx = FloatArray(4)

        for (dy in 0 until dstH) {
            val fy = (dy + 0.5f) * scaleY - 0.5f
            val iy = floor(fy).toInt()
            val ty = fy - iy
            catmullRomWeights(ty, wy)
            val yIdx0 = (iy - 1).coerceIn(0, srcH - 1)
            val yIdx1 = iy.coerceIn(0, srcH - 1)
            val yIdx2 = (iy + 1).coerceIn(0, srcH - 1)
            val yIdx3 = (iy + 2).coerceIn(0, srcH - 1)

            val row0 = yIdx0 * srcW
            val row1 = yIdx1 * srcW
            val row2 = yIdx2 * srcW
            val row3 = yIdx3 * srcW
            val dstOff = dy * dstW

            for (dx in 0 until dstW) {
                val fx = (dx + 0.5f) * scaleX - 0.5f
                val ix = floor(fx).toInt()
                val tx = fx - ix
                catmullRomWeights(tx, wx)
                val xIdx0 = (ix - 1).coerceIn(0, srcW - 1)
                val xIdx1 = ix.coerceIn(0, srcW - 1)
                val xIdx2 = (ix + 1).coerceIn(0, srcW - 1)
                val xIdx3 = (ix + 2).coerceIn(0, srcW - 1)

                var rAcc = 0f
                var gAcc = 0f
                var bAcc = 0f
                for (ky in 0 until 4) {
                    val rBase = when (ky) {
                        0 -> row0
                        1 -> row1
                        2 -> row2
                        else -> row3
                    }
                    val wRow = wy[ky]
                    val p0 = rBase + xIdx0
                    val p1 = rBase + xIdx1
                    val p2 = rBase + xIdx2
                    val p3 = rBase + xIdx3
                    rAcc += wRow * (wx[0] * srcR[p0] + wx[1] * srcR[p1] + wx[2] * srcR[p2] + wx[3] * srcR[p3])
                    gAcc += wRow * (wx[0] * srcG[p0] + wx[1] * srcG[p1] + wx[2] * srcG[p2] + wx[3] * srcG[p3])
                    bAcc += wRow * (wx[0] * srcB[p0] + wx[1] * srcB[p1] + wx[2] * srcB[p2] + wx[3] * srcB[p3])
                }
                val outIdx = dstOff + dx
                dstR[outIdx] = rAcc.coerceIn(0f, 255f)
                dstG[outIdx] = gAcc.coerceIn(0f, 255f)
                dstB[outIdx] = bAcc.coerceIn(0f, 255f)
            }
        }

        val outPixels = IntArray(dstN)
        if (unsharpAmount <= 1e-3f || dstW < 3 || dstH < 3) {
            for (i in 0 until dstN) {
                val r = dstR[i].roundToInt().coerceIn(0, 255)
                val g = dstG[i].roundToInt().coerceIn(0, 255)
                val b = dstB[i].roundToInt().coerceIn(0, 255)
                outPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
            return outPixels
        }

        for (y in 0 until dstH) {
            val ym = max(0, y - 1) * dstW
            val yc = y * dstW
            val yp = min(dstH - 1, y + 1) * dstW
            for (x in 0 until dstW) {
                val xm = max(0, x - 1)
                val xp = min(dstW - 1, x + 1)
                val idx = yc + x

                val blurR = (
                    dstR[ym + xm] + 2f * dstR[ym + x] + dstR[ym + xp] +
                        2f * dstR[yc + xm] + 4f * dstR[idx] + 2f * dstR[yc + xp] +
                        dstR[yp + xm] + 2f * dstR[yp + x] + dstR[yp + xp]
                    ) * (1f / 16f)
                val blurG = (
                    dstG[ym + xm] + 2f * dstG[ym + x] + dstG[ym + xp] +
                        2f * dstG[yc + xm] + 4f * dstG[idx] + 2f * dstG[yc + xp] +
                        dstG[yp + xm] + 2f * dstG[yp + x] + dstG[yp + xp]
                    ) * (1f / 16f)
                val blurB = (
                    dstB[ym + xm] + 2f * dstB[ym + x] + dstB[ym + xp] +
                        2f * dstB[yc + xm] + 4f * dstB[idx] + 2f * dstB[yc + xp] +
                        dstB[yp + xm] + 2f * dstB[yp + x] + dstB[yp + xp]
                    ) * (1f / 16f)

                val r = (dstR[idx] + unsharpAmount * (dstR[idx] - blurR)).roundToInt().coerceIn(0, 255)
                val g = (dstG[idx] + unsharpAmount * (dstG[idx] - blurG)).roundToInt().coerceIn(0, 255)
                val b = (dstB[idx] + unsharpAmount * (dstB[idx] - blurB)).roundToInt().coerceIn(0, 255)
                outPixels[idx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return outPixels
    }

    private fun catmullRomWeights(t: Float, out: FloatArray) {
        val t2 = t * t
        val t3 = t2 * t
        out[0] = -0.5f * t + t2 - 0.5f * t3
        out[1] = 1f - 2.5f * t2 + 1.5f * t3
        out[2] = 0.5f * t + 2f * t2 - 1.5f * t3
        out[3] = -0.5f * t2 + 0.5f * t3
    }

    /**
     * Forward OpenCV `cv2.getRotationMatrix2D(center=(cx, cy), angle=+ccwAngleDeg, scale=1.0)`
     * point transform in y-down image coordinates (`+deg` rotates CCW on screen).
     */
    fun rotatePointOpenCvCcw(
        x: Float,
        y: Float,
        cx: Float,
        cy: Float,
        ccwAngleDeg: Float
    ): Pair<Float, Float> {
        val rad = ccwAngleDeg * (PI / 180.0)
        val cosA = cos(rad).toFloat()
        val sinA = sin(rad).toFloat()
        val dx = x - cx
        val dy = y - cy
        return (cx + dx * cosA + dy * sinA) to (cy - dx * sinA + dy * cosA)
    }

    /**
     * Forward Android `android.graphics.Matrix.postRotate(+cwAngleDeg, cx, cy)` point transform
     * in y-down image coordinates (`+deg` rotates CW on screen).
     */
    fun rotatePointAndroidPostRotateCw(
        x: Float,
        y: Float,
        cx: Float,
        cy: Float,
        cwAngleDeg: Float
    ): Pair<Float, Float> {
        val rad = cwAngleDeg * (PI / 180.0)
        val cosA = cos(rad).toFloat()
        val sinA = sin(rad).toFloat()
        val dx = x - cx
        val dy = y - cy
        return (cx + dx * cosA - dy * sinA) to (cy + dx * sinA + dy * cosA)
    }

    /** Warps a binary/grayscale mask by OpenCV `+ccwAngleDeg` CCW around center (for rotation-sign testing). */
    fun warpMaskOpenCvCcw(src: IntArray, width: Int, height: Int, ccwAngleDeg: Float): IntArray =
        warpMask(src, width, height, padValue = 0) { dstX, dstY, cx, cy ->
            // Inverse of OpenCV CCW(+deg) is OpenCV CCW(-deg) = Android CW(+deg)
            rotatePointOpenCvCcw(dstX, dstY, cx, cy, -ccwAngleDeg)
        }

    /** Warps a binary/grayscale mask by Android `Matrix.postRotate(+cwAngleDeg, cx, cy)` CW around center. */
    fun warpMaskAndroidPostRotateCw(src: IntArray, width: Int, height: Int, cwAngleDeg: Float): IntArray =
        warpMask(src, width, height, padValue = 0) { dstX, dstY, cx, cy ->
            // Inverse of Android CW(+deg) is Android CW(-deg)
            rotatePointAndroidPostRotateCw(dstX, dstY, cx, cy, -cwAngleDeg)
        }

    private inline fun warpMask(
        src: IntArray,
        width: Int,
        height: Int,
        padValue: Int,
        inverseMap: (dstX: Float, dstY: Float, cx: Float, cy: Float) -> Pair<Float, Float>
    ): IntArray {
        val out = IntArray(width * height) { padValue }
        val cx = width * 0.5f
        val cy = height * 0.5f
        for (y in 0 until height) {
            for (x in 0 until width) {
                val (sx, sy) = inverseMap(x + 0.5f, y + 0.5f, cx, cy)
                val ix = floor(sx).toInt()
                val iy = floor(sy).toInt()
                if (ix in 0 until width && iy in 0 until height) {
                    out[y * width + x] = src[iy * width + ix]
                }
            }
        }
        return out
    }
}
