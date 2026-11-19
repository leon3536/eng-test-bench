package com.engtestbench.hockeynano.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import com.engtestbench.hockeynano.core.model.RectNorm
import com.engtestbench.hockeynano.core.tracking.JerseyCropGeometry
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Shared Android bitmap helpers for stick-immune torso-centered cropping,
 * `2.4x` bicubic + unsharp-mask upscaling, and canonical CW `+orientationDeg` derotation.
 */
internal object TorsoCropHelper {

    private val DEFAULT_WB = floatArrayOf(1f, 1f, 1f)
    private const val PAD_GREY = 114

    fun cropUpperTorso(
        highResBitmap: Bitmap,
        box: RectNorm,
        iceWb: FloatArray = DEFAULT_WB,
        yStartFrac: Float = JerseyCropGeometry.DEFAULT_TORSO_Y_START,
        yEndFrac: Float = JerseyCropGeometry.DEFAULT_TORSO_Y_END,
        normalXPadFrac: Float = JerseyCropGeometry.DEFAULT_TORSO_X_PAD_FRAC,
        minWidthPx: Int = 12,
        minHeightPx: Int = 18,
    ): Bitmap? {
        if (highResBitmap.isRecycled) return null
        val w = highResBitmap.width
        val h = highResBitmap.height
        if (w <= 1 || h <= 1) return null

        val wbR = iceWb.getOrElse(0) { 1f }.coerceIn(0.6f, 1.6f)
        val wbG = iceWb.getOrElse(1) { 1f }.coerceIn(0.6f, 1.6f)
        val wbB = iceWb.getOrElse(2) { 1f }.coerceIn(0.6f, 1.6f)

        val (y0, y1) = JerseyCropGeometry.computeVerticalBounds(h, box, yStartFrac, yEndFrac)
        val by0 = (box.y1 * h).toInt().coerceIn(0, h - 1)
        val by1 = (box.y2 * h).toInt().coerceIn(by0 + 1, h)
        val boxPxH = (by1 - by0).toFloat()

        val scanBand = JerseyCropGeometry.computeTorsoScanBand(w, h, box)
        val (x0, x1) = if (scanBand != null) {
            val sw = scanBand.width
            val sh = scanBand.height
            val scanPixels = IntArray(sw * sh)
            highResBitmap.getPixels(scanPixels, 0, sw, scanBand.x0, scanBand.y0, sw, sh)
            JerseyCropGeometry.computeStickImmuneHorizontalBounds(
                imageWidth = w,
                box = box,
                boxPxH = boxPxH,
                scanPixels = scanPixels,
                scanWidth = sw,
                scanHeight = sh,
                wbR = wbR,
                wbG = wbG,
                wbB = wbB,
                normalPadFrac = normalXPadFrac,
            )
        } else {
            JerseyCropGeometry.computeStickImmuneHorizontalBounds(
                imageWidth = w,
                box = box,
                boxPxH = boxPxH,
                scanPixels = null,
                scanWidth = 0,
                scanHeight = 0,
                wbR = wbR,
                wbG = wbG,
                wbB = wbB,
                normalPadFrac = normalXPadFrac,
            )
        }

        val cw = x1 - x0
        val ch = y1 - y0
        if (cw < minWidthPx || ch < minHeightPx) return null
        return Bitmap.createBitmap(highResBitmap, x0, y0, cw, ch)
    }

    fun sampleMedianBorderColor(bitmap: Bitmap): Int {
        if (bitmap.isRecycled || bitmap.width < 2 || bitmap.height < 2) {
            return Color.rgb(PAD_GREY, PAD_GREY, PAD_GREY)
        }
        val w = bitmap.width
        val h = bitmap.height
        val samplesPerEdge = 8
        val rVals = IntArray(samplesPerEdge * 4)
        val gVals = IntArray(samplesPerEdge * 4)
        val bVals = IntArray(samplesPerEdge * 4)
        var n = 0

        fun addSample(x: Int, y: Int) {
            val p = bitmap.getPixel(x.coerceIn(0, w - 1), y.coerceIn(0, h - 1))
            rVals[n] = (p ushr 16) and 0xFF
            gVals[n] = (p ushr 8) and 0xFF
            bVals[n] = p and 0xFF
            n++
        }

        for (i in 0 until samplesPerEdge) {
            val x = ((i + 0.5f) * w / samplesPerEdge).toInt()
            val y = ((i + 0.5f) * h / samplesPerEdge).toInt()
            addSample(x, 0)
            addSample(x, h - 1)
            addSample(0, y)
            addSample(w - 1, y)
        }
        if (n <= 0) return Color.rgb(PAD_GREY, PAD_GREY, PAD_GREY)
        rVals.sort(0, n)
        gVals.sort(0, n)
        bVals.sort(0, n)
        val mid = n / 2
        return Color.rgb(rVals[mid], gVals[mid], bVals[mid])
    }

    /**
     * Derotates [src] from OpenCV CCW `+ccwOrientationDeg` back to upright (`0 deg`)
     * via Android `Matrix.postRotate(+deg, cx, cy)` (clockwise in image coordinates).
     */
    fun derotateBitmapUpright(
        src: Bitmap,
        ccwOrientationDeg: Float,
        expandBounds: Boolean = true,
        bgColor: Int = sampleMedianBorderColor(src),
    ): Bitmap {
        val deg = JerseyCropGeometry.androidDerotationAngleDeg(ccwOrientationDeg)
        val cw = src.width.coerceAtLeast(1)
        val ch = src.height.coerceAtLeast(1)
        val (outW, outH) = if (expandBounds) {
            val rad = abs(deg) * Math.PI / 180.0
            val absCos = abs(cos(rad)).toFloat()
            val absSin = abs(sin(rad)).toFloat()
            val rotW = (cw * absCos + ch * absSin).roundToInt()
            val rotH = (cw * absSin + ch * absCos).roundToInt()
            max(cw, min(rotW, (cw * 1.25f).roundToInt())) to
                max(ch, min(rotH, (ch * 1.25f).roundToInt()))
        } else {
            cw to ch
        }

        val matrix = Matrix().apply {
            postRotate(deg, cw * 0.5f, ch * 0.5f)
            if (outW != cw || outH != ch) {
                postTranslate((outW - cw) * 0.5f, (outH - ch) * 0.5f)
            }
        }
        val out = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(bgColor)
        canvas.drawBitmap(src, matrix, Paint(Paint.FILTER_BITMAP_FLAG))
        return out
    }

    fun buildZoomedJerseyCrop(
        highResBitmap: Bitmap,
        box: RectNorm,
        iceWb: FloatArray = DEFAULT_WB,
        ccwOrientationDeg: Float = 0f,
        zoomFactor: Float = 2.4f,
        maxOutputSidePx: Int = 216,
        contrastAlpha: Float = 1.15f,
        unsharpAmount: Float = 0.55f,
    ): Bitmap? {
        val rawCrop = cropUpperTorso(
            highResBitmap = highResBitmap,
            box = box,
            iceWb = iceWb,
            yStartFrac = JerseyCropGeometry.NANO_ZOOM_Y_START,
            yEndFrac = JerseyCropGeometry.NANO_ZOOM_Y_END,
            normalXPadFrac = JerseyCropGeometry.DEFAULT_TORSO_X_PAD_FRAC,
            minWidthPx = 10,
            minHeightPx = 12,
        ) ?: return null

        val sw = rawCrop.width
        val sh = rawCrop.height
        val rawPixels = IntArray(sw * sh)
        rawCrop.getPixels(rawPixels, 0, sw, 0, 0, sw, sh)
        if (!rawCrop.isRecycled) rawCrop.recycle()

        val wbR = iceWb.getOrElse(0) { 1f }.coerceIn(0.6f, 1.6f)
        val wbG = iceWb.getOrElse(1) { 1f }.coerceIn(0.6f, 1.6f)
        val wbB = iceWb.getOrElse(2) { 1f }.coerceIn(0.6f, 1.6f)

        val scale = min(zoomFactor, maxOutputSidePx.toFloat() / max(sw, sh).coerceAtLeast(1)).coerceAtLeast(1f)
        val dstW = (sw * scale).roundToInt().coerceAtLeast(sw)
        val dstH = (sh * scale).roundToInt().coerceAtLeast(sh)

        val outPixels = JerseyCropGeometry.upscaleBicubicAndUnsharp(
            srcPixels = rawPixels,
            srcW = sw,
            srcH = sh,
            dstW = dstW,
            dstH = dstH,
            wbR = wbR,
            wbG = wbG,
            wbB = wbB,
            contrastAlpha = contrastAlpha,
            unsharpAmount = unsharpAmount,
        )
        if (outPixels.size < dstW * dstH) return null

        val zoomBmp = Bitmap.createBitmap(outPixels, dstW, dstH, Bitmap.Config.ARGB_8888)
        if (abs(ccwOrientationDeg) < JerseyCropGeometry.DEROTATE_MIN_ANGLE_DEG) {
            return zoomBmp
        }
        return try {
            derotateBitmapUpright(zoomBmp, ccwOrientationDeg, expandBounds = true)
        } finally {
            if (!zoomBmp.isRecycled) zoomBmp.recycle()
        }
    }
}
