package com.engtestbench.hockeynano.core.tracking

import com.engtestbench.hockeynano.core.model.RectNorm
import kotlin.math.abs

/**
 * Spatial ice-surface filter rejecting off-ice detections (overhead scoreboards, banners,
 * and near-side player benches) while preserving all skaters across both wide panoramic
 * rink feeds and 16:9 zoomed crops.
 */
object OnIceFilter {

    /**
     * Canonical 11-vertex normalized ice surface polygon for Dublin Iceland Rink 1
     * (4080x1808 panoramic camera).
     */
    private val DUBLIN_ICELAND_POLYGON: List<Pair<Float, Float>> = listOf(
        0.044f to 0.420f,
        0.233f to 0.288f,
        0.500f to 0.254f,
        0.767f to 0.288f,
        0.956f to 0.420f,
        0.985f to 0.653f,
        0.772f to 0.785f,
        0.600f to 0.918f,
        0.400f to 0.918f,
        0.228f to 0.785f,
        0.015f to 0.653f
    )

    fun isOnIce(box: RectNorm, aspectRatio: Float, panoramicEnabled: Boolean): Boolean {
        if (!box.x1.isFinite() || !box.y1.isFinite() || !box.x2.isFinite() || !box.y2.isFinite()) {
            return false
        }
        if (box.width <= 0.003f || box.height <= 0.008f) return false
        if (box.width > 0.45f || box.height > 0.85f) return false

        val cx = box.centerX
        val fy = box.footY
        if (cx < 0f || cx > 1f || fy < 0f || fy > 1.02f) return false
        val clampedFy = fy.coerceIn(0f, 1f)

        val usePanoramicPolygon = when {
            aspectRatio >= 2.0f -> true
            aspectRatio > 0f -> false
            else -> panoramicEnabled
        }

        return if (usePanoramicPolygon) {
            pointInPolygon(cx, clampedFy, DUBLIN_ICELAND_POLYGON)
        } else {
            isInsideZoomed16x9Ice(cx, clampedFy)
        }
    }

    private fun isInsideZoomed16x9Ice(cx: Float, fy: Float): Boolean {
        if (cx < 0.015f || cx > 0.985f) return false
        if (fy < 0.20f || fy > 0.94f) return false
        // Reject near-side bottom-left and bottom-right bench triangles in 16:9 zoomed crops.
        if (fy > 0.83f && (cx < 0.14f || cx > 0.86f)) return false
        return true
    }

    private fun pointInPolygon(x: Float, y: Float, poly: List<Pair<Float, Float>>): Boolean {
        var inside = false
        var j = poly.lastIndex
        for (i in poly.indices) {
            val xi = poly[i].first
            val yi = poly[i].second
            val xj = poly[j].first
            val yj = poly[j].second
            val denom = (yj - yi).let { if (abs(it) > 1e-6f) it else 1e-6f }
            val intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / denom + xi)
            if (intersect) inside = !inside
            j = i
        }
        return inside
    }
}
