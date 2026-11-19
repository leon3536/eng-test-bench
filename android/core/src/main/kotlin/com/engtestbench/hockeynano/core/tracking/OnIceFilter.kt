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
     * Normalized white-balanced ice-pixel fractions sampled around a candidate person box.
     * Any NaN component defaults to neutral/non-blocking for partial contexts.
     */
    data class IceContext(
        val footIceFraction: Float,
        val surroundingIceFraction: Float,
        val lowerSideIceFraction: Float = Float.NaN,
        val upperSideIceFraction: Float = Float.NaN,
    )

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

    fun isOnIce(box: RectNorm, aspectRatio: Float, panoramicEnabled: Boolean): Boolean =
        isOnIce(box, aspectRatio, panoramicEnabled, iceContext = null)

    fun isOnIce(
        box: RectNorm,
        aspectRatio: Float,
        panoramicEnabled: Boolean,
        iceContext: IceContext?,
    ): Boolean {
        if (!box.x1.isFinite() || !box.y1.isFinite() || !box.x2.isFinite() || !box.y2.isFinite()) {
            return false
        }
        if (box.width <= 0.005f || box.height < 0.026f) return false
        if (box.width > 0.45f || box.height > 0.85f) return false

        val cx = box.centerX
        val fy = box.footY
        if (cx < 0f || cx > 1f || fy < 0f || fy > 1.02f) return false

        val usePanoramicPolygon = when {
            aspectRatio >= 2.0f -> true
            aspectRatio > 0f -> false
            else -> panoramicEnabled
        }

        return if (usePanoramicPolygon) {
            val insidePoly = pointInPolygon(cx, fy.coerceIn(0f, 1f), DUBLIN_ICELAND_POLYGON)
            if (!insidePoly) return false
            if (iceContext != null && iceContext.surroundingIceFraction.isFinite() && iceContext.footIceFraction.isFinite()) {
                val lo = if (iceContext.lowerSideIceFraction.isFinite()) iceContext.lowerSideIceFraction else iceContext.surroundingIceFraction
                if (iceContext.surroundingIceFraction < 0.25f && iceContext.footIceFraction < 0.15f && lo < 0.28f) {
                    return false
                }
            }
            true
        } else {
            isInsideZoomed16x9Ice(cx, fy, box.height, iceContext)
        }
    }

    private fun isInsideZoomed16x9Ice(cx: Float, fy: Float, h: Float, iceContext: IceContext?): Boolean {
        if (cx < 0.015f || cx > 0.985f) return false
        if (iceContext == null) {
            val clampedFy = fy.coerceIn(0f, 1f)
            if (clampedFy < 0.20f || clampedFy > 0.94f) return false
            // Reject near-side bottom-left and bottom-right bench triangles in 16:9 zoomed crops.
            if (clampedFy > 0.83f && (cx < 0.14f || cx > 0.86f)) return false
            return true
        }

        if (fy < 0.20f || fy > 1.005f) return false
        val surr = if (iceContext.surroundingIceFraction.isFinite()) iceContext.surroundingIceFraction else 1f
        val foot = if (iceContext.footIceFraction.isFinite()) iceContext.footIceFraction else 1f
        val lo = if (iceContext.lowerSideIceFraction.isFinite()) iceContext.lowerSideIceFraction else surr
        val hi = if (iceContext.upperSideIceFraction.isFinite()) iceContext.upperSideIceFraction else surr

        // Bottom-clipped foreground recovery: admit fy in (0.94, 1.005] only for tall skaters on ice.
        if (fy > 0.94f) {
            return h >= 0.10f && surr >= 0.50f && lo >= 0.45f
        }

        // Near-side bottom corner triangles: reject unless ice context confirms an on-ice corner player/official.
        if (fy > 0.83f && (cx < 0.14f || cx > 0.86f)) {
            if (surr < 0.50f || lo < 0.45f) return false
        }

        // Off-ice bench coaches and glass-behind spectators inside [0.20, 0.94].
        if ((surr < 0.34f && foot < 0.22f && lo < 0.42f) || (foot < 0.08f && surr < 0.36f)) {
            return false
        }

        // Far-wall stand spectators / elevated bench & penalty-box strip.
        if (fy < 0.27f && hi < 0.08f && foot < 0.65f) {
            return false
        }
        if (fy < 0.365f) {
            if (h < 0.049f && !(surr >= 0.80f && lo >= 0.80f && hi >= 0.80f)) {
                return false
            }
            if (lo < 0.38f || (hi < 0.15f && surr < 0.55f && foot < 0.55f)) {
                return false
            }
        }
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
