package com.engtestbench.hockeynano.core

import com.engtestbench.hockeynano.core.model.RectNorm
import com.engtestbench.hockeynano.core.tracking.OnIceFilter
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class OnIceFilterTest {

    private fun skaterAt(cx: Float, footY: Float, w: Float = 0.03f, h: Float = 0.08f): RectNorm =
        RectNorm(cx - w * 0.5f, footY - h, cx + w * 0.5f, footY)

    @Test
    fun `panoramic 11-vertex polygon keeps on-ice skaters and rejects scoreboard and benches`() {
        val aspectPano = 4080f / 1808f // ~2.256

        // Center ice and both goal creases should be on ice.
        assertTrue(OnIceFilter.isOnIce(skaterAt(0.50f, 0.55f), aspectPano, panoramicEnabled = true))
        assertTrue(OnIceFilter.isOnIce(skaterAt(0.15f, 0.45f), aspectPano, panoramicEnabled = true))
        assertTrue(OnIceFilter.isOnIce(skaterAt(0.85f, 0.45f), aspectPano, panoramicEnabled = true))
        assertTrue(OnIceFilter.isOnIce(skaterAt(0.50f, 0.85f), aspectPano, panoramicEnabled = false))

        // Overhead scoreboard & banners should be rejected.
        assertFalse(OnIceFilter.isOnIce(skaterAt(0.50f, 0.22f), aspectPano, panoramicEnabled = true))
        assertFalse(OnIceFilter.isOnIce(skaterAt(0.20f, 0.15f), aspectPano, panoramicEnabled = true))

        // Bottom-left and bottom-right player benches should be rejected.
        assertFalse(OnIceFilter.isOnIce(skaterAt(0.08f, 0.86f), aspectPano, panoramicEnabled = true))
        assertFalse(OnIceFilter.isOnIce(skaterAt(0.92f, 0.86f), aspectPano, panoramicEnabled = true))
    }

    @Test
    fun `16x9 zoomed mode uses lenient bounds without clipping goal skaters while rejecting outliers`() {
        val aspect16x9 = 1920f / 1080f // ~1.778

        // Even with panoramicEnabled=true default, 16:9 zoom must not clip corner/goal skaters.
        assertTrue(OnIceFilter.isOnIce(skaterAt(0.06f, 0.44f), aspect16x9, panoramicEnabled = true))
        assertTrue(OnIceFilter.isOnIce(skaterAt(0.96f, 0.50f), aspect16x9, panoramicEnabled = true))
        assertTrue(OnIceFilter.isOnIce(skaterAt(0.12f, 0.75f), aspect16x9, panoramicEnabled = false))
        assertTrue(OnIceFilter.isOnIce(skaterAt(0.50f, 0.60f), aspect16x9, panoramicEnabled = false))

        // Top scoreboard and bottom-corner bench outliers are still rejected.
        assertFalse(OnIceFilter.isOnIce(skaterAt(0.50f, 0.14f), aspect16x9, panoramicEnabled = true))
        assertFalse(OnIceFilter.isOnIce(skaterAt(0.06f, 0.90f), aspect16x9, panoramicEnabled = false))
        assertFalse(OnIceFilter.isOnIce(skaterAt(0.94f, 0.90f), aspect16x9, panoramicEnabled = true))
    }

    @Test
    fun `degenerate or NaN boxes are rejected`() {
        assertFalse(OnIceFilter.isOnIce(RectNorm(Float.NaN, 0.4f, 0.5f, 0.5f), 2.25f, true))
        assertFalse(OnIceFilter.isOnIce(RectNorm(0.5f, 0.5f, 0.5f, 0.502f), 2.25f, true))
        assertFalse(OnIceFilter.isOnIce(RectNorm(0.0f, 0.0f, 0.9f, 0.9f), 1.77f, false))
    }
}
