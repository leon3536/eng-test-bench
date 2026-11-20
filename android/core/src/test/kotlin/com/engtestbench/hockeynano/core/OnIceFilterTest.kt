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
    fun `degenerate, tiny board-letter FP, or NaN boxes are rejected`() {
        assertFalse(OnIceFilter.isOnIce(RectNorm(Float.NaN, 0.4f, 0.5f, 0.5f), 2.25f, true))
        assertFalse(OnIceFilter.isOnIce(RectNorm(0.5f, 0.5f, 0.5f, 0.502f), 2.25f, true))
        assertFalse(OnIceFilter.isOnIce(RectNorm(0.0f, 0.0f, 0.9f, 0.9f), 1.77f, false))
        // 20x22px false-positive board letter 'A' on 1920x1080 (h ~ 0.0204 < 0.026) must be rejected,
        // while a far-board 0.038f skater on the ice passes.
        assertFalse(OnIceFilter.isOnIce(skaterAt(0.48f, 0.32f, w = 0.011f, h = 0.021f), 1920f / 1080f, panoramicEnabled = false))
        assertTrue(OnIceFilter.isOnIce(skaterAt(0.48f, 0.36f, w = 0.016f, h = 0.038f), 1920f / 1080f, panoramicEnabled = false))
    }

    @Test
    fun `ice context rejects off-ice bench and far-wall spectators while recovering clipped foreground players`() {
        val aspect16x9 = 1920f / 1080f
        val offIceBench = OnIceFilter.IceContext(
            footIceFraction = 0.02f,
            surroundingIceFraction = 0.18f,
            lowerSideIceFraction = 0.22f,
            upperSideIceFraction = 0.08f,
        )
        val benchCoachZeroFoot = OnIceFilter.IceContext(
            footIceFraction = 0.0f,
            surroundingIceFraction = 0.31f,
            lowerSideIceFraction = 0.47f,
            upperSideIceFraction = 0.50f,
        )
        val onIceFarGoalie = OnIceFilter.IceContext(
            footIceFraction = 0.92f,
            surroundingIceFraction = 0.68f,
            lowerSideIceFraction = 0.65f,
            upperSideIceFraction = 0.42f,
        )
        val onIceClippedFg = OnIceFilter.IceContext(
            footIceFraction = 0.45f,
            surroundingIceFraction = 0.77f,
            lowerSideIceFraction = 0.68f,
            upperSideIceFraction = 0.85f,
        )

        // Bench coach standing inside nominal mid-frame or far-wall spectator seats must be rejected.
        assertFalse(OnIceFilter.isOnIce(skaterAt(0.11f, 0.68f, w = 0.045f, h = 0.14f), aspect16x9, false, benchCoachZeroFoot))
        assertFalse(OnIceFilter.isOnIce(skaterAt(0.55f, 0.33f, w = 0.018f, h = 0.040f), aspect16x9, false, offIceBench))

        // On-ice far crease goalie passes; off-ice corner coach is rejected while clipped corner ref passes.
        assertTrue(OnIceFilter.isOnIce(skaterAt(0.33f, 0.33f, w = 0.030f, h = 0.068f), aspect16x9, false, onIceFarGoalie))
        assertFalse(OnIceFilter.isOnIce(skaterAt(0.05f, 0.90f, w = 0.055f, h = 0.16f), aspect16x9, false, offIceBench))
        assertTrue(OnIceFilter.isOnIce(skaterAt(0.04f, 0.99f, w = 0.065f, h = 0.21f), aspect16x9, false, onIceClippedFg))
        assertTrue(OnIceFilter.isOnIce(skaterAt(0.40f, 1.00f, w = 0.055f, h = 0.15f), aspect16x9, false, onIceClippedFg))

        // Far-wall glass / penalty-box spectator (fy < 0.27f with upperSideIce < 0.08f and footIce < 0.65f) must be rejected.
        val farGlassSpectator = OnIceFilter.IceContext(
            footIceFraction = 0.48f,
            surroundingIceFraction = 0.56f,
            lowerSideIceFraction = 0.55f,
            upperSideIceFraction = 0.04f,
        )
        assertFalse(OnIceFilter.isOnIce(skaterAt(0.75f, 0.25f, w = 0.025f, h = 0.060f), aspect16x9, false, farGlassSpectator))
    }
}
