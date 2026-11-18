package com.engtestbench.hockeynano.core

import com.engtestbench.hockeynano.core.model.FrameMarkSample
import com.engtestbench.hockeynano.core.model.RectNorm
import com.engtestbench.hockeynano.core.model.TeamId
import com.engtestbench.hockeynano.core.tracking.ChainNumberInput
import com.engtestbench.hockeynano.core.tracking.JerseyNumberPosterior
import com.engtestbench.hockeynano.core.tracking.JerseyReadKind
import com.engtestbench.hockeynano.core.tracking.JerseyReadObservation
import com.engtestbench.hockeynano.core.tracking.JerseyReadSource
import com.engtestbench.hockeynano.core.tracking.KitLabFeatures
import com.engtestbench.hockeynano.core.tracking.MultiTeamPlayerTracker
import com.engtestbench.hockeynano.core.tracking.RawDetection
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class MultiTeamPlayerTrackerSotaTest {

    private fun boxAt(cx: Float, footY: Float, w: Float = 0.04f, h: Float = 0.10f): RectNorm =
        RectNorm(cx - w * 0.5f, footY - h, cx + w * 0.5f, footY)

    @Test
    fun `sRGB to CIELAB and orderFreeKitDistSq are exact, symmetric, and swap-invariant`() {
        val blackLab = KitLabFeatures.rgbToLab(0, 0, 0)
        assertEquals(0f, blackLab[0], 1e-3f)
        assertEquals(0f, blackLab[1], 1e-3f)
        assertEquals(0f, blackLab[2], 1e-3f)

        val whiteLab = KitLabFeatures.rgbToLab(255, 255, 255)
        assertEquals(100f, whiteLab[0], 1e-3f)
        assertEquals(0f, whiteLab[1], 0.05f)
        assertEquals(0f, whiteLab[2], 0.05f)

        val greyLab = KitLabFeatures.rgbToLab(128, 128, 128)
        assertTrue(greyLab[0] in 53.0f..54.5f, "Expected mid-grey L* near 53.6, got ${greyLab[0]}")
        assertTrue(KitLabFeatures.labChroma(greyLab[1], greyLab[2]) < 1.0f)

        val x = floatArrayOf(24f, 5f, -22f, 82f, -1f, -2f, 55f)
        val swapped = KitLabFeatures.swapSummary(x)
        assertEquals(0f, KitLabFeatures.orderFreeKitDistSq(x, swapped), 1e-5f)

        val y = floatArrayOf(20f, 4f, -18f, 46f, 2f, -9f, 18f)
        assertEquals(
            KitLabFeatures.orderFreeKitDistSq(x, y),
            KitLabFeatures.orderFreeKitDistSq(y, x),
            1e-4f
        )

        val synthDark = KitLabFeatures.synthesizeKitSummary7(listOf(22, 38, 78))
        val synthLight = KitLabFeatures.synthesizeKitSummary7(listOf(232, 234, 236))
        assertEquals(0f, synthDark[6], 1e-4f)
        assertEquals(100f, synthLight[6], 1e-4f)
    }

    @Test
    fun `dark-yoke white jersey #97 stays TEAM_B and resists contact swap against TEAM_A navy`() {
        val tracker = MultiTeamPlayerTracker()
        val navyRgb = listOf(22, 38, 78)
        val navyKit = floatArrayOf(20f, 5f, -20f, 45f, 2f, -8f, 15f)
        val whiteRgb = listOf(200, 208, 214)
        val whiteKit = floatArrayOf(26f, 4f, -18f, 84f, -1f, -2f, 68f)

        // #97 Begic: chest RGB looks dark due to navy shoulder yoke, but light cluster + lowerBellyLuma reveal white body!
        val yokeContaminatedRgb = listOf(38, 52, 88)
        val begicKit = floatArrayOf(24f, 5f, -22f, 82f, -1f, -2f, 46f)

        var latest: FrameMarkSample? = null
        var begicId = -1
        var navyCrossId = -1

        for (step in 0..4) {
            val t = step * 0.25
            // Move #97 and one Navy skater toward a close crossing at step 2.
            val begicX = 0.42f + step * 0.03f
            val navyX = 0.54f - step * 0.03f
            val dets = listOf(
                RawDetection(boxAt(0.14f, 0.50f), navyRgb, kitSummaryLab7 = navyKit, lowerBellyLuma = 34f, pantsDarkness = 0.78f),
                RawDetection(boxAt(0.24f, 0.52f), navyRgb, kitSummaryLab7 = navyKit, lowerBellyLuma = 36f, pantsDarkness = 0.76f),
                RawDetection(boxAt(0.32f, 0.48f), navyRgb, kitSummaryLab7 = navyKit, lowerBellyLuma = 35f, pantsDarkness = 0.80f),
                RawDetection(boxAt(navyX, 0.55f), navyRgb, kitSummaryLab7 = navyKit, lowerBellyLuma = 35f, pantsDarkness = 0.78f),
                RawDetection(boxAt(0.68f, 0.50f), whiteRgb, kitSummaryLab7 = whiteKit, lowerBellyLuma = 178f, pantsDarkness = 0.65f),
                RawDetection(boxAt(0.76f, 0.52f), whiteRgb, kitSummaryLab7 = whiteKit, lowerBellyLuma = 182f, pantsDarkness = 0.66f),
                RawDetection(boxAt(0.84f, 0.48f), whiteRgb, kitSummaryLab7 = whiteKit, lowerBellyLuma = 175f, pantsDarkness = 0.64f),
                RawDetection(boxAt(0.90f, 0.54f), whiteRgb, kitSummaryLab7 = whiteKit, lowerBellyLuma = 180f, pantsDarkness = 0.68f),
                RawDetection(boxAt(begicX, 0.55f), yokeContaminatedRgb, kitSummaryLab7 = begicKit, lowerBellyLuma = 170f, pantsDarkness = 0.66f)
            )
            val f = tracker.updateFrame(t, dets, 0.94f)
            latest = f
            if (step == 0) {
                navyCrossId = f.players[3].trackId
                begicId = f.players[8].trackId
            }
        }

        val finalFrame = latest!!
        val begicFinal = finalFrame.players.first { it.trackId == begicId }
        val navyFinal = finalFrame.players.first { it.trackId == navyCrossId }
        assertEquals(TeamId.TEAM_B, begicFinal.team, "Dark-yoke white jersey #97 must classify as TEAM_B")
        assertEquals(TeamId.TEAM_A, navyFinal.team, "Crossing Navy skater must remain TEAM_A")
        assertTrue(begicFinal.box.centerX > 0.50f, "#97 must retain its own trackId across contact crossing")
        assertTrue(navyFinal.box.centerX < 0.46f, "Navy skater must retain its own trackId across contact crossing")
        assertEquals(4, finalFrame.players.count { it.team == TeamId.TEAM_A })
        assertEquals(5, finalFrame.players.count { it.team == TeamId.TEAM_B })

        // Even if the other 4 Team B teammates show almost pure white (90% light share), #97's lowerBellyLuma keeps it in TEAM_B!
        tracker.reset()
        val pureWhiteKit = floatArrayOf(76f, 0f, 0f, 85f, 0f, 0f, 90f)
        var hardFrame: FrameMarkSample? = null
        for (step in 0..2) {
            hardFrame = tracker.updateFrame(
                step * 0.25,
                listOf(
                    RawDetection(boxAt(0.14f, 0.50f), navyRgb, kitSummaryLab7 = navyKit, lowerBellyLuma = 35f),
                    RawDetection(boxAt(0.24f, 0.52f), navyRgb, kitSummaryLab7 = navyKit, lowerBellyLuma = 35f),
                    RawDetection(boxAt(0.32f, 0.48f), navyRgb, kitSummaryLab7 = navyKit, lowerBellyLuma = 34f),
                    RawDetection(boxAt(0.42f, 0.54f), navyRgb, kitSummaryLab7 = navyKit, lowerBellyLuma = 36f),
                    RawDetection(boxAt(0.60f, 0.50f), whiteRgb, kitSummaryLab7 = pureWhiteKit, lowerBellyLuma = 185f),
                    RawDetection(boxAt(0.70f, 0.52f), whiteRgb, kitSummaryLab7 = pureWhiteKit, lowerBellyLuma = 186f),
                    RawDetection(boxAt(0.80f, 0.48f), whiteRgb, kitSummaryLab7 = pureWhiteKit, lowerBellyLuma = 184f),
                    RawDetection(boxAt(0.90f, 0.54f), whiteRgb, kitSummaryLab7 = pureWhiteKit, lowerBellyLuma = 185f),
                    RawDetection(boxAt(0.50f, 0.56f), yokeContaminatedRgb, kitSummaryLab7 = begicKit, lowerBellyLuma = 170f)
                ),
                0.94f
            )
        }
        assertEquals(TeamId.TEAM_B, hardFrame!!.players.last().team, "lowerBellyLuma must keep #97 in TEAM_B even against mostly-white teammates")

        // Real-frame fs_12_674 regression: cool-blue shadow on white waist (bellyLuma=139.1f, lightL=45.8f, lightShare=60.6%)
        // stays TEAM_B, while #43 (dark navy body with light highlight bellyLuma=85f, lightShare=54%) stays TEAM_A at bellyWeight=1.25f.
        tracker.reset()
        val realNavyKit = floatArrayOf(15f, 4f, -15f, 35f, 1f, -10f, 20f)
        val real43Kit = floatArrayOf(18f, 4f, -16f, 52f, 0f, -6f, 54f)
        val realWhiteKit = floatArrayOf(26f, 4f, -16f, 75f, -1f, -3f, 72f)
        val real97Kit = floatArrayOf(19.4f, 4.9f, -16.8f, 45.8f, 0.0f, -7.1f, 60.6f)
        var real97Frame: FrameMarkSample? = null
        for (step in 0..2) {
            real97Frame = tracker.updateFrame(
                step * 0.25,
                listOf(
                    RawDetection(boxAt(0.14f, 0.50f), navyRgb, kitSummaryLab7 = realNavyKit, lowerBellyLuma = 21f),
                    RawDetection(boxAt(0.24f, 0.52f), navyRgb, kitSummaryLab7 = realNavyKit, lowerBellyLuma = 32f),
                    RawDetection(boxAt(0.34f, 0.48f), navyRgb, kitSummaryLab7 = realNavyKit, lowerBellyLuma = 42f),
                    RawDetection(boxAt(0.44f, 0.54f), listOf(38, 48, 72), kitSummaryLab7 = real43Kit, lowerBellyLuma = 85f),
                    RawDetection(boxAt(0.60f, 0.50f), whiteRgb, kitSummaryLab7 = realWhiteKit, lowerBellyLuma = 162f),
                    RawDetection(boxAt(0.70f, 0.52f), whiteRgb, kitSummaryLab7 = realWhiteKit, lowerBellyLuma = 168f),
                    RawDetection(boxAt(0.80f, 0.48f), whiteRgb, kitSummaryLab7 = realWhiteKit, lowerBellyLuma = 174f),
                    RawDetection(boxAt(0.90f, 0.54f), whiteRgb, kitSummaryLab7 = realWhiteKit, lowerBellyLuma = 165f),
                    RawDetection(boxAt(0.52f, 0.56f), listOf(88, 102, 128), kitSummaryLab7 = real97Kit, lowerBellyLuma = 139.1f)
                ),
                0.94f
            )
        }
        assertEquals(TeamId.TEAM_A, real97Frame!!.players[3].team, "Real-frame #43 (belly=85, share=54%) must remain TEAM_A")
        assertEquals(TeamId.TEAM_B, real97Frame.players.last().team, "Real-frame cool-shadow #97 must stay TEAM_B")
    }

    @Test
    fun `multi-cue official detection separates armband and far-board officials without false positives on #14`() {
        val tracker = MultiTeamPlayerTracker()
        val navyRgb = listOf(22, 38, 78)
        val navyKit = floatArrayOf(18f, 6f, -22f, 40f, 3f, -10f, 12f)
        val whiteRgb = listOf(225, 228, 232)
        val whiteKit = floatArrayOf(26f, 4f, -18f, 88f, -1f, -1f, 75f)
        val officialGrayRgb = listOf(128, 130, 127)
        val officialGrayKit = floatArrayOf(34f, -1f, 1f, 66f, 0f, 1f, 45f)

        // 1) Below 9 skaters: armband-assisted striped official (stripe=0.38, armband=0.08) classifies REFEREE.
        var fSmall: FrameMarkSample? = null
        for (i in 0..2) {
            fSmall = tracker.updateFrame(
                i * 0.25,
                listOf(
                    RawDetection(boxAt(0.15f, 0.50f), navyRgb, stripeScore = 0.08f, kitSummaryLab7 = navyKit, lowerBellyLuma = 35f),
                    RawDetection(boxAt(0.28f, 0.52f), navyRgb, stripeScore = 0.10f, kitSummaryLab7 = navyKit, lowerBellyLuma = 35f),
                    RawDetection(boxAt(0.41f, 0.48f), navyRgb, stripeScore = 0.09f, kitSummaryLab7 = navyKit, lowerBellyLuma = 35f),
                    RawDetection(boxAt(0.55f, 0.50f), whiteRgb, stripeScore = 0.11f, kitSummaryLab7 = whiteKit, lowerBellyLuma = 185f),
                    RawDetection(boxAt(0.68f, 0.52f), whiteRgb, stripeScore = 0.12f, kitSummaryLab7 = whiteKit, lowerBellyLuma = 185f),
                    RawDetection(boxAt(0.81f, 0.48f), whiteRgb, stripeScore = 0.10f, kitSummaryLab7 = whiteKit, lowerBellyLuma = 185f),
                    RawDetection(boxAt(0.48f, 0.40f), officialGrayRgb, stripeScore = 0.38f, kitSummaryLab7 = officialGrayKit, lowerBellyLuma = 128f, armbandScore = 0.08f)
                ),
                0.92f
            )
        }
        assertEquals(TeamId.REFEREE, fSmall!!.players.last().team)
        assertEquals(3, fSmall.players.count { it.team == TeamId.TEAM_A })
        assertEquals(3, fSmall.players.count { it.team == TeamId.TEAM_B })

        // 2) >= 9 skaters: blurred far-board officials (stripe=0.05, pantsDarkness=0.78, neutral grey) classify REFEREE,
        //    while dark-navy #14 (pantsDarkness=0.78, dark navy) and dark-yoke #97 NEVER trigger REFEREE.
        tracker.reset()
        val begicKit = floatArrayOf(24f, 5f, -22f, 82f, -1f, -2f, 52f)
        var fLarge: FrameMarkSample? = null
        for (i in 0..3) {
            fLarge = tracker.updateFrame(
                i * 0.25,
                listOf(
                    // 4 Navy skaters including #14 Matthews (index 3)
                    RawDetection(boxAt(0.12f, 0.50f), navyRgb, stripeScore = 0.05f, kitSummaryLab7 = navyKit, lowerBellyLuma = 35f, pantsDarkness = 0.78f),
                    RawDetection(boxAt(0.20f, 0.52f), navyRgb, stripeScore = 0.06f, kitSummaryLab7 = navyKit, lowerBellyLuma = 34f, pantsDarkness = 0.76f),
                    RawDetection(boxAt(0.28f, 0.48f), navyRgb, stripeScore = 0.04f, kitSummaryLab7 = navyKit, lowerBellyLuma = 36f, pantsDarkness = 0.80f),
                    RawDetection(boxAt(0.36f, 0.54f), listOf(26, 40, 74), stripeScore = 0.22f, kitSummaryLab7 = navyKit, lowerBellyLuma = 38f, pantsDarkness = 0.78f),
                    // 5 White skaters including #97 (index 8)
                    RawDetection(boxAt(0.50f, 0.50f), whiteRgb, stripeScore = 0.08f, kitSummaryLab7 = whiteKit, lowerBellyLuma = 182f, pantsDarkness = 0.65f),
                    RawDetection(boxAt(0.58f, 0.52f), whiteRgb, stripeScore = 0.09f, kitSummaryLab7 = whiteKit, lowerBellyLuma = 180f, pantsDarkness = 0.64f),
                    RawDetection(boxAt(0.66f, 0.48f), whiteRgb, stripeScore = 0.07f, kitSummaryLab7 = whiteKit, lowerBellyLuma = 184f, pantsDarkness = 0.66f),
                    RawDetection(boxAt(0.74f, 0.54f), whiteRgb, stripeScore = 0.08f, kitSummaryLab7 = whiteKit, lowerBellyLuma = 178f, pantsDarkness = 0.62f),
                    RawDetection(boxAt(0.82f, 0.50f), listOf(45, 58, 92), stripeScore = 0.14f, kitSummaryLab7 = begicKit, lowerBellyLuma = 168f, pantsDarkness = 0.65f),
                    // 2 Blurred far-board officials (indices 9, 10):
                    // REF_C (pants=0.99, meanL~49, C*~3) and REF_L (pants=0.84, meanL~35, meanChroma~13.5 from board/armband bleed)
                    RawDetection(boxAt(0.42f, 0.34f), officialGrayRgb, stripeScore = 0.05f, kitSummaryLab7 = officialGrayKit, lowerBellyLuma = 122f, pantsDarkness = 0.99f),
                    RawDetection(boxAt(0.88f, 0.35f), listOf(92, 82, 65), stripeScore = 0.06f, kitSummaryLab7 = floatArrayOf(15.9f, 3.5f, 11.5f, 43.3f, 2.0f, 12.0f, 77.0f), lowerBellyLuma = 85f, pantsDarkness = 0.84f)
                ),
                0.93f
            )
        }

        val frame = fLarge!!
        assertEquals(4, frame.players.count { it.team == TeamId.TEAM_A }, "#14 and all 4 Navy skaters must stay TEAM_A")
        assertEquals(5, frame.players.count { it.team == TeamId.TEAM_B }, "#97 and all 5 White skaters must stay TEAM_B")
        assertEquals(2, frame.players.count { it.team == TeamId.REFEREE }, "Both blurred far-board officials must classify REFEREE")
        assertEquals(TeamId.TEAM_A, frame.players[3].team, "#14 solid dark-navy must NEVER trigger REFEREE")
        assertEquals(TeamId.TEAM_B, frame.players[8].team, "#97 white-bodied jersey must NEVER trigger REFEREE")
        assertEquals(TeamId.REFEREE, frame.players[9].team)
        assertEquals(TeamId.REFEREE, frame.players[10].team)
    }

    @Test
    fun `burst crossfade ghost suppression drops hitCount 1 tracks only above 15 detections`() {
        val tracker = MultiTeamPlayerTracker()
        val dark = listOf(24, 42, 76)
        val light = listOf(232, 234, 236)

        val baseReal = (0 until 6).map { RawDetection(boxAt(0.10f + it * 0.06f, 0.52f), dark) } +
            (0 until 6).map { RawDetection(boxAt(0.50f + it * 0.06f, 0.52f), light) }

        // Frame 0 (12 <= 15): real tracks emit immediately on hit 1.
        val f0 = tracker.updateFrame(0.0, baseReal, 0.90f)
        assertEquals(12, f0.players.size)
        val baseIds = f0.players.map { it.trackId }.toSet()

        // Frame 1 (13 <= 15, e.g. 12 real + 1 edge newcomer): still <= 15 so emits immediately on hit 1.
        val newcomer = RawDetection(boxAt(0.92f, 0.55f), dark)
        val f1 = tracker.updateFrame(0.25, baseReal + listOf(newcomer), 0.90f)
        assertEquals(13, f1.players.size, "<=15 detections must preserve all 13 hit-1 detections")
        val newcomerId = f1.players.last().trackId

        // Frame 2 (crossfade burst: 13 confirmed + 6 transient ghosts = 19 > 15):
        // Only the 13 confirmed tracks emit; all 6 hitCount==1 ghosts are suppressed!
        val ghosts = (0 until 6).map { RawDetection(boxAt(0.15f + it * 0.12f, 0.72f), light) }
        val f2 = tracker.updateFrame(0.50, baseReal + listOf(newcomer) + ghosts, 0.85f)
        assertEquals(13, f2.players.size, "Burst frame >15 detections must suppress all hitCount==1 transient ghosts")
        assertTrue(f2.players.map { it.trackId }.containsAll(baseIds))
        assertTrue(f2.players.any { it.trackId == newcomerId })

        // Frame 3 (burst frame 17 > 15 with 4 brand-new far tracks): suppressed at hitCount==1 on Frame 3...
        val ghosts2 = (0 until 4).map { RawDetection(boxAt(0.20f + it * 0.20f, 0.25f), dark) }
        val f3 = tracker.updateFrame(1.00, baseReal + listOf(newcomer) + ghosts2, 0.88f)
        assertEquals(13, f3.players.size)

        // ...and confirmed on Frame 4 when hitCount reaches 2!
        val f4 = tracker.updateFrame(1.25, baseReal + listOf(newcomer) + ghosts2, 0.90f)
        assertEquals(17, f4.players.size, "Real skaters entering during >15 burst must confirm on hitCount>=2")
    }

    @Test
    fun `Bayesian posterior fuses RIGHT_PARTIAL 7 and LEFT_PARTIAL 9 into #97 with uniform prior and confusion handling`() {
        // Uniform prior verification over 1..99.
        val emptyLogP = JerseyNumberPosterior.computeLogPosterior(emptyList())
        val emptyProbs = JerseyNumberPosterior.posteriorProbs(emptyLogP)
        assertEquals(99, emptyProbs.size)
        assertEquals(1f / 99f, emptyProbs["97"]!!, 1e-5f)

        // Single RIGHT_PARTIAL '7' must NOT claim single-digit #7!
        val tracker = MultiTeamPlayerTracker()
        val f0 = tracker.updateFrame(
            0.0,
            listOf(
                RawDetection(boxAt(0.40f, 0.55f), listOf(230, 232, 234)),
                RawDetection(boxAt(0.75f, 0.55f), listOf(22, 38, 76))
            ),
            0.9f
        )
        val tid97 = f0.players[0].trackId
        val tidLow = f0.players[1].trackId

        tracker.recordJerseyRead(
            trackId = tid97,
            jerseyNumber = "7",
            confidence = 0.85f,
            timestampSec = 0.0,
            kind = JerseyReadKind.RIGHT_PARTIAL,
            qualityWeight = 1.0f,
            source = JerseyReadSource.MLKIT_OCR
        )
        val afterRightOnly = tracker.annotateHistory(listOf(f0)).first().players[0]
        assertNull(afterRightOnly.jerseyNumber, "Single right-partial '7' must not prematurely lock #7 or #77")

        // Now add LEFT_PARTIAL '9' from ONNX_STR distribution callback -> fuses cleanly to #97!
        tracker.recordJerseyDistribution(
            trackId = tid97,
            timestampSec = 0.5,
            candidates = emptyList(),
            qualityWeight = 1.0f,
            source = JerseyReadSource.ONNX_STR,
            partialLeftDigit = '9' to 0.88f,
            partialRightDigit = null
        )
        val afterBoth = tracker.annotateHistory(listOf(f0)).first().players[0]
        assertEquals("97", afterBoth.jerseyNumber, "RIGHT_PARTIAL '7' + LEFT_PARTIAL '9' must fuse to #97")

        // Wild low-confidence guess (0.35f MLKIT_OCR) stays unlabelled until verified.
        tracker.recordJerseyRead(
            trackId = tidLow,
            jerseyNumber = "88",
            confidence = 0.35f,
            timestampSec = 0.0,
            kind = JerseyReadKind.FULL,
            qualityWeight = 1.0f,
            source = JerseyReadSource.MLKIT_OCR
        )
        val lowGuessPlayer = tracker.annotateHistory(listOf(f0)).first().players[1]
        assertNull(lowGuessPlayer.jerseyNumber, "Single wild low-conf MLKit guess must remain unlabelled")

        // Digit confusion check: reading "37" + "87" (3<->8 confusion) gives high support to "37"/"87" over unrelated "12".
        val confObs = listOf(
            JerseyReadObservation(0.0, "37", 0.70f, JerseyReadKind.FULL, 1f, JerseyReadSource.MLKIT_OCR),
            JerseyReadObservation(0.5, "87", 0.70f, JerseyReadKind.FULL, 1f, JerseyReadSource.MLKIT_OCR),
            JerseyReadObservation(1.0, "37", 0.75f, JerseyReadKind.FULL, 1f, JerseyReadSource.ONNX_STR)
        )
        val confProbs = JerseyNumberPosterior.posteriorProbs(JerseyNumberPosterior.computeLogPosterior(confObs))
        assertTrue((confProbs["37"] ?: 0f) > (confProbs["87"] ?: 0f))
        assertTrue((confProbs["87"] ?: 0f) > 10f * (confProbs["12"] ?: 0f))

        // Uniform outlier floor check: one bad 66:0.79 spike must NOT wipe out two agreeing 67:0.98 frames.
        val spikeRobustObs = listOf(
            JerseyReadObservation(0.0, null, 0.98f, JerseyReadKind.FULL, 1f, JerseyReadSource.ONNX_STR, listOf("67" to 0.98f)),
            JerseyReadObservation(0.5, null, 0.98f, JerseyReadKind.FULL, 1f, JerseyReadSource.ONNX_STR, listOf("67" to 0.98f)),
            JerseyReadObservation(1.0, null, 0.79f, JerseyReadKind.FULL, 1f, JerseyReadSource.ONNX_STR, listOf("66" to 0.79f))
        )
        val spikeWinner = JerseyNumberPosterior.bestCandidate(JerseyNumberPosterior.computeLogPosterior(spikeRobustObs))
        assertEquals("67", spikeWinner.first, "Two 67:0.98 frames must beat one 66:0.79 outlier spike")
        assertTrue(spikeWinner.second > 0.90f)
    }

    @Test
    fun `Hungarian co-occurrence exclusivity resolves concurrent #77 vs #7 clash and allows strong multi-bin duplicates`() {
        val tracker = MultiTeamPlayerTracker()
        val dark = listOf(22, 38, 76)
        val light = listOf(232, 234, 236)

        // Two concurrent TEAM_A skaters from t=0..12s (overlap = 12s > max(1.5, 0.12*12)).
        val history = ArrayList<FrameMarkSample>()
        var idStrong77 = -1
        var idClash7 = -1
        for (i in 0..4) {
            val t = i * 3.0 // 0, 3, 6, 9, 12s
            val f = tracker.updateFrame(
                t,
                listOf(
                    RawDetection(boxAt(0.25f, 0.52f), dark),
                    RawDetection(boxAt(0.45f, 0.52f), dark),
                    RawDetection(boxAt(0.80f, 0.52f), light)
                ),
                0.92f
            )
            if (i == 0) {
                idStrong77 = f.players[0].trackId
                idClash7 = f.players[1].trackId
            }
            history.add(f)
        }
        assertEquals(idStrong77, history.last().players[0].trackId)
        assertEquals(idClash7, history.last().players[1].trackId)

        // Skater 1: strong #77 across 2 distinct 10s bins (t=0s and t=12s).
        tracker.recordJerseyRead(idStrong77, "77", 0.94f, 0.0, JerseyReadKind.FULL, 1f, JerseyReadSource.GEMINI_NANO)
        tracker.recordJerseyRead(idStrong77, "77", 0.92f, 12.0, JerseyReadKind.FULL, 1f, JerseyReadSource.ONNX_STR)

        // Skater 2: corroborated distribution leaning #77 (0.50) vs #7 (0.42), which without Hungarian would collide on #77.
        tracker.recordJerseyDistribution(
            trackId = idClash7,
            timestampSec = 3.0,
            candidates = listOf("77" to 0.50f, "7" to 0.42f),
            qualityWeight = 1.0f,
            source = JerseyReadSource.ONNX_STR
        )
        tracker.recordJerseyDistribution(
            trackId = idClash7,
            timestampSec = 4.0,
            candidates = listOf("77" to 0.48f, "7" to 0.44f),
            qualityWeight = 1.0f,
            source = JerseyReadSource.ONNX_STR
        )

        val resolved = tracker.annotateHistory(history).last().players
        val p1 = resolved.first { it.trackId == idStrong77 }
        val p2 = resolved.first { it.trackId == idClash7 }
        assertEquals("77", p1.jerseyNumber, "Strong multi-bin #77 must keep #77")
        assertEquals("7", p2.jerseyNumber, "Concurrent teammate clashing on #77 must move to runner-up #7")
        assertNotEquals(p1.jerseyNumber, p2.jerseyNumber)

        // Soft duplicate fallback: two concurrent chains that BOTH have strong independent multi-bin support keep #77.
        val dupInputs = listOf(
            ChainNumberInput(
                chainId = 101,
                firstSeenSec = 0.0,
                lastSeenSec = 15.0,
                observations = listOf(
                    JerseyReadObservation(1.0, "77", 0.92f, JerseyReadKind.FULL, 1f, JerseyReadSource.GEMINI_NANO),
                    JerseyReadObservation(12.0, "77", 0.91f, JerseyReadKind.FULL, 1f, JerseyReadSource.ONNX_STR)
                ),
                nanoVerifiedNumbers = setOf("77"),
                bestConfidenceByNumber = mapOf("77" to 0.92f)
            ),
            ChainNumberInput(
                chainId = 102,
                firstSeenSec = 0.0,
                lastSeenSec = 15.0,
                observations = listOf(
                    JerseyReadObservation(2.0, "77", 0.90f, JerseyReadKind.FULL, 1f, JerseyReadSource.GEMINI_NANO),
                    JerseyReadObservation(13.0, "77", 0.89f, JerseyReadKind.FULL, 1f, JerseyReadSource.ONNX_STR)
                ),
                nanoVerifiedNumbers = setOf("77"),
                bestConfidenceByNumber = mapOf("77" to 0.90f)
            )
        )
        val dupResult = JerseyNumberPosterior.assignNumbersJointly(dupInputs)
        assertEquals("77", dupResult[101]?.number)
        assertEquals("77", dupResult[102]?.number)
        assertEquals(listOf(102), dupResult[101]?.duplicateWithChainIds)
        assertEquals(listOf(101), dupResult[102]?.duplicateWithChainIds)

        // Non-concurrent handoff (<= 1.5s overlap) allows both chains to share #77 without duplicate clash.
        val handoffInputs = listOf(
            ChainNumberInput(
                chainId = 201,
                firstSeenSec = 0.0,
                lastSeenSec = 10.0,
                observations = listOf(JerseyReadObservation(1.0, "77", 0.90f, JerseyReadKind.FULL, 1f, JerseyReadSource.ONNX_STR)),
                bestConfidenceByNumber = mapOf("77" to 0.90f)
            ),
            ChainNumberInput(
                chainId = 202,
                firstSeenSec = 9.0, // 1.0s overlap <= 1.5s
                lastSeenSec = 20.0,
                observations = listOf(
                    JerseyReadObservation(15.0, "77", 0.68f, JerseyReadKind.FULL, 1f, JerseyReadSource.ONNX_STR),
                    JerseyReadObservation(16.2, "77", 0.68f, JerseyReadKind.FULL, 1f, JerseyReadSource.ONNX_STR)
                ),
                bestConfidenceByNumber = mapOf("77" to 0.68f)
            )
        )
        val handoffRes = JerseyNumberPosterior.assignNumbersJointly(handoffInputs)
        assertEquals("77", handoffRes[201]?.number)
        assertEquals("77", handoffRes[202]?.number)
        assertTrue(handoffRes[201]?.duplicateWithChainIds?.isEmpty() == true)

        // Unpinned clashing duplicate with NO valid alternative must unassign to null instead of duplicating #97!
        val unpinnedClash = listOf(
            ChainNumberInput(
                chainId = 301,
                firstSeenSec = 0.0,
                lastSeenSec = 10.0,
                observations = listOf(
                    JerseyReadObservation(1.0, "97", 0.94f, JerseyReadKind.FULL, 1f, JerseyReadSource.GEMINI_NANO),
                    JerseyReadObservation(2.0, "97", 0.92f, JerseyReadKind.FULL, 1f, JerseyReadSource.ONNX_STR)
                ),
                nanoVerifiedNumbers = setOf("97"),
                bestConfidenceByNumber = mapOf("97" to 0.94f)
            ),
            ChainNumberInput(
                chainId = 302,
                firstSeenSec = 0.0,
                lastSeenSec = 10.0,
                observations = listOf(
                    JerseyReadObservation(3.0, "97", 0.85f, JerseyReadKind.FULL, 1f, JerseyReadSource.GEMINI_NANO)
                ),
                nanoVerifiedNumbers = setOf("97"),
                bestConfidenceByNumber = mapOf("97" to 0.85f)
            )
        )
        val unpinnedRes = JerseyNumberPosterior.assignNumbersJointly(unpinnedClash)
        assertEquals("97", unpinnedRes[301]?.number, "Stronger #97 track must keep #97")
        assertNull(unpinnedRes[302]?.number, "Unpinned clashing track with no runner-up must unassign to null instead of duplicating #97")
    }

    @Test
    fun `minimum cluster-support guard and Huberized chroma prevent 1-vs-N team collapse on extreme chroma and bright outliers`() {
        val tracker = MultiTeamPlayerTracker()
        val navyRgb = listOf(22, 38, 78)
        val navyKit = floatArrayOf(18f, 5f, -18f, 40f, 2f, -8f, 15f)
        val whiteRgb = listOf(222, 226, 230)
        val whiteKit = floatArrayOf(26f, 4f, -16f, 82f, -1f, -2f, 68f)
        val officialGrayRgb = listOf(128, 130, 127)
        val officialGrayKit = floatArrayOf(34f, -1f, 1f, 66f, 0f, 1f, 45f)

        // Exact T34 blue board-post outlier (maxZoneChroma ~ 41.3, effL ~ 25.4) + extreme bright glare outlier (effL ~ 90)
        val t34BoardOutlierKit = floatArrayOf(16.1f, 17.8f, -37.3f, 45.1f, -1.8f, -3.3f, 51.1f)
        val brightOutlierKit = floatArrayOf(82f, -12f, 18f, 96f, -8f, 14f, 92f)

        // Establish 20 historical skater tracks (10 Navy + 10 White) + 2 far-board officials across 2 shifts.
        val shift1 = (0 until 5).map { RawDetection(boxAt(0.08f + it * 0.06f, 0.45f), navyRgb, kitSummaryLab7 = navyKit, lowerBellyLuma = 35f, pantsDarkness = 0.78f) } +
            (0 until 5).map { RawDetection(boxAt(0.45f + it * 0.06f, 0.45f), whiteRgb, kitSummaryLab7 = whiteKit, lowerBellyLuma = 175f, pantsDarkness = 0.65f) }
        for (i in 0 until 4) {
            tracker.updateFrame(i * 0.25, shift1, 0.92f)
        }

        // Jump > 3.0s so new tracks spawn in registry while preserving the 10 historical skaters, adding 10 more + 2 refs + T34 + bright glare.
        val shift2Base = (0 until 5).map { RawDetection(boxAt(0.10f + it * 0.06f, 0.55f), navyRgb, kitSummaryLab7 = navyKit, lowerBellyLuma = 34f, pantsDarkness = 0.78f) } +
            (0 until 5).map { RawDetection(boxAt(0.48f + it * 0.06f, 0.55f), whiteRgb, kitSummaryLab7 = whiteKit, lowerBellyLuma = 172f, pantsDarkness = 0.64f) } +
            listOf(
                RawDetection(boxAt(0.84f, 0.36f), officialGrayRgb, stripeScore = 0.05f, kitSummaryLab7 = officialGrayKit, lowerBellyLuma = 124f, pantsDarkness = 0.98f),
                RawDetection(boxAt(0.92f, 0.36f), listOf(90, 84, 68), stripeScore = 0.05f, kitSummaryLab7 = floatArrayOf(15.9f, 3.5f, 11.5f, 43.3f, 2.0f, 12.0f, 77.0f), lowerBellyLuma = 85f, pantsDarkness = 0.84f)
            )

        var latest: FrameMarkSample? = null
        for (i in 0 until 5) {
            val frameDets = shift2Base + listOf(
                RawDetection(boxAt(0.20f, 0.38f), listOf(38, 62, 165), kitSummaryLab7 = t34BoardOutlierKit, lowerBellyLuma = 65f),
                RawDetection(boxAt(0.76f, 0.38f), listOf(245, 250, 238), kitSummaryLab7 = brightOutlierKit, lowerBellyLuma = 240f)
            )
            latest = tracker.updateFrame(5.0 + i * 0.25, frameDets, 0.93f)
        }

        val frame = latest!!
        val navyPlayers = frame.players.subList(0, 5)
        val whitePlayers = frame.players.subList(5, 10)
        val refPlayers = frame.players.subList(10, 12)
        assertTrue(navyPlayers.all { it.team == TeamId.TEAM_A }, "All 5 dark Navy skaters must stay TEAM_A despite T34 & bright outliers: ${navyPlayers.map { it.team }}")
        assertTrue(whitePlayers.all { it.team == TeamId.TEAM_B }, "All 5 White skaters must stay TEAM_B despite T34 & bright outliers: ${whitePlayers.map { it.team }}")
        assertTrue(refPlayers.all { it.team == TeamId.REFEREE }, "Both far-board officials must remain REFEREE: ${refPlayers.map { it.team }}")

        // Symmetric YouTube #74 regression: 6 Dark + 6 White + 1 extreme bright/chroma outlier must remain 6 vs 6.
        tracker.reset()
        var ytLatest: FrameMarkSample? = null
        val ytDets = (0 until 6).map { RawDetection(boxAt(0.08f + it * 0.06f, 0.50f), navyRgb, kitSummaryLab7 = navyKit, lowerBellyLuma = 36f) } +
            (0 until 6).map { RawDetection(boxAt(0.46f + it * 0.06f, 0.50f), whiteRgb, kitSummaryLab7 = whiteKit, lowerBellyLuma = 176f) } +
            listOf(RawDetection(boxAt(0.90f, 0.38f), listOf(248, 252, 230), kitSummaryLab7 = brightOutlierKit, lowerBellyLuma = 242f))
        for (i in 0 until 3) {
            ytLatest = tracker.updateFrame(i * 0.25, ytDets, 0.91f)
        }
        assertEquals(6, ytLatest!!.players.subList(0, 6).count { it.team == TeamId.TEAM_A })
        assertEquals(6, ytLatest.players.subList(6, 12).count { it.team == TeamId.TEAM_B })

        // Chromatic Red vs White game (5 Red C*~70 vs 5 White C*~2) must not drop Red from seeding.
        tracker.reset()
        val redRgb = listOf(195, 34, 38)
        val redKit = floatArrayOf(42f, 58f, 34f, 46f, 60f, 36f, 15f)
        var redLatest: FrameMarkSample? = null
        val redWhiteDets = (0 until 5).map { RawDetection(boxAt(0.10f + it * 0.07f, 0.50f), redRgb, kitSummaryLab7 = redKit, lowerBellyLuma = 95f) } +
            (0 until 5).map { RawDetection(boxAt(0.50f + it * 0.07f, 0.50f), whiteRgb, kitSummaryLab7 = whiteKit, lowerBellyLuma = 180f) }
        for (i in 0 until 3) {
            redLatest = tracker.updateFrame(i * 0.25, redWhiteDets, 0.92f)
        }
        assertEquals(5, redLatest!!.players.count { it.team == TeamId.TEAM_A })
        assertEquals(5, redLatest.players.count { it.team == TeamId.TEAM_B })
    }

    @Test
    fun `referee vote persistence keeps established officials REFEREE across late board and crowd overlap`() {
        val tracker = MultiTeamPlayerTracker()
        val navyRgb = listOf(22, 38, 78)
        val navyKit = floatArrayOf(18f, 6f, -22f, 40f, 3f, -10f, 12f)
        val whiteRgb = listOf(225, 228, 232)
        val whiteKit = floatArrayOf(26f, 4f, -18f, 88f, -1f, -1f, 75f)
        val officialGrayRgb = listOf(128, 130, 127)
        val officialGrayKit = floatArrayOf(34f, -1f, 1f, 66f, 0f, 1f, 45f)

        val history = ArrayList<FrameMarkSample>()
        // Frames 0..4: establish REF_C & REF_L cleanly (colorCount >= 2 on frame 1 -> 4 referee votes >= 3 by frame 4).
        for (i in 0..4) {
            val dets = (0 until 4).map { RawDetection(boxAt(0.12f + it * 0.08f, 0.50f), navyRgb, kitSummaryLab7 = navyKit, lowerBellyLuma = 35f, pantsDarkness = 0.78f) } +
                (0 until 4).map { RawDetection(boxAt(0.50f + it * 0.08f, 0.50f), whiteRgb, kitSummaryLab7 = whiteKit, lowerBellyLuma = 182f, pantsDarkness = 0.65f) } +
                listOf(
                    RawDetection(boxAt(0.42f, 0.34f), officialGrayRgb, stripeScore = 0.05f, kitSummaryLab7 = officialGrayKit, lowerBellyLuma = 122f, pantsDarkness = 0.99f),
                    RawDetection(boxAt(0.88f, 0.35f), listOf(92, 82, 65), stripeScore = 0.06f, kitSummaryLab7 = floatArrayOf(15.9f, 3.5f, 11.5f, 43.3f, 2.0f, 12.0f, 77.0f), lowerBellyLuma = 85f, pantsDarkness = 0.84f)
                )
            history.add(tracker.updateFrame(i * 0.25, dets, 0.92f))
        }
        val refCId = history.last().players[8].trackId
        val refLId = history.last().players[9].trackId
        assertEquals(TeamId.REFEREE, history.last().players[8].team)
        assertEquals(TeamId.REFEREE, history.last().players[9].team)

        // Frames 5..7: late board/crowd overlap temporarily washes pantsDarkness down & brings bright crowd into belly,
        // yet meanChroma < 15f & skaterVotes <= 2 * refVotes so both stay REFEREE in live AND marked annotateHistory!
        for (i in 5..7) {
            val dets = (0 until 4).map { RawDetection(boxAt(0.12f + it * 0.08f, 0.50f), navyRgb, kitSummaryLab7 = navyKit, lowerBellyLuma = 35f, pantsDarkness = 0.78f) } +
                (0 until 4).map { RawDetection(boxAt(0.50f + it * 0.08f, 0.50f), whiteRgb, kitSummaryLab7 = whiteKit, lowerBellyLuma = 182f, pantsDarkness = 0.65f) } +
                listOf(
                    RawDetection(boxAt(0.42f, 0.34f), listOf(148, 150, 152), stripeScore = 0.02f, kitSummaryLab7 = whiteKit, lowerBellyLuma = 160f, pantsDarkness = 0.25f),
                    RawDetection(boxAt(0.88f, 0.35f), listOf(78, 82, 92), stripeScore = 0.02f, kitSummaryLab7 = navyKit, lowerBellyLuma = 60f, pantsDarkness = 0.25f)
                )
            history.add(tracker.updateFrame(i * 0.25, dets, 0.92f))
        }

        val annotated = tracker.annotateHistory(history)
        assertTrue(annotated.all { f -> f.players.first { it.trackId == refCId }.team == TeamId.REFEREE },
            "Established REF_C must stay REFEREE across late crowd overlap")
        assertTrue(annotated.all { f -> f.players.first { it.trackId == refLId }.team == TeamId.REFEREE },
            "Established REF_L must stay REFEREE across late board overlap")
    }

    @Test
    fun `1s time-bin deduplication keeps repeated within-1s sub-threshold ONNX reads tentative until a second bin confirms`() {
        val tracker = MultiTeamPlayerTracker()
        val dark = listOf(22, 38, 76)
        val light = listOf(232, 234, 236)

        val history = ArrayList<FrameMarkSample>()
        for (i in 0..4) {
            history.add(
                tracker.updateFrame(
                    i * 0.4,
                    listOf(
                        RawDetection(boxAt(0.30f, 0.52f), light),
                        RawDetection(boxAt(0.60f, 0.52f), light),
                        RawDetection(boxAt(0.85f, 0.52f), dark)
                    ),
                    0.92f
                )
            )
        }
        val fake77TrackId = history.first().players[0].trackId // e.g. #97 getting 3 sub-second 0.55 reads of #77
        val real77TrackId = history.first().players[1].trackId // real #77 getting multi-bin strong reads

        // 3 within-1s reads (t=0.04, 0.56, 0.92 -> all floor() == 0) at conf=0.55f (< 0.80f):
        // 1.0s bin deduplication discounts 2nd/3rd to 0.15x and single-bin < 0.80f stays tentative (null)!
        tracker.recordJerseyDistribution(fake77TrackId, 0.04, listOf("77" to 0.55f), source = JerseyReadSource.ONNX_STR)
        tracker.recordJerseyDistribution(fake77TrackId, 0.56, listOf("77" to 0.55f), source = JerseyReadSource.ONNX_STR)
        tracker.recordJerseyDistribution(fake77TrackId, 0.92, listOf("77" to 0.55f), source = JerseyReadSource.ONNX_STR)

        val tentativeFrame = tracker.annotateHistory(history).last().players
        assertNull(
            tentativeFrame.first { it.trackId == fake77TrackId }.jerseyNumber,
            "Three sub-1s 0.55f reads in a single 1.0s bin must stay tentative (null) and not lock #77"
        )

        // Now give real #77 a corroborated read -> real #77 cleanly claims #77 without being blocked!
        tracker.recordJerseyDistribution(real77TrackId, 0.20, listOf("77" to 0.88f), source = JerseyReadSource.ONNX_STR)
        val afterReal = tracker.annotateHistory(history).last().players
        assertEquals("77", afterReal.first { it.trackId == real77TrackId }.jerseyNumber)

        // Meanwhile, on an independent track, 2 reads of #97 at conf=0.55f across 2 distinct 1.0s bins (t=0.20s, t=1.40s) DOES commit #97!
        val twoBinInputs = listOf(
            ChainNumberInput(
                chainId = 401,
                firstSeenSec = 0.0,
                lastSeenSec = 3.0,
                observations = listOf(
                    JerseyReadObservation(0.20, null, 0.55f, JerseyReadKind.FULL, 1f, JerseyReadSource.ONNX_STR, listOf("97" to 0.55f)),
                    JerseyReadObservation(1.40, null, 0.55f, JerseyReadKind.FULL, 1f, JerseyReadSource.ONNX_STR, listOf("97" to 0.55f))
                ),
                bestConfidenceByNumber = mapOf("97" to 0.55f)
            )
        )
        val twoBinRes = JerseyNumberPosterior.assignNumbersJointly(twoBinInputs)
        assertEquals("97", twoBinRes[401]?.number, "Two distinct 1.0s bins at 0.55f ONNX_STR must commit #97")
    }
}
