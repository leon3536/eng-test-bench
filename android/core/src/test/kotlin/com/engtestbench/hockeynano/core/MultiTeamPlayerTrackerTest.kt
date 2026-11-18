package com.engtestbench.hockeynano.core

import com.engtestbench.hockeynano.core.model.FrameMarkSample
import com.engtestbench.hockeynano.core.model.RectNorm
import com.engtestbench.hockeynano.core.model.TeamId
import com.engtestbench.hockeynano.core.tracking.MultiTeamPlayerTracker
import com.engtestbench.hockeynano.core.tracking.RawDetection
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class MultiTeamPlayerTrackerTest {

    private fun boxAt(cx: Float, footY: Float, w: Float = 0.04f, h: Float = 0.10f): RectNorm =
        RectNorm(cx - w * 0.5f, footY - h, cx + w * 0.5f, footY)

    @Test
    fun `isolated skater survives single-frame specular shadow color shift without over-splitting`() {
        val tracker = MultiTeamPlayerTracker()
        val white = listOf(230, 232, 235)
        val shadowGlare = listOf(72, 78, 84)
        val darkOpponent = listOf(25, 65, 38)

        val ids = ArrayList<Int>()
        for (i in 0..5) {
            val t = i * 0.25
            val color = if (i == 3) shadowGlare else white
            val frame = tracker.updateFrame(
                timestampSec = t,
                detections = listOf(
                    RawDetection(boxAt(0.25f + i * 0.025f, 0.55f), color),
                    RawDetection(boxAt(0.80f, 0.55f), darkOpponent)
                ),
                livePlayProb = 0.92f
            )
            ids.add(frame.players.first().trackId)
        }

        assertEquals(1, ids.toSet().size, "Isolated skater should keep one trackId across 1-frame shadow: $ids")
    }

    @Test
    fun `contact-only kit gating prevents ID swap when dark and light skaters cross closely`() {
        val tracker = MultiTeamPlayerTracker()
        val dark = listOf(22, 62, 34)
        val light = listOf(235, 236, 238)

        // Frame 0 & 1: establish Team A dark moving right, Team B light moving left.
        val f0 = tracker.updateFrame(
            0.0,
            listOf(
                RawDetection(boxAt(0.40f, 0.55f), dark),
                RawDetection(boxAt(0.60f, 0.55f), light)
            ),
            0.95f
        )
        val darkId = f0.players[0].trackId
        val lightId = f0.players[1].trackId

        tracker.updateFrame(
            0.25,
            listOf(
                RawDetection(boxAt(0.46f, 0.55f), dark),
                RawDetection(boxAt(0.54f, 0.55f), light)
            ),
            0.95f
        )

        // Frame 2: skaters cross and overlap (within 0.08 norm / IoU >= 0.08),
        // present detections in swapped list order to ensure matching is not order-dependent.
        val f2 = tracker.updateFrame(
            0.50,
            listOf(
                RawDetection(boxAt(0.49f, 0.55f), light),
                RawDetection(boxAt(0.51f, 0.55f), dark)
            ),
            0.95f
        )
        val f3 = tracker.updateFrame(
            0.75,
            listOf(
                RawDetection(boxAt(0.44f, 0.55f), light),
                RawDetection(boxAt(0.56f, 0.55f), dark)
            ),
            0.95f
        )

        val lightAfter = f3.players.first { it.box.centerX < 0.50f }
        val darkAfter = f3.players.first { it.box.centerX > 0.50f }
        assertEquals(lightId, lightAfter.trackId, "Light skater ID must not swap onto dark skater during contact")
        assertEquals(darkId, darkAfter.trackId, "Dark skater ID must not swap onto light skater during contact")
        assertEquals(TeamId.TEAM_B, lightAfter.team)
        assertEquals(TeamId.TEAM_A, darkAfter.team)
        assertEquals(2, f2.players.size)
    }

    @Test
    fun `weighted 2-means clusters teams and referee outlier only triggers above 10 skaters`() {
        val tracker = MultiTeamPlayerTracker()
        val dark = listOf(22, 68, 38)
        val light = listOf(232, 234, 236)
        val refGray = listOf(128, 128, 128)

        // 8 skaters (<= 10): neutral gray should not be classified as REFEREE.
        val smallList = (0 until 4).map { RawDetection(boxAt(0.15f + it * 0.07f, 0.45f), dark) } +
            (0 until 3).map { RawDetection(boxAt(0.50f + it * 0.07f, 0.45f), light) } +
            listOf(RawDetection(boxAt(0.75f, 0.65f), refGray))
        val fSmall = tracker.updateFrame(0.0, smallList, 0.9f)
        assertFalse(fSmall.players.any { it.team == TeamId.REFEREE }, "Referee outlier requires >10 skaters")

        // Reset and run 12 skaters (> 10): neutral gray outlier becomes REFEREE.
        tracker.reset()
        val largeList = (0 until 6).map { RawDetection(boxAt(0.10f + it * 0.07f, 0.45f), dark) } +
            (0 until 5).map { RawDetection(boxAt(0.52f + it * 0.07f, 0.45f), light) } +
            listOf(RawDetection(boxAt(0.45f, 0.72f), refGray))

        tracker.updateFrame(0.0, largeList, 0.9f)
        val fLarge = tracker.updateFrame(0.25, largeList, 0.9f)
        assertEquals(TeamId.REFEREE, fLarge.players.last().team)
        assertEquals(6, fLarge.players.count { it.team == TeamId.TEAM_A })
        assertEquals(5, fLarge.players.count { it.team == TeamId.TEAM_B })

        tracker.updateTeamProfile(TeamId.TEAM_A, "Dublin Green", "Dark green & white trim", true)
        val profiles = tracker.getTeamProfiles()
        assertEquals("Dublin Green", profiles[TeamId.TEAM_A]?.displayName)
        assertTrue(profiles[TeamId.TEAM_A]?.nanoVerified == true)
        assertTrue(profiles.containsKey(TeamId.REFEREE))
    }

    @Test
    fun `4s stitcher and 2_5x Nano weighting propagate number across stitched shift`() {
        val tracker = MultiTeamPlayerTracker()
        val dark = listOf(24, 64, 36)
        val light = listOf(234, 236, 238)
        val history = ArrayList<FrameMarkSample>()

        // Shift 1 (0.0..1.0s): track1 visible, OCR misreads #13 twice (weight = 2 * 0.6 = 1.2)
        var track1Id = -1
        for (i in 0..4) {
            val f = tracker.updateFrame(
                i * 0.25,
                listOf(
                    RawDetection(boxAt(0.30f + i * 0.01f, 0.55f), dark),
                    RawDetection(boxAt(0.80f, 0.55f), light)
                ),
                0.9f
            )
            track1Id = f.players.first().trackId
            history.add(f)
        }
        tracker.recordJerseyRead(track1Id, "13", confidence = 0.6f, fromNano = false)
        tracker.recordJerseyRead(track1Id, "13", confidence = 0.6f, fromNano = false)

        // Gap from 1.0s to 3.5s (2.5s occlusion <= 4.0s stitch window; > 1.0s active expiry)
        // Meanwhile keep opponent visible so time advances.
        for (t in listOf(1.5, 2.0, 2.5, 3.0)) {
            history.add(
                tracker.updateFrame(
                    t,
                    listOf(RawDetection(boxAt(0.80f, 0.55f), light)),
                    0.9f
                )
            )
        }

        // Shift 2 (3.5..4.5s): skater reappears nearby as a new active trackId,
        // and Nano reads #97 once at conf 0.90 (weight = 0.9 * 2.5 = 2.25 > 1.2!).
        var track2Id = -1
        for (i in 0..4) {
            val f = tracker.updateFrame(
                3.5 + i * 0.25,
                listOf(
                    RawDetection(boxAt(0.38f + i * 0.01f, 0.56f), dark),
                    RawDetection(boxAt(0.80f, 0.55f), light)
                ),
                0.95f
            )
            track2Id = f.players.first { it.box.centerX < 0.5f }.trackId
            history.add(f)
        }
        assertNotEquals(track1Id, track2Id, "Occlusion > 1.0s should create a fresh segment trackId")
        tracker.recordJerseyRead(track2Id, "97", confidence = 0.90f, fromNano = true)

        val annotated = tracker.annotateHistory(history)
        val shift1Players = annotated.take(5).map { it.players.first() }
        val shift2Players = annotated.takeLast(5).map { it.players.first { p -> p.box.centerX < 0.5f } }

        assertTrue(shift1Players.all { it.jerseyNumber == "97" && it.nanoVerified },
            "Nano read on stitched shift 2 must propagate #97 back to shift 1")
        assertTrue(shift2Players.all { it.jerseyNumber == "97" && it.nanoVerified })

        val discovered = tracker.getDiscoveredPlayers(annotated, videoDurationSec = 10.0)
        assertEquals(1, discovered.size)
        val p97 = discovered.first()
        assertEquals("97", p97.jerseyNumber)
        assertEquals(TeamId.TEAM_A, p97.team)
        assertEquals(setOf(track1Id, track2Id), p97.trackIds)
        assertTrue(p97.nanoVerified)
        assertEquals(1, p97.clipCount30s)
    }

    @Test
    fun `stitcher does not merge tracks across opposing teams or gaps beyond 4s`() {
        val tracker = MultiTeamPlayerTracker()
        val dark = listOf(20, 50, 35)
        val light = listOf(235, 235, 235)
        val history = ArrayList<FrameMarkSample>()

        val f0 = tracker.updateFrame(0.0, listOf(
            RawDetection(boxAt(0.40f, 0.55f), dark),
            RawDetection(boxAt(0.80f, 0.55f), light)
        ), 0.9f)
        history.add(f0)
        val idDark = f0.players[0].trackId

        // Advance past 4.0s gap (to t = 5.0s)
        for (t in listOf(1.5, 3.0, 4.5)) {
            history.add(tracker.updateFrame(t, listOf(RawDetection(boxAt(0.80f, 0.55f), light)), 0.9f))
        }
        val fLate = tracker.updateFrame(5.0, listOf(
            RawDetection(boxAt(0.42f, 0.55f), dark),
            RawDetection(boxAt(0.80f, 0.55f), light)
        ), 0.9f)
        history.add(fLate)
        val idLate = fLate.players.first { it.box.centerX < 0.5f }.trackId
        tracker.recordJerseyRead(idLate, "19", 0.9f, true)

        val annotated = tracker.annotateHistory(history)
        assertEquals(null, annotated.first().players.first { it.trackId == idDark }.jerseyNumber,
            "Tracks separated by >4.0s must not be stitched")

        // Manual override assigns number & team cleanly.
        tracker.assignTrackJerseyNumber(idDark, "88", TeamId.TEAM_B)
        val afterAssign = tracker.annotateHistory(history)
        val updated = afterAssign.first().players.first { it.trackId == idDark }
        assertEquals("88", updated.jerseyNumber)
        assertEquals(TeamId.TEAM_B, updated.team)

        // Two Nano-verified differing numbers within <=4.0s gap must NOT stitch together.
        tracker.reset()
        val h2 = ArrayList<FrameMarkSample>()
        val fa = tracker.updateFrame(0.0, listOf(
            RawDetection(boxAt(0.40f, 0.55f), dark),
            RawDetection(boxAt(0.80f, 0.55f), light)
        ), 0.9f)
        h2.add(fa)
        val idA = fa.players.first().trackId
        tracker.recordJerseyRead(idA, "97", 0.92f, fromNano = true)
        for (t in listOf(0.5, 1.2, 2.0)) {
            h2.add(tracker.updateFrame(t, listOf(RawDetection(boxAt(0.80f, 0.55f), light)), 0.9f))
        }
        val fb = tracker.updateFrame(2.5, listOf(
            RawDetection(boxAt(0.42f, 0.55f), dark),
            RawDetection(boxAt(0.80f, 0.55f), light)
        ), 0.9f)
        h2.add(fb)
        val idB = fb.players.first { it.box.centerX < 0.5f }.trackId
        tracker.recordJerseyRead(idB, "19", 0.92f, fromNano = true)

        val ann2 = tracker.annotateHistory(h2)
        assertEquals("97", ann2.first().players.first { it.trackId == idA }.jerseyNumber)
        assertEquals("19", ann2.last().players.first { it.trackId == idB }.jerseyNumber)
    }
}
