package com.engtestbench.hockeynano.core

import com.engtestbench.hockeynano.core.clip.PlayerClipSegmenter
import com.engtestbench.hockeynano.core.model.FrameMarkSample
import com.engtestbench.hockeynano.core.model.RectNorm
import com.engtestbench.hockeynano.core.model.TeamId
import com.engtestbench.hockeynano.core.model.TrackedPlayer
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class PlayerClipSegmenterTest {

    private fun sampleWith(t: Double, number: String, team: TeamId = TeamId.TEAM_A, live: Float = 0.9f): FrameMarkSample =
        FrameMarkSample(
            timestampSec = t,
            livePlayProb = live,
            players = listOf(
                TrackedPlayer(
                    trackId = 1,
                    box = RectNorm(0.40f, 0.45f, 0.44f, 0.55f),
                    team = team,
                    jerseyNumber = number,
                    numberConfidence = 0.9f
                )
            )
        )

    @Test
    fun `short video under 30s returns single clamped full window when visibility meets 1s`() {
        val history = (0 until 8).map { sampleWith(it * 0.25 + 5.0, "97") } // 2.0s visibility
        val windows = PlayerClipSegmenter.compute30sWindows(history, "97", null, videoDurationSec = 28.0)
        assertEquals(1, windows.size)
        assertEquals(0.0, windows[0].startSec, 1e-6)
        assertEquals(28.0, windows[0].endSec, 1e-6)
        assertEquals(2.0, windows[0].visibilitySec, 1e-6)

        val tooShort = (0 until 3).map { sampleWith(it * 0.25 + 5.0, "97") } // 0.75s < 1.0s
        assertTrue(PlayerClipSegmenter.compute30sWindows(tooShort, "97", null, videoDurationSec = 28.0).isEmpty())
    }

    @Test
    fun `multi-minute video segments non-overlapping 30s windows sorted chronologically`() {
        val history = ArrayList<FrameMarkSample>()
        // Shift 1: 10..20s (40 hits = 10s)
        for (i in 0 until 40) history.add(sampleWith(10.0 + i * 0.25, "97", TeamId.TEAM_A, live = 0.85f))
        // Tiny 0.5s blip at 45s (2 hits < 1.0s minimum visibility) -> must be ignored
        history.add(sampleWith(45.0, "97", TeamId.TEAM_A))
        history.add(sampleWith(45.25, "97", TeamId.TEAM_A))
        // Shift 2: 72..86s (56 hits = 14s)
        for (i in 0 until 56) history.add(sampleWith(72.0 + i * 0.25, "97", TeamId.TEAM_A, live = 0.95f))
        // Shift 3: 165..175s (40 hits = 10s, near end of 180s video)
        for (i in 0 until 40) history.add(sampleWith(165.0 + i * 0.25, "97", TeamId.TEAM_A, live = 0.80f))

        val windows = PlayerClipSegmenter.compute30sWindows(
            history = history,
            playerNumber = "#97",
            teamFilter = TeamId.TEAM_A,
            videoDurationSec = 180.0
        )

        assertEquals(3, windows.size, "Should find exactly 3 valid 30s shift windows")
        for (i in windows.indices) {
            val w = windows[i]
            assertEquals(30.0, w.durationSec, 1e-6)
            assertTrue(w.startSec >= 0.0 && w.endSec <= 180.0)
            assertTrue(w.visibilitySec >= 10.0 - 1e-6)
            if (i + 1 < windows.size) {
                assertTrue(w.endSec <= windows[i + 1].startSec + 1e-6, "Windows must be non-overlapping and chronological")
            }
        }
        // Last window should clamp cleanly inside [150.0, 180.0]
        assertTrue(windows.last().endSec <= 180.0)
        assertTrue(windows.last().startSec <= 150.0 + 1e-6)
    }

    @Test
    fun `team filter isolates players sharing the same jersey number on opposing teams`() {
        val history = ArrayList<FrameMarkSample>()
        for (i in 0 until 12) {
            history.add(sampleWith(5.0 + i * 0.25, "19", TeamId.TEAM_A))
            history.add(sampleWith(60.0 + i * 0.25, "19", TeamId.TEAM_B))
        }

        val aWindows = PlayerClipSegmenter.compute30sWindows(history, "19", TeamId.TEAM_A, 120.0)
        val bWindows = PlayerClipSegmenter.compute30sWindows(history, "19", TeamId.TEAM_B, 120.0)

        assertEquals(1, aWindows.size)
        assertEquals(TeamId.TEAM_A, aWindows[0].team)
        assertTrue(aWindows[0].startSec <= 5.0 && aWindows[0].endSec >= 7.75)

        assertEquals(1, bWindows.size)
        assertEquals(TeamId.TEAM_B, bWindows[0].team)
        assertTrue(bWindows[0].startSec <= 60.0 && bWindows[0].endSec >= 62.75)
    }

    @Test
    fun `scoring formula prefers higher livePlayProb when visibility ties`() {
        // Two isolated clusters 20s apart so only one 30s window can cover each or pick peak:
        // Cluster A at t=10..12s (live=0.2f), Cluster B at t=38..40s (live=0.98f), video=45s
        // Wait: [10,40] could cover both if 30s apart, so put them at t=1..3s and t=40..42s in 60s video -> both picked.
        // Instead put Cluster A at t=10..12s (8 hits, live=0.2f) and Cluster B at t=26..28s (8 hits, live=0.98f)
        // Wait: [0,30] covers both! Put Cluster A at t=2..4s (live=0.1f) and Cluster B at t=33..35s (live=0.99f)
        // Wait: [2,32] and [32,62] don't overlap. What if videoDuration=38s?
        // In a 38s video, any 30s window covers [s, s+30] where s in [0, 8].
        // Cluster A at t=1..3s (only in [0, 30]) vs Cluster B at t=35..37s (only in [8, 38]).
        val history = ArrayList<FrameMarkSample>()
        for (i in 0 until 8) {
            history.add(sampleWith(1.0 + i * 0.25, "11", TeamId.TEAM_A, live = 0.10f))
            history.add(sampleWith(35.0 + i * 0.25, "11", TeamId.TEAM_A, live = 0.99f))
        }
        val windows = PlayerClipSegmenter.compute30sWindows(history, "11", TeamId.TEAM_A, videoDurationSec = 38.0)
        assertEquals(1, windows.size)
        assertTrue(windows[0].avgLiveProb > 0.9f, "Window with higher livePlayProb must win tie in visibility")
        assertTrue(windows[0].startSec >= 7.0, "Window should capture high-livePlayProb shift at 35..37s")
    }
}
