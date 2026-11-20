package com.engtestbench.hockeynano.core

import com.engtestbench.hockeynano.core.bench.BenchSeedReport
import com.engtestbench.hockeynano.core.model.AnalysisDurationPreset
import com.engtestbench.hockeynano.core.model.ClipWindow30s
import com.engtestbench.hockeynano.core.model.DiscoveredPlayerSummary
import com.engtestbench.hockeynano.core.model.FrameMarkSample
import com.engtestbench.hockeynano.core.model.PipelineState
import com.engtestbench.hockeynano.core.model.PipelineTelemetry
import com.engtestbench.hockeynano.core.model.RectNorm
import com.engtestbench.hockeynano.core.model.TeamId
import com.engtestbench.hockeynano.core.model.TeamProfile
import com.engtestbench.hockeynano.core.model.TrackedPlayer
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.io.File

class HockeyModelsAndBenchSerializationTest {

    private val json = Json { ignoreUnknownKeys = true }

    @Test
    fun `rect norm computed geometry and clip duration work`() {
        val rect = RectNorm(0.10f, 0.20f, 0.30f, 0.60f)
        assertEquals(0.20f, rect.width, 1e-5f)
        assertEquals(0.40f, rect.height, 1e-5f)
        assertEquals(0.20f, rect.centerX, 1e-5f)
        assertEquals(0.40f, rect.centerY, 1e-5f)
        assertEquals(0.60f, rect.footY, 1e-5f)

        val degenerate = RectNorm(0.5f, 0.5f, 0.4f, 0.4f)
        assertEquals(0f, degenerate.width, 1e-5f)
        assertEquals(0f, degenerate.height, 1e-5f)

        val clip = ClipWindow30s(
            id = "c1",
            playerNumber = "97",
            team = TeamId.TEAM_A,
            startSec = 15.0,
            endSec = 45.0,
            visibilitySec = 12.5,
            sampleHits = 50,
            avgLiveProb = 0.91f
        )
        assertEquals(30.0, clip.durationSec, 1e-6)
    }

    @Test
    fun `pipeline state round trips cleanly with kotlinx serialization`() {
        val player = TrackedPlayer(
            trackId = 7,
            box = RectNorm(0.40f, 0.45f, 0.44f, 0.56f),
            team = TeamId.TEAM_A,
            jerseyNumber = "97",
            numberConfidence = 0.93f,
            nanoVerified = true,
            chestRgb = listOf(28, 72, 44),
            speedNormPerSec = 0.14f,
            trail = listOf(0.41f to 0.55f, 0.42f to 0.56f)
        )
        val frame = FrameMarkSample(timestampSec = 12.5, livePlayProb = 0.88f, players = listOf(player))
        val state = PipelineState(
            videoTitle = "dublin_iceland_game_180s.mp4",
            videoSource = "/sdcard/Download/HockeyClips/dublin_iceland_game_180s.mp4",
            teams = mapOf(
                TeamId.TEAM_A to TeamProfile(TeamId.TEAM_A, "Dark Green", listOf(28, 72, 44), "Dark green", true),
                TeamId.TEAM_B to TeamProfile(TeamId.TEAM_B, "White", listOf(232, 234, 236), "White", true)
            ),
            currentFrame = frame,
            history = listOf(frame),
            discoveredPlayers = listOf(
                DiscoveredPlayerSummary("97", TeamId.TEAM_A, setOf(7, 11), 18.5, 0.93f, true, 2)
            ),
            selectedPlayerNumber = "97",
            selectedTeamFilter = TeamId.TEAM_A,
            telemetry = PipelineTelemetry(nanoStatus = "ready", fps = 5.2f, panoramicTilingEnabled = true)
        )

        val encoded = json.encodeToString(state)
        val decoded = json.decodeFromString<PipelineState>(encoded)
        assertEquals(state, decoded)
    }

    @Test
    fun `bench summary seed json deserializes into BenchSeedReport`() {
        val candidates = listOf(
            File("../app/src/main/assets/bench_summary_seed.json"),
            File("app/src/main/assets/bench_summary_seed.json"),
            File("../../bench_summary_seed.json")
        )
        val seedFile = candidates.firstOrNull { it.exists() }
        assertTrue(seedFile != null, "bench_summary_seed.json should exist in app assets")

        val report = json.decodeFromString<BenchSeedReport>(seedFile!!.readText())
        assertEquals("Engineering Test Bench", report.title)
        assertEquals(2468, report.totalTrials)
        assertEquals(27, report.taskCount)
        assertEquals(11, report.harnesses.size)
        assertEquals(9, report.models.size)
        assertEquals(89, report.cells.size)
        assertEquals(28, report.designsCount)
        assertTrue(report.findings.isNotEmpty())
    }

    @Test
    fun `analysis duration preset fromToken parses presets custom durations and invalid tokens`() {
        assertEquals(AnalysisDurationPreset.MIN_5 to 300.0, AnalysisDurationPreset.fromToken("5m"))
        assertEquals(AnalysisDurationPreset.MIN_5 to 300.0, AnalysisDurationPreset.fromToken("5min"))
        assertEquals(AnalysisDurationPreset.MIN_5 to 300.0, AnalysisDurationPreset.fromToken("5 min"))
        assertEquals(AnalysisDurationPreset.MIN_5 to 300.0, AnalysisDurationPreset.fromToken(" 5 MIN "))

        assertEquals(AnalysisDurationPreset.MIN_10 to 600.0, AnalysisDurationPreset.fromToken("10m"))
        assertEquals(AnalysisDurationPreset.MIN_10 to 600.0, AnalysisDurationPreset.fromToken("10min"))
        assertEquals(AnalysisDurationPreset.MIN_10 to 600.0, AnalysisDurationPreset.fromToken("10 min"))

        assertEquals(AnalysisDurationPreset.HOUR_30 to 108_000.0, AnalysisDurationPreset.fromToken("30h"))
        assertEquals(AnalysisDurationPreset.HOUR_30 to 108_000.0, AnalysisDurationPreset.fromToken("30hr"))
        assertEquals(AnalysisDurationPreset.HOUR_30 to 108_000.0, AnalysisDurationPreset.fromToken("30 hr"))
        assertEquals(AnalysisDurationPreset.HOUR_30 to 108_000.0, AnalysisDurationPreset.fromToken("30hour"))
        assertEquals(AnalysisDurationPreset.HOUR_30 to 108_000.0, AnalysisDurationPreset.fromToken("30 hours"))

        assertEquals(AnalysisDurationPreset.ALL to null, AnalysisDurationPreset.fromToken("all"))
        assertEquals(AnalysisDurationPreset.ALL to null, AnalysisDurationPreset.fromToken("ALL"))
        assertEquals(AnalysisDurationPreset.ALL to null, AnalysisDurationPreset.fromToken("full"))
        assertEquals(AnalysisDurationPreset.ALL to null, AnalysisDurationPreset.fromToken("unlimited"))
        assertEquals(AnalysisDurationPreset.ALL to null, AnalysisDurationPreset.fromToken("none"))

        assertEquals(null to 45.0, AnalysisDurationPreset.fromToken("45s"))
        assertEquals(null to 45.0, AnalysisDurationPreset.fromToken("45"))
        assertEquals(null to 900.0, AnalysisDurationPreset.fromToken("15m"))
        assertEquals(null to 900.0, AnalysisDurationPreset.fromToken("15min"))
        assertEquals(null to 7200.0, AnalysisDurationPreset.fromToken("2h"))
        assertEquals(null to 7200.0, AnalysisDurationPreset.fromToken("2hr"))

        assertEquals(null to null, AnalysisDurationPreset.fromToken(null))
        assertEquals(null to null, AnalysisDurationPreset.fromToken(""))
        assertEquals(null to null, AnalysisDurationPreset.fromToken("   "))
        assertEquals(null to null, AnalysisDurationPreset.fromToken("0"))
        assertEquals(null to null, AnalysisDurationPreset.fromToken("-5s"))
        assertEquals(null to null, AnalysisDurationPreset.fromToken("invalid"))
    }

    @Test
    fun `pipeline telemetry round trips budget fields and preserves defaults`() {
        val telemetry = PipelineTelemetry(
            nanoStatus = "ready",
            processedSec = 645.0,
            videoDurationSec = 5489.0,
            analysisStartSec = 600.0,
            analyzedSpanSec = 45.0,
            analysisLimitSec = 45.0,
            analysisBudgetReached = true
        )
        val encoded = json.encodeToString(telemetry)
        val decoded = json.decodeFromString<PipelineTelemetry>(encoded)
        assertEquals(telemetry, decoded)

        val legacyDecoded = json.decodeFromString<PipelineTelemetry>("""{"nanoStatus":"ready"}""")
        assertEquals(0.0, legacyDecoded.analysisStartSec)
        assertEquals(0.0, legacyDecoded.analyzedSpanSec)
        assertEquals(null, legacyDecoded.analysisLimitSec)
        assertEquals(false, legacyDecoded.analysisBudgetReached)
    }
}
