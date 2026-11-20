package com.engtestbench.hockeynano.core.model

import kotlinx.serialization.Serializable

@Serializable
enum class TeamId {
    TEAM_A,
    TEAM_B,
    REFEREE,
    UNKNOWN
}

@Serializable
data class RectNorm(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
) {
    val width: Float get() = (x2 - x1).coerceAtLeast(0f)
    val height: Float get() = (y2 - y1).coerceAtLeast(0f)
    val centerX: Float get() = (x1 + x2) * 0.5f
    val centerY: Float get() = (y1 + y2) * 0.5f
    val footY: Float get() = y2
}

@Serializable
data class TeamProfile(
    val id: TeamId,
    val displayName: String,
    val meanRgb: List<Int>,
    val kitSummary: String = "",
    val nanoVerified: Boolean = false
)

@Serializable
data class TrackedPlayer(
    val trackId: Int,
    val box: RectNorm,
    val team: TeamId,
    val jerseyNumber: String? = null,
    val numberConfidence: Float = 0f,
    val nanoVerified: Boolean = false,
    val roleHint: String? = null,
    val chestRgb: List<Int> = listOf(128, 128, 128),
    val speedNormPerSec: Float = 0f,
    val trail: List<Pair<Float, Float>> = emptyList()
)

@Serializable
data class FrameMarkSample(
    val timestampSec: Double,
    val livePlayProb: Float = 1f,
    val players: List<TrackedPlayer> = emptyList()
)

@Serializable
data class ClipWindow30s(
    val id: String,
    val playerNumber: String,
    val team: TeamId,
    val startSec: Double,
    val endSec: Double,
    val visibilitySec: Double,
    val sampleHits: Int,
    val avgLiveProb: Float,
    val nanoSummary: String? = null,
    val fastClipPath: String? = null,
    val markedClipPath: String? = null
) {
    val durationSec: Double get() = (endSec - startSec).coerceAtLeast(0.0)
}

@Serializable
data class DiscoveredPlayerSummary(
    val jerseyNumber: String,
    val team: TeamId,
    val trackIds: Set<Int>,
    val totalSeenSec: Double,
    val bestConfidence: Float,
    val nanoVerified: Boolean,
    val clipCount30s: Int
)

@Serializable
enum class AnalysisDurationPreset(val label: String, val seconds: Double?) {
    MIN_5("5 min", 300.0),
    MIN_10("10 min", 600.0),
    HOUR_30("30 hr", 108_000.0),
    ALL("All", null);

    companion object {
        private val CUSTOM_DURATION_REGEX =
            Regex("""^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours)?$""")

        fun fromToken(token: String?): Pair<AnalysisDurationPreset?, Double?> {
            val raw = token?.trim()?.lowercase() ?: return null to null
            if (raw.isEmpty()) return null to null

            val compact = raw.replace(Regex("""\s+"""), "")
            when (compact) {
                "5m", "5min", "5mins", "5minute", "5minutes" -> return MIN_5 to MIN_5.seconds
                "10m", "10min", "10mins", "10minute", "10minutes" -> return MIN_10 to MIN_10.seconds
                "30h", "30hr", "30hrs", "30hour", "30hours" -> return HOUR_30 to HOUR_30.seconds
                "all", "full", "unlimited", "none" -> return ALL to null
            }

            val match = CUSTOM_DURATION_REGEX.matchEntire(compact) ?: return null to null
            val numeric = match.groupValues[1].toDoubleOrNull() ?: return null to null
            if (!numeric.isFinite() || numeric <= 0.0) return null to null

            val multiplier = when (match.groupValues[2]) {
                "", "s", "sec", "secs", "second", "seconds" -> 1.0
                "m", "min", "mins", "minute", "minutes" -> 60.0
                "h", "hr", "hrs", "hour", "hours" -> 3600.0
                else -> return null to null
            }
            val totalSeconds = numeric * multiplier
            if (!totalSeconds.isFinite() || totalSeconds <= 0.0) return null to null
            return null to totalSeconds
        }
    }
}

@Serializable
data class PipelineTelemetry(
    val nanoStatus: String = "initializing",
    val fps: Float = 0f,
    val detectorMs: Float = 0f,
    val ocrReadsTotal: Int = 0,
    val nanoCallsTotal: Int = 0,
    val lastNanoLatencyMs: Long = 0L,
    val lastNanoMessage: String = "",
    val panoramicTilingEnabled: Boolean = true,
    val processedSec: Double = 0.0,
    val videoDurationSec: Double = 0.0,
    val analysisStartSec: Double = 0.0,
    val analyzedSpanSec: Double = 0.0,
    val analysisLimitSec: Double? = null,
    val analysisBudgetReached: Boolean = false,
    val framesAnalyzed: Int = 0
)

@Serializable
data class PipelineState(
    val videoTitle: String = "",
    val videoSource: String = "",
    val teams: Map<TeamId, TeamProfile> = emptyMap(),
    val currentFrame: FrameMarkSample? = null,
    val history: List<FrameMarkSample> = emptyList(),
    val discoveredPlayers: List<DiscoveredPlayerSummary> = emptyList(),
    val selectedPlayerNumber: String? = null,
    val selectedTeamFilter: TeamId? = null,
    val selectedPlayerClips: List<ClipWindow30s> = emptyList(),
    val telemetry: PipelineTelemetry = PipelineTelemetry()
)
