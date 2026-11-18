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
    val videoDurationSec: Double = 0.0
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
