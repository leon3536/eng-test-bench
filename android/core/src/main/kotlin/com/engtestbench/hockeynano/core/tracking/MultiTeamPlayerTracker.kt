package com.engtestbench.hockeynano.core.tracking

import com.engtestbench.hockeynano.core.clip.PlayerClipSegmenter
import com.engtestbench.hockeynano.core.model.DiscoveredPlayerSummary
import com.engtestbench.hockeynano.core.model.FrameMarkSample
import com.engtestbench.hockeynano.core.model.RectNorm
import com.engtestbench.hockeynano.core.model.TeamId
import com.engtestbench.hockeynano.core.model.TeamProfile
import com.engtestbench.hockeynano.core.model.TrackedPlayer
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class RawDetection(
    val box: RectNorm,
    val chestRgb: List<Int>,
    val stripeScore: Float = 0f
)

/**
 * Maintains temporal track segments, hierarchical 4.0s same-team stitching, and
 * weighted majority voting (with Gemini Nano VLM reads weighted 2.5x over OCR reads)
 * so a jersey read anywhere on a stitched shift propagates across the entire shift.
 */
class TrackIdentityRegistry {

    internal class TrackRecord(
        val trackId: Int,
        val firstSeenSec: Double,
        var lastSeenSec: Double,
        val firstBox: RectNorm,
        var lastBox: RectNorm,
        var lastVx: Float = 0f,
        var lastVy: Float = 0f,
        var hitCount: Int = 1,
        var sumR: Double = 0.0,
        var sumG: Double = 0.0,
        var sumB: Double = 0.0,
        var colorCount: Int = 0,
        var emaStripeScore: Float = 0f,
        var stripeObservations: Int = 0,
        var clusteredTeam: TeamId = TeamId.UNKNOWN,
        var teamOverride: TeamId? = null,
        var manualJerseyNumber: String? = null,
        val jerseyVotes: MutableMap<String, Float> = LinkedHashMap(),
        val bestConfidence: MutableMap<String, Float> = LinkedHashMap(),
        val nanoVerifiedNumbers: MutableSet<String> = LinkedHashSet()
    ) {
        fun addColor(rgb: List<Int>) {
            val r = rgb.getOrNull(0)?.coerceIn(0, 255) ?: return
            val g = rgb.getOrNull(1)?.coerceIn(0, 255) ?: return
            val b = rgb.getOrNull(2)?.coerceIn(0, 255) ?: return
            sumR += r
            sumG += g
            sumB += b
            colorCount++
        }

        fun addStripeScore(score: Float) {
            val clean = if (score.isFinite()) score.coerceIn(0f, 1f) else 0f
            emaStripeScore = if (stripeObservations == 0) {
                clean
            } else {
                0.8f * emaStripeScore + 0.2f * clean
            }
            stripeObservations++
        }

        fun meanRgb(): FloatArray? {
            if (colorCount <= 0) return null
            val n = colorCount.toFloat()
            return floatArrayOf((sumR / n).toFloat(), (sumG / n).toFloat(), (sumB / n).toFloat())
        }

        fun effectiveTeam(): TeamId = teamOverride ?: clusteredTeam
    }

    data class ResolvedIdentity(
        val team: TeamId,
        val jerseyNumber: String?,
        val confidence: Float,
        val nanoVerified: Boolean,
        val chainTrackIds: Set<Int>
    )

    private val records = LinkedHashMap<Int, TrackRecord>()
    private var cachedResolution: Map<Int, ResolvedIdentity>? = null

    fun clear() {
        records.clear()
        cachedResolution = null
    }

    internal fun getOrCreateRecord(
        trackId: Int,
        timestampSec: Double,
        box: RectNorm,
        chestRgb: List<Int>,
        stripeScore: Float = 0f
    ): TrackRecord {
        cachedResolution = null
        val existing = records[trackId]
        if (existing != null) {
            existing.lastSeenSec = max(existing.lastSeenSec, timestampSec)
            existing.lastBox = box
            existing.hitCount++
            existing.addColor(chestRgb)
            existing.addStripeScore(stripeScore)
            return existing
        }
        return TrackRecord(
            trackId = trackId,
            firstSeenSec = timestampSec,
            lastSeenSec = timestampSec,
            firstBox = box,
            lastBox = box
        ).also {
            it.addColor(chestRgb)
            it.addStripeScore(stripeScore)
            records[trackId] = it
        }
    }

    internal fun allRecords(): Collection<TrackRecord> = records.values

    internal fun invalidateCache() {
        cachedResolution = null
    }

    fun recordJerseyRead(
        trackId: Int,
        jerseyNumber: String,
        confidence: Float,
        fromNano: Boolean
    ) {
        val clean = normalizeJerseyNumber(jerseyNumber) ?: return
        val rec = records[trackId] ?: return
        val conf = confidence.coerceIn(0.05f, 1.0f)
        val weight = conf * if (fromNano) NANO_WEIGHT_MULTIPLIER else OCR_WEIGHT_MULTIPLIER
        rec.jerseyVotes[clean] = (rec.jerseyVotes[clean] ?: 0f) + weight
        rec.bestConfidence[clean] = max(rec.bestConfidence[clean] ?: 0f, conf)
        if (fromNano) {
            rec.nanoVerifiedNumbers.add(clean)
        }
        cachedResolution = null
    }

    fun assignTrackJerseyNumber(
        trackId: Int,
        jerseyNumber: String,
        teamOverride: TeamId? = null
    ) {
        val rec = records[trackId] ?: return
        val trimmed = jerseyNumber.trim().trimStart('#')
        if (trimmed.isEmpty()) {
            rec.manualJerseyNumber = null
        } else {
            val clean = normalizeJerseyNumber(trimmed) ?: trimmed
            rec.manualJerseyNumber = clean
            rec.bestConfidence[clean] = max(rec.bestConfidence[clean] ?: 0f, 1.0f)
        }
        if (teamOverride != null) {
            rec.teamOverride = teamOverride
        }
        cachedResolution = null
    }

    fun resolveAll(): Map<Int, ResolvedIdentity> {
        cachedResolution?.let { return it }
        if (records.isEmpty()) return emptyMap()

        // Build hierarchical temporal chains (Stage 1: <= 1.5s tight, Stage 2: <= 4.0s).
        val chains = buildStitchedChains()
        val resolved = HashMap<Int, ResolvedIdentity>(records.size)

        for (chainIds in chains) {
            val chainRecords = chainIds.mapNotNull { records[it] }
            val resolvedTeam = resolveChainTeam(chainRecords)
            val manualNum = chainRecords
                .sortedByDescending { it.lastSeenSec }
                .firstOrNull { !it.manualJerseyNumber.isNullOrBlank() }
                ?.manualJerseyNumber

            val aggregatedVotes = LinkedHashMap<String, Float>()
            val aggregatedConf = LinkedHashMap<String, Float>()
            val nanoNumbers = LinkedHashSet<String>()

            for (rec in chainRecords) {
                for ((num, score) in rec.jerseyVotes) {
                    aggregatedVotes[num] = (aggregatedVotes[num] ?: 0f) + score
                }
                for ((num, c) in rec.bestConfidence) {
                    aggregatedConf[num] = max(aggregatedConf[num] ?: 0f, c)
                }
                nanoNumbers.addAll(rec.nanoVerifiedNumbers)
            }

            val winningNumber: String?
            val winConf: Float
            val winNano: Boolean

            if (manualNum != null) {
                winningNumber = manualNum
                winConf = max(aggregatedConf[manualNum] ?: 1.0f, 1.0f)
                winNano = manualNum in nanoNumbers
            } else if (aggregatedVotes.isNotEmpty()) {
                val totalVotes = aggregatedVotes.values.sum()
                val topEntry = aggregatedVotes.entries.maxByOrNull { it.value }!!
                val topNum = topEntry.key
                val topScore = topEntry.value
                val isNano = topNum in nanoNumbers
                val bestC = aggregatedConf[topNum] ?: 0f
                val meetsSupport = (topScore * 2f >= totalVotes) || isNano || bestC >= 0.45f
                if (meetsSupport) {
                    winningNumber = topNum
                    winConf = bestC
                    winNano = isNano
                } else {
                    winningNumber = null
                    winConf = 0f
                    winNano = false
                }
            } else {
                winningNumber = null
                winConf = 0f
                winNano = false
            }

            val idSet = chainIds.toSet()
            val identity = ResolvedIdentity(
                team = resolvedTeam,
                jerseyNumber = winningNumber,
                confidence = winConf,
                nanoVerified = winNano,
                chainTrackIds = idSet
            )
            for (tid in idSet) {
                resolved[tid] = identity
            }
        }

        cachedResolution = resolved
        return resolved
    }

    private fun resolveChainTeam(chainRecords: List<TrackRecord>): TeamId {
        val override = chainRecords.lastOrNull { it.teamOverride != null }?.teamOverride
        if (override != null) return override

        val weights = LinkedHashMap<TeamId, Int>()
        for (rec in chainRecords) {
            val t = rec.clusteredTeam
            if (t != TeamId.UNKNOWN) {
                weights[t] = (weights[t] ?: 0) + rec.hitCount
            }
        }
        return weights.maxByOrNull { it.value }?.key ?: TeamId.UNKNOWN
    }

    private data class ChainNode(
        val trackIds: MutableList<Int>,
        var startSec: Double,
        var endSec: Double,
        var firstBox: RectNorm,
        var lastBox: RectNorm,
        var lastVx: Float,
        var lastVy: Float
    )

    private data class StitchCandidate(
        val prevIdx: Int,
        val nextIdx: Int,
        val cost: Float
    )

    private fun buildStitchedChains(): List<List<Int>> {
        val sorted = records.values.sortedBy { it.firstSeenSec }
        var nodes = sorted.map {
            ChainNode(
                trackIds = mutableListOf(it.trackId),
                startSec = it.firstSeenSec,
                endSec = it.lastSeenSec,
                firstBox = it.firstBox,
                lastBox = it.lastBox,
                lastVx = it.lastVx,
                lastVy = it.lastVy
            )
        }

        // Stage 1: short dropouts (<= 1.5s)
        nodes = stitchPass(nodes, maxGapSec = 1.5, maxDistCap = 0.24f, heightMultiplierBase = 2.4f, heightSlopePerSec = 1.6f)
        // Stage 2: longer occlusions / shifts (<= 4.0s)
        nodes = stitchPass(nodes, maxGapSec = MAX_STITCH_GAP_SEC, maxDistCap = 0.38f, heightMultiplierBase = 2.2f, heightSlopePerSec = 1.8f)

        return nodes.map { it.trackIds }
    }

    private fun stitchPass(
        nodes: List<ChainNode>,
        maxGapSec: Double,
        maxDistCap: Float,
        heightMultiplierBase: Float,
        heightSlopePerSec: Float
    ): List<ChainNode> {
        if (nodes.size < 2) return nodes
        val candidates = ArrayList<StitchCandidate>()

        for (i in nodes.indices) {
            val a = nodes[i]
            val teamA = resolveChainTeam(a.trackIds.mapNotNull { records[it] })
            val estA = establishedChainNumber(a.trackIds)
            val domA = dominantChainNumber(a.trackIds)

            for (j in nodes.indices) {
                if (i == j) continue
                val b = nodes[j]
                val gap = b.startSec - a.endSec
                if (gap <= 1e-6 || gap > maxGapSec) continue

                val teamB = resolveChainTeam(b.trackIds.mapNotNull { records[it] })
                if (!teamsCompatibleForStitch(teamA, teamB)) continue

                val estB = establishedChainNumber(b.trackIds)
                if (estA != null && estB != null && estA != estB) continue
                val domB = dominantChainNumber(b.trackIds)

                val avgH = max(0.03f, 0.5f * (a.lastBox.height + b.firstBox.height))
                val rawDist = hypot(a.lastBox.centerX - b.firstBox.centerX, a.lastBox.footY - b.firstBox.footY)
                val predDt = min(gap, 1.0).toFloat() * 0.5f
                val predX = a.lastBox.centerX + a.lastVx * predDt
                val predY = a.lastBox.footY + a.lastVy * predDt
                val predDist = hypot(predX - b.firstBox.centerX, predY - b.firstBox.footY)
                val bestDist = min(rawDist, predDist)

                val maxAllowed = min(maxDistCap, max(0.10f, (heightMultiplierBase + heightSlopePerSec * gap.toFloat()) * avgH))
                if (bestDist > maxAllowed) continue

                val numberAdjust = when {
                    domA != null && domB != null && domA == domB -> -0.04f
                    domA != null && domB != null && domA != domB -> 0.06f
                    else -> 0f
                }
                val cost = bestDist + 0.04f * gap.toFloat() + numberAdjust
                candidates.add(StitchCandidate(i, j, cost))
            }
        }

        if (candidates.isEmpty()) return nodes
        candidates.sortBy { it.cost }

        val nextOf = IntArray(nodes.size) { -1 }
        val prevOf = IntArray(nodes.size) { -1 }
        val compHead = IntArray(nodes.size) { it }
        val compTail = IntArray(nodes.size) { it }
        val compTeam = Array(nodes.size) { i -> resolveChainTeam(nodes[i].trackIds.mapNotNull { records[it] }) }
        val compEstNum = Array(nodes.size) { i -> establishedChainNumber(nodes[i].trackIds) }

        for (c in candidates) {
            if (nextOf[c.prevIdx] != -1 || prevOf[c.nextIdx] != -1) continue
            val hA = compHead[c.prevIdx]
            val hB = compHead[c.nextIdx]
            if (hA == hB) continue
            // Ensure transitive compatibility across merged chains (no conflicting established numbers/teams).
            val tA = compTeam[hA]
            val tB = compTeam[hB]
            if (!teamsCompatibleForStitch(tA, tB)) continue
            val nA = compEstNum[hA]
            val nB = compEstNum[hB]
            if (nA != null && nB != null && nA != nB) continue

            nextOf[c.prevIdx] = c.nextIdx
            prevOf[c.nextIdx] = c.prevIdx

            val mergedTeam = if (tA != TeamId.UNKNOWN) tA else tB
            val mergedNum = nA ?: nB
            val newTail = compTail[hB]
            compTail[hA] = newTail
            compTeam[hA] = mergedTeam
            compEstNum[hA] = mergedNum
            var walk = hB
            while (walk != -1) {
                compHead[walk] = hA
                walk = nextOf[walk]
            }
        }

        val merged = ArrayList<ChainNode>(nodes.size)
        for (i in nodes.indices) {
            if (prevOf[i] != -1) continue
            val root = nodes[i]
            val combinedIds = ArrayList(root.trackIds)
            var tail = root
            var cur = nextOf[i]
            while (cur != -1) {
                val n = nodes[cur]
                combinedIds.addAll(n.trackIds)
                tail = n
                cur = nextOf[cur]
            }
            merged.add(
                ChainNode(
                    trackIds = combinedIds,
                    startSec = root.startSec,
                    endSec = tail.endSec,
                    firstBox = root.firstBox,
                    lastBox = tail.lastBox,
                    lastVx = tail.lastVx,
                    lastVy = tail.lastVy
                )
            )
        }
        return merged
    }

    private fun teamsCompatibleForStitch(a: TeamId, b: TeamId): Boolean {
        if (a == TeamId.UNKNOWN || b == TeamId.UNKNOWN) return a != TeamId.REFEREE && b != TeamId.REFEREE
        return a == b
    }

    private fun establishedChainNumber(trackIds: List<Int>): String? {
        var manual: String? = null
        val votes = LinkedHashMap<String, Float>()
        val nanoNums = HashSet<String>()
        for (id in trackIds) {
            val rec = records[id] ?: continue
            if (!rec.manualJerseyNumber.isNullOrBlank()) manual = rec.manualJerseyNumber
            for ((num, w) in rec.jerseyVotes) {
                votes[num] = (votes[num] ?: 0f) + w
            }
            nanoNums.addAll(rec.nanoVerifiedNumbers)
        }
        if (manual != null) return manual
        if (votes.isEmpty()) return null
        val total = votes.values.sum()
        val top = votes.entries.maxByOrNull { it.value } ?: return null
        val hasStrongSupport = top.key in nanoNums || top.value >= 2.0f
        return if (hasStrongSupport && top.value * 2f >= total) top.key else null
    }

    private fun dominantChainNumber(trackIds: List<Int>): String? {
        var manual: String? = null
        val votes = LinkedHashMap<String, Float>()
        for (id in trackIds) {
            val rec = records[id] ?: continue
            if (!rec.manualJerseyNumber.isNullOrBlank()) manual = rec.manualJerseyNumber
            for ((num, w) in rec.jerseyVotes) {
                votes[num] = (votes[num] ?: 0f) + w
            }
        }
        if (manual != null) return manual
        if (votes.isEmpty()) return null
        val total = votes.values.sum()
        val top = votes.entries.maxByOrNull { it.value } ?: return null
        return if (top.value * 2f >= total) top.key else null
    }

    companion object {
        const val OCR_WEIGHT_MULTIPLIER = 1.0f
        const val NANO_WEIGHT_MULTIPLIER = 2.5f
        const val MAX_STITCH_GAP_SEC = 4.0

        fun normalizeJerseyNumber(raw: String): String? {
            val s = raw.trim().trimStart('#')
            if (s.isEmpty() || s.length > 2 || !s.all { it.isDigit() }) return null
            if (s == "0" || s == "00") return null
            val n = s.toIntOrNull() ?: return null
            if (n !in 1..99) return null
            return n.toString()
        }
    }
}

/**
 * Multi-team hockey player tracker combining:
 * - Constant-velocity + skate-foot distance + IoU frame association
 * - Contact-only kit compatibility gating (prevents 1-frame specular shadow over-splits)
 * - Weighted 2-means RGB+luminance team clustering + >10-skater referee outlier detection
 * - 4.0s same-team hierarchical stitching & Nano-weighted (2.5x) jersey voting
 */
class MultiTeamPlayerTracker {

    private class ActiveTrack(
        val id: Int,
        var box: RectNorm,
        var lastSeenSec: Double,
        var emaRgb: FloatArray,
        var emaStripeScore: Float = 0f,
        var vx: Float = 0f,
        var vy: Float = 0f,
        var misses: Int = 0,
        var hits: Int = 1,
        val trail: ArrayDeque<Pair<Float, Float>> = ArrayDeque()
    ) {
        init {
            trail.addLast(box.centerX to box.footY)
        }
    }

    private data class TeamOverrideMeta(
        val displayName: String,
        val kitSummary: String,
        val nanoVerified: Boolean
    )

    private val lock = Any()
    private val activeTracks = ArrayList<ActiveTrack>()
    private val registry = TrackIdentityRegistry()
    private val teamMetaOverrides = LinkedHashMap<TeamId, TeamOverrideMeta>()
    private var clusteredTeamRgb = LinkedHashMap<TeamId, List<Int>>()
    private var nextTrackId = 1
    private var lastFrameTimestampSec = Double.NaN

    fun reset() = synchronized(lock) {
        activeTracks.clear()
        registry.clear()
        teamMetaOverrides.clear()
        clusteredTeamRgb.clear()
        nextTrackId = 1
        lastFrameTimestampSec = Double.NaN
    }

    fun updateFrame(
        timestampSec: Double,
        detections: List<RawDetection>,
        livePlayProb: Float
    ): FrameMarkSample = synchronized(lock) {
        val safeTime = if (timestampSec.isFinite()) timestampSec else 0.0
        val safeLive = livePlayProb.takeIf { it.isFinite() }?.coerceIn(0f, 1f) ?: 1f

        if (!lastFrameTimestampSec.isNaN()) {
            val jump = safeTime - lastFrameTimestampSec
            if (jump > 3.0 || jump < -1.0) {
                activeTracks.clear()
            }
        }
        lastFrameTimestampSec = safeTime

        // Predict each active track forward by elapsed time since it was last seen.
        val predictedBoxes = activeTracks.map { t ->
            val dt = (safeTime - t.lastSeenSec).coerceIn(0.0, 1.0).toFloat()
            val dx = t.vx * dt
            val dy = t.vy * dt
            RectNorm(
                x1 = (t.box.x1 + dx).coerceIn(0f, 1f),
                y1 = (t.box.y1 + dy).coerceIn(0f, 1f),
                x2 = (t.box.x2 + dx).coerceIn(0f, 1f),
                y2 = (t.box.y2 + dy).coerceIn(0f, 1f)
            )
        }

        // Precompute contact flags per detection and per predicted track.
        val detInContact = BooleanArray(detections.size) { i ->
            detections.indices.any { j ->
                i != j && areCloseOrOverlapping(detections[i].box, detections[j].box)
            }
        }
        val trackInContact = BooleanArray(activeTracks.size) { i ->
            predictedBoxes.indices.any { j ->
                i != j && areCloseOrOverlapping(predictedBoxes[i], predictedBoxes[j])
            }
        }

        val candidates = ArrayList<Triple<Float, Int, Int>>() // (cost, trackIdx, detIdx)
        for (ti in activeTracks.indices) {
            val t = activeTracks[ti]
            val pred = predictedBoxes[ti]
            for (di in detections.indices) {
                val det = detections[di]
                val iou = computeIou(pred, det.box)
                val avgH = max(0.025f, 0.5f * (pred.height + det.box.height))
                val footDist = hypot(pred.centerX - det.box.centerX, pred.footY - det.box.footY)
                val normFootDist = footDist / avgH

                if (iou < MIN_MATCH_IOU && (normFootDist > MAX_NORM_FOOT_DIST || footDist > MAX_ABS_FOOT_DIST)) {
                    continue
                }

                val detRgb = toFloatRgb(det.chestRgb)
                val colorDist = rgbDistance(t.emaRgb, detRgb)
                val lumDiff = abs(luminance(t.emaRgb) - luminance(detRgb))

                val inContact = detInContact[di] || trackInContact[ti]
                if (inContact && isCrossKitIncompatible(t.emaRgb, detRgb, colorDist, lumDiff)) {
                    continue
                }

                val cost = normFootDist - 2.0f * iou + 0.0025f * colorDist
                candidates.add(Triple(cost, ti, di))
            }
        }

        candidates.sortBy { it.first }
        val detToTrackId = IntArray(detections.size) { -1 }
        val matchedTrack = BooleanArray(activeTracks.size)

        for ((_, ti, di) in candidates) {
            if (matchedTrack[ti] || detToTrackId[di] >= 0) continue
            matchedTrack[ti] = true

            val t = activeTracks[ti]
            val det = detections[di]
            val dt = (safeTime - t.lastSeenSec).coerceAtLeast(0.0)
            if (dt > 1e-3) {
                val instVx = ((det.box.centerX - t.box.centerX) / dt).toFloat()
                val instVy = ((det.box.footY - t.box.footY) / dt).toFloat()
                t.vx = 0.6f * t.vx + 0.4f * instVx
                t.vy = 0.6f * t.vy + 0.4f * instVy
            }
            t.box = det.box
            t.lastSeenSec = safeTime
            t.misses = 0
            t.hits++
            val detRgb = toFloatRgb(det.chestRgb)
            for (c in 0..2) {
                t.emaRgb[c] = 0.8f * t.emaRgb[c] + 0.2f * detRgb[c]
            }
            val cleanStripe = sanitizeStripeScore(det.stripeScore)
            t.emaStripeScore = 0.8f * t.emaStripeScore + 0.2f * cleanStripe
            t.trail.addLast(det.box.centerX to det.box.footY)
            while (t.trail.size > MAX_TRAIL_POINTS) {
                t.trail.removeFirst()
            }

            val rec = registry.getOrCreateRecord(t.id, safeTime, det.box, det.chestRgb, cleanStripe)
            rec.lastVx = t.vx
            rec.lastVy = t.vy
            detToTrackId[di] = t.id
        }

        // Age unmatched tracks and prune expired ones.
        for (ti in activeTracks.indices.reversed()) {
            if (!matchedTrack[ti]) {
                val t = activeTracks[ti]
                t.misses++
                val missedSec = safeTime - t.lastSeenSec
                if (t.misses > MAX_ACTIVE_MISSES || missedSec > MAX_ACTIVE_GAP_SEC) {
                    activeTracks.removeAt(ti)
                }
            }
        }

        // Spawn tracks for unmatched detections.
        for (di in detections.indices) {
            if (detToTrackId[di] >= 0) continue
            val det = detections[di]
            val newId = nextTrackId++
            val cleanStripe = sanitizeStripeScore(det.stripeScore)
            val t = ActiveTrack(
                id = newId,
                box = det.box,
                lastSeenSec = safeTime,
                emaRgb = toFloatRgb(det.chestRgb),
                emaStripeScore = cleanStripe
            )
            activeTracks.add(t)
            registry.getOrCreateRecord(newId, safeTime, det.box, det.chestRgb, cleanStripe)
            detToTrackId[di] = newId
        }

        // Re-cluster teams using weighted 2-means RGB+luminance and referee outlier detection.
        reclusterTeams(currentSkaterCount = detections.size)
        val identities = registry.resolveAll()

        val trackedPlayers = detections.indices.map { di ->
            val det = detections[di]
            val tid = detToTrackId[di]
            val track = activeTracks.firstOrNull { it.id == tid }
            val resolved = identities[tid]
            val speed = if (track != null) hypot(track.vx, track.vy) else 0f
            TrackedPlayer(
                trackId = tid,
                box = det.box,
                team = resolved?.team ?: TeamId.UNKNOWN,
                jerseyNumber = resolved?.jerseyNumber,
                numberConfidence = resolved?.confidence ?: 0f,
                nanoVerified = resolved?.nanoVerified ?: false,
                roleHint = if (resolved?.team == TeamId.REFEREE) "Referee" else null,
                chestRgb = normalizeRgbList(det.chestRgb),
                speedNormPerSec = speed,
                trail = track?.trail?.toList() ?: listOf(det.box.centerX to det.box.footY)
            )
        }

        FrameMarkSample(
            timestampSec = safeTime,
            livePlayProb = safeLive,
            players = trackedPlayers
        )
    }

    fun recordJerseyRead(
        trackId: Int,
        jerseyNumber: String,
        confidence: Float,
        fromNano: Boolean
    ) = synchronized(lock) {
        registry.recordJerseyRead(trackId, jerseyNumber, confidence, fromNano)
    }

    fun assignTrackJerseyNumber(
        trackId: Int,
        jerseyNumber: String,
        teamOverride: TeamId? = null
    ) = synchronized(lock) {
        registry.assignTrackJerseyNumber(trackId, jerseyNumber, teamOverride)
    }

    fun updateTeamProfile(
        team: TeamId,
        displayName: String,
        kitSummary: String,
        nanoVerified: Boolean
    ) = synchronized(lock) {
        teamMetaOverrides[team] = TeamOverrideMeta(
            displayName = displayName,
            kitSummary = kitSummary,
            nanoVerified = nanoVerified
        )
    }

    fun getTeamProfiles(): Map<TeamId, TeamProfile> = synchronized(lock) {
        val out = LinkedHashMap<TeamId, TeamProfile>()
        val teamsToEmit = listOf(TeamId.TEAM_A, TeamId.TEAM_B, TeamId.REFEREE)
        for (team in teamsToEmit) {
            val rgb = clusteredTeamRgb[team]
            val meta = teamMetaOverrides[team]
            if (rgb == null && meta == null) continue
            val defaultName = when (team) {
                TeamId.TEAM_A -> "Team A (Dark)"
                TeamId.TEAM_B -> "Team B (Light)"
                TeamId.REFEREE -> "Officials"
                TeamId.UNKNOWN -> "Unknown"
            }
            out[team] = TeamProfile(
                id = team,
                displayName = meta?.displayName ?: defaultName,
                meanRgb = rgb ?: defaultRgbFor(team),
                kitSummary = meta?.kitSummary ?: "",
                nanoVerified = meta?.nanoVerified ?: false
            )
        }
        out
    }

    fun annotateHistory(history: List<FrameMarkSample>): List<FrameMarkSample> = synchronized(lock) {
        if (history.isEmpty()) return@synchronized emptyList()
        val identities = registry.resolveAll()
        history.map { frame ->
            frame.copy(
                players = frame.players.map { player ->
                    val resolved = identities[player.trackId]
                    if (resolved == null) {
                        player
                    } else {
                        player.copy(
                            team = if (resolved.team != TeamId.UNKNOWN) resolved.team else player.team,
                            jerseyNumber = resolved.jerseyNumber ?: player.jerseyNumber,
                            numberConfidence = if (resolved.jerseyNumber != null) resolved.confidence else player.numberConfidence,
                            nanoVerified = player.nanoVerified || resolved.nanoVerified,
                            roleHint = if (resolved.team == TeamId.REFEREE) "Referee" else player.roleHint
                        )
                    }
                }
            )
        }
    }

    fun getDiscoveredPlayers(
        history: List<FrameMarkSample>,
        videoDurationSec: Double
    ): List<DiscoveredPlayerSummary> = synchronized(lock) {
        if (history.isEmpty()) return@synchronized emptyList()
        val annotated = annotateHistory(history).sortedBy { it.timestampSec }
        val sampleDurations = estimateSampleDurations(annotated)

        data class Accumulator(
            val trackIds: MutableSet<Int> = LinkedHashSet(),
            var totalSeenSec: Double = 0.0,
            var bestConfidence: Float = 0f,
            var nanoVerified: Boolean = false
        )

        val groups = LinkedHashMap<Pair<String, TeamId>, Accumulator>()
        for (idx in annotated.indices) {
            val frame = annotated[idx]
            val dt = sampleDurations[idx]
            val seenInFrame = HashSet<Pair<String, TeamId>>()

            for (player in frame.players) {
                if (player.team == TeamId.REFEREE) continue
                val cleanNum = player.jerseyNumber?.trim()?.trimStart('#')?.takeIf { it.isNotEmpty() } ?: continue
                val key = cleanNum to player.team
                val acc = groups.getOrPut(key) { Accumulator() }
                acc.trackIds.add(player.trackId)
                if (seenInFrame.add(key)) {
                    acc.totalSeenSec += dt
                }
                acc.bestConfidence = max(acc.bestConfidence, player.numberConfidence)
                if (player.nanoVerified) {
                    acc.nanoVerified = true
                }
            }
        }

        val effectiveDuration = if (videoDurationSec > 0.0) {
            videoDurationSec
        } else {
            (annotated.last().timestampSec + 0.25).coerceAtLeast(1.0)
        }

        groups.entries.map { (key, acc) ->
            val (num, team) = key
            val windows = PlayerClipSegmenter.compute30sWindows(
                history = annotated,
                playerNumber = num,
                teamFilter = team,
                videoDurationSec = effectiveDuration
            )
            DiscoveredPlayerSummary(
                jerseyNumber = num,
                team = team,
                trackIds = acc.trackIds.toSet(),
                totalSeenSec = acc.totalSeenSec,
                bestConfidence = acc.bestConfidence,
                nanoVerified = acc.nanoVerified,
                clipCount30s = windows.size
            )
        }.sortedWith(
            compareByDescending<DiscoveredPlayerSummary> { it.totalSeenSec }
                .thenByDescending { it.nanoVerified }
                .thenByDescending { it.bestConfidence }
        )
    }

    private fun estimateSampleDurations(sortedSamples: List<FrameMarkSample>): DoubleArray {
        val n = sortedSamples.size
        val out = DoubleArray(n)
        if (n == 0) return out
        if (n == 1) {
            out[0] = 0.25
            return out
        }
        val positiveDeltas = ArrayList<Double>(n - 1)
        for (i in 0 until n - 1) {
            val d = sortedSamples[i + 1].timestampSec - sortedSamples[i].timestampSec
            if (d > 1e-4 && d <= 2.0) positiveDeltas.add(d)
        }
        val typicalStep = if (positiveDeltas.isNotEmpty()) {
            positiveDeltas.sorted()[positiveDeltas.size / 2].coerceIn(0.05, 0.5)
        } else {
            0.25
        }
        for (i in 0 until n) {
            val d = if (i < n - 1) {
                (sortedSamples[i + 1].timestampSec - sortedSamples[i].timestampSec)
                    .let { if (it > 1e-4) min(it, 0.5) else typicalStep }
            } else {
                typicalStep
            }
            out[i] = d
        }
        return out
    }

    private fun reclusterTeams(currentSkaterCount: Int) {
        val all = registry.allRecords().toList()
        val withColor = all.mapNotNull { rec -> rec.meanRgb()?.let { rec to it } }
        if (withColor.isEmpty()) return

        var refSumR = 0.0
        var refSumG = 0.0
        var refSumB = 0.0
        var refWeight = 0.0
        val skaterWithColor = ArrayList<Pair<TrackIdentityRegistry.TrackRecord, FloatArray>>(withColor.size)

        // Pre-classify striped low-saturation officials at any skater count and exclude them
        // from Team A/B weighted 2-means so officials never skew team kit centroids.
        for (pair in withColor) {
            val (rec, rgb) = pair
            if (rec.emaStripeScore >= STRIPE_REFEREE_MIN_SCORE &&
                hsvSaturation(rgb) < STRIPE_REFEREE_MAX_HSV_SAT
            ) {
                rec.clusteredTeam = TeamId.REFEREE
                val w = rec.colorCount.toDouble()
                refSumR += rgb[0] * w
                refSumG += rgb[1] * w
                refSumB += rgb[2] * w
                refWeight += w
            } else {
                skaterWithColor.add(pair)
            }
        }

        if (skaterWithColor.isEmpty()) {
            clusteredTeamRgb.remove(TeamId.TEAM_A)
            clusteredTeamRgb.remove(TeamId.TEAM_B)
            updateRefereeCentroid(refSumR, refSumG, refSumB, refWeight)
            registry.invalidateCache()
            return
        }

        if (skaterWithColor.size == 1) {
            val (rec, rgb) = skaterWithColor.first()
            val singleTeam = if (luminance(rgb) < 128f) TeamId.TEAM_A else TeamId.TEAM_B
            rec.clusteredTeam = singleTeam
            clusteredTeamRgb.remove(if (singleTeam == TeamId.TEAM_A) TeamId.TEAM_B else TeamId.TEAM_A)
            clusteredTeamRgb[singleTeam] = rgb.map { it.roundToInt().coerceIn(0, 255) }
            updateRefereeCentroid(refSumR, refSumG, refSumB, refWeight)
            registry.invalidateCache()
            return
        }

        // Prefer tracks with >=3 hits (fallback >=2, then all) and seed p15/p85 to resist glare outliers.
        val anchors = skaterWithColor.filter { it.first.colorCount >= 3 }
            .ifEmpty { skaterWithColor.filter { it.first.colorCount >= 2 } }
            .ifEmpty { skaterWithColor }
        val sortedAnchors = anchors.sortedBy { luminance(it.second) }
        var centroidA = if (sortedAnchors.size >= 4) {
            sortedAnchors[(sortedAnchors.size * 15) / 100].second.copyOf()
        } else {
            sortedAnchors.first().second.copyOf()
        }
        var centroidB = if (sortedAnchors.size >= 4) {
            sortedAnchors[(sortedAnchors.size * 85) / 100].second.copyOf()
        } else {
            sortedAnchors.last().second.copyOf()
        }

        // Run weighted 2-means (RGB + luminance) for 12 iterations with capped per-track weight.
        repeat(12) {
            var sumAR = 0.0; var sumAG = 0.0; var sumAB = 0.0; var weightA = 0.0
            var sumBR = 0.0; var sumBG = 0.0; var sumBB = 0.0; var weightB = 0.0

            for ((rec, rgb) in skaterWithColor) {
                val dA = colorLumDistance(rgb, centroidA)
                val dB = colorLumDistance(rgb, centroidB)
                val team = if (dA <= dB) TeamId.TEAM_A else TeamId.TEAM_B

                val w = min(rec.colorCount, 15).toDouble()
                if (team == TeamId.TEAM_A) {
                    sumAR += rgb[0] * w; sumAG += rgb[1] * w; sumAB += rgb[2] * w; weightA += w
                } else {
                    sumBR += rgb[0] * w; sumBG += rgb[1] * w; sumBB += rgb[2] * w; weightB += w
                }
            }
            if (weightA > 0.0) {
                centroidA = floatArrayOf((sumAR / weightA).toFloat(), (sumAG / weightA).toFloat(), (sumAB / weightA).toFloat())
            }
            if (weightB > 0.0) {
                centroidB = floatArrayOf((sumBR / weightB).toFloat(), (sumBG / weightB).toFloat(), (sumBB / weightB).toFloat())
            }
        }

        // Enforce TEAM_A = darker kit, TEAM_B = lighter kit.
        if (luminance(centroidA) > luminance(centroidB)) {
            val tmp = centroidA
            centroidA = centroidB
            centroidB = tmp
        }

        val abSeparation = colorLumDistance(centroidA, centroidB)
        if (abSeparation < 25f) {
            // Only one distinct kit color visible so far; assign uniformly by dark/light luminance.
            val singleTeam = if (luminance(centroidA) < 128f) TeamId.TEAM_A else TeamId.TEAM_B
            val meanRgb = centroidA.map { it.roundToInt().coerceIn(0, 255) }
            for ((rec, _) in skaterWithColor) {
                rec.clusteredTeam = singleTeam
            }
            clusteredTeamRgb.remove(TeamId.TEAM_A)
            clusteredTeamRgb.remove(TeamId.TEAM_B)
            clusteredTeamRgb[singleTeam] = meanRgb
            updateRefereeCentroid(refSumR, refSumG, refSumB, refWeight)
            registry.invalidateCache()
            return
        }
        val allowRefereeOutlier = currentSkaterCount > 10 || activeTracks.size > 10

        for ((rec, rgb) in skaterWithColor) {
            val dA = colorLumDistance(rgb, centroidA)
            val dB = colorLumDistance(rgb, centroidB)
            var assigned = if (dA <= dB) TeamId.TEAM_A else TeamId.TEAM_B

            if (allowRefereeOutlier && abSeparation >= 35f) {
                val sat = saturationSpread(rgb)
                val minTeamDist = min(dA, dB)
                val outlierThreshold = max(45f, 0.30f * abSeparation)
                val isNeutralStriped = sat < 35f
                val isBrightStripedOrOrange = sat > 95f
                if (minTeamDist > outlierThreshold && (isNeutralStriped || isBrightStripedOrOrange)) {
                    assigned = TeamId.REFEREE
                    val w = rec.colorCount.toDouble()
                    refSumR += rgb[0] * w; refSumG += rgb[1] * w; refSumB += rgb[2] * w; refWeight += w
                }
            }
            rec.clusteredTeam = assigned
        }

        clusteredTeamRgb[TeamId.TEAM_A] = centroidA.map { it.roundToInt().coerceIn(0, 255) }
        clusteredTeamRgb[TeamId.TEAM_B] = centroidB.map { it.roundToInt().coerceIn(0, 255) }
        updateRefereeCentroid(refSumR, refSumG, refSumB, refWeight)
        registry.invalidateCache()
    }

    private fun updateRefereeCentroid(sumR: Double, sumG: Double, sumB: Double, weight: Double) {
        if (weight > 0.0) {
            clusteredTeamRgb[TeamId.REFEREE] = listOf(
                (sumR / weight).roundToInt().coerceIn(0, 255),
                (sumG / weight).roundToInt().coerceIn(0, 255),
                (sumB / weight).roundToInt().coerceIn(0, 255)
            )
        } else {
            clusteredTeamRgb.remove(TeamId.REFEREE)
        }
    }

    private fun isCrossKitIncompatible(
        trackRgb: FloatArray,
        detRgb: FloatArray,
        rgbDist: Float,
        lumDiff: Float
    ): Boolean {
        val centA = clusteredTeamRgb[TeamId.TEAM_A]?.let { toFloatRgb(it) }
        val centB = clusteredTeamRgb[TeamId.TEAM_B]?.let { toFloatRgb(it) }
        if (centA != null && centB != null && colorLumDistance(centA, centB) >= 40f) {
            val tIsA = colorLumDistance(trackRgb, centA) <= colorLumDistance(trackRgb, centB)
            val dIsA = colorLumDistance(detRgb, centA) <= colorLumDistance(detRgb, centB)
            if (tIsA != dIsA && rgbDist > 65f) return true
        }
        return rgbDist > 80f && lumDiff > 42f
    }

    companion object {
        private const val CONTACT_DIST_NORM = 0.08f
        private const val CONTACT_IOU_MIN = 0.08f
        private const val MIN_MATCH_IOU = 0.10f
        private const val MAX_NORM_FOOT_DIST = 2.5f
        private const val MAX_ABS_FOOT_DIST = 0.20f
        private const val MAX_ACTIVE_MISSES = 4
        private const val MAX_ACTIVE_GAP_SEC = 1.0
        private const val MAX_TRAIL_POINTS = 16
        private const val STRIPE_REFEREE_MIN_SCORE = 0.58f
        private const val STRIPE_REFEREE_MAX_HSV_SAT = 0.22f

        private fun sanitizeStripeScore(score: Float): Float =
            if (score.isFinite()) score.coerceIn(0f, 1f) else 0f

        private fun areCloseOrOverlapping(a: RectNorm, b: RectNorm): Boolean {
            if (computeIou(a, b) >= CONTACT_IOU_MIN) return true
            val footDist = hypot(a.centerX - b.centerX, a.footY - b.footY)
            if (footDist <= CONTACT_DIST_NORM) return true
            val centerDist = hypot(a.centerX - b.centerX, a.centerY - b.centerY)
            return centerDist <= CONTACT_DIST_NORM
        }

        private fun computeIou(a: RectNorm, b: RectNorm): Float {
            val iw = min(a.x2, b.x2) - max(a.x1, b.x1)
            val ih = min(a.y2, b.y2) - max(a.y1, b.y1)
            if (iw <= 0f || ih <= 0f) return 0f
            val inter = iw * ih
            val union = a.width * a.height + b.width * b.height - inter
            return if (union <= 1e-6f) 0f else inter / union
        }

        private fun toFloatRgb(rgb: List<Int>): FloatArray = floatArrayOf(
            (rgb.getOrNull(0) ?: 128).coerceIn(0, 255).toFloat(),
            (rgb.getOrNull(1) ?: 128).coerceIn(0, 255).toFloat(),
            (rgb.getOrNull(2) ?: 128).coerceIn(0, 255).toFloat()
        )

        private fun normalizeRgbList(rgb: List<Int>): List<Int> = listOf(
            (rgb.getOrNull(0) ?: 128).coerceIn(0, 255),
            (rgb.getOrNull(1) ?: 128).coerceIn(0, 255),
            (rgb.getOrNull(2) ?: 128).coerceIn(0, 255)
        )

        private fun rgbDistance(a: FloatArray, b: FloatArray): Float {
            val dr = a[0] - b[0]
            val dg = a[1] - b[1]
            val db = a[2] - b[2]
            return kotlin.math.sqrt(dr * dr + dg * dg + db * db)
        }

        private fun luminance(rgb: FloatArray): Float =
            0.299f * rgb[0] + 0.587f * rgb[1] + 0.114f * rgb[2]

        private fun saturationSpread(rgb: FloatArray): Float =
            maxOf(rgb[0], rgb[1], rgb[2]) - minOf(rgb[0], rgb[1], rgb[2])

        private fun hsvSaturation(rgb: FloatArray): Float {
            val maxC = maxOf(rgb[0], rgb[1], rgb[2])
            val minC = minOf(rgb[0], rgb[1], rgb[2])
            return if (maxC <= 1e-4f) 0f else (maxC - minC) / maxC
        }

        private fun colorLumDistance(a: FloatArray, b: FloatArray): Float {
            val dr = a[0] - b[0]
            val dg = a[1] - b[1]
            val db = a[2] - b[2]
            val dl = luminance(a) - luminance(b)
            return kotlin.math.sqrt(dr * dr + dg * dg + db * db + 1.5f * dl * dl)
        }

        private fun defaultRgbFor(team: TeamId): List<Int> = when (team) {
            TeamId.TEAM_A -> listOf(32, 68, 52)
            TeamId.TEAM_B -> listOf(228, 232, 236)
            TeamId.REFEREE -> listOf(128, 128, 128)
            TeamId.UNKNOWN -> listOf(128, 128, 128)
        }
    }
}
