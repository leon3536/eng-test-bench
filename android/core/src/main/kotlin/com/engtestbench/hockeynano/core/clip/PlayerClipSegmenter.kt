package com.engtestbench.hockeynano.core.clip

import com.engtestbench.hockeynano.core.model.ClipWindow30s
import com.engtestbench.hockeynano.core.model.FrameMarkSample
import com.engtestbench.hockeynano.core.model.TeamId
import java.util.TreeSet
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Segments player appearances into non-overlapping 30.0-second windows scored by
 * `visibilitySec * (0.7 + 0.3 * avgLiveProb)` with a minimum visibility of 1.0s.
 */
object PlayerClipSegmenter {

    private const val WINDOW_LEN_SEC = 30.0
    private const val MIN_VISIBILITY_SEC = 1.0
    private const val EPS = 1e-6

    private data class PlayerHit(
        val timestampSec: Double,
        val livePlayProb: Float,
        val team: TeamId
    )

    private data class ScoredWindow(
        val startSec: Double,
        val endSec: Double,
        val visibilitySec: Double,
        val sampleHits: Int,
        val avgLiveProb: Float,
        val score: Double,
        val centeringError: Double,
        val team: TeamId,
        val hitIndices: IntRange
    )

    fun compute30sWindows(
        history: List<FrameMarkSample>,
        playerNumber: String,
        teamFilter: TeamId?,
        videoDurationSec: Double,
        stepSec: Double = 0.25
    ): List<ClipWindow30s> {
        val targetNum = playerNumber.trim().trimStart('#')
        if (targetNum.isEmpty() || history.isEmpty() || !videoDurationSec.isFinite() || videoDurationSec <= 0.0) {
            return emptyList()
        }
        val safeStep = if (stepSec.isFinite() && stepSec > 0.0) stepSec else 0.25

        val hits = ArrayList<PlayerHit>()
        for (sample in history.sortedBy { it.timestampSec }) {
            if (!sample.timestampSec.isFinite()) continue
            val t = sample.timestampSec.coerceIn(0.0, videoDurationSec)
            val matched = sample.players.firstOrNull { p ->
                val pNum = p.jerseyNumber?.trim()?.trimStart('#')
                val numMatches = pNum == targetNum
                val teamMatches = if (teamFilter != null) {
                    p.team == teamFilter
                } else {
                    p.team != TeamId.REFEREE
                }
                numMatches && teamMatches
            } ?: continue
            hits.add(
                PlayerHit(
                    timestampSec = t,
                    livePlayProb = sample.livePlayProb.takeIf { it.isFinite() }?.coerceIn(0f, 1f) ?: 1f,
                    team = matched.team
                )
            )
        }

        if (hits.isEmpty()) return emptyList()

        // Short video <= 30.0s produces at most one full-clip window [0.0, videoDurationSec].
        if (videoDurationSec <= WINDOW_LEN_SEC + EPS) {
            val sampleHits = hits.size
            val vis = min(videoDurationSec, sampleHits * safeStep)
            if (vis + EPS < MIN_VISIBILITY_SEC) return emptyList()
            val avgLive = hits.map { it.livePlayProb.toDouble() }.average().toFloat()
            val team = teamFilter ?: majorityTeam(hits, 0 until hits.size)
            val endSec = videoDurationSec
            return listOf(
                ClipWindow30s(
                    id = buildClipId(team, targetNum, 0.0, endSec),
                    playerNumber = targetNum,
                    team = team,
                    startSec = 0.0,
                    endSec = endSec,
                    visibilitySec = vis,
                    sampleHits = sampleHits,
                    avgLiveProb = avgLive
                )
            )
        }

        val maxStart = (videoDurationSec - WINDOW_LEN_SEC).coerceAtLeast(0.0)
        val candidateStarts = TreeSet<Double>()
        var grid = 0.0
        while (grid <= maxStart + EPS) {
            candidateStarts.add(roundGrid(min(grid, maxStart)))
            grid += 1.0
        }
        candidateStarts.add(roundGrid(maxStart))
        for (hit in hits) {
            for (offset in doubleArrayOf(0.0, -2.0, -5.0, -10.0, -15.0, -25.0, -29.0)) {
                candidateStarts.add(roundGrid((hit.timestampSec + offset).coerceIn(0.0, maxStart)))
            }
        }

        val selected = ArrayList<ScoredWindow>()
        val usedHit = BooleanArray(hits.size)

        while (true) {
            var best: ScoredWindow? = null

            for (start in candidateStarts) {
                val end = min(videoDurationSec, start + WINDOW_LEN_SEC)
                if (overlapsAny(start, end, selected)) continue

                val range = findHitRange(hits, start, end)
                if (range.isEmpty()) continue

                var count = 0
                var liveSum = 0.0
                var timeSum = 0.0
                for (idx in range) {
                    if (usedHit[idx]) continue
                    count++
                    liveSum += hits[idx].livePlayProb
                    timeSum += hits[idx].timestampSec
                }
                if (count == 0) continue

                val vis = min(end - start, count * safeStep)
                if (vis + EPS < MIN_VISIBILITY_SEC) continue

                val avgLive = (liveSum / count).toFloat()
                val score = vis * (0.7 + 0.3 * avgLive)
                val meanHitT = timeSum / count
                val centeringErr = abs((start + WINDOW_LEN_SEC * 0.5) - meanHitT)
                val team = teamFilter ?: majorityTeam(hits, range)

                val candidate = ScoredWindow(
                    startSec = start,
                    endSec = end,
                    visibilitySec = vis,
                    sampleHits = count,
                    avgLiveProb = avgLive,
                    score = score,
                    centeringError = centeringErr,
                    team = team,
                    hitIndices = range
                )

                if (isBetterCandidate(candidate, best)) {
                    best = candidate
                }
            }

            if (best == null) break
            selected.add(best)
            for (idx in best.hitIndices) {
                usedHit[idx] = true
            }
            // Add exact adjacent boundary starts around the newly selected window.
            if (best.endSec <= maxStart + EPS) {
                candidateStarts.add(roundGrid(min(best.endSec, maxStart)))
            }
            if (best.startSec - WINDOW_LEN_SEC >= -EPS) {
                candidateStarts.add(roundGrid(max(0.0, best.startSec - WINDOW_LEN_SEC)))
            }
        }

        return selected
            .sortedBy { it.startSec }
            .map { w ->
                ClipWindow30s(
                    id = buildClipId(w.team, targetNum, w.startSec, w.endSec),
                    playerNumber = targetNum,
                    team = w.team,
                    startSec = w.startSec,
                    endSec = w.endSec,
                    visibilitySec = w.visibilitySec,
                    sampleHits = w.sampleHits,
                    avgLiveProb = w.avgLiveProb
                )
            }
    }

    private fun isBetterCandidate(cand: ScoredWindow, current: ScoredWindow?): Boolean {
        if (current == null) return true
        val scoreDiff = cand.score - current.score
        if (scoreDiff > 1e-6) return true
        if (scoreDiff < -1e-6) return false
        val centerDiff = cand.centeringError - current.centeringError
        if (abs(centerDiff) > 1e-4) return centerDiff < 0.0
        return cand.startSec < current.startSec
    }

    private fun overlapsAny(start: Double, end: Double, chosen: List<ScoredWindow>): Boolean {
        for (w in chosen) {
            if (start < w.endSec - EPS && end > w.startSec + EPS) return true
        }
        return false
    }

    private fun findHitRange(hits: List<PlayerHit>, start: Double, end: Double): IntRange {
        var lo = 0
        var hi = hits.size
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (hits[mid].timestampSec < start - EPS) lo = mid + 1 else hi = mid
        }
        val first = lo

        lo = first
        hi = hits.size
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (hits[mid].timestampSec <= end + EPS) lo = mid + 1 else hi = mid
        }
        val lastExclusive = lo
        return first until lastExclusive
    }

    private fun majorityTeam(hits: List<PlayerHit>, range: IntRange): TeamId {
        val counts = LinkedHashMap<TeamId, Int>()
        for (i in range) {
            val t = hits[i].team
            counts[t] = (counts[t] ?: 0) + 1
        }
        return counts.maxByOrNull { it.value }?.key ?: TeamId.UNKNOWN
    }

    private fun roundGrid(t: Double): Double = (t * 100.0).roundToInt() / 100.0

    private fun buildClipId(team: TeamId, number: String, startSec: Double, endSec: Double): String {
        val sTenths = (startSec * 10.0).roundToInt()
        val eTenths = (endSec * 10.0).roundToInt()
        return "clip_${team.name}_${number}_${sTenths}_${eTenths}"
    }
}
