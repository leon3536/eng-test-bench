package com.engtestbench.hockeynano.core.clip

import com.engtestbench.hockeynano.core.model.ClipWindow30s
import com.engtestbench.hockeynano.core.model.FrameMarkSample
import com.engtestbench.hockeynano.core.model.TeamId
import java.util.TreeMap
import java.util.TreeSet
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Segments player appearances into non-overlapping 30.0-second windows scored by
 * `visibilitySec * (0.7 + 0.3 * avgLiveProb)` with a minimum visibility of 1.0s.
 *
 * Uses O(hits) hit-neighborhood candidate generation plus incremental TreeSet/TreeMap
 * maintenance so 30-hour (`108_000.0s`) streams segment in milliseconds.
 */
object PlayerClipSegmenter {

    private const val WINDOW_LEN_SEC = 30.0
    private const val MIN_VISIBILITY_SEC = 1.0
    private const val EPS = 1e-6
    private const val BOUNDARY_RECOMPUTE_MARGIN_SEC = 0.02
    private val HIT_OFFSETS = doubleArrayOf(0.0, -2.0, -5.0, -10.0, -15.0, -25.0, -29.0, -30.0)

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

    private val SCORED_COMPARATOR = Comparator<ScoredWindow> { a, b ->
        val scoreCmp = b.score.compareTo(a.score)
        if (scoreCmp != 0) return@Comparator scoreCmp
        val centerCmp = a.centeringError.compareTo(b.centeringError)
        if (centerCmp != 0) return@Comparator centerCmp
        a.startSec.compareTo(b.startSec)
    }

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
        val candidateStarts = buildHitLocalCandidates(hits, maxStart)

        // Prefix sums allow O(1) initial scoring across all hit-local candidates via two pointers.
        val hitCount = hits.size
        val prefixLive = DoubleArray(hitCount + 1)
        val prefixTime = DoubleArray(hitCount + 1)
        val teamOrdinals = TeamId.entries.size
        val prefixTeam = Array(teamOrdinals) { IntArray(hitCount + 1) }
        for (i in 0 until hitCount) {
            val h = hits[i]
            prefixLive[i + 1] = prefixLive[i] + h.livePlayProb
            prefixTime[i + 1] = prefixTime[i] + h.timestampSec
            for (tIdx in 0 until teamOrdinals) {
                prefixTeam[tIdx][i + 1] = prefixTeam[tIdx][i]
            }
            prefixTeam[h.team.ordinal][i + 1]++
        }

        val scoredTree = TreeSet(SCORED_COMPARATOR)
        val scoredByStart = HashMap<Double, ScoredWindow>(candidateStarts.size * 2)

        var loIdx = 0
        var hiIdx = 0
        for (start in candidateStarts) {
            val end = min(videoDurationSec, start + WINDOW_LEN_SEC)
            while (hiIdx < hitCount && hits[hiIdx].timestampSec <= end + EPS) hiIdx++
            while (loIdx < hiIdx && hits[loIdx].timestampSec < start - EPS) loIdx++

            val count = hiIdx - loIdx
            if (count <= 0) continue
            val vis = min(end - start, count * safeStep)
            if (vis + EPS < MIN_VISIBILITY_SEC) continue

            val liveSum = prefixLive[hiIdx] - prefixLive[loIdx]
            val timeSum = prefixTime[hiIdx] - prefixTime[loIdx]
            val avgLive = (liveSum / count).toFloat()
            val score = vis * (0.7 + 0.3 * avgLive)
            val centeringErr = abs((start + WINDOW_LEN_SEC * 0.5) - (timeSum / count))
            val team = teamFilter ?: majorityTeamFromPrefix(hits, loIdx, hiIdx, prefixTeam)

            val scored = ScoredWindow(
                startSec = start,
                endSec = end,
                visibilitySec = vis,
                sampleHits = count,
                avgLiveProb = avgLive,
                score = score,
                centeringError = centeringErr,
                team = team,
                hitIndices = loIdx until hiIdx
            )
            scoredTree.add(scored)
            scoredByStart[start] = scored
        }

        val selected = ArrayList<ScoredWindow>()
        val selectedByStart = TreeMap<Double, Double>()
        val usedHit = BooleanArray(hitCount)

        while (scoredTree.isNotEmpty()) {
            val best = selectBestWindow(scoredTree) ?: break
            selected.add(best)
            selectedByStart[best.startSec] = best.endSec
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

            // Only candidates whose windows overlap `best` or touch its `EPS` hit edges can change.
            val bandLo = best.startSec - WINDOW_LEN_SEC - BOUNDARY_RECOMPUTE_MARGIN_SEC
            val bandHi = best.endSec + BOUNDARY_RECOMPUTE_MARGIN_SEC
            val affectedStarts = candidateStarts.subSet(bandLo, true, bandHi, true).toList()

            for (start in affectedStarts) {
                scoredByStart.remove(start)?.let { prev -> scoredTree.remove(prev) }

                val end = min(videoDurationSec, start + WINDOW_LEN_SEC)
                if (start < best.endSec - EPS && end > best.startSec + EPS) {
                    candidateStarts.remove(start)
                    continue
                }
                if (overlapsAny(start, end, selectedByStart)) {
                    candidateStarts.remove(start)
                    continue
                }

                val updated = scoreCandidate(
                    start = start,
                    videoDurationSec = videoDurationSec,
                    safeStep = safeStep,
                    hits = hits,
                    usedHit = usedHit,
                    teamFilter = teamFilter
                )
                if (updated != null) {
                    scoredTree.add(updated)
                    scoredByStart[start] = updated
                }
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

    private fun buildHitLocalCandidates(hits: List<PlayerHit>, maxStart: Double): TreeSet<Double> {
        val candidateStarts = TreeSet<Double>()
        candidateStarts.add(0.0)
        candidateStarts.add(roundGrid(maxStart))

        val maxGridSec = floor(maxStart + EPS).toInt().coerceAtLeast(0)
        var nextGridSec = 0

        for (hit in hits) {
            val lo = (hit.timestampSec - WINDOW_LEN_SEC).coerceIn(0.0, maxStart)
            val hi = hit.timestampSec.coerceIn(0.0, maxStart)

            val gStart = max(nextGridSec, ceil(lo - EPS).toInt()).coerceAtLeast(0)
            val gEnd = min(maxGridSec, floor(hi + EPS).toInt())
            if (gStart <= gEnd) {
                for (g in gStart..gEnd) {
                    candidateStarts.add(g.toDouble())
                }
                nextGridSec = gEnd + 1
            }

            for (offset in HIT_OFFSETS) {
                candidateStarts.add(roundGrid((hit.timestampSec + offset).coerceIn(0.0, maxStart)))
            }
        }

        return candidateStarts
    }

    private fun selectBestWindow(scoredTree: TreeSet<ScoredWindow>): ScoredWindow? {
        val top = scoredTree.firstOrNull() ?: return null
        var best = top
        var inTopTier: ScoredWindow? = scoredTree.higher(top)
        while (inTopTier != null && inTopTier.score == top.score && abs(inTopTier.centeringError - best.centeringError) <= 1e-4) {
            if (isBetterCandidate(inTopTier, best)) {
                best = inTopTier
            }
            inTopTier = scoredTree.higher(inTopTier)
        }
        var cursorScore = top.score

        while (true) {
            val tierEndSentinel = ScoredWindow(
                startSec = Double.POSITIVE_INFINITY,
                endSec = Double.POSITIVE_INFINITY,
                visibilitySec = 0.0,
                sampleHits = 0,
                avgLiveProb = 0f,
                score = cursorScore,
                centeringError = Double.POSITIVE_INFINITY,
                team = TeamId.UNKNOWN,
                hitIndices = IntRange.EMPTY
            )
            val next = scoredTree.higher(tierEndSentinel) ?: break
            if (top.score - next.score > 1e-6) break
            if (isBetterCandidate(next, best)) {
                best = next
            }
            var inTier: ScoredWindow? = scoredTree.higher(next)
            while (inTier != null && inTier.score == next.score && abs(inTier.centeringError - best.centeringError) <= 1e-4) {
                if (isBetterCandidate(inTier, best)) {
                    best = inTier
                }
                inTier = scoredTree.higher(inTier)
            }
            cursorScore = next.score
        }
        return best
    }

    private fun scoreCandidate(
        start: Double,
        videoDurationSec: Double,
        safeStep: Double,
        hits: List<PlayerHit>,
        usedHit: BooleanArray,
        teamFilter: TeamId?
    ): ScoredWindow? {
        val end = min(videoDurationSec, start + WINDOW_LEN_SEC)
        val range = findHitRange(hits, start, end)
        if (range.isEmpty()) return null

        var count = 0
        var liveSum = 0.0
        var timeSum = 0.0
        for (idx in range) {
            if (usedHit[idx]) continue
            count++
            liveSum += hits[idx].livePlayProb
            timeSum += hits[idx].timestampSec
        }
        if (count == 0) return null

        val vis = min(end - start, count * safeStep)
        if (vis + EPS < MIN_VISIBILITY_SEC) return null

        val avgLive = (liveSum / count).toFloat()
        val score = vis * (0.7 + 0.3 * avgLive)
        val meanHitT = timeSum / count
        val centeringErr = abs((start + WINDOW_LEN_SEC * 0.5) - meanHitT)
        val team = teamFilter ?: majorityTeam(hits, range)

        return ScoredWindow(
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

    private fun overlapsAny(start: Double, end: Double, selectedByStart: TreeMap<Double, Double>): Boolean {
        val nearby = selectedByStart.subMap(start - WINDOW_LEN_SEC - 1.0, true, end, true)
        for ((wStart, wEnd) in nearby) {
            if (start < wEnd - EPS && end > wStart + EPS) return true
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

    private fun majorityTeamFromPrefix(
        hits: List<PlayerHit>,
        loIdx: Int,
        hiIdx: Int,
        prefixTeam: Array<IntArray>
    ): TeamId {
        var bestTeam = TeamId.UNKNOWN
        var bestCount = -1
        var tieCount = 0
        for (team in TeamId.entries) {
            val c = prefixTeam[team.ordinal][hiIdx] - prefixTeam[team.ordinal][loIdx]
            if (c > bestCount) {
                bestCount = c
                bestTeam = team
                tieCount = 1
            } else if (c == bestCount && c > 0) {
                tieCount++
            }
        }
        return if (tieCount > 1) majorityTeam(hits, loIdx until hiIdx) else bestTeam
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
