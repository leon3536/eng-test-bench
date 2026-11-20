package com.engtestbench.hockeynano.core.tracking

import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/** Geometry / visibility classification of a jersey number observation. */
enum class JerseyReadKind {
    FULL,
    LEFT_PARTIAL,
    RIGHT_PARTIAL,
    SINGLE,
    SLEEVE
}

/** Source engine producing a jersey read observation and its calibrated fusion multiplier. */
enum class JerseyReadSource(val defaultWeight: Float) {
    ONNX_STR(1.6f),
    GEMINI_NANO(1.6f),
    MLKIT_OCR(1.0f)
}

/**
 * Single timestamped jersey number read or top-K distribution observation on a track.
 * Either [number] (for point reads / partials) or [candidates] (for softmax distributions) is populated.
 */
data class JerseyReadObservation(
    val timestampSec: Double,
    val number: String?,
    val confidence: Float,
    val kind: JerseyReadKind = JerseyReadKind.FULL,
    val qualityWeight: Float = 1f,
    val source: JerseyReadSource = JerseyReadSource.MLKIT_OCR,
    val candidates: List<Pair<String, Float>>? = null
)

/** Per-chain input passed into [JerseyNumberPosterior.assignNumbersJointly]. */
data class ChainNumberInput(
    val chainId: Int,
    val firstSeenSec: Double,
    val lastSeenSec: Double,
    val observations: List<JerseyReadObservation>,
    val manualNumber: String? = null,
    val nanoVerifiedNumbers: Set<String> = emptySet(),
    val bestConfidenceByNumber: Map<String, Float> = emptyMap()
)

/** Joint assignment output for a single stitched chain. */
data class NumberAssignment(
    val number: String?,
    val probability: Float,
    val duplicateWithChainIds: List<Int> = emptyList(),
    val eliminatedNumbers: List<String> = emptyList(),
    val nanoVerified: Boolean = false
)

/**
 * 100% roster-free Bayesian jersey-number posterior over `1..99` (uniform prior `1/99`)
 * with symmetric sports-font digit confusion, left/right partial digit aggregation (`HIDE_P = 0.38`),
 * and exact pure-Kotlin O(N^3) Hungarian co-occurrence exclusivity per team.
 */
object JerseyNumberPosterior {

    const val NUM_CLASSES = 99
    const val UNIFORM_PRIOR = 1.0 / NUM_CLASSES
    const val HIDE_P = 0.38
    const val SPURIOUS_P = 0.05
    const val P_CORRECT_DIGIT = 0.85
    const val DUPLICATE_PRIOR = 0.04
    const val UNASSIGN_PRIOR = 0.08
    const val DUPLICATE_MIN_OVERLAP_SEC = 1.5
    const val DUPLICATE_MIN_SHARE = 0.12
    const val MIN_POSTERIOR_PROB = 0.48f
    const val MIN_EVIDENCE_WEIGHT = 0.90f
    const val EVIDENCE_BIN_SEC = 1.0
    const val SAME_BIN_MARGINAL_WEIGHT = 0.15f
    const val SINGLE_BIN_COMMIT_MIN_CONF = 0.90f
    const val HIGH_QUALITY_SINGLE_BIN_MIN_CONF = 0.86f
    const val HIGH_QUALITY_MIN_WEIGHT = 1.20f
    const val HIGH_QUALITY_MIN_AGREEING_FRAMES = 3
    const val NANO_SINGLE_BIN_MIN_CONF = 0.80f
    const val CROSS_ENGINE_CORROB_MIN_CONF = 0.55f
    const val COMPLEMENTARY_PARTIAL_MIN_CONF = 0.80f
    const val DEFAULT_MIN_MOVE_PROB = 0.05f
    const val STRONG_MIN_CONF = 0.85f
    const val STRONG_BIN_SEC = 10.0

    /** Canonical numbers `"1".."99"` at indices `0..98`. */
    val ALL_NUMBERS: Array<String> = Array(NUM_CLASSES) { (it + 1).toString() }
    private val NUMBER_INDEX: Map<String, Int> = ALL_NUMBERS.withIndex().associate { it.value to it.index }

    private val CONFUSION_MATRIX: Array<DoubleArray> = buildConfusionMatrix()
    private val SINGLE_CONFUSION_NEIGHBORS: Array<IntArray> = buildSingleConfusionNeighbors()

    private fun buildConfusionMatrix(): Array<DoubleArray> {
        val m = Array(10) { DoubleArray(10) }
        val pairs = listOf(
            Triple(1, 7, 0.050),
            Triple(3, 8, 0.045),
            Triple(6, 8, 0.035),
            Triple(6, 9, 0.030),
            Triple(8, 9, 0.030),
            Triple(3, 9, 0.025),
            Triple(0, 8, 0.035),
            Triple(2, 7, 0.025),
            Triple(4, 1, 0.025)
        )
        for ((a, b, w) in pairs) {
            m[a][b] += w
            m[b][a] += w
        }
        for (d in 0..9) {
            val offSum = m[d].sum()
            val rest = max(0.0, 1.0 - P_CORRECT_DIGIT - offSum)
            val perDigit = rest / 9.0
            var totalOff = 0.0
            for (r in 0..9) {
                if (r != d) {
                    m[d][r] += perDigit
                    totalOff += m[d][r]
                }
            }
            m[d][d] = max(0.0, 1.0 - totalOff)
        }
        return m
    }

    private fun buildSingleConfusionNeighbors(): Array<IntArray> {
        val neighbors = Array(NUM_CLASSES) { IntArray(0) }
        val pairs = setOf(
            1 to 7, 7 to 1,
            3 to 8, 8 to 3,
            6 to 8, 8 to 6,
            6 to 9, 9 to 6,
            8 to 9, 9 to 8,
            3 to 9, 9 to 3,
            0 to 8, 8 to 0,
            2 to 7, 7 to 2,
            4 to 1, 1 to 4
        )
        for (idx in 0 until NUM_CLASSES) {
            val s = ALL_NUMBERS[idx]
            val list = ArrayList<Int>()
            for (j in 0 until NUM_CLASSES) {
                if (j == idx) continue
                val t = ALL_NUMBERS[j]
                if (s.length != t.length) continue
                val diffs = s.indices.count { s[it] != t[it] }
                if (diffs == 1) {
                    val pos = s.indices.first { s[it] != t[it] }
                    val c1 = s[pos] - '0'
                    val c2 = t[pos] - '0'
                    if ((c1 to c2) in pairs) {
                        list.add(j)
                    }
                }
            }
            neighbors[idx] = list.toIntArray()
        }
        return neighbors
    }

    /** Returns `P(reader sees readDigit | true jersey digit is trueDigit)`. */
    fun pDigit(readDigit: Char, trueDigit: Char): Double {
        val r = readDigit - '0'
        val t = trueDigit - '0'
        if (r !in 0..9 || t !in 0..9) return 1e-4
        return CONFUSION_MATRIX[t][r]
    }

    /** Normalizes a candidate string onto `"1".."99"` or returns `null`. */
    fun normalizeNumber(raw: String): String? {
        val s = raw.trim().trimStart('#')
        if (s.isEmpty() || s.length > 2 || !s.all { it.isDigit() }) return null
        if (s == "0" || s == "00") return null
        val n = s.toIntOrNull() ?: return null
        return if (n in 1..99) n.toString() else null
    }

    /** Normalizes a partial/sleeve digit `'0'..'9'` or full number `"1".."99"`. */
    fun normalizeObservationNumber(raw: String, kind: JerseyReadKind): String? {
        val s = raw.trim().trimStart('#')
        if (kind == JerseyReadKind.LEFT_PARTIAL || kind == JerseyReadKind.RIGHT_PARTIAL || kind == JerseyReadKind.SLEEVE) {
            if (s.length == 1 && s[0] in '0'..'9') {
                if (kind == JerseyReadKind.LEFT_PARTIAL && s[0] == '0') return null
                return s
            }
        }
        return normalizeNumber(raw)
    }

    /** Computes `P(obs | trueNumber = candidateNumber)` for a single point observation. */
    fun readLikelihood(obs: JerseyReadObservation, candidateNumber: String): Double {
        val r = obs.number ?: return 1.0
        val c = obs.confidence.toDouble().coerceIn(0.0, 0.99)
        if (c <= 0.0) return 1.0

        val kind = obs.kind
        return if (r.length == 2) {
            val noise = 1.0 / NUM_CLASSES
            val pm = if (candidateNumber.length == 2) {
                pDigit(r[0], candidateNumber[0]) * pDigit(r[1], candidateNumber[1])
            } else {
                val d = candidateNumber[0]
                SPURIOUS_P * 0.5 * (pDigit(r[0], d) + pDigit(r[1], d)) * 0.1
            }
            c * pm + (1.0 - c) * noise
        } else if (r.length == 1) {
            val d = r[0]
            val noise = 1.0 / 10.0
            val pm = if (candidateNumber.length == 1) {
                val base = pDigit(d, candidateNumber[0])
                when (kind) {
                    JerseyReadKind.FULL, JerseyReadKind.SINGLE -> base * (1.0 - 0.3 * HIDE_P)
                    JerseyReadKind.SLEEVE -> 0.8 * base
                    JerseyReadKind.LEFT_PARTIAL, JerseyReadKind.RIGHT_PARTIAL -> 0.25 * base
                }
            } else {
                val left = pDigit(d, candidateNumber[0])
                val right = pDigit(d, candidateNumber[1])
                when (kind) {
                    JerseyReadKind.RIGHT_PARTIAL -> HIDE_P * right + 0.05 * left
                    JerseyReadKind.LEFT_PARTIAL -> HIDE_P * left + 0.05 * right
                    JerseyReadKind.SLEEVE -> 0.5 * HIDE_P * max(left, right)
                    JerseyReadKind.SINGLE, JerseyReadKind.FULL -> 0.3 * HIDE_P * 0.5 * (left + right)
                }
            }
            c * pm + (1.0 - c) * noise
        } else {
            1.0
        }
    }

    /**
     * Fuses all [observations] of one track/chain into a 99-element log-posterior array
     * initialized from the uniform `1.0 / 99.0` prior.
     */
    fun computeLogPosterior(observations: List<JerseyReadObservation>): DoubleArray {
        val logPrior = ln(UNIFORM_PRIOR)
        val logP = DoubleArray(NUM_CLASSES) { logPrior }
        if (observations.isEmpty()) return logP

        val effWeights = computeEffectiveObservationWeights(observations)
        val distLikelihood = DoubleArray(NUM_CLASSES)
        for (i in observations.indices) {
            val obs = observations[i]
            val effWeight = effWeights[i]
            if (effWeight <= 0.0) continue

            val cands = obs.candidates
            if (!cands.isNullOrEmpty()) {
                if (fillDistributionLikelihood(cands, distLikelihood)) {
                    for (idx in 0 until NUM_CLASSES) {
                        logP[idx] += effWeight * ln(max(distLikelihood[idx], 1e-12))
                    }
                }
                continue
            }

            if (obs.number == null || !obs.confidence.isFinite() || obs.confidence <= 0f) continue
            for (idx in 0 until NUM_CLASSES) {
                val lk = readLikelihood(obs, ALL_NUMBERS[idx])
                logP[idx] += effWeight * ln(max(lk, 1e-12))
            }
        }
        return logP
    }

    private fun observationChannel(obs: JerseyReadObservation): Int = when {
        !obs.candidates.isNullOrEmpty() -> 0
        obs.kind == JerseyReadKind.LEFT_PARTIAL -> 1
        obs.kind == JerseyReadKind.RIGHT_PARTIAL -> 2
        obs.kind == JerseyReadKind.SLEEVE -> 3
        else -> 0
    }

    private fun observationRawStrength(obs: JerseyReadObservation): Float {
        val qWeight = (if (obs.qualityWeight.isFinite()) obs.qualityWeight else 1f)
            .coerceIn(0.05f, 1.5f)
        val conf = if (!obs.candidates.isNullOrEmpty()) {
            obs.candidates.maxOfOrNull { if (it.second.isFinite()) it.second else 0f } ?: 0f
        } else if (obs.number != null && obs.confidence.isFinite()) {
            obs.confidence
        } else {
            0f
        }
        if (conf <= 0.02f) return 0f
        return conf.coerceIn(0f, 1f) * qWeight * obs.source.defaultWeight
    }

    /**
     * Computes per-observation effective fusion weight with 1.0s time-bin deduplication:
     * within each 1.0s bin (`floor(timestampSec / 1.0)`), the strongest non-Nano sample per
     * channel gets full weight (`1.0x`) while additional sub-second samples are discounted to `0.15x`.
     */
    internal fun computeEffectiveObservationWeights(observations: List<JerseyReadObservation>): DoubleArray {
        val out = DoubleArray(observations.size)
        if (observations.isEmpty()) return out

        val bestIdxByBinChannel = HashMap<Pair<Long, Int>, Int>()
        val bestStrengthByBinChannel = HashMap<Pair<Long, Int>, Float>()

        for (i in observations.indices) {
            val obs = observations[i]
            val strength = observationRawStrength(obs)
            if (strength <= 0f || obs.source == JerseyReadSource.GEMINI_NANO) continue
            val safeT = if (obs.timestampSec.isFinite()) obs.timestampSec else 0.0
            val bin = floor(safeT / EVIDENCE_BIN_SEC).toLong()
            val key = bin to observationChannel(obs)
            val prevBest = bestStrengthByBinChannel[key]
            if (prevBest == null || strength > prevBest) {
                bestStrengthByBinChannel[key] = strength
                bestIdxByBinChannel[key] = i
            }
        }

        for (i in observations.indices) {
            val obs = observations[i]
            val qWeight = (if (obs.qualityWeight.isFinite()) obs.qualityWeight else 1f)
                .coerceIn(0.05f, 1.5f)
            val baseWeight = (qWeight * obs.source.defaultWeight).toDouble()
            if (obs.source == JerseyReadSource.GEMINI_NANO) {
                out[i] = baseWeight
                continue
            }
            val safeT = if (obs.timestampSec.isFinite()) obs.timestampSec else 0.0
            val bin = floor(safeT / EVIDENCE_BIN_SEC).toLong()
            val key = bin to observationChannel(obs)
            val bestIdx = bestIdxByBinChannel[key]
            val binDiscount = if (bestIdx == null || bestIdx == i) 1.0 else SAME_BIN_MARGINAL_WEIGHT.toDouble()
            out[i] = baseWeight * binDiscount
        }
        return out
    }

    private fun fillDistributionLikelihood(
        candidates: List<Pair<String, Float>>,
        out: DoubleArray
    ): Boolean {
        val candMap = HashMap<Int, Double>(candidates.size)
        var sum = 0.0
        for ((rawNum, rawProb) in candidates) {
            if (!rawProb.isFinite() || rawProb <= 0f) continue
            val clean = normalizeNumber(rawNum) ?: continue
            val idx = NUMBER_INDEX[clean] ?: continue
            val p = rawProb.toDouble().coerceAtLeast(0.0)
            candMap[idx] = (candMap[idx] ?: 0.0) + p
            sum += p
        }
        if (candMap.isEmpty() || sum <= 1e-6) return false

        if (sum > 1.0) {
            val inv = 1.0 / sum
            for (k in candMap.keys.toList()) {
                candMap[k] = (candMap[k] ?: 0.0) * inv
            }
            sum = 1.0
        }

        val residualMass = (1.0 - sum).coerceIn(0.01, 0.99)
        val unlistedCount = max(1, NUM_CLASSES - candMap.size)
        val residualPerOther = residualMass / unlistedCount

        // Mix base distribution with mild confusion-neighbor smoothing to stay calibrated.
        val base = DoubleArray(NUM_CLASSES) { idx -> candMap[idx] ?: residualPerOther }
        for (idx in 0 until NUM_CLASSES) {
            val self = base[idx]
            val neigh = SINGLE_CONFUSION_NEIGHBORS[idx]
            var neighMass = 0.0
            for (nIdx in neigh) {
                neighMass += base[nIdx]
            }
            val smoothed = if (neigh.isNotEmpty()) {
                0.92 * self + 0.08 * (neighMass / neigh.size)
            } else {
                self
            }
            // Uniform heavy-tailed outlier floor so one bad overconfident spike cannot veto good reads.
            out[idx] = max(0.92 * smoothed + 0.08 * UNIFORM_PRIOR, 1e-6)
        }
        return true
    }

    /** Total time-bin-deduplicated weighted evidence accumulated across non-empty observations. */
    fun totalEvidenceWeight(observations: List<JerseyReadObservation>): Float {
        if (observations.isEmpty()) return 0f
        val effWeights = computeEffectiveObservationWeights(observations)
        var total = 0f
        for (i in observations.indices) {
            val obs = observations[i]
            val conf = if (!obs.candidates.isNullOrEmpty()) {
                obs.candidates.maxOfOrNull { if (it.second.isFinite()) it.second else 0f } ?: 0f
            } else if (obs.number != null && obs.confidence.isFinite()) {
                obs.confidence
            } else {
                0f
            }
            if (conf > 0.02f) {
                total += conf.coerceIn(0f, 1f) * effWeights[i].toFloat()
            }
        }
        return total
    }

    /**
     * Checks whether [candidateNumber] is corroborated by >= 2 distinct 1.0s time bins,
     * >= 2 distinct engines (`bestConf >= 0.55f` across `ONNX_STR` / `GEMINI_NANO` / `MLKIT_OCR`),
     * OR a high-confidence single-bin observation (`>= 0.90f` non-Nano / `>= 0.80f` Nano) /
     * Gemini Nano verification.
     */
    fun meetsIndependentBinOrHighConfGate(
        observations: List<JerseyReadObservation>,
        candidateNumber: String?,
        isNanoVerified: Boolean
    ): Boolean {
        if (candidateNumber == null) return false
        if (isNanoVerified) return true
        val supportingBins = HashSet<Long>()
        val bestConfBySource = FloatArray(JerseyReadSource.entries.size)
        var maxNonNanoConf = 0f
        var maxHighQualityConf = 0f
        var strongAgreeingFrames = 0
        var maxNanoConf = 0f
        var hasStrongLeftPartial = false
        var hasStrongRightPartial = false
        for (obs in observations) {
            val safeT = if (obs.timestampSec.isFinite()) obs.timestampSec else 0.0
            val bin = floor(safeT / EVIDENCE_BIN_SEC).toLong()
            val cands = obs.candidates
            if (!cands.isNullOrEmpty()) {
                var candProb = 0f
                var topCandNum: String? = null
                var topCandProb = -1f
                for ((raw, p) in cands) {
                    if (!p.isFinite() || p <= 0f) continue
                    val norm = normalizeNumber(raw) ?: continue
                    if (norm == candidateNumber) {
                        candProb += p
                    }
                    if (p > topCandProb) {
                        topCandProb = p
                        topCandNum = norm
                    }
                }
                if (candProb >= 0.25f) {
                    supportingBins.add(bin)
                    val cleanP = candProb.coerceIn(0f, 1f)
                    val sIdx = obs.source.ordinal
                    bestConfBySource[sIdx] = max(bestConfBySource[sIdx], cleanP)
                    if (obs.source == JerseyReadSource.GEMINI_NANO) {
                        maxNanoConf = max(maxNanoConf, cleanP)
                    } else {
                        maxNonNanoConf = max(maxNonNanoConf, cleanP)
                        if (obs.qualityWeight >= HIGH_QUALITY_MIN_WEIGHT && topCandNum == candidateNumber) {
                            maxHighQualityConf = max(maxHighQualityConf, cleanP)
                        }
                        if (cleanP >= 0.80f && topCandNum == candidateNumber) {
                            strongAgreeingFrames++
                        }
                    }
                }
            } else if (obs.number != null && obs.confidence.isFinite() && obs.confidence > 0f) {
                val matches = when (obs.kind) {
                    JerseyReadKind.FULL, JerseyReadKind.SINGLE -> normalizeNumber(obs.number) == candidateNumber
                    JerseyReadKind.LEFT_PARTIAL -> candidateNumber.length == 2 && obs.number == candidateNumber[0].toString()
                    JerseyReadKind.RIGHT_PARTIAL -> candidateNumber.length == 2 && obs.number == candidateNumber[1].toString()
                    JerseyReadKind.SLEEVE -> obs.number in candidateNumber
                }
                if (matches) {
                    supportingBins.add(bin)
                    val cleanConf = obs.confidence.coerceIn(0f, 1f)
                    val isFullOrSingle = obs.kind == JerseyReadKind.FULL || obs.kind == JerseyReadKind.SINGLE
                    if (isFullOrSingle) {
                        val sIdx = obs.source.ordinal
                        bestConfBySource[sIdx] = max(bestConfBySource[sIdx], cleanConf)
                    }
                    if (obs.kind == JerseyReadKind.LEFT_PARTIAL && cleanConf >= COMPLEMENTARY_PARTIAL_MIN_CONF) {
                        hasStrongLeftPartial = true
                    } else if (obs.kind == JerseyReadKind.RIGHT_PARTIAL && cleanConf >= COMPLEMENTARY_PARTIAL_MIN_CONF) {
                        hasStrongRightPartial = true
                    }
                    if (obs.source == JerseyReadSource.GEMINI_NANO) {
                        maxNanoConf = max(maxNanoConf, cleanConf)
                    } else {
                        maxNonNanoConf = max(maxNonNanoConf, cleanConf)
                        if (isFullOrSingle && obs.qualityWeight >= HIGH_QUALITY_MIN_WEIGHT) {
                            maxHighQualityConf = max(maxHighQualityConf, cleanConf)
                        }
                        if (isFullOrSingle && cleanConf >= 0.80f) {
                            strongAgreeingFrames++
                        }
                    }
                }
            }
        }
        val corroboratingEngines = bestConfBySource.count { it >= CROSS_ENGINE_CORROB_MIN_CONF }
        return supportingBins.size >= 2 ||
            corroboratingEngines >= 2 ||
            (hasStrongLeftPartial && hasStrongRightPartial) ||
            maxNonNanoConf >= SINGLE_BIN_COMMIT_MIN_CONF ||
            (maxHighQualityConf >= HIGH_QUALITY_SINGLE_BIN_MIN_CONF &&
                strongAgreeingFrames >= HIGH_QUALITY_MIN_AGREEING_FRAMES) ||
            maxNanoConf >= NANO_SINGLE_BIN_MIN_CONF
    }

    /** Softmax probabilities over `"1".."99"` excluding [excludeNumbers]. */
    fun posteriorProbs(
        logP: DoubleArray,
        excludeNumbers: Set<String> = emptySet()
    ): Map<String, Float> {
        var maxLog = Double.NEGATIVE_INFINITY
        for (idx in 0 until NUM_CLASSES) {
            if (ALL_NUMBERS[idx] !in excludeNumbers && logP[idx] > maxLog) {
                maxLog = logP[idx]
            }
        }
        if (!maxLog.isFinite()) return emptyMap()
        val weights = DoubleArray(NUM_CLASSES)
        var sum = 0.0
        for (idx in 0 until NUM_CLASSES) {
            if (ALL_NUMBERS[idx] !in excludeNumbers) {
                val w = exp(logP[idx] - maxLog)
                weights[idx] = w
                sum += w
            }
        }
        if (sum <= 0.0) return emptyMap()
        val out = LinkedHashMap<String, Float>(NUM_CLASSES)
        for (idx in 0 until NUM_CLASSES) {
            val num = ALL_NUMBERS[idx]
            if (num !in excludeNumbers) {
                out[num] = (weights[idx] / sum).toFloat()
            }
        }
        return out
    }

    /** Top-K `(number, probability)` candidates sorted descending by probability. */
    fun topCandidates(
        logP: DoubleArray,
        k: Int = 5,
        excludeNumbers: Set<String> = emptySet()
    ): List<Pair<String, Float>> {
        val probs = posteriorProbs(logP, excludeNumbers)
        if (probs.isEmpty()) return emptyList()
        return probs.entries
            .sortedWith(compareByDescending<Map.Entry<String, Float>> { it.value }.thenBy { it.key.length }.thenBy { it.key })
            .take(k)
            .map { it.key to it.value }
    }

    /** Best `(number, probability)` candidate excluding [excludeNumbers]. */
    fun bestCandidate(
        logP: DoubleArray,
        excludeNumbers: Set<String> = emptySet()
    ): Pair<String?, Float> {
        val top = topCandidates(logP, 1, excludeNumbers)
        return if (top.isEmpty()) null to 0f else top[0]
    }

    /**
     * Checks whether [observations] provide independent multi-bin strong support for [number]
     * across at least 2 distinct [binSec] time bins (so borrowed-jersey duplicates can keep their number).
     */
    fun hasStrongSupport(
        observations: List<JerseyReadObservation>,
        number: String?,
        minConf: Float = STRONG_MIN_CONF,
        binSec: Double = STRONG_BIN_SEC
    ): Boolean {
        if (number == null) return false
        val bins = HashSet<Long>()
        for (obs in observations) {
            val safeT = if (obs.timestampSec.isFinite()) obs.timestampSec else 0.0
            val cands = obs.candidates
            if (!cands.isNullOrEmpty()) {
                val bestCand = cands
                    .mapNotNull { (raw, p) -> normalizeNumber(raw)?.let { it to p } }
                    .maxByOrNull { it.second }
                if (bestCand != null && bestCand.first == number && bestCand.second >= minConf) {
                    bins.add(floor(safeT / binSec).toLong())
                }
            } else if (obs.number == number &&
                (obs.kind == JerseyReadKind.FULL || obs.kind == JerseyReadKind.SINGLE) &&
                obs.confidence >= minConf
            ) {
                bins.add(floor(safeT / binSec).toLong())
            }
        }
        return bins.size >= 2
    }

    /**
     * Checks whether two temporal spans overlap by more than `max(1.5, 0.12 * minSpan)` seconds.
     */
    fun spansOverlapSignificantly(
        aStart: Double,
        aEnd: Double,
        bStart: Double,
        bEnd: Double
    ): Boolean {
        val overlap = min(aEnd, bEnd) - max(aStart, bStart)
        if (overlap <= 0.0) return false
        val minSpan = min(max(0.0, aEnd - aStart), max(0.0, bEnd - bStart))
        return overlap > max(DUPLICATE_MIN_OVERLAP_SEC, DUPLICATE_MIN_SHARE * minSpan)
    }

    /**
     * Jointly resolves jersey numbers across all same-team chains using an exact O(N^3)
     * Hungarian solver on clashing temporal cliques, enforcing soft co-occurrence exclusivity
     * with `DUPLICATE_PRIOR = 0.04` fallback for genuine multi-bin strong duplicates.
     */
    fun assignNumbersJointly(
        inputs: List<ChainNumberInput>,
        minProb: Float = MIN_POSTERIOR_PROB,
        minMoveProb: Float = DEFAULT_MIN_MOVE_PROB,
        maxRounds: Int = 50
    ): Map<Int, NumberAssignment> {
        if (inputs.isEmpty()) return emptyMap()

        val sortedInputs = inputs.sortedBy { it.chainId }
        val byId = sortedInputs.associateBy { it.chainId }
        val ids = sortedInputs.map { it.chainId }

        val conc = ids.associateWith { HashSet<Int>() }
        for (i in sortedInputs.indices) {
            val a = sortedInputs[i]
            for (j in i + 1 until sortedInputs.size) {
                val b = sortedInputs[j]
                if (spansOverlapSignificantly(a.firstSeenSec, a.lastSeenSec, b.firstSeenSec, b.lastSeenSec)) {
                    conc.getValue(a.chainId).add(b.chainId)
                    conc.getValue(b.chainId).add(a.chainId)
                }
            }
        }

        val logPMap = HashMap<Int, DoubleArray>(sortedInputs.size)
        val probsMap = HashMap<Int, Map<String, Float>>(sortedInputs.size)
        val assigned = LinkedHashMap<Int, String?>(sortedInputs.size)
        val pinned = HashSet<Int>()

        for (inp in sortedInputs) {
            val lp = computeLogPosterior(inp.observations)
            logPMap[inp.chainId] = lp
            probsMap[inp.chainId] = posteriorProbs(lp)

            val manual = inp.manualNumber?.let { normalizeNumber(it) ?: it }
            if (manual != null) {
                assigned[inp.chainId] = manual
                pinned.add(inp.chainId)
                continue
            }

            if (inp.observations.isEmpty()) {
                assigned[inp.chainId] = null
                continue
            }

            val (bestNum, bestProb) = bestCandidate(lp)
            val evidenceWeight = totalEvidenceWeight(inp.observations)
            val isNano = bestNum != null && bestNum in inp.nanoVerifiedNumbers
            val meetsSupport = bestNum != null &&
                bestProb >= minProb &&
                (evidenceWeight >= MIN_EVIDENCE_WEIGHT || (isNano && evidenceWeight >= 0.75f)) &&
                meetsIndependentBinOrHighConfGate(inp.observations, bestNum, isNano)

            if (meetsSupport) {
                assigned[inp.chainId] = bestNum
                if (hasStrongSupport(inp.observations, bestNum)) {
                    pinned.add(inp.chainId)
                }
            } else {
                assigned[inp.chainId] = null
            }
        }

        val keptDuplicatePairs = HashSet<Pair<Int, Int>>()

        fun probExcludingConcurrent(chainId: Int, candidateNum: String): Float {
            val taken = conc.getValue(chainId)
                .mapNotNull { h -> assigned[h]?.takeIf { it != candidateNum } }
                .toSet()
            return posteriorProbs(logPMap.getValue(chainId), taken)[candidateNum] ?: 0f
        }

        for (round in 0 until maxRounds) {
            var clashPair: Pair<Int, Int>? = null
            outer@ for (a in ids) {
                val numA = assigned[a] ?: continue
                val neighbors = conc.getValue(a).sorted()
                for (b in neighbors) {
                    if (b <= a) continue
                    if (assigned[b] == numA &&
                        (a to b) !in keptDuplicatePairs &&
                        !(a in pinned && b in pinned)
                    ) {
                        clashPair = a to b
                        break@outer
                    }
                }
            }
            if (clashPair == null) break

            val (a, b) = clashPair
            val clique = ArrayList<Int>().apply {
                add(a)
                add(b)
            }
            val common = conc.getValue(a).intersect(conc.getValue(b)).sorted()
            for (g in common) {
                if (clique.all { h -> g in conc.getValue(h) }) {
                    clique.add(g)
                }
            }

            val group = clique.filter { it !in pinned }
            if (group.isEmpty()) {
                keptDuplicatePairs.add(a to b)
                continue
            }

            val cand = ArrayList<String>()
            val candSeen = HashSet<String>()
            for (g in group) {
                val own = assigned[g]
                if (own != null && candSeen.add(own)) {
                    cand.add(own)
                }
                for ((num, p) in topCandidates(logPMap.getValue(g), k = 6)) {
                    if (p >= 0.005f && candSeen.add(num)) {
                        cand.add(num)
                    }
                }
            }

            val bigCost = 1e6
            val nRows = group.size
            // Columns: [0..cand.size-1] = candidate numbers,
            //          [cand.size..cand.size+nRows-1] = explicit null/unassign action,
            //          [cand.size+nRows..cand.size+2*nRows-1] = keep duplicate fallback.
            val nCols = cand.size + 2 * nRows
            val cost = Array(nRows) { DoubleArray(nCols) { bigCost } }
            val unassignCost = -ln(UNASSIGN_PRIOR)

            for (i in group.indices) {
                val g = group[i]
                val p = probsMap.getValue(g)
                val blocked = conc.getValue(g)
                    .filter { h -> h !in group && assigned[h] != null }
                    .mapNotNull { assigned[it] }
                    .toSet()
                for (j in cand.indices) {
                    val num = cand[j]
                    if (num !in blocked) {
                        val prob = max((p[num] ?: 0f).toDouble(), 1e-9)
                        cost[i][j] = -ln(prob)
                    }
                }
                cost[i][cand.size + i] = unassignCost
                val own = assigned[g]
                val keepProb = if (own != null) max((p[own] ?: 0f).toDouble() * DUPLICATE_PRIOR, 1e-9) else 1e-4
                cost[i][cand.size + nRows + i] = -ln(keepProb)
            }

            val before = assigned.toMap()
            val rowToCol = hungarianAssign(cost)
            for (i in group.indices) {
                val col = rowToCol[i]
                if (col in cand.indices) {
                    assigned[group[i]] = cand[col]
                } else if (col in cand.size until (cand.size + nRows)) {
                    assigned[group[i]] = null
                }
            }

            for (g in group) {
                val nextNum = assigned[g]
                if (nextNum != before[g] && nextNum != null) {
                    if (probExcludingConcurrent(g, nextNum) < minMoveProb) {
                        val prevNum = before[g]
                        val prevTaken = prevNum != null && conc.getValue(g).any { assigned[it] == prevNum }
                        assigned[g] = if (prevTaken) null else prevNum
                    }
                }
            }

            if (assigned[a] == assigned[b]) {
                keptDuplicatePairs.add(a to b)
            }
        }

        // Compute remaining concurrent duplicates after joint optimization.
        val dupMap = ids.associateWith { ArrayList<Int>() }
        for (i in ids.indices) {
            val a = ids[i]
            val numA = assigned[a] ?: continue
            for (j in i + 1 until ids.size) {
                val b = ids[j]
                if (b in conc.getValue(a) && assigned[b] == numA) {
                    dupMap.getValue(a).add(b)
                    dupMap.getValue(b).add(a)
                }
            }
        }

        val out = LinkedHashMap<Int, NumberAssignment>(ids.size)
        for (g in ids) {
            val inp = byId.getValue(g)
            val num = assigned[g]
            val taken = conc.getValue(g)
                .mapNotNull { h -> assigned[h]?.takeIf { it != num } }
                .distinct()
                .sorted()
            if (num == null) {
                out[g] = NumberAssignment(
                    number = null,
                    probability = 0f,
                    duplicateWithChainIds = emptyList(),
                    eliminatedNumbers = taken.take(5),
                    nanoVerified = false
                )
                continue
            }
            val manual = inp.manualNumber?.let { normalizeNumber(it) ?: it }
            val renormProb = if (manual != null && num == manual) {
                max(inp.bestConfidenceByNumber[num] ?: 1f, 1f)
            } else {
                posteriorProbs(logPMap.getValue(g), taken.toSet())[num] ?: 0f
            }
            val finalNum = if (manual != null || renormProb >= minProb) num else null
            out[g] = NumberAssignment(
                number = finalNum,
                probability = if (finalNum != null) renormProb.coerceIn(0f, 1f) else 0f,
                duplicateWithChainIds = if (finalNum != null) dupMap.getValue(g).sorted() else emptyList(),
                eliminatedNumbers = taken.filter { (probsMap.getValue(g)[it] ?: 0f) > 0.01f }.take(5),
                nanoVerified = finalNum != null && finalNum in inp.nanoVerifiedNumbers
            )
        }
        return out
    }

    /**
     * Exact O(N^3) Kuhn-Munkres / Jonker-Volgenant Hungarian assignment for rectangular
     * cost matrices (`rows <= cols`). Returns `rowToCol[row]`.
     */
    internal fun hungarianAssign(cost: Array<DoubleArray>): IntArray {
        val n = cost.size
        if (n == 0) return IntArray(0)
        val m = cost[0].size
        require(m >= n) { "Hungarian cost matrix requires cols >= rows ($m < $n)" }

        val u = DoubleArray(n + 1)
        val v = DoubleArray(m + 1)
        val p = IntArray(m + 1)
        val way = IntArray(m + 1)

        for (i in 1..n) {
            p[0] = i
            var j0 = 0
            val minv = DoubleArray(m + 1) { Double.POSITIVE_INFINITY }
            val used = BooleanArray(m + 1)
            do {
                used[j0] = true
                val i0 = p[j0]
                var delta = Double.POSITIVE_INFINITY
                var j1 = 0
                for (j in 1..m) {
                    if (!used[j]) {
                        val cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                        if (cur < minv[j]) {
                            minv[j] = cur
                            way[j] = j0
                        }
                        if (minv[j] < delta) {
                            delta = minv[j]
                            j1 = j
                        }
                    }
                }
                for (j in 0..m) {
                    if (used[j]) {
                        u[p[j]] += delta
                        v[j] -= delta
                    } else {
                        minv[j] -= delta
                    }
                }
                j0 = j1
            } while (p[j0] != 0)
            do {
                val j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while (j0 != 0)
        }

        val rowToCol = IntArray(n) { -1 }
        for (j in 1..m) {
            if (p[j] in 1..n) {
                rowToCol[p[j] - 1] = j - 1
            }
        }
        return rowToCol
    }
}
