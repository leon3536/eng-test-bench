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
import kotlin.math.sqrt

data class RawDetection(
    val box: RectNorm,
    val chestRgb: List<Int>,
    val stripeScore: Float = 0f,
    val kitSummaryLab7: FloatArray? = null,
    val lowerBellyLuma: Float = Float.NaN,
    val pantsDarkness: Float = Float.NaN,
    val armbandScore: Float = 0f,
    val torsoUpperLabL: Float = Float.NaN
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is RawDetection) return false
        if (box != other.box) return false
        if (chestRgb != other.chestRgb) return false
        if (stripeScore != other.stripeScore) return false
        if (!kitSummaryLab7.contentEquals(other.kitSummaryLab7)) return false
        if (lowerBellyLuma != other.lowerBellyLuma && !(lowerBellyLuma.isNaN() && other.lowerBellyLuma.isNaN())) {
            return false
        }
        if (pantsDarkness != other.pantsDarkness && !(pantsDarkness.isNaN() && other.pantsDarkness.isNaN())) {
            return false
        }
        if (armbandScore != other.armbandScore) return false
        if (torsoUpperLabL != other.torsoUpperLabL && !(torsoUpperLabL.isNaN() && other.torsoUpperLabL.isNaN())) {
            return false
        }
        return true
    }

    override fun hashCode(): Int {
        var result = box.hashCode()
        result = 31 * result + chestRgb.hashCode()
        result = 31 * result + stripeScore.hashCode()
        result = 31 * result + (kitSummaryLab7?.contentHashCode() ?: 0)
        result = 31 * result + lowerBellyLuma.hashCode()
        result = 31 * result + pantsDarkness.hashCode()
        result = 31 * result + armbandScore.hashCode()
        result = 31 * result + torsoUpperLabL.hashCode()
        return result
    }
}

/**
 * Maintains temporal track segments, hierarchical 4.0s same-team stitching, and
 * roster-free Bayesian jersey-number posterior fusion + Hungarian co-occurrence assignment
 * across stitched shifts.
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
        val sumKit7: DoubleArray = DoubleArray(KitLabFeatures.SUMMARY_SIZE),
        var explicitKitCount: Int = 0,
        var sumLowerBellyLuma: Double = 0.0,
        var lowerBellyCount: Int = 0,
        var sumTorsoUpperLabL: Double = 0.0,
        var torsoUpperCount: Int = 0,
        var emaPantsDarkness: Float = Float.NaN,
        var pantsObservations: Int = 0,
        var emaArmbandScore: Float = 0f,
        var armbandObservations: Int = 0,
        var emaStripeScore: Float = 0f,
        var stripeObservations: Int = 0,
        var refereeFrameVotes: Int = 0,
        var stripeRefVotes: Int = 0,
        var lastStripeFrameHit: Boolean = false,
        var skaterFrameVotes: Int = 0,
        var clusteredTeam: TeamId = TeamId.UNKNOWN,
        var teamOverride: TeamId? = null,
        var manualJerseyNumber: String? = null,
        val observations: MutableList<JerseyReadObservation> = ArrayList(),
        val bestConfidence: MutableMap<String, Float> = LinkedHashMap(),
        val nanoVerifiedNumbers: MutableSet<String> = LinkedHashSet()
    ) {
        fun addDetectionFeatures(det: RawDetection) {
            addColor(det.chestRgb)
            addStripeScore(det.stripeScore)
            addArmbandScore(det.armbandScore)
            addPantsDarkness(det.pantsDarkness)
            addLowerBellyLuma(det.lowerBellyLuma)
            addTorsoUpperLabL(det.torsoUpperLabL)
            addKitSummaryLab7(det.kitSummaryLab7, det.chestRgb)
            val cleanSt = if (det.stripeScore.isFinite()) det.stripeScore.coerceIn(0f, 1f) else 0f
            val pdOk = !det.pantsDarkness.isFinite() || det.pantsDarkness >= 0.35f
            val r = det.chestRgb.getOrNull(0)?.coerceIn(0, 255)?.toFloat() ?: 128f
            val g = det.chestRgb.getOrNull(1)?.coerceIn(0, 255)?.toFloat() ?: 128f
            val b = det.chestRgb.getOrNull(2)?.coerceIn(0, 255)?.toFloat() ?: 128f
            val lab = KitLabFeatures.rgbToLab(floatArrayOf(r, g, b))
            val detChroma = KitLabFeatures.labChroma(lab[1], lab[2])
            lastStripeFrameHit = cleanSt >= 0.44f && detChroma < 8.0f && pdOk
            if (lastStripeFrameHit) {
                stripeRefVotes++
            }
        }

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

        fun addArmbandScore(score: Float) {
            val clean = if (score.isFinite()) score.coerceIn(0f, 1f) else 0f
            emaArmbandScore = if (armbandObservations == 0) {
                clean
            } else {
                0.8f * emaArmbandScore + 0.2f * clean
            }
            armbandObservations++
        }

        fun addPantsDarkness(score: Float) {
            if (!score.isFinite()) return
            val clean = score.coerceIn(0f, 1f)
            emaPantsDarkness = if (pantsObservations == 0) {
                clean
            } else {
                0.8f * emaPantsDarkness + 0.2f * clean
            }
            pantsObservations++
        }

        fun addLowerBellyLuma(luma: Float) {
            if (!luma.isFinite()) return
            sumLowerBellyLuma += luma.coerceIn(0f, 255f)
            lowerBellyCount++
        }

        fun addTorsoUpperLabL(labL: Float) {
            if (!labL.isFinite()) return
            sumTorsoUpperLabL += labL.coerceIn(0f, 100f)
            torsoUpperCount++
        }

        fun addKitSummaryLab7(kitSummaryLab7: FloatArray?, fallbackRgb: List<Int>) {
            if (kitSummaryLab7 == null || kitSummaryLab7.size < KitLabFeatures.SUMMARY_SIZE) return
            for (i in 0 until KitLabFeatures.SUMMARY_SIZE) {
                if (!kitSummaryLab7[i].isFinite()) return
            }
            val canonical = KitLabFeatures.sanitizeKitSummary7(kitSummaryLab7, fallbackRgb)
            for (i in 0 until KitLabFeatures.SUMMARY_SIZE) {
                sumKit7[i] += canonical[i]
            }
            explicitKitCount++
        }

        fun meanRgb(): FloatArray? {
            if (colorCount <= 0) return null
            val n = colorCount.toFloat()
            return floatArrayOf((sumR / n).toFloat(), (sumG / n).toFloat(), (sumB / n).toFloat())
        }

        fun meanKitSummaryLab7(): FloatArray {
            if (explicitKitCount > 0) {
                val n = explicitKitCount.toFloat()
                return FloatArray(KitLabFeatures.SUMMARY_SIZE) { i -> (sumKit7[i] / n).toFloat() }
            }
            val rgb = meanRgb() ?: return floatArrayOf(50f, 0f, 0f, 50f, 0f, 0f, 0f)
            return KitLabFeatures.synthesizeKitSummary7(rgb)
        }

        fun meanLowerBellyLuma(): Float =
            if (lowerBellyCount > 0) (sumLowerBellyLuma / lowerBellyCount).toFloat() else Float.NaN

        fun meanTorsoUpperLabL(): Float =
            if (torsoUpperCount > 0) (sumTorsoUpperLabL / torsoUpperCount).toFloat() else Float.NaN

        fun isEstablishedOfficial(meanChroma: Float): Boolean {
            if (stripeRefVotes >= 3 && meanChroma < 8.5f) return true
            if (refereeFrameVotes < 3) return false
            if (skaterFrameVotes > 2 * refereeFrameVotes && meanChroma >= 15f) return false
            val lightSkaterEscape = skaterFrameVotes > 3 * refereeFrameVotes &&
                ((emaPantsDarkness.isFinite() && emaPantsDarkness < 0.60f) ||
                    (torsoUpperCount > 0 && meanTorsoUpperLabL() >= 58f))
            return !lightSkaterEscape
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
        det: RawDetection
    ): TrackRecord {
        cachedResolution = null
        val existing = records[trackId]
        if (existing != null) {
            existing.lastSeenSec = max(existing.lastSeenSec, timestampSec)
            existing.lastBox = det.box
            existing.hitCount++
            existing.addDetectionFeatures(det)
            return existing
        }
        return TrackRecord(
            trackId = trackId,
            firstSeenSec = timestampSec,
            lastSeenSec = timestampSec,
            firstBox = det.box,
            lastBox = det.box
        ).also {
            it.addDetectionFeatures(det)
            records[trackId] = it
        }
    }

    internal fun getRecord(trackId: Int): TrackRecord? = records[trackId]

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
        val rec = records[trackId] ?: return
        recordJerseyRead(
            trackId = trackId,
            jerseyNumber = jerseyNumber,
            confidence = confidence,
            timestampSec = rec.lastSeenSec,
            kind = JerseyReadKind.FULL,
            qualityWeight = 1f,
            source = if (fromNano) JerseyReadSource.GEMINI_NANO else JerseyReadSource.MLKIT_OCR
        )
    }

    fun recordJerseyRead(
        trackId: Int,
        jerseyNumber: String,
        confidence: Float,
        timestampSec: Double,
        kind: JerseyReadKind = JerseyReadKind.FULL,
        qualityWeight: Float = 1f,
        source: JerseyReadSource = JerseyReadSource.MLKIT_OCR
    ) {
        val rec = records[trackId] ?: return
        val clean = JerseyNumberPosterior.normalizeObservationNumber(jerseyNumber, kind) ?: return
        val conf = if (confidence.isFinite()) confidence.coerceIn(0.05f, 1.0f) else return
        val qWeight = if (qualityWeight.isFinite()) qualityWeight.coerceIn(0.05f, 1.5f) else 1f
        val safeT = if (timestampSec.isFinite()) timestampSec else rec.lastSeenSec

        rec.observations.add(
            JerseyReadObservation(
                timestampSec = safeT,
                number = clean,
                confidence = conf,
                kind = kind,
                qualityWeight = qWeight,
                source = source
            )
        )
        if (kind == JerseyReadKind.FULL || kind == JerseyReadKind.SINGLE) {
            normalizeJerseyNumber(clean)?.let { fullNum ->
                rec.bestConfidence[fullNum] = max(rec.bestConfidence[fullNum] ?: 0f, conf)
                if (source == JerseyReadSource.GEMINI_NANO) {
                    rec.nanoVerifiedNumbers.add(fullNum)
                }
            }
        }
        cachedResolution = null
    }

    fun recordJerseyDistribution(
        trackId: Int,
        timestampSec: Double,
        candidates: List<Pair<String, Float>>,
        qualityWeight: Float = 1f,
        source: JerseyReadSource = JerseyReadSource.ONNX_STR,
        partialLeftDigit: Pair<Char, Float>? = null,
        partialRightDigit: Pair<Char, Float>? = null
    ) {
        val rec = records[trackId] ?: return
        val qWeight = if (qualityWeight.isFinite()) qualityWeight.coerceIn(0.05f, 1.5f) else 1f
        val safeT = if (timestampSec.isFinite()) timestampSec else rec.lastSeenSec

        val cleanCandidates = candidates.mapNotNull { (rawNum, rawProb) ->
            if (!rawProb.isFinite() || rawProb <= 0f) return@mapNotNull null
            val norm = normalizeJerseyNumber(rawNum) ?: return@mapNotNull null
            norm to rawProb.coerceIn(0f, 1f)
        }
        val topCandProb = cleanCandidates.maxOfOrNull { it.second } ?: 0f
        var recordedAny = false

        if (cleanCandidates.isNotEmpty() && topCandProb >= 0.05f) {
            rec.observations.add(
                JerseyReadObservation(
                    timestampSec = safeT,
                    number = null,
                    confidence = topCandProb,
                    kind = JerseyReadKind.FULL,
                    qualityWeight = qWeight,
                    source = source,
                    candidates = cleanCandidates
                )
            )
            for ((num, p) in cleanCandidates) {
                if (p >= 0.20f) {
                    rec.bestConfidence[num] = max(rec.bestConfidence[num] ?: 0f, p)
                    if (source == JerseyReadSource.GEMINI_NANO) {
                        rec.nanoVerifiedNumbers.add(num)
                    }
                }
            }
            recordedAny = true
        }

        val hasLeft = partialLeftDigit != null &&
            partialLeftDigit.first in '1'..'9' &&
            partialLeftDigit.second.isFinite() &&
            partialLeftDigit.second > 0.05f
        val hasRight = partialRightDigit != null &&
            partialRightDigit.first in '0'..'9' &&
            partialRightDigit.second.isFinite() &&
            partialRightDigit.second > 0.05f

        val partialScale = when {
            topCandProb >= 0.15f -> 0.35f
            hasLeft && hasRight -> 0.65f
            else -> 1.0f
        }

        if (hasLeft) {
            val (ch, conf) = partialLeftDigit
            rec.observations.add(
                JerseyReadObservation(
                    timestampSec = safeT,
                    number = ch.toString(),
                    confidence = conf.coerceIn(0.05f, 0.99f),
                    kind = JerseyReadKind.LEFT_PARTIAL,
                    qualityWeight = (qWeight * partialScale).coerceIn(0.05f, 1.5f),
                    source = source
                )
            )
            recordedAny = true
        }
        if (hasRight) {
            val (ch, conf) = partialRightDigit
            rec.observations.add(
                JerseyReadObservation(
                    timestampSec = safeT,
                    number = ch.toString(),
                    confidence = conf.coerceIn(0.05f, 0.99f),
                    kind = JerseyReadKind.RIGHT_PARTIAL,
                    qualityWeight = (qWeight * partialScale).coerceIn(0.05f, 1.5f),
                    source = source
                )
            )
            recordedAny = true
        }

        if (recordedAny) {
            cachedResolution = null
        }
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

        data class ChainMeta(
            val chainId: Int,
            val trackIds: Set<Int>,
            val team: TeamId,
            val input: ChainNumberInput
        )

        val byTeam = LinkedHashMap<TeamId, ArrayList<ChainMeta>>()
        val metas = ArrayList<ChainMeta>(chains.size)

        for (idx in chains.indices) {
            val chainIds = chains[idx]
            val chainRecords = chainIds.mapNotNull { records[it] }
            if (chainRecords.isEmpty()) continue
            val resolvedTeam = resolveChainTeam(chainRecords)
            val manualNum = chainRecords
                .sortedByDescending { it.lastSeenSec }
                .firstOrNull { !it.manualJerseyNumber.isNullOrBlank() }
                ?.manualJerseyNumber

            val obs = ArrayList<JerseyReadObservation>()
            val bestConf = LinkedHashMap<String, Float>()
            val nanoNums = LinkedHashSet<String>()
            var firstSeen = Double.POSITIVE_INFINITY
            var lastSeen = Double.NEGATIVE_INFINITY

            for (rec in chainRecords) {
                if (rec.firstSeenSec < firstSeen) firstSeen = rec.firstSeenSec
                if (rec.lastSeenSec > lastSeen) lastSeen = rec.lastSeenSec
                obs.addAll(rec.observations)
                for ((num, c) in rec.bestConfidence) {
                    bestConf[num] = max(bestConf[num] ?: 0f, c)
                }
                nanoNums.addAll(rec.nanoVerifiedNumbers)
            }
            obs.sortBy { it.timestampSec }

            val chainId = chainIds.minOrNull() ?: idx
            val meta = ChainMeta(
                chainId = chainId,
                trackIds = chainIds.toSet(),
                team = resolvedTeam,
                input = ChainNumberInput(
                    chainId = chainId,
                    firstSeenSec = firstSeen,
                    lastSeenSec = lastSeen,
                    observations = obs,
                    manualNumber = manualNum,
                    nanoVerifiedNumbers = nanoNums,
                    bestConfidenceByNumber = bestConf
                )
            )
            metas.add(meta)
            byTeam.getOrPut(resolvedTeam) { ArrayList() }.add(meta)
        }

        val assignments = HashMap<Int, NumberAssignment>(metas.size)

        for (team in listOf(TeamId.TEAM_A, TeamId.TEAM_B)) {
            val teamChains = byTeam[team] ?: continue
            val teamRes = JerseyNumberPosterior.assignNumbersJointly(teamChains.map { it.input })
            assignments.putAll(teamRes)
        }

        // UNKNOWN chains resolve individually without team-wide co-occurrence blocking.
        for (meta in byTeam[TeamId.UNKNOWN].orEmpty()) {
            val singleRes = JerseyNumberPosterior.assignNumbersJointly(listOf(meta.input))
            assignments.putAll(singleRes)
        }

        // REFEREE tracks do not hold skater numbers unless manually overridden.
        for (meta in byTeam[TeamId.REFEREE].orEmpty()) {
            val manual = meta.input.manualNumber
            assignments[meta.chainId] = NumberAssignment(
                number = manual,
                probability = if (manual != null) 1f else 0f,
                nanoVerified = manual != null && manual in meta.input.nanoVerifiedNumbers
            )
        }

        for (meta in metas) {
            val assign = assignments[meta.chainId]
            val identity = ResolvedIdentity(
                team = meta.team,
                jerseyNumber = assign?.number,
                confidence = assign?.probability ?: 0f,
                nanoVerified = assign?.nanoVerified ?: false,
                chainTrackIds = meta.trackIds
            )
            for (tid in meta.trackIds) {
                resolved[tid] = identity
            }
        }

        cachedResolution = resolved
        return resolved
    }

    private fun resolveChainTeam(chainRecords: List<TrackRecord>): TeamId {
        val override = chainRecords.lastOrNull { it.teamOverride != null }?.teamOverride
        if (override != null) return override

        var totalRefVotes = 0
        var totalStripeRefVotes = 0
        var totalSkaterVotes = 0
        var sumR = 0.0
        var sumG = 0.0
        var sumB = 0.0
        var colorW = 0.0
        var sumPd = 0.0
        var pdW = 0.0
        for (rec in chainRecords) {
            totalRefVotes += rec.refereeFrameVotes
            totalStripeRefVotes += rec.stripeRefVotes
            totalSkaterVotes += rec.skaterFrameVotes
            val rgb = rec.meanRgb()
            if (rgb != null) {
                val lab = KitLabFeatures.rgbToLab(rgb)
                val kit7 = rec.meanKitSummaryLab7()
                val belly = rec.meanLowerBellyLuma()
                val upperL = rec.meanTorsoUpperLabL()
                val chroma = KitLabFeatures.labChroma(lab[1], lab[2])
                val brightStripe = kit7[3] >= 44f ||
                    lab[0] >= 36f ||
                    (belly.isFinite() && belly >= 85f) ||
                    (upperL.isFinite() && upperL >= 46f)
                val pantsOk = !rec.emaPantsDarkness.isFinite() ||
                    rec.emaPantsDarkness >= 0.30f ||
                    rec.emaArmbandScore >= MultiTeamPlayerTracker.ARMBAND_MIN_SCORE
                if (rec.emaStripeScore >= 0.46f && chroma < 14.5f && brightStripe && pantsOk) {
                    return TeamId.REFEREE
                }
                val w = rec.colorCount.toDouble()
                sumR += rgb[0] * w
                sumG += rgb[1] * w
                sumB += rgb[2] * w
                colorW += w
                if (rec.emaPantsDarkness.isFinite()) {
                    sumPd += rec.emaPantsDarkness * w
                    pdW += w
                }
            }
        }
        val avgChroma = if (colorW > 0.0) {
            val avgLab = KitLabFeatures.rgbToLab(floatArrayOf((sumR / colorW).toFloat(), (sumG / colorW).toFloat(), (sumB / colorW).toFloat()))
            KitLabFeatures.labChroma(avgLab[1], avgLab[2])
        } else {
            0f
        }
        if (totalStripeRefVotes >= 3 && avgChroma < 8.5f) {
            return TeamId.REFEREE
        }
        if (totalRefVotes >= 3) {
            val avgPd = if (pdW > 0.0) (sumPd / pdW).toFloat() else Float.NaN
            val skaterEscape = (totalSkaterVotes > 2 * totalRefVotes && avgChroma >= 15f) ||
                (totalSkaterVotes > 3 * totalRefVotes && avgPd.isFinite() && avgPd < 0.60f)
            if (!skaterEscape) {
                return TeamId.REFEREE
            }
        }

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
        val obs = ArrayList<JerseyReadObservation>()
        val nanoNums = HashSet<String>()
        for (id in trackIds) {
            val rec = records[id] ?: continue
            if (!rec.manualJerseyNumber.isNullOrBlank()) manual = rec.manualJerseyNumber
            obs.addAll(rec.observations)
            nanoNums.addAll(rec.nanoVerifiedNumbers)
        }
        if (manual != null) return manual
        if (obs.isEmpty()) return null
        val logP = JerseyNumberPosterior.computeLogPosterior(obs)
        val (bestNum, bestProb) = JerseyNumberPosterior.bestCandidate(logP)
        if (bestNum == null || bestProb < 0.55f) return null
        val evidence = JerseyNumberPosterior.totalEvidenceWeight(obs)
        val hasStrongSupport = bestNum in nanoNums || evidence >= 1.80f
        return if (hasStrongSupport) bestNum else null
    }

    private fun dominantChainNumber(trackIds: List<Int>): String? {
        var manual: String? = null
        val obs = ArrayList<JerseyReadObservation>()
        for (id in trackIds) {
            val rec = records[id] ?: continue
            if (!rec.manualJerseyNumber.isNullOrBlank()) manual = rec.manualJerseyNumber
            obs.addAll(rec.observations)
        }
        if (manual != null) return manual
        if (obs.isEmpty()) return null
        val logP = JerseyNumberPosterior.computeLogPosterior(obs)
        val (bestNum, bestProb) = JerseyNumberPosterior.bestCandidate(logP)
        if (bestNum == null || bestProb < 0.45f) return null
        val evidence = JerseyNumberPosterior.totalEvidenceWeight(obs)
        return if (evidence >= JerseyNumberPosterior.MIN_EVIDENCE_WEIGHT) bestNum else null
    }

    companion object {
        const val OCR_WEIGHT_MULTIPLIER = 1.0f
        const val NANO_WEIGHT_MULTIPLIER = 2.5f
        const val ONNX_STR_WEIGHT_MULTIPLIER = 1.6f
        const val MAX_STITCH_GAP_SEC = 4.0

        fun normalizeJerseyNumber(raw: String): String? =
            JerseyNumberPosterior.normalizeNumber(raw)
    }
}

/**
 * Multi-team hockey player tracker combining:
 * - Constant-velocity + skate-foot distance + IoU frame association
 * - Contact-only order-free CIELAB + lower-belly kit compatibility gating
 * - Order-free 2-cluster CIELAB + lower-belly team clustering & multi-cue referee exclusion
 * - Crossfade/burst (>12 detections) transient 1-hit ghost suppression
 * - 4.0s same-team hierarchical stitching & Bayesian + Hungarian co-occurrence jersey number solver
 */
class MultiTeamPlayerTracker {

    private class ActiveTrack(
        val id: Int,
        var box: RectNorm,
        var lastSeenSec: Double,
        var emaRgb: FloatArray,
        var kitSummaryLab7: FloatArray,
        var lowerBellyLuma: Float = Float.NaN,
        var torsoUpperLabL: Float = Float.NaN,
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

    private data class KitCentroid(
        var kit7: FloatArray,
        var bellyLuma: Float,
        var torsoUpperLabL: Float,
        var rgb: FloatArray
    )

    private data class SkaterSnapshot(
        val rec: TrackIdentityRegistry.TrackRecord,
        val rgb: FloatArray,
        val meanLab: FloatArray,
        val kit7: FloatArray,
        val bellyLuma: Float,
        val torsoUpperLabL: Float
    )

    private val lock = Any()
    private val activeTracks = ArrayList<ActiveTrack>()
    private val registry = TrackIdentityRegistry()
    private val teamMetaOverrides = LinkedHashMap<TeamId, TeamOverrideMeta>()
    private var clusteredTeamRgb = LinkedHashMap<TeamId, List<Int>>()
    private var clusteredTeamKit7 = LinkedHashMap<TeamId, FloatArray>()
    private var clusteredTeamBellyLuma = LinkedHashMap<TeamId, Float>()
    private var clusteredTeamUpperLabL = LinkedHashMap<TeamId, Float>()
    private var nextTrackId = 1
    private var lastFrameTimestampSec = Double.NaN

    fun reset() = synchronized(lock) {
        activeTracks.clear()
        registry.clear()
        teamMetaOverrides.clear()
        clusteredTeamRgb.clear()
        clusteredTeamKit7.clear()
        clusteredTeamBellyLuma.clear()
        clusteredTeamUpperLabL.clear()
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
                val detKit7 = KitLabFeatures.sanitizeKitSummary7(det.kitSummaryLab7, det.chestRgb)
                val colorDist = rgbDistance(t.emaRgb, detRgb)
                val lumDiff = abs(luminance(t.emaRgb) - luminance(detRgb))

                val inContact = detInContact[di] || trackInContact[ti]
                if (inContact && isCrossKitIncompatible(
                        t.emaRgb, t.kitSummaryLab7, t.lowerBellyLuma, t.torsoUpperLabL,
                        detRgb, detKit7, det.lowerBellyLuma, det.torsoUpperLabL,
                        colorDist, lumDiff
                    )
                ) {
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

            val rec = registry.getOrCreateRecord(t.id, safeTime, det)
            rec.lastVx = t.vx
            rec.lastVy = t.vy
            t.kitSummaryLab7 = rec.meanKitSummaryLab7()
            t.lowerBellyLuma = rec.meanLowerBellyLuma()
            t.torsoUpperLabL = rec.meanTorsoUpperLabL()
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
            val rec = registry.getOrCreateRecord(newId, safeTime, det)
            val t = ActiveTrack(
                id = newId,
                box = det.box,
                lastSeenSec = safeTime,
                emaRgb = toFloatRgb(det.chestRgb),
                kitSummaryLab7 = rec.meanKitSummaryLab7(),
                lowerBellyLuma = rec.meanLowerBellyLuma(),
                torsoUpperLabL = rec.meanTorsoUpperLabL(),
                emaStripeScore = cleanStripe
            )
            activeTracks.add(t)
            detToTrackId[di] = newId
        }

        // Re-cluster teams using order-free CIELAB + lower-belly 2-means and multi-cue referee exclusion.
        val currentFrameTrackIds = HashSet<Int>(detToTrackId.size)
        for (tid in detToTrackId) {
            if (tid >= 0) currentFrameTrackIds.add(tid)
        }
        reclusterTeams(currentSkaterCount = detections.size, currentFrameTrackIds = currentFrameTrackIds)
        val identities = registry.resolveAll()

        val suppressBurstUnconfirmed = detections.size > BURST_GHOST_DETECTION_THRESHOLD
        val trackedPlayers = ArrayList<TrackedPlayer>(detections.size)
        for (di in detections.indices) {
            val det = detections[di]
            val tid = detToTrackId[di]
            val rec = registry.getRecord(tid)
            if (suppressBurstUnconfirmed && (rec?.hitCount ?: 1) <= 1) {
                continue
            }
            val track = activeTracks.firstOrNull { it.id == tid }
            val resolved = identities[tid]
            val speed = if (track != null) hypot(track.vx, track.vy) else 0f
            trackedPlayers.add(
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

    fun recordJerseyRead(
        trackId: Int,
        jerseyNumber: String,
        confidence: Float,
        timestampSec: Double,
        kind: JerseyReadKind = JerseyReadKind.FULL,
        qualityWeight: Float = 1f,
        source: JerseyReadSource = JerseyReadSource.MLKIT_OCR
    ) = synchronized(lock) {
        registry.recordJerseyRead(trackId, jerseyNumber, confidence, timestampSec, kind, qualityWeight, source)
    }

    fun recordJerseyDistribution(
        trackId: Int,
        timestampSec: Double,
        candidates: List<Pair<String, Float>>,
        qualityWeight: Float = 1f,
        source: JerseyReadSource = JerseyReadSource.ONNX_STR,
        partialLeftDigit: Pair<Char, Float>? = null,
        partialRightDigit: Pair<Char, Float>? = null
    ) = synchronized(lock) {
        registry.recordJerseyDistribution(
            trackId = trackId,
            timestampSec = timestampSec,
            candidates = candidates,
            qualityWeight = qualityWeight,
            source = source,
            partialLeftDigit = partialLeftDigit,
            partialRightDigit = partialRightDigit
        )
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

    private fun reclusterTeams(currentSkaterCount: Int, currentFrameTrackIds: Set<Int>) {
        val all = registry.allRecords().toList()
        val snapshots = all.mapNotNull { rec ->
            val rgb = rec.meanRgb() ?: return@mapNotNull null
            SkaterSnapshot(
                rec = rec,
                rgb = rgb,
                meanLab = KitLabFeatures.rgbToLab(rgb),
                kit7 = rec.meanKitSummaryLab7(),
                bellyLuma = rec.meanLowerBellyLuma(),
                torsoUpperLabL = rec.meanTorsoUpperLabL()
            )
        }
        if (snapshots.isEmpty()) return

        var refSumR = 0.0
        var refSumG = 0.0
        var refSumB = 0.0
        var refWeight = 0.0
        val remaining = ArrayList<SkaterSnapshot>(snapshots.size)
        val persistedRefCandidates = ArrayList<SkaterSnapshot>()

        // Pass 1: Pre-classify close/medium striped officials and established officials so they never skew Team A/B centroids.
        for (snap in snapshots) {
            val isCurFrame = snap.rec.trackId in currentFrameTrackIds
            val meanChroma = KitLabFeatures.labChroma(snap.meanLab[1], snap.meanLab[2])
            if (isCloseOrMediumStripedOfficial(snap)) {
                if (isCurFrame) snap.rec.refereeFrameVotes++
                snap.rec.clusteredTeam = TeamId.REFEREE
                val w = snap.rec.colorCount.toDouble()
                refSumR += snap.rgb[0] * w
                refSumG += snap.rgb[1] * w
                refSumB += snap.rgb[2] * w
                refWeight += w
            } else if (snap.rec.isEstablishedOfficial(meanChroma)) {
                persistedRefCandidates.add(snap)
            } else {
                remaining.add(snap)
            }
        }

        if (remaining.isEmpty() && persistedRefCandidates.isEmpty()) {
            clusteredTeamRgb.remove(TeamId.TEAM_A)
            clusteredTeamRgb.remove(TeamId.TEAM_B)
            clusteredTeamKit7.clear()
            clusteredTeamBellyLuma.clear()
            clusteredTeamUpperLabL.clear()
            updateRefereeCentroid(refSumR, refSumG, refSumB, refWeight)
            registry.invalidateCache()
            return
        }

        if (remaining.isEmpty()) {
            for (snap in persistedRefCandidates) {
                snap.rec.clusteredTeam = TeamId.REFEREE
                val w = snap.rec.colorCount.toDouble()
                refSumR += snap.rgb[0] * w; refSumG += snap.rgb[1] * w; refSumB += snap.rgb[2] * w; refWeight += w
            }
            clusteredTeamRgb.remove(TeamId.TEAM_A)
            clusteredTeamRgb.remove(TeamId.TEAM_B)
            clusteredTeamKit7.clear()
            clusteredTeamBellyLuma.clear()
            clusteredTeamUpperLabL.clear()
            updateRefereeCentroid(refSumR, refSumG, refSumB, refWeight)
            registry.invalidateCache()
            return
        }

        if (remaining.size == 1 && persistedRefCandidates.isEmpty()) {
            val only = remaining.first()
            if (only.rec.trackId in currentFrameTrackIds) only.rec.skaterFrameVotes++
            val singleTeam = if (KitLabFeatures.kitEffectiveLightness(only.kit7, only.bellyLuma, only.torsoUpperLabL) < 53.6f) {
                TeamId.TEAM_A
            } else {
                TeamId.TEAM_B
            }
            only.rec.clusteredTeam = singleTeam
            val other = if (singleTeam == TeamId.TEAM_A) TeamId.TEAM_B else TeamId.TEAM_A
            clusteredTeamRgb.remove(other)
            clusteredTeamKit7.remove(other)
            clusteredTeamBellyLuma.remove(other)
            clusteredTeamUpperLabL.remove(other)
            clusteredTeamRgb[singleTeam] = only.rgb.map { it.roundToInt().coerceIn(0, 255) }
            clusteredTeamKit7[singleTeam] = only.kit7
            if (only.bellyLuma.isFinite()) clusteredTeamBellyLuma[singleTeam] = only.bellyLuma else clusteredTeamBellyLuma.remove(singleTeam)
            if (only.torsoUpperLabL.isFinite()) clusteredTeamUpperLabL[singleTeam] = only.torsoUpperLabL else clusteredTeamUpperLabL.remove(singleTeam)
            updateRefereeCentroid(refSumR, refSumG, refSumB, refWeight)
            registry.invalidateCache()
            return
        }

        // Initial 2-means across remaining tracks with minimum cluster-support guard against 1-vs-N collapse.
        var (centroidA, centroidB) = if (remaining.size >= 2) {
            runRobustOrderFree2Means(remaining)
        } else {
            runRobustOrderFree2Means(remaining + persistedRefCandidates)
        }
        var kitSep = sqrt(KitLabFeatures.orderFreeKitDistSq(centroidA.kit7, centroidB.kit7))
        var rgbSep = colorLumDistance(centroidA.rgb, centroidB.rgb)

        // Pass 2: Multi-cue blurred far-board official detection + persisted-official hysteresis.
        val allowBlurredOfficial = currentSkaterCount >= 9 || activeTracks.size >= 9
        val canTestBlurred = allowBlurredOfficial && (kitSep >= MIN_KIT_LAB_SEP || rgbSep >= MIN_RGB_LUM_SEP)
        val labA = KitLabFeatures.rgbToLab(centroidA.rgb)
        val labB = KitLabFeatures.rgbToLab(centroidB.rgb)

        val skatersAfterRefs = ArrayList<SkaterSnapshot>(remaining.size + persistedRefCandidates.size)
        var needRecomputeCentroids = false

        for (snap in persistedRefCandidates) {
            val isCurFrame = snap.rec.trackId in currentFrameTrackIds
            val meanChroma = KitLabFeatures.labChroma(snap.meanLab[1], snap.meanLab[2])
            val rawBlurred = canTestBlurred && isBlurredFarBoardOfficial(snap, labA, labB)
            if (isCurFrame) {
                if (rawBlurred) snap.rec.refereeFrameVotes++ else snap.rec.skaterFrameVotes++
            }
            if (rawBlurred || snap.rec.isEstablishedOfficial(meanChroma)) {
                snap.rec.clusteredTeam = TeamId.REFEREE
                val w = snap.rec.colorCount.toDouble()
                refSumR += snap.rgb[0] * w; refSumG += snap.rgb[1] * w; refSumB += snap.rgb[2] * w; refWeight += w
            } else {
                skatersAfterRefs.add(snap)
                needRecomputeCentroids = true
            }
        }

        for (snap in remaining) {
            val isCurFrame = snap.rec.trackId in currentFrameTrackIds
            val meanChroma = KitLabFeatures.labChroma(snap.meanLab[1], snap.meanLab[2])
            val rawBlurred = canTestBlurred && isBlurredFarBoardOfficial(snap, labA, labB)
            if (isCurFrame) {
                if (rawBlurred) snap.rec.refereeFrameVotes++ else snap.rec.skaterFrameVotes++
            }
            if (rawBlurred || snap.rec.isEstablishedOfficial(meanChroma)) {
                snap.rec.clusteredTeam = TeamId.REFEREE
                val w = snap.rec.colorCount.toDouble()
                refSumR += snap.rgb[0] * w
                refSumG += snap.rgb[1] * w
                refSumB += snap.rgb[2] * w
                refWeight += w
                needRecomputeCentroids = true
            } else {
                skatersAfterRefs.add(snap)
            }
        }

        if (skatersAfterRefs.isEmpty()) {
            clusteredTeamRgb.remove(TeamId.TEAM_A)
            clusteredTeamRgb.remove(TeamId.TEAM_B)
            clusteredTeamKit7.clear()
            clusteredTeamBellyLuma.clear()
            clusteredTeamUpperLabL.clear()
            updateRefereeCentroid(refSumR, refSumG, refSumB, refWeight)
            registry.invalidateCache()
            return
        }

        if (skatersAfterRefs.size == 1) {
            val only = skatersAfterRefs.first()
            val singleTeam = if (KitLabFeatures.kitEffectiveLightness(only.kit7, only.bellyLuma, only.torsoUpperLabL) < 53.6f) {
                TeamId.TEAM_A
            } else {
                TeamId.TEAM_B
            }
            only.rec.clusteredTeam = singleTeam
            val other = if (singleTeam == TeamId.TEAM_A) TeamId.TEAM_B else TeamId.TEAM_A
            clusteredTeamRgb.remove(other)
            clusteredTeamKit7.remove(other)
            clusteredTeamBellyLuma.remove(other)
            clusteredTeamUpperLabL.remove(other)
            clusteredTeamRgb[singleTeam] = only.rgb.map { it.roundToInt().coerceIn(0, 255) }
            clusteredTeamKit7[singleTeam] = only.kit7
            if (only.bellyLuma.isFinite()) clusteredTeamBellyLuma[singleTeam] = only.bellyLuma else clusteredTeamBellyLuma.remove(singleTeam)
            if (only.torsoUpperLabL.isFinite()) clusteredTeamUpperLabL[singleTeam] = only.torsoUpperLabL else clusteredTeamUpperLabL.remove(singleTeam)
            updateRefereeCentroid(refSumR, refSumG, refSumB, refWeight)
            registry.invalidateCache()
            return
        }

        if (needRecomputeCentroids) {
            val recomputed = runRobustOrderFree2Means(skatersAfterRefs)
            centroidA = recomputed.first
            centroidB = recomputed.second
            kitSep = sqrt(KitLabFeatures.orderFreeKitDistSq(centroidA.kit7, centroidB.kit7))
            rgbSep = colorLumDistance(centroidA.rgb, centroidB.rgb)
        }

        if (kitSep < MIN_KIT_LAB_SEP && rgbSep < MIN_RGB_LUM_SEP) {
            // Only one distinct kit color visible so far; assign uniformly by dark/light lightness.
            val singleTeam = if (KitLabFeatures.kitEffectiveLightness(centroidA.kit7, centroidA.bellyLuma, centroidA.torsoUpperLabL) < 53.6f) {
                TeamId.TEAM_A
            } else {
                TeamId.TEAM_B
            }
            var sR = 0.0; var sG = 0.0; var sB = 0.0; var sW = 0.0
            for (snap in skatersAfterRefs) {
                snap.rec.clusteredTeam = singleTeam
                val w = min(snap.rec.colorCount, 15).toDouble()
                sR += snap.rgb[0] * w; sG += snap.rgb[1] * w; sB += snap.rgb[2] * w; sW += w
            }
            val other = if (singleTeam == TeamId.TEAM_A) TeamId.TEAM_B else TeamId.TEAM_A
            clusteredTeamRgb.remove(other)
            clusteredTeamKit7.remove(other)
            clusteredTeamBellyLuma.remove(other)
            clusteredTeamUpperLabL.remove(other)
            if (sW > 0.0) {
                clusteredTeamRgb[singleTeam] = listOf(
                    (sR / sW).roundToInt().coerceIn(0, 255),
                    (sG / sW).roundToInt().coerceIn(0, 255),
                    (sB / sW).roundToInt().coerceIn(0, 255)
                )
            }
            clusteredTeamKit7[singleTeam] = centroidA.kit7
            if (centroidA.bellyLuma.isFinite()) clusteredTeamBellyLuma[singleTeam] = centroidA.bellyLuma
            if (centroidA.torsoUpperLabL.isFinite()) clusteredTeamUpperLabL[singleTeam] = centroidA.torsoUpperLabL
            updateRefereeCentroid(refSumR, refSumG, refSumB, refWeight)
            registry.invalidateCache()
            return
        }

        val allowLegacyOutlier = currentSkaterCount > 10 || activeTracks.size > 10
        var sumAR = 0.0; var sumAG = 0.0; var sumAB = 0.0; var weightA = 0.0
        var sumBR = 0.0; var sumBG = 0.0; var sumBB = 0.0; var weightB = 0.0

        for (snap in skatersAfterRefs) {
            val dA = KitLabFeatures.robustKitAndBellyOrUpperDistSq(
                snap.kit7, snap.bellyLuma, snap.torsoUpperLabL,
                centroidA.kit7, centroidA.bellyLuma, centroidA.torsoUpperLabL
            )
            val dB = KitLabFeatures.robustKitAndBellyOrUpperDistSq(
                snap.kit7, snap.bellyLuma, snap.torsoUpperLabL,
                centroidB.kit7, centroidB.bellyLuma, centroidB.torsoUpperLabL
            )
            var assigned = if (dA <= dB) TeamId.TEAM_A else TeamId.TEAM_B

            // Legacy neutral-gray fallback only when pantsDarkness was not supplied (NaN).
            if (allowLegacyOutlier && !snap.rec.emaPantsDarkness.isFinite() && rgbSep >= 35f) {
                val rgbDistA = colorLumDistance(snap.rgb, centroidA.rgb)
                val rgbDistB = colorLumDistance(snap.rgb, centroidB.rgb)
                val sat = saturationSpread(snap.rgb)
                val minTeamDist = min(rgbDistA, rgbDistB)
                val outlierThreshold = max(45f, 0.30f * rgbSep)
                val isNeutralStriped = sat < 35f
                val isBrightStripedOrOrange = sat > 95f
                if (minTeamDist > outlierThreshold && (isNeutralStriped || isBrightStripedOrOrange)) {
                    assigned = TeamId.REFEREE
                    val w = snap.rec.colorCount.toDouble()
                    refSumR += snap.rgb[0] * w; refSumG += snap.rgb[1] * w; refSumB += snap.rgb[2] * w; refWeight += w
                }
            }

            snap.rec.clusteredTeam = assigned
            val w = min(snap.rec.colorCount, 15).toDouble()
            if (assigned == TeamId.TEAM_A) {
                sumAR += snap.rgb[0] * w; sumAG += snap.rgb[1] * w; sumAB += snap.rgb[2] * w; weightA += w
            } else if (assigned == TeamId.TEAM_B) {
                sumBR += snap.rgb[0] * w; sumBG += snap.rgb[1] * w; sumBB += snap.rgb[2] * w; weightB += w
            }
        }

        clusteredTeamRgb[TeamId.TEAM_A] = if (weightA > 0.0) {
            listOf(
                (sumAR / weightA).roundToInt().coerceIn(0, 255),
                (sumAG / weightA).roundToInt().coerceIn(0, 255),
                (sumAB / weightA).roundToInt().coerceIn(0, 255)
            )
        } else {
            centroidA.rgb.map { it.roundToInt().coerceIn(0, 255) }
        }
        clusteredTeamRgb[TeamId.TEAM_B] = if (weightB > 0.0) {
            listOf(
                (sumBR / weightB).roundToInt().coerceIn(0, 255),
                (sumBG / weightB).roundToInt().coerceIn(0, 255),
                (sumBB / weightB).roundToInt().coerceIn(0, 255)
            )
        } else {
            centroidB.rgb.map { it.roundToInt().coerceIn(0, 255) }
        }

        clusteredTeamKit7[TeamId.TEAM_A] = centroidA.kit7
        clusteredTeamKit7[TeamId.TEAM_B] = centroidB.kit7
        if (centroidA.bellyLuma.isFinite()) clusteredTeamBellyLuma[TeamId.TEAM_A] = centroidA.bellyLuma else clusteredTeamBellyLuma.remove(TeamId.TEAM_A)
        if (centroidB.bellyLuma.isFinite()) clusteredTeamBellyLuma[TeamId.TEAM_B] = centroidB.bellyLuma else clusteredTeamBellyLuma.remove(TeamId.TEAM_B)
        if (centroidA.torsoUpperLabL.isFinite()) clusteredTeamUpperLabL[TeamId.TEAM_A] = centroidA.torsoUpperLabL else clusteredTeamUpperLabL.remove(TeamId.TEAM_A)
        if (centroidB.torsoUpperLabL.isFinite()) clusteredTeamUpperLabL[TeamId.TEAM_B] = centroidB.torsoUpperLabL else clusteredTeamUpperLabL.remove(TeamId.TEAM_B)

        updateRefereeCentroid(refSumR, refSumG, refSumB, refWeight)
        registry.invalidateCache()
    }

    private fun runRobustOrderFree2Means(skaters: List<SkaterSnapshot>): Pair<KitCentroid, KitCentroid> {
        val initial = runOrderFree2Means(skaters)
        if (skaters.size < 6) return initial

        val (itemsA, itemsB, wA, wB) = partitionByCentroids(skaters, initial.first, initial.second)
        if (isBalancedClusterSplit(itemsA, itemsB, wA, wB, skaters.size)) {
            return initial
        }

        // Degenerate 1-vs-N / outlier-hijack split detected: trim tiny minority outlier cluster and retry.
        val majority = if (itemsA.size >= itemsB.size) itemsA else itemsB
        if (majority.size >= 4) {
            val trimmed = runOrderFree2Means(majority)
            val (tA, tB, twA, twB) = partitionByCentroids(majority, trimmed.first, trimmed.second)
            if (isBalancedClusterSplit(tA, tB, twA, twB, majority.size)) {
                val tKitSep = sqrt(KitLabFeatures.orderFreeKitDistSq(trimmed.first.kit7, trimmed.second.kit7))
                val tRgbSep = colorLumDistance(trimmed.first.rgb, trimmed.second.rgb)
                if (tKitSep >= MIN_KIT_LAB_SEP || tRgbSep >= MIN_RGB_LUM_SEP) {
                    return trimmed
                }
            }
        }

        // Fallback: 1D kitEffectiveLightness weighted 2-means (immune to extreme chroma outliers).
        val oneDim = runLightness1D2Means(skaters)
        if (oneDim != null) {
            return oneDim
        }
        return initial
    }

    private data class PartitionResult(
        val itemsA: List<SkaterSnapshot>,
        val itemsB: List<SkaterSnapshot>,
        val weightA: Double,
        val weightB: Double
    )

    private fun partitionByCentroids(
        skaters: List<SkaterSnapshot>,
        cA: KitCentroid,
        cB: KitCentroid
    ): PartitionResult {
        val aList = ArrayList<SkaterSnapshot>(skaters.size)
        val bList = ArrayList<SkaterSnapshot>(skaters.size)
        var wA = 0.0
        var wB = 0.0
        for (snap in skaters) {
            val dA = KitLabFeatures.robustKitAndBellyOrUpperDistSq(
                snap.kit7, snap.bellyLuma, snap.torsoUpperLabL,
                cA.kit7, cA.bellyLuma, cA.torsoUpperLabL
            )
            val dB = KitLabFeatures.robustKitAndBellyOrUpperDistSq(
                snap.kit7, snap.bellyLuma, snap.torsoUpperLabL,
                cB.kit7, cB.bellyLuma, cB.torsoUpperLabL
            )
            val w = min(snap.rec.colorCount, 15).toDouble()
            if (dA <= dB) {
                aList.add(snap)
                wA += w
            } else {
                bList.add(snap)
                wB += w
            }
        }
        return PartitionResult(aList, bList, wA, wB)
    }

    private fun isBalancedClusterSplit(
        itemsA: List<SkaterSnapshot>,
        itemsB: List<SkaterSnapshot>,
        weightA: Double,
        weightB: Double,
        totalCount: Int
    ): Boolean {
        val minCount = min(itemsA.size, itemsB.size)
        val minRequired = max(2, totalCount / 6)
        if (minCount < minRequired) return false
        val totalWeight = weightA + weightB
        if (totalWeight <= 0.0) return false
        if (min(weightA, weightB) < 0.12 * totalWeight) return false

        // Detect outlier-hijack splits where a tiny minority (<25%) splits off a bimodal majority containing both teams.
        if (minCount * 4 < totalCount) {
            val majority = if (itemsA.size >= itemsB.size) itemsA else itemsB
            if (majority.size >= 4) {
                val effs = majority.map {
                    KitLabFeatures.kitEffectiveLightness(it.kit7, it.bellyLuma, it.torsoUpperLabL)
                }.sorted()
                val spread = effs[(effs.size * 85) / 100] - effs[(effs.size * 15) / 100]
                if (spread >= 22f) return false
            }
        }
        return true
    }

    private fun runLightness1D2Means(skaters: List<SkaterSnapshot>): Pair<KitCentroid, KitCentroid>? {
        val anchors = skaters.filter { it.rec.colorCount >= 3 }
            .ifEmpty { skaters.filter { it.rec.colorCount >= 2 } }
            .ifEmpty { skaters }
        val sortedEff = anchors.map {
            KitLabFeatures.kitEffectiveLightness(it.kit7, it.bellyLuma, it.torsoUpperLabL)
        }.sorted()
        var muA = if (sortedEff.size >= 4) sortedEff[(sortedEff.size * 15) / 100] else sortedEff.first()
        var muB = if (sortedEff.size >= 4) sortedEff[(sortedEff.size * 85) / 100] else sortedEff.last()
        if (abs(muB - muA) < 10f) return null

        repeat(12) {
            var sA = 0.0; var wA = 0.0
            var sB = 0.0; var wB = 0.0
            for (snap in skaters) {
                val l = KitLabFeatures.kitEffectiveLightness(snap.kit7, snap.bellyLuma, snap.torsoUpperLabL)
                val w = min(snap.rec.colorCount, 15).toDouble()
                if (abs(l - muA) <= abs(l - muB)) {
                    sA += l * w; wA += w
                } else {
                    sB += l * w; wB += w
                }
            }
            if (wA > 0.0) muA = (sA / wA).toFloat()
            if (wB > 0.0) muB = (sB / wB).toFloat()
        }
        if (abs(muB - muA) < 10f) return null
        if (muA > muB) {
            val tmp = muA; muA = muB; muB = tmp
        }

        val groupA = ArrayList<SkaterSnapshot>()
        val groupB = ArrayList<SkaterSnapshot>()
        var wA = 0.0
        var wB = 0.0
        for (snap in skaters) {
            val l = KitLabFeatures.kitEffectiveLightness(snap.kit7, snap.bellyLuma, snap.torsoUpperLabL)
            val w = min(snap.rec.colorCount, 15).toDouble()
            if (abs(l - muA) <= abs(l - muB)) {
                groupA.add(snap); wA += w
            } else {
                groupB.add(snap); wB += w
            }
        }
        if (!isBalancedClusterSplit(groupA, groupB, wA, wB, skaters.size)) return null
        val cA = computeCentroidFromGroup(groupA) ?: return null
        val cB = computeCentroidFromGroup(groupB) ?: return null
        val lA = KitLabFeatures.kitEffectiveLightness(cA.kit7, cA.bellyLuma, cA.torsoUpperLabL)
        val lB = KitLabFeatures.kitEffectiveLightness(cB.kit7, cB.bellyLuma, cB.torsoUpperLabL)
        return if (lA <= lB) cA to cB else cB to cA
    }

    private fun computeCentroidFromGroup(group: List<SkaterSnapshot>): KitCentroid? {
        if (group.isEmpty()) return null
        val ref = group.first().kit7
        val sumKit = DoubleArray(KitLabFeatures.SUMMARY_SIZE)
        var sumBelly = 0.0; var wBelly = 0.0
        var sumUpper = 0.0; var wUpper = 0.0
        var sumR = 0.0; var sumG = 0.0; var sumB = 0.0; var wTot = 0.0
        for (snap in group) {
            val w = min(snap.rec.colorCount, 15).toDouble()
            val aligned = KitLabFeatures.robustOrderFreeBestAligned(snap.kit7, ref)
            for (k in 0 until KitLabFeatures.SUMMARY_SIZE) sumKit[k] += aligned[k] * w
            sumR += snap.rgb[0] * w; sumG += snap.rgb[1] * w; sumB += snap.rgb[2] * w
            wTot += w
            if (snap.bellyLuma.isFinite()) {
                sumBelly += snap.bellyLuma * w
                wBelly += w
            }
            if (snap.torsoUpperLabL.isFinite()) {
                sumUpper += snap.torsoUpperLabL * w
                wUpper += w
            }
        }
        if (wTot <= 0.0) return null
        val kit7 = FloatArray(KitLabFeatures.SUMMARY_SIZE) { k -> (sumKit[k] / wTot).toFloat() }
        val finalKit = if (kit7[0] > kit7[3]) KitLabFeatures.swapSummary(kit7) else kit7
        return KitCentroid(
            kit7 = finalKit,
            bellyLuma = if (wBelly > 0.0) (sumBelly / wBelly).toFloat() else Float.NaN,
            torsoUpperLabL = if (wUpper > 0.0) (sumUpper / wUpper).toFloat() else Float.NaN,
            rgb = floatArrayOf((sumR / wTot).toFloat(), (sumG / wTot).toFloat(), (sumB / wTot).toFloat())
        )
    }

    private fun runOrderFree2Means(skaters: List<SkaterSnapshot>): Pair<KitCentroid, KitCentroid> {
        val baseAnchors = skaters.filter { it.rec.colorCount >= 3 }
            .ifEmpty { skaters.filter { it.rec.colorCount >= 2 } }
            .ifEmpty { skaters }
        // Trim rare extreme-chroma outliers from seeding only when they form a small tail (<25%).
        val normalChromaAnchors = baseAnchors.filter { KitLabFeatures.maxZoneChroma(it.kit7) <= 25f }
        val anchors = if (normalChromaAnchors.size >= max(4, (baseAnchors.size * 3) / 4)) {
            normalChromaAnchors
        } else {
            baseAnchors
        }
        val sortedByLight = anchors.sortedBy {
            KitLabFeatures.kitEffectiveLightness(it.kit7, it.bellyLuma, it.torsoUpperLabL)
        }

        val seedA = if (sortedByLight.size >= 4) {
            sortedByLight[(sortedByLight.size * 15) / 100]
        } else {
            sortedByLight.first()
        }
        var seedB = if (sortedByLight.size >= 4) {
            sortedByLight[(sortedByLight.size * 85) / 100]
        } else {
            sortedByLight.last()
        }

        // If lightness spread is narrow (e.g. dark green vs dark royal blue), seed B by kit distance from A.
        if (sortedByLight.size >= 4) {
            val lA = KitLabFeatures.kitEffectiveLightness(seedA.kit7, seedA.bellyLuma, seedA.torsoUpperLabL)
            val lB = KitLabFeatures.kitEffectiveLightness(seedB.kit7, seedB.bellyLuma, seedB.torsoUpperLabL)
            if (abs(lB - lA) < 10f) {
                val sortedByDist = anchors.sortedBy {
                    KitLabFeatures.robustKitAndBellyOrUpperDistSq(
                        it.kit7, it.bellyLuma, it.torsoUpperLabL,
                        seedA.kit7, seedA.bellyLuma, seedA.torsoUpperLabL
                    )
                }
                seedB = sortedByDist[(sortedByDist.size * 85) / 100]
            }
        }

        var cA = KitCentroid(seedA.kit7.copyOf(), seedA.bellyLuma, seedA.torsoUpperLabL, seedA.rgb.copyOf())
        var cB = KitCentroid(seedB.kit7.copyOf(), seedB.bellyLuma, seedB.torsoUpperLabL, seedB.rgb.copyOf())

        repeat(12) {
            val sumKitA = DoubleArray(KitLabFeatures.SUMMARY_SIZE)
            val sumKitB = DoubleArray(KitLabFeatures.SUMMARY_SIZE)
            var sumBellyA = 0.0; var wBellyA = 0.0
            var sumBellyB = 0.0; var wBellyB = 0.0
            var sumUpperA = 0.0; var wUpperA = 0.0
            var sumUpperB = 0.0; var wUpperB = 0.0
            var sumRA = 0.0; var sumGA = 0.0; var sumBA = 0.0; var wA = 0.0
            var sumRB = 0.0; var sumGB = 0.0; var sumBB = 0.0; var wB = 0.0

            for (snap in skaters) {
                val dA = KitLabFeatures.robustKitAndBellyOrUpperDistSq(
                    snap.kit7, snap.bellyLuma, snap.torsoUpperLabL,
                    cA.kit7, cA.bellyLuma, cA.torsoUpperLabL
                )
                val dB = KitLabFeatures.robustKitAndBellyOrUpperDistSq(
                    snap.kit7, snap.bellyLuma, snap.torsoUpperLabL,
                    cB.kit7, cB.bellyLuma, cB.torsoUpperLabL
                )
                val w = min(snap.rec.colorCount, 15).toDouble()
                if (dA <= dB) {
                    val aligned = KitLabFeatures.robustOrderFreeBestAligned(snap.kit7, cA.kit7)
                    for (k in 0 until KitLabFeatures.SUMMARY_SIZE) sumKitA[k] += aligned[k] * w
                    sumRA += snap.rgb[0] * w; sumGA += snap.rgb[1] * w; sumBA += snap.rgb[2] * w; wA += w
                    if (snap.bellyLuma.isFinite()) {
                        sumBellyA += snap.bellyLuma * w
                        wBellyA += w
                    }
                    if (snap.torsoUpperLabL.isFinite()) {
                        sumUpperA += snap.torsoUpperLabL * w
                        wUpperA += w
                    }
                } else {
                    val aligned = KitLabFeatures.robustOrderFreeBestAligned(snap.kit7, cB.kit7)
                    for (k in 0 until KitLabFeatures.SUMMARY_SIZE) sumKitB[k] += aligned[k] * w
                    sumRB += snap.rgb[0] * w; sumGB += snap.rgb[1] * w; sumBB += snap.rgb[2] * w; wB += w
                    if (snap.bellyLuma.isFinite()) {
                        sumBellyB += snap.bellyLuma * w
                        wBellyB += w
                    }
                    if (snap.torsoUpperLabL.isFinite()) {
                        sumUpperB += snap.torsoUpperLabL * w
                        wUpperB += w
                    }
                }
            }

            if (wA > 0.0) {
                cA = KitCentroid(
                    kit7 = FloatArray(KitLabFeatures.SUMMARY_SIZE) { k -> (sumKitA[k] / wA).toFloat() },
                    bellyLuma = if (wBellyA > 0.0) (sumBellyA / wBellyA).toFloat() else Float.NaN,
                    torsoUpperLabL = if (wUpperA > 0.0) (sumUpperA / wUpperA).toFloat() else Float.NaN,
                    rgb = floatArrayOf((sumRA / wA).toFloat(), (sumGA / wA).toFloat(), (sumBA / wA).toFloat())
                )
            }
            if (wB > 0.0) {
                cB = KitCentroid(
                    kit7 = FloatArray(KitLabFeatures.SUMMARY_SIZE) { k -> (sumKitB[k] / wB).toFloat() },
                    bellyLuma = if (wBellyB > 0.0) (sumBellyB / wBellyB).toFloat() else Float.NaN,
                    torsoUpperLabL = if (wUpperB > 0.0) (sumUpperB / wUpperB).toFloat() else Float.NaN,
                    rgb = floatArrayOf((sumRB / wB).toFloat(), (sumGB / wB).toFloat(), (sumBB / wB).toFloat())
                )
            }
        }

        // Canonicalize dark-first summary slots for final centroids.
        if (cA.kit7[0] > cA.kit7[3]) cA.kit7 = KitLabFeatures.swapSummary(cA.kit7)
        if (cB.kit7[0] > cB.kit7[3]) cB.kit7 = KitLabFeatures.swapSummary(cB.kit7)

        // Enforce TEAM_A = darker kit, TEAM_B = lighter kit.
        val lightA = KitLabFeatures.kitEffectiveLightness(cA.kit7, cA.bellyLuma, cA.torsoUpperLabL)
        val lightB = KitLabFeatures.kitEffectiveLightness(cB.kit7, cB.bellyLuma, cB.torsoUpperLabL)
        return if (lightA <= lightB) cA to cB else cB to cA
    }

    private fun isCloseOrMediumStripedOfficial(snap: SkaterSnapshot): Boolean {
        val rec = snap.rec
        val chroma = KitLabFeatures.labChroma(snap.meanLab[1], snap.meanLab[2])
        if (rec.lastStripeFrameHit && chroma < 8.5f) return true

        val stripeHit = rec.emaStripeScore >= STRIPE_REFEREE_MIN_SCORE ||
            (rec.emaStripeScore >= STRIPE_ARMBAND_REFEREE_MIN_SCORE && rec.emaArmbandScore >= ARMBAND_MIN_SCORE)
        if (!stripeHit) return false
        if (chroma >= STRIPE_REFEREE_MAX_CHROMA) return false

        val pantsOk = !rec.emaPantsDarkness.isFinite() ||
            rec.emaPantsDarkness >= 0.30f ||
            rec.emaArmbandScore >= ARMBAND_MIN_SCORE
        if (!pantsOk) return false

        // Require bright stripe support so solid dark navy skaters never trigger REFEREE.
        return snap.kit7[3] >= 44f ||
            snap.meanLab[0] >= 36f ||
            (snap.bellyLuma.isFinite() && snap.bellyLuma >= 85f) ||
            (snap.torsoUpperLabL.isFinite() && snap.torsoUpperLabL >= 46f)
    }

    private fun isBlurredFarBoardOfficial(
        snap: SkaterSnapshot,
        teamLabA: FloatArray,
        teamLabB: FloatArray
    ): Boolean {
        val rec = snap.rec
        if (!rec.emaPantsDarkness.isFinite() || rec.emaPantsDarkness < BLURRED_PANTS_DARKNESS_MIN) return false

        val meanL = snap.meanLab[0]
        val meanChroma = KitLabFeatures.labChroma(snap.meanLab[1], snap.meanLab[2])
        val maxZoneChroma = KitLabFeatures.maxZoneChroma(snap.kit7)
        val distToA = sqrt(KitLabFeatures.labDistSq(snap.meanLab, teamLabA))
        val distToB = sqrt(KitLabFeatures.labDistSq(snap.meanLab, teamLabB))
        val minTeamDist = min(distToA, distToB)

        // Legacy fallback path when torsoUpperLabL is not supplied (NaN).
        if (!snap.torsoUpperLabL.isFinite()) {
            if (rec.colorCount < 2) return false
            if (meanL < 30f || meanL > 72f) return false
            if (meanChroma >= BLURRED_MAX_MEAN_CHROMA) return false
            if (maxZoneChroma >= BLURRED_MAX_ZONE_CHROMA) return false
            if (snap.kit7[3] >= 75f && snap.kit7[6] >= 42f) return false
            if (snap.bellyLuma.isFinite() && snap.bellyLuma >= 150f) return false
            val distToGrey = sqrt(KitLabFeatures.labDistSq(snap.meanLab, NEUTRAL_GREY_LAB))
            if (rec.emaPantsDarkness >= 0.75f &&
                meanL in 31f..72f &&
                meanChroma < 10.0f &&
                (!snap.bellyLuma.isFinite() || snap.bellyLuma < 135f)
            ) {
                return minTeamDist >= 3.0f && distToGrey <= minTeamDist * 1.6f
            }
            return minTeamDist >= 8.0f && distToGrey <= minTeamDist * 1.15f
        }

        // Round 2 shadow-resilient neutral blurred-official detector using torsoUpperLabL & light share.
        if (rec.colorCount < 6) return false
        if (meanL < 25f || meanL > 66f) return false
        if (meanChroma >= 13.0f || maxZoneChroma >= 19.0f) return false
        if (snap.torsoUpperLabL >= 56f || snap.kit7[3] >= 56f || (snap.bellyLuma.isFinite() && snap.bellyLuma >= 135f)) {
            return false
        }
        if (snap.kit7[6] < 68f) return false
        if (rec.emaPantsDarkness >= 0.72f && meanChroma < 10.5f && minTeamDist >= 5.0f && meanL <= teamLabB[0] + 2.0f) {
            return true
        }
        return rec.emaPantsDarkness >= 0.60f && meanChroma < 4.5f && maxZoneChroma < 6.0f && minTeamDist >= 8.0f && meanL <= teamLabB[0]
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
        trackKit7: FloatArray,
        trackBelly: Float,
        trackUpperL: Float,
        detRgb: FloatArray,
        detKit7: FloatArray,
        detBelly: Float,
        detUpperL: Float,
        rgbDist: Float,
        lumDiff: Float
    ): Boolean {
        val kitA = clusteredTeamKit7[TeamId.TEAM_A]
        val kitB = clusteredTeamKit7[TeamId.TEAM_B]
        val centA = clusteredTeamRgb[TeamId.TEAM_A]?.let { toFloatRgb(it) }
        val centB = clusteredTeamRgb[TeamId.TEAM_B]?.let { toFloatRgb(it) }
        val bellyA = clusteredTeamBellyLuma[TeamId.TEAM_A] ?: Float.NaN
        val bellyB = clusteredTeamBellyLuma[TeamId.TEAM_B] ?: Float.NaN
        val upperA = clusteredTeamUpperLabL[TeamId.TEAM_A] ?: Float.NaN
        val upperB = clusteredTeamUpperLabL[TeamId.TEAM_B] ?: Float.NaN

        if (kitA != null && kitB != null && centA != null && centB != null) {
            val kitSep = sqrt(KitLabFeatures.orderFreeKitDistSq(kitA, kitB))
            val rgbSep = colorLumDistance(centA, centB)
            if (kitSep >= 12f || rgbSep >= 40f) {
                val tDistA = KitLabFeatures.kitAndBellyOrUpperDistSq(trackKit7, trackBelly, trackUpperL, kitA, bellyA, upperA)
                val tDistB = KitLabFeatures.kitAndBellyOrUpperDistSq(trackKit7, trackBelly, trackUpperL, kitB, bellyB, upperB)
                val dDistA = KitLabFeatures.kitAndBellyOrUpperDistSq(detKit7, detBelly, detUpperL, kitA, bellyA, upperA)
                val dDistB = KitLabFeatures.kitAndBellyOrUpperDistSq(detKit7, detBelly, detUpperL, kitB, bellyB, upperB)
                val tIsA = tDistA <= tDistB
                val dIsA = dDistA <= dDistB
                if (tIsA != dIsA) {
                    val kitCrossDist = KitLabFeatures.kitAndBellyOrUpperDistSq(
                        trackKit7, trackBelly, trackUpperL, detKit7, detBelly, detUpperL
                    )
                    if (rgbDist > 65f || kitCrossDist > 250f) return true
                }
            }
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
        private const val BURST_GHOST_DETECTION_THRESHOLD = 15

        private const val STRIPE_REFEREE_MIN_SCORE = 0.46f
        private const val STRIPE_ARMBAND_REFEREE_MIN_SCORE = 0.34f
        internal const val ARMBAND_MIN_SCORE = 0.04f
        private const val STRIPE_REFEREE_MAX_CHROMA = 14.5f

        private const val BLURRED_PANTS_DARKNESS_MIN = 0.60f
        private const val BLURRED_MAX_MEAN_CHROMA = 15.0f
        private const val BLURRED_MAX_ZONE_CHROMA = 17.5f

        private const val MIN_KIT_LAB_SEP = 11.0f
        private const val MIN_RGB_LUM_SEP = 25.0f

        private val NEUTRAL_GREY_LAB = floatArrayOf(55f, 0f, 0f)

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
            return sqrt(dr * dr + dg * dg + db * db)
        }

        private fun luminance(rgb: FloatArray): Float =
            0.299f * rgb[0] + 0.587f * rgb[1] + 0.114f * rgb[2]

        private fun saturationSpread(rgb: FloatArray): Float =
            maxOf(rgb[0], rgb[1], rgb[2]) - minOf(rgb[0], rgb[1], rgb[2])

        private fun colorLumDistance(a: FloatArray, b: FloatArray): Float {
            val dr = a[0] - b[0]
            val dg = a[1] - b[1]
            val db = a[2] - b[2]
            val dl = luminance(a) - luminance(b)
            return sqrt(dr * dr + dg * dg + db * db + 1.5f * dl * dl)
        }

        private fun defaultRgbFor(team: TeamId): List<Int> = when (team) {
            TeamId.TEAM_A -> listOf(32, 68, 52)
            TeamId.TEAM_B -> listOf(228, 232, 236)
            TeamId.REFEREE -> listOf(128, 128, 128)
            TeamId.UNKNOWN -> listOf(128, 128, 128)
        }
    }
}
