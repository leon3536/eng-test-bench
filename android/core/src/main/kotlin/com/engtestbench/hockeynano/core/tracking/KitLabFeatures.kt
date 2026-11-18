package com.engtestbench.hockeynano.core.tracking

import kotlin.math.abs
import kotlin.math.cbrt
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Pure-Kotlin sRGB -> CIELAB (D65) conversion and order-free 2-cluster kit feature geometry.
 *
 * Each detection's torso kit is summarized as a 7-element vector:
 * `[darkL, darkA, darkB, lightL, lightA, lightB, lightSharePct(0..100)]`
 * paired with an optional `lowerBellyLuma` (`0..255` median luma on `y: 42%..56%` below shoulder yoke)
 * so white away jerseys with dark shoulder yokes stay bound to the Light team (`TEAM_B`).
 */
object KitLabFeatures {

    const val SUMMARY_SIZE = 7
    const val WEIGHT_L = 0.45f
    const val WEIGHT_AB = 1.0f
    const val WEIGHT_SHARE = 0.55f
    const val DEFAULT_BELLY_WEIGHT = 1.25f
    const val HUBER_AB_DELTA = 22f

    private const val LAB_DELTA = 6.0 / 29.0
    private const val LAB_DELTA_CUBED = LAB_DELTA * LAB_DELTA * LAB_DELTA
    private const val LAB_SCALE = 1.0 / (3.0 * LAB_DELTA * LAB_DELTA)
    private const val LAB_OFFSET = 4.0 / 29.0

    private const val D65_XN = 0.95047
    private const val D65_YN = 1.00000
    private const val D65_ZN = 1.08883

    /** Converts sRGB (`0..255`) to CIELAB `[L*(0..100), a*, b*]`. */
    fun rgbToLab(r: Int, g: Int, b: Int): FloatArray =
        rgbToLab(r.coerceIn(0, 255).toFloat(), g.coerceIn(0, 255).toFloat(), b.coerceIn(0, 255).toFloat())

    /** Converts a 3-element sRGB list (`0..255`) to CIELAB `[L*, a*, b*]`. */
    fun rgbToLab(rgb: List<Int>): FloatArray = rgbToLab(
        rgb.getOrNull(0) ?: 128,
        rgb.getOrNull(1) ?: 128,
        rgb.getOrNull(2) ?: 128
    )

    /** Converts a 3-element float sRGB array (`0..255`) to CIELAB `[L*, a*, b*]`. */
    fun rgbToLab(rgb: FloatArray): FloatArray = rgbToLab(
        rgb.getOrNull(0) ?: 128f,
        rgb.getOrNull(1) ?: 128f,
        rgb.getOrNull(2) ?: 128f
    )

    /** Converts continuous sRGB (`0..255`) to CIELAB `[L*(0..100), a*, b*]`. */
    fun rgbToLab(r: Float, g: Float, b: Float): FloatArray {
        val sr = (if (r.isFinite()) r else 128f).coerceIn(0f, 255f) / 255.0
        val sg = (if (g.isFinite()) g else 128f).coerceIn(0f, 255f) / 255.0
        val sb = (if (b.isFinite()) b else 128f).coerceIn(0f, 255f) / 255.0

        val lr = srgbToLinear(sr)
        val lg = srgbToLinear(sg)
        val lb = srgbToLinear(sb)

        val x = 0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb
        val y = 0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb
        val z = 0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb

        val fx = labF(x / D65_XN)
        val fy = labF(y / D65_YN)
        val fz = labF(z / D65_ZN)

        val l = (116.0 * fy - 16.0).coerceIn(0.0, 100.0)
        val labA = 500.0 * (fx - fy)
        val labB = 200.0 * (fy - fz)
        return floatArrayOf(l.toFloat(), labA.toFloat(), labB.toFloat())
    }

    /** Maps a neutral luma (`0..255`) to perceptual CIELAB `L*` (`0..100`). */
    fun lumaToLabL(luma0To255: Float): Float {
        val clamped = (if (luma0To255.isFinite()) luma0To255 else 128f).coerceIn(0f, 255f) / 255.0
        val linY = srgbToLinear(clamped)
        val fy = labF(linY / D65_YN)
        return (116.0 * fy - 16.0).coerceIn(0.0, 100.0).toFloat()
    }

    /** Returns CIELAB chroma `C* = hypot(a*, b*)`. */
    fun labChroma(a: Float, b: Float): Float = hypot(a, b)

    /** Squared weighted CIELAB distance between two `[L*, a*, b*]` colors. */
    fun labDistSq(lab1: FloatArray, lab2: FloatArray, wL: Float = 1f, wAB: Float = 1f): Float {
        val dL = (lab1[0] - lab2[0]) * wL
        val dA = (lab1[1] - lab2[1]) * wAB
        val dB = (lab1[2] - lab2[2]) * wAB
        return dL * dL + dA * dA + dB * dB
    }

    /** Exchanges the two color slots and complements `lightSharePct` -> `100f - ls`. */
    fun swapSummary(summary7: FloatArray): FloatArray = floatArrayOf(
        summary7[3], summary7[4], summary7[5],
        summary7[0], summary7[1], summary7[2],
        100f - summary7[6]
    )

    /** Synthesizes a 7-element kit summary when explicit 2-cluster Lab features are absent. */
    fun synthesizeKitSummary7(chestRgb: List<Int>): FloatArray {
        val lab = rgbToLab(chestRgb)
        val share = if (lab[0] >= 62f) 100f else 0f
        return floatArrayOf(lab[0], lab[1], lab[2], lab[0], lab[1], lab[2], share)
    }

    /** Synthesizes a 7-element kit summary from float RGB (`0..255`). */
    fun synthesizeKitSummary7(chestRgb: FloatArray): FloatArray {
        val lab = rgbToLab(chestRgb)
        val share = if (lab[0] >= 62f) 100f else 0f
        return floatArrayOf(lab[0], lab[1], lab[2], lab[0], lab[1], lab[2], share)
    }

    /**
     * Validates and canonicalizes a `kitSummaryLab7` (`darkL <= lightL`, `lightSharePct in 0..100`),
     * falling back to `synthesizeKitSummary7(fallbackRgb)` if null or non-finite.
     */
    fun sanitizeKitSummary7(kitSummaryLab7: FloatArray?, fallbackRgb: List<Int>): FloatArray {
        if (kitSummaryLab7 == null || kitSummaryLab7.size < SUMMARY_SIZE) {
            return synthesizeKitSummary7(fallbackRgb)
        }
        for (i in 0 until SUMMARY_SIZE) {
            if (!kitSummaryLab7[i].isFinite()) {
                return synthesizeKitSummary7(fallbackRgb)
            }
        }
        val l0 = kitSummaryLab7[0].coerceIn(0f, 100f)
        val a0 = kitSummaryLab7[1].coerceIn(-128f, 128f)
        val b0 = kitSummaryLab7[2].coerceIn(-128f, 128f)
        val l1 = kitSummaryLab7[3].coerceIn(0f, 100f)
        val a1 = kitSummaryLab7[4].coerceIn(-128f, 128f)
        val b1 = kitSummaryLab7[5].coerceIn(-128f, 128f)
        val share = kitSummaryLab7[6].coerceIn(0f, 100f)
        return if (l0 <= l1) {
            floatArrayOf(l0, a0, b0, l1, a1, b1, share)
        } else {
            floatArrayOf(l1, a1, b1, l0, a0, b0, 100f - share)
        }
    }

    /** Weighted squared distance in 7-D Lab+share space without swapping. */
    fun weightedKitDistSq(x: FloatArray, c: FloatArray): Float {
        val dL0 = (x[0] - c[0]) * WEIGHT_L
        val dA0 = (x[1] - c[1]) * WEIGHT_AB
        val dB0 = (x[2] - c[2]) * WEIGHT_AB
        val dL1 = (x[3] - c[3]) * WEIGHT_L
        val dA1 = (x[4] - c[4]) * WEIGHT_AB
        val dB1 = (x[5] - c[5]) * WEIGHT_AB
        val dShare = (x[6] - c[6]) * WEIGHT_SHARE
        return dL0 * dL0 + dA0 * dA0 + dB0 * dB0 +
            dL1 * dL1 + dA1 * dA1 + dB1 * dB1 +
            dShare * dShare
    }

    /**
     * Order-free squared distance between two 7-element kit summaries:
     * `min(weightedDistSq(x, c), weightedDistSq(swap(x), c))`.
     */
    fun orderFreeKitDistSq(x: FloatArray, c: FloatArray): Float {
        val direct = weightedKitDistSq(x, c)
        val swapped = weightedKitDistSq(swapSummary(x), c)
        return min(direct, swapped)
    }

    /** Returns `x` or `swapSummary(x)`, whichever aligns more closely with centroid `c`. */
    fun orderFreeBestAligned(x: FloatArray, c: FloatArray): FloatArray {
        val swapped = swapSummary(x)
        return if (weightedKitDistSq(x, c) <= weightedKitDistSq(swapped, c)) x else swapped
    }

    /**
     * Computes perceptual lightness (`0..100`) of a kit summary combined with optional
     * `lowerBellyLuma` (`0..255`), keeping dark-yoke white jerseys brighter than dark jerseys.
     */
    fun kitEffectiveLightness(summary7: FloatArray, lowerBellyLuma: Float = Float.NaN): Float {
        val shareFrac = (summary7[6] / 100f).coerceIn(0f, 1f)
        val kitMeanL = (1f - shareFrac) * summary7[0] + shareFrac * summary7[3]
        return if (lowerBellyLuma.isFinite()) {
            0.55f * kitMeanL + 0.45f * lumaToLabL(lowerBellyLuma)
        } else {
            kitMeanL
        }
    }

    /**
     * Combined order-free kit distance plus lower-belly luminance separation penalty when both
     * `bellyX` and `bellyC` are finite.
     */
    fun kitAndBellyDistSq(
        kitX: FloatArray,
        bellyX: Float,
        kitC: FloatArray,
        bellyC: Float,
        bellyWeight: Float = DEFAULT_BELLY_WEIGHT
    ): Float {
        var dist = orderFreeKitDistSq(kitX, kitC)
        if (bellyX.isFinite() && bellyC.isFinite()) {
            val dBelly = (lumaToLabL(bellyX) - lumaToLabL(bellyC)) * bellyWeight
            dist += dBelly * dBelly
        }
        return dist
    }

    /** Symmetric Huber-like quadratic penalty: quadratic for `|diff| <= delta`, linear tail beyond. */
    fun huberSq(diff: Float, delta: Float = HUBER_AB_DELTA): Float {
        val a = abs(diff)
        return if (a <= delta) a * a else delta * (2f * a - delta)
    }

    /** Huberized weighted squared distance clamping extreme chroma differences at `abDelta`. */
    fun robustWeightedKitDistSq(
        x: FloatArray,
        c: FloatArray,
        abDelta: Float = HUBER_AB_DELTA
    ): Float {
        val dL0 = (x[0] - c[0]) * WEIGHT_L
        val dL1 = (x[3] - c[3]) * WEIGHT_L
        val dShare = (x[6] - c[6]) * WEIGHT_SHARE
        return dL0 * dL0 +
            huberSq((x[1] - c[1]) * WEIGHT_AB, abDelta) +
            huberSq((x[2] - c[2]) * WEIGHT_AB, abDelta) +
            dL1 * dL1 +
            huberSq((x[4] - c[4]) * WEIGHT_AB, abDelta) +
            huberSq((x[5] - c[5]) * WEIGHT_AB, abDelta) +
            dShare * dShare
    }

    /** Order-free distance with Huberized `(a*, b*)` chroma channels to resist board/glare outliers. */
    fun robustOrderFreeKitDistSq(
        x: FloatArray,
        c: FloatArray,
        abDelta: Float = HUBER_AB_DELTA
    ): Float {
        val direct = robustWeightedKitDistSq(x, c, abDelta)
        val swapped = robustWeightedKitDistSq(swapSummary(x), c, abDelta)
        return min(direct, swapped)
    }

    /** Robust best orientation (`x` or `swapSummary(x)`) under Huberized kit distance. */
    fun robustOrderFreeBestAligned(
        x: FloatArray,
        c: FloatArray,
        abDelta: Float = HUBER_AB_DELTA
    ): FloatArray {
        val swapped = swapSummary(x)
        return if (robustWeightedKitDistSq(x, c, abDelta) <= robustWeightedKitDistSq(swapped, c, abDelta)) x else swapped
    }

    /** Combined Huberized order-free kit distance plus lower-belly luminance separation penalty. */
    fun robustKitAndBellyDistSq(
        kitX: FloatArray,
        bellyX: Float,
        kitC: FloatArray,
        bellyC: Float,
        bellyWeight: Float = DEFAULT_BELLY_WEIGHT,
        abDelta: Float = HUBER_AB_DELTA
    ): Float {
        var dist = robustOrderFreeKitDistSq(kitX, kitC, abDelta)
        if (bellyX.isFinite() && bellyC.isFinite()) {
            val dBelly = (lumaToLabL(bellyX) - lumaToLabL(bellyC)) * bellyWeight
            dist += dBelly * dBelly
        }
        return dist
    }

    /** Returns max chroma `C*` across the two color slots of a 7-element summary. */
    fun maxZoneChroma(summary7: FloatArray): Float =
        max(abs(labChroma(summary7[1], summary7[2])), abs(labChroma(summary7[4], summary7[5])))

    private fun srgbToLinear(u: Double): Double =
        if (u <= 0.04045) u / 12.92 else ((u + 0.055) / 1.055).pow(2.4)

    private fun labF(t: Double): Double =
        if (t > LAB_DELTA_CUBED) cbrt(t) else (t * LAB_SCALE) + LAB_OFFSET
}
