package com.engtestbench.hockeynano.core.bench

import kotlinx.serialization.Serializable

@Serializable
data class HarnessCell(
    val harness: String,
    val model: String,
    val passed: Int,
    val total: Int = 27,
    val costUsd: Double? = null,
    val medianSec: Double? = null,
    val badge: String? = null,
    val lookupFail: Int = 0
)

@Serializable
data class BenchSeedReport(
    val title: String,
    val subtitle: String,
    val totalTrials: Int,
    val totalSpendUsd: Double,
    val taskCount: Int,
    val harnesses: List<String>,
    val models: List<String>,
    val cells: List<HarnessCell>,
    val findings: List<String>,
    val designsCount: Int
)
