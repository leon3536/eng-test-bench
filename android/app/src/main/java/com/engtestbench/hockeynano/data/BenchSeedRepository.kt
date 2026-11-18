package com.engtestbench.hockeynano.data

import android.content.Context
import com.engtestbench.hockeynano.core.bench.BenchSeedReport
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException
import java.util.concurrent.TimeUnit

class BenchSeedRepository(private val context: Context) {
    private val json = Json {
        ignoreUnknownKeys = true
    }

    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(3, TimeUnit.SECONDS)
        .readTimeout(4, TimeUnit.SECONDS)
        .callTimeout(5, TimeUnit.SECONDS)
        .build()

    suspend fun loadSeedReport(): BenchSeedReport = withContext(Dispatchers.IO) {
        val raw = context.assets.open(SEED_ASSET_NAME).bufferedReader().use { it.readText() }
        json.decodeFromString<BenchSeedReport>(raw)
    }

    suspend fun probeLiveBenchServer(baseUrl: String): Result<String> = withContext(Dispatchers.IO) {
        runCatching {
            val cleanBase = normalizeBaseUrl(baseUrl)
            require(cleanBase.isNotEmpty()) { "Base URL is empty" }

            var lastError: Throwable? = null
            var authProtectedResponse: String? = null

            for (endpoint in PROBE_ENDPOINTS) {
                val url = "$cleanBase$endpoint"
                val req = Request.Builder()
                    .url(url)
                    .get()
                    .header("Accept", "application/json")
                    .build()

                try {
                    client.newCall(req).execute().use { resp ->
                        val code = resp.code
                        val bodySnippet = resp.body?.string()
                            ?.trim()
                            ?.replace(Regex("\\s+"), " ")
                            ?.take(220)
                            .orEmpty()

                        if (resp.isSuccessful) {
                            val summary = if (bodySnippet.isNotEmpty()) bodySnippet else "OK"
                            return@runCatching "Live ($endpoint HTTP $code): $summary"
                        }
                        if (code == 401 || code == 403) {
                            authProtectedResponse = "Reachable ($endpoint HTTP $code): session token required"
                        }
                        lastError = IOException("HTTP $code from $url")
                    }
                } catch (e: IOException) {
                    lastError = e
                }
            }

            authProtectedResponse?.let { return@runCatching it }
            throw lastError ?: IOException("Failed to reach $cleanBase")
        }
    }

    private fun normalizeBaseUrl(raw: String): String {
        val trimmed = raw.trim().trimEnd('/')
        if (trimmed.isEmpty()) return ""
        return if (trimmed.startsWith("http://", ignoreCase = true) ||
            trimmed.startsWith("https://", ignoreCase = true)
        ) {
            trimmed
        } else {
            "http://$trimmed"
        }
    }

    companion object {
        private const val SEED_ASSET_NAME = "bench_summary_seed.json"
        private val PROBE_ENDPOINTS = listOf(
            "/api/status",
            "/api/summary",
            "/api/state",
            "/api/ledger/status"
        )
    }
}
