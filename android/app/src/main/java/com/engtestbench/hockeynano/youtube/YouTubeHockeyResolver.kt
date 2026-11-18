package com.engtestbench.hockeynano.youtube

import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import org.schabi.newpipe.extractor.MediaFormat
import org.schabi.newpipe.extractor.NewPipe
import org.schabi.newpipe.extractor.ServiceList
import org.schabi.newpipe.extractor.downloader.Downloader
import org.schabi.newpipe.extractor.downloader.Request
import org.schabi.newpipe.extractor.downloader.Response
import org.schabi.newpipe.extractor.exceptions.ReCaptchaException
import org.schabi.newpipe.extractor.stream.AudioStream
import org.schabi.newpipe.extractor.stream.AudioTrackType
import org.schabi.newpipe.extractor.stream.DeliveryMethod
import org.schabi.newpipe.extractor.stream.StreamInfo
import org.schabi.newpipe.extractor.stream.VideoStream
import java.net.URI
import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt

@Serializable
data class ResolvedYouTubeVideo(
    val originalUrl: String,
    val videoId: String,
    val title: String,
    val videoStreamUrl: String,
    val audioStreamUrl: String? = null,
    val muxedFallbackUrl: String? = null,
    val width: Int = 1920,
    val height: Int = 1080,
    val hasEmbeddedAudio: Boolean = false,
    val durationSec: Double = 0.0,
    val startOffsetSec: Double? = null,
    val endOffsetSec: Double? = null,
    val resolvedAtElapsedMs: Long = 0L
)

/**
 * Resolves YouTube watch, mobile, youtu.be, live, shorts, and clip links to direct progressive
 * HTTP streams via NewPipeExtractor without solving/bypassing anti-bot or ReCaptcha challenges.
 */
object YouTubeHockeyResolver {

    const val USER_AGENT: String =
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"

    private const val LOG_TAG = "HockeyNanoBench"

    private val TIME_COMPOUND_REGEX =
        Regex("""^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s?)?$""", RegexOption.IGNORE_CASE)

    private val client: OkHttpClient = OkHttpClient.Builder()
        .readTimeout(30, TimeUnit.SECONDS)
        .connectTimeout(15, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    @Volatile
    private var initialized: Boolean = false

    private fun ensureInit() {
        if (!initialized) {
            synchronized(this) {
                if (!initialized) {
                    NewPipe.init(OkHttpDownloader(client))
                    initialized = true
                }
            }
        }
    }

    fun isYouTubeUrl(input: String?): Boolean {
        val trimmed = input?.trim().orEmpty()
        if (trimmed.isEmpty() || trimmed.startsWith("/")) return false

        val candidate = if ("://" in trimmed) {
            val scheme = trimmed.substringBefore("://").lowercase()
            if (scheme != "http" && scheme != "https") return false
            trimmed
        } else {
            "https://$trimmed"
        }

        val uri = runCatching { URI(candidate) }.getOrNull() ?: return false
        val host = uri.host?.lowercase()?.removeSuffix(".") ?: return false
        val isYouTubeHost = host == "youtube.com" ||
            host.endsWith(".youtube.com") ||
            host == "youtu.be" ||
            host.endsWith(".youtu.be") ||
            host == "youtube-nocookie.com" ||
            host.endsWith(".youtube-nocookie.com")
        if (!isYouTubeHost) return false

        val path = uri.path.orEmpty()
        val query = uri.rawQuery.orEmpty()
        if (host == "youtu.be" || host.endsWith(".youtu.be")) {
            return path.trim('/').isNotEmpty()
        }

        return (path.startsWith("/watch") && query.contains("v=")) ||
            path.startsWith("/live/") ||
            path.startsWith("/shorts/") ||
            path.startsWith("/clip/") ||
            path.startsWith("/embed/") ||
            path.startsWith("/v/")
    }

    suspend fun resolve(url: String, preferMaxHeight: Int = 1080): ResolvedYouTubeVideo =
        withContext(Dispatchers.IO) {
            ensureInit()
            val trimmed = url.trim()
            require(trimmed.isNotEmpty() && isYouTubeUrl(trimmed)) {
                "Unsupported YouTube URL: $url"
            }
            val normalizedUrl = if ("://" in trimmed) trimmed else "https://$trimmed"

            var clipSpan: Pair<Double, Double>? = null
            val watchUrl = if ("/clip/" in normalizedUrl) {
                val clip = resolveClip(normalizedUrl)
                clipSpan = clip.second
                "https://www.youtube.com/watch?v=${clip.first}"
            } else {
                normalizedUrl
            }

            val info = StreamInfo.getInfo(ServiceList.YouTube, watchUrl)
            val maxHeight = preferMaxHeight.coerceAtLeast(144)

            val muxedStreams = info.videoStreams.filter { it.isPlayableProgressiveAvc(maxHeight) }
            val videoOnlyStreams = info.videoOnlyStreams.filter { it.isPlayableProgressiveAvc(maxHeight) }

            val muxedBest = muxedStreams.maxWithOrNull(
                compareBy<VideoStream> { it.height }
                    .thenBy { it.fps }
                    .thenBy { it.bitrate }
            )

            val best = (videoOnlyStreams.filter { it.height >= 480 } + muxedStreams)
                .ifEmpty { videoOnlyStreams }
                .maxWithOrNull(
                    compareBy<VideoStream> { it.height }
                        .thenBy { !it.isVideoOnly }
                        .thenBy { it.fps }
                        .thenBy { it.bitrate }
                )
                ?: throw IllegalStateException(
                    "No playable progressive MPEG-4 AVC stream <= ${maxHeight}p for ${info.id}"
                )

            val bestAudio = if (best.isVideoOnly) {
                info.audioStreams
                    .filter { it.isPlayableProgressiveM4a() }
                    .maxWithOrNull(
                        compareBy<AudioStream> {
                            it.audioTrackType == null || it.audioTrackType == AudioTrackType.ORIGINAL
                        }
                            .thenBy { it.averageBitrate }
                            .thenBy { it.bitrate }
                    )
            } else {
                null
            }

            val startOffsetSec = clipSpan?.first
                ?: info.startPosition.takeIf { it > 0 }?.toDouble()
                ?: parseTimeParamSec(normalizedUrl)
            val endOffsetSec = clipSpan?.second

            val resolvedHeight = best.height.takeIf { it > 0 } ?: 1080
            val resolvedWidth = best.width.takeIf { it > 0 }
                ?: (((resolvedHeight * 16.0) / 9.0).roundToInt() and 1.inv()).coerceAtLeast(2)

            Log.i(
                LOG_TAG,
                "YOUTUBE_RESOLVED id=${info.id} title=\"${info.name}\" res=${best.resolution} dur=${info.duration}s hasAudio=${!best.isVideoOnly} muxedFallback=${muxedBest != null}"
            )

            ResolvedYouTubeVideo(
                originalUrl = url,
                videoId = info.id,
                title = info.name,
                videoStreamUrl = best.content,
                audioStreamUrl = bestAudio?.content,
                muxedFallbackUrl = muxedBest?.content,
                width = resolvedWidth,
                height = resolvedHeight,
                hasEmbeddedAudio = !best.isVideoOnly,
                durationSec = info.duration.toDouble().coerceAtLeast(0.0),
                startOffsetSec = startOffsetSec,
                endOffsetSec = endOffsetSec,
                resolvedAtElapsedMs = runCatching { SystemClock.elapsedRealtime() }.getOrElse { 0L }
            )
        }

    private fun VideoStream.isPlayableProgressiveAvc(maxHeight: Int): Boolean =
        isUrl &&
            deliveryMethod == DeliveryMethod.PROGRESSIVE_HTTP &&
            format == MediaFormat.MPEG_4 &&
            height in 1..maxHeight &&
            (codec?.lowercase()?.startsWith("avc") ?: true)

    private fun AudioStream.isPlayableProgressiveM4a(): Boolean =
        isUrl &&
            deliveryMethod == DeliveryMethod.PROGRESSIVE_HTTP &&
            format == MediaFormat.M4A &&
            (codec?.lowercase()?.startsWith("mp4a") ?: true)

    /** Resolves a YouTube `/clip/` page to `(parentVideoId, startSec to endSec)`. */
    private fun resolveClip(url: String): Pair<String, Pair<Double, Double>> {
        val request = okhttp3.Request.Builder()
            .url(url)
            .header("User-Agent", USER_AGENT)
            .header("Accept-Language", "en-US")
            .build()
        val html = client.newCall(request).execute().use { resp ->
            if (resp.code == 429) {
                throw ReCaptchaException("reCaptcha challenge requested", url)
            }
            resp.body?.string().orEmpty()
        }
        val startMs = Regex("\"startTimeMs\":\"(\\d+)\"").find(html)?.groupValues?.get(1)?.toLongOrNull()
        val endMs = Regex("\"endTimeMs\":\"(\\d+)\"").find(html)?.groupValues?.get(1)?.toLongOrNull()
        val videoId = Regex("\"videoDetails\":\\{\"videoId\":\"([\\w-]{11})\"")
            .find(html)?.groupValues?.get(1)
            ?: Regex("/vi(?:_webp)?/([\\w-]{11})/").find(html)?.groupValues?.get(1)

        if (videoId == null || startMs == null || endMs == null) {
            throw IllegalStateException("Could not extract clip metadata from YouTube clip link")
        }
        return videoId to (startMs / 1000.0 to endMs / 1000.0)
    }

    private fun parseTimeParamSec(rawUrl: String): Double? {
        val uri = runCatching { URI(rawUrl) }.getOrNull() ?: return null
        val candidates = listOfNotNull(uri.rawQuery, uri.rawFragment)
        for (segment in candidates) {
            for (pair in segment.split('&', ';')) {
                val eq = pair.indexOf('=')
                if (eq <= 0) continue
                val key = pair.substring(0, eq).lowercase()
                if (key != "t" && key != "start") continue
                val rawVal = URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8.name()).trim()
                val parsed = parseYouTubeTimestamp(rawVal)
                if (parsed != null && parsed > 0.0) return parsed
            }
        }
        return null
    }

    private fun parseYouTubeTimestamp(token: String): Double? {
        if (token.isEmpty()) return null
        token.toDoubleOrNull()?.let { sec ->
            return sec.takeIf { it.isFinite() && it > 0.0 }
        }
        val m = TIME_COMPOUND_REGEX.matchEntire(token) ?: return null
        val hours = m.groupValues[1].toDoubleOrNull() ?: 0.0
        val mins = m.groupValues[2].toDoubleOrNull() ?: 0.0
        val secs = m.groupValues[3].toDoubleOrNull() ?: 0.0
        val total = hours * 3600.0 + mins * 60.0 + secs
        return total.takeIf { it.isFinite() && it > 0.0 }
    }

    private class OkHttpDownloader(private val client: OkHttpClient) : Downloader() {
        override fun execute(request: Request): Response {
            val method = request.httpMethod()
            val rawBytes = request.dataToSend()
            val body = rawBytes?.toRequestBody()
                ?: if (method.equals("POST", ignoreCase = true) || method.equals("PUT", ignoreCase = true)) {
                    ByteArray(0).toRequestBody()
                } else {
                    null
                }

            val builder = okhttp3.Request.Builder()
                .method(method, body)
                .url(request.url())
                .header("User-Agent", USER_AGENT)
            for ((name, values) in request.headers()) {
                builder.removeHeader(name)
                values.forEach { builder.addHeader(name, it) }
            }
            client.newCall(builder.build()).execute().use { resp ->
                if (resp.code == 429) {
                    throw ReCaptchaException("reCaptcha challenge requested", request.url())
                }
                return Response(
                    resp.code,
                    resp.message,
                    resp.headers.toMultimap(),
                    resp.body?.string(),
                    resp.request.url.toString()
                )
            }
        }
    }
}
