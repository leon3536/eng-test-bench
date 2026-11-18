package com.engtestbench.hockeynano.video

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.util.Log
import com.engtestbench.hockeynano.core.model.TeamId
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.Locale
import kotlin.math.max
import kotlin.math.roundToLong

@Serializable
data class ExportedClipInfo(
    val id: String,
    val filePath: String,
    val playerNumber: String,
    val team: TeamId,
    val startSec: Double,
    val endSec: Double,
    val durationSec: Double,
    val sizeBytes: Long,
    val burnedMarks: Boolean,
    val nanoSummary: String? = null,
    val createdAtMs: Long
)

internal object MediaSourceHelper {
    private const val LOG_TAG = "HockeyNanoBench"

    fun openForExtractor(
        context: Context,
        sourcePathOrUri: String,
        extractor: MediaExtractor
    ): Closeable {
        val trimmed = sourcePathOrUri.trim()
        return when {
            trimmed.startsWith("android.resource://") -> {
                val uri = Uri.parse(trimmed)
                val afd = openResourceAfd(context, uri)
                if (afd.length >= 0L) {
                    extractor.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
                } else {
                    extractor.setDataSource(afd.fileDescriptor)
                }
                Closeable { runCatching { afd.close() } }
            }
            trimmed.startsWith("content://") -> {
                val uri = Uri.parse(trimmed)
                val pfd: ParcelFileDescriptor = context.contentResolver.openFileDescriptor(uri, "r")
                    ?: error("Unable to open content URI: $trimmed")
                extractor.setDataSource(pfd.fileDescriptor)
                Closeable { runCatching { pfd.close() } }
            }
            trimmed.startsWith("file://") -> {
                val path = Uri.parse(trimmed).path ?: trimmed.removePrefix("file://")
                extractor.setDataSource(path)
                Closeable {}
            }
            trimmed.startsWith("http://") || trimmed.startsWith("https://") -> {
                extractor.setDataSource(trimmed, emptyMap())
                Closeable {}
            }
            else -> {
                extractor.setDataSource(trimmed)
                Closeable {}
            }
        }
    }

    fun openForRetriever(
        context: Context,
        sourcePathOrUri: String,
        retriever: MediaMetadataRetriever
    ): Closeable {
        val trimmed = sourcePathOrUri.trim()
        return when {
            trimmed.startsWith("android.resource://") -> {
                val uri = Uri.parse(trimmed)
                val afd = openResourceAfd(context, uri)
                if (afd.length >= 0L) {
                    retriever.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
                } else {
                    retriever.setDataSource(afd.fileDescriptor)
                }
                Closeable { runCatching { afd.close() } }
            }
            trimmed.startsWith("content://") -> {
                val uri = Uri.parse(trimmed)
                val pfd = context.contentResolver.openFileDescriptor(uri, "r")
                    ?: error("Unable to open content URI for retriever: $trimmed")
                retriever.setDataSource(pfd.fileDescriptor)
                Closeable { runCatching { pfd.close() } }
            }
            trimmed.startsWith("file://") -> {
                val path = Uri.parse(trimmed).path ?: trimmed.removePrefix("file://")
                retriever.setDataSource(path)
                Closeable {}
            }
            trimmed.startsWith("http://") || trimmed.startsWith("https://") -> {
                retriever.setDataSource(trimmed, emptyMap())
                Closeable {}
            }
            else -> {
                retriever.setDataSource(trimmed)
                Closeable {}
            }
        }
    }

    private fun openResourceAfd(context: Context, uri: Uri): AssetFileDescriptor {
        runCatching { context.contentResolver.openAssetFileDescriptor(uri, "r") }
            .getOrNull()
            ?.let { return it }

        val segments = uri.pathSegments
        if (segments.size >= 2) {
            val type = segments[0]
            val name = segments[1].substringBeforeLast('.')
            val pkg = uri.authority ?: context.packageName
            val resId = context.resources.getIdentifier(name, type, pkg)
            if (resId != 0) {
                return context.resources.openRawResourceFd(resId)
            }
        } else if (segments.size == 1) {
            val resId = segments[0].toIntOrNull()
            if (resId != null && resId != 0) {
                return context.resources.openRawResourceFd(resId)
            }
        }
        error("Unable to resolve android.resource URI: $uri")
    }

    fun moveAtomically(tmpFile: File, outputFile: File) {
        outputFile.parentFile?.mkdirs()
        if (outputFile.exists()) {
            outputFile.delete()
        }
        val moved = runCatching {
            Files.move(
                tmpFile.toPath(),
                outputFile.toPath(),
                StandardCopyOption.REPLACE_EXISTING
            )
            true
        }.getOrDefault(false)
        if (!moved && !tmpFile.renameTo(outputFile)) {
            tmpFile.copyTo(outputFile, overwrite = true)
            tmpFile.delete()
        }
    }

    fun formatMmSs(sec: Double): String {
        val total = sec.coerceAtLeast(0.0).roundToLong()
        val m = total / 60L
        val s = total % 60L
        return "%02d:%02d".format(Locale.US, m, s)
    }

    fun logError(msg: String, tr: Throwable) {
        Log.e(LOG_TAG, msg, tr)
    }
}

object FastMp4ClipSplitter {
    private const val LOG_TAG = "HockeyNanoBench"
    private const val DEFAULT_BUFFER_BYTES = 2 * 1024 * 1024

    suspend fun split30sClip(
        context: Context,
        sourcePathOrUri: String,
        startSec: Double,
        durationSec: Double = 30.0,
        playerNumber: String,
        team: TeamId,
        outputFile: File,
        nanoSummary: String? = null
    ): Result<ExportedClipInfo> = withContext(Dispatchers.IO) {
        try {
            val safeStartSec = startSec.coerceAtLeast(0.0)
            val safeTargetDurSec = durationSec.coerceAtLeast(0.5)
            val startUs = (safeStartSec * 1_000_000.0).roundToLong()
            val targetDurUs = (safeTargetDurSec * 1_000_000.0).roundToLong()

            outputFile.parentFile?.mkdirs()
            val tmpFile = File(outputFile.parentFile, "${outputFile.name}.${System.nanoTime()}.tmp")

            val extractor = MediaExtractor()
            var muxer: MediaMuxer? = null
            var muxerStarted = false

            try {
                MediaSourceHelper.openForExtractor(context, sourcePathOrUri, extractor).use {
                    var srcVideoTrack = -1
                    var srcAudioTrack = -1
                    var videoFormat: MediaFormat? = null
                    var audioFormat: MediaFormat? = null

                    for (i in 0 until extractor.trackCount) {
                        val format = extractor.getTrackFormat(i)
                        val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
                        if (srcVideoTrack < 0 && mime.startsWith("video/")) {
                            srcVideoTrack = i
                            videoFormat = format
                        } else if (srcAudioTrack < 0 && mime.startsWith("audio/")) {
                            srcAudioTrack = i
                            audioFormat = format
                        }
                    }

                    require(srcVideoTrack >= 0 && videoFormat != null) {
                        "No video track found in $sourcePathOrUri"
                    }

                    extractor.selectTrack(srcVideoTrack)
                    if (srcAudioTrack >= 0) {
                        extractor.selectTrack(srcAudioTrack)
                    }

                    val m = MediaMuxer(tmpFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
                    muxer = m

                    if (videoFormat.containsKey(MediaFormat.KEY_ROTATION)) {
                        runCatching {
                            m.setOrientationHint(videoFormat.getInteger(MediaFormat.KEY_ROTATION))
                        }
                    }

                    val dstVideoTrack = m.addTrack(videoFormat)
                    val dstAudioTrack = if (audioFormat != null) {
                        runCatching { m.addTrack(audioFormat) }.getOrDefault(-1)
                    } else {
                        -1
                    }
                    if (dstAudioTrack < 0 && srcAudioTrack >= 0) {
                        extractor.unselectTrack(srcAudioTrack)
                    }

                    val videoMaxBuf = if (videoFormat.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)) {
                        videoFormat.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE)
                    } else {
                        DEFAULT_BUFFER_BYTES
                    }
                    val audioMaxBuf = if (audioFormat != null && audioFormat.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)) {
                        audioFormat.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE)
                    } else {
                        256 * 1024
                    }
                    val buffer = ByteBuffer.allocateDirect(max(DEFAULT_BUFFER_BYTES, max(videoMaxBuf, audioMaxBuf)))
                    val bufferInfo = MediaCodec.BufferInfo()

                    extractor.seekTo(startUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
                    m.start()
                    muxerStarted = true

                    var seenFirstVideo = false
                    var pts0 = 0L
                    var endUs = Long.MAX_VALUE
                    var maxRelVideoPtsUs = 0L
                    var lastAudioRelPtsUs = -1L
                    var videoFramesWritten = 0
                    var videoPastEndCount = 0

                    while (true) {
                        coroutineContext.ensureActive()
                        val trackIdx = extractor.sampleTrackIndex
                        if (trackIdx < 0) break

                        val sampleTime = extractor.sampleTime
                        if (sampleTime < 0L) {
                            extractor.advance()
                            continue
                        }

                        if (!seenFirstVideo) {
                            // Align baseline PTS to the first video keyframe after seekTo
                            if (trackIdx != srcVideoTrack) {
                                extractor.advance()
                                continue
                            }
                            pts0 = sampleTime
                            endUs = pts0 + targetDurUs
                            seenFirstVideo = true
                        }

                        if (sampleTime < pts0) {
                            extractor.advance()
                            continue
                        }
                        if (sampleTime > endUs + 500_000L) {
                            break
                        }
                        if (sampleTime > endUs) {
                            if (trackIdx == srcVideoTrack) {
                                videoPastEndCount++
                                if (videoPastEndCount >= 6) break
                            }
                            extractor.advance()
                            continue
                        }

                        val dstTrack = when (trackIdx) {
                            srcVideoTrack -> dstVideoTrack
                            srcAudioTrack -> dstAudioTrack
                            else -> -1
                        }
                        if (dstTrack < 0) {
                            extractor.advance()
                            continue
                        }

                        buffer.clear()
                        val sampleSize = extractor.readSampleData(buffer, 0)
                        if (sampleSize < 0) break

                        val relPtsUs = sampleTime - pts0
                        if (relPtsUs < 0L) {
                            extractor.advance()
                            continue
                        }

                        if (dstTrack == dstAudioTrack) {
                            if (relPtsUs <= lastAudioRelPtsUs) {
                                extractor.advance()
                                continue
                            }
                            lastAudioRelPtsUs = relPtsUs
                        } else {
                            if (relPtsUs > maxRelVideoPtsUs) {
                                maxRelVideoPtsUs = relPtsUs
                            }
                            videoFramesWritten++
                        }

                        val flags = if ((extractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC) != 0) {
                            MediaCodec.BUFFER_FLAG_KEY_FRAME
                        } else {
                            0
                        }

                        buffer.position(0)
                        buffer.limit(sampleSize)
                        bufferInfo.set(0, sampleSize, relPtsUs, flags)
                        m.writeSampleData(dstTrack, buffer, bufferInfo)
                        extractor.advance()
                    }

                    require(videoFramesWritten > 0) {
                        "No video frames extracted in range [${safeStartSec}s, +${safeTargetDurSec}s]"
                    }

                    m.stop()
                    muxerStarted = false
                    m.release()
                    muxer = null

                    MediaSourceHelper.moveAtomically(tmpFile, outputFile)

                    val actualStartSec = pts0 / 1_000_000.0
                    val actualDurationSec = (maxRelVideoPtsUs / 1_000_000.0).coerceAtLeast(0.033)
                    val actualEndSec = actualStartSec + actualDurationSec
                    val sizeBytes = outputFile.length()
                    val cleanPlayer = playerNumber.trim().trimStart('#').ifBlank { "all" }

                    Log.i(
                        LOG_TAG,
                        "CLIP_EXPORTED path=${outputFile.absolutePath} dur=${"%.2f".format(Locale.US, actualDurationSec)}s size=$sizeBytes"
                    )

                    Result.success(
                        ExportedClipInfo(
                            id = "fast_${cleanPlayer}_${(actualStartSec * 10).roundToLong()}_${System.currentTimeMillis()}",
                            filePath = outputFile.absolutePath,
                            playerNumber = cleanPlayer,
                            team = team,
                            startSec = actualStartSec,
                            endSec = actualEndSec,
                            durationSec = actualDurationSec,
                            sizeBytes = sizeBytes,
                            burnedMarks = false,
                            nanoSummary = nanoSummary,
                            createdAtMs = System.currentTimeMillis()
                        )
                    )
                }
            } finally {
                if (muxerStarted) {
                    runCatching { muxer?.stop() }
                }
                runCatching { muxer?.release() }
                runCatching { extractor.release() }
                if (tmpFile.exists()) {
                    runCatching { tmpFile.delete() }
                }
            }
        } catch (ce: CancellationException) {
            throw ce
        } catch (t: Throwable) {
            MediaSourceHelper.logError("FastMp4ClipSplitter failed for $sourcePathOrUri", t)
            Result.failure(t)
        }
    }
}
