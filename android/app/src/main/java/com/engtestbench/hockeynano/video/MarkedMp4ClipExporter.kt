package com.engtestbench.hockeynano.video

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.graphics.Typeface
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import com.engtestbench.hockeynano.core.model.FrameMarkSample
import com.engtestbench.hockeynano.core.model.RectNorm
import com.engtestbench.hockeynano.core.model.TeamId
import com.engtestbench.hockeynano.core.model.TeamProfile
import com.engtestbench.hockeynano.core.model.TrackedPlayer
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.coroutines.coroutineContext
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.roundToLong

object MarkedMp4ClipExporter {
    private const val LOG_TAG = "HockeyNanoBench"
    private const val TARGET_FPS = 12
    private const val BITRATE_BPS = 4_000_000
    private const val I_FRAME_INTERVAL_SEC = 1
    private const val MAX_EXPORT_WIDTH = 1280
    private const val FRAME_INTERVAL_US = 1_000_000L / TARGET_FPS
    private const val MAX_SAMPLE_GAP_SEC = 1.5

    private const val COLOR_TEAM_A = 0xFF00E5FF.toInt()
    private const val COLOR_TEAM_B = 0xFFFF6D00.toInt()
    private const val COLOR_REFEREE = 0xFFB0BEC5.toInt()
    private const val COLOR_UNKNOWN = 0xFF90A4AE.toInt()
    private const val COLOR_TARGET_GOLD = 0xFFFFD600.toInt()
    private const val COLOR_HUD_BG = 0xD6080B09.toInt()
    private const val COLOR_BENCH_GREEN = 0xFF4CBB17.toInt()

    suspend fun exportMarked30sClip(
        context: Context,
        sourcePathOrUri: String,
        startSec: Double,
        durationSec: Double = 30.0,
        samples: List<FrameMarkSample>,
        teams: Map<TeamId, TeamProfile>,
        highlightPlayerNumber: String?,
        teamId: TeamId,
        outputFile: File,
        nanoSummary: String? = null,
        onProgress: (Float) -> Unit = {},
        audioSourcePathOrUri: String? = null
    ): Result<ExportedClipInfo> {
        val executor = Executors.newSingleThreadExecutor { r ->
            Thread(r, "MarkedMp4Exporter").apply { isDaemon = true }
        }
        val singleDispatcher = executor.asCoroutineDispatcher()
        return try {
            withContext(singleDispatcher) {
                try {
                    val safeStartSec = startSec.coerceAtLeast(0.0)
                    val safeTargetDurSec = durationSec.coerceAtLeast(0.5)
                    val sortedSamples = samples
                        .asSequence()
                        .filter { it.timestampSec in (safeStartSec - 2.0)..(safeStartSec + safeTargetDurSec + 2.0) }
                        .sortedBy { it.timestampSec }
                        .toList()
                        .ifEmpty { samples.sortedBy { it.timestampSec } }

                    outputFile.parentFile?.mkdirs()
                    val tmpFile = File(outputFile.parentFile, "${outputFile.name}.${System.nanoTime()}.tmp")

                    val exportStats = try {
                        exportViaMediaCodecGl(
                            context = context,
                            sourcePathOrUri = sourcePathOrUri,
                            audioSourcePathOrUri = audioSourcePathOrUri,
                            startSec = safeStartSec,
                            durationSec = safeTargetDurSec,
                            samples = sortedSamples,
                            teams = teams,
                            highlightPlayerNumber = highlightPlayerNumber,
                            teamId = teamId,
                            tmpOutputFile = tmpFile,
                            nanoSummary = nanoSummary,
                            onProgress = onProgress
                        )
                    } catch (ce: CancellationException) {
                        if (tmpFile.exists()) tmpFile.delete()
                        throw ce
                    } catch (glErr: Throwable) {
                        Log.w(LOG_TAG, "GL MediaCodec decode fallback to retriever: ${glErr.message}")
                        if (tmpFile.exists()) tmpFile.delete()
                        exportViaRetrieverFallback(
                            context = context,
                            sourcePathOrUri = sourcePathOrUri,
                            audioSourcePathOrUri = audioSourcePathOrUri,
                            startSec = safeStartSec,
                            durationSec = safeTargetDurSec,
                            samples = sortedSamples,
                            teams = teams,
                            highlightPlayerNumber = highlightPlayerNumber,
                            teamId = teamId,
                            tmpOutputFile = tmpFile,
                            nanoSummary = nanoSummary,
                            onProgress = onProgress
                        )
                    }

                    MediaSourceHelper.moveAtomically(tmpFile, outputFile)
                    onProgress(1.0f)

                    val sizeBytes = outputFile.length()
                    val cleanPlayer = highlightPlayerNumber
                        ?.trim()
                        ?.trimStart('#')
                        ?.takeIf { it.isNotEmpty() }
                        ?: "all"

                    Log.i(
                        LOG_TAG,
                        "MARKED_CLIP_EXPORTED path=${outputFile.absolutePath} frames=${exportStats.frameCount} size=$sizeBytes"
                    )

                    Result.success(
                        ExportedClipInfo(
                            id = "marked_${cleanPlayer}_${(safeStartSec * 10).roundToLong()}_${System.currentTimeMillis()}",
                            filePath = outputFile.absolutePath,
                            playerNumber = cleanPlayer,
                            team = teamId,
                            startSec = safeStartSec,
                            endSec = safeStartSec + exportStats.actualDurationSec,
                            durationSec = exportStats.actualDurationSec,
                            sizeBytes = sizeBytes,
                            burnedMarks = true,
                            nanoSummary = nanoSummary,
                            createdAtMs = System.currentTimeMillis()
                        )
                    )
                } catch (ce: CancellationException) {
                    throw ce
                } catch (t: Throwable) {
                    MediaSourceHelper.logError("MarkedMp4ClipExporter failed for $sourcePathOrUri", t)
                    Result.failure(t)
                }
            }
        } finally {
            runCatching { singleDispatcher.close() }
            runCatching { executor.shutdownNow() }
        }
    }

    private data class ExportStats(
        val frameCount: Int,
        val actualDurationSec: Double
    )

    private fun computeTargetResolution(srcW: Int, srcH: Int, rotationDeg: Int): Pair<Int, Int> {
        val dispW = if (rotationDeg == 90 || rotationDeg == 270) srcH else srcW
        val dispH = if (rotationDeg == 90 || rotationDeg == 270) srcW else srcH
        val scale = if (dispW > MAX_EXPORT_WIDTH) MAX_EXPORT_WIDTH.toDouble() / dispW else 1.0
        var outW = (dispW * scale).roundToInt().coerceAtLeast(32)
        var outH = (dispH * scale).roundToInt().coerceAtLeast(32)
        outW = (outW / 2) * 2
        outH = (outH / 2) * 2
        return outW to outH
    }

    private suspend fun exportViaMediaCodecGl(
        context: Context,
        sourcePathOrUri: String,
        audioSourcePathOrUri: String?,
        startSec: Double,
        durationSec: Double,
        samples: List<FrameMarkSample>,
        teams: Map<TeamId, TeamProfile>,
        highlightPlayerNumber: String?,
        teamId: TeamId,
        tmpOutputFile: File,
        nanoSummary: String?,
        onProgress: (Float) -> Unit
    ): ExportStats {
        val startUs = (startSec * 1_000_000.0).roundToLong()
        val durationUs = (durationSec * 1_000_000.0).roundToLong()
        val endUs = startUs + durationUs

        val extractor = MediaExtractor()
        var decoder: MediaCodec? = null
        var glDecoderSurface: GlSurfaceDecoder? = null
        var encoderSession: VideoMuxEncoderSession? = null
        var frameBitmap: Bitmap? = null
        var argbPixels: IntArray? = null

        return MediaSourceHelper.openForExtractor(context, sourcePathOrUri, extractor).use {
            try {
                var srcVideoTrack = -1
                var srcAudioTrack = -1
                var videoFormat: MediaFormat? = null
                var audioFormat: MediaFormat? = null

                for (i in 0 until extractor.trackCount) {
                    val fmt = extractor.getTrackFormat(i)
                    val mime = fmt.getString(MediaFormat.KEY_MIME) ?: continue
                    if (srcVideoTrack < 0 && mime.startsWith("video/")) {
                        srcVideoTrack = i
                        videoFormat = fmt
                    } else if (srcAudioTrack < 0 && mime.startsWith("audio/")) {
                        srcAudioTrack = i
                        audioFormat = fmt
                    }
                }
                require(srcVideoTrack >= 0 && videoFormat != null) {
                    "No video track in $sourcePathOrUri"
                }

                var resolvedAudioSource = sourcePathOrUri
                if (srcAudioTrack < 0 && !audioSourcePathOrUri.isNullOrBlank()) {
                    MediaSourceHelper.probeFirstAudioTrack(
                        context = context,
                        sourcePathOrUri = audioSourcePathOrUri,
                        verifySeekUs = startUs
                    )?.let { (idx, fmt) ->
                        srcAudioTrack = idx
                        audioFormat = fmt
                        resolvedAudioSource = audioSourcePathOrUri
                    }
                }

                val mime = videoFormat.getString(MediaFormat.KEY_MIME)!!
                val srcW = videoFormat.getInteger(MediaFormat.KEY_WIDTH)
                val srcH = videoFormat.getInteger(MediaFormat.KEY_HEIGHT)
                val rotation = if (videoFormat.containsKey(MediaFormat.KEY_ROTATION)) {
                    videoFormat.getInteger(MediaFormat.KEY_ROTATION)
                } else {
                    0
                }
                val (outW, outH) = computeTargetResolution(srcW, srcH, rotation)

                extractor.selectTrack(srcVideoTrack)
                extractor.seekTo(startUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)

                val glSurf = GlSurfaceDecoder(outW, outH)
                glDecoderSurface = glSurf

                val dec = MediaCodec.createDecoderByType(mime)
                decoder = dec
                dec.configure(videoFormat, glSurf.surface, null, 0)
                dec.start()

                val encSession = VideoMuxEncoderSession(
                    outputFile = tmpOutputFile,
                    width = outW,
                    height = outH,
                    audioFormat = audioFormat
                )
                encoderSession = encSession

                val bmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
                frameBitmap = bmp
                val canvas = Canvas(bmp)
                val annotator = FrameAnnotator()
                val argb = IntArray(outW * outH)
                argbPixels = argb

                val decInfo = MediaCodec.BufferInfo()
                var inputDone = false
                var decoderDone = false
                var reachedTargetEnd = false
                var nextTargetPtsUs = startUs
                var encodedCount = 0
                var lastEncPtsUs = -1L
                var lastReportedProgress = -1f

                while (!decoderDone) {
                    coroutineContext.ensureActive()

                    if (!inputDone) {
                        val inIdx = dec.dequeueInputBuffer(10_000L)
                        if (inIdx >= 0) {
                            if (reachedTargetEnd) {
                                dec.queueInputBuffer(inIdx, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                val inBuf = dec.getInputBuffer(inIdx)!!
                                inBuf.clear()
                                val sampleSize = extractor.readSampleData(inBuf, 0)
                                if (sampleSize < 0) {
                                    dec.queueInputBuffer(inIdx, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                    inputDone = true
                                } else {
                                    val pts = extractor.sampleTime
                                    dec.queueInputBuffer(inIdx, 0, sampleSize, pts, extractor.sampleFlags)
                                    extractor.advance()
                                    if (pts > endUs + 500_000L) {
                                        reachedTargetEnd = true
                                    }
                                }
                            }
                        }
                    }

                    val outIdx = dec.dequeueOutputBuffer(decInfo, 10_000L)
                    if (outIdx >= 0) {
                        val isEos = (decInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                        val ptsUs = decInfo.presentationTimeUs
                        val doRender = decInfo.size > 0 &&
                            !reachedTargetEnd &&
                            ptsUs >= startUs &&
                            ptsUs <= endUs &&
                            ptsUs >= nextTargetPtsUs

                        if (ptsUs > endUs) {
                            reachedTargetEnd = true
                        }

                        dec.releaseOutputBuffer(outIdx, doRender)

                        if (doRender && glSurf.awaitAndReadFrame(bmp)) {
                            val tSec = ptsUs / 1_000_000.0
                            annotator.drawFrameOverlay(
                                canvas = canvas,
                                width = outW,
                                height = outH,
                                timestampSec = tSec,
                                samples = samples,
                                teams = teams,
                                highlightPlayerNumber = highlightPlayerNumber,
                                teamId = teamId,
                                nanoSummary = nanoSummary
                            )
                            bmp.getPixels(argb, 0, outW, 0, 0, outW, outH)

                            val rawRelUs = (ptsUs - startUs).coerceAtLeast(0L)
                            val encPtsUs = if (encodedCount == 0) {
                                rawRelUs
                            } else {
                                max(rawRelUs, lastEncPtsUs + 1_000L)
                            }
                            lastEncPtsUs = encPtsUs

                            encSession.encodeFrame(argb, encPtsUs)
                            encodedCount++

                            while (nextTargetPtsUs <= ptsUs) {
                                nextTargetPtsUs += FRAME_INTERVAL_US
                            }
                            val frac = ((ptsUs - startUs).toFloat() / durationUs.toFloat()).coerceIn(0f, 0.94f)
                            if (frac - lastReportedProgress >= 0.02f) {
                                lastReportedProgress = frac
                                onProgress(frac)
                            }
                        }

                        if (isEos) {
                            decoderDone = true
                        }
                    }
                }

                require(encodedCount > 0) {
                    "No decoded frames in [${startSec}s, +${durationSec}s]"
                }

                val totalClipUs = max(lastEncPtsUs + FRAME_INTERVAL_US, FRAME_INTERVAL_US)
                encSession.finishVideoAndMuxAudio(
                    context = context,
                    sourcePathOrUri = resolvedAudioSource,
                    srcAudioTrack = if (encSession.hasAudioTrack) srcAudioTrack else -1,
                    startUs = startUs,
                    endUs = startUs + totalClipUs
                )
                onProgress(0.99f)

                ExportStats(
                    frameCount = encodedCount,
                    actualDurationSec = (totalClipUs / 1_000_000.0).coerceAtLeast(0.5)
                )
            } finally {
                runCatching { decoder?.stop() }
                runCatching { decoder?.release() }
                runCatching { glDecoderSurface?.release() }
                runCatching { encoderSession?.release() }
                runCatching { frameBitmap?.recycle() }
                runCatching { extractor.release() }
            }
        }
    }

    private suspend fun exportViaRetrieverFallback(
        context: Context,
        sourcePathOrUri: String,
        audioSourcePathOrUri: String?,
        startSec: Double,
        durationSec: Double,
        samples: List<FrameMarkSample>,
        teams: Map<TeamId, TeamProfile>,
        highlightPlayerNumber: String?,
        teamId: TeamId,
        tmpOutputFile: File,
        nanoSummary: String?,
        onProgress: (Float) -> Unit
    ): ExportStats {
        val retriever = MediaMetadataRetriever()
        var encoderSession: VideoMuxEncoderSession? = null
        var workBmp: Bitmap? = null

        return MediaSourceHelper.openForRetriever(context, sourcePathOrUri, retriever).use {
            try {
                val srcW = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
                    ?.toIntOrNull() ?: 1280
                val srcH = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
                    ?.toIntOrNull() ?: 720
                val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                    ?.toIntOrNull() ?: 0
                val totalDurMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                    ?.toLongOrNull() ?: ((startSec + durationSec) * 1000.0).roundToLong()

                val (outW, outH) = computeTargetResolution(srcW, srcH, rotation)

                val extractor = MediaExtractor()
                var srcAudioTrack = -1
                var audioFormat: MediaFormat? = null
                runCatching {
                    MediaSourceHelper.openForExtractor(context, sourcePathOrUri, extractor).use {
                        for (i in 0 until extractor.trackCount) {
                            val fmt = extractor.getTrackFormat(i)
                            if (fmt.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                                srcAudioTrack = i
                                audioFormat = fmt
                                break
                            }
                        }
                    }
                }
                runCatching { extractor.release() }

                val startUs = (startSec * 1_000_000.0).roundToLong()
                val maxEndUs = min(
                    startUs + (durationSec * 1_000_000.0).roundToLong(),
                    totalDurMs * 1_000L
                ).coerceAtLeast(startUs + FRAME_INTERVAL_US)

                var resolvedAudioSource = sourcePathOrUri
                if (srcAudioTrack < 0 && !audioSourcePathOrUri.isNullOrBlank()) {
                    MediaSourceHelper.probeFirstAudioTrack(
                        context = context,
                        sourcePathOrUri = audioSourcePathOrUri,
                        verifySeekUs = startUs
                    )?.let { (idx, fmt) ->
                        srcAudioTrack = idx
                        audioFormat = fmt
                        resolvedAudioSource = audioSourcePathOrUri
                    }
                }

                val encSession = VideoMuxEncoderSession(
                    outputFile = tmpOutputFile,
                    width = outW,
                    height = outH,
                    audioFormat = audioFormat
                )
                encoderSession = encSession

                val bmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
                workBmp = bmp
                val canvas = Canvas(bmp)
                val annotator = FrameAnnotator()
                val argb = IntArray(outW * outH)

                var curUs = startUs
                var encodedCount = 0
                val totalSteps = max(1L, (maxEndUs - startUs) / FRAME_INTERVAL_US)

                while (curUs <= maxEndUs) {
                    coroutineContext.ensureActive()
                    val frame = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                        retriever.getScaledFrameAtTime(
                            curUs,
                            MediaMetadataRetriever.OPTION_CLOSEST,
                            outW,
                            outH
                        )
                    } else {
                        retriever.getFrameAtTime(curUs, MediaMetadataRetriever.OPTION_CLOSEST)
                    }
                    if (frame != null) {
                        val drawFrame = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
                            frame.config == Bitmap.Config.HARDWARE
                        ) {
                            frame.copy(Bitmap.Config.ARGB_8888, false).also { frame.recycle() }
                        } else {
                            frame
                        }
                        val dstRect = RectF(0f, 0f, outW.toFloat(), outH.toFloat())
                        canvas.drawBitmap(drawFrame, null, dstRect, null)
                        if (drawFrame !== bmp) drawFrame.recycle()

                        annotator.drawFrameOverlay(
                            canvas = canvas,
                            width = outW,
                            height = outH,
                            timestampSec = curUs / 1_000_000.0,
                            samples = samples,
                            teams = teams,
                            highlightPlayerNumber = highlightPlayerNumber,
                            teamId = teamId,
                            nanoSummary = nanoSummary
                        )
                        bmp.getPixels(argb, 0, outW, 0, 0, outW, outH)
                        encSession.encodeFrame(argb, curUs - startUs)
                        encodedCount++
                        onProgress((encodedCount.toFloat() / totalSteps.toFloat()).coerceIn(0f, 0.94f))
                    }
                    curUs += FRAME_INTERVAL_US
                }

                require(encodedCount > 0) {
                    "Retriever fallback extracted 0 frames from $sourcePathOrUri"
                }

                val actualDurUs = max(encodedCount * FRAME_INTERVAL_US, FRAME_INTERVAL_US)
                encSession.finishVideoAndMuxAudio(
                    context = context,
                    sourcePathOrUri = resolvedAudioSource,
                    srcAudioTrack = if (encSession.hasAudioTrack) srcAudioTrack else -1,
                    startUs = startUs,
                    endUs = startUs + actualDurUs
                )
                onProgress(0.99f)

                ExportStats(
                    frameCount = encodedCount,
                    actualDurationSec = actualDurUs / 1_000_000.0
                )
            } finally {
                runCatching { encoderSession?.release() }
                runCatching { workBmp?.recycle() }
                runCatching { retriever.release() }
            }
        }
    }

    private class VideoMuxEncoderSession(
        outputFile: File,
        private val width: Int,
        private val height: Int,
        audioFormat: MediaFormat?
    ) {
        private val encoder: MediaCodec
        private val muxer: MediaMuxer
        private val bufferInfo = MediaCodec.BufferInfo()
        private val yRowScratch = ByteArray(width)
        private val pendingAudioFormat: MediaFormat? = audioFormat

        private var muxerVideoTrack = -1
        private var muxerAudioTrack = -1
        private var muxerStarted = false
        private var eosSent = false
        private var eosReceived = false
        private var lastEncodedPtsUs = 0L

        val hasAudioTrack: Boolean get() = pendingAudioFormat != null

        init {
            val encFormat = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
                setInteger(
                    MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
                )
                setInteger(MediaFormat.KEY_BIT_RATE, BITRATE_BPS)
                setInteger(MediaFormat.KEY_FRAME_RATE, TARGET_FPS)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL_SEC)
            }
            encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
            encoder.configure(encFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            encoder.start()
            muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        }

        fun encodeFrame(argb: IntArray, ptsUs: Long) {
            lastEncodedPtsUs = ptsUs
            while (true) {
                drainEncoder(endOfStream = false)
                val inIdx = encoder.dequeueInputBuffer(10_000L)
                if (inIdx >= 0) {
                    val inputImage = encoder.getInputImage(inIdx)
                        ?: error("MediaCodec encoder returned null getInputImage($inIdx)")
                    writeRgbToYuv420Image(argb, width, height, inputImage, yRowScratch)
                    encoder.queueInputBuffer(inIdx, 0, width * height * 3 / 2, ptsUs, 0)
                    drainEncoder(endOfStream = false)
                    return
                }
            }
        }

        fun finishVideoAndMuxAudio(
            context: Context,
            sourcePathOrUri: String,
            srcAudioTrack: Int,
            startUs: Long,
            endUs: Long
        ) {
            if (!eosSent) {
                while (true) {
                    val inIdx = encoder.dequeueInputBuffer(10_000L)
                    if (inIdx >= 0) {
                        encoder.queueInputBuffer(
                            inIdx,
                            0,
                            0,
                            lastEncodedPtsUs + FRAME_INTERVAL_US,
                            MediaCodec.BUFFER_FLAG_END_OF_STREAM
                        )
                        eosSent = true
                        break
                    }
                    drainEncoder(endOfStream = false)
                }
            }

            var idleCycles = 0
            while (!eosReceived && idleCycles < 200) {
                val progressed = drainEncoder(endOfStream = true)
                idleCycles = if (progressed) 0 else idleCycles + 1
            }

            if (muxerStarted && muxerAudioTrack >= 0 && srcAudioTrack >= 0) {
                runCatching {
                    MediaSourceHelper.muxAudioWindow(
                        context = context,
                        audioSourcePathOrUri = sourcePathOrUri,
                        srcAudioTrack = srcAudioTrack,
                        muxer = muxer,
                        dstAudioTrack = muxerAudioTrack,
                        seekUs = startUs,
                        windowStartUs = startUs,
                        windowEndUs = endUs,
                        ptsOriginUs = startUs
                    )
                }.onFailure {
                    Log.w(LOG_TAG, "Skipping audio mux on marked export: ${it.message}")
                }
            }
        }

        private fun drainEncoder(endOfStream: Boolean): Boolean {
            var progressed = false
            val timeoutUs = if (endOfStream) 10_000L else 0L
            while (true) {
                val outIdx = encoder.dequeueOutputBuffer(bufferInfo, timeoutUs)
                when {
                    outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> break
                    outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        if (!muxerStarted) {
                            muxerVideoTrack = muxer.addTrack(encoder.outputFormat)
                            pendingAudioFormat?.let { af ->
                                muxerAudioTrack = runCatching { muxer.addTrack(af) }
                                    .getOrDefault(-1)
                            }
                            muxer.start()
                            muxerStarted = true
                            progressed = true
                        }
                    }
                    outIdx >= 0 -> {
                        progressed = true
                        val outBuf = encoder.getOutputBuffer(outIdx)
                        if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                            bufferInfo.size = 0
                        }
                        if (bufferInfo.size > 0 && muxerStarted && muxerVideoTrack >= 0 && outBuf != null) {
                            outBuf.position(bufferInfo.offset)
                            outBuf.limit(bufferInfo.offset + bufferInfo.size)
                            muxer.writeSampleData(muxerVideoTrack, outBuf, bufferInfo)
                        }
                        val isEos = (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                        encoder.releaseOutputBuffer(outIdx, false)
                        if (isEos) {
                            eosReceived = true
                            break
                        }
                    }
                }
            }
            return progressed
        }

        fun release() {
            runCatching { encoder.stop() }
            runCatching { encoder.release() }
            if (muxerStarted) {
                runCatching { muxer.stop() }
                muxerStarted = false
            }
            runCatching { muxer.release() }
        }

        companion object {
            /**
             * Writes ARGB pixels to MediaCodec YUV_420_888 Image respecting plane.rowStride and plane.pixelStride
             * on planes [0, 1, 2] (critical on Tensor G6 hardware encoder).
             */
            fun writeRgbToYuv420Image(
                argb: IntArray,
                width: Int,
                height: Int,
                image: android.media.Image,
                yRowScratch: ByteArray
            ) {
                val yPlane = image.planes[0]
                val uPlane = image.planes[1]
                val vPlane = image.planes[2]

                val yBuf = yPlane.buffer
                val uBuf = uPlane.buffer
                val vBuf = vPlane.buffer

                val yRowStride = yPlane.rowStride
                val yPixelStride = yPlane.pixelStride
                val uRowStride = uPlane.rowStride
                val uPixelStride = uPlane.pixelStride
                val vRowStride = vPlane.rowStride
                val vPixelStride = vPlane.pixelStride

                if (yPixelStride == 1) {
                    for (y in 0 until height) {
                        val rowBase = y * width
                        for (x in 0 until width) {
                            val c = argb[rowBase + x]
                            val r = (c shr 16) and 0xFF
                            val g = (c shr 8) and 0xFF
                            val b = c and 0xFF
                            val yVal = (((66 * r + 129 * g + 25 * b + 128) shr 8) + 16).coerceIn(0, 255)
                            yRowScratch[x] = yVal.toByte()
                        }
                        yBuf.position(y * yRowStride)
                        yBuf.put(yRowScratch, 0, width)
                    }
                } else {
                    for (y in 0 until height) {
                        val rowBase = y * width
                        val dstRowBase = y * yRowStride
                        for (x in 0 until width) {
                            val c = argb[rowBase + x]
                            val r = (c shr 16) and 0xFF
                            val g = (c shr 8) and 0xFF
                            val b = c and 0xFF
                            val yVal = (((66 * r + 129 * g + 25 * b + 128) shr 8) + 16).coerceIn(0, 255)
                            yBuf.put(dstRowBase + x * yPixelStride, yVal.toByte())
                        }
                    }
                }

                val halfH = height / 2
                val halfW = width / 2
                for (cy in 0 until halfH) {
                    val y0 = cy * 2
                    val y1 = y0 + 1
                    val row0 = y0 * width
                    val row1 = y1 * width
                    val uDstRow = cy * uRowStride
                    val vDstRow = cy * vRowStride

                    for (cx in 0 until halfW) {
                        val x0 = cx * 2
                        val x1 = x0 + 1
                        val c00 = argb[row0 + x0]
                        val c01 = argb[row0 + x1]
                        val c10 = argb[row1 + x0]
                        val c11 = argb[row1 + x1]

                        val rAvg = (((c00 shr 16) and 0xFF) + ((c01 shr 16) and 0xFF) +
                            ((c10 shr 16) and 0xFF) + ((c11 shr 16) and 0xFF)) shr 2
                        val gAvg = (((c00 shr 8) and 0xFF) + ((c01 shr 8) and 0xFF) +
                            ((c10 shr 8) and 0xFF) + ((c11 shr 8) and 0xFF)) shr 2
                        val bAvg = ((c00 and 0xFF) + (c01 and 0xFF) +
                            (c10 and 0xFF) + (c11 and 0xFF)) shr 2

                        val uVal = (((-38 * rAvg - 74 * gAvg + 112 * bAvg + 128) shr 8) + 128).coerceIn(0, 255)
                        val vVal = (((112 * rAvg - 94 * gAvg - 18 * bAvg + 128) shr 8) + 128).coerceIn(0, 255)

                        uBuf.put(uDstRow + cx * uPixelStride, uVal.toByte())
                        vBuf.put(vDstRow + cx * vPixelStride, vVal.toByte())
                    }
                }
            }
        }
    }

    /**
     * Hardware GPU SurfaceTexture + EGL Pbuffer renderer for fast YUV->RGBA decode & downscale.
     */
    private class GlSurfaceDecoder(
        private val width: Int,
        private val height: Int
    ) {
        private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
        private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
        private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
        private var texId = 0
        private var program = 0
        private var posHandle = 0
        private var texHandle = 0
        private var stMatrixHandle = 0

        private val stThread = HandlerThread("HockeyExportST").apply { start() }
        private val stHandler = Handler(stThread.looper)
        private val frameLock = ReentrantLock()
        private val frameCond = frameLock.newCondition()
        private var frameAvailable = false

        private val surfaceTexture: SurfaceTexture
        val surface: Surface

        private val stMatrix = FloatArray(16)
        private val pixelBuffer: ByteBuffer = ByteBuffer.allocateDirect(width * height * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        // Map GL bottom (-1) -> Bitmap top (tex v=1) so glReadPixels directly yields top-down ARGB_8888
        private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(4 * 4 * 4).run {
            order(ByteOrder.nativeOrder())
            asFloatBuffer().apply {
                put(
                    floatArrayOf(
                        -1f, -1f, 0f, 1f,
                        1f, -1f, 1f, 1f,
                        -1f, 1f, 0f, 0f,
                        1f, 1f, 1f, 0f
                    )
                )
                position(0)
            }
        }

        init {
            setupEgl()
            setupGl()
            surfaceTexture = SurfaceTexture(texId).apply {
                setOnFrameAvailableListener({
                    frameLock.withLock {
                        frameAvailable = true
                        frameCond.signalAll()
                    }
                }, stHandler)
            }
            surface = Surface(surfaceTexture)
        }

        private fun setupEgl() {
            eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            require(eglDisplay != EGL14.EGL_NO_DISPLAY) { "eglGetDisplay failed" }
            val version = IntArray(2)
            require(EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) { "eglInitialize failed" }

            val configAttribs = intArrayOf(
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_NONE
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            require(
                EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0) &&
                    numConfigs[0] > 0
            ) { "eglChooseConfig failed" }

            val ctxAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
            eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, ctxAttribs, 0)
            require(eglContext != EGL14.EGL_NO_CONTEXT) { "eglCreateContext failed" }

            val surfAttribs = intArrayOf(
                EGL14.EGL_WIDTH, width,
                EGL14.EGL_HEIGHT, height,
                EGL14.EGL_NONE
            )
            eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, configs[0], surfAttribs, 0)
            require(eglSurface != EGL14.EGL_NO_SURFACE) { "eglCreatePbufferSurface failed" }

            require(EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
                "eglMakeCurrent failed"
            }
        }

        private fun setupGl() {
            val vsSrc = """
                uniform mat4 uSTMatrix;
                attribute vec4 aPosition;
                attribute vec4 aTexCoord;
                varying vec2 vTexCoord;
                void main() {
                    gl_Position = aPosition;
                    vTexCoord = (uSTMatrix * aTexCoord).xy;
                }
            """.trimIndent()

            val fsSrc = """
                #extension GL_OES_EGL_image_external : require
                precision mediump float;
                varying vec2 vTexCoord;
                uniform samplerExternalOES sTexture;
                void main() {
                    gl_FragColor = texture2D(sTexture, vTexCoord);
                }
            """.trimIndent()

            val vs = compileShader(GLES20.GL_VERTEX_SHADER, vsSrc)
            val fs = compileShader(GLES20.GL_FRAGMENT_SHADER, fsSrc)
            program = GLES20.glCreateProgram().also { prog ->
                GLES20.glAttachShader(prog, vs)
                GLES20.glAttachShader(prog, fs)
                GLES20.glLinkProgram(prog)
                val status = IntArray(1)
                GLES20.glGetProgramiv(prog, GLES20.GL_LINK_STATUS, status, 0)
                require(status[0] == GLES20.GL_TRUE) {
                    "GL link failed: ${GLES20.glGetProgramInfoLog(prog)}"
                }
            }
            GLES20.glDeleteShader(vs)
            GLES20.glDeleteShader(fs)

            posHandle = GLES20.glGetAttribLocation(program, "aPosition")
            texHandle = GLES20.glGetAttribLocation(program, "aTexCoord")
            stMatrixHandle = GLES20.glGetUniformLocation(program, "uSTMatrix")

            val textures = IntArray(1)
            GLES20.glGenTextures(1, textures, 0)
            texId = textures[0]
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texId)
            GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR.toFloat())
            GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR.toFloat())
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        }

        private fun compileShader(type: Int, src: String): Int {
            val shader = GLES20.glCreateShader(type)
            GLES20.glShaderSource(shader, src)
            GLES20.glCompileShader(shader)
            val status = IntArray(1)
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
            require(status[0] == GLES20.GL_TRUE) {
                "Shader compile error: ${GLES20.glGetShaderInfoLog(shader)}"
            }
            return shader
        }

        fun awaitAndReadFrame(targetBitmap: Bitmap): Boolean {
            frameLock.withLock {
                val deadlineMs = System.currentTimeMillis() + 1_500L
                while (!frameAvailable) {
                    val remaining = deadlineMs - System.currentTimeMillis()
                    if (remaining <= 0L) return false
                    frameCond.await(remaining, TimeUnit.MILLISECONDS)
                }
                frameAvailable = false
            }

            surfaceTexture.updateTexImage()
            surfaceTexture.getTransformMatrix(stMatrix)

            GLES20.glViewport(0, 0, width, height)
            GLES20.glDisable(GLES20.GL_CULL_FACE)
            GLES20.glDisable(GLES20.GL_DEPTH_TEST)
            GLES20.glClearColor(0f, 0f, 0f, 1f)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

            GLES20.glUseProgram(program)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texId)
            GLES20.glUniformMatrix4fv(stMatrixHandle, 1, false, stMatrix, 0)

            quadVertices.position(0)
            GLES20.glVertexAttribPointer(posHandle, 2, GLES20.GL_FLOAT, false, 16, quadVertices)
            GLES20.glEnableVertexAttribArray(posHandle)

            quadVertices.position(2)
            GLES20.glVertexAttribPointer(texHandle, 2, GLES20.GL_FLOAT, false, 16, quadVertices)
            GLES20.glEnableVertexAttribArray(texHandle)

            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
            GLES20.glFinish()

            pixelBuffer.clear()
            GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixelBuffer)
            pixelBuffer.rewind()
            targetBitmap.copyPixelsFromBuffer(pixelBuffer)
            return true
        }

        fun release() {
            runCatching { surface.release() }
            runCatching { surfaceTexture.release() }
            if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
                runCatching {
                    EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
                }
                if (texId != 0) runCatching { GLES20.glDeleteTextures(1, intArrayOf(texId), 0) }
                if (program != 0) runCatching { GLES20.glDeleteProgram(program) }
                if (eglSurface != EGL14.EGL_NO_SURFACE) runCatching { EGL14.eglDestroySurface(eglDisplay, eglSurface) }
                if (eglContext != EGL14.EGL_NO_CONTEXT) runCatching { EGL14.eglDestroyContext(eglDisplay, eglContext) }
                runCatching { EGL14.eglTerminate(eglDisplay) }
            }
            eglDisplay = EGL14.EGL_NO_DISPLAY
            eglContext = EGL14.EGL_NO_CONTEXT
            eglSurface = EGL14.EGL_NO_SURFACE
            runCatching { stThread.quitSafely() }
        }
    }

    /**
     * Interpolates RectNorms across FrameMarkSamples and renders hockey overlays onto each exported frame.
     */
    private class FrameAnnotator {
        private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
        private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
        }
        private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        }
        private val badgeRects = ArrayList<RectF>(32)
        private val trailPath = Path()

        fun drawFrameOverlay(
            canvas: Canvas,
            width: Int,
            height: Int,
            timestampSec: Double,
            samples: List<FrameMarkSample>,
            teams: Map<TeamId, TeamProfile>,
            highlightPlayerNumber: String?,
            teamId: TeamId,
            nanoSummary: String?
        ) {
            val scale = (width / 1280f).coerceIn(0.6f, 1.5f)
            val w = width.toFloat()
            val h = height.toFloat()

            val normTarget = highlightPlayerNumber
                ?.trim()
                ?.trimStart('#')
                ?.takeIf { it.isNotEmpty() }

            val players = interpolatePlayersAt(samples, timestampSec)
            val sortedPlayers = players.sortedBy { isTargetPlayer(it, normTarget, teamId) }
            badgeRects.clear()

            for (player in sortedPlayers) {
                val isTarget = isTargetPlayer(player, normTarget, teamId)
                val baseColor = teamColor(player.team)
                val box = player.box

                val left = (box.x1 * w).coerceIn(0f, w - 2f)
                val top = (box.y1 * h).coerceIn(0f, h - 2f)
                val right = (box.x2 * w).coerceIn(left + 2f, w)
                val bottom = (box.y2 * h).coerceIn(top + 2f, h)
                val boxRect = RectF(left, top, right, bottom)

                val footX = (box.centerX * w).coerceIn(0f, w)
                val footY = (box.footY * h).coerceIn(0f, h)
                val rx = max((right - left) * 0.56f, 16f * scale)
                val ry = max(rx * 0.34f, 6f * scale)

                if (isTarget && player.trail.isNotEmpty()) {
                    drawMotionTrail(canvas, player.trail, footX, footY, w, h, scale)
                }

                // Skate ring oval at footY
                fillPaint.color = withAlpha(if (isTarget) COLOR_TARGET_GOLD else baseColor, if (isTarget) 54 else 32)
                canvas.drawOval(RectF(footX - rx, footY - ry, footX + rx, footY + ry), fillPaint)

                strokePaint.color = baseColor
                strokePaint.strokeWidth = (if (isTarget) 2.5f else 2.2f) * scale
                canvas.drawOval(RectF(footX - rx, footY - ry, footX + rx, footY + ry), strokePaint)

                if (isTarget) {
                    // Neon Gold double halo ring
                    strokePaint.color = COLOR_TARGET_GOLD
                    strokePaint.strokeWidth = 3.2f * scale
                    canvas.drawOval(
                        RectF(footX - rx * 1.15f, footY - ry * 1.15f, footX + rx * 1.15f, footY + ry * 1.15f),
                        strokePaint
                    )
                    strokePaint.color = withAlpha(COLOR_TARGET_GOLD, 210)
                    strokePaint.strokeWidth = 2.0f * scale
                    canvas.drawOval(
                        RectF(footX - rx * 1.45f, footY - ry * 1.45f, footX + rx * 1.45f, footY + ry * 1.45f),
                        strokePaint
                    )
                }

                // Translucent box + rounded outline
                val corner = 6f * scale
                fillPaint.color = withAlpha(if (isTarget) COLOR_TARGET_GOLD else baseColor, if (isTarget) 34 else 18)
                canvas.drawRoundRect(boxRect, corner, corner, fillPaint)

                strokePaint.color = baseColor
                strokePaint.strokeWidth = (if (isTarget) 2.4f else 1.8f) * scale
                canvas.drawRoundRect(boxRect, corner, corner, strokePaint)

                if (isTarget) {
                    drawCornerTargetBrackets(canvas, boxRect, COLOR_TARGET_GOLD, scale)
                }

                drawPlayerBadge(
                    canvas = canvas,
                    player = player,
                    boxRect = boxRect,
                    baseColor = baseColor,
                    isTarget = isTarget,
                    frameW = w,
                    frameH = h,
                    scale = scale
                )
            }

            drawHudBar(
                canvas = canvas,
                frameW = w,
                frameH = h,
                scale = scale,
                timestampSec = timestampSec,
                teams = teams,
                highlightPlayerNumber = normTarget,
                teamId = teamId,
                nanoSummary = nanoSummary
            )
        }

        private fun drawMotionTrail(
            canvas: Canvas,
            trail: List<Pair<Float, Float>>,
            footX: Float,
            footY: Float,
            w: Float,
            h: Float,
            scale: Float
        ) {
            if (trail.isEmpty()) return
            trailPath.reset()
            val first = trail.first()
            trailPath.moveTo(first.first * w, first.second * h)
            for (i in 1 until trail.size) {
                val pt = trail[i]
                trailPath.lineTo(pt.first * w, pt.second * h)
            }
            trailPath.lineTo(footX, footY)

            strokePaint.color = withAlpha(COLOR_TARGET_GOLD, 180)
            strokePaint.strokeWidth = 3.2f * scale
            canvas.drawPath(trailPath, strokePaint)

            fillPaint.color = COLOR_TARGET_GOLD
            for (i in trail.indices) {
                val alpha = (90 + (150 * (i + 1) / trail.size)).coerceIn(90, 240)
                fillPaint.color = withAlpha(COLOR_TARGET_GOLD, alpha)
                val pt = trail[i]
                canvas.drawCircle(pt.first * w, pt.second * h, 3.0f * scale, fillPaint)
            }
        }

        private fun drawCornerTargetBrackets(
            canvas: Canvas,
            rect: RectF,
            color: Int,
            scale: Float
        ) {
            val arm = min(rect.width(), rect.height()).coerceAtLeast(16f * scale) * 0.28f
            strokePaint.color = color
            strokePaint.strokeWidth = 3.6f * scale

            // Top-left
            canvas.drawLine(rect.left, rect.top + arm, rect.left, rect.top, strokePaint)
            canvas.drawLine(rect.left, rect.top, rect.left + arm, rect.top, strokePaint)
            // Top-right
            canvas.drawLine(rect.right - arm, rect.top, rect.right, rect.top, strokePaint)
            canvas.drawLine(rect.right, rect.top, rect.right, rect.top + arm, strokePaint)
            // Bottom-left
            canvas.drawLine(rect.left, rect.bottom - arm, rect.left, rect.bottom, strokePaint)
            canvas.drawLine(rect.left, rect.bottom, rect.left + arm, rect.bottom, strokePaint)
            // Bottom-right
            canvas.drawLine(rect.right - arm, rect.bottom, rect.right, rect.bottom, strokePaint)
            canvas.drawLine(rect.right, rect.bottom - arm, rect.right, rect.bottom, strokePaint)
        }

        private fun drawPlayerBadge(
            canvas: Canvas,
            player: TrackedPlayer,
            boxRect: RectF,
            baseColor: Int,
            isTarget: Boolean,
            frameW: Float,
            frameH: Float,
            scale: Float
        ) {
            val teamTag = when (player.team) {
                TeamId.TEAM_A -> "A"
                TeamId.TEAM_B -> "B"
                TeamId.REFEREE -> "REF"
                TeamId.UNKNOWN -> "?"
            }
            val numTag = player.jerseyNumber
                ?.trim()
                ?.trimStart('#')
                ?.takeIf { it.isNotEmpty() }
                ?.let { "#$it" }
                ?: "T${player.trackId}"
            val star = if (player.nanoVerified) " ★" else ""
            val label = "[$teamTag] $numTag$star"

            textPaint.textSize = (if (isTarget) 15.5f else 13.5f) * scale
            val textW = textPaint.measureText(label)
            val fm = textPaint.fontMetrics
            val padX = 7f * scale
            val padY = 4f * scale
            val badgeW = textW + padX * 2f
            val badgeH = (fm.descent - fm.ascent) + padY * 2f

            val x = (boxRect.centerX() - badgeW / 2f).coerceIn(4f, frameW - badgeW - 4f)
            var y = (boxRect.top - badgeH - 5f * scale).coerceIn(4f, frameH - badgeH - 4f)

            val candidate = RectF(x, y, x + badgeW, y + badgeH)
            var nudges = 0
            while (nudges < 4 && badgeRects.any { RectF.intersects(it, candidate) }) {
                val nextY = y - (badgeH + 3f * scale)
                if (nextY >= 4f) {
                    y = nextY
                } else {
                    y = (y + badgeH + 3f * scale).coerceAtMost(frameH - badgeH - 4f)
                }
                candidate.set(x, y, x + badgeW, y + badgeH)
                nudges++
            }
            badgeRects.add(RectF(candidate))

            fillPaint.color = COLOR_HUD_BG
            canvas.drawRoundRect(candidate, 5f * scale, 5f * scale, fillPaint)

            strokePaint.color = if (isTarget) COLOR_TARGET_GOLD else baseColor
            strokePaint.strokeWidth = (if (isTarget) 2.2f else 1.5f) * scale
            canvas.drawRoundRect(candidate, 5f * scale, 5f * scale, strokePaint)

            textPaint.color = if (isTarget) COLOR_TARGET_GOLD else Color.WHITE
            canvas.drawText(label, candidate.left + padX, candidate.top + padY - fm.ascent, textPaint)
        }

        private fun drawHudBar(
            canvas: Canvas,
            frameW: Float,
            frameH: Float,
            scale: Float,
            timestampSec: Double,
            teams: Map<TeamId, TeamProfile>,
            highlightPlayerNumber: String?,
            teamId: TeamId,
            nanoSummary: String?
        ) {
            val teamName = teams[teamId]?.displayName?.takeIf { it.isNotBlank() } ?: when (teamId) {
                TeamId.TEAM_A -> "Team A"
                TeamId.TEAM_B -> "Team B"
                TeamId.REFEREE -> "Referee"
                TeamId.UNKNOWN -> "All Teams"
            }
            val mmSs = MediaSourceHelper.formatMmSs(timestampSec)
            val hudText = if (highlightPlayerNumber != null) {
                "TARGET #$highlightPlayerNumber • $teamName • $mmSs"
            } else {
                "$teamName • $mmSs"
            }

            textPaint.textSize = 18f * scale
            val fm = textPaint.fontMetrics
            val padX = 14f * scale
            val padY = 8f * scale
            val hudW = min(textPaint.measureText(hudText) + padX * 2f, frameW - 32f * scale)
            val hudH = (fm.descent - fm.ascent) + padY * 2f
            val hudRect = RectF(16f * scale, 14f * scale, 16f * scale + hudW, 14f * scale + hudH)

            fillPaint.color = COLOR_HUD_BG
            canvas.drawRoundRect(hudRect, 8f * scale, 8f * scale, fillPaint)

            strokePaint.color = if (highlightPlayerNumber != null) COLOR_TARGET_GOLD else COLOR_BENCH_GREEN
            strokePaint.strokeWidth = 2.0f * scale
            canvas.drawRoundRect(hudRect, 8f * scale, 8f * scale, strokePaint)

            textPaint.color = if (highlightPlayerNumber != null) COLOR_TARGET_GOLD else Color.WHITE
            canvas.drawText(hudText, hudRect.left + padX, hudRect.top + padY - fm.ascent, textPaint)

            val cleanSummary = nanoSummary?.trim()?.replace('\n', ' ')?.takeIf { it.isNotEmpty() }
            if (cleanSummary != null) {
                val clipped = if (cleanSummary.length > 84) cleanSummary.take(81) + "..." else cleanSummary
                val note = "NANO: $clipped"
                textPaint.textSize = 14f * scale
                val sfm = textPaint.fontMetrics
                val sPadX = 12f * scale
                val sPadY = 6f * scale
                val sW = min(textPaint.measureText(note) + sPadX * 2f, frameW - 32f * scale)
                val sH = (sfm.descent - sfm.ascent) + sPadY * 2f
                val sTop = frameH - sH - 14f * scale
                val sRect = RectF(16f * scale, sTop, 16f * scale + sW, sTop + sH)

                fillPaint.color = withAlpha(0x080B09, 200)
                canvas.drawRoundRect(sRect, 6f * scale, 6f * scale, fillPaint)
                strokePaint.color = withAlpha(COLOR_BENCH_GREEN, 180)
                strokePaint.strokeWidth = 1.4f * scale
                canvas.drawRoundRect(sRect, 6f * scale, 6f * scale, strokePaint)

                textPaint.color = Color.WHITE
                canvas.drawText(note, sRect.left + sPadX, sRect.top + sPadY - sfm.ascent, textPaint)
            }
        }

        private fun isTargetPlayer(
            player: TrackedPlayer,
            normTarget: String?,
            targetTeam: TeamId
        ): Boolean {
            if (normTarget == null) return false
            val pNum = player.jerseyNumber?.trim()?.trimStart('#') ?: return false
            if (pNum != normTarget) return false
            return targetTeam == TeamId.UNKNOWN || player.team == targetTeam || player.team == TeamId.UNKNOWN
        }

        private fun teamColor(team: TeamId): Int = when (team) {
            TeamId.TEAM_A -> COLOR_TEAM_A
            TeamId.TEAM_B -> COLOR_TEAM_B
            TeamId.REFEREE -> COLOR_REFEREE
            TeamId.UNKNOWN -> COLOR_UNKNOWN
        }

        private fun withAlpha(color: Int, alpha: Int): Int =
            (color and 0x00FFFFFF) or ((alpha.coerceIn(0, 255)) shl 24)

        companion object {
            fun interpolatePlayersAt(
                samples: List<FrameMarkSample>,
                timestampSec: Double
            ): List<TrackedPlayer> {
                if (samples.isEmpty()) return emptyList()

                var lo = 0
                var hi = samples.lastIndex
                while (lo <= hi) {
                    val mid = (lo + hi) ushr 1
                    val tMid = samples[mid].timestampSec
                    when {
                        tMid < timestampSec -> lo = mid + 1
                        tMid > timestampSec -> hi = mid - 1
                        else -> return samples[mid].players
                    }
                }

                val prev = samples.getOrNull(hi)
                val next = samples.getOrNull(lo)

                if (prev != null && next != null) {
                    val gap = next.timestampSec - prev.timestampSec
                    if (gap in 1e-6..MAX_SAMPLE_GAP_SEC) {
                        val alpha = ((timestampSec - prev.timestampSec) / gap).toFloat().coerceIn(0f, 1f)
                        val prevMap = prev.players.associateBy { it.trackId }
                        val nextMap = next.players.associateBy { it.trackId }
                        val merged = ArrayList<TrackedPlayer>(max(prev.players.size, next.players.size))
                        val allowUnmatchedCarryover = abs(prev.players.size - next.players.size) <= 3

                        for (p in prev.players) {
                            val q = nextMap[p.trackId]
                            if (q != null) {
                                merged.add(
                                    TrackedPlayer(
                                        trackId = p.trackId,
                                        box = RectNorm(
                                            x1 = lerp(p.box.x1, q.box.x1, alpha),
                                            y1 = lerp(p.box.y1, q.box.y1, alpha),
                                            x2 = lerp(p.box.x2, q.box.x2, alpha),
                                            y2 = lerp(p.box.y2, q.box.y2, alpha)
                                        ),
                                        team = when {
                                            alpha >= 0.5f && q.team != TeamId.UNKNOWN -> q.team
                                            p.team != TeamId.UNKNOWN -> p.team
                                            else -> q.team
                                        },
                                        jerseyNumber = when {
                                            alpha >= 0.5f && !q.jerseyNumber.isNullOrBlank() -> q.jerseyNumber
                                            !p.jerseyNumber.isNullOrBlank() -> p.jerseyNumber
                                            else -> q.jerseyNumber
                                        },
                                        numberConfidence = max(p.numberConfidence, q.numberConfidence),
                                        nanoVerified = p.nanoVerified || q.nanoVerified,
                                        roleHint = if (alpha >= 0.5f) q.roleHint ?: p.roleHint else p.roleHint ?: q.roleHint,
                                        chestRgb = if (alpha >= 0.5f) q.chestRgb else p.chestRgb,
                                        speedNormPerSec = lerp(p.speedNormPerSec, q.speedNormPerSec, alpha),
                                        trail = if (alpha >= 0.5f && q.trail.isNotEmpty()) q.trail else p.trail
                                    )
                                )
                            } else if (allowUnmatchedCarryover && (timestampSec - prev.timestampSec) <= min(0.12, gap * 0.5)) {
                                merged.add(p)
                            }
                        }

                        for (q in next.players) {
                            if (q.trackId !in prevMap && allowUnmatchedCarryover && (next.timestampSec - timestampSec) <= min(0.12, gap * 0.5)) {
                                merged.add(q)
                            }
                        }
                        return merged
                    }
                }

                val prevDt = if (prev != null) timestampSec - prev.timestampSec else Double.MAX_VALUE
                val nextDt = if (next != null) next.timestampSec - timestampSec else Double.MAX_VALUE
                return when {
                    prev != null && prevDt <= nextDt && prevDt <= MAX_SAMPLE_GAP_SEC -> prev.players
                    next != null && nextDt <= MAX_SAMPLE_GAP_SEC -> next.players
                    else -> emptyList()
                }
            }

            private fun lerp(a: Float, b: Float, t: Float): Float = a + (b - a) * t
        }
    }
}
