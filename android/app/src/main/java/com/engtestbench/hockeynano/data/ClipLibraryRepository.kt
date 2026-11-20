package com.engtestbench.hockeynano.data

import android.content.ContentUris
import android.content.Context
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import com.engtestbench.hockeynano.core.model.TeamId
import com.engtestbench.hockeynano.video.ExportedClipInfo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.Locale
import kotlin.math.roundToInt

@Serializable
private data class RecentYouTubeEntry(
    val title: String,
    val watchUrl: String,
    val savedAtMs: Long = 0L
)

class ClipLibraryRepository(private val context: Context) {
    private val json = Json {
        ignoreUnknownKeys = true
        prettyPrint = true
        encodeDefaults = true
    }

    private val indexFile: File
        get() = File(context.filesDir, INDEX_FILE_NAME)

    private val recentYouTubeFile: File
        get() = File(context.filesDir, RECENT_YT_FILE_NAME)

    private fun clipsDir(): File {
        val moviesDir = context.getExternalFilesDir(Environment.DIRECTORY_MOVIES)
            ?: File(context.filesDir, "Movies")
        return File(moviesDir, CLIPS_DIR_NAME).apply { mkdirs() }
    }

    fun outputFileFor(
        playerNumber: String,
        startSec: Double,
        endSec: Double,
        burnedMarks: Boolean
    ): File {
        val safePlayer = playerNumber
            .trim()
            .trimStart('#')
            .replace(Regex("[^A-Za-z0-9_-]"), "")
            .ifBlank { "all" }
        val sInt = startSec.roundToInt().coerceAtLeast(0)
        val eInt = endSec.roundToInt().coerceAtLeast(sInt)
        val suffix = if (burnedMarks) "marked" else "fast"
        return File(clipsDir(), "hockey_p${safePlayer}_${sInt}s_${eInt}s_${suffix}.mp4")
    }

    suspend fun listExportedClips(): List<ExportedClipInfo> = withContext(Dispatchers.IO) {
        indexMutex.withLock {
            val indexed = readIndexLocked()
                .filter { info ->
                    val f = File(info.filePath)
                    f.exists() && f.isFile && f.length() > 0L
                }
            val knownPaths = indexed.mapTo(HashSet()) { it.filePath }
            val extras = discoverOrphanExportsLocked(knownPaths)
            (indexed + extras).sortedByDescending { it.createdAtMs }
        }
    }

    suspend fun recordExport(info: ExportedClipInfo) = withContext(Dispatchers.IO) {
        indexMutex.withLock {
            val current = readIndexLocked()
                .filterNot { it.id == info.id || it.filePath == info.filePath }
                .filter { File(it.filePath).let { f -> f.exists() && f.length() > 0L } }
            val updated = (listOf(info) + current).sortedByDescending { it.createdAtMs }
            writeIndexLocked(updated)
        }
    }

    suspend fun saveRecentYouTubeVideo(title: String, watchUrl: String) = withContext(Dispatchers.IO) {
        val cleanUrl = watchUrl.trim()
        if (cleanUrl.isBlank()) return@withContext
        val cleanTitle = title.trim().ifBlank { "YouTube Hockey" }

        ytMutex.withLock {
            val current = readRecentYouTubeLocked()
                .filterNot { it.watchUrl.trim() == cleanUrl }
            val updated = (listOf(
                RecentYouTubeEntry(
                    title = cleanTitle,
                    watchUrl = cleanUrl,
                    savedAtMs = System.currentTimeMillis()
                )
            ) + current).take(MAX_RECENT_YOUTUBE)
            writeJsonAtomically(recentYouTubeFile, json.encodeToString(updated))
        }
    }

    suspend fun discoverLocalHockeyVideos(): List<Pair<String, String>> = withContext(Dispatchers.IO) {
        val results = ArrayList<Pair<String, String>>()
        val seenNames = HashSet<String>()
        val seenPaths = HashSet<String>()

        fun addEntry(title: String, pathOrUri: String, fileNameKey: String? = null) {
            if (seenPaths.add(pathOrUri)) {
                if (fileNameKey != null) {
                    seenNames.add(fileNameKey.lowercase(Locale.US))
                }
                results.add(title to pathOrUri)
            }
        }

        // 1) /sdcard/Download/HockeyClips
        val downloadClipsDir = File("/sdcard/Download/HockeyClips")
        scanVideoDirectory(downloadClipsDir).forEach { file ->
            addEntry(
                title = "${file.name} (${formatMb(file.length())})",
                pathOrUri = file.absolutePath,
                fileNameKey = file.name
            )
        }

        // 2) App external Movies/HockeyInputs
        val appInputsDir = context.getExternalFilesDir(Environment.DIRECTORY_MOVIES)?.let {
            File(it, "HockeyInputs")
        }
        if (appInputsDir != null) {
            scanVideoDirectory(appInputsDir).forEach { file ->
                addEntry(
                    title = "Inputs: ${file.name} (${formatMb(file.length())})",
                    pathOrUri = file.absolutePath,
                    fileNameKey = file.name
                )
            }
        }

        // 3) MediaStore query (Scoped Storage fallback on Android 17 when direct File listing is restricted)
        queryMediaStoreMp4s(seenNames).forEach { (title, uri) ->
            addEntry(title = title, pathOrUri = uri)
        }

        // 4) Bundled raw sample
        val rawResId = context.resources.getIdentifier("hockey_sample_30s", "raw", context.packageName)
        if (rawResId != 0) {
            addEntry(
                title = "Bundled Demo: hockey_sample_30s.mp4",
                pathOrUri = "android.resource://${context.packageName}/raw/hockey_sample_30s"
            )
        }

        // 5) Saved recent YouTube URLs (newest first, deduplicated)
        val recentYt = ytMutex.withLock { readRecentYouTubeLocked() }
        recentYt.forEach { entry ->
            val url = entry.watchUrl.trim()
            if (url.isNotEmpty()) {
                val label = if (entry.title.contains("(YT)", ignoreCase = true) ||
                    entry.title.startsWith("YT:", ignoreCase = true)
                ) {
                    entry.title
                } else {
                    "${entry.title} (YT)"
                }
                addEntry(title = label, pathOrUri = url)
            }
        }

        // 6) Built-in YouTube hockey preset
        addEntry(
            title = PRESET_YOUTUBE_TITLE,
            pathOrUri = PRESET_YOUTUBE_URL
        )

        results
    }

    private fun scanVideoDirectory(dir: File): List<File> {
        if (!dir.exists() || !dir.isDirectory) return emptyList()
        val files = runCatching { dir.listFiles() }.getOrNull() ?: return emptyList()
        return files
            .asSequence()
            .filter { file ->
                file.isFile &&
                    file.length() > 0L &&
                    file.canRead() &&
                    isSupportedVideoExt(file.name)
            }
            .sortedBy { it.name.lowercase(Locale.US) }
            .toList()
    }

    private fun queryMediaStoreMp4s(seenNames: Set<String>): List<Pair<String, String>> {
        val hockeyHits = ArrayList<Pair<String, String>>()
        val otherMp4s = ArrayList<Pair<String, String>>()

        val projection = arrayOf(
            MediaStore.Video.Media._ID,
            MediaStore.Video.Media.DISPLAY_NAME,
            MediaStore.Video.Media.SIZE,
            MediaStore.Video.Media.DATA,
            MediaStore.Video.Media.RELATIVE_PATH
        )
        val selection = "${MediaStore.Video.Media.MIME_TYPE} = ? OR ${MediaStore.Video.Media.DISPLAY_NAME} LIKE ?"
        val selectionArgs = arrayOf("video/mp4", "%.mp4")
        val sortOrder = "${MediaStore.Video.Media.DATE_MODIFIED} DESC"

        runCatching {
            context.contentResolver.query(
                MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                projection,
                selection,
                selectionArgs,
                sortOrder
            )?.use { cursor ->
                val idCol = cursor.getColumnIndexOrThrow(MediaStore.Video.Media._ID)
                val nameCol = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DISPLAY_NAME)
                val sizeCol = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.SIZE)
                val dataCol = cursor.getColumnIndex(MediaStore.Video.Media.DATA)
                val relCol = cursor.getColumnIndex(MediaStore.Video.Media.RELATIVE_PATH)

                while (cursor.moveToNext()) {
                    val id = cursor.getLong(idCol)
                    val name = cursor.getString(nameCol) ?: continue
                    val size = cursor.getLong(sizeCol)
                    val dataPath = if (dataCol >= 0) cursor.getString(dataCol) else null
                    val relPath = if (relCol >= 0) cursor.getString(relCol) else null

                    // Exclude our own exported 30s clips
                    if (dataPath?.contains(CLIPS_DIR_NAME, ignoreCase = true) == true ||
                        relPath?.contains(CLIPS_DIR_NAME, ignoreCase = true) == true
                    ) {
                        continue
                    }

                    val nameKey = name.lowercase(Locale.US)
                    if (nameKey in seenNames) continue
                    if (size <= 0L) continue

                    val contentUri = ContentUris.withAppendedId(
                        MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                        id
                    ).toString()
                    val label = "$name (${formatMb(size)})"

                    val looksLikeHockey = nameKey.contains("hockey") ||
                        nameKey.contains("dublin") ||
                        nameKey.contains("iceland") ||
                        (relPath?.contains("hockey", ignoreCase = true) == true)

                    if (looksLikeHockey) {
                        hockeyHits.add(label to contentUri)
                    } else if (otherMp4s.size < 15) {
                        otherMp4s.add(label to contentUri)
                    }
                }
            }
        }.onFailure {
            Log.w(LOG_TAG, "MediaStore query skipped/unavailable: ${it.message}")
        }

        return hockeyHits + otherMp4s
    }

    private fun readIndexLocked(): List<ExportedClipInfo> {
        val file = indexFile
        if (!file.exists() || !file.isFile) return emptyList()
        return runCatching {
            val raw = file.readText()
            if (raw.isBlank()) emptyList() else json.decodeFromString<List<ExportedClipInfo>>(raw)
        }.getOrElse { err ->
            Log.w(LOG_TAG, "Failed parsing clips_index.json: ${err.message}")
            emptyList()
        }
    }

    private fun writeIndexLocked(items: List<ExportedClipInfo>) {
        writeJsonAtomically(indexFile, json.encodeToString(items))
    }

    private fun readRecentYouTubeLocked(): List<RecentYouTubeEntry> {
        val file = recentYouTubeFile
        if (!file.exists() || !file.isFile) return emptyList()
        return runCatching {
            val raw = file.readText()
            if (raw.isBlank()) emptyList() else json.decodeFromString<List<RecentYouTubeEntry>>(raw)
        }.getOrElse { err ->
            Log.w(LOG_TAG, "Failed parsing $RECENT_YT_FILE_NAME: ${err.message}")
            emptyList()
        }
    }

    private fun writeJsonAtomically(target: File, content: String) {
        target.parentFile?.mkdirs()
        val tmp = File(target.parentFile, "${target.name}.tmp")
        tmp.writeText(content)
        val moved = runCatching {
            Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
            true
        }.getOrDefault(false)
        if (!moved && !tmp.renameTo(target)) {
            tmp.copyTo(target, overwrite = true)
            tmp.delete()
        }
    }

    private fun discoverOrphanExportsLocked(knownPaths: Set<String>): List<ExportedClipInfo> {
        val files = clipsDir().listFiles() ?: return emptyList()
        val extras = ArrayList<ExportedClipInfo>()
        for (file in files) {
            if (!file.isFile || file.length() <= 0L) continue
            if (!file.name.endsWith(".mp4", ignoreCase = true)) continue
            if (file.absolutePath in knownPaths) continue

            val match = ORPHAN_NAME_REGEX.matchEntire(file.name)
            val player = match?.groupValues?.getOrNull(1) ?: "all"
            val sSec = match?.groupValues?.getOrNull(2)?.toDoubleOrNull() ?: 0.0
            val eSec = match?.groupValues?.getOrNull(3)?.toDoubleOrNull() ?: (sSec + 30.0)
            val burned = match?.groupValues?.getOrNull(4)?.equals("marked", ignoreCase = true) == true

            extras.add(
                ExportedClipInfo(
                    id = "disk_${file.name}",
                    filePath = file.absolutePath,
                    playerNumber = player,
                    team = TeamId.UNKNOWN,
                    startSec = sSec,
                    endSec = eSec,
                    durationSec = (eSec - sSec).coerceAtLeast(0.0),
                    sizeBytes = file.length(),
                    burnedMarks = burned,
                    nanoSummary = null,
                    createdAtMs = file.lastModified()
                )
            )
        }
        return extras
    }

    private fun isSupportedVideoExt(name: String): Boolean {
        val lower = name.lowercase(Locale.US)
        return lower.endsWith(".mp4") || lower.endsWith(".mov") || lower.endsWith(".mkv")
    }

    private fun formatMb(bytes: Long): String {
        val mb = bytes / (1024.0 * 1024.0)
        return "%.1f MB".format(Locale.US, mb)
    }

    companion object {
        private const val LOG_TAG = "HockeyNanoBench"
        private const val CLIPS_DIR_NAME = "HockeyClips30s"
        private const val INDEX_FILE_NAME = "clips_index.json"
        private const val RECENT_YT_FILE_NAME = "recent_youtube.json"
        private const val MAX_RECENT_YOUTUBE = 15
        private const val PRESET_YOUTUBE_TITLE = "TV Bulls vs Jr Reign (YT)"
        private const val PRESET_YOUTUBE_URL = "https://www.youtube.com/watch?v=1W1ZfGruhP4"
        private val indexMutex = Mutex()
        private val ytMutex = Mutex()
        private val ORPHAN_NAME_REGEX = Regex("^hockey_p([A-Za-z0-9_-]+)_(\\d+)s_(\\d+)s_(fast|marked)\\.mp4$")
    }
}
