package com.engtestbench.hockeynano.ui.components

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.net.Uri
import android.view.ViewGroup
import android.webkit.CookieManager
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.SportsHockey
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.engtestbench.hockeynano.ui.theme.BenchGreen
import com.engtestbench.hockeynano.ui.theme.BorderSubtle
import com.engtestbench.hockeynano.ui.theme.ObsidianBg
import com.engtestbench.hockeynano.ui.theme.PanelBg
import com.engtestbench.hockeynano.ui.theme.PanelBgElevated
import com.engtestbench.hockeynano.ui.theme.TargetGold
import com.engtestbench.hockeynano.ui.theme.TextMuted
import com.engtestbench.hockeynano.ui.theme.TextPrimary
import org.json.JSONObject
import org.json.JSONTokener
import java.util.Locale

private const val LEONXZHAO_GAMES_URL = "https://coach.leonxzhao.com/games"
private const val COACHBOX_GAMES_URL = "https://coachbox.tvbdhockey.com/games"
private const val EXTRACT_VIDEO_JS =
    "(function(){try{" +
        "var isHttp=function(u){return /^https?:\\/\\//i.test((u||'').trim());};" +
        "var pickVideoSrc=function(v){" +
            "var c=[];" +
            "if(v.currentSrc){c.push(v.currentSrc);}" +
            "if(v.src){c.push(v.src);}" +
            "var ss=v.querySelectorAll('source');" +
            "for(var i=0;i<ss.length;i++){" +
                "var su=ss[i].src||ss[i].getAttribute('src')||'';" +
                "if(su){c.push(su);}" +
            "}" +
            "for(var j=0;j<c.length;j++){if(isHttp(c[j])){return c[j];}}" +
            "return c.length?c[0]:'';" +
        "};" +
        "var vs=document.querySelectorAll('video');" +
        "var best='';" +
        "for(var k=0;k<vs.length;k++){" +
            "var v=vs[k];" +
            "var cand=pickVideoSrc(v);" +
            "if(!cand){continue;}" +
            "if(!best||(isHttp(cand)&&!isHttp(best))){best=cand;}" +
            "if(isHttp(cand)&&(!v.paused||v.readyState>0)){best=cand;break;}" +
        "}" +
        "if(!isHttp(best)){" +
            "var ifs=document.querySelectorAll('iframe[src]');" +
            "for(var m=0;m<ifs.length;m++){" +
                "var fs=ifs[m].src||ifs[m].getAttribute('src')||'';" +
                "if(/^https?:\\/\\/([^\\/]+\\.)?(youtube\\.com|youtube-nocookie\\.com|youtu\\.be)\\//i.test(fs)){best=fs;break;}" +
            "}" +
        "}" +
        "return JSON.stringify({src:(best||''),title:(document.title||'')});" +
    "}catch(e){return JSON.stringify({src:'',title:(document.title||'')});}})();"
private const val COACHBOX_HINT_TEXT =
    "Open a CoachBox game and tap Play first so the <video> stream loads."
private const val COACHBOX_BLOB_HINT_TEXT =
    "This video is playing via a browser blob stream; open a direct MP4/GCS or YouTube embed stream."

private data class CoachBoxHostPreset(
    val label: String,
    val host: String,
    val url: String
)

private val COACHBOX_HOST_PRESETS = listOf(
    CoachBoxHostPreset(
        label = "LeonxZhao (5 Games)",
        host = "coach.leonxzhao.com",
        url = LEONXZHAO_GAMES_URL
    ),
    CoachBoxHostPreset(
        label = "TVBD CoachBox",
        host = "coachbox.tvbdhockey.com",
        url = COACHBOX_GAMES_URL
    )
)

/**
 * Full-screen WebView connector for CoachBox hosts (`coach.leonxzhao.com/games` and
 * `coachbox.tvbdhockey.com/games`) allowing the user to browse/sign in, play any game
 * video, and send its active `<video>` stream URL directly into the on-device Nano Hockey analyzer.
 */
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun CoachBoxWebDialog(
    initialUrl: String = LEONXZHAO_GAMES_URL,
    onVideoSelected: (title: String, streamUrl: String) -> Unit,
    onDismiss: () -> Unit
) {
    var webViewRef by remember { mutableStateOf<WebView?>(null) }
    var canGoBack by remember { mutableStateOf(false) }
    var pageProgress by remember { mutableIntStateOf(0) }
    var currentPageUrl by remember(initialUrl) { mutableStateOf(initialUrl) }
    var currentPageTitle by remember(initialUrl) { mutableStateOf(formatCoachBoxHeader(initialUrl)) }
    var inlineHint by remember { mutableStateOf<String?>(null) }

    val activeHost = remember(currentPageUrl) {
        extractCoachBoxHost(currentPageUrl).lowercase(Locale.US)
    }
    val activeHeaderTitle = remember(currentPageUrl) {
        formatCoachBoxHeader(currentPageUrl)
    }

    DisposableEffect(Unit) {
        onDispose {
            runCatching { CookieManager.getInstance().flush() }
            webViewRef?.let { wv ->
                wv.stopLoading()
                wv.destroy()
            }
            webViewRef = null
        }
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(
            usePlatformDefaultWidth = false,
            dismissOnBackPress = true,
            dismissOnClickOutside = false
        )
    ) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = ObsidianBg
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                // Top navigation + active CoachBox host title bar
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(PanelBg)
                        .padding(horizontal = 6.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(2.dp)
                ) {
                    IconButton(
                        onClick = onDismiss,
                        modifier = Modifier.size(36.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Rounded.Close,
                            contentDescription = "Close",
                            tint = TextPrimary
                        )
                    }

                    Icon(
                        imageVector = Icons.Rounded.SportsHockey,
                        contentDescription = null,
                        tint = BenchGreen,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(Modifier.width(4.dp))

                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = activeHeaderTitle,
                            color = TextPrimary,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        Text(
                            text = currentPageUrl,
                            color = TextMuted,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 9.5.sp,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }

                    IconButton(
                        onClick = {
                            val wv = webViewRef
                            if (wv != null && wv.canGoBack()) {
                                inlineHint = null
                                wv.goBack()
                            }
                        },
                        enabled = canGoBack,
                        modifier = Modifier.size(36.dp)
                    ) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Rounded.ArrowBack,
                            contentDescription = "Back",
                            tint = if (canGoBack) TextPrimary else TextMuted.copy(alpha = 0.4f)
                        )
                    }

                    IconButton(
                        onClick = {
                            inlineHint = null
                            webViewRef?.reload()
                        },
                        modifier = Modifier.size(36.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Rounded.Refresh,
                            contentDescription = "Refresh",
                            tint = TextPrimary
                        )
                    }
                }

                // Compact 1-tap host switcher chips
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(PanelBg)
                        .horizontalScroll(rememberScrollState())
                        .padding(horizontal = 10.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    COACHBOX_HOST_PRESETS.forEach { preset ->
                        val isSelected = activeHost == preset.host || activeHost.endsWith(".${preset.host}")
                        FilterChip(
                            selected = isSelected,
                            onClick = {
                                inlineHint = null
                                currentPageUrl = preset.url
                                currentPageTitle = formatCoachBoxHeader(preset.url)
                                webViewRef?.loadUrl(preset.url)
                            },
                            label = {
                                Text(
                                    text = preset.label,
                                    fontSize = 11.sp,
                                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                                    maxLines = 1,
                                    softWrap = false
                                )
                            },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = BenchGreen.copy(alpha = 0.22f),
                                selectedLabelColor = BenchGreen,
                                containerColor = PanelBgElevated,
                                labelColor = TextMuted
                            ),
                            border = FilterChipDefaults.filterChipBorder(
                                enabled = true,
                                selected = isSelected,
                                borderColor = BorderSubtle,
                                selectedBorderColor = BenchGreen.copy(alpha = 0.75f)
                            ),
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier.height(30.dp)
                        )
                    }
                }

                // Action strip with bright "Analyze Active Video" trigger
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(PanelBgElevated)
                        .border(1.dp, BorderSubtle)
                        .padding(horizontal = 10.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = currentPageTitle.ifBlank { "Browse games & press Play" },
                        color = TextMuted,
                        fontSize = 11.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier
                            .weight(1f)
                            .padding(end = 8.dp)
                    )

                    Button(
                        onClick = {
                            val wv = webViewRef
                            if (wv == null) {
                                inlineHint = COACHBOX_HINT_TEXT
                                return@Button
                            }
                            wv.evaluateJavascript(EXTRACT_VIDEO_JS) { rawResult ->
                                val parsed = parseCoachBoxVideoJson(rawResult)
                                val src = parsed?.first.orEmpty()
                                val title = parsed?.second.orEmpty()
                                if (isCoachBoxHttpMp4Url(src)) {
                                    inlineHint = null
                                    runCatching { CookieManager.getInstance().flush() }
                                    val cleanTitle = title
                                        .ifBlank { currentPageTitle }
                                        .ifBlank { "CoachBox Hockey" }
                                    onVideoSelected(cleanTitle, src)
                                    onDismiss()
                                } else if (src.startsWith("blob:", ignoreCase = true)) {
                                    inlineHint = COACHBOX_BLOB_HINT_TEXT
                                } else {
                                    inlineHint = COACHBOX_HINT_TEXT
                                }
                            }
                        },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = BenchGreen,
                            contentColor = ObsidianBg
                        ),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 0.dp),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.height(34.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Rounded.PlayArrow,
                            contentDescription = null,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(Modifier.width(4.dp))
                        Text(
                            text = "Analyze Active Video",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            maxLines = 1,
                            softWrap = false
                        )
                    }
                }

                if (pageProgress in 1..99) {
                    LinearProgressIndicator(
                        progress = { (pageProgress / 100f).coerceIn(0f, 1f) },
                        color = BenchGreen,
                        trackColor = BorderSubtle,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(2.dp)
                    )
                }

                if (!inlineHint.isNullOrBlank()) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(TargetGold.copy(alpha = 0.16f))
                            .border(1.dp, TargetGold.copy(alpha = 0.65f))
                            .padding(horizontal = 10.dp, vertical = 7.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = inlineHint.orEmpty(),
                            color = TargetGold,
                            fontSize = 11.5.sp,
                            fontWeight = FontWeight.Medium,
                            modifier = Modifier.weight(1f)
                        )
                        IconButton(
                            onClick = { inlineHint = null },
                            modifier = Modifier.size(24.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Rounded.Close,
                                contentDescription = "Dismiss hint",
                                tint = TargetGold,
                                modifier = Modifier.size(14.dp)
                            )
                        }
                    }
                }

                AndroidView(
                    factory = { ctx ->
                        CookieManager.getInstance().setAcceptCookie(true)
                        WebView(ctx).apply {
                            layoutParams = ViewGroup.LayoutParams(
                                ViewGroup.LayoutParams.MATCH_PARENT,
                                ViewGroup.LayoutParams.MATCH_PARENT
                            )
                            CookieManager.getInstance().setAcceptThirdPartyCookies(this, true)
                            settings.apply {
                                javaScriptEnabled = true
                                domStorageEnabled = true
                                mediaPlaybackRequiresUserGesture = false
                                useWideViewPort = true
                                loadWithOverviewMode = true
                                builtInZoomControls = true
                                displayZoomControls = false
                                mixedContentMode = WebSettings.MIXED_CONTENT_COMPATIBILITY_MODE
                            }
                            webChromeClient = object : WebChromeClient() {
                                override fun onProgressChanged(view: WebView?, newProgress: Int) {
                                    pageProgress = newProgress
                                }

                                override fun onReceivedTitle(view: WebView?, title: String?) {
                                    if (!title.isNullOrBlank()) {
                                        currentPageTitle = title
                                    }
                                }
                            }
                            webViewClient = object : WebViewClient() {
                                private fun syncNavState(view: WebView?, url: String?) {
                                    canGoBack = view?.canGoBack() == true
                                    if (!url.isNullOrBlank()) {
                                        currentPageUrl = url
                                    }
                                }

                                override fun shouldOverrideUrlLoading(
                                    view: WebView?,
                                    request: WebResourceRequest?
                                ): Boolean = false

                                override fun onPageStarted(
                                    view: WebView?,
                                    url: String?,
                                    favicon: Bitmap?
                                ) {
                                    syncNavState(view, url)
                                }

                                override fun onPageFinished(view: WebView?, url: String?) {
                                    syncNavState(view, url)
                                }

                                override fun doUpdateVisitedHistory(
                                    view: WebView?,
                                    url: String?,
                                    isReload: Boolean
                                ) {
                                    syncNavState(view, url)
                                }
                            }
                            webViewRef = this
                            loadUrl(initialUrl)
                        }
                    },
                    update = { wv ->
                        webViewRef = wv
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .background(Color.Black)
                )
            }
        }
    }
}

private fun extractCoachBoxHost(url: String): String =
    runCatching { Uri.parse(url).host?.trim().orEmpty() }.getOrDefault("")

private fun formatCoachBoxHeader(url: String): String {
    val host = extractCoachBoxHost(url)
    return if (host.isNotBlank()) "CoachBox · $host" else "CoachBox"
}

private fun parseCoachBoxVideoJson(rawResult: String?): Pair<String, String>? {
    if (rawResult.isNullOrBlank() || rawResult == "null") return null
    return runCatching {
        val tokenerVal = JSONTokener(rawResult).nextValue()
        val jsonObj = when (tokenerVal) {
            is JSONObject -> tokenerVal
            is String -> JSONObject(tokenerVal)
            else -> return null
        }
        val src = jsonObj.optString("src", "").trim()
        val title = jsonObj.optString("title", "").trim()
        src to title
    }.getOrNull()
}

private fun isCoachBoxHttpMp4Url(url: String): Boolean {
    val clean = url.trim()
    if (clean.isEmpty()) return false
    val lower = clean.lowercase(Locale.US)
    if (!lower.startsWith("http://") && !lower.startsWith("https://")) return false
    val path = runCatching { Uri.parse(clean).path.orEmpty().lowercase(Locale.US) }.getOrDefault("")
    if (path.endsWith(".m3u8") || path.endsWith(".mpd")) return false
    return path.endsWith(".mp4") || lower.contains(".mp4") || path.isNotBlank()
}
