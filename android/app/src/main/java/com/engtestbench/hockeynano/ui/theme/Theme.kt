package com.engtestbench.hockeynano.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.engtestbench.hockeynano.core.model.TeamId
import java.util.Locale
import kotlin.math.roundToInt

val ObsidianBg = Color(0xFF080B09)
val PanelBg = Color(0xFF101612)
val PanelBgElevated = Color(0xFF161F19)
val BorderSubtle = Color(0xFF233228)

val BenchGreen = Color(0xFF4CBB17)
val TeamACyan = Color(0xFF00E5FF)
val TeamBOrange = Color(0xFFFF6D00)
val TargetGold = Color(0xFFFFD600)
val RefGray = Color(0xFFB0BEC5)
val UnknownPurple = Color(0xFFA78BFA)

val TextPrimary = Color(0xFFECF4EE)
val TextMuted = Color(0xFF8CA092)
val DangerRed = Color(0xFFFF5252)

val HockeyDarkColorScheme = darkColorScheme(
    primary = BenchGreen,
    onPrimary = ObsidianBg,
    primaryContainer = Color(0xFF1A3A12),
    onPrimaryContainer = Color(0xFFC8F7B0),
    secondary = TeamACyan,
    onSecondary = ObsidianBg,
    tertiary = TargetGold,
    onTertiary = ObsidianBg,
    background = ObsidianBg,
    onBackground = TextPrimary,
    surface = PanelBg,
    onSurface = TextPrimary,
    surfaceVariant = PanelBgElevated,
    onSurfaceVariant = TextMuted,
    outline = BorderSubtle,
    error = DangerRed,
    onError = Color.Black
)

private val HockeyTypography = Typography(
    titleMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 16.sp,
        letterSpacing = 0.1.sp,
        color = TextPrimary
    ),
    bodyMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 13.sp,
        lineHeight = 18.sp,
        color = TextPrimary
    ),
    labelSmall = TextStyle(
        fontFamily = FontFamily.Monospace,
        fontWeight = FontWeight.Medium,
        fontSize = 11.sp,
        letterSpacing = 0.2.sp,
        color = TextMuted
    )
)

@Composable
fun HockeyNanoTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = HockeyDarkColorScheme,
        typography = HockeyTypography,
        content = content
    )
}

fun teamColor(team: TeamId): Color = when (team) {
    TeamId.TEAM_A -> TeamACyan
    TeamId.TEAM_B -> TeamBOrange
    TeamId.REFEREE -> RefGray
    TeamId.UNKNOWN -> UnknownPurple
}

fun teamTag(team: TeamId): String = when (team) {
    TeamId.TEAM_A -> "A"
    TeamId.TEAM_B -> "B"
    TeamId.REFEREE -> "R"
    TeamId.UNKNOWN -> "?"
}

fun formatClock(seconds: Double): String {
    val total = seconds.coerceAtLeast(0.0).roundToInt()
    val mins = total / 60
    val secs = total % 60
    return String.format(Locale.US, "%02d:%02d", mins, secs)
}

fun formatBytes(bytes: Long): String {
    if (bytes < 1024L) return "${bytes}B"
    val kb = bytes / 1024.0
    if (kb < 1024.0) return String.format(Locale.US, "%.0fKB", kb)
    val mb = kb / 1024.0
    return String.format(Locale.US, "%.1fMB", mb)
}
