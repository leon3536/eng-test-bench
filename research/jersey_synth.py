"""Roster-free synthetic hockey jersey dataset generator v8 (100% zero prior / zero test leakage).

Generates realistic 128x128 RGB letterboxed upper-torso hockey crops matching Android
`JerseyStrOnnxRecognizer.kt` v5 (`y: 4%..58%`, torso-centered, pad 114, bicubic upscale,
RGB-preserving 2%..98% local stretch):
  - Overhead camera pitch foreshortening (`v_scale in [0.54, 0.96]` on 75% of crops, higher
    numeral placement `py0 in 0.08..0.38 * H`)
  - Wide skating stride tilt (`50%` in `[-18, +18]` deg, `50%` in `[+/-18, +/-52]` deg)
    rotated via `cv2.getRotationMatrix2D(center, +tilt_deg, 1.0)` (`+` = CCW in image space)
  - Low-luma chromatic twill (`28%` of positives): dark crimson/red/scarlet/orange/royal digits
    (`|DeltaLuma| < 35`) separated by crisp 1-2 px white/gold/light tackle-twill outline on
    dark navy/charcoal/forest/black jerseys
  - Telephoto low-res pre-upscale (`min_short_side in 18..62 px`), directional motion blur,
    H.264 4:2:0 chroma subsampling, JPEG/codec ringing, and 16% translucent Plexiglas glare,
    rink-glass stanchion, or netting overlap
  - Uniform jersey classes 1..99 and "00" (class 100), plus class 0 (illegible/hard negative)
  - Multi-task targets: `sr_target` (32x32 uint8 clean numeral stroke mask, 0 on hard negatives),
    `tilt_deg` (float32 in [-55, +55], + = CCW), compositional digit labels, 14% partial
    occlusions, and 32% hard negatives.

All priors over 1..99 and "00" are strictly uniform (1/100). Never imports or reads any test frame.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = Path(os.environ.get("HOCKEY_SCRATCH_DIR", str(REPO_ROOT / "research" / "scratch")))

NUM_CLASSES = 101  # 0 = none/illegible, 1..99 = "1".."99", 100 = "00"
DIGIT_BLANK = 10   # 0..9 = digit, 10 = blank / single-right-blank

PARTIAL_NONE = 0
PARTIAL_LEFT_ONLY = 1   # left digit visible, right occluded ("a?")
PARTIAL_RIGHT_ONLY = 2  # right digit visible, left occluded ("?b")

LETTERBOX_PAD = 114
TARGET_SIZE = 128
SR_SIZE = 32

_SYSTEM_FONT_PATHS = [
    "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/URWGothic-Demi.otf",
    "/usr/share/fonts/opentype/urw-base35/C059-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-BoldOblique.otf",
]

# Style 0: Squared/beveled classic NHL hockey varsity block (squared '2', '3', '5', '6', '8', '9', '0')
_NHL_SQUARED_BLOCK: Dict[str, List[List[Tuple[float, float]]]] = {
    "0": [[(0.22, 0.08), (0.78, 0.08), (0.88, 0.18), (0.88, 0.82), (0.78, 0.92), (0.22, 0.92), (0.12, 0.82), (0.12, 0.18), (0.22, 0.08)]],
    "1": [[(0.26, 0.18), (0.54, 0.08), (0.54, 0.92)], [(0.22, 0.92), (0.84, 0.92)]],
    "2": [[(0.14, 0.22), (0.22, 0.08), (0.78, 0.08), (0.86, 0.18), (0.86, 0.42), (0.76, 0.48), (0.22, 0.48), (0.14, 0.56), (0.14, 0.90), (0.88, 0.90)]],
    "3": [[(0.14, 0.18), (0.24, 0.08), (0.78, 0.08), (0.86, 0.18), (0.86, 0.40), (0.76, 0.48), (0.34, 0.48)],
          [(0.76, 0.48), (0.86, 0.56), (0.86, 0.82), (0.78, 0.92), (0.24, 0.92), (0.14, 0.82)]],
    "4": [[(0.74, 0.92), (0.74, 0.08), (0.14, 0.62), (0.88, 0.62)]],
    "5": [[(0.86, 0.08), (0.16, 0.08), (0.16, 0.46), (0.78, 0.46), (0.86, 0.56), (0.86, 0.82), (0.78, 0.92), (0.22, 0.92), (0.14, 0.82)]],
    "6": [[(0.84, 0.18), (0.76, 0.08), (0.24, 0.08), (0.14, 0.20), (0.14, 0.82), (0.24, 0.92), (0.78, 0.92), (0.86, 0.82), (0.86, 0.56), (0.78, 0.46), (0.16, 0.46)]],
    "7": [[(0.14, 0.08), (0.88, 0.08), (0.44, 0.92)]],
    "8": [[(0.24, 0.08), (0.76, 0.08), (0.86, 0.18), (0.86, 0.40), (0.76, 0.48), (0.24, 0.48), (0.14, 0.40), (0.14, 0.18), (0.24, 0.08)],
          [(0.24, 0.48), (0.76, 0.48), (0.86, 0.58), (0.86, 0.82), (0.76, 0.92), (0.24, 0.92), (0.14, 0.82), (0.14, 0.58), (0.24, 0.48)]],
    "9": [[(0.16, 0.82), (0.24, 0.92), (0.76, 0.92), (0.86, 0.80), (0.86, 0.18), (0.76, 0.08), (0.22, 0.08), (0.14, 0.18), (0.14, 0.44), (0.22, 0.54), (0.84, 0.54)]],
}

# Style 1: Angled varsity block ('2' diagonal neck, pillar '1' without base bar)
_NHL_ANGLED_BLOCK: Dict[str, List[List[Tuple[float, float]]]] = {
    **_NHL_SQUARED_BLOCK,
    "1": [[(0.32, 0.20), (0.56, 0.08), (0.56, 0.92)]],
    "2": [[(0.14, 0.24), (0.26, 0.08), (0.76, 0.08), (0.88, 0.22), (0.86, 0.40), (0.70, 0.54), (0.14, 0.90), (0.88, 0.90)]],
    "7": [[(0.14, 0.08), (0.88, 0.08), (0.64, 0.48), (0.44, 0.92)]],
}

# Style 2: Rounded athletic
_ROUNDED_BLOCK: Dict[str, List[List[Tuple[float, float]]]] = {
    "0": [[(0.50, 0.08), (0.80, 0.14), (0.88, 0.50), (0.80, 0.86), (0.50, 0.92), (0.20, 0.86), (0.12, 0.50), (0.20, 0.14), (0.50, 0.08)]],
    "1": [[(0.28, 0.22), (0.54, 0.08), (0.54, 0.92)], [(0.24, 0.92), (0.82, 0.92)]],
    "2": [[(0.16, 0.26), (0.34, 0.08), (0.72, 0.08), (0.86, 0.24), (0.82, 0.44), (0.16, 0.90), (0.88, 0.90)]],
    "3": [[(0.16, 0.18), (0.48, 0.08), (0.82, 0.18), (0.82, 0.38), (0.46, 0.48), (0.84, 0.60), (0.84, 0.80), (0.50, 0.92), (0.16, 0.80)]],
    "4": [[(0.70, 0.92), (0.70, 0.08), (0.14, 0.64), (0.88, 0.64)]],
    "5": [[(0.84, 0.08), (0.22, 0.08), (0.18, 0.44), (0.68, 0.44), (0.86, 0.62), (0.82, 0.84), (0.50, 0.92), (0.16, 0.80)]],
    "6": [[(0.80, 0.16), (0.50, 0.08), (0.18, 0.22), (0.14, 0.62), (0.24, 0.88), (0.56, 0.92), (0.86, 0.78), (0.86, 0.56), (0.54, 0.46), (0.16, 0.54)]],
    "7": [[(0.14, 0.08), (0.88, 0.08), (0.56, 0.52), (0.42, 0.92)]],
    "8": [[(0.50, 0.08), (0.82, 0.16), (0.82, 0.38), (0.50, 0.48), (0.18, 0.38), (0.18, 0.16), (0.50, 0.08)],
          [(0.50, 0.48), (0.84, 0.60), (0.84, 0.82), (0.50, 0.92), (0.16, 0.82), (0.16, 0.60), (0.50, 0.48)]],
    "9": [[(0.20, 0.84), (0.50, 0.92), (0.82, 0.78), (0.86, 0.38), (0.76, 0.12), (0.46, 0.08), (0.14, 0.22), (0.14, 0.44), (0.48, 0.54), (0.84, 0.46)]],
}

# Style 3: Wide-serif hockey block ('1' with broad horizontal top flag + pedestal, squared '4' & '7')
_HOCKEY_FLAT_SERIF: Dict[str, List[List[Tuple[float, float]]]] = {
    **_NHL_SQUARED_BLOCK,
    "1": [[(0.20, 0.14), (0.56, 0.10), (0.56, 0.92)], [(0.20, 0.92), (0.86, 0.92)]],
    "4": [[(0.72, 0.92), (0.72, 0.08), (0.50, 0.08), (0.14, 0.62), (0.88, 0.62)]],
}

_DARK_BASES = [
    (24, 34, 64),    # navy
    (26, 28, 34),    # black / charcoal
    (24, 58, 42),    # forest green
    (92, 24, 32),    # maroon / crimson
    (26, 54, 104),   # royal blue
    (124, 28, 34),   # red
    (22, 66, 80),    # teal
    (48, 32, 74),    # purple
    (42, 48, 56),    # dark slate
    (28, 42, 68),    # midnight blue
]

_TWILL_DARK_BODIES = [
    (22, 32, 62),    # dark navy
    (26, 28, 34),    # charcoal / black
    (22, 54, 40),    # dark forest
    (28, 42, 68),    # midnight navy
    (36, 40, 48),    # dark charcoal slate
]

_TWILL_CHROMATIC_DIGITS = [
    (128, 22, 34),   # dark crimson
    (144, 26, 30),   # deep red
    (152, 34, 26),   # dark scarlet
    (124, 52, 20),   # dark burnt orange
    (28, 68, 138),   # dark royal blue
    (106, 24, 42),   # dark maroon
    (22, 82, 92),    # deep teal
]

_LIGHT_BASES = [
    (196, 200, 206), # realistic ice-lit white
    (178, 184, 192), # shaded white
    (216, 220, 224), # bright white
    (168, 174, 182), # silver / grey jersey
    (204, 184, 86),  # muted athletic gold
]

_ACCENT_COLORS = [
    (210, 214, 220),
    (26, 38, 74),
    (148, 34, 42),
    (28, 30, 36),
    (198, 152, 54),
    (28, 82, 52),
    (184, 82, 34),
    (42, 102, 162),
]

_LOADED_FONT_PATHS = [p for p in _SYSTEM_FONT_PATHS if os.path.exists(p)]
_FONT_CACHE: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    f = _FONT_CACHE.get(key)
    if f is None:
        f = ImageFont.truetype(path, size=size)
        _FONT_CACHE[key] = f
    return f


def _jitter_color(rng: np.random.Generator, rgb: Tuple[int, int, int], jitter: int = 14) -> Tuple[int, int, int]:
    arr = np.clip(np.array(rgb, dtype=np.int32) + rng.integers(-jitter, jitter + 1, size=3), 0, 255)
    return int(arr[0]), int(arr[1]), int(arr[2])


def _luma(rgb: Tuple[int, int, int]) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _pick_jersey_colors(
    rng: np.random.Generator,
    force_low_luma_twill: bool = False,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], bool]:
    """Sample (body, yoke, digit, outline, is_low_luma_twill) palette."""
    if force_low_luma_twill:
        body = _jitter_color(rng, _TWILL_DARK_BODIES[int(rng.integers(0, len(_TWILL_DARK_BODIES)))], 10)
        digit = _jitter_color(rng, _TWILL_CHROMATIC_DIGITS[int(rng.integers(0, len(_TWILL_CHROMATIC_DIGITS)))], 10)
        # Ensure |DeltaLuma| < 35 between digit and dark jersey body
        body_l = _luma(body)
        dig_l = _luma(digit)
        if abs(dig_l - body_l) >= 34.0:
            scale = (body_l + np.sign(dig_l - body_l) * 26.0) / max(1.0, dig_l)
            digit = tuple(int(np.clip(round(c * scale), 10, 220)) for c in digit)  # type: ignore[assignment]
        outline_pool = [(216, 220, 226), (212, 178, 72), (192, 198, 206)]
        outline = _jitter_color(rng, outline_pool[int(rng.integers(0, len(outline_pool)))], 10)
        yoke = _jitter_color(rng, digit if rng.random() < 0.65 else outline, 12)
        return body, yoke, digit, outline, True

    is_light = rng.random() < 0.56
    if is_light:
        body = _jitter_color(rng, _LIGHT_BASES[int(rng.integers(0, len(_LIGHT_BASES)))], 12)
        yoke = _jitter_color(rng, _DARK_BASES[int(rng.integers(0, len(_DARK_BASES)))], 14)
        digit = yoke if rng.random() < 0.60 else _jitter_color(rng, _DARK_BASES[int(rng.integers(0, len(_DARK_BASES)))], 12)
        outline = _jitter_color(rng, _ACCENT_COLORS[int(rng.integers(0, len(_ACCENT_COLORS)))], 12)
    else:
        body = _jitter_color(rng, _DARK_BASES[int(rng.integers(0, len(_DARK_BASES)))], 14)
        yoke = _jitter_color(rng, _ACCENT_COLORS[int(rng.integers(0, len(_ACCENT_COLORS)))], 14)
        if rng.random() < 0.74:
            digit = _jitter_color(rng, (208, 212, 218), 14)
            outline = _jitter_color(rng, _DARK_BASES[int(rng.integers(0, len(_DARK_BASES)))], 10) if rng.random() < 0.5 else _jitter_color(rng, _ACCENT_COLORS[int(rng.integers(0, len(_ACCENT_COLORS)))], 12)
        else:
            digit = _jitter_color(rng, (208, 174, 72), 14)
            outline = _jitter_color(rng, (212, 216, 222), 10) if rng.random() < 0.45 else _jitter_color(rng, _DARK_BASES[int(rng.integers(0, len(_DARK_BASES)))], 10)
    return body, yoke, digit, outline, False


def _make_rink_background(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """Muted realistic ice, boards, glass, or dark rink corner background."""
    bg = np.empty((h, w, 3), dtype=np.uint8)
    kind = int(rng.choice([0, 1, 2, 3], p=[0.55, 0.22, 0.15, 0.08]))
    if kind == 0:
        v = int(rng.integers(156, 218))
        base = np.array([v - rng.integers(0, 8), v, v + rng.integers(0, 6)], dtype=np.int32)
        base = np.clip(base, 130, 232)
        bg[:, :] = base
        if rng.random() < 0.35:
            line_y = int(rng.integers(int(h * 0.2), int(h * 0.8)))
            lcol = (80, 105, 165) if rng.random() < 0.5 else (165, 85, 85)
            bg[max(0, line_y - 2):min(h, line_y + 2), :] = lcol
        for _ in range(4):
            sy = int(rng.integers(0, h))
            bg[max(0, sy - 1):min(h, sy + 1), :] = np.clip(base + int(rng.integers(-14, 10)), 110, 240)
    elif kind == 1:
        bw_val = int(rng.integers(165, 215))
        bg[:, :] = (bw_val, bw_val, bw_val)
        cap_y = int(h * rng.uniform(0.15, 0.35))
        bg[max(0, cap_y - 3):cap_y, :] = (65, 95, 145) if rng.random() < 0.6 else (155, 55, 55)
        if rng.random() < 0.65:
            kick_y = int(h * rng.uniform(0.65, 0.85))
            bg[kick_y:min(h, kick_y + int(h * 0.12)), :] = (182, 166, 78)
        bg[int(h * 0.80):, :] = (int(rng.integers(160, 210)),) * 3
    elif kind == 2:
        top = np.array(rng.integers(38, 95, size=3), dtype=np.float32)
        bot = np.array(rng.integers(155, 210, size=3), dtype=np.float32)
        alpha = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        bg[:, :] = np.clip((1.0 - alpha) * top + alpha * bot, 0, 255).astype(np.uint8)
        for _ in range(6):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, int(h * 0.55)))
            cv2.circle(bg, (cx, cy), int(rng.integers(4, 12)), tuple(int(v) for v in rng.integers(40, 165, size=3)), -1)
    else:
        bg[:, :] = _jitter_color(rng, (55, 62, 72), 20)
    return bg


def _draw_vector_digit(
    canvas: np.ndarray,
    ch: str,
    x0: int,
    y0: int,
    dw: int,
    dh: int,
    color: Tuple[int, int, int] | Tuple[int],
    thickness: int,
    style: int,
    slant: float,
    draw_serif_7: bool = False,
) -> None:
    if style == 0:
        segs = _NHL_SQUARED_BLOCK[ch]
    elif style == 1:
        segs = _NHL_ANGLED_BLOCK[ch]
    elif style == 2:
        segs = _ROUNDED_BLOCK[ch]
    else:
        segs = _HOCKEY_FLAT_SERIF[ch]
    for poly in segs:
        pts = []
        for px, py in poly:
            sx = x0 + (px + slant * (0.5 - py)) * dw
            sy = y0 + py * dh
            pts.append([int(round(sx)), int(round(sy))])
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts_arr], isClosed=False, color=color, thickness=max(2, thickness), lineType=cv2.LINE_AA)
    if draw_serif_7 and style in (0, 1) and ch == "7":
        p0 = (int(round(x0 + (0.14 + slant * 0.42) * dw)), int(round(y0 + 0.08 * dh)))
        p1 = (int(round(x0 + (0.14 + slant * 0.30) * dw)), int(round(y0 + 0.22 * dh)))
        cv2.line(canvas, p0, p1, color, max(2, thickness), lineType=cv2.LINE_AA)


def _render_jersey_number_patch(
    rng: np.random.Generator,
    text: str,
    patch_w: int,
    patch_h: int,
    digit_color: Tuple[int, int, int],
    outline_color: Tuple[int, int, int],
    body_color: Tuple[int, int, int],
    x_shift_frac: float = 0.0,
    force_outline: bool = False,
    v_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render (RGBA patch, uint8 clean foreground stroke mask) with tackle-twill outline and optional pitch v_scale."""
    raw_h = max(24, int(round(patch_h / max(0.50, v_scale))))
    use_pil = (len(_LOADED_FONT_PATHS) > 0) and (rng.random() < 0.34)
    alpha_mask = np.zeros((raw_h, patch_w), dtype=np.uint8)
    stroke_mask = np.zeros((raw_h, patch_w), dtype=np.uint8)
    rgb = np.zeros((raw_h, patch_w, 3), dtype=np.uint8)

    if force_outline:
        outline_mode = int(rng.choice([1, 2], p=[0.85, 0.15]))
    else:
        outline_mode = int(rng.choice([0, 1, 2], p=[0.20, 0.58, 0.22]))
        if abs(_luma(digit_color) - _luma(body_color)) < 48.0 and outline_mode == 0:
            outline_mode = 1

    if use_pil:
        pil_img = Image.new("RGBA", (patch_w, raw_h), (0, 0, 0, 0))
        pil_fg = Image.new("L", (patch_w, raw_h), 0)
        draw = ImageDraw.Draw(pil_img)
        draw_fg = ImageDraw.Draw(pil_fg)
        fpath = _LOADED_FONT_PATHS[int(rng.integers(0, len(_LOADED_FONT_PATHS)))]
        fsize = int(raw_h * rng.uniform(0.64, 0.80))
        font = _get_font(fpath, max(14, fsize))
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = int(round((patch_w - tw) * 0.5 - bbox[0] + x_shift_frac * patch_w))
        ty = int(round((raw_h - th) * 0.5 - bbox[1]))
        sw_scale = 1.0 / math.sqrt(max(0.50, v_scale))
        sw = max(3, min(7, int(round(raw_h * rng.uniform(0.055, 0.088) * sw_scale)))) if force_outline else max(2, min(5, int(round(raw_h * rng.uniform(0.028, 0.052)))))
        if outline_mode == 2:
            draw.text((tx, ty), text, font=font, fill=outline_color + (255,), stroke_width=sw * 2, stroke_fill=outline_color + (255,))
            draw.text((tx, ty), text, font=font, fill=body_color + (255,), stroke_width=max(1, sw // (3 if force_outline else 2)), stroke_fill=body_color + (255,))
            draw.text((tx, ty), text, font=font, fill=digit_color + (255,))
        elif outline_mode == 1:
            draw.text((tx, ty), text, font=font, fill=outline_color + (255,), stroke_width=sw, stroke_fill=outline_color + (255,))
            draw.text((tx, ty), text, font=font, fill=digit_color + (255,))
        else:
            draw.text((tx, ty), text, font=font, fill=digit_color + (255,))
        draw_fg.text((tx, ty), text, font=font, fill=255)
        rgba = np.array(pil_img)
        fg = np.array(pil_fg)
    else:
        style = int(rng.choice([0, 1, 2, 3], p=[0.38, 0.24, 0.22, 0.16]))
        draw_serif_7 = bool(rng.random() < 0.35)
        slant = float(rng.uniform(-0.10, 0.10)) if rng.random() < 0.30 else 0.0
        n_chars = len(text)
        dh = int(raw_h * rng.uniform(0.60, 0.78))
        dw = int(dh * rng.uniform(0.50, 0.74))
        gap = int(dw * rng.uniform(0.16, 0.32)) if n_chars == 2 else 0
        total_w = n_chars * dw + (n_chars - 1) * gap
        start_x = int(round((patch_w - total_w) * 0.5 + x_shift_frac * patch_w))
        start_y = int(round((raw_h - dh) * 0.5))
        base_thick = max(3, min(int(round(dh * rng.uniform(0.130, 0.190))), max(3, int(dw * 0.26))))

        def _draw_layer(col: Tuple[int, int, int], thick_delta: int, is_fg: bool = False) -> None:
            layer_m = np.zeros((raw_h, patch_w), dtype=np.uint8)
            for ci, ch in enumerate(text):
                cx0 = start_x + ci * (dw + gap)
                _draw_vector_digit(layer_m, ch, cx0, start_y, dw, dh, (255,), base_thick + thick_delta, style, slant, draw_serif_7)
            m = layer_m > 20
            rgb[m] = col
            alpha_mask[m] = np.maximum(alpha_mask[m], layer_m[m])
            if is_fg:
                stroke_mask[m] = np.maximum(stroke_mask[m], layer_m[m])

        if outline_mode == 2:
            out_d = max(4, min(8, int(round(base_thick * rng.uniform(0.56, 0.86))))) if force_outline else max(3, min(8, int(round(base_thick * rng.uniform(0.45, 0.74)))))
            _draw_layer(outline_color, out_d)
            _draw_layer(body_color, max(1, min(3, int(round(base_thick * rng.uniform(0.18, 0.30))))))
            _draw_layer(digit_color, 0, is_fg=True)
        elif outline_mode == 1:
            out_d = max(4, min(8, int(round(base_thick * rng.uniform(0.54, 0.84))))) if force_outline else max(2, min(7, int(round(base_thick * rng.uniform(0.36, 0.68)))))
            _draw_layer(outline_color, out_d)
            _draw_layer(digit_color, 0, is_fg=True)
        else:
            _draw_layer(digit_color, 0, is_fg=True)

        rgba = np.dstack([rgb, alpha_mask])
        fg = stroke_mask

    if raw_h != patch_h:
        rgba = cv2.resize(rgba, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
        fg = cv2.resize(fg, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
    return rgba, fg


def _draw_referee_jersey(rng: np.random.Generator, jersey: np.ndarray, w: int, h: int) -> None:
    n_stripes = int(rng.integers(5, 11))
    stripe_w = max(2, w // n_stripes)
    phase = int(rng.integers(0, stripe_w * 2))
    dark_col = _jitter_color(rng, (24, 26, 30), 10)
    light_col = _jitter_color(rng, (198, 202, 208), 16)
    xs = np.arange(w)
    is_white = (((xs + phase) // stripe_w) % 2) == 0
    jersey[:, is_white] = light_col
    jersey[:, ~is_white] = dark_col
    if rng.random() < 0.45:
        yoke_h = int(h * rng.uniform(0.10, 0.20))
        jersey[:yoke_h, :] = dark_col
    if rng.random() < 0.38:
        arm_col = _jitter_color(rng, (206, 84, 28), 18)
        ay0 = int(h * rng.uniform(0.24, 0.38))
        ay1 = ay0 + int(h * rng.uniform(0.10, 0.16))
        aw = int(w * rng.uniform(0.14, 0.22))
        jersey[ay0:ay1, :aw] = arm_col
        jersey[ay0:ay1, w - aw:] = arm_col


def _draw_front_crest(rng: np.random.Generator, jersey: np.ndarray, body_col: Tuple[int, int, int], yoke_col: Tuple[int, int, int], outline_col: Tuple[int, int, int]) -> None:
    h, w = jersey.shape[:2]
    cx, cy = w // 2, int(h * rng.uniform(0.42, 0.56))
    crest_kind = int(rng.integers(0, 5))
    c1 = outline_col if abs(_luma(outline_col) - _luma(body_col)) > 35 else (205, 205, 210)
    c2 = yoke_col if abs(_luma(yoke_col) - _luma(body_col)) > 35 else (165, 40, 40)
    if crest_kind == 0:
        rx, ry = int(w * rng.uniform(0.16, 0.26)), int(h * rng.uniform(0.16, 0.25))
        cv2.ellipse(jersey, (cx, cy), (rx, ry), 0, 0, 360, c1, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(jersey, (cx, cy), (max(2, int(rx * 0.70)), max(2, int(ry * 0.70))), 0, 0, 360, c2, -1, lineType=cv2.LINE_AA)
        cv2.line(jersey, (cx - rx, cy), (cx + rx, cy), body_col, max(1, w // 28), lineType=cv2.LINE_AA)
    elif crest_kind == 1:
        rw, rh = int(w * 0.20), int(h * 0.22)
        pts = np.array([[cx - rw, cy - rh], [cx + rw, cy - rh], [cx + rw, cy], [cx, cy + rh], [cx - rw, cy]], dtype=np.int32)
        cv2.fillPoly(jersey, [pts], c1, lineType=cv2.LINE_AA)
        cv2.polylines(jersey, [pts], True, c2, max(1, w // 24), lineType=cv2.LINE_AA)
    elif crest_kind == 2:
        thick = max(4, int(h * 0.12))
        cv2.line(jersey, (int(w * 0.14), int(h * 0.24)), (int(w * 0.86), int(h * 0.74)), c1, thick, lineType=cv2.LINE_AA)
        cv2.circle(jersey, (cx, cy), max(6, int(w * 0.14)), c2, -1, lineType=cv2.LINE_AA)
    elif crest_kind == 3:
        word = str(rng.choice(["COUGARS", "CHIEFS", "DEVILS", "WOLVES", "HAWKS", "SHARKS"]))
        lx, ly = int(w * rng.uniform(0.12, 0.20)), int(h * rng.uniform(0.38, 0.56))
        fsc = max(0.38, w / 260.0)
        cv2.putText(jersey, word, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, fsc, c2, max(2, w // 42), cv2.LINE_AA)
        cv2.putText(jersey, word, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, fsc, c1, max(1, w // 70), cv2.LINE_AA)
        cv2.line(jersey, (int(w * 0.14), ly + 8), (int(w * 0.86), ly + 8), c1, max(2, h // 40), lineType=cv2.LINE_AA)
    else:
        letter = str(rng.choice(["C", "A", "TEAM", "PRO", "HC"]))
        lx, ly = int(w * rng.uniform(0.18, 0.34)), int(h * rng.uniform(0.32, 0.56))
        fsc = max(0.45, h / 180.0) if len(letter) == 1 else max(0.34, h / 250.0)
        cv2.putText(jersey, letter, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, fsc, c1, max(2, w // 28), cv2.LINE_AA)
        cv2.circle(jersey, (cx, int(h * 0.62)), max(4, int(w * 0.12)), c2, -1, lineType=cv2.LINE_AA)


def _apply_cylindrical_warp_and_folds(
    rng: np.random.Generator,
    jersey_rgba: np.ndarray,
    yaw_deg: float,
    torso_width_frac: float = 0.84,
    y_shift_frac: float = 0.0,
    stroke_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Simulate 3D cylindrical jersey torso curvature, yaw rotation, pitch foreshortening, and cloth folds."""
    h, w = jersey_rgba.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = ((xs / (w - 1.0)) * 2.0 - 1.0) / max(0.45, torso_width_frac)
    ny = (ys / (h - 1.0)) * 2.0 - 1.0 - y_shift_frac * 2.0

    yaw = math.radians(yaw_deg)
    arg = np.clip(nx / max(0.35, math.cos(yaw * 0.35)), -0.995, 0.995)
    theta = np.arcsin(arg) - yaw * 0.65
    u = np.sin(theta)
    map_x = ((u + 1.0) * 0.5 * (w - 1.0)).astype(np.float32)
    pitch = float(rng.uniform(-0.10, 0.10))
    map_y = np.clip(((ny + 1.0) * 0.5 * (h - 1.0)) + pitch * h * (1.0 - np.clip(nx, -1, 1) ** 2), 0, h - 1).astype(np.float32)

    warped = cv2.remap(jersey_rgba, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    shoulder_boost = 0.14 * np.exp(-((ny + 0.55) / 0.38) ** 2)
    waist_lim = 0.94 + shoulder_boost - 0.10 * np.clip(ny, -0.3, 1.0)
    valid = (np.abs(nx) <= waist_lim) & (np.abs(theta) < 1.36) & (np.abs(ny) <= 1.05)
    warped[~valid, 3] = 0

    warped_mask: Optional[np.ndarray] = None
    if stroke_mask is not None:
        warped_mask = cv2.remap(stroke_mask, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_mask[~valid] = 0

    fold_freq = float(rng.uniform(2.0, 5.5))
    fold_phase = float(rng.uniform(0.0, 6.28))
    fold_amp = float(rng.uniform(0.05, 0.16))
    shading = 1.0 - 0.22 * np.clip(nx, -1, 1) ** 2 + fold_amp * np.sin(fold_freq * np.clip(nx, -1, 1) + 1.5 * ny + fold_phase)
    shading = np.clip(shading, 0.60, 1.12)[..., None]
    warped[:, :, :3] = np.clip(warped[:, :, :3].astype(np.float32) * shading, 0, 255).astype(np.uint8)
    return warped, warped_mask


def _apply_occlusion(
    rng: np.random.Generator,
    img: np.ndarray,
    occl_kind: str,
    stroke_mask: Optional[np.ndarray] = None,
) -> None:
    h, w = img.shape[:2]
    occl_bin = np.zeros((h, w), dtype=np.uint8)
    if occl_kind in ("left", "right"):
        side = occl_kind
        mode = int(rng.integers(0, 3))
        if mode == 0:
            glove_col = _jitter_color(rng, (26, 28, 34), 12)
            sleeve_col = _jitter_color(rng, (45, 55, 85), 22)
            x_center = int(w * (0.40 if side == "left" else 0.60))
            x_edge = 0 if side == "left" else w
            pts = np.array([
                [x_edge, int(h * rng.uniform(0.14, 0.30))],
                [x_center + int(w * 0.05 * (1 if side == "left" else -1)), int(h * rng.uniform(0.24, 0.38))],
                [x_center + int(w * 0.04 * (1 if side == "left" else -1)), int(h * rng.uniform(0.70, 0.86))],
                [x_edge, int(h * rng.uniform(0.76, 0.94))],
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], sleeve_col, lineType=cv2.LINE_AA)
            cv2.fillPoly(occl_bin, [pts], 255)
            gx = x_center - int(w * 0.05 * (1 if side == "left" else -1))
            gy = int(h * rng.uniform(0.36, 0.60))
            ang = int(rng.integers(-25, 25))
            cv2.ellipse(img, (gx, gy), (int(w * 0.14), int(h * 0.18)), ang, 0, 360, glove_col, -1, lineType=cv2.LINE_AA)
            cv2.ellipse(occl_bin, (gx, gy), (int(w * 0.14), int(h * 0.18)), ang, 0, 360, 255, -1)
        elif mode == 1:
            col = _jitter_color(rng, (34, 40, 54) if rng.random() < 0.5 else (188, 194, 200), 20)
            col_dark = _jitter_color(rng, (24, 26, 30), 10)
            cx = int(w * (rng.uniform(0.12, 0.20) if side == "left" else rng.uniform(0.80, 0.88)))
            rx = int(w * rng.uniform(0.24, 0.30))
            ry = int(h * rng.uniform(0.46, 0.64))
            cy = int(h * rng.uniform(0.42, 0.58))
            cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, col, -1, lineType=cv2.LINE_AA)
            cv2.ellipse(img, (cx, int(cy - ry * 0.82)), (int(rx * 0.45), int(ry * 0.25)), 0, 0, 360, col_dark, -1, lineType=cv2.LINE_AA)
            cv2.ellipse(occl_bin, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        else:
            stick_col = _jitter_color(rng, (34, 32, 30), 15)
            x0 = int(w * (0.28 if side == "left" else 0.72))
            x1 = int(w * (0.41 if side == "left" else 0.59))
            thick = max(6, int(w * 0.13))
            rad = int(w * 0.15)
            cv2.line(img, (x0, 0), (x1, h), stick_col, thick, lineType=cv2.LINE_AA)
            cv2.circle(img, (int((x0 + x1) * 0.5), int(h * 0.48)), rad, stick_col, -1, lineType=cv2.LINE_AA)
            cv2.line(occl_bin, (x0, 0), (x1, h), 255, thick)
            cv2.circle(occl_bin, (int((x0 + x1) * 0.5), int(h * 0.48)), rad, 255, -1)
    elif occl_kind == "mild":
        stick_col = _jitter_color(rng, (30, 32, 34) if rng.random() < 0.5 else (164, 168, 158), 14)
        # Mild thin hockey stick near edge/diagonal so legible digits stay unobstructed
        if rng.random() < 0.5:
            x0, y0 = int(w * rng.uniform(0.0, 0.28)), int(h * rng.uniform(0.72, 1.0))
            x1, y1 = int(w * rng.uniform(0.68, 1.0)), int(h * rng.uniform(0.0, 0.28))
        else:
            x0, y0 = int(w * rng.uniform(0.72, 1.0)), int(h * rng.uniform(0.72, 1.0))
            x1, y1 = int(w * rng.uniform(0.0, 0.32)), int(h * rng.uniform(0.0, 0.28))
        thick = max(2, int(w * rng.uniform(0.018, 0.032)))
        cv2.line(img, (x0, y0), (x1, y1), stick_col, thick, lineType=cv2.LINE_AA)
        cv2.line(occl_bin, (x0, y0), (x1, y1), 255, thick)

    if stroke_mask is not None and occl_bin.any():
        stroke_mask[occl_bin > 127] = 0


def _apply_plexiglas_artifacts(rng: np.random.Generator, img: np.ndarray, is_legible: bool = False) -> None:
    """Apply 16% translucent rink-glass stanchions, protective netting, or Plexiglas glare streaks."""
    if rng.random() >= 0.16:
        return
    h, w = img.shape[:2]
    mode = int(rng.choice([0, 1, 2], p=[0.42, 0.30, 0.28]))
    overlay = img.astype(np.float32)

    if mode == 0:
        # Translucent vertical rink-glass stanchion seam; keep off dead-center on legible=1
        bar_w = max(4, int(w * rng.uniform(0.04, 0.10)))
        if is_legible:
            cx = int(w * (rng.uniform(0.08, 0.28) if rng.random() < 0.5 else rng.uniform(0.72, 0.92)))
            a_val = float(rng.uniform(0.16, 0.30))
        else:
            cx = int(w * rng.uniform(0.14, 0.86))
            a_val = float(rng.uniform(0.22, 0.42))
        x0, x1 = max(0, cx - bar_w // 2), min(w, cx + bar_w // 2 + 1)
        alpha_map = np.zeros((h, w), dtype=np.float32)
        alpha_map[:, x0:x1] = a_val
        alpha_map = cv2.GaussianBlur(alpha_map, (5, 1), 0)[..., None]
        st_col = np.array(_jitter_color(rng, (145, 152, 160) if rng.random() < 0.55 else (62, 68, 78), 18), dtype=np.float32)
        out = overlay * (1.0 - alpha_map) + st_col[None, None, :] * alpha_map
    elif mode == 1:
        # Translucent protective arena netting mesh
        mesh = np.zeros((h, w), dtype=np.float32)
        step = max(8, int(round(min(h, w) * rng.uniform(0.07, 0.13))))
        phase = int(rng.integers(0, step))
        for k in range(-h, w + h, step):
            cv2.line(mesh, (k + phase, 0), (k + phase + h // 2, h), 1.0, 1, lineType=cv2.LINE_AA)
            cv2.line(mesh, (k + phase + h // 2, 0), (k + phase, h), 1.0, 1, lineType=cv2.LINE_AA)
        alpha_map = (mesh * float(rng.uniform(0.12, 0.24)))[..., None]
        net_col = np.array(_jitter_color(rng, (36, 38, 42) if rng.random() < 0.7 else (175, 180, 186), 14), dtype=np.float32)
        out = overlay * (1.0 - alpha_map) + net_col[None, None, :] * alpha_map
    else:
        # Diagonal / vertical Plexiglas specular glare haze
        glare = np.zeros((h, w), dtype=np.float32)
        gx0 = int(w * rng.uniform(0.05, 0.85))
        gx1 = int(w * rng.uniform(0.15, 0.95))
        cv2.line(glare, (gx0, 0), (gx1, h), 1.0, max(10, int(w * rng.uniform(0.12, 0.26))), lineType=cv2.LINE_AA)
        glare = cv2.GaussianBlur(glare, (15, 15), 0)
        glare_max = max(1e-6, float(glare.max()))
        alpha_map = ((glare / glare_max) * float(rng.uniform(0.14, 0.30)))[..., None]
        gl_col = np.array(_jitter_color(rng, (226, 232, 238), 10), dtype=np.float32)
        out = overlay * (1.0 - alpha_map) + gl_col[None, None, :] * alpha_map

    img[:] = np.clip(out, 0, 255).astype(np.uint8)


def _apply_yuv420_subsampling(rgb: np.ndarray) -> np.ndarray:
    """Simulate H.264 4:2:0 chroma subsampling at native crop resolution."""
    lh, lw = rgb.shape[:2]
    if lw < 4 or lh < 4:
        return rgb
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cw_c, ch_c = max(2, lw // 2), max(2, lh // 2)
    cr = cv2.resize(cv2.resize(ycrcb[..., 1], (cw_c, ch_c), interpolation=cv2.INTER_AREA), (lw, lh), interpolation=cv2.INTER_LINEAR)
    cb = cv2.resize(cv2.resize(ycrcb[..., 2], (cw_c, ch_c), interpolation=cv2.INTER_AREA), (lw, lh), interpolation=cv2.INTER_LINEAR)
    ycrcb[..., 1] = cr
    ycrcb[..., 2] = cb
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _degrade_camera_crop_and_mask(
    rng: np.random.Generator,
    rgb: np.ndarray,
    stroke_mask: Optional[np.ndarray],
    tilt_deg: float,
    is_illegible_blur: bool = False,
    min_short_side: int = 18,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match Android JerseyStrOnnxRecognizer.kt v5 128x128 pipeline and build aligned 32x32 sr_target."""
    h_nat, w_nat = rgb.shape[:2]
    aspect = w_nat / max(1.0, float(h_nat))
    out = rgb.astype(np.float32)

    # 1. Rink color cast (including cool teal-cyan broadcast lighting) + exposure variation
    cast_type = int(rng.choice([0, 1, 2, 3, 4], p=[0.30, 0.28, 0.14, 0.08, 0.20]))
    if cast_type == 1:
        gains = np.array([rng.uniform(0.80, 0.94), rng.uniform(1.02, 1.14), rng.uniform(0.90, 1.04)], dtype=np.float32)
    elif cast_type == 2:
        gains = np.array([rng.uniform(0.88, 0.97), rng.uniform(0.95, 1.03), rng.uniform(1.04, 1.16)], dtype=np.float32)
    elif cast_type == 3:
        gains = np.array([rng.uniform(1.03, 1.14), rng.uniform(0.95, 1.04), rng.uniform(0.84, 0.95)], dtype=np.float32)
    elif cast_type == 4:
        gains = np.array([rng.uniform(0.74, 0.88), rng.uniform(0.90, 1.04), rng.uniform(0.86, 1.00)], dtype=np.float32)
    else:
        gains = np.array([rng.uniform(0.95, 1.05), rng.uniform(0.95, 1.05), rng.uniform(0.95, 1.05)], dtype=np.float32)
    out *= gains[None, None, :]
    out *= float(rng.uniform(0.74, 1.14))
    out += float(rng.uniform(-10.0, 10.0))
    out = np.clip(out, 0, 255).astype(np.uint8)

    # 2. Target short side: 18% sharp high-res downscale (96..154 px), 82% telephoto (lo_sh..62 px)
    is_sharp_hi = False
    if is_illegible_blur:
        short_side = int(rng.integers(12, 24))
    elif rng.random() < 0.18:
        short_side = int(rng.integers(96, 155))
        is_sharp_hi = bool(rng.random() < 0.25)
    else:
        lo_sh = min(58, max(20, int(min_short_side)))
        r = rng.random()
        if r < 0.36:
            short_side = int(rng.integers(lo_sh, max(lo_sh + 4, 36)))
        elif r < 0.76:
            lo2 = max(lo_sh, 32)
            short_side = int(rng.integers(lo2, max(lo2 + 4, 52)))
        else:
            lo3 = max(lo_sh, 48)
            short_side = int(rng.integers(lo3, max(lo3 + 4, 63)))
        short_side = min(62, short_side)

    # 3. Directional skating motion blur (avoid stacking harsh blur onto <30px telephoto crops or sharp high-res)
    if is_illegible_blur:
        do_blur, k = True, int(rng.integers(5, 11))
    elif is_sharp_hi:
        do_blur, k = False, 2
    elif short_side < 30:
        do_blur, k = (rng.random() < 0.16), 2
    else:
        do_blur, k = (rng.random() < 0.42), int(rng.integers(2, 6))
    if do_blur:
        ang = float(rng.uniform(-40.0, 40.0))
        kern = np.zeros((k, k), dtype=np.float32)
        kern[k // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), ang, 1.0)
        kern = cv2.warpAffine(kern, M, (k, k))
        s = kern.sum()
        if s > 1e-6:
            kern /= s
            out = cv2.filter2D(out, -1, kern)

    if aspect >= 1.0:
        sh = short_side
        sw = max(12, int(round(short_side * aspect)))
    else:
        sw = short_side
        sh = max(12, int(round(short_side / aspect)))
    low = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)

    # 4. H.264 4:2:0 chroma subsampling + JPEG/codec ringing + soft optical/H.264 blur or sensor noise
    is_soft_telephoto = (not is_illegible_blur) and (not is_sharp_hi) and (short_side >= 36) and (rng.random() < 0.54)
    if not is_sharp_hi and rng.random() < 0.82:
        low = _apply_yuv420_subsampling(low)
    if not is_sharp_hi and rng.random() < 0.82:
        q = int(rng.integers(40, 92))
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(low, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                low = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    if not is_soft_telephoto and rng.random() < 0.22:
        blur_l = cv2.GaussianBlur(low, (3, 3), 0.7)
        amt = float(rng.uniform(0.15, 0.35))
        low = np.clip(low.astype(np.float32) * (1.0 + amt) - blur_l.astype(np.float32) * amt, 0, 255).astype(np.uint8)
    if not is_sharp_hi and not is_soft_telephoto and rng.random() < 0.45:
        sigma = float(rng.uniform(1.0, 4.0))
        low = np.clip(low.astype(np.float32) + rng.normal(0.0, sigma, size=low.shape).astype(np.float32), 0, 255).astype(np.uint8)

    # 5. Aspect-preserving letterbox to 128x128: mix 50% INTER_LINEAR + 50% INTER_CUBIC on upscales (and AREA/LINEAR/CUBIC on high-res downscales)
    scale = min(float(TARGET_SIZE) / max(1, sw), float(TARGET_SIZE) / max(1, sh))
    nw = max(1, min(TARGET_SIZE, int(round(sw * scale))))
    nh = max(1, min(TARGET_SIZE, int(round(sh * scale))))
    pad_x, pad_y = (TARGET_SIZE - nw) // 2, (TARGET_SIZE - nh) // 2
    if sw > nw or sh > nh:
        interp = int(rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC], p=[0.50, 0.25, 0.25]))
    else:
        interp = cv2.INTER_LINEAR if rng.random() < 0.50 else cv2.INTER_CUBIC
    resized = cv2.resize(low, (nw, nh), interpolation=interp).astype(np.float32)
    if is_soft_telephoto:
        # Realistic H.264 deblocking + distant lens softness & Plexiglas haze (interior Laplacian S ~ 4..20)
        sig_s = float(rng.uniform(1.15, 2.20)) if nw < 96 else float(rng.uniform(1.25, 2.35))
        k_s = 7 if sig_s >= 1.65 else 5
        resized = cv2.GaussianBlur(resized, (k_s, k_s), sigmaX=sig_s, sigmaY=sig_s)
        if rng.random() < 0.62:
            haze_c = float(rng.uniform(0.54, 0.86))
            m_v = float(resized.mean())
            resized = np.clip((resized - m_v) * haze_c + m_v, 0.0, 255.0)

    # Simulate 2nd-pass derotation PAD_GREY corner triangles on 18% of near-upright samples
    if abs(tilt_deg) < 14.0 and rng.random() < 0.18:
        yc = np.abs(np.linspace(-1.0, 1.0, nh, dtype=np.float32))[:, None]
        xc = np.abs(np.linspace(-1.0, 1.0, nw, dtype=np.float32))[None, :]
        resized[(xc + yc) > float(rng.uniform(1.38, 1.66))] = float(LETTERBOX_PAD)

    luma = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
    p_low, p_high = np.percentile(luma, [2.0, 98.0])
    raw_span = max(32.0, float(p_high - p_low))
    gain = min(max(1.0 + 0.12 * min(max(220.0 / raw_span, 0.8), 2.0), 1.05), 1.25)
    mid = 0.5 * float(p_low + p_high)
    resized = np.clip((resized - mid) * gain + mid, 0.0, 255.0)

    pad_val = LETTERBOX_PAD if rng.random() < 0.96 else int(rng.choice([0, 96, 128]))
    canvas = np.full((TARGET_SIZE, TARGET_SIZE, 3), pad_val, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = np.clip(resized, 0, 255).astype(np.uint8)

    # 6. Build aligned 32x32 clean stroke mask in 128x128 crop space (0 on hard negatives)
    if stroke_mask is None:
        sr_target = np.zeros((SR_SIZE, SR_SIZE), dtype=np.uint8)
    else:
        mask_full = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
        mask_resized = cv2.resize(stroke_mask, (nw, nh), interpolation=cv2.INTER_AREA)
        mask_full[pad_y:pad_y + nh, pad_x:pad_x + nw] = mask_resized
        sr_target = cv2.resize(mask_full, (SR_SIZE, SR_SIZE), interpolation=cv2.INTER_AREA)

    return canvas, sr_target


def _apply_sleeve_shadow_folds(rng: np.random.Generator, jersey: np.ndarray, yoke_col: Tuple[int, int, int]) -> None:
    """Inject realistic lateral curved sleeve folds & armpit shadows next to outer digits ('7','1','4','9')."""
    if rng.random() >= 0.52:
        return
    h, w = jersey.shape[:2]
    fold_mask = np.zeros((h, w), dtype=np.float32)
    shadow_col = _jitter_color(rng, yoke_col if rng.random() < 0.40 else (28, 34, 46), 14)
    sides = ["right"] if rng.random() < 0.55 else (["left"] if rng.random() < 0.5 else ["left", "right"])
    for side in sides:
        thick = int(w * rng.uniform(0.05, 0.11))
        x_edge = int(w * rng.uniform(0.82, 0.96)) if side == "right" else int(w * rng.uniform(0.04, 0.18))
        x_mid = int(w * rng.uniform(0.72, 0.86)) if side == "right" else int(w * rng.uniform(0.14, 0.28))
        y0 = int(h * rng.uniform(0.24, 0.42))
        y1 = int(h * rng.uniform(0.46, 0.64))
        y2 = int(h * rng.uniform(0.68, 0.92))
        pts = np.array([[x_edge, y0], [x_mid, y1], [x_edge, y2]], dtype=np.int32)
        cv2.polylines(fold_mask, [pts], False, 1.0, max(3, thick), lineType=cv2.LINE_AA)
    k = int(rng.choice([7, 9, 11, 13]))
    fold_mask = cv2.GaussianBlur(fold_mask, (k, k), 0)
    alpha = np.clip(fold_mask * float(rng.uniform(0.20, 0.48)), 0.0, 0.52)[..., None]
    sc = np.array(shadow_col, dtype=np.float32)[None, None, :]
    jersey[:] = np.clip(jersey.astype(np.float32) * (1.0 - alpha) + sc * alpha, 0, 255).astype(np.uint8)


def _sample_tilt_deg(rng: np.random.Generator) -> float:
    """Sample skating stride tilt: 50% in [-18, +18], 50% in [+/-18, +/-52] deg (+ = CCW)."""
    if rng.random() < 0.50:
        return float(rng.uniform(-18.0, 18.0))
    sign = -1.0 if rng.random() < 0.50 else 1.0
    return float(sign * rng.uniform(18.0, 52.0))


def generate_sample(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float, int, int, int, int, int, int]:
    """Generate one synthetic 128x128 hockey torso crop and multi-task labels."""
    H = 160
    TOP_EXT = 40
    BOT_EXT = 40
    H_FULL = H + TOP_EXT + BOT_EXT
    # Torso-centered crop aspect ratios matching y: 4%..58% stick-immune crop geometry (real ~0.72..1.42)
    r_asp = float(rng.random())
    if r_asp < 0.36:
        crop_aspect = float(rng.uniform(0.72, 0.95))
    elif r_asp < 0.88:
        crop_aspect = float(rng.uniform(0.95, 1.26))
    else:
        crop_aspect = float(rng.uniform(1.26, 1.44))
    W = max(112, int(round(H * crop_aspect)))

    # 54% legible positive, 14% partial occlusion, 32% hard negative
    u = float(rng.random())
    if u < 0.54:
        cat = "legible"
    elif u < 0.68:
        cat = "partial"
    elif u < 0.75:
        cat = "ref_stripes"
    elif u < 0.81:
        cat = "front_crest"
    elif u < 0.88:
        cat = "double_body_overlap"
    elif u < 0.95:
        cat = "blank_or_blur"
    else:
        cat = "rink_clutter"

    bg = _make_rink_background(rng, W, H)

    if cat == "rink_clutter":
        if rng.random() < 0.55:
            txt = str(int(rng.integers(1, 99))) if rng.random() < 0.4 else rng.choice(["ICE", "HOCKEY", "PRO", "2026", "RINK", "A", "C"])
            cv2.putText(
                bg, txt,
                (int(rng.integers(8, max(12, W - 48))), int(rng.integers(28, H - 20))),
                cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.55, 1.35)),
                tuple(int(v) for v in rng.integers(35, 195, size=3)),
                int(rng.integers(2, 4)), cv2.LINE_AA,
            )
        _apply_plexiglas_artifacts(rng, bg)
        img_128, sr_32 = _degrade_camera_crop_and_mask(rng, bg, None, 0.0, is_illegible_blur=False)
        return img_128, sr_32, np.float32(0.0), 0, DIGIT_BLANK, DIGIT_BLANK, 0, 0, PARTIAL_NONE

    # 28% of positives (legible + partial) use low-luma chromatic twill on dark jerseys
    is_pos = cat in ("legible", "partial")
    force_twill = is_pos and (rng.random() < 0.28)
    body_col, yoke_col, digit_col, outline_col, is_twill = _pick_jersey_colors(rng, force_low_luma_twill=force_twill)

    J_W = 160
    jersey = np.empty((H_FULL, J_W, 3), dtype=np.uint8)
    jersey[:, :] = body_col

    # Overhead camera pitch foreshortening: v_scale in [0.54, 0.96] on 75% of crops
    v_scale = float(rng.uniform(0.54, 0.96)) if rng.random() < 0.75 else 1.0
    tilt_deg = _sample_tilt_deg(rng)

    if cat != "ref_stripes" and rng.random() < 0.74:
        yoke_style = int(rng.choice([0, 1, 2], p=[0.30, 0.55, 0.15]))
        yoke_bottom = TOP_EXT + int(H * rng.uniform(0.12, 0.26) * (0.65 + 0.35 * v_scale))
        trim_h = max(2, int(H * rng.uniform(0.022, 0.040)))
        if yoke_style == 0:
            jersey[:yoke_bottom, :] = yoke_col
            if rng.random() < 0.60:
                jersey[yoke_bottom:yoke_bottom + trim_h, :] = outline_col
        elif yoke_style == 1:
            nx_j = (np.arange(J_W, dtype=np.float32) - J_W * 0.5) / (J_W * 0.5)
            arc_amp = float(H * rng.uniform(0.05, 0.13) * v_scale)
            if rng.random() < 0.55:
                y_curve = (yoke_bottom + arc_amp * (nx_j ** 2)).astype(np.int32)
            else:
                y_curve = (yoke_bottom + arc_amp * (1.0 - nx_j ** 2)).astype(np.int32)
            ys_col = np.arange(H_FULL, dtype=np.int32)[:, None]
            y_c = np.clip(y_curve[None, :], TOP_EXT + 4, TOP_EXT + int(H * 0.40))
            jersey[ys_col < y_c] = yoke_col
            if rng.random() < 0.65:
                jersey[(ys_col >= y_c) & (ys_col < y_c + trim_h)] = outline_col
        else:
            jersey[:yoke_bottom, :] = yoke_col
            sw_side = int(J_W * rng.uniform(0.14, 0.24))
            sh_end = TOP_EXT + int(H * rng.uniform(0.30, 0.46))
            jersey[:sh_end, :sw_side] = yoke_col
            jersey[:sh_end, J_W - sw_side:] = yoke_col
        if rng.random() < 0.50:
            cv2.ellipse(jersey, (J_W // 2, TOP_EXT), (int(J_W * 0.18), int(H * 0.08)), 0, 0, 180, outline_col, max(2, int(H * 0.022)), lineType=cv2.LINE_AA)

    if rng.random() < 0.72:
        hem_y = TOP_EXT + int(H * rng.uniform(0.72, 0.88))
        band_h = max(3, int(H * rng.uniform(0.035, 0.060)))
        stripe_c1 = outline_col if cat != "ref_stripes" else (24, 24, 28)
        stripe_c2 = yoke_col if cat != "ref_stripes" else (200, 204, 210)
        jersey[hem_y:min(H_FULL, hem_y + band_h), :] = stripe_c1
        if rng.random() < 0.65:
            gap_h = max(2, int(H * 0.022))
            band2_h = max(2, int(rng.uniform(0.026, 0.048) * H))
            y2 = hem_y - gap_h - band2_h
            if y2 > TOP_EXT + int(H * 0.66):
                jersey[y2:y2 + band2_h, :] = stripe_c2
            elif hem_y + band_h + gap_h + band2_h < H_FULL:
                jersey[hem_y + band_h + gap_h:hem_y + band_h + gap_h + band2_h, :] = stripe_c2
    if rng.random() < 0.78:
        pants_y = TOP_EXT + int(H * rng.uniform(0.84, 0.96))
        jersey[pants_y:, :] = _jitter_color(rng, (24, 26, 30), 10)

    # Extended helmet / neck region above TOP_EXT so tilted crops never expose cut-off top corners
    if rng.random() < 0.74:
        helm_y = TOP_EXT + int(H * rng.uniform(0.02, 0.11))
        helm_col = _jitter_color(rng, (22, 24, 30) if rng.random() < 0.65 else yoke_col, 14)
        jersey[:helm_y, int(J_W * 0.24):int(J_W * 0.76)] = helm_col
    elif rng.random() < 0.50:
        jersey[:TOP_EXT, :] = yoke_col

    torso_w = float(rng.uniform(0.60, 0.92))
    y_shift = float(rng.uniform(-0.04, 0.06))
    use_framed_skater = bool(rng.random() < 0.45)
    body_v_scale = float(rng.uniform(0.76, 0.94)) if use_framed_skater else 1.0

    def _draw_nameplate(target_j: np.ndarray, y_num_top: int) -> None:
        if y_num_top < TOP_EXT + 12:
            return
        nb_len = int(rng.integers(5, 9))
        nb_txt = "".join(chr(int(rng.integers(65, 91))) for _ in range(nb_len))
        gap_px = int(rng.integers(1, 5))
        y_base = max(TOP_EXT + 10, y_num_top - gap_px)
        name_col = digit_col if abs(_luma(digit_col) - _luma(body_col)) > 35 else outline_col
        if rng.random() < 0.55:
            bar_h = max(7, int(H * rng.uniform(0.065, 0.110) * v_scale))
            by0 = max(TOP_EXT + 2, y_base - bar_h)
            bar_col = yoke_col if abs(_luma(yoke_col) - _luma(body_col)) > 18 else _jitter_color(rng, body_col, 24)
            target_j[by0:y_base + 1, int(J_W * 0.16):int(J_W * 0.84)] = bar_col
            txt_col = _jitter_color(rng, (214, 218, 222), 12) if _luma(bar_col) < 120 else _jitter_color(rng, (26, 30, 38), 10)
        else:
            txt_col = name_col
        f_scale = float(rng.uniform(0.32, 0.52)) * (0.75 + 0.25 * v_scale)
        f_thick = 1 if rng.random() < 0.55 else 2
        font_id = cv2.FONT_HERSHEY_SIMPLEX if rng.random() < 0.65 else cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            target_j, nb_txt,
            (int(J_W * rng.uniform(0.20, 0.30)), y_base),
            font_id, f_scale, txt_col, f_thick, cv2.LINE_AA,
        )

    def _composite_tilted_torso(
        j_arr: np.ndarray,
        yaw_d: float,
        t_w: float,
        t_deg: float,
        x_bias: float = 0.0,
        y_bias: float = 0.0,
        base_bg: Optional[np.ndarray] = None,
        mask_arr: Optional[np.ndarray] = None,
        framed: Optional[bool] = None,
        b_scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        is_framed = use_framed_skater if framed is None else framed
        cur_v_scale = (body_v_scale if is_framed else 1.0) if b_scale is None else b_scale
        alpha_init = np.full((H_FULL, J_W), 255, dtype=np.uint8)
        j_work = j_arr.copy()
        if is_framed:
            # Carve realistic upper-body helmet dome + sloping shoulders exposing rink background above/around shoulders
            xs_idx = np.arange(J_W, dtype=np.float32)
            nx_s = np.abs(xs_idx - J_W * 0.5) / (J_W * 0.5)
            shoulder_y = (TOP_EXT + int(H * 0.06) + int(H * 0.12) * (nx_s ** 1.4)).astype(np.int32)
            ys_idx = np.arange(H_FULL, dtype=np.int32)[:, None]
            alpha_init[ys_idx < shoulder_y[None, :]] = 0
            # Draw distinct helmet + neck centered at top of shoulders
            h_col = _jitter_color(rng, (22, 24, 30) if rng.random() < 0.68 else (208, 212, 218), 14)
            hx = int(J_W * 0.5 + np.clip(yaw_d / 60.0, -0.5, 0.5) * J_W * 0.14)
            hy = TOP_EXT + int(H * 0.02)
            hrx, hry = int(J_W * rng.uniform(0.15, 0.19)), int(H * rng.uniform(0.10, 0.13))
            cv2.ellipse(j_work, (hx, hy), (hrx, hry), 0, 0, 360, h_col, -1, lineType=cv2.LINE_AA)
            cv2.ellipse(alpha_init, (hx, hy), (hrx, hry), 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)

        rgba = np.dstack([j_work, alpha_init])
        warped_sq, warped_m = _apply_cylindrical_warp_and_folds(rng, rgba, yaw_d, t_w, y_shift, stroke_mask=mask_arr)

        out_comp = (bg if base_bg is None else base_bg).copy()
        if is_framed:
            place_w = min(W - 4, max(72, int(round(H * rng.uniform(0.68, 0.92)))))
            place_h = max(140, int(round(H_FULL * cur_v_scale)))
            eff_x_bias = x_bias + float(rng.uniform(-0.10, 0.10))
            eff_y_off = int(round((y_bias + float(rng.uniform(-0.02, 0.08))) * H))
        else:
            place_w = min(W, max(94, int(round(H * rng.uniform(0.84, 1.12)))))
            place_h = H_FULL
            eff_x_bias = x_bias + float(rng.uniform(-0.05, 0.05))
            eff_y_off = int(round(y_bias * H))
        warped_resized = cv2.resize(warped_sq, (place_w, place_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(warped_m, (place_w, place_h), interpolation=cv2.INTER_LINEAR) if warped_m is not None else None

        pad_m = 64
        can_h, can_w = H_FULL + 2 * pad_m, W + 2 * pad_m
        tmp_rgba = np.zeros((can_h, can_w, 4), dtype=np.uint8)
        tmp_mask = np.zeros((can_h, can_w), dtype=np.uint8) if mask_resized is not None else None

        x_off = int(round((W - place_w) * 0.5 + eff_x_bias * W))
        y_off = eff_y_off + int(round(TOP_EXT * (1.0 - cur_v_scale)))
        dx_c = pad_m + x_off
        dy_c = pad_m + y_off

        # Draw slanted skating sleeves/gloves BEHIND torso anchored cleanly inside shoulders on 68% of framed silhouettes
        if is_framed and rng.random() < 0.68:
            arm_col = _jitter_color(rng, body_col if rng.random() < 0.65 else yoke_col, 8)
            glove_col = _jitter_color(rng, (26, 28, 34), 10)
            sh_y = dy_c + int(round((TOP_EXT + H * rng.uniform(0.16, 0.24)) * cur_v_scale))
            for sgn in (-1, 1):
                if rng.random() < 0.20:
                    continue
                ax0 = dx_c + (int(place_w * 0.34) if sgn < 0 else int(place_w * 0.66))
                ax1 = dx_c + (int(place_w * 0.16) if sgn < 0 else int(place_w * 0.84)) + sgn * int(place_w * rng.uniform(0.14, 0.32))
                ay1 = sh_y + int(place_h * rng.uniform(0.16, 0.32))
                a_th = max(6, int(place_w * rng.uniform(0.14, 0.20)))
                cv2.line(tmp_rgba, (ax0, sh_y), (ax1, ay1), arm_col + (255,), a_th, lineType=cv2.LINE_AA)
                cv2.circle(tmp_rgba, (ax1, ay1), max(4, a_th // 2), glove_col + (255,), -1, lineType=cv2.LINE_AA)

        sx0 = max(0, -dx_c)
        sy0 = max(0, -dy_c)
        dx0 = max(0, dx_c)
        dy0 = max(0, dy_c)
        copy_w = min(place_w - sx0, can_w - dx0)
        copy_h = min(place_h - sy0, can_h - dy0)
        if copy_w > 0 and copy_h > 0:
            src_patch = warped_resized[sy0:sy0 + copy_h, sx0:sx0 + copy_w]
            dst_patch = tmp_rgba[dy0:dy0 + copy_h, dx0:dx0 + copy_w]
            pa_src = src_patch[:, :, 3:4].astype(np.float32) / 255.0
            dst_patch[:, :, :3] = np.clip(src_patch[:, :, :3].astype(np.float32) * pa_src + dst_patch[:, :, :3].astype(np.float32) * (1.0 - pa_src), 0, 255).astype(np.uint8)
            dst_patch[:, :, 3] = np.maximum(dst_patch[:, :, 3], src_patch[:, :, 3])
            if tmp_mask is not None and mask_resized is not None:
                tmp_mask[dy0:dy0 + copy_h, dx0:dx0 + copy_w] = mask_resized[sy0:sy0 + copy_h, sx0:sx0 + copy_w]

        # Canonical rotation sign: +t_deg => CCW in OpenCV image space around torso/numeral center
        if abs(t_deg) > 0.05:
            center = (pad_m + W * 0.5 + eff_x_bias * W, pad_m + TOP_EXT + H * 0.42 + y_bias * H)
            M = cv2.getRotationMatrix2D(center, float(t_deg), 1.0)
            tmp_rgba = cv2.warpAffine(tmp_rgba, M, (can_w, can_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            if tmp_mask is not None:
                tmp_mask = cv2.warpAffine(tmp_mask, M, (can_w, can_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        crop_rgba = tmp_rgba[pad_m + TOP_EXT:pad_m + TOP_EXT + H, pad_m:pad_m + W]
        a = crop_rgba[:, :, 3:4].astype(np.float32) / 255.0
        out_comp = np.clip(crop_rgba[:, :, :3].astype(np.float32) * a + out_comp.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)
        crop_mask = tmp_mask[pad_m + TOP_EXT:pad_m + TOP_EXT + H, pad_m:pad_m + W] if tmp_mask is not None else None
        return out_comp, crop_mask

    if cat == "double_body_overlap":
        is_back = int(rng.random() < 0.50)
        shift_side = -0.22 if rng.random() < 0.5 else 0.22
        if rng.random() < 0.65:
            num_idx = int(rng.integers(10, 100))
            ph_d = int(H * rng.uniform(0.28, 0.44))
            patch, _ = _render_jersey_number_patch(rng, str(num_idx), J_W, ph_d, digit_col, outline_col, body_col, v_scale=v_scale)
            py0 = TOP_EXT + int(H * rng.uniform(0.14, 0.36))
            py0 = min(TOP_EXT + H - ph_d - 4, max(TOP_EXT + 6, py0))
            pa = patch[:, :, 3:4].astype(np.float32) / 255.0
            jersey[py0:py0 + ph_d, :] = np.clip(patch[:, :, :3].astype(np.float32) * pa + jersey[py0:py0 + ph_d, :].astype(np.float32) * (1.0 - pa), 0, 255).astype(np.uint8)
        _apply_sleeve_shadow_folds(rng, jersey, yoke_col)
        comp1, _ = _composite_tilted_torso(jersey, float(rng.uniform(-42.0, 42.0)), torso_w, tilt_deg * 0.6, x_bias=shift_side)
        b2, y2, d2, o2, _ = _pick_jersey_colors(rng)
        j2 = np.empty((H_FULL, J_W, 3), dtype=np.uint8)
        j2[:, :] = b2
        j2[:TOP_EXT + int(H * rng.uniform(0.16, 0.32)), :] = y2
        j2[TOP_EXT + int(H * 0.84):, :] = _jitter_color(rng, (24, 26, 30), 10)
        if rng.random() < 0.55:
            p2h = int(H * 0.36)
            p2, _ = _render_jersey_number_patch(rng, str(int(rng.integers(1, 100))), J_W, p2h, d2, o2, b2, v_scale=v_scale)
            p2y = TOP_EXT + int(H * 0.18)
            pa2 = p2[:, :, 3:4].astype(np.float32) / 255.0
            j2[p2y:p2y + p2h, :] = np.clip(p2[:, :, :3].astype(np.float32) * pa2 + j2[p2y:p2y + p2h, :].astype(np.float32) * (1.0 - pa2), 0, 255).astype(np.uint8)
        comp2, _ = _composite_tilted_torso(j2, float(rng.uniform(-48.0, 48.0)), torso_w, _sample_tilt_deg(rng) * 0.6, x_bias=-shift_side, base_bg=comp1)
        _apply_plexiglas_artifacts(rng, comp2)
        img_128, sr_32 = _degrade_camera_crop_and_mask(rng, comp2, None, tilt_deg, is_illegible_blur=bool(rng.random() < 0.35))
        return img_128, sr_32, np.float32(tilt_deg), 0, DIGIT_BLANK, DIGIT_BLANK, 0, is_back, PARTIAL_NONE

    if cat == "ref_stripes":
        _draw_referee_jersey(rng, jersey, J_W, H_FULL)
        is_back = int(rng.random() < 0.55)
        comp, _ = _composite_tilted_torso(jersey, float(rng.uniform(-65.0, 65.0)), torso_w, tilt_deg)
        _apply_plexiglas_artifacts(rng, comp)
        img_128, sr_32 = _degrade_camera_crop_and_mask(rng, comp, None, tilt_deg, is_illegible_blur=False)
        return img_128, sr_32, np.float32(tilt_deg), 0, DIGIT_BLANK, DIGIT_BLANK, 0, is_back, PARTIAL_NONE

    if cat == "front_crest":
        _draw_front_crest(rng, jersey[TOP_EXT:TOP_EXT + H], body_col, yoke_col, outline_col)
        comp, _ = _composite_tilted_torso(jersey, float(rng.uniform(-35.0, 35.0)), torso_w, tilt_deg)
        _apply_plexiglas_artifacts(rng, comp)
        img_128, sr_32 = _degrade_camera_crop_and_mask(rng, comp, None, tilt_deg, is_illegible_blur=False)
        return img_128, sr_32, np.float32(tilt_deg), 0, DIGIT_BLANK, DIGIT_BLANK, 0, 0, PARTIAL_NONE

    if cat == "blank_or_blur":
        is_back = int(rng.random() < 0.50)
        illegible_blur = rng.random() < 0.65
        is_side_num = bool(rng.random() < 0.35)
        if illegible_blur or is_side_num:
            num_idx = int(rng.integers(1, 101))
            num_str = "00" if num_idx == 100 else str(num_idx)
            patch_h = max(24, int(H * rng.uniform(0.22, 0.42) * v_scale))
            patch, _ = _render_jersey_number_patch(rng, num_str, J_W, patch_h, digit_col, outline_col, body_col, v_scale=v_scale)
            py0 = min(TOP_EXT + H - patch_h - 4, max(TOP_EXT + 6, TOP_EXT + int(H * rng.uniform(0.12, 0.36))))
            pa = patch[:, :, 3:4].astype(np.float32) / 255.0
            jersey[py0:py0 + patch_h, :] = np.clip(
                patch[:, :, :3].astype(np.float32) * pa + jersey[py0:py0 + patch_h, :].astype(np.float32) * (1.0 - pa),
                0, 255,
            ).astype(np.uint8)
            if is_back and rng.random() < 0.70:
                _draw_nameplate(jersey, py0)
            yaw_deg = float(rng.choice([-1.0, 1.0]) * rng.uniform(62.0, 82.0)) if (is_side_num or rng.random() < 0.55) else float(rng.uniform(-28.0, 28.0))
            illegible_blur = not is_side_num or (rng.random() < 0.55)
        else:
            yaw_deg = float(rng.uniform(-45.0, 45.0))
        comp, _ = _composite_tilted_torso(jersey, yaw_deg, torso_w, tilt_deg)
        _apply_plexiglas_artifacts(rng, comp)
        img_128, sr_32 = _degrade_camera_crop_and_mask(rng, comp, None, tilt_deg, is_illegible_blur=illegible_blur)
        return img_128, sr_32, np.float32(tilt_deg), 0, DIGIT_BLANK, DIGIT_BLANK, 0, is_back, PARTIAL_NONE

    def _nyquist_min_sh(p_h: int, is_single: bool = False) -> int:
        # Enforce physical Nyquist floor >= 7.5 px digit height scaled for framed skater bodies
        eff_p_h = p_h * body_v_scale
        dh_frac = max(0.13, (eff_p_h / float(H)) * 0.62)
        min_sh = int(math.ceil(7.5 / dh_frac))
        if crop_aspect < 1.0:
            min_sh = int(math.ceil(min_sh * crop_aspect))
        floor_sh = 22 if (is_single and p_h >= int(0.42 * H)) else 30
        return max(floor_sh, min(58, min_sh))

    jersey_mask = np.zeros((H_FULL, J_W), dtype=np.uint8)

    if cat == "partial":
        p_choice = int(rng.integers(10, 101))
        num_str = "00" if p_choice == 100 else str(p_choice)
        hide_left = bool(rng.random() < 0.5)
        is_back = 1 if rng.random() < 0.85 else 0
        patch_h_raw = int(H * rng.uniform(0.38, 0.58))
        patch_eff_v = max(0.76, v_scale) if use_framed_skater else v_scale
        patch_h = max(48 if use_framed_skater else 36, int(round(patch_h_raw * patch_eff_v)))
        patch, fg_patch = _render_jersey_number_patch(
            rng, num_str, J_W, patch_h, digit_col, outline_col, body_col,
            x_shift_frac=float(rng.uniform(-0.08, 0.08)), force_outline=is_twill, v_scale=patch_eff_v,
        )
        py0_rel = min(H - patch_h - 10, max(14, int(H * rng.uniform(0.12, 0.36))))
        py0 = TOP_EXT + py0_rel
        pa = patch[:, :, 3:4].astype(np.float32) / 255.0
        jersey[py0:py0 + patch_h, :] = np.clip(
            patch[:, :, :3].astype(np.float32) * pa + jersey[py0:py0 + patch_h, :].astype(np.float32) * (1.0 - pa),
            0, 255,
        ).astype(np.uint8)
        jersey_mask[py0:py0 + patch_h, :] = fg_patch
        if is_back and rng.random() < 0.72:
            _draw_nameplate(jersey, py0)
        # Apply occlusion in torso space on central torso band before tilt rotation
        _apply_occlusion(rng, jersey[TOP_EXT:TOP_EXT + H], "left" if hide_left else "right", stroke_mask=jersey_mask[TOP_EXT:TOP_EXT + H])
        comp, comp_mask = _composite_tilted_torso(
            jersey, float(rng.uniform(-24.0, 24.0)), max(0.66, torso_w), tilt_deg,
            y_bias=0.04 if py0_rel < int(0.14 * H) else 0.0, mask_arr=jersey_mask,
        )
        _apply_plexiglas_artifacts(rng, comp, is_legible=True)
        img_128, sr_32 = _degrade_camera_crop_and_mask(rng, comp, comp_mask, tilt_deg, False, min_short_side=_nyquist_min_sh(patch_h, is_single=False))
        d0, d1 = int(num_str[0]), int(num_str[1])
        if hide_left:
            return img_128, sr_32, np.float32(tilt_deg), p_choice, DIGIT_BLANK, d1, 0, is_back, PARTIAL_RIGHT_ONLY
        return img_128, sr_32, np.float32(tilt_deg), p_choice, d0, DIGIT_BLANK, 0, is_back, PARTIAL_LEFT_ONLY

    # Legible full number sample: strictly uniform 1..99 + "00" (1/100 each)
    num_cls = int(rng.integers(1, 101))
    num_str = "00" if num_cls == 100 else str(num_cls)
    is_single = len(num_str) == 1
    is_back = 1 if rng.random() < 0.88 else 0
    patch_h_raw = int(H * (rng.uniform(0.40, 0.60) if is_back else rng.uniform(0.36, 0.54)))
    patch_eff_v = max(0.76, v_scale) if use_framed_skater else v_scale
    patch_h = max(48 if use_framed_skater else 36, int(round(patch_h_raw * patch_eff_v)))
    x_shift = float(rng.uniform(-0.13, 0.13))
    patch, fg_patch = _render_jersey_number_patch(
        rng, num_str, J_W, patch_h, digit_col, outline_col, body_col,
        x_shift_frac=x_shift, force_outline=is_twill, v_scale=patch_eff_v,
    )
    py0_rel = min(H - patch_h - 12, max(16, int(H * (rng.uniform(0.12, 0.36) if is_back else rng.uniform(0.14, 0.38)))))
    py0 = TOP_EXT + py0_rel
    pa = patch[:, :, 3:4].astype(np.float32) / 255.0
    jersey[py0:py0 + patch_h, :] = np.clip(
        patch[:, :, :3].astype(np.float32) * pa + jersey[py0:py0 + patch_h, :].astype(np.float32) * (1.0 - pa),
        0, 255,
    ).astype(np.uint8)
    jersey_mask[py0:py0 + patch_h, :] = fg_patch

    if is_back and rng.random() < 0.85:
        _draw_nameplate(jersey, py0)
    if rng.random() < 0.28 and (py0 + patch_h + 10) < TOP_EXT + H:
        f_y0 = py0 + patch_h + int(rng.integers(3, 8))
        f_h = int(rng.integers(3, 8))
        f_col = _jitter_color(rng, (24, 28, 36) if _luma(body_col) > 100 else (18, 20, 26), 8)
        f_alpha = float(rng.uniform(0.28, 0.55))
        band = jersey[f_y0:min(H_FULL, f_y0 + f_h), :]
        jersey[f_y0:min(H_FULL, f_y0 + f_h), :] = np.clip(band.astype(np.float32) * (1.0 - f_alpha) + np.array(f_col, dtype=np.float32) * f_alpha, 0, 255).astype(np.uint8)
    _apply_sleeve_shadow_folds(rng, jersey, yoke_col)

    y_bias = 0.05 if (py0_rel < int(0.16 * H) and abs(tilt_deg) > 25.0) else 0.0
    is_flatter = rng.random() < 0.08
    yaw_cur = float(rng.uniform(-8.0, 8.0)) if is_flatter else float(rng.uniform(-38.0, 38.0))
    torso_cur = float(rng.uniform(0.90, 1.04)) if is_flatter else torso_w
    comp, comp_mask = _composite_tilted_torso(
        jersey, yaw_cur, torso_cur, tilt_deg, y_bias=y_bias, mask_arr=jersey_mask,
    )
    if rng.random() < 0.20:
        _apply_occlusion(rng, comp, "mild", stroke_mask=comp_mask)
    _apply_plexiglas_artifacts(rng, comp, is_legible=True)
    img_128, sr_32 = _degrade_camera_crop_and_mask(rng, comp, comp_mask, tilt_deg, False, min_short_side=_nyquist_min_sh(patch_h, is_single=is_single))

    if is_single:
        return img_128, sr_32, np.float32(tilt_deg), num_cls, int(num_str[0]), DIGIT_BLANK, 1, is_back, PARTIAL_NONE
    return img_128, sr_32, np.float32(tilt_deg), num_cls, int(num_str[0]), int(num_str[1]), 1, is_back, PARTIAL_NONE


def _worker_chunk(args: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed, start_idx, count = args
    rng = np.random.default_rng(seed + start_idx * 9973)
    imgs = np.empty((count, TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    sr_t = np.empty((count, SR_SIZE, SR_SIZE), dtype=np.uint8)
    tilt = np.empty((count,), dtype=np.float32)
    num_cls = np.empty((count,), dtype=np.int16)
    left_d = np.empty((count,), dtype=np.int16)
    right_d = np.empty((count,), dtype=np.int16)
    legible = np.empty((count,), dtype=np.uint8)
    is_back = np.empty((count,), dtype=np.uint8)
    partial_f = np.empty((count,), dtype=np.uint8)
    for i in range(count):
        im, sr, td, nc, ld, rd, lg, bk, pf = generate_sample(rng)
        imgs[i] = im
        sr_t[i] = sr
        tilt[i] = td
        num_cls[i] = nc
        left_d[i] = ld
        right_d[i] = rd
        legible[i] = lg
        is_back[i] = bk
        partial_f[i] = pf
    return imgs, sr_t, tilt, num_cls, left_d, right_d, legible, is_back, partial_f


def generate_dataset(out_dir: Path, n_samples: int, seed: int, prefix: str = "train", shard_size: int = 20000, workers: int = 12) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: List[Path] = []
    t0 = time.time()
    n_shards = (n_samples + shard_size - 1) // shard_size

    with mp.Pool(processes=workers) as pool:
        idx_cursor = 0
        for s_idx in range(n_shards):
            cur_n = min(shard_size, n_samples - idx_cursor)
            chunk = max(250, (cur_n + workers - 1) // workers)
            tasks = []
            off = 0
            while off < cur_n:
                cnt = min(chunk, cur_n - off)
                tasks.append((seed, idx_cursor + off, cnt))
                off += cnt
            parts = pool.map(_worker_chunk, tasks)
            imgs = np.concatenate([p[0] for p in parts], axis=0)
            sr_t = np.concatenate([p[1] for p in parts], axis=0)
            tilt = np.concatenate([p[2] for p in parts], axis=0)
            num_cls = np.concatenate([p[3] for p in parts], axis=0)
            left_d = np.concatenate([p[4] for p in parts], axis=0)
            right_d = np.concatenate([p[5] for p in parts], axis=0)
            legible = np.concatenate([p[6] for p in parts], axis=0)
            is_back = np.concatenate([p[7] for p in parts], axis=0)
            partial_f = np.concatenate([p[8] for p in parts], axis=0)

            shard_path = out_dir / f"{prefix}_shard_{s_idx:02d}.npz"
            np.savez(
                shard_path,
                images=imgs,
                sr_target=sr_t,
                tilt_deg=tilt,
                number_cls=num_cls,
                left_digit=left_d,
                right_digit=right_d,
                legible=legible,
                is_back=is_back,
                partial_flag=partial_f,
            )
            shard_paths.append(shard_path)
            idx_cursor += cur_n
            dt = time.time() - t0
            print(f"[jersey_synth] {prefix} shard {s_idx + 1}/{n_shards}: {cur_n} samples saved to {shard_path.name} ({idx_cursor}/{n_samples}, {dt:.1f}s)", flush=True)
    return shard_paths


def save_preview_grid(out_png: Path, seed: int = 42, cols: int = 8, rows: int = 6) -> None:
    """Render 128x128 RGB crop alongside its aligned 32x32 SR stroke mask (upscaled 128x128) for visual check."""
    rng = np.random.default_rng(seed)
    cell = TARGET_SIZE
    sr_w = 64
    tile_w = cell + sr_w + 4
    pad = 22
    canvas = np.full((rows * (cell + pad), cols * tile_w, 3), 24, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            im, sr, td, nc, ld, rd, lg, bk, pf = generate_sample(rng)
            y0 = r * (cell + pad)
            x0 = c * tile_w
            canvas[y0:y0 + cell, x0:x0 + cell] = im
            sr_vis = cv2.resize(sr, (sr_w, sr_w), interpolation=cv2.INTER_NEAREST)
            canvas[y0:y0 + sr_w, x0 + cell + 2:x0 + cell + 2 + sr_w] = np.stack([sr_vis] * 3, axis=-1)
            # Also show derotated crop below SR mask to visually prove rotation sign!
            if nc != 0:
                M = cv2.getRotationMatrix2D((cell * 0.5, cell * 0.5), -float(td), 1.0)
                derot = cv2.warpAffine(im, M, (cell, cell), flags=cv2.INTER_CUBIC, borderValue=(LETTERBOX_PAD, LETTERBOX_PAD, LETTERBOX_PAD))
                derot_sm = cv2.resize(derot, (sr_w, sr_w), interpolation=cv2.INTER_AREA)
                canvas[y0 + sr_w:y0 + sr_w * 2, x0 + cell + 2:x0 + cell + 2 + sr_w] = derot_sm
            if nc == 0:
                tag = f"NEG b{bk} {td:+.0f}d"
            elif pf == PARTIAL_RIGHT_ONLY:
                tag = f"?{rd}({nc}) {td:+.0f}d"
            elif pf == PARTIAL_LEFT_ONLY:
                tag = f"{ld}?({nc}) {td:+.0f}d"
            else:
                tag = f"#{('00' if nc == 100 else nc)} {td:+.0f}d"
            cv2.putText(canvas, tag, (x0 + 2, y0 + cell + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 230, 240), 1, cv2.LINE_AA)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_WORK_DIR / "v5_str" / "shards")
    ap.add_argument("--train-samples", type=int, default=200000)
    ap.add_argument("--val-samples", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--preview", type=Path, default=None)
    args = ap.parse_args()

    if args.preview is not None:
        save_preview_grid(args.preview, seed=args.seed)
        print(f"Saved preview to {args.preview}")
        return

    generate_dataset(args.out_dir, args.train_samples, seed=args.seed, prefix="train", shard_size=20000, workers=args.workers)
    generate_dataset(args.out_dir, args.val_samples, seed=args.seed + 500000, prefix="val", shard_size=16000, workers=args.workers)


if __name__ == "__main__":
    main()
