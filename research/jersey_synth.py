"""Roster-free synthetic hockey jersey dataset generator (100% zero prior / zero test leakage).

Generates realistic 96x96 RGB letterboxed upper-torso hockey crops matching Android
`JerseyStrOnnxRecognizer.kt` (`x: -6%..106%, y: 10%..62%`, pad 114, 2%..98% local stretch):
  - Authentic squared/beveled NHL varsity block numerals (both squared-step '2' and diagonal '2',
    heavy stroke thickness 20..31% of height, tackle-twill 1/2-layer outlines, + bold system fonts)
  - Full upper-skater silhouette with shoulder yokes, flanking sleeves/arms, namebars, waist stripes,
    hockey pants hem, and wide pose/crouch offsets (vertical & horizontal jitter, lean +/-22 deg)
  - Muted realistic hockey rink backgrounds & low-contrast camera dynamic range + low-res blur + H.264
  - Uniform jersey classes 1..99 and "00" (class 100), plus class 0 (illegible/hard negative)
  - Compositional digit targets + 15% partial occlusions + 30% hard negatives (ref stripes, crests, blur, boards).

All priors over 1..99 and "00" are strictly uniform (1/100). Never imports or reads any test frame.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NUM_CLASSES = 101  # 0 = none/illegible, 1..99 = "1".."99", 100 = "00"
DIGIT_BLANK = 10   # 0..9 = digit, 10 = blank / single-right-blank

PARTIAL_NONE = 0
PARTIAL_LEFT_ONLY = 1   # left digit visible, right occluded ("a?")
PARTIAL_RIGHT_ONLY = 2  # right digit visible, left occluded ("?b")

LETTERBOX_PAD = 114

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

# Style 1: Angled varsity block ('2' diagonal neck, pillar '1' without base bar sometimes)
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

_DARK_BASES = [
    (26, 36, 66),    # navy
    (28, 30, 36),    # black / charcoal
    (26, 62, 46),    # forest green
    (96, 26, 34),    # maroon / crimson
    (28, 56, 108),   # royal blue
    (130, 30, 36),   # red
    (24, 68, 82),    # teal
    (52, 34, 78),    # purple
    (46, 52, 60),    # dark slate
    (32, 46, 72),    # midnight blue
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


def _pick_jersey_colors(rng: np.random.Generator) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    is_light = rng.random() < 0.48
    if is_light:
        body = _jitter_color(rng, _LIGHT_BASES[int(rng.integers(0, len(_LIGHT_BASES)))], 12)
        yoke = _jitter_color(rng, _DARK_BASES[int(rng.integers(0, len(_DARK_BASES)))], 14)
        digit = yoke if rng.random() < 0.60 else _jitter_color(rng, _DARK_BASES[int(rng.integers(0, len(_DARK_BASES)))], 12)
        outline = _jitter_color(rng, _ACCENT_COLORS[int(rng.integers(0, len(_ACCENT_COLORS)))], 12)
    else:
        body = _jitter_color(rng, _DARK_BASES[int(rng.integers(0, len(_DARK_BASES)))], 14)
        yoke = _jitter_color(rng, _ACCENT_COLORS[int(rng.integers(0, len(_ACCENT_COLORS)))], 14)
        u_col = rng.random()
        if u_col < 0.56:
            digit = _jitter_color(rng, (206, 210, 214), 14)
            outline = _jitter_color(rng, _DARK_BASES[int(rng.integers(0, len(_DARK_BASES)))], 10) if rng.random() < 0.5 else _jitter_color(rng, _ACCENT_COLORS[int(rng.integers(0, len(_ACCENT_COLORS)))], 12)
        elif u_col < 0.72:
            digit = _jitter_color(rng, (208, 174, 72), 14)
            outline = _jitter_color(rng, (212, 216, 222), 10) if rng.random() < 0.4 else _jitter_color(rng, _DARK_BASES[int(rng.integers(0, len(_DARK_BASES)))], 10)
        else:
            # Saturated colored numerals on dark jerseys (crimson/red/orange/royal/teal) with light/white outline
            sat_digits = [
                (172, 34, 44),   # crimson / red
                (198, 48, 38),   # scarlet
                (204, 96, 32),   # burnt orange
                (44, 102, 184),  # royal blue
                (38, 146, 156),  # teal
            ]
            digit = _jitter_color(rng, sat_digits[int(rng.integers(0, len(sat_digits)))], 12)
            outline = _jitter_color(rng, (214, 216, 220), 12) if rng.random() < 0.80 else _jitter_color(rng, (212, 180, 74), 12)
    return body, yoke, digit, outline


def _make_rink_background(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """Muted realistic ice, boards, glass, or dark rink corner background."""
    bg = np.empty((h, w, 3), dtype=np.uint8)
    kind = int(rng.choice([0, 1, 2, 3], p=[0.55, 0.22, 0.15, 0.08]))
    if kind == 0:
        # Muted ice surface with subtle blue/red line or skate scuffs
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
        # Realistic white rink boards with muted yellow kickplate & thin blue/red cap
        bw_val = int(rng.integers(165, 215))
        bg[:, :] = (bw_val, bw_val, bw_val)
        cap_y = int(h * rng.uniform(0.15, 0.35))
        bg[max(0, cap_y - 3):cap_y, :] = (65, 95, 145) if rng.random() < 0.6 else (155, 55, 55)
        if rng.random() < 0.65:
            kick_y = int(h * rng.uniform(0.65, 0.85))
            bg[kick_y:min(h, kick_y + int(h * 0.12)), :] = (182, 166, 78)
        bg[int(h * 0.80):, :] = (int(rng.integers(160, 210)),) * 3
    elif kind == 2:
        # Blurred arena stands / glass transition
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
    color: Tuple[int, int, int],
    thickness: int,
    style: int,
    slant: float,
    draw_serif_7: bool = False,
) -> None:
    if style == 0:
        segs = _NHL_SQUARED_BLOCK[ch]
    elif style == 1:
        segs = _NHL_ANGLED_BLOCK[ch]
    else:
        segs = _ROUNDED_BLOCK[ch]
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
) -> np.ndarray:
    """Render RGBA (H, W, 4) chunky athletic numeral patch with 1- or 2-layer tackle-twill outline."""
    use_pil = (len(_LOADED_FONT_PATHS) > 0) and (rng.random() < 0.35)
    alpha_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    rgb = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
    outline_mode = int(rng.choice([0, 1, 2], p=[0.22, 0.56, 0.22]))
    if abs(_luma(digit_color) - _luma(body_color)) < 48.0 and outline_mode == 0:
        outline_mode = 1

    if use_pil:
        pil_img = Image.new("RGBA", (patch_w, patch_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)
        fpath = _LOADED_FONT_PATHS[int(rng.integers(0, len(_LOADED_FONT_PATHS)))]
        fsize = int(patch_h * rng.uniform(0.64, 0.80))
        font = _get_font(fpath, max(14, fsize))
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = int(round((patch_w - tw) * 0.5 - bbox[0] + x_shift_frac * patch_w))
        ty = int(round((patch_h - th) * 0.5 - bbox[1]))
        sw = max(1, min(3, int(round(patch_h * rng.uniform(0.026, 0.048)))))
        if outline_mode == 2:
            draw.text((tx, ty), text, font=font, fill=outline_color + (255,), stroke_width=sw * 2, stroke_fill=outline_color + (255,))
            draw.text((tx, ty), text, font=font, fill=body_color + (255,), stroke_width=sw, stroke_fill=body_color + (255,))
            draw.text((tx, ty), text, font=font, fill=digit_color + (255,))
        elif outline_mode == 1:
            draw.text((tx, ty), text, font=font, fill=outline_color + (255,), stroke_width=sw, stroke_fill=outline_color + (255,))
            draw.text((tx, ty), text, font=font, fill=digit_color + (255,))
        else:
            draw.text((tx, ty), text, font=font, fill=digit_color + (255,))
        return np.array(pil_img)

    # Heavy + medium-thin NHL varsity block renderer covering both wide bold & slim real digits
    style = int(rng.choice([0, 1, 2], p=[0.46, 0.28, 0.26]))
    draw_serif_7 = bool(rng.random() < 0.35)
    slant = float(rng.uniform(-0.12, 0.12)) if rng.random() < 0.35 else 0.0
    n_chars = len(text)
    dh = int(patch_h * rng.uniform(0.56, 0.76))
    dw = int(dh * rng.uniform(0.50, 0.76))
    gap = int(dw * rng.uniform(0.16, 0.32)) if n_chars == 2 else 0
    total_w = n_chars * dw + (n_chars - 1) * gap
    start_x = int(round((patch_w - total_w) * 0.5 + x_shift_frac * patch_w))
    start_y = int(round((patch_h - dh) * 0.5))
    # Broad range 0.095..0.175*dh covers both slim real-video digits and chunky varsity twill
    base_thick = max(2, min(int(round(dh * rng.uniform(0.095, 0.175))), max(2, int(dw * 0.24))))

    def _draw_layer(col: Tuple[int, int, int], thick_delta: int) -> None:
        layer_m = np.zeros((patch_h, patch_w), dtype=np.uint8)
        for ci, ch in enumerate(text):
            cx0 = start_x + ci * (dw + gap)
            _draw_vector_digit(layer_m, ch, cx0, start_y, dw, dh, (255,), base_thick + thick_delta, style, slant, draw_serif_7)
        m = layer_m > 20
        rgb[m] = col
        alpha_mask[m] = np.maximum(alpha_mask[m], layer_m[m])

    if outline_mode == 2:
        _draw_layer(outline_color, max(2, min(7, int(round(base_thick * rng.uniform(0.42, 0.72))))))
        _draw_layer(body_color, max(1, min(4, int(round(base_thick * rng.uniform(0.20, 0.35))))))
        _draw_layer(digit_color, 0)
    elif outline_mode == 1:
        _draw_layer(outline_color, max(2, min(6, int(round(base_thick * rng.uniform(0.32, 0.65))))))
        _draw_layer(digit_color, 0)
    else:
        _draw_layer(digit_color, 0)

    return np.dstack([rgb, alpha_mask])


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
    crest_kind = int(rng.integers(0, 4))
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
    else:
        letter = "C" if rng.random() < 0.55 else "A"
        lx, ly = int(w * rng.uniform(0.24, 0.36)), int(h * rng.uniform(0.24, 0.34))
        cv2.putText(jersey, letter, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, max(0.55, h / 160.0), c1, max(2, w // 26), cv2.LINE_AA)
        cv2.circle(jersey, (cx, int(h * 0.56)), max(4, int(w * 0.14)), c2, -1, lineType=cv2.LINE_AA)


def _apply_cylindrical_warp_and_folds(
    rng: np.random.Generator,
    jersey_rgba: np.ndarray,
    yaw_deg: float,
    torso_width_frac: float = 0.84,
    lean_deg: float = 0.0,
    y_shift_frac: float = 0.0,
) -> np.ndarray:
    """Simulate 3D cylindrical jersey torso curvature, yaw rotation, skating lean, and cloth folds."""
    h, w = jersey_rgba.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nx_raw = (xs / (w - 1.0)) * 2.0 - 1.0
    ny_raw = (ys / (h - 1.0)) * 2.0 - 1.0 - y_shift_frac * 2.0
    ang = math.radians(lean_deg)
    nx_rot = nx_raw * math.cos(ang) - ny_raw * math.sin(ang)
    ny = nx_raw * math.sin(ang) + ny_raw * math.cos(ang)
    nx = nx_rot / max(0.45, torso_width_frac)

    yaw = math.radians(yaw_deg)
    arg = np.clip(nx / max(0.35, math.cos(yaw * 0.35)), -0.995, 0.995)
    theta = np.arcsin(arg) - yaw * 0.65
    u = np.sin(theta)
    map_x = ((u + 1.0) * 0.5 * (w - 1.0)).astype(np.float32)
    pitch = float(rng.uniform(-0.08, 0.08))
    map_y = np.clip(((ny + 1.0) * 0.5 * (h - 1.0)) + pitch * h * (1.0 - np.clip(nx, -1, 1) ** 2), 0, h - 1).astype(np.float32)

    warped = cv2.remap(jersey_rgba, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # Add wider shoulder silhouette at top so realistic hockey shoulder pads/elbows appear
    shoulder_boost = 0.14 * np.exp(-((ny + 0.55) / 0.38) ** 2)
    waist_lim = 0.94 + shoulder_boost - 0.10 * np.clip(ny, -0.3, 1.0)
    valid = (np.abs(nx) <= waist_lim) & (np.abs(theta) < 1.36) & (np.abs(ny) <= 1.05)
    warped[~valid, 3] = 0

    fold_freq = float(rng.uniform(2.0, 5.5))
    fold_phase = float(rng.uniform(0.0, 6.28))
    fold_amp = float(rng.uniform(0.05, 0.16))
    shading = 1.0 - 0.22 * np.clip(nx, -1, 1) ** 2 + fold_amp * np.sin(fold_freq * np.clip(nx, -1, 1) + 1.5 * ny + fold_phase)
    shading = np.clip(shading, 0.60, 1.12)[..., None]
    warped[:, :, :3] = np.clip(warped[:, :, :3].astype(np.float32) * shading, 0, 255).astype(np.uint8)
    return warped


def _apply_occlusion(rng: np.random.Generator, img: np.ndarray, occl_kind: str) -> None:
    h, w = img.shape[:2]
    if occl_kind in ("left", "right"):
        side = occl_kind
        mode = int(rng.integers(0, 3))
        if mode == 0:
            glove_col = _jitter_color(rng, (26, 28, 34), 12)
            sleeve_col = _jitter_color(rng, (45, 55, 85), 22)
            x_center = int(w * (0.34 if side == "left" else 0.66))
            x_edge = 0 if side == "left" else w
            pts = np.array([
                [x_edge, int(h * rng.uniform(0.14, 0.30))],
                [x_center + int(w * 0.11 * (1 if side == "left" else -1)), int(h * rng.uniform(0.24, 0.38))],
                [x_center + int(w * 0.10 * (1 if side == "left" else -1)), int(h * rng.uniform(0.70, 0.86))],
                [x_edge, int(h * rng.uniform(0.76, 0.94))],
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], sleeve_col, lineType=cv2.LINE_AA)
            gx = x_center + int(w * 0.04 * (1 if side == "left" else -1))
            gy = int(h * rng.uniform(0.38, 0.62))
            cv2.ellipse(img, (gx, gy), (int(w * 0.16), int(h * 0.20)), int(rng.integers(-25, 25)), 0, 360, glove_col, -1, lineType=cv2.LINE_AA)
        elif mode == 1:
            col = _jitter_color(rng, (34, 40, 54) if rng.random() < 0.5 else (188, 194, 200), 20)
            col_dark = _jitter_color(rng, (24, 26, 30), 10)
            cx = int(w * (rng.uniform(0.16, 0.24) if side == "left" else rng.uniform(0.76, 0.84)))
            rx = int(w * rng.uniform(0.28, 0.36))
            ry = int(h * rng.uniform(0.50, 0.68))
            cy = int(h * rng.uniform(0.45, 0.60))
            cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, col, -1, lineType=cv2.LINE_AA)
            cv2.ellipse(img, (cx, int(cy - ry * 0.82)), (int(rx * 0.45), int(ry * 0.25)), 0, 0, 360, col_dark, -1, lineType=cv2.LINE_AA)
        else:
            stick_col = _jitter_color(rng, (34, 32, 30), 15)
            x0 = int(w * (0.28 if side == "left" else 0.72))
            x1 = int(w * (0.46 if side == "left" else 0.54))
            cv2.line(img, (x0, 0), (x1, h), stick_col, max(6, int(w * 0.16)), lineType=cv2.LINE_AA)
            cv2.circle(img, (int((x0 + x1) * 0.5), int(h * 0.50)), int(w * 0.19), stick_col, -1, lineType=cv2.LINE_AA)
    elif occl_kind == "mild":
        stick_col = _jitter_color(rng, (30, 32, 34), 12)
        if rng.random() < 0.5:
            x0, y0 = int(w * rng.uniform(0.0, 0.3)), int(h * rng.uniform(0.65, 1.0))
            x1, y1 = int(w * rng.uniform(0.6, 1.0)), int(h * rng.uniform(0.0, 0.35))
        else:
            x0, y0 = int(w * rng.uniform(0.7, 1.0)), int(h * rng.uniform(0.65, 1.0))
            x1, y1 = int(w * rng.uniform(0.0, 0.4)), int(h * rng.uniform(0.0, 0.35))
        cv2.line(img, (x0, y0), (x1, y1), stick_col, max(2, int(w * 0.025)), lineType=cv2.LINE_AA)


def _degrade_camera_crop(
    rng: np.random.Generator,
    rgb: np.ndarray,
    is_illegible_blur: bool = False,
    min_short_side: int = 24,
) -> np.ndarray:
    """Match Android JerseyStrOnnxRecognizer.kt pipeline preserving native crop aspect ratio."""
    h_nat, w_nat = rgb.shape[:2]
    aspect = w_nat / max(1.0, float(h_nat))
    out = rgb.astype(np.float32)

    # 1. Rink color cast + exposure variation
    cast_type = int(rng.choice([0, 1, 2, 3], p=[0.46, 0.24, 0.18, 0.12]))
    if cast_type == 1:
        gains = np.array([rng.uniform(0.84, 0.96), rng.uniform(1.01, 1.13), rng.uniform(0.88, 1.02)], dtype=np.float32)
    elif cast_type == 2:
        gains = np.array([rng.uniform(0.88, 0.97), rng.uniform(0.95, 1.03), rng.uniform(1.04, 1.16)], dtype=np.float32)
    elif cast_type == 3:
        gains = np.array([rng.uniform(1.03, 1.14), rng.uniform(0.95, 1.04), rng.uniform(0.84, 0.95)], dtype=np.float32)
    else:
        gains = np.array([rng.uniform(0.95, 1.05), rng.uniform(0.95, 1.05), rng.uniform(0.95, 1.05)], dtype=np.float32)
    out *= gains[None, None, :]
    out *= float(rng.uniform(0.80, 1.12))
    out += float(rng.uniform(-10.0, 10.0))
    out = np.clip(out, 0, 255).astype(np.uint8)

    # 2. Directional skating motion blur
    if is_illegible_blur or rng.random() < 0.45:
        k = int(rng.integers(8, 16)) if is_illegible_blur else int(rng.integers(2, 6))
        ang = float(rng.uniform(-35.0, 35.0))
        kern = np.zeros((k, k), dtype=np.float32)
        kern[k // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), ang, 1.0)
        kern = cv2.warpAffine(kern, M, (k, k))
        s = kern.sum()
        if s > 1e-6:
            kern /= s
            out = cv2.filter2D(out, -1, kern)

    # 3. Low-res camera downscale preserving native crop aspect ratio (enforcing legible Nyquist floor)
    if is_illegible_blur:
        short_side = int(rng.integers(10, 21))
    else:
        lo_sh = min(60, max(24, int(min_short_side)))
        r = rng.random()
        if r < 0.40:
            short_side = int(rng.integers(lo_sh, max(lo_sh + 6, 44)))
        elif r < 0.80:
            short_side = int(rng.integers(max(lo_sh, 40), 68))
        else:
            short_side = int(rng.integers(max(lo_sh, 64), 92))
    if aspect >= 1.0:
        sh = short_side
        sw = max(14, int(round(short_side * aspect)))
    else:
        sw = short_side
        sh = max(14, int(round(short_side / aspect)))
    low = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)

    # 4. H.264 / JPEG compression at native low-res scale
    if rng.random() < 0.82:
        q = int(rng.integers(35, 90))
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(low, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                low = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    if rng.random() < 0.55:
        sigma = float(rng.uniform(1.5, 5.5))
        low = np.clip(low.astype(np.float32) + rng.normal(0.0, sigma, size=low.shape).astype(np.float32), 0, 255).astype(np.uint8)

    # 5. Upscale & aspect-preserving letterbox to 96x96 + JerseyStrOnnxRecognizer 2%..98% contrast stretch
    scale = min(96.0 / max(1, sw), 96.0 / max(1, sh))
    nw, nh = max(1, min(96, int(round(sw * scale)))), max(1, min(96, int(round(sh * scale))))
    pad_x, pad_y = (96 - nw) // 2, (96 - nh) // 2
    resized = cv2.resize(low, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    luma = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
    p_low, p_high = np.percentile(luma, [2.0, 98.0])
    raw_span = max(32.0, float(p_high - p_low))
    gain = min(max(1.0 + 0.12 * min(max(220.0 / raw_span, 0.8), 2.0), 1.05), 1.25)
    mid = 0.5 * float(p_low + p_high)
    resized = np.clip((resized - mid) * gain + mid, 0.0, 255.0)

    pad_val = LETTERBOX_PAD if rng.random() < 0.96 else int(rng.choice([0, 96, 128]))
    canvas = np.full((96, 96, 3), pad_val, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = np.clip(resized, 0, 255).astype(np.uint8)
    return canvas


def _apply_sleeve_shadow_folds(rng: np.random.Generator, jersey: np.ndarray, yoke_col: Tuple[int, int, int]) -> None:
    """Inject realistic lateral curved sleeve folds & armpit shadows next to outer digits ('7','1','4','9')."""
    if rng.random() >= 0.52:
        return
    h, w = jersey.shape[:2]
    fold_mask = np.zeros((h, w), dtype=np.float32)
    shadow_col = _jitter_color(rng, yoke_col if rng.random() < 0.40 else (28, 34, 46), 14)
    sides = ["right"] if rng.random() < 0.55 else (["left"] if rng.random() < 0.5 else ["left", "right"])
    for side in sides:
        thick = int(w * rng.uniform(0.06, 0.14))
        x_edge = int(w * rng.uniform(0.76, 0.94)) if side == "right" else int(w * rng.uniform(0.06, 0.24))
        x_mid = int(w * rng.uniform(0.64, 0.80)) if side == "right" else int(w * rng.uniform(0.20, 0.36))
        y0 = int(h * rng.uniform(0.24, 0.42))
        y1 = int(h * rng.uniform(0.46, 0.64))
        y2 = int(h * rng.uniform(0.68, 0.92))
        pts = np.array([[x_edge, y0], [x_mid, y1], [x_edge, y2]], dtype=np.int32)
        cv2.polylines(fold_mask, [pts], False, 1.0, max(3, thick), lineType=cv2.LINE_AA)
    k = int(rng.choice([7, 9, 11, 13]))
    fold_mask = cv2.GaussianBlur(fold_mask, (k, k), 0)
    alpha = np.clip(fold_mask * float(rng.uniform(0.28, 0.68)), 0.0, 0.72)[..., None]
    sc = np.array(shadow_col, dtype=np.float32)[None, None, :]
    jersey[:] = np.clip(jersey.astype(np.float32) * (1.0 - alpha) + sc * alpha, 0, 255).astype(np.uint8)


def generate_sample(rng: np.random.Generator) -> Tuple[np.ndarray, int, int, int, int, int, int]:
    """Generate one synthetic 96x96 hockey torso crop and multi-task labels."""
    H = 128
    # Sample realistic crop aspect ratio matching real YOLO x:-6..106%, y:10..62% crops:
    # ~22% upright (0.72..0.95), ~48% medium (0.95..1.35), ~30% wide crouched (1.35..2.10)
    r_asp = float(rng.random())
    if r_asp < 0.22:
        crop_aspect = float(rng.uniform(0.72, 0.95))
    elif r_asp < 0.70:
        crop_aspect = float(rng.uniform(0.95, 1.35))
    else:
        crop_aspect = float(rng.uniform(1.35, 2.10))
    W = max(92, int(round(H * crop_aspect)))

    # 56% legible positive, 12% partial occlusion, 32% hard negative (including double-skater overlaps & heavy blurs)
    u = float(rng.random())
    if u < 0.56:
        cat = "legible"
    elif u < 0.68:
        cat = "partial"
    elif u < 0.75:
        cat = "ref_stripes"
    elif u < 0.81:
        cat = "front_crest"
    elif u < 0.87:
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
                (int(rng.integers(8, max(12, W - 48))), int(rng.integers(28, 108))),
                cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.45, 1.2)),
                tuple(int(v) for v in rng.integers(35, 195, size=3)),
                int(rng.integers(2, 4)), cv2.LINE_AA,
            )
        return _degrade_camera_crop(rng, bg, is_illegible_blur=False), 0, DIGIT_BLANK, DIGIT_BLANK, 0, 0, PARTIAL_NONE

    body_col, yoke_col, digit_col, outline_col = _pick_jersey_colors(rng)
    # Render jersey torso onto a square 128x128 surface, then composite onto aspect-matched (H, W)
    J_W = 128
    jersey = np.empty((H, J_W, 3), dtype=np.uint8)
    jersey[:, :] = body_col

    if cat != "ref_stripes" and rng.random() < 0.74:
        yoke_style = int(rng.choice([0, 1, 2], p=[0.30, 0.55, 0.15]))
        yoke_bottom = int(H * rng.uniform(0.18, 0.34))
        trim_h = max(2, int(H * rng.uniform(0.025, 0.045)))
        if yoke_style == 0:
            jersey[:yoke_bottom, :] = yoke_col
            if rng.random() < 0.60:
                jersey[yoke_bottom:yoke_bottom + trim_h, :] = outline_col
        elif yoke_style == 1:
            # Curved NHL shoulder yoke (either dipping along shoulder caps or arched center crest)
            nx_j = (np.arange(J_W, dtype=np.float32) - J_W * 0.5) / (J_W * 0.5)
            arc_amp = float(H * rng.uniform(0.06, 0.15))
            if rng.random() < 0.55:
                y_curve = (yoke_bottom + arc_amp * (nx_j ** 2)).astype(np.int32)
            else:
                y_curve = (yoke_bottom + arc_amp * (1.0 - nx_j ** 2)).astype(np.int32)
            ys_col = np.arange(H, dtype=np.int32)[:, None]
            y_c = np.clip(y_curve[None, :], 8, int(H * 0.46))
            jersey[ys_col < y_c] = yoke_col
            if rng.random() < 0.65:
                jersey[(ys_col >= y_c) & (ys_col < y_c + trim_h)] = outline_col
        else:
            # Upper yoke + side shoulder/sleeve color blocks
            jersey[:yoke_bottom, :] = yoke_col
            sw_side = int(J_W * rng.uniform(0.14, 0.24))
            sh_end = int(H * rng.uniform(0.36, 0.52))
            jersey[:sh_end, :sw_side] = yoke_col
            jersey[:sh_end, J_W - sw_side:] = yoke_col
        if rng.random() < 0.50:
            cv2.ellipse(jersey, (J_W // 2, 0), (int(J_W * 0.18), int(H * 0.09)), 0, 0, 180, outline_col, max(2, int(H * 0.025)), lineType=cv2.LINE_AA)

    if rng.random() < 0.70:
        hem_y = int(H * rng.uniform(0.72, 0.86))
        band_h = max(3, int(H * rng.uniform(0.035, 0.065)))
        stripe_c1 = outline_col if cat != "ref_stripes" else (24, 24, 28)
        stripe_c2 = yoke_col if cat != "ref_stripes" else (200, 204, 210)
        jersey[hem_y:min(H, hem_y + band_h), :] = stripe_c1
        if rng.random() < 0.65:
            gap_h = max(2, int(H * 0.024))
            band2_h = max(2, int(rng.uniform(0.028, 0.050) * H))
            y2 = hem_y - gap_h - band2_h
            if y2 > int(H * 0.66):
                jersey[y2:y2 + band2_h, :] = stripe_c2
            elif hem_y + band_h + gap_h + band2_h < H:
                jersey[hem_y + band_h + gap_h:hem_y + band_h + gap_h + band2_h, :] = stripe_c2
    if rng.random() < 0.78:
        pants_y = int(H * rng.uniform(0.84, 0.94))
        jersey[pants_y:, :] = _jitter_color(rng, (24, 26, 30), 10)

    if rng.random() < 0.72:
        helm_y = int(H * rng.uniform(0.05, 0.16))
        helm_col = _jitter_color(rng, (22, 24, 30) if rng.random() < 0.65 else yoke_col, 14)
        jersey[:helm_y, int(J_W * 0.28):int(J_W * 0.72)] = helm_col

    torso_w = float(rng.uniform(0.56, 0.92))
    lean_deg = float(rng.uniform(-20.0, 20.0))
    y_shift = float(rng.uniform(-0.08, 0.14))

    def _draw_nameplate(target_j: np.ndarray, y_num_top: int) -> None:
        if y_num_top < 15:
            return
        nb_len = int(rng.integers(5, 9))
        nb_txt = "".join(chr(int(rng.integers(65, 91))) for _ in range(nb_len))
        gap_px = int(rng.integers(2, 7))
        y_base = max(12, y_num_top - gap_px)
        name_col = digit_col if abs(_luma(digit_col) - _luma(body_col)) > 35 else outline_col
        if rng.random() < 0.45:
            # Stitched fabric nameplate bar with contrasting uppercase lettering
            bar_h = int(H * rng.uniform(0.08, 0.12))
            by0 = max(4, y_base - bar_h)
            bar_col = yoke_col if abs(_luma(yoke_col) - _luma(body_col)) > 18 else _jitter_color(rng, body_col, 24)
            target_j[by0:y_base + 1, int(J_W * 0.18):int(J_W * 0.82)] = bar_col
            txt_col = _jitter_color(rng, (214, 218, 222), 12) if _luma(bar_col) < 120 else _jitter_color(rng, (26, 30, 38), 10)
        else:
            txt_col = name_col
        f_scale = float(rng.uniform(0.27, 0.42))
        f_thick = 1 if rng.random() < 0.70 else 2
        font_id = cv2.FONT_HERSHEY_SIMPLEX if rng.random() < 0.65 else cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            target_j, nb_txt,
            (int(J_W * rng.uniform(0.20, 0.30)), y_base),
            font_id, f_scale, txt_col, f_thick, cv2.LINE_AA,
        )

    def _composite_torso(j_arr: np.ndarray, yaw_d: float, t_w: float, x_bias: float = 0.0, base_bg: np.ndarray | None = None) -> np.ndarray:
        rgba = np.dstack([j_arr, np.full((H, J_W), 255, dtype=np.uint8)])
        warped_sq = _apply_cylindrical_warp_and_folds(rng, rgba, yaw_d, t_w, lean_deg, y_shift)
        # Place warped torso horizontally on the aspect-matched W canvas without stretching digits!
        out_comp = (bg if base_bg is None else base_bg).copy()
        place_w = min(W, max(78, int(round(H * rng.uniform(0.82, 1.18)))))
        warped_resized = cv2.resize(warped_sq, (place_w, H), interpolation=cv2.INTER_LINEAR)
        x_off = int(round((W - place_w) * 0.5 + (x_bias + rng.uniform(-0.07, 0.07)) * W))
        x_off = max(-place_w // 3, min(W - (2 * place_w) // 3, x_off))
        sx0 = max(0, -x_off)
        dx0 = max(0, x_off)
        copy_w = min(place_w - sx0, W - dx0)
        if copy_w > 0:
            a = warped_resized[:, sx0:sx0 + copy_w, 3:4].astype(np.float32) / 255.0
            out_comp[:, dx0:dx0 + copy_w] = np.clip(
                warped_resized[:, sx0:sx0 + copy_w, :3].astype(np.float32) * a + out_comp[:, dx0:dx0 + copy_w].astype(np.float32) * (1.0 - a),
                0, 255,
            ).astype(np.uint8)
        return out_comp

    if cat == "double_body_overlap":
        # Two overlapping skaters colliding/crossing (kills T38:#85 false positives)
        is_back = int(rng.random() < 0.50)
        shift_side = -0.22 if rng.random() < 0.5 else 0.22
        if rng.random() < 0.65:
            num_idx = int(rng.integers(10, 100))
            patch = _render_jersey_number_patch(rng, str(num_idx), J_W, int(H * rng.uniform(0.30, 0.48)), digit_col, outline_col, body_col)
            py0 = int(H * rng.uniform(0.24, 0.42))
            pa = patch[:, :, 3:4].astype(np.float32) / 255.0
            jersey[py0:py0 + patch.shape[0], :] = np.clip(
                patch[:, :, :3].astype(np.float32) * pa + jersey[py0:py0 + patch.shape[0], :].astype(np.float32) * (1.0 - pa),
                0, 255,
            ).astype(np.uint8)
        _apply_sleeve_shadow_folds(rng, jersey, yoke_col)
        comp1 = _composite_torso(jersey, float(rng.uniform(-42.0, 42.0)), torso_w, x_bias=shift_side)
        b2, y2, d2, o2 = _pick_jersey_colors(rng)
        j2 = np.empty((H, J_W, 3), dtype=np.uint8)
        j2[:, :] = b2
        j2[:int(H * rng.uniform(0.18, 0.34)), :] = y2
        j2[int(H * 0.85):, :] = _jitter_color(rng, (24, 26, 30), 10)
        if rng.random() < 0.55:
            p2 = _render_jersey_number_patch(rng, str(int(rng.integers(1, 100))), J_W, int(H * 0.38), d2, o2, b2)
            p2y = int(H * 0.28)
            pa2 = p2[:, :, 3:4].astype(np.float32) / 255.0
            j2[p2y:p2y + p2.shape[0], :] = np.clip(p2[:, :, :3].astype(np.float32) * pa2 + j2[p2y:p2y + p2.shape[0], :].astype(np.float32) * (1.0 - pa2), 0, 255).astype(np.uint8)
        comp2 = _composite_torso(j2, float(rng.uniform(-48.0, 48.0)), torso_w, x_bias=-shift_side, base_bg=comp1)
        return _degrade_camera_crop(rng, comp2, bool(rng.random() < 0.35)), 0, DIGIT_BLANK, DIGIT_BLANK, 0, is_back, PARTIAL_NONE

    if cat == "ref_stripes":
        _draw_referee_jersey(rng, jersey, J_W, H)
        is_back = int(rng.random() < 0.55)
        comp = _composite_torso(jersey, float(rng.uniform(-32.0, 32.0)), torso_w)
        return _degrade_camera_crop(rng, comp, False), 0, DIGIT_BLANK, DIGIT_BLANK, 0, is_back, PARTIAL_NONE

    if cat == "front_crest":
        _draw_front_crest(rng, jersey, body_col, yoke_col, outline_col)
        comp = _composite_torso(jersey, float(rng.uniform(-35.0, 35.0)), torso_w)
        return _degrade_camera_crop(rng, comp, False), 0, DIGIT_BLANK, DIGIT_BLANK, 0, 0, PARTIAL_NONE

    if cat == "blank_or_blur":
        is_back = int(rng.random() < 0.50)
        illegible_blur = rng.random() < 0.65
        if illegible_blur:
            num_idx = int(rng.integers(1, 101))
            num_str = "00" if num_idx == 100 else str(num_idx)
            patch_h = int(H * rng.uniform(0.24, 0.44))
            patch = _render_jersey_number_patch(rng, num_str, J_W, patch_h, digit_col, outline_col, body_col)
            py0 = int(H * rng.uniform(0.26, 0.42))
            py0 = min(H - patch_h, max(0, py0))
            pa = patch[:, :, 3:4].astype(np.float32) / 255.0
            jersey[py0:py0 + patch_h, :] = np.clip(
                patch[:, :, :3].astype(np.float32) * pa + jersey[py0:py0 + patch_h, :].astype(np.float32) * (1.0 - pa),
                0, 255,
            ).astype(np.uint8)
            if is_back and rng.random() < 0.70:
                _draw_nameplate(jersey, py0)
            # Mix oblique edge angles + straight-on extreme tiny blur
            yaw_deg = float(rng.choice([-1.0, 1.0]) * rng.uniform(48.0, 74.0)) if rng.random() < 0.55 else float(rng.uniform(-28.0, 28.0))
        else:
            yaw_deg = float(rng.uniform(-45.0, 45.0))
        comp = _composite_torso(jersey, yaw_deg, torso_w)
        return _degrade_camera_crop(rng, comp, illegible_blur), 0, DIGIT_BLANK, DIGIT_BLANK, 0, is_back, PARTIAL_NONE

    def _nyquist_min_sh(p_h: int) -> int:
        dh_frac = max(0.10, (p_h / float(H)) * 0.64)
        min_sh = int(math.ceil(7.6 / dh_frac))
        if crop_aspect < 1.0:
            min_sh = int(math.ceil(min_sh * crop_aspect))
        return max(24, min(58, min_sh))

    if cat == "partial":
        p_choice = int(rng.integers(10, 101))
        num_str = "00" if p_choice == 100 else str(p_choice)
        hide_left = bool(rng.random() < 0.5)
        is_back = 1 if rng.random() < 0.85 else 0
        patch_h = int(H * rng.uniform(0.32, 0.56))
        patch = _render_jersey_number_patch(rng, num_str, J_W, patch_h, digit_col, outline_col, body_col, float(rng.uniform(-0.05, 0.05)))
        py0 = int(H * rng.uniform(0.24, 0.44))
        py0 = min(H - patch_h, max(0, py0))
        pa = patch[:, :, 3:4].astype(np.float32) / 255.0
        jersey[py0:py0 + patch_h, :] = np.clip(
            patch[:, :, :3].astype(np.float32) * pa + jersey[py0:py0 + patch_h, :].astype(np.float32) * (1.0 - pa),
            0, 255,
        ).astype(np.uint8)
        if is_back and rng.random() < 0.72:
            _draw_nameplate(jersey, py0)
        comp = _composite_torso(jersey, float(rng.uniform(-24.0, 24.0)), max(0.66, torso_w))
        _apply_occlusion(rng, comp, "left" if hide_left else "right")
        img_96 = _degrade_camera_crop(rng, comp, False, min_short_side=_nyquist_min_sh(patch_h))
        d0, d1 = int(num_str[0]), int(num_str[1])
        if hide_left:
            return img_96, p_choice, DIGIT_BLANK, d1, 0, is_back, PARTIAL_RIGHT_ONLY
        return img_96, p_choice, d0, DIGIT_BLANK, 0, is_back, PARTIAL_LEFT_ONLY

    # Legible full number sample: strictly uniform 1..99 + "00" (1/100 each)
    num_cls = int(rng.integers(1, 101))
    num_str = "00" if num_cls == 100 else str(num_cls)
    is_back = 1 if rng.random() < 0.88 else 0
    # Cover smaller real-video digit sizes (26..58% of jersey height)
    patch_h = int(H * (rng.uniform(0.26, 0.58) if is_back else rng.uniform(0.24, 0.48)))
    x_shift = float(rng.uniform(-0.10, 0.10))
    patch = _render_jersey_number_patch(rng, num_str, J_W, patch_h, digit_col, outline_col, body_col, x_shift_frac=x_shift)
    py0 = int(H * (rng.uniform(0.22, 0.46) if is_back else rng.uniform(0.26, 0.48)))
    py0 = min(H - patch_h, max(0, py0))
    pa = patch[:, :, 3:4].astype(np.float32) / 255.0
    jersey[py0:py0 + patch_h, :] = np.clip(
        patch[:, :, :3].astype(np.float32) * pa + jersey[py0:py0 + patch_h, :].astype(np.float32) * (1.0 - pa),
        0, 255,
    ).astype(np.uint8)

    if is_back and rng.random() < 0.82:
        _draw_nameplate(jersey, py0)
    _apply_sleeve_shadow_folds(rng, jersey, yoke_col)

    comp = _composite_torso(jersey, float(rng.uniform(-32.0, 32.0)), torso_w)
    if rng.random() < 0.18:
        _apply_occlusion(rng, comp, "mild")
    img_96 = _degrade_camera_crop(rng, comp, False, min_short_side=_nyquist_min_sh(patch_h))

    if len(num_str) == 1:
        return img_96, num_cls, int(num_str[0]), DIGIT_BLANK, 1, is_back, PARTIAL_NONE
    return img_96, num_cls, int(num_str[0]), int(num_str[1]), 1, is_back, PARTIAL_NONE


def _worker_chunk(args: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed, start_idx, count = args
    rng = np.random.default_rng(seed + start_idx * 9973)
    imgs = np.empty((count, 96, 96, 3), dtype=np.uint8)
    num_cls = np.empty((count,), dtype=np.int16)
    left_d = np.empty((count,), dtype=np.int16)
    right_d = np.empty((count,), dtype=np.int16)
    legible = np.empty((count,), dtype=np.uint8)
    is_back = np.empty((count,), dtype=np.uint8)
    partial_f = np.empty((count,), dtype=np.uint8)
    for i in range(count):
        im, nc, ld, rd, lg, bk, pf = generate_sample(rng)
        imgs[i] = im
        num_cls[i] = nc
        left_d[i] = ld
        right_d[i] = rd
        legible[i] = lg
        is_back[i] = bk
        partial_f[i] = pf
    return imgs, num_cls, left_d, right_d, legible, is_back, partial_f


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
            num_cls = np.concatenate([p[1] for p in parts], axis=0)
            left_d = np.concatenate([p[2] for p in parts], axis=0)
            right_d = np.concatenate([p[3] for p in parts], axis=0)
            legible = np.concatenate([p[4] for p in parts], axis=0)
            is_back = np.concatenate([p[5] for p in parts], axis=0)
            partial_f = np.concatenate([p[6] for p in parts], axis=0)

            shard_path = out_dir / f"{prefix}_shard_{s_idx:02d}.npz"
            np.savez(
                shard_path,
                images=imgs,
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
    rng = np.random.default_rng(seed)
    cell = 96
    pad = 18
    canvas = np.full((rows * (cell + pad), cols * cell, 3), 24, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            im, nc, ld, rd, lg, bk, pf = generate_sample(rng)
            y0 = r * (cell + pad)
            x0 = c * cell
            canvas[y0:y0 + cell, x0:x0 + cell] = im
            if nc == 0:
                tag = f"NEG b{bk}"
            elif pf == PARTIAL_RIGHT_ONLY:
                tag = f"?{rd} (t:{nc})"
            elif pf == PARTIAL_LEFT_ONLY:
                tag = f"{ld}? (t:{nc})"
            else:
                tag = f"#{('00' if nc == 100 else nc)} L{ld}R{rd}"
            cv2.putText(canvas, tag, (x0 + 2, y0 + cell + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 230, 240), 1, cv2.LINE_AA)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/home/windgone/.gemini/antigravity/brain/d4f6c85d-b213-4653-af5c-c83e4f23795d/scratch/coachbox_v2/synth_shards"))
    ap.add_argument("--train-samples", type=int, default=160000)
    ap.add_argument("--val-samples", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--preview", type=Path, default=None)
    args = ap.parse_args()

    if args.preview is not None:
        save_preview_grid(args.preview, seed=args.seed)
        print(f"Saved preview to {args.preview}")
        return

    generate_dataset(args.out_dir, args.train_samples, seed=args.seed, prefix="train", shard_size=20000, workers=args.workers)
    generate_dataset(args.out_dir, args.val_samples, seed=args.seed + 500000, prefix="val", shard_size=12000, workers=args.workers)


if __name__ == "__main__":
    main()
