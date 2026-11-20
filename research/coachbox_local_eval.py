"""Roster-free local end-to-end validation harness: Baseline vs. Proposed SOTA.

Evaluates on:
  1. CoachBox compare stills (frame_00..15.png, 11:14 frame_08.png GT) + crossfade burst ghost check
  2. CoachBox cropped 1080p raw frames (fs_00..22_*.png, 11:14 fs_12_674.png GT)
  3. eng-test-bench sample_hockey_30s.mp4 (2D schematic false-positive sanity check)
  4. eng-test-bench tvi_iceland_clip_01_0240s_0270s.mp4 (non-CoachBox real rink sanity check)

100% Roster-Free: uniform prior over 1..99 (+ "00" in ONNX) with zero roster inputs.
Ground-truth boxes/labels are used strictly for scoring outputs after inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = Path(os.environ.get("HOCKEY_SCRATCH_DIR", str(REPO_ROOT / "research" / "scratch")))
DEFAULT_BENCH_DIR = Path(os.environ.get("ENG_TEST_BENCH_DIR", str(REPO_ROOT)))

# ---------------------------------------------------------------------------
# Color & CIELAB Utilities (exact mirror of KitLabFeatures.kt)
# ---------------------------------------------------------------------------

_D65_XN = 0.95047
_D65_YN = 1.00000
_D65_ZN = 1.08883
_LAB_DELTA = 6.0 / 29.0
_LAB_DELTA_CUBED = _LAB_DELTA ** 3
_LAB_SCALE = 1.0 / (3.0 * _LAB_DELTA ** 2)
_LAB_OFFSET = 4.0 / 29.0

WEIGHT_L = 0.45
WEIGHT_AB = 1.0
WEIGHT_SHARE = 0.55
BELLY_WEIGHT = 1.25


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) float RGB in 0..255 -> (..., 3) CIELAB [L*(0..100), a*, b*]."""
    s = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 255.0) / 255.0
    lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    x = (0.4124564 * lin[..., 0] + 0.3575761 * lin[..., 1] + 0.1804375 * lin[..., 2]) / _D65_XN
    y = (0.2126729 * lin[..., 0] + 0.7151522 * lin[..., 1] + 0.0721750 * lin[..., 2]) / _D65_YN
    z = (0.0193339 * lin[..., 0] + 0.1191920 * lin[..., 1] + 0.9503041 * lin[..., 2]) / _D65_ZN
    fx = np.where(x > _LAB_DELTA_CUBED, np.cbrt(x), x * _LAB_SCALE + _LAB_OFFSET)
    fy = np.where(y > _LAB_DELTA_CUBED, np.cbrt(y), y * _LAB_SCALE + _LAB_OFFSET)
    fz = np.where(z > _LAB_DELTA_CUBED, np.cbrt(z), z * _LAB_SCALE + _LAB_OFFSET)
    L = np.clip(116.0 * fy - 16.0, 0.0, 100.0)
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def luma_to_lab_l(luma: float) -> float:
    s = min(max(luma, 0.0), 255.0) / 255.0
    lin = s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4
    fy = math.cbrt(lin) if lin > _LAB_DELTA_CUBED else lin * _LAB_SCALE + _LAB_OFFSET
    return min(max(116.0 * fy - 16.0, 0.0), 100.0)


def swap_summary(s7: np.ndarray) -> np.ndarray:
    return np.array([s7[3], s7[4], s7[5], s7[0], s7[1], s7[2], 100.0 - s7[6]], dtype=np.float64)


def weighted_kit_dist_sq(x: np.ndarray, c: np.ndarray) -> float:
    d = np.array([
        (x[0] - c[0]) * WEIGHT_L, (x[1] - c[1]) * WEIGHT_AB, (x[2] - c[2]) * WEIGHT_AB,
        (x[3] - c[3]) * WEIGHT_L, (x[4] - c[4]) * WEIGHT_AB, (x[5] - c[5]) * WEIGHT_AB,
        (x[6] - c[6]) * WEIGHT_SHARE,
    ], dtype=np.float64)
    return float((d * d).sum())


def order_free_kit_dist_sq(x: np.ndarray, c: np.ndarray) -> float:
    return min(weighted_kit_dist_sq(x, c), weighted_kit_dist_sq(swap_summary(x), c))


def order_free_best_aligned(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    sw = swap_summary(x)
    return x if weighted_kit_dist_sq(x, c) <= weighted_kit_dist_sq(sw, c) else sw


def kit_effective_lightness(s7: np.ndarray, belly_luma: float) -> float:
    share = min(max(s7[6] / 100.0, 0.0), 1.0)
    mean_l = (1.0 - share) * s7[0] + share * s7[3]
    if math.isfinite(belly_luma):
        return 0.55 * mean_l + 0.45 * luma_to_lab_l(belly_luma)
    return float(mean_l)


def kit_and_belly_dist_sq(kx: np.ndarray, bx: float, kc: np.ndarray, bc: float) -> float:
    dist = order_free_kit_dist_sq(kx, kc)
    if math.isfinite(bx) and math.isfinite(bc):
        db = (luma_to_lab_l(bx) - luma_to_lab_l(bc)) * BELLY_WEIGHT
        dist += db * db
    return dist


HUBER_AB_DELTA = 22.0


def huber_sq(diff: float, delta: float = HUBER_AB_DELTA) -> float:
    a = abs(diff)
    return a * a if a <= delta else delta * (2.0 * a - delta)


def robust_weighted_kit_dist_sq(x: np.ndarray, c: np.ndarray, delta: float = HUBER_AB_DELTA) -> float:
    dL0 = (x[0] - c[0]) * WEIGHT_L
    dL1 = (x[3] - c[3]) * WEIGHT_L
    dShare = (x[6] - c[6]) * WEIGHT_SHARE
    return float(
        dL0 * dL0
        + huber_sq((x[1] - c[1]) * WEIGHT_AB, delta)
        + huber_sq((x[2] - c[2]) * WEIGHT_AB, delta)
        + dL1 * dL1
        + huber_sq((x[4] - c[4]) * WEIGHT_AB, delta)
        + huber_sq((x[5] - c[5]) * WEIGHT_AB, delta)
        + dShare * dShare
    )


def robust_order_free_kit_dist_sq(x: np.ndarray, c: np.ndarray, delta: float = HUBER_AB_DELTA) -> float:
    return min(robust_weighted_kit_dist_sq(x, c, delta), robust_weighted_kit_dist_sq(swap_summary(x), c, delta))


def robust_order_free_best_aligned(x: np.ndarray, c: np.ndarray, delta: float = HUBER_AB_DELTA) -> np.ndarray:
    sw = swap_summary(x)
    return x if robust_weighted_kit_dist_sq(x, c, delta) <= robust_weighted_kit_dist_sq(sw, c, delta) else sw


def robust_kit_and_belly_dist_sq(kx: np.ndarray, bx: float, kc: np.ndarray, bc: float, delta: float = HUBER_AB_DELTA) -> float:
    dist = robust_order_free_kit_dist_sq(kx, kc, delta)
    if math.isfinite(bx) and math.isfinite(bc):
        db = (luma_to_lab_l(bx) - luma_to_lab_l(bc)) * BELLY_WEIGHT
        dist += db * db
    return dist


def max_zone_chroma(s7: np.ndarray) -> float:
    return max(math.hypot(float(s7[1]), float(s7[2])), math.hypot(float(s7[4]), float(s7[5])))


# ---------------------------------------------------------------------------
# Detector & Feature Extractors (Baseline vs. Proposed SOTA)
# ---------------------------------------------------------------------------

def is_on_ice(box: Tuple[float, float, float, float], aspect_ratio: float) -> bool:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0.005 or h < 0.026 or w > 0.45 or h > 0.85:
        return False
    cx, fy = 0.5 * (x1 + x2), min(max(y2, 0.0), 1.0)
    if aspect_ratio >= 2.0:
        poly = [
            (0.044, 0.420), (0.233, 0.288), (0.500, 0.254), (0.767, 0.288),
            (0.956, 0.420), (0.985, 0.653), (0.772, 0.785), (0.600, 0.918),
            (0.400, 0.918), (0.228, 0.785), (0.015, 0.653),
        ]
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            denom = (yj - yi) if abs(yj - yi) > 1e-6 else 1e-6
            if ((yi > fy) != (yj > fy)) and (cx < (xj - xi) * (fy - yi) / denom + xi):
                inside = not inside
            j = i
        return inside
    if cx < 0.015 or cx > 0.985 or fy < 0.20 or fy > 0.94:
        return False
    if fy > 0.83 and (cx < 0.14 or cx > 0.86):
        return False
    return True


class YoloDetector:
    def __init__(self, onnx_path: Path) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    def detect(self, rgb: np.ndarray, panoramic_tiling: bool = True) -> List[Tuple[float, float, float, float, float]]:
        h, w = rgb.shape[:2]
        # Full 640x360 letterboxed to 640x384
        det = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((384, 640, 3), 114, dtype=np.uint8)
        canvas[12:372, :] = det
        out = self.sess.run(None, {"images": (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]})[0][0]
        cands: List[Tuple[float, float, float, float, float, bool]] = []
        for r in out:
            if r[4] >= 0.30 and r[5] == 0.0:
                x1 = float(np.clip(r[0] / 640.0, 0.0, 1.0))
                y1 = float(np.clip((r[1] - 12.0) / 360.0, 0.0, 1.0))
                x2 = float(np.clip(r[2] / 640.0, 0.0, 1.0))
                y2 = float(np.clip((r[3] - 12.0) / 360.0, 0.0, 1.0))
                if x2 - x1 >= 0.004 and y2 - y1 >= 0.008:
                    cands.append((x1, y1, x2, y2, float(r[4]), False))

        aspect = w / max(1.0, float(h))
        if panoramic_tiling and aspect >= 1.65:
            for cx0, cy0, cx1, cy1 in ((0.08, 0.20, 0.56, 0.72), (0.44, 0.20, 0.92, 0.72)):
                sx0, sy0 = int(cx0 * w), int(cy0 * h)
                sx1, sy1 = int(cx1 * w), int(cy1 * h)
                crop = rgb[sy0:sy1, sx0:sx1]
                ch, cw = crop.shape[:2]
                scale = min(640.0 / cw, 384.0 / ch)
                dw, dh = max(1, int(cw * scale)), max(1, int(ch * scale))
                px, py = (640 - dw) // 2, (384 - dh) // 2
                tcan = np.full((384, 640, 3), 114, dtype=np.uint8)
                tcan[py:py + dh, px:px + dw] = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_LINEAR)
                tout = self.sess.run(None, {"images": (tcan.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]})[0][0]
                for r in tout:
                    if r[4] >= 0.30 and r[5] == 0.0:
                        nx1 = float(np.clip((r[0] - px) / dw, 0.0, 1.0))
                        ny1 = float(np.clip((r[1] - py) / dh, 0.0, 1.0))
                        nx2 = float(np.clip((r[2] - px) / dw, 0.0, 1.0))
                        ny2 = float(np.clip((r[3] - py) / dh, 0.0, 1.0))
                        if nx2 - nx1 < 0.008 or ny2 - ny1 < 0.015:
                            continue
                        fx1 = cx0 + nx1 * (cx1 - cx0)
                        fy1 = cy0 + ny1 * (cy1 - cy0)
                        fx2 = cx0 + nx2 * (cx1 - cx0)
                        fy2 = cy0 + ny2 * (cy1 - cy0)
                        edge = nx1 < 0.008 or nx2 > 0.992 or ny1 < 0.008 or ny2 > 0.992
                        cands.append((fx1, fy1, fx2, fy2, float(r[4]), edge))

        cands.sort(key=lambda c: c[4] - (0.06 if c[5] else 0.0), reverse=True)
        kept = []
        for c in cands:
            sup = False
            for k in kept:
                iw = max(0.0, min(c[2], k[2]) - max(c[0], k[0]))
                ih = max(0.0, min(c[3], k[3]) - max(c[1], k[1]))
                inter = iw * ih
                aa = (c[2] - c[0]) * (c[3] - c[1])
                ab = (k[2] - k[0]) * (k[3] - k[1])
                iou = inter / max(1e-6, aa + ab - inter)
                iom = inter / max(1e-6, min(aa, ab))
                if iou > 0.42 or iom > 0.60:
                    sup = True
                    break
            if not sup:
                kept.append(c)

        return [(c[0], c[1], c[2], c[3], c[4]) for c in kept if is_on_ice((c[0], c[1], c[2], c[3]), aspect)]


def compute_ice_wb(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    ys = np.linspace(int(0.60 * h), int(0.85 * h), 12, dtype=int)
    xs = np.linspace(int(0.20 * w), int(0.80 * w), 16, dtype=int)
    grid = rgb[np.ix_(ys, xs)].reshape(-1, 3)
    med = np.median(grid, axis=0)
    luma = 0.299 * med[0] + 0.587 * med[1] + 0.114 * med[2]
    if luma < 100.0:
        return np.ones(3, dtype=np.float32)
    mean_ice = float(med.mean())
    return np.clip(mean_ice / np.maximum(1.0, med), 0.70, 1.45).astype(np.float32)


def _baseline_stripe_score(wb_crop: np.ndarray) -> float:
    ch, cw = wb_crop.shape[:2]
    if cw < 4 or ch < 2:
        return 0.0
    luma = 0.299 * wb_crop[..., 0] + 0.587 * wb_crop[..., 1] + 0.114 * wb_crop[..., 2]
    p10, p90 = np.percentile(luma, [10, 90])
    if p90 < 125.0 or 0.5 * (p10 + p90) < 82.0 or (p90 - p10) < 24.0:
        return 0.0
    r_s = np.sort(wb_crop[..., 0].ravel())
    g_s = np.sort(wb_crop[..., 1].ravel())
    b_s = np.sort(wb_crop[..., 2].ravel())
    n = len(r_s)
    lo, hi = (n * 2) // 5 if n >= 5 else 0, max(1, (n * 4) // 5) if n >= 5 else n
    mr, mg, mb = float(r_s[lo:hi].mean()), float(g_s[lo:hi].mean()), float(b_s[lo:hi].mean())
    max_c, min_c = max(mr, mg, mb), min(mr, mg, mb)
    hsv_sat = 0.0 if max_c <= 0 else (max_c - min_c) / max_c
    sat_w = min(max((0.40 - hsv_sat) / 0.18, 0.0), 1.0)
    if sat_w <= 0.0:
        return 0.0
    band_scores = []
    n_bands = min(3, ch)
    for bi in range(n_bands):
        y0, y1 = (bi * ch) // n_bands, max((bi * ch) // n_bands + 1, ((bi + 1) * ch) // n_bands)
        prof = luma[y0:y1].mean(axis=0)
        bins = min(cw, 20)
        if cw > bins:
            sb = np.array([prof[(i * cw) // bins:max((i * cw) // bins + 1, ((i + 1) * cw) // bins)].mean() for i in range(bins)])
        else:
            sb = prof
        rng = float(sb.max() - sb.min())
        if rng < 16.0:
            band_scores.append(0.0)
            continue
        dt = max(7.0, 0.12 * rng)
        last_s, flips = 0, 0
        for i in range(len(sb) - 1):
            d = sb[i + 1] - sb[i]
            if abs(d) >= dt:
                sgn = 1 if d > 0 else -1
                if last_s != 0 and sgn != last_s:
                    flips += 1
                last_s = sgn
        fs = 1.0 if flips >= 3 else (0.35 if flips == 2 else (0.10 if flips == 1 else 0.0))
        cs = min(max((rng - 16.0) / 56.0, 0.0), 1.0)
        mid = 0.5 * (float(sb.min()) + float(sb.max()))
        df = float((sb < mid).mean())
        cb = min(max(min(df, 1.0 - df) / 0.25, 0.0), 1.0)
        band_scores.append(fs * cs * cb)
    band_scores.sort()
    combined = 0.55 * band_scores[0] + 0.45 * band_scores[n_bands // 2]
    mid_l = 0.5 * (p10 + p90)
    df_all = float((luma < mid_l).mean())
    px_bal = min(max(min(df_all, 1.0 - df_all) / 0.30, 0.0), 1.0)
    return float(min(max(combined * sat_w * px_bal, 0.0), 1.0))


def extract_baseline_features(rgb: np.ndarray, ice_wb: np.ndarray, box: Tuple[float, float, float, float]) -> Dict[str, Any]:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px0, px1 = max(0, int((x1 + 0.36 * bw) * w)), min(w, max(1, int((x1 + 0.64 * bw) * w)))
    py0, py1 = max(0, int((y1 + 0.30 * bh) * h)), min(h, max(1, int((y1 + 0.54 * bh) * h)))
    crop = rgb[py0:py1, px0:px1]
    if crop.size == 0:
        return {"chest_rgb": [128, 128, 128], "stripe_score": 0.0}
    wb = np.clip(crop.astype(np.float32) * ice_wb[None, None, :], 0, 255)
    r_s = np.sort(wb[..., 0].ravel())
    g_s = np.sort(wb[..., 1].ravel())
    b_s = np.sort(wb[..., 2].ravel())
    n = len(r_s)
    lo, hi = (n * 2) // 5 if n >= 5 else 0, max(1, (n * 4) // 5) if n >= 5 else n
    chest = [int(round(float(r_s[lo:hi].mean()))), int(round(float(g_s[lo:hi].mean()))), int(round(float(b_s[lo:hi].mean())))]
    return {"chest_rgb": chest, "stripe_score": _baseline_stripe_score(wb)}


def _dft_stripe_score_64x36(rgb: np.ndarray, ice_wb: np.ndarray, box: Tuple[float, float, float, float]) -> float:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px0, px1 = max(0, int((x1 + 0.14 * bw) * w)), min(w, int((x1 + 0.86 * bw) * w))
    py0, py1 = max(0, int((y1 + 0.16 * bh) * h)), min(h, int((y1 + 0.55 * bh) * h))
    if px1 - px0 < 4 or py1 - py0 < 4:
        return 0.0
    roi = np.clip(rgb[py0:py1, px0:px1].astype(np.float32) * ice_wb[None, None, :], 0, 255)
    roi_64 = cv2.resize(roi, (64, 36), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    luma_64 = 0.299 * roi_64[..., 0] + 0.587 * roi_64[..., 1] + 0.114 * roi_64[..., 2]
    p10, p90 = np.percentile(luma_64, [10.0, 90.0])
    if p90 < 122.0 or 0.5 * (p10 + p90) < 78.0 or (p90 - p10) < 22.0:
        return 0.0
    max_c = roi_64.max(axis=-1)
    min_c = roi_64.min(axis=-1)
    spread = max_c - min_c
    sat = np.where(max_c <= 1.0, 0.0, spread / np.maximum(1.0, max_c))
    achromatic_frac = float(((spread < 36.0) | (sat < 0.24)).mean())
    mean_sat = float(sat.mean())
    chroma_gate = min(max((achromatic_frac - 0.45) / 0.35, 0.0), 1.0) * min(max((0.40 - mean_sat) / 0.18, 0.0), 1.0)
    if chroma_gate <= 0.0:
        return 0.0
    mid_luma = 0.5 * (p10 + p90)
    dark_frac = float((luma_64 < mid_luma).mean())
    px_balance = min(max(min(dark_frac, 1.0 - dark_frac) / 0.25, 0.0), 1.0)
    if px_balance <= 0.0:
        return 0.0

    hann = np.hanning(64)
    hann_sum_sq = float((hann * hann).sum())
    rows_centered = luma_64 - luma_64.mean(axis=1, keepdims=True)
    rows_win = rows_centered * hann[None, :]
    ac_energy = (rows_win ** 2).sum(axis=1) / hann_sum_sq
    spec = (np.abs(np.fft.rfft(rows_win, axis=1)) ** 2) * (2.0 / (64.0 * hann_sum_sq))
    low_pow = spec[:, 1:3].sum(axis=1)
    band_pow = spec[:, 3:16].sum(axis=1)
    row_amp = np.sqrt(band_pow)

    group_scores = []
    for g in range(6):
        eb = float(band_pow[g * 6:(g + 1) * 6].mean())
        el = float(low_pow[g * 6:(g + 1) * 6].mean())
        etot = max(1e-6, float(ac_energy[g * 6:(g + 1) * 6].mean()))
        ratio_band = eb / max(1e-6, eb + el)
        ratio_total = eb / etot
        amp = math.sqrt(eb)
        gs = (
            min(max((ratio_band - 0.22) / 0.36, 0.0), 1.0)
            * min(max((ratio_total - 0.20) / 0.35, 0.0), 1.0)
            * min(max(amp / 6.5, 0.0), 1.0)
        )
        group_scores.append(gs)
    group_scores.sort()
    band_consistency = 0.55 * group_scores[1] + 0.45 * group_scores[2]
    if band_consistency <= 0.0:
        return 0.0

    ref_amp = float(np.percentile(row_amp, 90.0))
    if ref_amp <= 1e-4:
        return 0.0
    extent = float((row_amp >= 0.40 * ref_amp).mean())
    extent_score = min(max((extent - 0.50) / 0.30, 0.0), 1.0)
    if extent_score <= 0.0:
        return 0.0
    return float(min(max(math.sqrt(band_consistency * extent_score * px_balance) * chroma_gate, 0.0), 1.0))


def _two_means_lab(X: np.ndarray, iters: int = 6) -> Tuple[np.ndarray, np.ndarray, float]:
    n = len(X)
    total = X.sum(axis=0)
    c0 = X[int(np.argmin(X[:, 0]))].copy()
    c1 = X[int(np.argmax(X[:, 0]))].copy()
    side = np.zeros(n, dtype=bool)
    for _ in range(iters):
        side = (X @ (c1 - c0)) > 0.5 * float(c1 @ c1 - c0 @ c0)
        n1 = int(side.sum())
        if n1 == 0 or n1 == n:
            break
        s1 = X[side].sum(axis=0)
        new0, new1 = (total - s1) / (n - n1), s1 / n1
        if max(np.abs(new0 - c0).max(), np.abs(new1 - c1).max()) < 0.05:
            c0, c1 = new0, new1
            break
        c0, c1 = new0, new1
    n1 = int(side.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        m = total / max(1, n)
        return m, m, (1.0 if m[0] >= 62.0 else 0.0)
    if c0[0] <= c1[0]:
        return c0, c1, n1 / n
    return c1, c0, n0 / n


def _find_torso_center_frac(
    rgb: np.ndarray,
    ice_wb: np.ndarray,
    x1: float,
    y1: float,
    bw: float,
    bh: float,
) -> float:
    h, w = rgb.shape[:2]
    bw_px = max(1, int(round(bw * w)))
    bh_px = max(1, int(round(bh * h)))
    if bw_px <= 0.65 * bh_px or bw_px < 10:
        return 0.50
    bx0 = max(0, int(x1 * w))
    bx1 = min(w, bx0 + bw_px)
    by0 = max(0, int((y1 + 0.20 * bh) * h))
    by1 = min(h, max(by0 + 1, int((y1 + 0.56 * bh) * h)))
    strip = rgb[by0:by1, bx0:bx1]
    if strip.size == 0 or strip.shape[1] < 10:
        return 0.50
    sw = strip.shape[1]
    wb_s = np.clip(strip.astype(np.float32) * ice_wb[None, None, :], 0.0, 255.0)
    col_luma = (0.299 * wb_s[..., 0] + 0.587 * wb_s[..., 1] + 0.114 * wb_s[..., 2]).mean(axis=0)
    win_w = min(sw, max(3, int(round(0.42 * sw))))
    min_x = max(0, min(sw - win_w, int(0.04 * sw)))
    max_x = max(min_x, sw - win_w - int(0.04 * sw))
    if max_x <= min_x:
        return 0.50
    best_score = float("inf")
    best_center = 0.50
    for sx0 in range(min_x, max_x + 1):
        mean_lum = float(col_luma[sx0:sx0 + win_w].mean())
        c_frac = (sx0 + 0.5 * win_w) / float(sw)
        score = mean_lum + 20.0 * abs(c_frac - 0.50)
        if score < best_score:
            best_score = score
            best_center = c_frac
    return min(max(best_center, 0.24), 0.76)


def extract_sota_features(rgb: np.ndarray, ice_wb: np.ndarray, box: Tuple[float, float, float, float]) -> Dict[str, Any]:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    bw_px = max(1, int(round(bw * w)))
    bh_px = max(1, int(round(bh * h)))
    wide_stick = bw_px > 0.65 * bh_px
    torso_center = _find_torso_center_frac(rgb, ice_wb, x1, y1, bw, bh)

    # 1. Baseline-compatible chest RGB centered on torsoCenterFrac (+/- 14% x, 30..54% y)
    cx0, cx1 = max(0, int((x1 + (torso_center - 0.14) * bw) * w)), min(w, max(1, int((x1 + (torso_center + 0.14) * bw) * w)))
    cy0, cy1 = max(0, int((y1 + 0.30 * bh) * h)), min(h, max(1, int((y1 + 0.54 * bh) * h)))
    c_crop = rgb[cy0:cy1, cx0:cx1]
    if c_crop.size > 0:
        wb_c = np.clip(c_crop.astype(np.float32) * ice_wb[None, None, :], 0, 255)
        r_s, g_s, b_s = np.sort(wb_c[..., 0].ravel()), np.sort(wb_c[..., 1].ravel()), np.sort(wb_c[..., 2].ravel())
        n = len(r_s)
        lo, hi = (n * 2) // 5 if n >= 5 else 0, max(1, (n * 4) // 5) if n >= 5 else n
        chest_rgb = [int(round(float(r_s[lo:hi].mean()))), int(round(float(g_s[lo:hi].mean()))), int(round(float(b_s[lo:hi].mean())))]
    else:
        chest_rgb = [128, 128, 128]

    # 2. Order-free 2-means CIELAB on torso (centered on torsoCenterFrac) with border ice BG subtraction
    half_w = 0.21 if wide_stick else 0.24
    tx0, tx1 = max(0, int((x1 + (torso_center - half_w) * bw) * w)), min(w, max(1, int((x1 + (torso_center + half_w) * bw) * w)))
    ty0, ty1 = max(0, int((y1 + 0.22 * bh) * h)), min(h, max(1, int((y1 + 0.56 * bh) * h)))
    t_crop = rgb[ty0:ty1, tx0:tx1]
    kit7 = None
    if t_crop.size >= 12:
        wb_t = np.clip(t_crop.astype(np.float32) * ice_wb[None, None, :], 0, 255)
        if max(wb_t.shape[:2]) > 32:
            s = 32.0 / max(wb_t.shape[:2])
            wb_t = cv2.resize(wb_t, (max(2, int(round(wb_t.shape[1] * s))), max(2, int(round(wb_t.shape[0] * s)))), interpolation=cv2.INTER_AREA)
        t_lab = srgb_to_lab(wb_t.reshape(-1, 3))
        # Border ice background estimation (0..10% & 90..100% x, 15..55% y)
        by0, by1 = max(0, int((y1 + 0.15 * bh) * h)), min(h, max(1, int((y1 + 0.55 * bh) * h)))
        bx_l0, bx_l1 = max(0, int((x1 + 0.00 * bw) * w)), min(w, max(1, int((x1 + 0.10 * bw) * w)))
        bx_r0, bx_r1 = max(0, int((x1 + 0.90 * bw) * w)), min(w, max(1, int((x1 + 1.00 * bw) * w)))
        bg_pts = []
        if by1 > by0 and bx_l1 > bx_l0:
            bg_pts.append(rgb[by0:by1, bx_l0:bx_l1].reshape(-1, 3))
        if by1 > by0 and bx_r1 > bx_r0:
            bg_pts.append(rgb[by0:by1, bx_r0:bx_r1].reshape(-1, 3))
        if bg_pts:
            bg_wb = np.clip(np.concatenate(bg_pts, axis=0).astype(np.float32) * ice_wb[None, :], 0, 255)
            bg_lab_pts = srgb_to_lab(bg_wb)
            bg_chroma = np.hypot(bg_lab_pts[:, 1], bg_lab_pts[:, 2])
            ice_ref_pts = bg_lab_pts[(bg_lab_pts[:, 0] >= 48.0) & (bg_chroma < 22.0)]
            if len(ice_ref_pts) >= max(4, len(bg_lab_pts) // 4):
                bg_lab = np.median(ice_ref_pts, axis=0)
                de = np.linalg.norm(t_lab - bg_lab[None, :], axis=1)
                bg_like = de < 11.0
                ice_frac = float(bg_like.mean())
                rem_pts = t_lab[~bg_like]
                max_drop = 0.75 if wide_stick else 0.40
                min_rem = 10 if wide_stick else max(4, int(0.35 * len(t_lab)))
                allow_wide = (not wide_stick) or ice_frac <= 0.40 or (len(rem_pts) >= 10 and float(rem_pts[:, 0].mean()) < 45.0)
                if allow_wide and 0.0 < ice_frac <= max_drop and len(rem_pts) >= min_rem:
                    t_lab = rem_pts
        dark, light, light_frac = _two_means_lab(t_lab)
        kit7 = np.array([dark[0], dark[1], dark[2], light[0], light[1], light[2], light_frac * 100.0], dtype=np.float64)

    # 3. lowerBellyLuma centered on torsoCenterFrac (+/- 18% x, 42..56% y, 40th-80th percentile)
    bx0, bx1 = max(0, int((x1 + (torso_center - 0.18) * bw) * w)), min(w, max(1, int((x1 + (torso_center + 0.18) * bw) * w)))
    by0, by1 = max(0, int((y1 + 0.42 * bh) * h)), min(h, max(1, int((y1 + 0.56 * bh) * h)))
    b_crop = rgb[by0:by1, bx0:bx1]
    belly_luma = float("nan")
    if b_crop.size >= 3:
        wb_b = np.clip(b_crop.astype(np.float32) * ice_wb[None, None, :], 0, 255)
        bl = np.sort((0.299 * wb_b[..., 0] + 0.587 * wb_b[..., 1] + 0.114 * wb_b[..., 2]).ravel())
        lo, hi = (len(bl) * 2) // 5 if len(bl) >= 5 else 0, max(1, (len(bl) * 4) // 5) if len(bl) >= 5 else len(bl)
        belly_luma = float(bl[lo:hi].mean())

    # 4. pantsDarkness (28..72% x, 56..84% y, luma < 72 & chromaSpread < 24)
    px0, px1 = max(0, int((x1 + 0.28 * bw) * w)), min(w, max(1, int((x1 + 0.72 * bw) * w)))
    py0, py1 = max(0, int((y1 + 0.56 * bh) * h)), min(h, max(1, int((y1 + 0.84 * bh) * h)))
    p_crop = rgb[py0:py1, px0:px1]
    pants_dark = float("nan")
    if p_crop.size >= 3:
        wb_p = np.clip(p_crop.astype(np.float32) * ice_wb[None, None, :], 0, 255)
        pl = 0.299 * wb_p[..., 0] + 0.587 * wb_p[..., 1] + 0.114 * wb_p[..., 2]
        spread = wb_p.max(axis=-1) - wb_p.min(axis=-1)
        pants_dark = float(((pl < 72.0) & (spread < 24.0)).mean())

    # 5. armbandScore (sleeves 0..25% & 75..100% x, 15..45% y, HSV orange)
    ay0, ay1 = max(0, int((y1 + 0.15 * bh) * h)), min(h, max(1, int((y1 + 0.45 * bh) * h)))
    ax_l0, ax_l1 = max(0, int((x1 + 0.00 * bw) * w)), min(w, max(1, int((x1 + 0.25 * bw) * w)))
    ax_r0, ax_r1 = max(0, int((x1 + 0.75 * bw) * w)), min(w, max(1, int((x1 + 1.00 * bw) * w)))
    arm_pts = []
    if ay1 > ay0 and ax_l1 > ax_l0:
        arm_pts.append(rgb[ay0:ay1, ax_l0:ax_l1].reshape(-1, 3))
    if ay1 > ay0 and ax_r1 > ax_r0:
        arm_pts.append(rgb[ay0:ay1, ax_r0:ax_r1].reshape(-1, 3))
    armband = 0.0
    if arm_pts:
        arm_wb = np.clip(np.concatenate(arm_pts, axis=0).astype(np.float32) * ice_wb[None, :], 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(arm_wb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        orange = (hsv[:, 0] >= 2) & (hsv[:, 0] <= 13) & (hsv[:, 1] >= 115) & (hsv[:, 2] >= 102)
        armband = float(orange.mean())

    stripe = _dft_stripe_score_64x36(rgb, ice_wb, box)
    return {
        "chest_rgb": chest_rgb,
        "kit_summary_lab7": kit7,
        "lower_belly_luma": belly_luma,
        "pants_darkness": pants_dark,
        "armband_score": armband,
        "stripe_score": stripe,
    }


# ---------------------------------------------------------------------------
# ONNX STR Recognizer (exact match to JerseyStrOnnxRecognizer contract)
# ---------------------------------------------------------------------------

class JerseyStrRecognizer:
    def __init__(self, onnx_path: Path) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    def infer(self, rgb: np.ndarray, ice_wb: np.ndarray, box: Tuple[float, float, float, float]) -> Dict[str, Any]:
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cx0 = max(0, int((x1 - 0.06 * bw) * w))
        cx1 = min(w, max(cx0 + 1, int((x2 + 0.06 * bw) * w)))
        cy0 = max(0, int((y1 + 0.10 * bh) * h))
        cy1 = min(h, max(cy0 + 1, int((y1 + 0.62 * bh) * h)))
        crop = rgb[cy0:cy1, cx0:cx1]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return {"legibility": 0.0, "back": 0.0, "sharpness": 0.0, "candidates": [], "partial_left": None, "partial_right": None}

        ch, cw = crop.shape[:2]
        scale = min(96.0 / cw, 96.0 / ch)
        nw, nh = max(1, min(96, int(round(cw * scale)))), max(1, min(96, int(round(ch * scale))))
        pad_x, pad_y = (96 - nw) // 2, (96 - nh) // 2
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Match JerseyStrOnnxRecognizer.kt: WB + 2%..98% local contrast stretch inside content rect only
        wb_content = np.clip(resized.astype(np.float32) * ice_wb[None, None, :], 0.0, 255.0)
        luma = 0.299 * wb_content[..., 0] + 0.587 * wb_content[..., 1] + 0.114 * wb_content[..., 2]
        p_low, p_high = np.percentile(luma, [2.0, 98.0])
        raw_span = max(32.0, float(p_high - p_low))
        gain = min(max(1.0 + 0.12 * min(max(220.0 / raw_span, 0.8), 2.0), 1.05), 1.25)
        mid = 0.5 * float(p_low + p_high)
        sharpness = float(cv2.Laplacian(luma.astype(np.float32), cv2.CV_32F).var())
        adj = np.clip((wb_content - mid) * gain + mid, 0.0, 255.0) / 255.0

        canvas = np.full((96, 96, 3), 114.0 / 255.0, dtype=np.float32)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = adj

        inp = canvas.transpose(2, 0, 1)[None]
        leg_l, back_l, num_l, ld_l, rd_l = self.sess.run(None, {"images": inp})

        leg_p = float(1.0 / (1.0 + math.exp(-min(max(float(leg_l[0, 0]), -20.0), 20.0))))
        back_p = float(1.0 / (1.0 + math.exp(-min(max(float(back_l[0, 0]), -20.0), 20.0))))

        num_v = num_l[0] - num_l[0].max()
        num_e = np.exp(num_v)
        num_p = num_e / num_e.sum()

        ld_v = ld_l[0] - ld_l[0].max()
        ld_p = np.exp(ld_v) / np.exp(ld_v).sum()
        rd_v = rd_l[0] - rd_l[0].max()
        rd_p = np.exp(rd_v) / np.exp(rd_v).sum()

        cands: List[Tuple[str, float]] = []
        if leg_p >= 0.55 and float(num_p[0]) < 0.60:
            indexed = [(n, float(num_p[n])) for n in range(1, 100) if float(num_p[n]) >= 0.02]
            indexed.sort(key=lambda kv: -kv[1])
            cands = [(str(n), p) for n, p in indexed[:5]]

        p_left = None
        p_right = None
        l_best = int(1 + np.argmax(ld_p[1:10]))
        l_best_p = float(ld_p[l_best])
        if (
            leg_p >= 0.30
            and l_best_p >= 0.45
            and l_best_p > float(ld_p[10]) * 1.15
            and (float(rd_p[10]) >= 0.28 or leg_p < 0.55)
            and float(num_p[l_best]) < 0.25
        ):
            p_left = (str(l_best), l_best_p)

        r_best = int(np.argmax(rd_p[:10]))
        r_best_p = float(rd_p[r_best])
        if (
            leg_p >= 0.30
            and r_best_p >= 0.45
            and r_best_p > float(rd_p[10]) * 1.15
            and (float(ld_p[10]) >= 0.28 or leg_p < 0.60)
        ):
            p_right = (str(r_best), r_best_p)

        return {
            "legibility": leg_p,
            "back": back_p,
            "sharpness": sharpness,
            "candidates": cands,
            "partial_left": p_left,
            "partial_right": p_right,
        }


# ---------------------------------------------------------------------------
# Roster-Free Bayesian + Hungarian Number Solver (mirror of JerseyNumberPosterior.kt)
# ---------------------------------------------------------------------------

ALL_NUMS = [str(i) for i in range(1, 100)]
NUM_IDX = {s: i for i, s in enumerate(ALL_NUMS)}
HIDE_P = 0.38
SPURIOUS_P = 0.05
P_CORRECT = 0.85
DUPLICATE_PRIOR = 0.04
UNASSIGN_PRIOR = 0.08
MIN_POSTERIOR_PROB = 0.48
MIN_EVIDENCE_WEIGHT = 0.90
EVIDENCE_BIN_SEC = 1.0
SAME_BIN_MARGINAL_WEIGHT = 0.15
SINGLE_BIN_COMMIT_MIN_CONF = 0.80


def _obs_channel(obs: Dict[str, Any]) -> int:
    if obs.get("candidates"):
        return 0
    k = obs.get("kind", "FULL")
    if k == "LEFT_PARTIAL":
        return 1
    if k == "RIGHT_PARTIAL":
        return 2
    if k == "SLEEVE":
        return 3
    return 0


def _obs_raw_strength(obs: Dict[str, Any]) -> float:
    qw = min(max(float(obs.get("weight", 1.0)), 0.05), 1.5)
    sw = float(obs.get("source_weight", 1.6))
    cands = obs.get("candidates")
    if cands:
        conf = max((float(pr) for _, pr in cands if math.isfinite(float(pr))), default=0.0)
    elif obs.get("number") and math.isfinite(float(obs.get("conf", 0.0))):
        conf = float(obs["conf"])
    else:
        conf = 0.0
    if conf <= 0.02:
        return 0.0
    return min(max(conf, 0.0), 1.0) * qw * sw


def _compute_effective_obs_weights(observations: List[Dict[str, Any]]) -> List[float]:
    if not observations:
        return []
    best_idx_by_key: Dict[Tuple[int, int], int] = {}
    best_str_by_key: Dict[Tuple[int, int], float] = {}
    for i, obs in enumerate(observations):
        st = _obs_raw_strength(obs)
        if st <= 0.0 or obs.get("is_nano"):
            continue
        t_sec = float(obs.get("timestamp_sec", 0.0))
        if not math.isfinite(t_sec):
            t_sec = 0.0
        key = (math.floor(t_sec / EVIDENCE_BIN_SEC), _obs_channel(obs))
        if key not in best_str_by_key or st > best_str_by_key[key]:
            best_str_by_key[key] = st
            best_idx_by_key[key] = i
    out: List[float] = []
    for i, obs in enumerate(observations):
        qw = min(max(float(obs.get("weight", 1.0)), 0.05), 1.5)
        sw = float(obs.get("source_weight", 1.6))
        base_w = qw * sw
        if obs.get("is_nano"):
            out.append(base_w)
            continue
        t_sec = float(obs.get("timestamp_sec", 0.0))
        if not math.isfinite(t_sec):
            t_sec = 0.0
        key = (math.floor(t_sec / EVIDENCE_BIN_SEC), _obs_channel(obs))
        disc = 1.0 if best_idx_by_key.get(key, i) == i else SAME_BIN_MARGINAL_WEIGHT
        out.append(base_w * disc)
    return out


def _meets_independent_bin_or_high_conf_gate(observations: List[Dict[str, Any]], candidate_num: Optional[str], is_nano: bool = False) -> bool:
    if candidate_num is None:
        return False
    if is_nano:
        return True
    bins = set()
    max_conf = 0.0
    for obs in observations:
        t_sec = float(obs.get("timestamp_sec", 0.0))
        if not math.isfinite(t_sec):
            t_sec = 0.0
        b_idx = math.floor(t_sec / EVIDENCE_BIN_SEC)
        cands = obs.get("candidates")
        if cands:
            c_prob = sum(float(pr) for raw, pr in cands if raw == candidate_num and math.isfinite(float(pr)) and float(pr) > 0.0)
            if c_prob >= 0.25:
                bins.add(b_idx)
                max_conf = max(max_conf, min(c_prob, 1.0))
        elif obs.get("number") and math.isfinite(float(obs.get("conf", 0.0))) and float(obs["conf"]) > 0.0:
            kind = obs.get("kind", "FULL")
            num = str(obs["number"])
            matches = (
                (kind in ("FULL", "SINGLE") and num == candidate_num)
                or (kind == "LEFT_PARTIAL" and len(candidate_num) == 2 and num == candidate_num[0])
                or (kind == "RIGHT_PARTIAL" and len(candidate_num) == 2 and num == candidate_num[1])
                or (kind == "SLEEVE" and num in candidate_num)
            )
            if matches:
                bins.add(b_idx)
                max_conf = max(max_conf, min(float(obs["conf"]), 1.0))
    return len(bins) >= 2 or max_conf >= SINGLE_BIN_COMMIT_MIN_CONF


def _build_confusion() -> np.ndarray:
    m = np.zeros((10, 10), dtype=np.float64)
    pairs = [(1, 7, 0.050), (3, 8, 0.045), (6, 8, 0.035), (6, 9, 0.030), (8, 9, 0.030),
             (3, 9, 0.025), (0, 8, 0.035), (2, 7, 0.025), (4, 1, 0.025)]
    for a, b, w in pairs:
        m[a, b] += w
        m[b, a] += w
    for d in range(10):
        off = float(m[d].sum())
        rest = max(0.0, 1.0 - P_CORRECT - off) / 9.0
        for r in range(10):
            if r != d:
                m[d, r] += rest
        m[d, d] = max(0.0, 1.0 - float(m[d].sum()))
    return m


CONF_MAT = _build_confusion()


def _confusion_neighbors() -> List[List[int]]:
    pairs = {(1, 7), (7, 1), (3, 8), (8, 3), (6, 8), (8, 6), (6, 9), (9, 6), (8, 9), (9, 8),
             (3, 9), (9, 3), (0, 8), (8, 0), (2, 7), (7, 2), (4, 1), (1, 4)}
    out = []
    for i, s in enumerate(ALL_NUMS):
        nb = []
        for j, t in enumerate(ALL_NUMS):
            if i != j and len(s) == len(t):
                diffs = [k for k in range(len(s)) if s[k] != t[k]]
                if len(diffs) == 1 and (int(s[diffs[0]]), int(t[diffs[0]])) in pairs:
                    nb.append(j)
        out.append(nb)
    return out


CONF_NEIGHBORS = _confusion_neighbors()


def solve_team_numbers_bayesian_hungarian(chains: List[Dict[str, Any]]) -> Dict[int, Tuple[Optional[str], float]]:
    if not chains:
        return {}
    log_ps: Dict[int, np.ndarray] = {}
    probs_map: Dict[int, np.ndarray] = {}
    assigned: Dict[int, Optional[str]] = {}
    pinned = set()

    for ch in chains:
        cid = ch["chain_id"]
        lp = np.full(99, math.log(1.0 / 99.0), dtype=np.float64)
        ev_w = 0.0
        eff_weights = _compute_effective_obs_weights(ch["observations"])
        for obs, eff in zip(ch["observations"], eff_weights):
            if eff <= 0.0:
                continue
            cands = obs.get("candidates")
            if cands:
                cmap: Dict[int, float] = {}
                s = 0.0
                for raw, pr in cands:
                    if raw in NUM_IDX and pr > 0.0:
                        cmap[NUM_IDX[raw]] = cmap.get(NUM_IDX[raw], 0.0) + pr
                        s += pr
                if cmap and s > 1e-6:
                    if s > 1.0:
                        for k in cmap:
                            cmap[k] /= s
                        s = 1.0
                    rem = min(max(1.0 - s, 0.01), 0.99)
                    per_other = rem / max(1, 99 - len(cmap))
                    base = np.array([cmap.get(i, per_other) for i in range(99)], dtype=np.float64)
                    sm = np.empty(99, dtype=np.float64)
                    for i in range(99):
                        nb = CONF_NEIGHBORS[i]
                        s0 = 0.92 * base[i] + 0.08 * (float(base[nb].mean()) if nb else base[i])
                        sm[i] = max(0.92 * s0 + 0.08 * (1.0 / 99.0), 1e-6)
                    lp += eff * np.log(sm)
                    ev_w += max(cmap.values()) * eff
            elif obs.get("number"):
                r = obs["number"]
                c = min(max(obs["conf"], 0.0), 0.99)
                kind = obs["kind"]
                lk = np.empty(99, dtype=np.float64)
                for idx, cand in enumerate(ALL_NUMS):
                    if len(r) == 2:
                        noise = 1.0 / 99.0
                        if len(cand) == 2:
                            pm = CONF_MAT[int(cand[0]), int(r[0])] * CONF_MAT[int(cand[1]), int(r[1])]
                        else:
                            pm = SPURIOUS_P * 0.5 * (CONF_MAT[int(cand), int(r[0])] + CONF_MAT[int(cand), int(r[1])]) * 0.1
                    else:
                        noise = 1.0 / 10.0
                        d = int(r[0])
                        if len(cand) == 1:
                            b = CONF_MAT[int(cand), d]
                            pm = b * (1.0 - 0.3 * HIDE_P) if kind in ("FULL", "SINGLE") else 0.25 * b
                        else:
                            l_p, r_p = CONF_MAT[int(cand[0]), d], CONF_MAT[int(cand[1]), d]
                            if kind == "RIGHT_PARTIAL":
                                pm = HIDE_P * r_p + 0.05 * l_p
                            elif kind == "LEFT_PARTIAL":
                                pm = HIDE_P * l_p + 0.05 * r_p
                            else:
                                pm = 0.3 * HIDE_P * 0.5 * (l_p + r_p)
                    lk[idx] = max(c * pm + (1.0 - c) * noise, 1e-12)
                lp += eff * np.log(lk)
                ev_w += c * eff

        log_ps[cid] = lp
        w = np.exp(lp - lp.max())
        p = w / w.sum()
        probs_map[cid] = p
        best_i = int(np.argmax(p))
        best_p = float(p[best_i])
        best_num = ALL_NUMS[best_i]
        if (
            ch["observations"]
            and best_p >= MIN_POSTERIOR_PROB
            and ev_w >= MIN_EVIDENCE_WEIGHT
            and _meets_independent_bin_or_high_conf_gate(ch["observations"], best_num)
        ):
            assigned[cid] = best_num
            bins = set()
            for obs in ch["observations"]:
                t_sec = float(obs.get("timestamp_sec", 0.0))
                b_idx = math.floor(t_sec / 10.0)
                cands = obs.get("candidates")
                if cands:
                    best_c = max(cands, key=lambda kv: kv[1])
                    if best_c[0] == assigned[cid] and best_c[1] >= 0.85:
                        bins.add(b_idx)
                elif obs.get("number") == assigned[cid] and obs.get("kind", "FULL") in ("FULL", "SINGLE") and obs.get("conf", 0.0) >= 0.85:
                    bins.add(b_idx)
            if len(bins) >= 2:
                pinned.add(cid)
        else:
            assigned[cid] = None

    # Concurrent overlap graph
    conc: Dict[int, set] = {ch["chain_id"]: set() for ch in chains}
    for i, a in enumerate(chains):
        for b in chains[i + 1:]:
            ov = min(a["last_sec"], b["last_sec"]) - max(a["first_sec"], b["first_sec"])
            min_sp = min(max(0.0, a["last_sec"] - a["first_sec"]), max(0.0, b["last_sec"] - b["first_sec"]))
            if ov > max(1.5, 0.12 * min_sp):
                conc[a["chain_id"]].add(b["chain_id"])
                conc[b["chain_id"]].add(a["chain_id"])

    kept_dups = set()
    ids = sorted(assigned.keys())
    for _ in range(50):
        clash = None
        for a in ids:
            if assigned[a] is None:
                continue
            for b in sorted(conc[a]):
                if b > a and assigned[b] == assigned[a] and (a, b) not in kept_dups and not (a in pinned and b in pinned):
                    clash = (a, b)
                    break
            if clash:
                break
        if not clash:
            break
        a, b = clash
        clique = [a, b]
        for g in sorted(conc[a] & conc[b]):
            if all(g in conc[h] for h in clique):
                clique.append(g)
        group = [g for g in clique if g not in pinned]
        if not group:
            kept_dups.add((a, b))
            continue
        cands_num: List[str] = []
        for g in group:
            if assigned[g] and assigned[g] not in cands_num:
                cands_num.append(assigned[g])
            for idx in np.argsort(-probs_map[g])[:6]:
                if probs_map[g][idx] >= 0.005 and ALL_NUMS[idx] not in cands_num:
                    cands_num.append(ALL_NUMS[idx])
        n_rows = len(group)
        n_cands = len(cands_num)
        cost = np.full((n_rows, n_cands + 2 * n_rows), 1e6, dtype=np.float64)
        unassign_cost = -math.log(UNASSIGN_PRIOR)
        for i, g in enumerate(group):
            p = probs_map[g]
            blocked = {assigned[h] for h in conc[g] if h not in group and assigned[h] is not None}
            for j, num in enumerate(cands_num):
                if num not in blocked:
                    cost[i, j] = -math.log(max(float(p[NUM_IDX[num]]), 1e-9))
            cost[i, n_cands + i] = unassign_cost
            own = assigned[g]
            kp = float(p[NUM_IDX[own]]) * DUPLICATE_PRIOR if own else 1e-4
            cost[i, n_cands + n_rows + i] = -math.log(max(kp, 1e-9))
        before = dict(assigned)
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if c < n_cands:
                assigned[group[r]] = cands_num[c]
            elif c < n_cands + n_rows:
                assigned[group[r]] = None
        for g in group:
            if assigned[g] != before[g] and assigned[g] is not None:
                taken = {assigned[h] for h in conc[g] if assigned[h] and assigned[h] != assigned[g]}
                mask = np.array([ALL_NUMS[k] not in taken for k in range(99)])
                lp_ex = log_ps[g]
                w_ex = np.where(mask, np.exp(lp_ex - lp_ex[mask].max()), 0.0)
                p_ex = float(w_ex[NUM_IDX[assigned[g]]] / max(w_ex.sum(), 1e-12))
                if p_ex < 0.05:
                    prev_num = before[g]
                    prev_taken = prev_num is not None and any(assigned[h] == prev_num for h in conc[g])
                    assigned[g] = None if prev_taken else prev_num
        if assigned[a] == assigned[b]:
            kept_dups.add((a, b))

    out: Dict[int, Tuple[Optional[str], float]] = {}
    for g in ids:
        num = assigned[g]
        if num is None:
            out[g] = (None, 0.0)
        else:
            taken = {assigned[h] for h in conc[g] if assigned[h] and assigned[h] != num}
            mask = np.array([ALL_NUMS[k] not in taken for k in range(99)])
            lp_ex = log_ps[g]
            w_ex = np.where(mask, np.exp(lp_ex - lp_ex[mask].max()), 0.0)
            p_ex = float(w_ex[NUM_IDX[num]] / max(w_ex.sum(), 1e-12))
            out[g] = (num if p_ex >= MIN_POSTERIOR_PROB else None, p_ex if p_ex >= MIN_POSTERIOR_PROB else 0.0)
    return out


# ---------------------------------------------------------------------------
# MultiTeamPlayerTracker Python Twin (Baseline vs. SOTA)
# ---------------------------------------------------------------------------

def _box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _rgb_luma(rgb: np.ndarray) -> float:
    return float(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])


def _color_lum_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    dl = _rgb_luma(a) - _rgb_luma(b)
    return float(math.sqrt(float((d * d).sum()) + 1.5 * dl * dl))


class LocalTracker:
    def __init__(self, sota_mode: bool = True) -> None:
        self.sota_mode = sota_mode
        self.active: List[Dict[str, Any]] = []
        self.records: Dict[int, Dict[str, Any]] = {}
        self.next_id = 1
        self.last_ts = float("nan")
        self.frame_history: List[Tuple[float, List[Dict[str, Any]]]] = []
        self.team_kit7: Dict[str, np.ndarray] = {}
        self.team_rgb: Dict[str, np.ndarray] = {}
        self.team_belly: Dict[str, float] = {}

    def _is_cross_kit_incompatible(self, tid: int, track_rgb: np.ndarray, det: Dict[str, Any], rgb_dist: float) -> bool:
        det_rgb = np.array(det["chest_rgb"], dtype=np.float64)
        lum_diff = abs(_rgb_luma(track_rgb) - _rgb_luma(det_rgb))
        det_kit7 = det.get("kit_summary_lab7")
        if det_kit7 is None:
            lab = srgb_to_lab(det_rgb)
            det_kit7 = np.array([lab[0], lab[1], lab[2], lab[0], lab[1], lab[2], 100.0 if lab[0] >= 62.0 else 0.0], dtype=np.float64)
        det_belly = float(det.get("lower_belly_luma", float("nan")))
        rec = self.records[tid]
        track_kit7 = self._mean_kit7(rec, track_rgb)
        track_belly = rec["sum_belly"] / rec["n_belly"] if rec["n_belly"] > 0 else float("nan")

        if "TEAM_A" in self.team_kit7 and "TEAM_B" in self.team_kit7 and "TEAM_A" in self.team_rgb and "TEAM_B" in self.team_rgb:
            kitA, kitB = self.team_kit7["TEAM_A"], self.team_kit7["TEAM_B"]
            centA, centB = self.team_rgb["TEAM_A"], self.team_rgb["TEAM_B"]
            bA, bB = self.team_belly.get("TEAM_A", float("nan")), self.team_belly.get("TEAM_B", float("nan"))
            kit_sep = math.sqrt(order_free_kit_dist_sq(kitA, kitB))
            rgb_sep = _color_lum_dist(centA, centB)
            if kit_sep >= 12.0 or rgb_sep >= 40.0:
                t_is_a = kit_and_belly_dist_sq(track_kit7, track_belly, kitA, bA) <= kit_and_belly_dist_sq(track_kit7, track_belly, kitB, bB)
                d_is_a = kit_and_belly_dist_sq(det_kit7, det_belly, kitA, bA) <= kit_and_belly_dist_sq(det_kit7, det_belly, kitB, bB)
                if t_is_a != d_is_a:
                    kit_cross = kit_and_belly_dist_sq(track_kit7, track_belly, det_kit7, det_belly)
                    if rgb_dist > 65.0 or kit_cross > 250.0:
                        return True
        return rgb_dist > 80.0 and lum_diff > 42.0

    def update(self, ts: float, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not math.isnan(self.last_ts) and (ts - self.last_ts > 3.0 or ts - self.last_ts < -1.0):
            self.active.clear()
        self.last_ts = ts

        preds = []
        for t in self.active:
            dt = min(max(ts - t["last_sec"], 0.0), 1.0)
            bx = t["box"]
            preds.append((
                min(max(bx[0] + t["vx"] * dt, 0.0), 1.0),
                min(max(bx[1] + t["vy"] * dt, 0.0), 1.0),
                min(max(bx[2] + t["vx"] * dt, 0.0), 1.0),
                min(max(bx[3] + t["vy"] * dt, 0.0), 1.0),
            ))

        cands = []
        for ti, t in enumerate(self.active):
            pr = preds[ti]
            pcx, pfy = 0.5 * (pr[0] + pr[2]), pr[3]
            ph = pr[3] - pr[1]
            for di, det in enumerate(dets):
                db = det["box"]
                iou = _box_iou(pr, db)
                dcx, dfy = 0.5 * (db[0] + db[2]), db[3]
                dh = db[3] - db[1]
                avg_h = max(0.025, 0.5 * (ph + dh))
                fdist = math.hypot(pcx - dcx, pfy - dfy)
                nfoot = fdist / avg_h
                if iou < 0.10 and (nfoot > 2.5 or fdist > 0.20):
                    continue
                cdist = float(np.linalg.norm(t["ema_rgb"] - np.array(det["chest_rgb"], dtype=np.float64)))
                if self.sota_mode and self._is_cross_kit_incompatible(t["id"], t["ema_rgb"], det, cdist):
                    continue
                cands.append((nfoot - 2.0 * iou + 0.0025 * cdist, ti, di))
        cands.sort(key=lambda x: x[0])

        det_to_tid = [-1] * len(dets)
        matched_t = [False] * len(self.active)
        for _, ti, di in cands:
            if matched_t[ti] or det_to_tid[di] >= 0:
                continue
            matched_t[ti] = True
            t = self.active[ti]
            det = dets[di]
            dt = max(ts - t["last_sec"], 0.0)
            if dt > 1e-3:
                ocx, ofy = 0.5 * (t["box"][0] + t["box"][2]), t["box"][3]
                ncx, nfy = 0.5 * (det["box"][0] + det["box"][2]), det["box"][3]
                t["vx"] = 0.6 * t["vx"] + 0.4 * ((ncx - ocx) / dt)
                t["vy"] = 0.6 * t["vy"] + 0.4 * ((nfy - ofy) / dt)
            t["box"] = det["box"]
            t["last_sec"] = ts
            t["misses"] = 0
            t["hits"] += 1
            t["ema_rgb"] = 0.8 * t["ema_rgb"] + 0.2 * np.array(det["chest_rgb"], dtype=np.float64)
            det_to_tid[di] = t["id"]
            self._update_record(t["id"], ts, det)

        for ti in range(len(self.active) - 1, -1, -1):
            if not matched_t[ti]:
                self.active[ti]["misses"] += 1
                if self.active[ti]["misses"] > 4 or (ts - self.active[ti]["last_sec"]) > 1.0:
                    self.active.pop(ti)

        for di, det in enumerate(dets):
            if det_to_tid[di] >= 0:
                continue
            tid = self.next_id
            self.next_id += 1
            self.active.append({
                "id": tid,
                "box": det["box"],
                "last_sec": ts,
                "vx": 0.0,
                "vy": 0.0,
                "misses": 0,
                "hits": 1,
                "ema_rgb": np.array(det["chest_rgb"], dtype=np.float64),
            })
            det_to_tid[di] = tid
            self._update_record(tid, ts, det)

        self._recluster(len(dets), {tid for tid in det_to_tid if tid >= 0})
        suppress_burst = self.sota_mode and (len(dets) > 15)
        emitted = []
        for di, det in enumerate(dets):
            tid = det_to_tid[di]
            rec = self.records[tid]
            if suppress_burst and rec["hits"] <= 1:
                continue
            emitted.append({"track_id": tid, "box": det["box"], "team": rec["team"]})
        self.frame_history.append((ts, emitted))
        return emitted

    def _update_record(self, tid: int, ts: float, det: Dict[str, Any]) -> None:
        rec = self.records.get(tid)
        rgb = np.array(det["chest_rgb"], dtype=np.float64)
        st = float(det.get("stripe_score", 0.0))
        if rec is None:
            rec = {
                "track_id": tid,
                "first_sec": ts,
                "last_sec": ts,
                "hits": 1,
                "sum_rgb": rgb.copy(),
                "n_rgb": 1,
                "ema_stripe": st,
                "n_stripe": 1,
                "ema_armband": float(det.get("armband_score", 0.0)),
                "sum_belly": 0.0,
                "n_belly": 0,
                "sum_pants": 0.0,
                "n_pants": 0,
                "kit_samples": [],
                "ref_votes": 0,
                "skater_votes": 0,
                "team": "UNKNOWN",
                "observations": [],
                "naive_votes": {},
            }
            self.records[tid] = rec
        else:
            rec["last_sec"] = max(rec["last_sec"], ts)
            rec["hits"] += 1
            rec["sum_rgb"] += rgb
            rec["n_rgb"] += 1
            rec["ema_stripe"] = 0.8 * rec["ema_stripe"] + 0.2 * st
            rec["n_stripe"] += 1
            rec["ema_armband"] = 0.8 * rec["ema_armband"] + 0.2 * float(det.get("armband_score", 0.0))

        bl = det.get("lower_belly_luma", float("nan"))
        if math.isfinite(bl):
            rec["sum_belly"] += bl
            rec["n_belly"] += 1
        pd = det.get("pants_darkness", float("nan"))
        if math.isfinite(pd):
            rec["sum_pants"] += pd
            rec["n_pants"] += 1
        if det.get("kit_summary_lab7") is not None:
            rec["kit_samples"].append(det["kit_summary_lab7"])

    def _mean_kit7(self, rec: Dict[str, Any], mean_rgb: np.ndarray) -> np.ndarray:
        ks = rec["kit_samples"]
        if not ks:
            lab = srgb_to_lab(mean_rgb)
            return np.array([lab[0], lab[1], lab[2], lab[0], lab[1], lab[2], 100.0 if lab[0] >= 62.0 else 0.0], dtype=np.float64)
        ref = ks[0]
        aligned = [order_free_best_aligned(k, ref) for k in ks]
        m = np.mean(aligned, axis=0)
        return m if m[0] <= m[3] else swap_summary(m)

    def _is_established_official(self, rec: Dict[str, Any], mean_chroma: float) -> bool:
        if rec["ref_votes"] < 3:
            return False
        overwhelmed = rec["skater_votes"] > 2 * rec["ref_votes"] and mean_chroma >= 15.0
        return not overwhelmed

    def _recluster(self, cur_count: int, cur_frame_tids: Optional[set] = None) -> None:
        if cur_frame_tids is None:
            cur_frame_tids = set(self.records.keys())
        snaps = []
        for rec in self.records.values():
            rgb = rec["sum_rgb"] / max(1, rec["n_rgb"])
            belly = rec["sum_belly"] / rec["n_belly"] if rec["n_belly"] > 0 else float("nan")
            pants = rec["sum_pants"] / rec["n_pants"] if rec["n_pants"] > 0 else float("nan")
            kit7 = self._mean_kit7(rec, rgb)
            mean_lab = srgb_to_lab(rgb)
            snaps.append({"rec": rec, "rgb": rgb, "belly": belly, "pants": pants, "kit7": kit7, "lab": mean_lab})

        if not snaps:
            return

        if not self.sota_mode:
            skaters = []
            for s in snaps:
                rgb = s["rgb"]
                max_c, min_c = float(rgb.max()), float(rgb.min())
                hsv_sat = 0.0 if max_c <= 1e-4 else (max_c - min_c) / max_c
                if s["rec"]["ema_stripe"] >= 0.58 and hsv_sat < 0.22:
                    s["rec"]["team"] = "REFEREE"
                else:
                    skaters.append(s)
            if len(skaters) == 1:
                skaters[0]["rec"]["team"] = "TEAM_A" if _rgb_luma(skaters[0]["rgb"]) < 128.0 else "TEAM_B"
                return
            if not skaters:
                return
            anchors = [s for s in skaters if s["rec"]["n_rgb"] >= 3] or [s for s in skaters if s["rec"]["n_rgb"] >= 2] or skaters
            anchors.sort(key=lambda s: _rgb_luma(s["rgb"]))
            cA = anchors[(len(anchors) * 15) // 100]["rgb"].copy() if len(anchors) >= 4 else anchors[0]["rgb"].copy()
            cB = anchors[(len(anchors) * 85) // 100]["rgb"].copy() if len(anchors) >= 4 else anchors[-1]["rgb"].copy()
            for _ in range(12):
                sA, wA = np.zeros(3), 0.0
                sB, wB = np.zeros(3), 0.0
                for s in skaters:
                    w = float(min(s["rec"]["n_rgb"], 15))
                    if _color_lum_dist(s["rgb"], cA) <= _color_lum_dist(s["rgb"], cB):
                        sA += s["rgb"] * w; wA += w
                    else:
                        sB += s["rgb"] * w; wB += w
                if wA > 0: cA = sA / wA
                if wB > 0: cB = sB / wB
            if _rgb_luma(cA) > _rgb_luma(cB):
                cA, cB = cB, cA
            sep = _color_lum_dist(cA, cB)
            allow_out = cur_count > 10 or len(self.active) > 10
            for s in skaters:
                dA, dB = _color_lum_dist(s["rgb"], cA), _color_lum_dist(s["rgb"], cB)
                team = "TEAM_A" if dA <= dB else "TEAM_B"
                if allow_out and sep >= 35.0:
                    sat = float(s["rgb"].max() - s["rgb"].min())
                    if min(dA, dB) > max(45.0, 0.30 * sep) and (sat < 35.0 or sat > 95.0):
                        team = "REFEREE"
                s["rec"]["team"] = team
            return

        # SOTA Pass 1: close/medium striped official + persisted official exclusion
        rem = []
        persisted_refs = []
        for s in snaps:
            rec = s["rec"]
            is_cur = rec["track_id"] in cur_frame_tids
            stripe_hit = rec["ema_stripe"] >= 0.46 or (rec["ema_stripe"] >= 0.34 and rec["ema_armband"] >= 0.04)
            chroma = math.hypot(s["lab"][1], s["lab"][2])
            bright_sup = s["kit7"][3] >= 52.0 or s["lab"][0] >= 36.0 or (math.isfinite(s["belly"]) and s["belly"] >= 85.0)
            if stripe_hit and chroma < 16.5 and bright_sup:
                if is_cur:
                    rec["ref_votes"] += 1
                rec["team"] = "REFEREE"
            elif self._is_established_official(rec, chroma):
                persisted_refs.append(s)
            else:
                rem.append(s)

        if not rem and not persisted_refs:
            return
        if not rem:
            for s in persisted_refs:
                s["rec"]["team"] = "REFEREE"
            return
        if len(rem) == 1 and not persisted_refs:
            if rem[0]["rec"]["track_id"] in cur_frame_tids:
                rem[0]["rec"]["skater_votes"] += 1
            rem[0]["rec"]["team"] = "TEAM_A" if kit_effective_lightness(rem[0]["kit7"], rem[0]["belly"]) < 53.6 else "TEAM_B"
            return

        def run_2means(items: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            base_anch = [s for s in items if s["rec"]["n_rgb"] >= 3] or [s for s in items if s["rec"]["n_rgb"] >= 2] or items
            normal_anch = [s for s in base_anch if max_zone_chroma(s["kit7"]) <= 25.0]
            anch = normal_anch if len(normal_anch) >= max(4, (len(base_anch) * 3) // 4) else base_anch
            anch.sort(key=lambda s: kit_effective_lightness(s["kit7"], s["belly"]))
            sa = anch[(len(anch) * 15) // 100] if len(anch) >= 4 else anch[0]
            sb = anch[(len(anch) * 85) // 100] if len(anch) >= 4 else anch[-1]
            if len(anch) >= 4:
                la = kit_effective_lightness(sa["kit7"], sa["belly"])
                lb = kit_effective_lightness(sb["kit7"], sb["belly"])
                if abs(lb - la) < 10.0:
                    anch_d = sorted(anch, key=lambda s: robust_kit_and_belly_dist_sq(s["kit7"], s["belly"], sa["kit7"], sa["belly"]))
                    sb = anch_d[(len(anch_d) * 85) // 100]
            cA = {"kit7": sa["kit7"].copy(), "belly": sa["belly"], "rgb": sa["rgb"].copy()}
            cB = {"kit7": sb["kit7"].copy(), "belly": sb["belly"], "rgb": sb["rgb"].copy()}
            for _ in range(12):
                skA, sbA, wbA, srA, wA = np.zeros(7), 0.0, 0.0, np.zeros(3), 0.0
                skB, sbB, wbB, srB, wB = np.zeros(7), 0.0, 0.0, np.zeros(3), 0.0
                for s in items:
                    dA = robust_kit_and_belly_dist_sq(s["kit7"], s["belly"], cA["kit7"], cA["belly"])
                    dB = robust_kit_and_belly_dist_sq(s["kit7"], s["belly"], cB["kit7"], cB["belly"])
                    w = float(min(s["rec"]["n_rgb"], 15))
                    if dA <= dB:
                        skA += robust_order_free_best_aligned(s["kit7"], cA["kit7"]) * w
                        srA += s["rgb"] * w; wA += w
                        if math.isfinite(s["belly"]): sbA += s["belly"] * w; wbA += w
                    else:
                        skB += robust_order_free_best_aligned(s["kit7"], cB["kit7"]) * w
                        srB += s["rgb"] * w; wB += w
                        if math.isfinite(s["belly"]): sbB += s["belly"] * w; wbB += w
                if wA > 0:
                    cA = {"kit7": skA / wA, "belly": (sbA / wbA if wbA > 0 else float("nan")), "rgb": srA / wA}
                if wB > 0:
                    cB = {"kit7": skB / wB, "belly": (sbB / wbB if wbB > 0 else float("nan")), "rgb": srB / wB}
            if cA["kit7"][0] > cA["kit7"][3]: cA["kit7"] = swap_summary(cA["kit7"])
            if cB["kit7"][0] > cB["kit7"][3]: cB["kit7"] = swap_summary(cB["kit7"])
            if kit_effective_lightness(cA["kit7"], cA["belly"]) > kit_effective_lightness(cB["kit7"], cB["belly"]):
                return cB, cA
            return cA, cB

        def is_balanced(itemsA: List[Dict[str, Any]], itemsB: List[Dict[str, Any]], wA: float, wB: float, total_n: int) -> bool:
            min_c = min(len(itemsA), len(itemsB))
            if min_c < max(2, total_n // 6):
                return False
            tot_w = wA + wB
            if tot_w <= 0.0 or min(wA, wB) < 0.12 * tot_w:
                return False
            if min_c * 4 < total_n:
                maj = itemsA if len(itemsA) >= len(itemsB) else itemsB
                if len(maj) >= 4:
                    effs = sorted(kit_effective_lightness(s["kit7"], s["belly"]) for s in maj)
                    if effs[(len(effs) * 85) // 100] - effs[(len(effs) * 15) // 100] >= 22.0:
                        return False
            return True

        def partition(items: List[Dict[str, Any]], cA: Dict[str, Any], cB: Dict[str, Any]):
            aL, bL, wA, wB = [], [], 0.0, 0.0
            for s in items:
                dA = robust_kit_and_belly_dist_sq(s["kit7"], s["belly"], cA["kit7"], cA["belly"])
                dB = robust_kit_and_belly_dist_sq(s["kit7"], s["belly"], cB["kit7"], cB["belly"])
                w = float(min(s["rec"]["n_rgb"], 15))
                if dA <= dB:
                    aL.append(s); wA += w
                else:
                    bL.append(s); wB += w
            return aL, bL, wA, wB

        def centroid_from_group(grp: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not grp:
                return None
            ref = grp[0]["kit7"]
            sk, sb, wb, sr, wt = np.zeros(7), 0.0, 0.0, np.zeros(3), 0.0
            for s in grp:
                w = float(min(s["rec"]["n_rgb"], 15))
                sk += robust_order_free_best_aligned(s["kit7"], ref) * w
                sr += s["rgb"] * w; wt += w
                if math.isfinite(s["belly"]): sb += s["belly"] * w; wb += w
            k7 = sk / wt
            if k7[0] > k7[3]:
                k7 = swap_summary(k7)
            return {"kit7": k7, "belly": (sb / wb if wb > 0 else float("nan")), "rgb": sr / wt}

        def run_1d_lightness_2means(items: List[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
            anch = [s for s in items if s["rec"]["n_rgb"] >= 3] or [s for s in items if s["rec"]["n_rgb"] >= 2] or items
            eff = sorted(kit_effective_lightness(s["kit7"], s["belly"]) for s in anch)
            muA = eff[(len(eff) * 15) // 100] if len(eff) >= 4 else eff[0]
            muB = eff[(len(eff) * 85) // 100] if len(eff) >= 4 else eff[-1]
            if abs(muB - muA) < 10.0:
                return None
            for _ in range(12):
                sA = wA = sB = wB = 0.0
                for s in items:
                    l = kit_effective_lightness(s["kit7"], s["belly"])
                    w = float(min(s["rec"]["n_rgb"], 15))
                    if abs(l - muA) <= abs(l - muB): sA += l * w; wA += w
                    else: sB += l * w; wB += w
                if wA > 0: muA = sA / wA
                if wB > 0: muB = sB / wB
            if abs(muB - muA) < 10.0:
                return None
            if muA > muB: muA, muB = muB, muA
            gA, gB, wA, wB = [], [], 0.0, 0.0
            for s in items:
                l = kit_effective_lightness(s["kit7"], s["belly"])
                w = float(min(s["rec"]["n_rgb"], 15))
                if abs(l - muA) <= abs(l - muB): gA.append(s); wA += w
                else: gB.append(s); wB += w
            if not is_balanced(gA, gB, wA, wB, len(items)):
                return None
            cA_1d, cB_1d = centroid_from_group(gA), centroid_from_group(gB)
            if cA_1d is None or cB_1d is None:
                return None
            if kit_effective_lightness(cA_1d["kit7"], cA_1d["belly"]) <= kit_effective_lightness(cB_1d["kit7"], cB_1d["belly"]):
                return cA_1d, cB_1d
            return cB_1d, cA_1d

        def run_robust_2means(items: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            initA, initB = run_2means(items)
            if len(items) < 6:
                return initA, initB
            iA, iB, wA, wB = partition(items, initA, initB)
            if is_balanced(iA, iB, wA, wB, len(items)):
                return initA, initB
            maj = iA if len(iA) >= len(iB) else iB
            if len(maj) >= 4:
                tA, tB = run_2means(maj)
                pA, pB, twA, twB = partition(maj, tA, tB)
                if is_balanced(pA, pB, twA, twB, len(maj)):
                    if math.sqrt(order_free_kit_dist_sq(tA["kit7"], tB["kit7"])) >= 11.0 or _color_lum_dist(tA["rgb"], tB["rgb"]) >= 25.0:
                        return tA, tB
            one_d = run_1d_lightness_2means(items)
            if one_d is not None:
                return one_d
            return initA, initB

        cA, cB = run_robust_2means(rem if len(rem) >= 2 else rem + persisted_refs)
        kit_sep = math.sqrt(order_free_kit_dist_sq(cA["kit7"], cB["kit7"]))
        rgb_sep = _color_lum_dist(cA["rgb"], cB["rgb"])

        # SOTA Pass 2: multi-cue blurred far-board official exclusion + persisted-official hysteresis
        allow_blur = cur_count >= 9 or len(self.active) >= 9
        can_test_blur = allow_blur and (kit_sep >= 11.0 or rgb_sep >= 25.0)
        labA = srgb_to_lab(cA["rgb"])
        labB = srgb_to_lab(cB["rgb"])
        grey_lab = np.array([55.0, 0.0, 0.0])
        skaters = []
        need_recompute = False

        def _is_raw_blurred_ref(s: Dict[str, Any]) -> bool:
            rec = s["rec"]
            mean_c = math.hypot(s["lab"][1], s["lab"][2])
            max_zc = max(math.hypot(s["kit7"][1], s["kit7"][2]), math.hypot(s["kit7"][4], s["kit7"][5]))
            is_white = (s["kit7"][3] >= 75.0 and s["kit7"][6] >= 42.0) or (math.isfinite(s["belly"]) and s["belly"] >= 150.0)
            d_grey = float(np.linalg.norm(s["lab"] - grey_lab))
            min_td = min(float(np.linalg.norm(s["lab"] - labA)), float(np.linalg.norm(s["lab"] - labB)))
            blur_relaxed = (
                s["pants"] >= 0.75
                and 31.0 <= s["lab"][0] <= 72.0
                and mean_c < 10.0
                and (not math.isfinite(s["belly"]) or s["belly"] < 135.0)
            )
            team_ok = (min_td >= 3.0 and d_grey <= min_td * 1.6) if blur_relaxed else (min_td >= 8.0 and d_grey <= min_td * 1.15)
            return (
                rec["n_rgb"] >= 2
                and math.isfinite(s["pants"]) and s["pants"] >= 0.60
                and 30.0 <= s["lab"][0] <= 72.0
                and mean_c < 15.0 and max_zc < 17.5
                and not is_white
                and team_ok
            )

        for s in persisted_refs:
            rec = s["rec"]
            is_cur = rec["track_id"] in cur_frame_tids
            mean_c = math.hypot(s["lab"][1], s["lab"][2])
            raw_b = can_test_blur and _is_raw_blurred_ref(s)
            if is_cur:
                if raw_b:
                    rec["ref_votes"] += 1
                else:
                    rec["skater_votes"] += 1
            if raw_b or self._is_established_official(rec, mean_c):
                rec["team"] = "REFEREE"
            else:
                skaters.append(s)
                need_recompute = True

        for s in rem:
            rec = s["rec"]
            is_cur = rec["track_id"] in cur_frame_tids
            mean_c = math.hypot(s["lab"][1], s["lab"][2])
            raw_b = can_test_blur and _is_raw_blurred_ref(s)
            if is_cur:
                if raw_b:
                    rec["ref_votes"] += 1
                else:
                    rec["skater_votes"] += 1
            if raw_b or self._is_established_official(rec, mean_c):
                rec["team"] = "REFEREE"
                need_recompute = True
            else:
                skaters.append(s)

        if not skaters:
            return
        if len(skaters) == 1:
            skaters[0]["rec"]["team"] = "TEAM_A" if kit_effective_lightness(skaters[0]["kit7"], skaters[0]["belly"]) < 53.6 else "TEAM_B"
            return
        if need_recompute:
            cA, cB = run_robust_2means(skaters)

        self.team_kit7 = {"TEAM_A": cA["kit7"], "TEAM_B": cB["kit7"]}
        self.team_rgb = {"TEAM_A": cA["rgb"], "TEAM_B": cB["rgb"]}
        self.team_belly = {"TEAM_A": cA["belly"], "TEAM_B": cB["belly"]}

        for s in skaters:
            dA = robust_kit_and_belly_dist_sq(s["kit7"], s["belly"], cA["kit7"], cA["belly"])
            dB = robust_kit_and_belly_dist_sq(s["kit7"], s["belly"], cB["kit7"], cB["belly"])
            s["rec"]["team"] = "TEAM_A" if dA <= dB else "TEAM_B"

    def resolve_numbers(self) -> Dict[int, Optional[str]]:
        if not self.sota_mode:
            out: Dict[int, Optional[str]] = {}
            for tid, rec in self.records.items():
                nv = rec["naive_votes"]
                if not nv:
                    out[tid] = None
                else:
                    best = max(nv.items(), key=lambda kv: kv[1])[0]
                    out[tid] = best
            return out

        out = {}
        for team in ("TEAM_A", "TEAM_B"):
            chains = [
                {
                    "chain_id": rec["track_id"],
                    "first_sec": rec["first_sec"],
                    "last_sec": rec["last_sec"],
                    "observations": rec["observations"],
                }
                for rec in self.records.values()
                if rec["team"] == team
            ]
            sol = solve_team_numbers_bayesian_hungarian(chains)
            for cid, (num, _) in sol.items():
                out[cid] = num
        return out


# ---------------------------------------------------------------------------
# Ground Truth at 11:14 (Evaluation Only, Zero Leakage)
# ---------------------------------------------------------------------------

GT_FRAME_08 = [
    {"name": "#11", "cx": 0.113, "cy": 0.542, "team": "TEAM_B", "number": "11"},
    {"name": "REF_L", "cx": 0.133, "cy": 0.403, "team": "REFEREE", "number": None},
    {"name": "#12", "cx": 0.283, "cy": 0.416, "team": "TEAM_B", "number": "12"},
    {"name": "#97", "cx": 0.308, "cy": 0.520, "team": "TEAM_B", "number": "97"},
    {"name": "#67", "cx": 0.383, "cy": 0.521, "team": "TEAM_A", "number": "67"},
    {"name": "DARK_U", "cx": 0.388, "cy": 0.420, "team": "TEAM_A", "number": None},
    {"name": "#34", "cx": 0.454, "cy": 0.405, "team": "TEAM_B", "number": "34"},
    {"name": "REF_C", "cx": 0.496, "cy": 0.340, "team": "REFEREE", "number": None},
    {"name": "#43", "cx": 0.522, "cy": 0.412, "team": "TEAM_A", "number": "43"},
    {"name": "#77", "cx": 0.542, "cy": 0.456, "team": "TEAM_B", "number": "77"},
    {"name": "#14", "cx": 0.687, "cy": 0.423, "team": "TEAM_A", "number": "14"},
    {"name": "#57", "cx": 0.893, "cy": 0.581, "team": "TEAM_A", "number": "57"},
]

GT_FS_674 = [
    {"name": "#11", "cx": 0.141, "cy": 0.541, "team": "TEAM_B", "number": "11"},
    {"name": "REF_L", "cx": 0.151, "cy": 0.400, "team": "REFEREE", "number": None},
    {"name": "#97", "cx": 0.254, "cy": 0.557, "team": "TEAM_B", "number": "97"},
    {"name": "#12", "cx": 0.287, "cy": 0.407, "team": "TEAM_B", "number": "12"},
    {"name": "#67", "cx": 0.390, "cy": 0.559, "team": "TEAM_A", "number": "67"},
    {"name": "DARK_U", "cx": 0.426, "cy": 0.426, "team": "TEAM_A", "number": None},
    {"name": "#34", "cx": 0.464, "cy": 0.396, "team": "TEAM_B", "number": "34"},
    {"name": "REF_C", "cx": 0.479, "cy": 0.343, "team": "REFEREE", "number": None},
    {"name": "#77", "cx": 0.507, "cy": 0.437, "team": "TEAM_B", "number": "77"},
    {"name": "#43", "cx": 0.531, "cy": 0.394, "team": "TEAM_A", "number": "43"},
    {"name": "#14", "cx": 0.687, "cy": 0.416, "team": "TEAM_A", "number": "14"},
    {"name": "#57", "cx": 0.903, "cy": 0.568, "team": "TEAM_A", "number": "57"},
]


def evaluate_image_sequence(
    name: str,
    frame_paths: List[Path],
    dt_sec: float,
    eval_idx: int,
    gt_items: List[Dict[str, Any]],
    detector: YoloDetector,
    str_rec: JerseyStrRecognizer,
) -> Dict[str, Any]:
    results = {}
    for mode_name, is_sota in (("baseline", False), ("proposed_sota", True)):
        tracker = LocalTracker(sota_mode=is_sota)
        eval_emitted = []
        for fi, fp in enumerate(frame_paths):
            bgr = cv2.imread(str(fp))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            boxes = detector.detect(rgb, panoramic_tiling=True)
            wb = compute_ice_wb(rgb)
            dets = []
            for b in boxes:
                b4 = b[:4]
                if is_sota:
                    feat = extract_sota_features(rgb, wb, b4)
                else:
                    feat = extract_baseline_features(rgb, wb, b4)
                feat["box"] = b4
                dets.append(feat)
            ts = fi * dt_sec
            emitted = tracker.update(ts, dets)

            # Run STR on every track across sequence (simulating 220ms/on-device STR cadence)
            for item in emitted:
                tid = item["track_id"]
                rec = tracker.records[tid]
                if rec["team"] == "REFEREE":
                    continue
                if is_sota and rec["n_pants"] >= 1 and (rec["sum_pants"] / rec["n_pants"]) >= 0.60:
                    m_lab = srgb_to_lab(rec["sum_rgb"] / max(1, rec["n_rgb"]))
                    if math.hypot(float(m_lab[1]), float(m_lab[2])) < 15.0:
                        continue
                res = str_rec.infer(rgb, wb, item["box"])
                cands = res["candidates"] if res["legibility"] >= 0.55 else []
                if cands:
                    q_w = min(max((0.35 + 0.65 * res["legibility"]) * (0.50 + 0.50 * res["back"]), 0.05), 1.25)
                else:
                    q_w = min(max(0.40 + 0.60 * res["back"], 0.10), 1.0)
                if is_sota:
                    if cands:
                        rec["observations"].append({
                            "timestamp_sec": ts,
                            "candidates": cands,
                            "weight": q_w,
                            "source_weight": 1.6,
                        })
                    if res["partial_left"]:
                        rec["observations"].append({
                            "timestamp_sec": ts,
                            "number": res["partial_left"][0],
                            "conf": res["partial_left"][1],
                            "kind": "LEFT_PARTIAL",
                            "weight": q_w,
                            "source_weight": 1.6,
                        })
                    if res["partial_right"]:
                        rec["observations"].append({
                            "timestamp_sec": ts,
                            "number": res["partial_right"][0],
                            "conf": res["partial_right"][1],
                            "kind": "RIGHT_PARTIAL",
                            "weight": q_w,
                            "source_weight": 1.6,
                        })
                else:
                    # Baseline naive: single argmax string only when legibility >= 0.50 (no partials, no Hungarian)
                    if res["candidates"] and res["legibility"] >= 0.50:
                        top_num, top_p = res["candidates"][0]
                        if top_p >= 0.40:
                            rec["naive_votes"][top_num] = rec["naive_votes"].get(top_num, 0.0) + top_p

            if fi == eval_idx:
                eval_emitted = emitted

        # Final re-resolve numbers after sequence
        num_map = tracker.resolve_numbers()
        for item in eval_emitted:
            item["team"] = tracker.records[item["track_id"]]["team"]
            item["number"] = num_map.get(item["track_id"]) if item["team"] != "REFEREE" else None

        # Score against 11:14 GT
        used = set()
        matches = []
        det_hits = 0
        team_correct = 0
        team_total = 0
        ref_tp = 0
        ref_fn = 0
        ref_fp = 0
        hash14_ref_fp = False
        hash97_team_ok = False
        hash97_num = None
        num_correct = 0
        num_pred_count = 0
        num_gt_total = 0

        for g in gt_items:
            best_i, best_d = -1, 1e9
            for i, p in enumerate(eval_emitted):
                if i in used:
                    continue
                pcx, pcy = 0.5 * (p["box"][0] + p["box"][2]), 0.5 * (p["box"][1] + p["box"][3])
                d = math.hypot(pcx - g["cx"], pcy - g["cy"])
                if d < 0.045 and d < best_d:
                    best_d = d
                    best_i = i
            pred = eval_emitted[best_i] if best_i >= 0 else None
            if best_i >= 0:
                used.add(best_i)
                det_hits += 1
                if g["team"] == "REFEREE":
                    if pred["team"] == "REFEREE":
                        ref_tp += 1
                    else:
                        ref_fn += 1
                else:
                    team_total += 1
                    if pred["team"] == g["team"]:
                        team_correct += 1
                    if pred["team"] == "REFEREE":
                        ref_fp += 1
                        if g["name"] == "#14":
                            hash14_ref_fp = True
                    if g["name"] == "#97":
                        hash97_team_ok = (pred["team"] == "TEAM_B")
                        hash97_num = pred["number"]
                    if g["number"] is not None:
                        num_gt_total += 1
                        if pred["number"] is not None:
                            num_pred_count += 1
                            if pred["number"] == g["number"]:
                                num_correct += 1
            else:
                if g["team"] == "REFEREE":
                    ref_fn += 1
                elif g["number"] is not None:
                    num_gt_total += 1

            matches.append({
                "gt": g["name"],
                "gt_team": g["team"],
                "gt_num": g["number"],
                "pred_team": pred["team"] if pred else None,
                "pred_num": pred["number"] if pred else None,
                "track_id": pred["track_id"] if pred else None,
            })

        ref_prec = ref_tp / max(1, ref_tp + ref_fp) if (ref_tp + ref_fp) > 0 else 0.0
        ref_rec = ref_tp / max(1, ref_tp + ref_fn)
        num_prec = num_correct / max(1, num_pred_count) if num_pred_count > 0 else 0.0
        num_rec = num_correct / max(1, num_gt_total)

        results[mode_name] = {
            "detection_recall": f"{det_hits}/{len(gt_items)} ({det_hits / len(gt_items) * 100:.1f}%)",
            "skater_team_acc": f"{team_correct}/{team_total} ({team_correct / max(1, team_total) * 100:.1f}%)",
            "hash97_team_b_ok": hash97_team_ok,
            "hash97_predicted_num": hash97_num,
            "referee_precision": round(ref_prec, 3),
            "referee_recall": round(ref_rec, 3),
            "referee_tp_fp_fn": [ref_tp, ref_fp, ref_fn],
            "hash14_referee_fp": hash14_ref_fp,
            "jersey_exact_hits": f"{num_correct}/{num_gt_total}",
            "jersey_precision": round(num_prec, 3),
            "jersey_recall": round(num_rec, 3),
            "matches": matches,
        }
    return {"benchmark": name, "frames": len(frame_paths), "results": results}


def evaluate_burst_ghost_suppression(mp4_crossfade: Path, mp4_cleanhold: Path, detector: YoloDetector) -> Dict[str, Any]:
    """Compare crossfaded MP4 (where blended frames hit det=19..23) vs cleanhold MP4 under Baseline vs SOTA."""
    out = {}
    for label, mp4_path in (("crossfade_30s", mp4_crossfade), ("cleanhold_30s", mp4_cleanhold)):
        cap = cv2.VideoCapture(str(mp4_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * 0.25)))  # 4 fps like Android
        max_raw_det = 0
        max_base_emitted = 0
        max_sota_emitted = 0
        t_base = LocalTracker(sota_mode=False)
        t_sota = LocalTracker(sota_mode=True)
        burst_frames = 0
        for fi in range(0, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            boxes = detector.detect(rgb, panoramic_tiling=True)
            max_raw_det = max(max_raw_det, len(boxes))
            if len(boxes) > 12:
                burst_frames += 1
            wb = compute_ice_wb(rgb)
            dets_b = [{**extract_baseline_features(rgb, wb, b[:4]), "box": b[:4]} for b in boxes]
            dets_s = [{**extract_sota_features(rgb, wb, b[:4]), "box": b[:4]} for b in boxes]
            eb = t_base.update(fi / fps, dets_b)
            es = t_sota.update(fi / fps, dets_s)
            max_base_emitted = max(max_base_emitted, len(eb))
            max_sota_emitted = max(max_sota_emitted, len(es))
        cap.release()
        out[label] = {
            "max_raw_detections": max_raw_det,
            "burst_frames_gt12": burst_frames,
            "baseline_max_emitted": max_base_emitted,
            "sota_max_emitted": max_sota_emitted,
        }
    return out


def evaluate_video_sanity(mp4_path: Path, detector: YoloDetector, str_rec: JerseyStrRecognizer, sample_stride_frames: int = 15) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(mp4_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = LocalTracker(sota_mode=True)
    det_counts = []
    for fi in range(0, n, sample_stride_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes = detector.detect(rgb, panoramic_tiling=True)
        det_counts.append(len(boxes))
        wb = compute_ice_wb(rgb)
        dets = [{**extract_sota_features(rgb, wb, b[:4]), "box": b[:4]} for b in boxes]
        emitted = tracker.update(fi / fps, dets)
        for item in emitted:
            tid = item["track_id"]
            rec = tracker.records[tid]
            if rec["team"] == "REFEREE":
                continue
            if rec["n_pants"] >= 1 and (rec["sum_pants"] / rec["n_pants"]) >= 0.60:
                m_lab = srgb_to_lab(rec["sum_rgb"] / max(1, rec["n_rgb"]))
                if math.hypot(float(m_lab[1]), float(m_lab[2])) < 15.0:
                    continue
            res = str_rec.infer(rgb, wb, item["box"])
            if res["candidates"]:
                q_w = (0.3 + 0.7 * res["legibility"]) * (0.5 + 0.5 * res["back"])
                rec["observations"].append({"timestamp_sec": fi / fps, "candidates": res["candidates"], "weight": q_w, "source_weight": 1.6})
    cap.release()
    nums = tracker.resolve_numbers()
    team_counts = {"TEAM_A": 0, "TEAM_B": 0, "REFEREE": 0, "UNKNOWN": 0}
    for r in tracker.records.values():
        team_counts[r["team"]] = team_counts.get(r["team"], 0) + 1
    found_nums = sorted({f"{tracker.records[tid]['team']}:#{num}" for tid, num in nums.items() if num is not None})
    return {
        "video": mp4_path.name,
        "frames_sampled": len(det_counts),
        "mean_on_ice_detections": round(float(np.mean(det_counts)) if det_counts else 0.0, 2),
        "max_on_ice_detections": int(max(det_counts)) if det_counts else 0,
        "track_team_counts": team_counts,
        "roster_free_jersey_numbers_resolved": found_nums,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir", type=Path, default=REPO_ROOT / "android" / "app" / "src" / "main" / "assets")
    ap.add_argument("--report-out", type=Path, default=REPO_ROOT / "research" / "local_validation_report.json")
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--external-bench-dir", type=Path, default=DEFAULT_BENCH_DIR)
    args = ap.parse_args()

    t0 = time.time()
    detector = YoloDetector(args.assets_dir / "coach_person_detector.onnx")
    str_rec = JerseyStrRecognizer(args.assets_dir / "coach_jersey_str.onnx")

    compare_frames = sorted((args.work_dir / "coachbox_compare" / "frames").glob("frame_*.png"))
    fs_frames = sorted((args.work_dir / "coachbox_v2" / "fs_1080p_crops").glob("fs_*.png"))

    res_compare = (
        evaluate_image_sequence("coachbox_compare_16frames_11m14s", compare_frames, dt_sec=2.0, eval_idx=8, gt_items=GT_FRAME_08, detector=detector, str_rec=str_rec)
        if compare_frames else {"benchmark": "coachbox_compare_16frames_11m14s", "skipped": True, "reason": "frames_not_found"}
    )
    res_1080p = (
        evaluate_image_sequence("coachbox_1080p_23frames_11m14s", fs_frames, dt_sec=1.4, eval_idx=12, gt_items=GT_FS_674, detector=detector, str_rec=str_rec)
        if fs_frames else {"benchmark": "coachbox_1080p_23frames_11m14s", "skipped": True, "reason": "frames_not_found"}
    )

    crossfade_mp4 = args.work_dir / "coachbox_compare" / "coachbox_tvbd_cougars_30s.mp4"
    cleanhold_mp4 = args.work_dir / "coachbox_v2" / "coachbox_tvbd_cougars_cleanhold_30s.mp4"
    res_burst = (
        evaluate_burst_ghost_suppression(crossfade_mp4, cleanhold_mp4, detector)
        if (crossfade_mp4.exists() and cleanhold_mp4.exists()) else {"skipped": True, "reason": "clips_not_found"}
    )

    sample_mp4 = args.external_bench_dir / "bench" / "hockey" / "sample_hockey_30s.mp4"
    tvi_mp4 = args.external_bench_dir / "bench" / "hockey" / "clips" / "tvi_iceland_clip_01_0240s_0270s.mp4"
    res_sample = (
        evaluate_video_sanity(sample_mp4, detector, str_rec, sample_stride_frames=30)
        if sample_mp4.exists() else {"video": sample_mp4.name, "skipped": True, "reason": "file_not_found"}
    )
    res_tvi = (
        evaluate_video_sanity(tvi_mp4, detector, str_rec, sample_stride_frames=15)
        if tvi_mp4.exists() else {"video": tvi_mp4.name, "skipped": True, "reason": "file_not_found"}
    )

    str_meta = json.loads((args.assets_dir / "coach_jersey_str.json").read_text())
    report = {
        "roster_free_verified": True,
        "elapsed_sec": round(time.time() - t0, 2),
        "onnx_str_summary": {
            "size_bytes": str_meta["size_bytes"],
            "opset": str_meta["opset"],
            "ort_cpu_ms": str_meta["benchmark"]["onnxruntime_cpu_2thread_ms"],
            "synthetic_val": str_meta["synthetic_validation_metrics"],
            "calibration": str_meta["calibration"],
        },
        "coachbox_compare": res_compare,
        "coachbox_1080p": res_1080p,
        "burst_ghost_suppression": res_burst,
        "non_coachbox_sanity": {
            "sample_hockey_30s": res_sample,
            "tvi_iceland_clip_01": res_tvi,
        },
    }
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
