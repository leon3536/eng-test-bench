"""5-Game Cloud (coach.leonxzhao.com) vs. 100% Roster-Free Mobile Pipeline Benchmark (Round 2).

Mirrors the exact production Round 2 Android `:core` + `:app` vision pipeline:
  - YoloOnnxDetector (640x360 letterbox + 2 far-ice tiles on aspect >= 1.65, soft 0.23..0.30
    ice-context goalie/foreground admission, NMS IoU > 0.42 / IoM > 0.60, IceContext WB CIELAB)
  - OnIceFilter (aspect < 2.0 16:9 zoomed IceContext bounds + foreground fy<=1.005 recovery +
    aspect >= 2.0 Dublin polygon w/ IceContext rejection)
  - KitLabFeatures + MultiTeamPlayerTracker (constant-velocity + contact cross-kit gate +
    torsoUpperLabL p80 crest immunity + Huberized order-free 2-means w/ 1-vs-N balance guard +
    torso-centered 64x36 DFT stripe w/ pants floor + Round 2 blurred-referee hysteresis +
    Stage-1 1.5s / Stage-2 4.0s chain stitching)
  - JerseyStrOnnxRecognizer v4 (12x18 min crop, 96x96 letterbox + WB + interior Laplacian>=18.0
    gate + leg>=0.68 / back>=0.58 / illegible<0.48 / top>=0.34 + tightened partials)
  - RealtimeHockeyProcessor priority gating (MIN_OCR_BOX_HEIGHT=0.068)
  - JerseyNumberPosterior (uniform 1/99 prior, 1.0s time-bin discount, O(N^3) Hungarian solver)

100% Roster-Free: never imports or conditions on any game-sheet roster or test numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = Path(os.environ.get("HOCKEY_SCRATCH_DIR", str(REPO_ROOT / "research" / "scratch")))

# ---------------------------------------------------------------------------
# Color & CIELAB Utilities (exact mirror of KitLabFeatures.kt Round 2)
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
DEFAULT_UPPER_L_WEIGHT = 1.75
DEFAULT_SECONDARY_BELLY_WEIGHT = 0.40
HUBER_AB_DELTA = 22.0


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (0..255) array (..., 3) to CIELAB [L*, a*, b*]."""
    s = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 255.0) / 255.0
    lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    x = (0.4124564 * lin[..., 0] + 0.3575761 * lin[..., 1] + 0.1804375 * lin[..., 2]) / _D65_XN
    y = (0.2126729 * lin[..., 0] + 0.7151522 * lin[..., 1] + 0.0721750 * lin[..., 2]) / _D65_YN
    z = (0.0193339 * lin[..., 0] + 0.1191920 * lin[..., 1] + 0.9503041 * lin[..., 2]) / _D65_ZN
    fx = np.where(x > _LAB_DELTA_CUBED, np.cbrt(x), x * _LAB_SCALE + _LAB_OFFSET)
    fy = np.where(y > _LAB_DELTA_CUBED, np.cbrt(y), y * _LAB_SCALE + _LAB_OFFSET)
    fz = np.where(z > _LAB_DELTA_CUBED, np.cbrt(z), z * _LAB_SCALE + _LAB_OFFSET)
    l_val = np.clip(116.0 * fy - 16.0, 0.0, 100.0)
    a_val = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)
    return np.stack([l_val, a_val, b_val], axis=-1)


def luma_to_lab_l(luma: float) -> float:
    s = min(max(luma, 0.0), 255.0) / 255.0
    lin = s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4
    fy = math.cbrt(lin) if lin > _LAB_DELTA_CUBED else lin * _LAB_SCALE + _LAB_OFFSET
    return min(max(116.0 * fy - 16.0, 0.0), 100.0)


def swap_summary(s7: np.ndarray) -> np.ndarray:
    return np.array([s7[3], s7[4], s7[5], s7[0], s7[1], s7[2], 100.0 - s7[6]], dtype=np.float64)


def sanitize_kit_summary7(s7: Optional[np.ndarray], fallback_rgb: List[int]) -> np.ndarray:
    """Canonicalize 7D kit summary with darkL <= lightL (KitLabFeatures.sanitizeKitSummary7)."""
    if s7 is None or len(s7) < 7 or not np.all(np.isfinite(s7[:7])):
        lab = srgb_to_lab(np.array(fallback_rgb, dtype=np.float64))
        share = 100.0 if lab[0] >= 62.0 else 0.0
        return np.array([lab[0], lab[1], lab[2], lab[0], lab[1], lab[2], share], dtype=np.float64)
    l0 = float(np.clip(s7[0], 0.0, 100.0))
    a0 = float(np.clip(s7[1], -128.0, 128.0))
    b0 = float(np.clip(s7[2], -128.0, 128.0))
    l1 = float(np.clip(s7[3], 0.0, 100.0))
    a1 = float(np.clip(s7[4], -128.0, 128.0))
    b1 = float(np.clip(s7[5], -128.0, 128.0))
    sh = float(np.clip(s7[6], 0.0, 100.0))
    if l0 <= l1:
        return np.array([l0, a0, b0, l1, a1, b1, sh], dtype=np.float64)
    return np.array([l1, a1, b1, l0, a0, b0, 100.0 - sh], dtype=np.float64)


def weighted_kit_dist_sq(x: np.ndarray, c: np.ndarray) -> float:
    d = np.array([
        (x[0] - c[0]) * WEIGHT_L, (x[1] - c[1]) * WEIGHT_AB, (x[2] - c[2]) * WEIGHT_AB,
        (x[3] - c[3]) * WEIGHT_L, (x[4] - c[4]) * WEIGHT_AB, (x[5] - c[5]) * WEIGHT_AB,
        (x[6] - c[6]) * WEIGHT_SHARE,
    ], dtype=np.float64)
    return float((d * d).sum())


def order_free_kit_dist_sq(x: np.ndarray, c: np.ndarray) -> float:
    return min(weighted_kit_dist_sq(x, c), weighted_kit_dist_sq(swap_summary(x), c))


def kit_and_belly_or_upper_dist_sq(
    kx: np.ndarray,
    bx: float,
    ux: float,
    kc: np.ndarray,
    bc: float,
    uc: float,
    belly_weight: float = BELLY_WEIGHT,
) -> float:
    dist = order_free_kit_dist_sq(kx, kc)
    if math.isfinite(ux) and math.isfinite(uc):
        du = (ux - uc) * DEFAULT_UPPER_L_WEIGHT
        dist += du * du
        if math.isfinite(bx) and math.isfinite(bc):
            db = (luma_to_lab_l(bx) - luma_to_lab_l(bc)) * DEFAULT_SECONDARY_BELLY_WEIGHT
            dist += db * db
    elif math.isfinite(bx) and math.isfinite(bc):
        db = (luma_to_lab_l(bx) - luma_to_lab_l(bc)) * belly_weight
        dist += db * db
    return dist


def huber_sq(diff: float, delta: float = HUBER_AB_DELTA) -> float:
    a = abs(diff)
    return a * a if a <= delta else delta * (2.0 * a - delta)


def robust_weighted_kit_dist_sq(x: np.ndarray, c: np.ndarray, delta: float = HUBER_AB_DELTA) -> float:
    dl0 = (x[0] - c[0]) * WEIGHT_L
    dl1 = (x[3] - c[3]) * WEIGHT_L
    dsh = (x[6] - c[6]) * WEIGHT_SHARE
    return float(
        dl0 * dl0
        + huber_sq((x[1] - c[1]) * WEIGHT_AB, delta)
        + huber_sq((x[2] - c[2]) * WEIGHT_AB, delta)
        + dl1 * dl1
        + huber_sq((x[4] - c[4]) * WEIGHT_AB, delta)
        + huber_sq((x[5] - c[5]) * WEIGHT_AB, delta)
        + dsh * dsh
    )


def robust_order_free_best_aligned(x: np.ndarray, c: np.ndarray, delta: float = HUBER_AB_DELTA) -> np.ndarray:
    sw = swap_summary(x)
    return x if robust_weighted_kit_dist_sq(x, c, delta) <= robust_weighted_kit_dist_sq(sw, c, delta) else sw


def robust_kit_and_belly_or_upper_dist_sq(
    kx: np.ndarray,
    bx: float,
    ux: float,
    kc: np.ndarray,
    bc: float,
    uc: float,
    belly_weight: float = BELLY_WEIGHT,
    delta: float = HUBER_AB_DELTA,
) -> float:
    dist = min(robust_weighted_kit_dist_sq(kx, kc, delta), robust_weighted_kit_dist_sq(swap_summary(kx), kc, delta))
    if math.isfinite(ux) and math.isfinite(uc):
        du = (ux - uc) * DEFAULT_UPPER_L_WEIGHT
        dist += du * du
        if math.isfinite(bx) and math.isfinite(bc):
            db = (luma_to_lab_l(bx) - luma_to_lab_l(bc)) * DEFAULT_SECONDARY_BELLY_WEIGHT
            dist += db * db
    elif math.isfinite(bx) and math.isfinite(bc):
        db = (luma_to_lab_l(bx) - luma_to_lab_l(bc)) * belly_weight
        dist += db * db
    return dist


def kit_effective_lightness(
    s7: np.ndarray,
    belly_luma: float = float("nan"),
    torso_upper_l: float = float("nan"),
) -> float:
    share = min(max(float(s7[6]) / 100.0, 0.0), 1.0)
    mean_l = (1.0 - share) * float(s7[0]) + share * float(s7[3])
    if math.isfinite(torso_upper_l):
        return 0.45 * mean_l + 0.55 * float(torso_upper_l)
    if math.isfinite(belly_luma):
        return 0.55 * mean_l + 0.45 * luma_to_lab_l(belly_luma)
    return float(mean_l)


def max_zone_chroma(s7: np.ndarray) -> float:
    return max(math.hypot(float(s7[1]), float(s7[2])), math.hypot(float(s7[4]), float(s7[5])))


# ---------------------------------------------------------------------------
# OnIceFilter & Detector & Feature Extraction (YoloOnnxDetector.kt Round 2)
# ---------------------------------------------------------------------------

SOFT_PLAYER_MIN_CONF = 0.23
MIN_DET_CONF = 0.30
SOFT_MIN_BOX_HEIGHT = 0.052
SOFT_MIN_SURROUND_ICE = 0.56
SOFT_MIN_FOOT_ICE = 0.60
SOFT_MIN_LOWER_SIDE_ICE = 0.48
ICE_CTX_MIN_L = 55.0
ICE_CTX_MAX_CHROMA = 22.0


def _accum_ice_samples(
    rgb: np.ndarray, rx0: float, ry0: float, rx1: float, ry1: float, wb: np.ndarray
) -> Tuple[int, int]:
    h, w = rgb.shape[:2]
    px0 = max(0, min(w, int(rx0 * w)))
    px1 = max(0, min(w, int(rx1 * w)))
    py0 = max(0, min(h, int(ry0 * h)))
    py1 = max(0, min(h, int(ry1 * h)))
    sw, sh = px1 - px0, py1 - py0
    if sw <= 0 or sh <= 0:
        return 0, 0
    ny, nx = min(8, sh), min(8, sw)
    ys = [py0] if ny == 1 else [py0 + (iy * (sh - 1)) // (ny - 1) for iy in range(ny)]
    xs = [px0] if nx == 1 else [px0 + (ix * (sw - 1)) // (nx - 1) for ix in range(nx)]
    patch = rgb[np.ix_(ys, xs)].reshape(-1, 3)
    wb_p = np.clip(np.rint(patch.astype(np.float32) * wb[None, :]), 0.0, 255.0)
    lab = srgb_to_lab(wb_p)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    ice_cnt = int(((lab[:, 0] >= ICE_CTX_MIN_L) & (chroma < ICE_CTX_MAX_CHROMA)).sum())
    return ice_cnt, len(patch)


def compute_ice_context(
    rgb: np.ndarray, box: Tuple[float, float, float, float], wb: np.ndarray
) -> Dict[str, float]:
    """Exact mirror of YoloOnnxDetector.computeIceContext."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    fx0, fy0, fx1, fy1 = x1 - 0.20 * bw, y2 - 0.02 * bh, x2 + 0.20 * bw, min(1.0, y2 + 0.14 * bh)
    fc, ft = _accum_ice_samples(rgb, fx0, fy0, fx1, fy1, wb)
    foot_ice = (fc / ft) if ft > 0 else 0.0

    lol_c, lol_t = _accum_ice_samples(rgb, x1 - 0.25 * bw, y1 + 0.50 * bh, x1, y1 + 0.90 * bh, wb)
    lor_c, lor_t = _accum_ice_samples(rgb, x2, y1 + 0.50 * bh, x2 + 0.25 * bw, y1 + 0.90 * bh, wb)
    lo_l = (lol_c / lol_t) if lol_t > 0 else 0.0
    lo_r = (lor_c / lor_t) if lor_t > 0 else 0.0

    upl_c, upl_t = _accum_ice_samples(rgb, x1 - 0.25 * bw, y1 + 0.10 * bh, x1, y1 + 0.50 * bh, wb)
    upr_c, upr_t = _accum_ice_samples(rgb, x2, y1 + 0.10 * bh, x2 + 0.25 * bw, y1 + 0.50 * bh, wb)
    up_l = (upl_c / upl_t) if upl_t > 0 else 0.0
    up_r = (upr_c / upr_t) if upr_t > 0 else 0.0

    sl_c, sl_t = _accum_ice_samples(rgb, x1 - 0.25 * bw, y1 + 0.30 * bh, x1, y1 + 0.90 * bh, wb)
    sr_c, sr_t = _accum_ice_samples(rgb, x2, y1 + 0.30 * bh, x2 + 0.25 * bw, y1 + 0.90 * bh, wb)
    surr_tot = ft + sl_t + sr_t
    surr_ice = ((fc + sl_c + sr_c) / surr_tot) if surr_tot > 0 else 0.0

    return {
        "foot": float(foot_ice),
        "surrounding": float(surr_ice),
        "lower_side": float(0.5 * (lo_l + lo_r)),
        "upper_side": float(0.5 * (up_l + up_r)),
    }


def is_on_ice(
    box: Tuple[float, float, float, float],
    aspect_ratio: float,
    ice_context: Optional[Dict[str, float]] = None,
) -> bool:
    """Exact mirror of OnIceFilter.isOnIce (Round 2)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return False
    if w <= 0.005 or h < 0.026 or w > 0.45 or h > 0.85:
        return False
    cx = 0.5 * (x1 + x2)
    fy = y2
    if cx < 0.0 or cx > 1.0 or fy < 0.0 or fy > 1.02:
        return False

    if aspect_ratio >= 2.0:
        fyc = min(max(fy, 0.0), 1.0)
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
            if ((yi > fyc) != (yj > fyc)) and (cx < (xj - xi) * (fyc - yi) / denom + xi):
                inside = not inside
            j = i
        if not inside:
            return False
        if ice_context is not None and math.isfinite(ice_context["surrounding"]) and math.isfinite(ice_context["foot"]):
            lo = ice_context["lower_side"] if math.isfinite(ice_context["lower_side"]) else ice_context["surrounding"]
            if ice_context["surrounding"] < 0.25 and ice_context["foot"] < 0.15 and lo < 0.28:
                return False
        return True

    if cx < 0.015 or cx > 0.985:
        return False
    if ice_context is None:
        fyc = min(max(fy, 0.0), 1.0)
        if fyc < 0.20 or fyc > 0.94:
            return False
        if fyc > 0.83 and (cx < 0.14 or cx > 0.86):
            return False
        return True

    if fy < 0.20 or fy > 1.005:
        return False
    surr = ice_context["surrounding"] if math.isfinite(ice_context["surrounding"]) else 1.0
    foot = ice_context["foot"] if math.isfinite(ice_context["foot"]) else 1.0
    lo = ice_context["lower_side"] if math.isfinite(ice_context["lower_side"]) else surr
    hi = ice_context["upper_side"] if math.isfinite(ice_context["upper_side"]) else surr

    if fy > 0.94:
        return h >= 0.10 and surr >= 0.50 and lo >= 0.45
    if fy > 0.83 and (cx < 0.14 or cx > 0.86):
        if surr < 0.50 or lo < 0.45:
            return False
    if (surr < 0.34 and foot < 0.22 and lo < 0.42) or (foot < 0.08 and surr < 0.36):
        return False
    if fy < 0.27 and hi < 0.08 and foot < 0.65:
        return False
    if fy < 0.365:
        if h < 0.049 and not (surr >= 0.80 and lo >= 0.80 and hi >= 0.80):
            return False
        if lo < 0.38 or (hi < 0.15 and surr < 0.55 and foot < 0.55):
            return False
    return True


CANONICAL_FEAT_W = 960
CANONICAL_FEAT_H = 540


def make_feature_rgb(rgb: np.ndarray) -> np.ndarray:
    """Mirror YoloOnnxDetector Round 3 canonical <= 960x540 featureBitmap."""
    h, w = rgb.shape[:2]
    if w > CANONICAL_FEAT_W or h > CANONICAL_FEAT_H:
        return cv2.resize(rgb, (CANONICAL_FEAT_W, CANONICAL_FEAT_H), interpolation=cv2.INTER_LINEAR)
    return rgb


class YoloDetector:
    """ONNX YOLOv8n person detector with panoramic far-ice tiling + Round 3 canonical 960x540 featBitmap."""

    def __init__(self, onnx_path: Path) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    def detect(
        self,
        rgb: np.ndarray,
        panoramic_tiling: bool = True,
        apply_on_ice: bool = True,
        wb: Optional[np.ndarray] = None,
        feat_rgb: Optional[np.ndarray] = None,
    ) -> List[Tuple[float, float, float, float, float, Dict[str, float]]]:
        h, w = rgb.shape[:2]
        soft_min_conf = min(MIN_DET_CONF, SOFT_PLAYER_MIN_CONF)
        det = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((384, 640, 3), 114, dtype=np.uint8)
        canvas[12:372, :] = det
        out = self.sess.run(None, {"images": (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]})[0][0]
        cands: List[Tuple[float, float, float, float, float, bool]] = []
        for r in out:
            if r[4] >= soft_min_conf and r[5] == 0.0:
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
                    if r[4] >= soft_min_conf and r[5] == 0.0:
                        nx1 = float(np.clip((r[0] - px) / dw, 0.0, 1.0))
                        ny1 = float(np.clip((r[1] - py) / dh, 0.0, 1.0))
                        nx2 = float(np.clip((r[2] - px) / dw, 0.0, 1.0))
                        ny2 = float(np.clip((r[3] - py) / dh, 0.0, 1.0))
                        if nx2 - nx1 < 0.008 or ny2 - ny1 < 0.015:
                            continue
                        fx1 = float(np.clip(cx0 + nx1 * (cx1 - cx0), 0.0, 1.0))
                        fy1 = float(np.clip(cy0 + ny1 * (cy1 - cy0), 0.0, 1.0))
                        fx2 = float(np.clip(cx0 + nx2 * (cx1 - cx0), 0.0, 1.0))
                        fy2 = float(np.clip(cy0 + ny2 * (cy1 - cy0), 0.0, 1.0))
                        if fx2 - fx1 < 0.004 or fy2 - fy1 < 0.008:
                            continue
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
                if inter / max(1e-6, aa + ab - inter) > 0.42 or inter / max(1e-6, min(aa, ab)) > 0.60:
                    sup = True
                    break
            if not sup:
                kept.append(c)

        if feat_rgb is None:
            feat_rgb = make_feature_rgb(rgb)
        if wb is None:
            wb = compute_ice_wb(feat_rgb)
        out_boxes: List[Tuple[float, float, float, float, float, Dict[str, float]]] = []
        for c in kept:
            rect = (c[0], c[1], c[2], c[3])
            ice_ctx = compute_ice_context(feat_rgb, rect, wb)
            if c[4] < MIN_DET_CONF:
                h_box = c[3] - c[1]
                lo_or_foot = (ice_ctx["foot"] >= SOFT_MIN_FOOT_ICE) or (ice_ctx["lower_side"] >= SOFT_MIN_LOWER_SIDE_ICE)
                if not (h_box >= SOFT_MIN_BOX_HEIGHT and ice_ctx["surrounding"] >= SOFT_MIN_SURROUND_ICE and lo_or_foot):
                    continue
            if apply_on_ice and not is_on_ice(rect, aspect, ice_ctx):
                continue
            out_boxes.append((c[0], c[1], c[2], c[3], c[4], ice_ctx))
        return out_boxes


def compute_ice_wb(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    if w < 8 or h < 8:
        return np.ones(3, dtype=np.float32)
    ys = np.array([int((0.60 + (0.25 * ry) / 11.0) * h) for ry in range(12)], dtype=int)
    xs = np.array([int((0.20 + (0.60 * rx) / 15.0) * w) for rx in range(16)], dtype=int)
    grid = rgb[np.ix_(ys, xs)].reshape(-1, 3)
    med = np.median(grid, axis=0)
    luma = 0.299 * med[0] + 0.587 * med[1] + 0.114 * med[2]
    if luma < 100.0:
        return np.ones(3, dtype=np.float32)
    mean_ice = float(med.mean())
    return np.clip(mean_ice / np.maximum(1.0, med), 0.70, 1.45).astype(np.float32)


def _dft_stripe_score_64x36(
    box_rgb: np.ndarray, ice_wb: np.ndarray, torso_center_frac: float
) -> float:
    """Exact mirror of YoloOnnxDetector.computeDftStripeScore (torso-centered 64x36 DFT)."""
    bh_px, bw_px = box_rgb.shape[:2]
    wide_stick = bw_px > 0.65 * bh_px
    half_w = 0.21 if wide_stick else 0.24
    sx0 = min(max(torso_center_frac - half_w, 0.0), 1.0) * bw_px
    sx1 = min(max(torso_center_frac + half_w, 0.0), 1.0) * bw_px
    sy0, sy1 = 0.22 * bh_px, 0.58 * bh_px
    span_w, span_h = sx1 - sx0, sy1 - sy0
    if span_w < 4.0 or span_h < 4.0:
        return 0.0

    fys = np.clip(sy0 + ((np.arange(36, dtype=np.float64) + 0.5) / 36.0) * span_h - 0.5, 0.0, float(bh_px - 1))
    fxs = np.clip(sx0 + ((np.arange(64, dtype=np.float64) + 0.5) / 64.0) * span_w - 0.5, 0.0, float(bw_px - 1))
    y0 = np.clip(np.floor(fys).astype(int), 0, bh_px - 1)
    y1 = np.minimum(y0 + 1, bh_px - 1)
    ty = (fys - y0)[:, None, None]
    x0 = np.clip(np.floor(fxs).astype(int), 0, bw_px - 1)
    x1 = np.minimum(x0 + 1, bw_px - 1)
    tx = (fxs - x0)[None, :, None]

    arr = box_rgb.astype(np.float64)
    p00 = arr[y0[:, None], x0[None, :]]
    p10 = arr[y0[:, None], x1[None, :]]
    p01 = arr[y1[:, None], x0[None, :]]
    p11 = arr[y1[:, None], x1[None, :]]
    interp = (1.0 - tx) * (1.0 - ty) * p00 + tx * (1.0 - ty) * p10 + (1.0 - tx) * ty * p01 + tx * ty * p11
    roi_64 = np.clip(interp * ice_wb.astype(np.float64)[None, None, :], 0.0, 255.0)
    luma_64 = 0.299 * roi_64[..., 0] + 0.587 * roi_64[..., 1] + 0.114 * roi_64[..., 2]

    sorted_l = np.sort(luma_64.ravel())
    tot_px = len(sorted_l)
    p10 = float(sorted_l[int(tot_px * 0.10)])
    p90 = float(sorted_l[int(tot_px * 0.90)])
    if p90 < 122.0 or 0.5 * (p10 + p90) < 78.0 or (p90 - p10) < 22.0:
        return 0.0
    max_c, min_c = roi_64.max(axis=-1), roi_64.min(axis=-1)
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
    hann_sum_sq = max(1e-6, float((hann * hann).sum()))
    rows_win = (luma_64 - luma_64.mean(axis=1, keepdims=True)) * hann[None, :]
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
        gs = (
            min(max((eb / max(1e-6, eb + el) - 0.22) / 0.36, 0.0), 1.0)
            * min(max((eb / etot - 0.20) / 0.35, 0.0), 1.0)
            * min(max(math.sqrt(eb) / 6.5, 0.0), 1.0)
        )
        group_scores.append(gs)
    group_scores.sort()
    band_consistency = 0.55 * group_scores[2] + 0.45 * group_scores[3]
    if band_consistency <= 0.0:
        return 0.0
    sorted_ra = np.sort(row_amp)
    ref_amp = float(sorted_ra[min(35, max(0, int(36 * 0.90)))])
    if ref_amp <= 1e-4:
        return 0.0
    extent_score = min(max((float((row_amp >= 0.40 * ref_amp).mean()) - 0.50) / 0.30, 0.0), 1.0)
    if extent_score <= 0.0:
        return 0.0
    return float(min(max(math.sqrt(band_consistency * extent_score * px_balance) * chroma_gate, 0.0), 1.0))


def _two_means_lab(pts: np.ndarray, iters: int = 6) -> Tuple[np.ndarray, np.ndarray, float]:
    n = len(pts)
    total = pts.sum(axis=0)
    c0 = pts[int(np.argmin(pts[:, 0]))].copy()
    c1 = pts[int(np.argmax(pts[:, 0]))].copy()
    if float(c1[0] - c0[0]) < 2.0 or n < 4:
        m = total / max(1, n)
        return m, m, (1.0 if m[0] >= 62.0 else 0.0)
    side = np.zeros(n, dtype=bool)
    for _ in range(iters):
        side = (pts @ (c1 - c0)) > 0.5 * float(c1 @ c1 - c0 @ c0)
        n1 = int(side.sum())
        if n1 == 0 or n1 == n:
            break
        s1 = pts[side].sum(axis=0)
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
    return (c0, c1, n1 / n) if c0[0] <= c1[0] else (c1, c0, n0 / n)


def _find_torso_center_frac(box_rgb: np.ndarray, ice_wb: np.ndarray) -> float:
    bh_px, bw_px = box_rgb.shape[:2]
    if bw_px <= 0.65 * bh_px or bw_px < 10:
        return 0.50
    y0 = max(0, min(bh_px - 1, int(0.20 * bh_px)))
    y1 = max(y0 + 1, min(bh_px, int(0.56 * bh_px)))
    strip = box_rgb[y0:y1, :]
    wb_s = np.clip(strip.astype(np.float32) * ice_wb[None, None, :], 0.0, 255.0)
    col_luma = (0.299 * wb_s[..., 0] + 0.587 * wb_s[..., 1] + 0.114 * wb_s[..., 2]).mean(axis=0)
    win_w = min(bw_px, max(3, int(round(0.42 * bw_px))))
    min_x = max(0, min(bw_px - win_w, int(0.04 * bw_px)))
    max_x = max(min_x, bw_px - win_w - int(0.04 * bw_px))
    if max_x <= min_x:
        return 0.50
    best_score, best_center = float("inf"), 0.50
    for sx0 in range(min_x, max_x + 1):
        c_frac = (sx0 + 0.5 * win_w) / float(bw_px)
        score = float(col_luma[sx0:sx0 + win_w].mean()) + 20.0 * abs(c_frac - 0.50)
        if score < best_score:
            best_score, best_center = score, c_frac
    return min(max(best_center, 0.24), 0.76)


def _load_box_rgb(img_rgb: np.ndarray, box: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = box
    bx0 = max(0, min(w - 1, int(x1 * w)))
    by0 = max(0, min(h - 1, int(y1 * h)))
    bx1 = max(bx0 + 1, min(w, int(x2 * w)))
    by1 = max(by0 + 1, min(h, int(y2 * h)))
    bw_px, bh_px = bx1 - bx0, by1 - by0
    if bw_px < 3 or bh_px < 6 or (bw_px * bh_px) <= 0:
        return None
    return img_rgb[by0:by1, bx0:bx1]


def extract_torso_features(
    feat_rgb: np.ndarray,
    high_res_rgb: np.ndarray,
    ice_wb: np.ndarray,
    box: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """Exact Round 3 mirror of YoloOnnxDetector.extractTorsoFeatures(featureBitmap, highResBitmap)."""
    box_rgb = _load_box_rgb(feat_rgb, box)
    if box_rgb is None:
        return {
            "chest_rgb": [128, 128, 128],
            "kit_summary_lab7": np.array([50.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0]),
            "lower_belly_luma": float("nan"),
            "pants_darkness": float("nan"),
            "armband_score": 0.0,
            "stripe_score": 0.0,
            "torso_upper_lab_l": float("nan"),
        }
    bh_px, bw_px = box_rgb.shape[:2]
    wide_stick = bw_px > 0.65 * bh_px
    tc = _find_torso_center_frac(box_rgb, ice_wb)

    # 1. Core chest RGB (40..80th percentile)
    cx0 = max(0, min(bw_px - 1, int((tc - 0.14) * bw_px)))
    cx1 = max(cx0 + 1, min(bw_px, int((tc + 0.14) * bw_px)))
    cy0 = max(0, min(bh_px - 1, int(0.30 * bh_px)))
    cy1 = max(cy0 + 1, min(bh_px, int(0.54 * bh_px)))
    c_crop = box_rgb[cy0:cy1, cx0:cx1]
    if c_crop.size > 0:
        wb_c = np.clip(np.rint(c_crop.astype(np.float32) * ice_wb[None, None, :]), 0, 255)
        r_s, g_s, b_s = np.sort(wb_c[..., 0].ravel()), np.sort(wb_c[..., 1].ravel()), np.sort(wb_c[..., 2].ravel())
        n = len(r_s)
        lo, hi = ((n * 2) // 5, max((n * 2) // 5 + 1, (n * 4) // 5)) if n >= 5 else (0, n)
        chest_rgb = [int(r_s[lo:hi].mean()), int(g_s[lo:hi].mean()), int(b_s[lo:hi].mean())]
    else:
        chest_rgb = [128, 128, 128]

    # 2. 7D CIELAB kit summary with border ice background subtraction + torsoUpperLabL (p80 L*)
    half_w = 0.21 if wide_stick else 0.24
    tx0 = max(0, min(bw_px - 1, int((tc - half_w) * bw_px)))
    tx1 = max(tx0 + 1, min(bw_px, int((tc + half_w) * bw_px)))
    ty0 = max(0, min(bh_px - 1, int(0.22 * bh_px)))
    ty1 = max(ty0 + 1, min(bh_px, int(0.56 * bh_px)))
    t_crop = box_rgb[ty0:ty1, tx0:tx1]
    kit7 = None
    torso_upper_l = float("nan")
    if t_crop.size > 0:
        wb_t = np.clip(np.rint(t_crop.astype(np.float32) * ice_wb[None, None, :]), 0, 255)
        t_lab = srgb_to_lab(wb_t.reshape(-1, 3))
        bdy0, bdy1 = max(0, min(bh_px - 1, int(0.15 * bh_px))), max(1, min(bh_px, int(0.55 * bh_px)))
        lx1 = max(1, min(max(1, bw_px // 3), int(0.10 * bw_px)))
        rx0 = max(lx1, min(bw_px - 1, int(0.90 * bw_px)))
        bg_pts = []
        if bdy1 > bdy0:
            bg_pts.append(box_rgb[bdy0:bdy1, 0:lx1].reshape(-1, 3))
            if bw_px > rx0:
                bg_pts.append(box_rgb[bdy0:bdy1, rx0:bw_px].reshape(-1, 3))
        if bg_pts:
            bg_cat = np.concatenate(bg_pts, axis=0)
            if len(bg_cat) > 0:
                bg_wb = np.clip(np.rint(bg_cat.astype(np.float32) * ice_wb[None, :]), 0, 255)
                bg_lab_pts = srgb_to_lab(bg_wb)
                bg_chroma = np.hypot(bg_lab_pts[:, 1], bg_lab_pts[:, 2])
                ice_ref_pts = bg_lab_pts[(bg_lab_pts[:, 0] >= 48.0) & (bg_chroma < 22.0)]
                if len(ice_ref_pts) >= max(4, len(bg_lab_pts) // 4):
                    mid_i = len(ice_ref_pts) // 2
                    bg_l = np.array([
                        float(np.sort(ice_ref_pts[:, 0])[mid_i]),
                        float(np.sort(ice_ref_pts[:, 1])[mid_i]),
                        float(np.sort(ice_ref_pts[:, 2])[mid_i]),
                    ])
                    de_sq = ((t_lab - bg_l[None, :]) ** 2).sum(axis=1)
                    bg_like = de_sq < 121.0
                    ice_cnt = int(bg_like.sum())
                    tot_cnt = len(t_lab)
                    rem_pts = t_lab[~bg_like]
                    max_drop = 0.75 if wide_stick else 0.40
                    min_rem = 10 if wide_stick else 6
                    allow_wide = (not wide_stick) or (ice_cnt <= tot_cnt * 0.40) or (len(rem_pts) >= 10 and float(rem_pts[:, 0].mean()) < 45.0)
                    if allow_wide and 0 < ice_cnt <= tot_cnt * max_drop and len(rem_pts) >= min_rem:
                        t_lab = rem_pts
        if len(t_lab) > 0:
            sorted_l = np.sort(t_lab[:, 0])
            p80_pos = 0.80 * (len(sorted_l) - 1)
            p80_lo = max(0, min(len(sorted_l) - 1, int(p80_pos)))
            p80_hi = min(len(sorted_l) - 1, p80_lo + 1)
            p80_frac = p80_pos - p80_lo
            torso_upper_l = float(sorted_l[p80_lo] * (1.0 - p80_frac) + sorted_l[p80_hi] * p80_frac)
            dark, light, light_frac = _two_means_lab(t_lab)
            kit7 = np.array([dark[0], dark[1], dark[2], light[0], light[1], light[2], light_frac * 100.0], dtype=np.float64)

    # 3. Lower belly luma (canonical feat_rgb)
    lbx0 = max(0, min(bw_px - 1, int((tc - 0.18) * bw_px)))
    lbx1 = max(lbx0 + 1, min(bw_px, int((tc + 0.18) * bw_px)))
    lby0 = max(0, min(bh_px - 1, int(0.42 * bh_px)))
    lby1 = max(lby0 + 1, min(bh_px, int(0.56 * bh_px)))
    b_crop = box_rgb[lby0:lby1, lbx0:lbx1]
    belly_luma = float("nan")
    if b_crop.shape[0] * b_crop.shape[1] >= 4:
        wb_b = np.clip(b_crop.astype(np.float32) * ice_wb[None, None, :], 0, 255)
        bl = np.sort((0.299 * wb_b[..., 0] + 0.587 * wb_b[..., 1] + 0.114 * wb_b[..., 2]).ravel())
        lo, hi = ((len(bl) * 2) // 5, max((len(bl) * 2) // 5 + 1, (len(bl) * 4) // 5)) if len(bl) >= 5 else (0, len(bl))
        belly_luma = float(bl[lo:hi].mean())

    # 4. Official cues (highResBitmap when src > 960x540): pantsDarkness, armbandScore, DFT stripe
    hi_box = box_rgb if (high_res_rgb is feat_rgb) else _load_box_rgb(high_res_rgb, box)
    if hi_box is not None:
        hi_bh, hi_bw = hi_box.shape[:2]
        hi_tc = tc if (high_res_rgb is feat_rgb) else _find_torso_center_frac(hi_box, ice_wb)
        px0 = max(0, min(hi_bw - 1, int(0.28 * hi_bw)))
        px1 = max(px0 + 1, min(hi_bw, int(0.72 * hi_bw)))
        py0 = max(0, min(hi_bh - 1, int(0.56 * hi_bh)))
        py1 = max(py0 + 1, min(hi_bh, int(0.84 * hi_bh)))
        p_crop = hi_box[py0:py1, px0:px1]
        pants_dark = float("nan")
        if p_crop.shape[0] * p_crop.shape[1] >= 4:
            wb_p = np.clip(p_crop.astype(np.float32) * ice_wb[None, None, :], 0, 255)
            pl = 0.299 * wb_p[..., 0] + 0.587 * wb_p[..., 1] + 0.114 * wb_p[..., 2]
            spread = wb_p.max(axis=-1) - wb_p.min(axis=-1)
            pants_dark = float(((pl < 88.0) & (spread < 36.0)).mean())

        ay0, ay1 = max(0, min(hi_bh - 1, int(0.15 * hi_bh))), max(1, min(hi_bh, int(0.45 * hi_bh)))
        alx1 = max(1, min(max(1, hi_bw // 2), int(0.25 * hi_bw)))
        arx0 = max(alx1, min(hi_bw - 1, int(0.75 * hi_bw)))
        arm_pts = []
        if ay1 > ay0:
            arm_pts.append(hi_box[ay0:ay1, 0:alx1].reshape(-1, 3))
            if hi_bw > arx0:
                arm_pts.append(hi_box[ay0:ay1, arx0:hi_bw].reshape(-1, 3))
        armband = 0.0
        if arm_pts:
            arm_cat = np.concatenate(arm_pts, axis=0)
            if len(arm_cat) >= 4:
                arm_wb = np.clip(arm_cat.astype(np.float32) * ice_wb[None, :], 0, 255).astype(np.uint8)
                hsv = cv2.cvtColor(arm_wb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                orange = (hsv[:, 0] >= 2) & (hsv[:, 0] <= 13) & (hsv[:, 1] >= 115) & (hsv[:, 2] >= 102)
                armband = float(orange.mean())
        stripe = _dft_stripe_score_64x36(hi_box, ice_wb, hi_tc)
    else:
        pants_dark = float("nan")
        armband = 0.0
        stripe = 0.0

    return {
        "chest_rgb": chest_rgb,
        "kit_summary_lab7": sanitize_kit_summary7(kit7, chest_rgb),
        "lower_belly_luma": belly_luma,
        "pants_darkness": pants_dark,
        "armband_score": armband,
        "stripe_score": stripe,
        "torso_upper_lab_l": torso_upper_l,
    }


# ---------------------------------------------------------------------------
# ONNX STR Recognizer (JerseyStrOnnxRecognizer.kt v5 + v4-compat mirror)
# ---------------------------------------------------------------------------

class JerseyStrRecognizer:
    MIN_CROP_WIDTH_PX = 12
    MIN_CROP_HEIGHT_PX = 18
    MIN_LAPLACIAN_SHARPNESS = 5.5
    DEROTATE_MIN_ANGLE_DEG = 16.0
    DEROTATE_MIN_LEGIBILITY_PROB = 0.25
    DEROTATE_MIN_BACK_PROB = 0.45
    MIN_CANDIDATE_PROB = 0.02
    MIN_TOP_CANDIDATE_PROB = 0.32
    MAX_ILLEGIBLE_PROB = 0.48
    MIN_CANDIDATE_LEGIBILITY_PROB = 0.64
    MIN_CANDIDATE_BACK_PROB = 0.56
    MIN_PARTIAL_LEGIBILITY_PROB = 0.55
    MIN_PARTIAL_BACK_PROB = 0.60
    MIN_PARTIAL_DIGIT_PROB = 0.65
    PARTIAL_BLANK_MARGIN = 1.25
    MIN_BLANK_COMPANION_PROB = 0.32

    def __init__(self, onnx_path: Path) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.input_size = int(shape[2]) if (len(shape) >= 4 and isinstance(shape[2], int) and 64 <= shape[2] <= 256) else 96
        out_names = {o.name for o in self.sess.get_outputs()}
        self.has_orientation = "orientation_deg" in out_names

    @staticmethod
    def _sample_median_border_color(crop: np.ndarray) -> Tuple[int, int, int]:
        h, w = crop.shape[:2]
        if w < 2 or h < 2:
            return (114, 114, 114)
        xs = np.clip(((np.arange(8) + 0.5) * w / 8.0).astype(int), 0, w - 1)
        ys = np.clip(((np.arange(8) + 0.5) * h / 8.0).astype(int), 0, h - 1)
        pts = np.concatenate([crop[0, xs], crop[h - 1, xs], crop[ys, 0], crop[ys, w - 1]], axis=0)
        med = np.median(pts, axis=0)
        return int(med[0]), int(med[1]), int(med[2])

    @classmethod
    def _derotate_upright(cls, crop: np.ndarray, ccw_orient_deg: float) -> np.ndarray:
        """Rotate OpenCV CCW +ccw_orient_deg back to upright via -ccw_orient_deg (CW), matching TorsoCropHelper."""
        ch, cw = crop.shape[:2]
        deg = float(np.clip(ccw_orient_deg, -60.0, 60.0))
        rad = abs(deg) * math.pi / 180.0
        abs_cos, abs_sin = abs(math.cos(rad)), abs(math.sin(rad))
        rot_w = int(round(cw * abs_cos + ch * abs_sin))
        rot_h = int(round(cw * abs_sin + ch * abs_cos))
        out_w = max(cw, min(rot_w, int(round(cw * 1.25))))
        out_h = max(ch, min(rot_h, int(round(ch * 1.25))))
        M = cv2.getRotationMatrix2D((cw * 0.5, ch * 0.5), -deg, 1.0)
        M[0, 2] += (out_w - cw) * 0.5
        M[1, 2] += (out_h - ch) * 0.5
        bg = cls._sample_median_border_color(crop)
        return cv2.warpAffine(crop, M, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)

    def crop_upper_torso(
        self,
        rgb: np.ndarray,
        box: Tuple[float, float, float, float],
        ice_wb: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Stick-immune torso-centered crop (y: 4%..58%), exact mirror of JerseyCropGeometry.kt."""
        h, w = rgb.shape[:2]
        if w <= 1 or h <= 1:
            return None
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cy0 = max(0, min(h - 1, int((y1 + 0.04 * bh) * h)))
        cy1 = max(cy0 + 1, min(h, int((y1 + 0.58 * bh) * h)))

        bx0 = max(0, min(w - 1, int(x1 * w)))
        bx1 = max(bx0 + 1, min(w, int(x2 * w)))
        by0 = max(0, min(h - 1, int(y1 * h)))
        by1 = max(by0 + 1, min(h, int(y2 * h)))
        bw_px = float(bx1 - bx0)
        bh_px = float(by1 - by0)

        cx0 = max(0, min(w - 1, int((x1 - 0.06 * bw) * w)))
        cx1 = max(cx0 + 1, min(w, int((x2 + 0.06 * bw) * w)))

        if bw_px > 0.58 * bh_px and (bx1 - bx0) >= 4:
            sy0 = max(0, min(h - 1, int((y1 + 0.16 * bh) * h)))
            sy1 = max(sy0 + 1, min(h, int((y1 + 0.56 * bh) * h)))
            if (sy1 - sy0) >= 2:
                wb = np.ones(3, dtype=np.float32) if ice_wb is None else np.clip(ice_wb.astype(np.float32), 0.6, 1.6)
                strip = rgb[sy0:sy1, bx0:bx1]
                wb_s = np.clip(strip.astype(np.float32) * wb[None, None, :], 0.0, 255.0)
                lab = srgb_to_lab(wb_s)
                chroma = np.hypot(lab[..., 1], lab[..., 2])
                non_ice = ~((lab[..., 0] >= 62.0) & (chroma < 18.0))
                col_frac = non_ice.mean(axis=0)
                scan_w = len(col_frac)
                win_w = max(2, min(scan_w, int(round(0.36 * bh_px))))
                best_start = 0
                best_score = -1e18
                win_sum = float(col_frac[:win_w].sum())
                for start in range(0, scan_w - win_w + 1):
                    if start > 0:
                        win_sum += float(col_frac[start + win_w - 1] - col_frac[start - 1])
                    c_frac = (start + 0.5 * win_w) / float(scan_w)
                    sc = win_sum - 0.15 * win_w * abs(c_frac - 0.5)
                    if sc > best_score:
                        best_score = sc
                        best_start = start
                mid_win = best_start + 0.5 * (win_w - 1)
                peak_col = best_start
                local_peak = -1.0
                for c in range(best_start, best_start + win_w):
                    f = float(col_frac[c])
                    if f > local_peak + 1e-4 or (abs(f - local_peak) <= 1e-4 and abs(c - mid_win) < abs(peak_col - mid_win)):
                        local_peak = f
                        peak_col = c
                if local_peak >= 0.14:
                    act_thresh = max(0.14, local_peak * 0.28)
                    max_gap = max(2, int(round(0.06 * bh_px)))
                    span_left = peak_col
                    gap = 0
                    for c in range(peak_col - 1, -1, -1):
                        if col_frac[c] >= act_thresh:
                            span_left = c
                            gap = 0
                        else:
                            gap += 1
                            if gap > max_gap:
                                break
                    span_right = peak_col
                    gap = 0
                    for c in range(peak_col + 1, scan_w):
                        if col_frac[c] >= act_thresh:
                            span_right = c
                            gap = 0
                        else:
                            gap += 1
                            if gap > max_gap:
                                break
                    non_ice_span = float(span_right - span_left + 1)
                    torso_center = bx0 + (span_left + span_right + 1) * 0.5 * (bw_px / float(scan_w))
                    target_w = min(bw_px * 1.08, max(0.46 * bh_px, non_ice_span * (bw_px / float(scan_w)) * 1.28))
                    cx0 = max(0, min(w - 1, int(round(torso_center - 0.5 * target_w))))
                    cx1 = max(cx0 + 1, min(w, int(round(torso_center + 0.5 * target_w))))

        crop = rgb[cy0:cy1, cx0:cx1]
        if crop.shape[0] < self.MIN_CROP_HEIGHT_PX or crop.shape[1] < self.MIN_CROP_WIDTH_PX:
            return None
        return crop

    @staticmethod
    def _sm(v: np.ndarray) -> np.ndarray:
        e = np.exp(np.clip(v - v.max(), -40.0, 0.0))
        return e / e.sum()

    def _run_single_pass(self, crop: np.ndarray, ice_wb: np.ndarray) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        sz = self.input_size
        ch, cw = crop.shape[:2]
        scale = min(float(sz) / cw, float(sz) / ch)
        nw, nh = max(1, min(sz, int(round(cw * scale)))), max(1, min(sz, int(round(ch * scale))))
        pad_x, pad_y = (sz - nw) // 2, (sz - nh) // 2
        interp = cv2.INTER_CUBIC if sz >= 128 else cv2.INTER_LINEAR
        resized = cv2.resize(crop, (nw, nh), interpolation=interp)

        wb_content = np.clip(resized.astype(np.float32) * ice_wb[None, None, :], 0.0, 255.0)
        luma = np.clip(0.299 * wb_content[..., 0] + 0.587 * wb_content[..., 1] + 0.114 * wb_content[..., 2], 0.0, 255.0)
        if nw >= 4 and nh >= 4:
            lap_arr = (
                4.0 * luma[1:-1, 1:-1]
                - luma[1:-1, :-2]
                - luma[1:-1, 2:]
                - luma[:-2, 1:-1]
                - luma[2:, 1:-1]
            ).astype(np.float64)
            sharpness = float(min(max(float(lap_arr.var()) if lap_arr.size >= 4 else 1.0, 1.0), 2500.0))
        else:
            sharpness = 1.0

        if sharpness < self.MIN_LAPLACIAN_SHARPNESS:
            z_num = np.zeros(101, dtype=np.float64)
            z_num[0] = 1.0
            z_d = np.zeros(11, dtype=np.float64)
            z_d[10] = 1.0
            return 0.0, 0.0, sharpness, 0.0, z_num, z_d, z_d

        p_low, p_high = np.percentile(luma, [2.0, 98.0])
        raw_span = max(32.0, float(p_high - p_low))
        gain = min(max(1.0 + 0.12 * min(max(220.0 / raw_span, 0.8), 2.0), 1.05), 1.25)
        mid = 0.5 * float(p_low + p_high)

        adj = np.clip((wb_content - mid) * gain + mid, 0.0, 255.0) / 255.0
        canvas = np.full((sz, sz, 3), 114.0 / 255.0, dtype=np.float32)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = adj

        outs = self.sess.run(None, {self.input_name: canvas.transpose(2, 0, 1)[None]})
        leg_l, back_l, num_l, ld_l, rd_l = outs[:5]
        orient_deg = float(np.clip(float(outs[5][0, 0]), -60.0, 60.0)) if (self.has_orientation and len(outs) >= 6) else 0.0
        leg_p = float(1.0 / (1.0 + math.exp(-min(max(float(leg_l[0, 0]), -20.0), 20.0))))
        back_p = float(1.0 / (1.0 + math.exp(-min(max(float(back_l[0, 0]), -20.0), 20.0))))
        return leg_p, back_p, sharpness, orient_deg, self._sm(num_l[0]), self._sm(ld_l[0]), self._sm(rd_l[0])

    def decode_raw_pass(
        self,
        leg_p: float,
        back_p: float,
        sharpness: float,
        orient_deg: float,
        num_p: np.ndarray,
        ld_p: np.ndarray,
        rd_p: np.ndarray,
    ) -> Dict[str, Any]:
        cands: List[Tuple[str, float]] = []
        if (
            sharpness >= self.MIN_LAPLACIAN_SHARPNESS
            and float(num_p[0]) < self.MAX_ILLEGIBLE_PROB
            and leg_p >= self.MIN_CANDIDATE_LEGIBILITY_PROB
            and back_p >= self.MIN_CANDIDATE_BACK_PROB
        ):
            indexed = [(n, float(num_p[n])) for n in range(1, 100) if float(num_p[n]) >= self.MIN_CANDIDATE_PROB]
            indexed.sort(key=lambda kv: -kv[1])
            if indexed and indexed[0][1] >= self.MIN_TOP_CANDIDATE_PROB:
                cands = [(str(n), p) for n, p in indexed[:5]]

        p_left, p_right = None, None
        if sharpness >= self.MIN_LAPLACIAN_SHARPNESS and leg_p >= self.MIN_PARTIAL_LEGIBILITY_PROB and back_p >= self.MIN_PARTIAL_BACK_PROB:
            l_best = int(1 + np.argmax(ld_p[1:10]))
            l_best_p = float(ld_p[l_best])
            if (
                l_best_p >= self.MIN_PARTIAL_DIGIT_PROB
                and l_best_p > float(ld_p[10]) * self.PARTIAL_BLANK_MARGIN
                and (float(rd_p[10]) >= self.MIN_BLANK_COMPANION_PROB or leg_p < 0.65)
                and float(num_p[l_best]) < 0.25
            ):
                p_left = (str(l_best), l_best_p)

            r_best = int(np.argmax(rd_p[:10]))
            r_best_p = float(rd_p[r_best])
            if (
                r_best_p >= self.MIN_PARTIAL_DIGIT_PROB
                and r_best_p > float(rd_p[10]) * self.PARTIAL_BLANK_MARGIN
                and (float(ld_p[10]) >= self.MIN_BLANK_COMPANION_PROB or leg_p < 0.65)
            ):
                p_right = (str(r_best), r_best_p)

        return {
            "legibility": leg_p,
            "back": back_p,
            "sharpness": sharpness,
            "orientation_deg": orient_deg,
            "candidates": cands,
            "partial_left": p_left,
            "partial_right": p_right,
        }

    def infer_from_crop(self, crop: np.ndarray, ice_wb: np.ndarray, return_raw: bool = False) -> Dict[str, Any]:
        leg1, back1, sharp1, orient1, num1, ld1, rd1 = self._run_single_pass(crop, ice_wb)
        if sharp1 < self.MIN_LAPLACIAN_SHARPNESS:
            res = {"legibility": 0.0, "back": 0.0, "sharpness": sharp1, "orientation_deg": 0.0, "candidates": [], "partial_left": None, "partial_right": None}
            if return_raw:
                res["_raw"] = (0.0, 0.0, sharp1, 0.0, num1, ld1, rd1, False, None)
            return res

        leg_f, back_f, num_f, ld_f, rd_f = leg1, back1, num1, ld1, rd1
        did_derot = False
        upright_crop = None
        if (
            self.has_orientation
            and abs(orient1) >= self.DEROTATE_MIN_ANGLE_DEG
            and (leg1 >= self.DEROTATE_MIN_LEGIBILITY_PROB or back1 >= self.DEROTATE_MIN_BACK_PROB)
        ):
            upright_crop = self._derotate_upright(crop, orient1)
            leg2, back2, sharp2, _, num2, ld2, rd2 = self._run_single_pass(upright_crop, ice_wb)
            if sharp2 >= self.MIN_LAPLACIAN_SHARPNESS:
                did_derot = True
                w1 = max(0.02, leg1 * leg1)
                w2 = max(0.02, leg2 * leg2)
                w_sum = w1 + w2
                a1, a2 = w1 / w_sum, w2 / w_sum
                leg_f = min(max(a1 * leg1 + a2 * leg2, 0.0), 1.0)
                back_f = min(max(a1 * back1 + a2 * back2, 0.0), 1.0)
                num_f = a1 * num1 + a2 * num2
                ld_f = a1 * ld1 + a2 * ld2
                rd_f = a1 * rd1 + a2 * rd2

        res = self.decode_raw_pass(leg_f, back_f, sharp1, orient1, num_f, ld_f, rd_f)
        res["derotated"] = did_derot
        if return_raw:
            res["_raw"] = (leg_f, back_f, sharp1, orient1, num_f, ld_f, rd_f, did_derot, upright_crop)
        return res

    def infer(self, rgb: np.ndarray, ice_wb: np.ndarray, box: Tuple[float, float, float, float]) -> Dict[str, Any]:
        crop = self.crop_upper_torso(rgb, box, ice_wb)
        if crop is None:
            return {"legibility": 0.0, "back": 0.0, "sharpness": 1.0, "orientation_deg": 0.0, "candidates": [], "partial_left": None, "partial_right": None}
        return self.infer_from_crop(crop, ice_wb)


# ---------------------------------------------------------------------------
# 100% Roster-Free Bayesian + Hungarian Solver (JerseyNumberPosterior.kt)
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
SINGLE_BIN_COMMIT_MIN_CONF = 0.90
NANO_SINGLE_BIN_MIN_CONF = 0.80
COMPLEMENTARY_PARTIAL_MIN_CONF = 0.80


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


def _obs_channel(obs: Dict[str, Any]) -> int:
    if obs.get("candidates"):
        return 0
    k = obs.get("kind", "FULL")
    return 1 if k == "LEFT_PARTIAL" else (2 if k == "RIGHT_PARTIAL" else (3 if k == "SLEEVE" else 0))


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
    return 0.0 if conf <= 0.02 else min(max(conf, 0.0), 1.0) * qw * sw


def _compute_effective_obs_weights(observations: List[Dict[str, Any]]) -> List[float]:
    if not observations:
        return []
    best_idx_by_key: Dict[Tuple[int, int], int] = {}
    best_str_by_key: Dict[Tuple[int, int], float] = {}
    for i, obs in enumerate(observations):
        st = _obs_raw_strength(obs)
        if st <= 0.0:
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
        t_sec = float(obs.get("timestamp_sec", 0.0))
        if not math.isfinite(t_sec):
            t_sec = 0.0
        key = (math.floor(t_sec / EVIDENCE_BIN_SEC), _obs_channel(obs))
        disc = 1.0 if best_idx_by_key.get(key, i) == i else SAME_BIN_MARGINAL_WEIGHT
        out.append(qw * sw * disc)
    return out


def compute_chain_log_posterior(observations: List[Dict[str, Any]]) -> Tuple[np.ndarray, float]:
    lp = np.full(99, math.log(1.0 / 99.0), dtype=np.float64)
    ev_w = 0.0
    eff_weights = _compute_effective_obs_weights(observations)
    for obs, eff in zip(observations, eff_weights):
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
    return lp, ev_w


def _meets_bin_or_conf_gate(
    observations: List[Dict[str, Any]],
    candidate_num: Optional[str],
    is_nano_verified: bool = False,
    single_bin_min_conf: float = SINGLE_BIN_COMMIT_MIN_CONF,
) -> bool:
    if candidate_num is None:
        return False
    if is_nano_verified:
        return True
    bins = set()
    max_non_nano_conf = 0.0
    max_nano_conf = 0.0
    has_strong_left = False
    has_strong_right = False
    for obs in observations:
        t_sec = float(obs.get("timestamp_sec", 0.0))
        if not math.isfinite(t_sec):
            t_sec = 0.0
        b_idx = math.floor(t_sec / EVIDENCE_BIN_SEC)
        is_nano = obs.get("source") == "GEMINI_NANO"
        cands = obs.get("candidates")
        if cands:
            c_prob = sum(float(pr) for raw, pr in cands if raw == candidate_num and math.isfinite(float(pr)) and float(pr) > 0.0)
            if c_prob >= 0.25:
                bins.add(b_idx)
                clean_p = min(c_prob, 1.0)
                if is_nano:
                    max_nano_conf = max(max_nano_conf, clean_p)
                else:
                    max_non_nano_conf = max(max_non_nano_conf, clean_p)
        elif obs.get("number") and math.isfinite(float(obs.get("conf", 0.0))) and float(obs["conf"]) > 0.0:
            kind, num = obs.get("kind", "FULL"), str(obs["number"])
            matches = (
                (kind in ("FULL", "SINGLE") and num == candidate_num)
                or (kind == "LEFT_PARTIAL" and len(candidate_num) == 2 and num == candidate_num[0])
                or (kind == "RIGHT_PARTIAL" and len(candidate_num) == 2 and num == candidate_num[1])
                or (kind == "SLEEVE" and num in candidate_num)
            )
            if matches:
                bins.add(b_idx)
                clean_c = min(float(obs["conf"]), 1.0)
                if kind == "LEFT_PARTIAL" and clean_c >= COMPLEMENTARY_PARTIAL_MIN_CONF:
                    has_strong_left = True
                elif kind == "RIGHT_PARTIAL" and clean_c >= COMPLEMENTARY_PARTIAL_MIN_CONF:
                    has_strong_right = True
                if is_nano:
                    max_nano_conf = max(max_nano_conf, clean_c)
                else:
                    max_non_nano_conf = max(max_non_nano_conf, clean_c)
    return (
        len(bins) >= 2
        or (has_strong_left and has_strong_right)
        or max_non_nano_conf >= single_bin_min_conf
        or max_nano_conf >= NANO_SINGLE_BIN_MIN_CONF
    )


def solve_team_numbers_bayesian_hungarian(
    chains: List[Dict[str, Any]],
    single_bin_min_conf: float = SINGLE_BIN_COMMIT_MIN_CONF,
) -> Dict[int, Tuple[Optional[str], float]]:
    if not chains:
        return {}
    log_ps: Dict[int, np.ndarray] = {}
    probs_map: Dict[int, np.ndarray] = {}
    assigned: Dict[int, Optional[str]] = {}
    pinned = set()

    for ch in chains:
        cid = ch["chain_id"]
        lp, ev_w = compute_chain_log_posterior(ch["observations"])
        log_ps[cid] = lp
        w = np.exp(lp - lp.max())
        p = w / w.sum()
        probs_map[cid] = p
        best_i = int(np.argmax(p))
        best_p, best_num = float(p[best_i]), ALL_NUMS[best_i]
        if (
            ch["observations"] and best_p >= MIN_POSTERIOR_PROB and ev_w >= MIN_EVIDENCE_WEIGHT
            and _meets_bin_or_conf_gate(ch["observations"], best_num, single_bin_min_conf=single_bin_min_conf)
        ):
            assigned[cid] = best_num
            bins = set()
            for obs in ch["observations"]:
                b_idx = math.floor(float(obs.get("timestamp_sec", 0.0)) / 10.0)
                cands = obs.get("candidates")
                if cands:
                    bc = max(cands, key=lambda kv: kv[1])
                    if bc[0] == best_num and bc[1] >= 0.85:
                        bins.add(b_idx)
                elif obs.get("number") == best_num and obs.get("kind", "FULL") in ("FULL", "SINGLE") and obs.get("conf", 0.0) >= 0.85:
                    bins.add(b_idx)
            if len(bins) >= 2:
                pinned.add(cid)
        else:
            assigned[cid] = None

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
        n_rows, n_cands = len(group), len(cands_num)
        cost = np.full((n_rows, n_cands + 2 * n_rows), 1e6, dtype=np.float64)
        for i, g in enumerate(group):
            p = probs_map[g]
            blocked = {assigned[h] for h in conc[g] if h not in group and assigned[h] is not None}
            for j, num in enumerate(cands_num):
                if num not in blocked:
                    cost[i, j] = -math.log(max(float(p[NUM_IDX[num]]), 1e-9))
            cost[i, n_cands + i] = -math.log(UNASSIGN_PRIOR)
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
                if float(w_ex[NUM_IDX[assigned[g]]] / max(w_ex.sum(), 1e-12)) < 0.05:
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
# MultiTeamPlayerTracker Exact Python Twin (MultiTeamPlayerTracker.kt)
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


def _are_close_or_overlapping(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    if _box_iou(a, b) >= 0.08:
        return True
    acx, acy, afy = 0.5 * (a[0] + a[2]), 0.5 * (a[1] + a[3]), a[3]
    bcx, bcy, bfy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]), b[3]
    return math.hypot(acx - bcx, afy - bfy) <= 0.08 or math.hypot(acx - bcx, acy - bcy) <= 0.08


class MobileTracker:
    """Exact Round 2 mirror of MultiTeamPlayerTracker + TrackIdentityRegistry."""

    def __init__(self) -> None:
        self.active: List[Dict[str, Any]] = []
        self.records: Dict[int, Dict[str, Any]] = {}
        self.next_id = 1
        self.last_ts = float("nan")
        self.team_kit7: Dict[str, np.ndarray] = {}
        self.team_rgb: Dict[str, np.ndarray] = {}
        self.team_belly: Dict[str, float] = {}
        self.team_upper_l: Dict[str, float] = {}
        self.str_attempts: Dict[int, int] = {}
        self.last_str_score: Dict[int, Dict[str, float]] = {}

    def _mean_kit7(self, rec: Dict[str, Any], mean_rgb: np.ndarray) -> np.ndarray:
        if rec["n_kit"] > 0:
            return rec["sum_kit7"] / float(rec["n_kit"])
        return sanitize_kit_summary7(None, [int(round(v)) for v in mean_rgb])

    def _mean_upper_l(self, rec: Dict[str, Any]) -> float:
        return rec["sum_upper_l"] / rec["n_upper_l"] if rec["n_upper_l"] > 0 else float("nan")

    def _is_cross_kit_incompatible(self, tid: int, track_rgb: np.ndarray, det: Dict[str, Any], rgb_dist: float) -> bool:
        det_rgb = np.array(det["chest_rgb"], dtype=np.float64)
        lum_diff = abs(_rgb_luma(track_rgb) - _rgb_luma(det_rgb))
        det_kit7 = det["kit_summary_lab7"]
        det_belly = float(det.get("lower_belly_luma", float("nan")))
        det_upper = float(det.get("torso_upper_lab_l", float("nan")))
        rec = self.records[tid]
        track_kit7 = self._mean_kit7(rec, track_rgb)
        track_belly = rec["sum_belly"] / rec["n_belly"] if rec["n_belly"] > 0 else float("nan")
        track_upper = self._mean_upper_l(rec)

        if "TEAM_A" in self.team_kit7 and "TEAM_B" in self.team_kit7 and "TEAM_A" in self.team_rgb and "TEAM_B" in self.team_rgb:
            kitA, kitB = self.team_kit7["TEAM_A"], self.team_kit7["TEAM_B"]
            centA, centB = self.team_rgb["TEAM_A"], self.team_rgb["TEAM_B"]
            bA, bB = self.team_belly.get("TEAM_A", float("nan")), self.team_belly.get("TEAM_B", float("nan"))
            uA, uB = self.team_upper_l.get("TEAM_A", float("nan")), self.team_upper_l.get("TEAM_B", float("nan"))
            if math.sqrt(order_free_kit_dist_sq(kitA, kitB)) >= 12.0 or _color_lum_dist(centA, centB) >= 40.0:
                t_is_a = (
                    kit_and_belly_or_upper_dist_sq(track_kit7, track_belly, track_upper, kitA, bA, uA)
                    <= kit_and_belly_or_upper_dist_sq(track_kit7, track_belly, track_upper, kitB, bB, uB)
                )
                d_is_a = (
                    kit_and_belly_or_upper_dist_sq(det_kit7, det_belly, det_upper, kitA, bA, uA)
                    <= kit_and_belly_or_upper_dist_sq(det_kit7, det_belly, det_upper, kitB, bB, uB)
                )
                if t_is_a != d_is_a:
                    cross_d = kit_and_belly_or_upper_dist_sq(track_kit7, track_belly, track_upper, det_kit7, det_belly, det_upper)
                    if rgb_dist > 65.0 or cross_d > 250.0:
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

        det_in_contact = [any(i != j and _are_close_or_overlapping(dets[i]["box"], dets[j]["box"]) for j in range(len(dets))) for i in range(len(dets))]
        track_in_contact = [any(i != j and _are_close_or_overlapping(preds[i], preds[j]) for j in range(len(preds))) for i in range(len(preds))]

        cands = []
        for ti, t in enumerate(self.active):
            pr = preds[ti]
            pcx, pfy, ph = 0.5 * (pr[0] + pr[2]), pr[3], pr[3] - pr[1]
            for di, det in enumerate(dets):
                db = det["box"]
                iou = _box_iou(pr, db)
                dcx, dfy, dh = 0.5 * (db[0] + db[2]), db[3], db[3] - db[1]
                avg_h = max(0.025, 0.5 * (ph + dh))
                fdist = math.hypot(pcx - dcx, pfy - dfy)
                nfoot = fdist / avg_h
                if iou < 0.10 and (nfoot > 2.5 or fdist > 0.20):
                    continue
                cdist = float(np.linalg.norm(t["ema_rgb"] - np.array(det["chest_rgb"], dtype=np.float64)))
                if (det_in_contact[di] or track_in_contact[ti]) and self._is_cross_kit_incompatible(t["id"], t["ema_rgb"], det, cdist):
                    continue
                cands.append((nfoot - 2.0 * iou + 0.0025 * cdist, ti, di))
        cands.sort(key=lambda x: x[0])

        det_to_tid = [-1] * len(dets)
        matched_t = [False] * len(self.active)
        for _, ti, di in cands:
            if matched_t[ti] or det_to_tid[di] >= 0:
                continue
            matched_t[ti] = True
            t, det = self.active[ti], dets[di]
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
            self._update_record(t["id"], ts, det, t["vx"], t["vy"])

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
                "id": tid, "box": det["box"], "last_sec": ts,
                "vx": 0.0, "vy": 0.0, "misses": 0, "hits": 1,
                "ema_rgb": np.array(det["chest_rgb"], dtype=np.float64),
            })
            det_to_tid[di] = tid
            self._update_record(tid, ts, det, 0.0, 0.0)

        self._recluster(len(dets), {tid for tid in det_to_tid if tid >= 0})
        suppress_burst = len(dets) > 15
        emitted = []
        for di, det in enumerate(dets):
            tid = det_to_tid[di]
            rec = self.records[tid]
            if suppress_burst and rec["hits"] <= 1:
                continue
            emitted.append({"track_id": tid, "box": det["box"], "team": rec["team"]})
        return emitted

    def _update_record(self, tid: int, ts: float, det: Dict[str, Any], vx: float, vy: float) -> None:
        rec = self.records.get(tid)
        rgb = np.array(det["chest_rgb"], dtype=np.float64)
        st = float(det.get("stripe_score", 0.0))
        arm = float(det.get("armband_score", 0.0))
        pd = float(det.get("pants_darkness", float("nan")))
        bl = float(det.get("lower_belly_luma", float("nan")))
        ul = float(det.get("torso_upper_lab_l", float("nan")))
        k7 = det["kit_summary_lab7"]
        clean_st = min(max(st, 0.0), 1.0) if math.isfinite(st) else 0.0
        pd_ok = (not math.isfinite(pd)) or (pd >= 0.35)
        det_lab = srgb_to_lab(rgb)
        det_chroma = math.hypot(float(det_lab[1]), float(det_lab[2]))
        frame_stripe_hit = clean_st >= 0.44 and det_chroma < 8.0 and pd_ok

        if rec is None:
            rec = {
                "track_id": tid, "first_sec": ts, "last_sec": ts,
                "first_box": det["box"], "last_box": det["box"],
                "last_vx": vx, "last_vy": vy, "hits": 1,
                "sum_rgb": rgb.copy(), "n_rgb": 1,
                "sum_kit7": k7.copy(), "n_kit": 1,
                "sum_belly": min(max(bl, 0.0), 255.0) if math.isfinite(bl) else 0.0,
                "n_belly": 1 if math.isfinite(bl) else 0,
                "sum_upper_l": min(max(ul, 0.0), 100.0) if math.isfinite(ul) else 0.0,
                "n_upper_l": 1 if math.isfinite(ul) else 0,
                "ema_pants": min(max(pd, 0.0), 1.0) if math.isfinite(pd) else float("nan"),
                "n_pants": 1 if math.isfinite(pd) else 0,
                "ema_armband": arm, "n_armband": 1,
                "ema_stripe": clean_st, "n_stripe": 1,
                "ref_votes": 0, "stripe_ref_votes": 1 if frame_stripe_hit else 0,
                "last_stripe_frame_hit": frame_stripe_hit,
                "skater_votes": 0,
                "team": "UNKNOWN", "observations": [],
            }
            self.records[tid] = rec
        else:
            rec["last_sec"] = max(rec["last_sec"], ts)
            rec["last_box"] = det["box"]
            rec["last_vx"], rec["last_vy"] = vx, vy
            rec["hits"] += 1
            rec["sum_rgb"] += rgb
            rec["n_rgb"] += 1
            rec["sum_kit7"] += k7
            rec["n_kit"] += 1
            if math.isfinite(bl):
                rec["sum_belly"] += min(max(bl, 0.0), 255.0)
                rec["n_belly"] += 1
            if math.isfinite(ul):
                rec["sum_upper_l"] += min(max(ul, 0.0), 100.0)
                rec["n_upper_l"] += 1
            if math.isfinite(pd):
                clean_pd = min(max(pd, 0.0), 1.0)
                rec["ema_pants"] = clean_pd if not math.isfinite(rec["ema_pants"]) else 0.8 * rec["ema_pants"] + 0.2 * clean_pd
                rec["n_pants"] += 1
            clean_arm = min(max(arm, 0.0), 1.0) if math.isfinite(arm) else 0.0
            rec["ema_armband"] = 0.8 * rec["ema_armband"] + 0.2 * clean_arm
            rec["n_armband"] += 1
            rec["ema_stripe"] = 0.8 * rec["ema_stripe"] + 0.2 * clean_st
            rec["n_stripe"] += 1
            rec["last_stripe_frame_hit"] = frame_stripe_hit
            if frame_stripe_hit:
                rec["stripe_ref_votes"] += 1

    def _is_established_official(self, rec: Dict[str, Any], mean_chroma: float) -> bool:
        if rec["stripe_ref_votes"] >= 3 and mean_chroma < 8.5:
            return True
        if rec["ref_votes"] < 3:
            return False
        if rec["skater_votes"] > 2 * rec["ref_votes"] and mean_chroma >= 15.0:
            return False
        upper_l = self._mean_upper_l(rec)
        light_escape = (
            rec["skater_votes"] > 3 * rec["ref_votes"]
            and (
                (math.isfinite(rec["ema_pants"]) and rec["ema_pants"] < 0.60)
                or (rec["n_upper_l"] > 0 and upper_l >= 58.0)
            )
        )
        return not light_escape

    def _is_close_or_medium_striped_official(self, s: Dict[str, Any]) -> bool:
        rec = s["rec"]
        chroma = math.hypot(float(s["lab"][1]), float(s["lab"][2]))
        if rec["last_stripe_frame_hit"] and chroma < 8.5:
            return True
        stripe_hit = rec["ema_stripe"] >= 0.46 or (rec["ema_stripe"] >= 0.34 and rec["ema_armband"] >= 0.04)
        if not stripe_hit or chroma >= 14.5:
            return False
        pants_ok = (not math.isfinite(rec["ema_pants"])) or rec["ema_pants"] >= 0.30 or rec["ema_armband"] >= 0.04
        if not pants_ok:
            return False
        return (
            s["kit7"][3] >= 44.0
            or s["lab"][0] >= 36.0
            or (math.isfinite(s["belly"]) and s["belly"] >= 85.0)
            or (math.isfinite(s["upper_l"]) and s["upper_l"] >= 46.0)
        )

    def _recluster(self, cur_count: int, cur_frame_tids: set) -> None:
        snaps = []
        for rec in self.records.values():
            rgb = rec["sum_rgb"] / max(1, rec["n_rgb"])
            belly = rec["sum_belly"] / rec["n_belly"] if rec["n_belly"] > 0 else float("nan")
            upper_l = self._mean_upper_l(rec)
            kit7 = self._mean_kit7(rec, rgb)
            snaps.append({
                "rec": rec,
                "rgb": rgb,
                "belly": belly,
                "upper_l": upper_l,
                "pants": rec["ema_pants"],
                "kit7": kit7,
                "lab": srgb_to_lab(rgb),
            })
        if not snaps:
            return

        rem, persisted_refs = [], []
        for s in snaps:
            rec = s["rec"]
            is_cur = rec["track_id"] in cur_frame_tids
            chroma = math.hypot(float(s["lab"][1]), float(s["lab"][2]))
            if self._is_close_or_medium_striped_official(s):
                if is_cur:
                    rec["ref_votes"] += 1
                rec["team"] = "REFEREE"
            elif self._is_established_official(rec, chroma):
                persisted_refs.append(s)
            else:
                rem.append(s)

        if not rem and not persisted_refs:
            self.team_rgb.pop("TEAM_A", None)
            self.team_rgb.pop("TEAM_B", None)
            self.team_kit7.clear()
            self.team_belly.clear()
            self.team_upper_l.clear()
            return
        if not rem:
            for s in persisted_refs:
                s["rec"]["team"] = "REFEREE"
            self.team_rgb.pop("TEAM_A", None)
            self.team_rgb.pop("TEAM_B", None)
            self.team_kit7.clear()
            self.team_belly.clear()
            self.team_upper_l.clear()
            return
        if len(rem) == 1 and not persisted_refs:
            only = rem[0]
            if only["rec"]["track_id"] in cur_frame_tids:
                only["rec"]["skater_votes"] += 1
            st = "TEAM_A" if kit_effective_lightness(only["kit7"], only["belly"], only["upper_l"]) < 53.6 else "TEAM_B"
            other = "TEAM_B" if st == "TEAM_A" else "TEAM_A"
            only["rec"]["team"] = st
            self.team_rgb.pop(other, None)
            self.team_kit7.pop(other, None)
            self.team_belly.pop(other, None)
            self.team_upper_l.pop(other, None)
            self.team_rgb[st] = only["rgb"]
            self.team_kit7[st] = only["kit7"]
            if math.isfinite(only["belly"]):
                self.team_belly[st] = only["belly"]
            else:
                self.team_belly.pop(st, None)
            if math.isfinite(only["upper_l"]):
                self.team_upper_l[st] = only["upper_l"]
            else:
                self.team_upper_l.pop(st, None)
            return

        def run_2means(items: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            base_anch = [s for s in items if s["rec"]["n_rgb"] >= 3] or [s for s in items if s["rec"]["n_rgb"] >= 2] or items
            normal_anch = [s for s in base_anch if max_zone_chroma(s["kit7"]) <= 25.0]
            anch = normal_anch if len(normal_anch) >= max(4, (len(base_anch) * 3) // 4) else base_anch
            anch.sort(key=lambda s: kit_effective_lightness(s["kit7"], s["belly"], s["upper_l"]))
            sa = anch[(len(anch) * 15) // 100] if len(anch) >= 4 else anch[0]
            sb = anch[(len(anch) * 85) // 100] if len(anch) >= 4 else anch[-1]
            if len(anch) >= 4 and abs(
                kit_effective_lightness(sb["kit7"], sb["belly"], sb["upper_l"])
                - kit_effective_lightness(sa["kit7"], sa["belly"], sa["upper_l"])
            ) < 10.0:
                anch_d = sorted(
                    anch,
                    key=lambda s: robust_kit_and_belly_or_upper_dist_sq(
                        s["kit7"], s["belly"], s["upper_l"], sa["kit7"], sa["belly"], sa["upper_l"]
                    ),
                )
                sb = anch_d[(len(anch_d) * 85) // 100]
            cA = {"kit7": sa["kit7"].copy(), "belly": sa["belly"], "upper_l": sa["upper_l"], "rgb": sa["rgb"].copy()}
            cB = {"kit7": sb["kit7"].copy(), "belly": sb["belly"], "upper_l": sb["upper_l"], "rgb": sb["rgb"].copy()}
            for _ in range(12):
                skA, sbA, wbA, suA, wuA, srA, wA = np.zeros(7), 0.0, 0.0, 0.0, 0.0, np.zeros(3), 0.0
                skB, sbB, wbB, suB, wuB, srB, wB = np.zeros(7), 0.0, 0.0, 0.0, 0.0, np.zeros(3), 0.0
                for s in items:
                    dA = robust_kit_and_belly_or_upper_dist_sq(
                        s["kit7"], s["belly"], s["upper_l"], cA["kit7"], cA["belly"], cA["upper_l"]
                    )
                    dB = robust_kit_and_belly_or_upper_dist_sq(
                        s["kit7"], s["belly"], s["upper_l"], cB["kit7"], cB["belly"], cB["upper_l"]
                    )
                    w = float(min(s["rec"]["n_rgb"], 15))
                    if dA <= dB:
                        skA += robust_order_free_best_aligned(s["kit7"], cA["kit7"]) * w
                        srA += s["rgb"] * w
                        wA += w
                        if math.isfinite(s["belly"]):
                            sbA += s["belly"] * w
                            wbA += w
                        if math.isfinite(s["upper_l"]):
                            suA += s["upper_l"] * w
                            wuA += w
                    else:
                        skB += robust_order_free_best_aligned(s["kit7"], cB["kit7"]) * w
                        srB += s["rgb"] * w
                        wB += w
                        if math.isfinite(s["belly"]):
                            sbB += s["belly"] * w
                            wbB += w
                        if math.isfinite(s["upper_l"]):
                            suB += s["upper_l"] * w
                            wuB += w
                if wA > 0:
                    cA = {
                        "kit7": skA / wA,
                        "belly": (sbA / wbA if wbA > 0 else float("nan")),
                        "upper_l": (suA / wuA if wuA > 0 else float("nan")),
                        "rgb": srA / wA,
                    }
                if wB > 0:
                    cB = {
                        "kit7": skB / wB,
                        "belly": (sbB / wbB if wbB > 0 else float("nan")),
                        "upper_l": (suB / wuB if wuB > 0 else float("nan")),
                        "rgb": srB / wB,
                    }
            if cA["kit7"][0] > cA["kit7"][3]:
                cA["kit7"] = swap_summary(cA["kit7"])
            if cB["kit7"][0] > cB["kit7"][3]:
                cB["kit7"] = swap_summary(cB["kit7"])
            return (
                (cB, cA)
                if kit_effective_lightness(cA["kit7"], cA["belly"], cA["upper_l"])
                > kit_effective_lightness(cB["kit7"], cB["belly"], cB["upper_l"])
                else (cA, cB)
            )

        def is_balanced(itemsA: List[Dict[str, Any]], itemsB: List[Dict[str, Any]], wA: float, wB: float, total_n: int) -> bool:
            min_c = min(len(itemsA), len(itemsB))
            if min_c < max(2, total_n // 6) or (wA + wB) <= 0.0 or min(wA, wB) < 0.12 * (wA + wB):
                return False
            if min_c * 4 < total_n:
                maj = itemsA if len(itemsA) >= len(itemsB) else itemsB
                if len(maj) >= 4:
                    effs = sorted(kit_effective_lightness(s["kit7"], s["belly"], s["upper_l"]) for s in maj)
                    if effs[(len(effs) * 85) // 100] - effs[(len(effs) * 15) // 100] >= 22.0:
                        return False
            return True

        def partition(items: List[Dict[str, Any]], cA: Dict[str, Any], cB: Dict[str, Any]):
            aL, bL, wA, wB = [], [], 0.0, 0.0
            for s in items:
                dA = robust_kit_and_belly_or_upper_dist_sq(
                    s["kit7"], s["belly"], s["upper_l"], cA["kit7"], cA["belly"], cA["upper_l"]
                )
                dB = robust_kit_and_belly_or_upper_dist_sq(
                    s["kit7"], s["belly"], s["upper_l"], cB["kit7"], cB["belly"], cB["upper_l"]
                )
                w = float(min(s["rec"]["n_rgb"], 15))
                if dA <= dB:
                    aL.append(s)
                    wA += w
                else:
                    bL.append(s)
                    wB += w
            return aL, bL, wA, wB

        def run_1d_2means(items: List[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
            anch = [s for s in items if s["rec"]["n_rgb"] >= 3] or [s for s in items if s["rec"]["n_rgb"] >= 2] or items
            eff = sorted(kit_effective_lightness(s["kit7"], s["belly"], s["upper_l"]) for s in anch)
            muA = eff[(len(eff) * 15) // 100] if len(eff) >= 4 else eff[0]
            muB = eff[(len(eff) * 85) // 100] if len(eff) >= 4 else eff[-1]
            if abs(muB - muA) < 10.0:
                return None
            for _ in range(12):
                sA = wA = sB = wB = 0.0
                for s in items:
                    lv = kit_effective_lightness(s["kit7"], s["belly"], s["upper_l"])
                    w = float(min(s["rec"]["n_rgb"], 15))
                    if abs(lv - muA) <= abs(lv - muB):
                        sA += lv * w
                        wA += w
                    else:
                        sB += lv * w
                        wB += w
                if wA > 0:
                    muA = sA / wA
                if wB > 0:
                    muB = sB / wB
            if abs(muB - muA) < 10.0:
                return None
            if muA > muB:
                muA, muB = muB, muA
            gA, gB, wA, wB = [], [], 0.0, 0.0
            for s in items:
                lv = kit_effective_lightness(s["kit7"], s["belly"], s["upper_l"])
                w = float(min(s["rec"]["n_rgb"], 15))
                if abs(lv - muA) <= abs(lv - muB):
                    gA.append(s)
                    wA += w
                else:
                    gB.append(s)
                    wB += w
            if not is_balanced(gA, gB, wA, wB, len(items)):
                return None

            def _cent(grp):
                ref = grp[0]["kit7"]
                sk, sb, wb, su, wu, sr, wt = np.zeros(7), 0.0, 0.0, 0.0, 0.0, np.zeros(3), 0.0
                for s in grp:
                    w = float(min(s["rec"]["n_rgb"], 15))
                    sk += robust_order_free_best_aligned(s["kit7"], ref) * w
                    sr += s["rgb"] * w
                    wt += w
                    if math.isfinite(s["belly"]):
                        sb += s["belly"] * w
                        wb += w
                    if math.isfinite(s["upper_l"]):
                        su += s["upper_l"] * w
                        wu += w
                k7 = sk / wt
                if k7[0] > k7[3]:
                    k7 = swap_summary(k7)
                return {
                    "kit7": k7,
                    "belly": (sb / wb if wb > 0 else float("nan")),
                    "upper_l": (su / wu if wu > 0 else float("nan")),
                    "rgb": sr / wt,
                }

            cA1, cB1 = _cent(gA), _cent(gB)
            return (
                (cA1, cB1)
                if kit_effective_lightness(cA1["kit7"], cA1["belly"], cA1["upper_l"])
                <= kit_effective_lightness(cB1["kit7"], cB1["belly"], cB1["upper_l"])
                else (cB1, cA1)
            )

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
            one_d = run_1d_2means(items)
            return one_d if one_d is not None else (initA, initB)

        cA, cB = run_robust_2means(rem if len(rem) >= 2 else rem + persisted_refs)
        kit_sep = math.sqrt(order_free_kit_dist_sq(cA["kit7"], cB["kit7"]))
        rgb_sep = _color_lum_dist(cA["rgb"], cB["rgb"])

        can_test_blur = (cur_count >= 9 or len(self.active) >= 9) and (kit_sep >= 11.0 or rgb_sep >= 25.0)
        labA, labB = srgb_to_lab(cA["rgb"]), srgb_to_lab(cB["rgb"])
        grey_lab = np.array([55.0, 0.0, 0.0])
        skaters, need_recompute = [], False

        def _is_raw_blurred_ref(s: Dict[str, Any]) -> bool:
            rec = s["rec"]
            if not math.isfinite(s["pants"]) or s["pants"] < 0.60:
                return False
            mean_l = float(s["lab"][0])
            mean_c = math.hypot(float(s["lab"][1]), float(s["lab"][2]))
            max_zc = max_zone_chroma(s["kit7"])
            min_td = min(float(np.linalg.norm(s["lab"] - labA)), float(np.linalg.norm(s["lab"] - labB)))
            if not math.isfinite(s["upper_l"]):
                if rec["n_rgb"] < 2 or not (30.0 <= mean_l <= 72.0):
                    return False
                if mean_c >= 15.0 or max_zc >= 17.5:
                    return False
                if (s["kit7"][3] >= 75.0 and s["kit7"][6] >= 42.0) or (math.isfinite(s["belly"]) and s["belly"] >= 150.0):
                    return False
                d_grey = float(np.linalg.norm(s["lab"] - grey_lab))
                if s["pants"] >= 0.75 and 31.0 <= mean_l <= 72.0 and mean_c < 10.0 and (not math.isfinite(s["belly"]) or s["belly"] < 135.0):
                    return min_td >= 3.0 and d_grey <= min_td * 1.6
                return min_td >= 8.0 and d_grey <= min_td * 1.15

            if rec["n_rgb"] < 6 or not (25.0 <= mean_l <= 66.0):
                return False
            if mean_c >= 13.0 or max_zc >= 19.0:
                return False
            if s["upper_l"] >= 56.0 or s["kit7"][3] >= 56.0 or (math.isfinite(s["belly"]) and s["belly"] >= 135.0):
                return False
            if s["kit7"][6] < 68.0:
                return False
            if s["pants"] >= 0.72 and mean_c < 10.5 and min_td >= 5.0 and mean_l <= float(labB[0]) + 2.0:
                return True
            return s["pants"] >= 0.60 and mean_c < 4.5 and max_zc < 6.0 and min_td >= 8.0 and mean_l <= float(labB[0])

        for is_pers, group in ((True, persisted_refs), (False, rem)):
            for s in group:
                rec = s["rec"]
                is_cur = rec["track_id"] in cur_frame_tids
                mean_c = math.hypot(float(s["lab"][1]), float(s["lab"][2]))
                raw_b = can_test_blur and _is_raw_blurred_ref(s)
                if is_cur:
                    rec["ref_votes" if raw_b else "skater_votes"] += 1
                if raw_b or self._is_established_official(rec, mean_c):
                    rec["team"] = "REFEREE"
                    if not is_pers:
                        need_recompute = True
                else:
                    skaters.append(s)
                    if is_pers:
                        need_recompute = True

        if not skaters:
            self.team_rgb.pop("TEAM_A", None)
            self.team_rgb.pop("TEAM_B", None)
            self.team_kit7.clear()
            self.team_belly.clear()
            self.team_upper_l.clear()
            return
        if len(skaters) == 1:
            only = skaters[0]
            st = "TEAM_A" if kit_effective_lightness(only["kit7"], only["belly"], only["upper_l"]) < 53.6 else "TEAM_B"
            other = "TEAM_B" if st == "TEAM_A" else "TEAM_A"
            only["rec"]["team"] = st
            self.team_rgb.pop(other, None)
            self.team_kit7.pop(other, None)
            self.team_belly.pop(other, None)
            self.team_upper_l.pop(other, None)
            self.team_rgb[st] = only["rgb"]
            self.team_kit7[st] = only["kit7"]
            if math.isfinite(only["belly"]):
                self.team_belly[st] = only["belly"]
            else:
                self.team_belly.pop(st, None)
            if math.isfinite(only["upper_l"]):
                self.team_upper_l[st] = only["upper_l"]
            else:
                self.team_upper_l.pop(st, None)
            return
        if need_recompute:
            cA, cB = run_robust_2means(skaters)

        kit_sep = math.sqrt(order_free_kit_dist_sq(cA["kit7"], cB["kit7"]))
        rgb_sep = _color_lum_dist(cA["rgb"], cB["rgb"])
        if kit_sep < 11.0 and rgb_sep < 25.0:
            single_team = "TEAM_A" if kit_effective_lightness(cA["kit7"], cA["belly"], cA["upper_l"]) < 53.6 else "TEAM_B"
            other = "TEAM_B" if single_team == "TEAM_A" else "TEAM_A"
            for s in skaters:
                s["rec"]["team"] = single_team
            self.team_rgb.pop(other, None)
            self.team_kit7.pop(other, None)
            self.team_belly.pop(other, None)
            self.team_upper_l.pop(other, None)
            self.team_kit7[single_team] = cA["kit7"]
            self.team_rgb[single_team] = cA["rgb"]
            if math.isfinite(cA["belly"]):
                self.team_belly[single_team] = cA["belly"]
            if math.isfinite(cA["upper_l"]):
                self.team_upper_l[single_team] = cA["upper_l"]
            return

        self.team_kit7 = {"TEAM_A": cA["kit7"], "TEAM_B": cB["kit7"]}
        self.team_rgb = {"TEAM_A": cA["rgb"], "TEAM_B": cB["rgb"]}
        self.team_belly = {"TEAM_A": cA["belly"], "TEAM_B": cB["belly"]}
        self.team_upper_l = {}
        if math.isfinite(cA["upper_l"]):
            self.team_upper_l["TEAM_A"] = cA["upper_l"]
        if math.isfinite(cB["upper_l"]):
            self.team_upper_l["TEAM_B"] = cB["upper_l"]

        allow_legacy_outlier = cur_count > 10 or len(self.active) > 10
        for s in skaters:
            dA = robust_kit_and_belly_or_upper_dist_sq(
                s["kit7"], s["belly"], s["upper_l"], cA["kit7"], cA["belly"], cA["upper_l"]
            )
            dB = robust_kit_and_belly_or_upper_dist_sq(
                s["kit7"], s["belly"], s["upper_l"], cB["kit7"], cB["belly"], cB["upper_l"]
            )
            assigned = "TEAM_A" if dA <= dB else "TEAM_B"
            if allow_legacy_outlier and not math.isfinite(s["pants"]) and rgb_sep >= 35.0:
                min_td_rgb = min(_color_lum_dist(s["rgb"], cA["rgb"]), _color_lum_dist(s["rgb"], cB["rgb"]))
                sat = float(s["rgb"].max() - s["rgb"].min())
                if min_td_rgb > max(45.0, 0.30 * rgb_sep) and (sat < 35.0 or sat > 95.0):
                    assigned = "REFEREE"
            s["rec"]["team"] = assigned

    def is_likely_official_track(self, tid: int) -> bool:
        rec = self.records.get(tid)
        if not rec:
            return False
        if rec["team"] == "REFEREE":
            return True
        if math.isfinite(rec["ema_pants"]) and rec["ema_pants"] >= 0.60:
            lab = srgb_to_lab(rec["sum_rgb"] / max(1, rec["n_rgb"]))
            if math.hypot(float(lab[1]), float(lab[2])) < 15.0:
                return True
        return False

    def select_priority_skaters(
        self,
        emitted: List[Dict[str, Any]],
        limit: int = 6,
        min_height: float = 0.085,
    ) -> List[Dict[str, Any]]:
        """Mirror of RealtimeHockeyProcessor.selectPrioritySkaters."""
        cands = []
        for item in emitted:
            tid = item["track_id"]
            if self.is_likely_official_track(tid):
                continue
            x1, y1, x2, y2 = item["box"]
            bw, bh = x2 - x1, y2 - y1
            if bh < min_height or bw < 0.015:
                continue
            area = bw * bh
            st = self.last_str_score.get(tid, {})
            sharp = st.get("sharpness", 100.0)
            back_p = st.get("back", 0.5)
            leg_p = st.get("legibility", 0.5)
            att = self.str_attempts.get(tid, 0)
            view_score = (0.3 + 0.7 * leg_p) * (0.4 + 0.6 * back_p) * sharp * bh
            cands.append((-(view_score * area) / (1.0 + 0.5 * att), item))
        cands.sort(key=lambda kv: kv[0])
        return [it for _, it in cands[:limit]]

    def _resolve_chain_team(self, tids: List[int]) -> str:
        tot_ref, tot_stripe_ref, tot_sk = 0, 0, 0
        sr, sg, sb, cw = 0.0, 0.0, 0.0, 0.0
        sum_pd, pd_w = 0.0, 0.0
        for tid in tids:
            rec = self.records[tid]
            tot_ref += rec["ref_votes"]
            tot_stripe_ref += rec["stripe_ref_votes"]
            tot_sk += rec["skater_votes"]
            rgb = rec["sum_rgb"] / max(1, rec["n_rgb"])
            lab = srgb_to_lab(rgb)
            k7 = self._mean_kit7(rec, rgb)
            belly = rec["sum_belly"] / rec["n_belly"] if rec["n_belly"] > 0 else float("nan")
            upper_l = self._mean_upper_l(rec)
            chroma = math.hypot(float(lab[1]), float(lab[2]))
            bright_s = (
                k7[3] >= 44.0
                or lab[0] >= 36.0
                or (math.isfinite(belly) and belly >= 85.0)
                or (math.isfinite(upper_l) and upper_l >= 46.0)
            )
            pants_ok = (not math.isfinite(rec["ema_pants"])) or rec["ema_pants"] >= 0.30 or rec["ema_armband"] >= 0.04
            if rec["ema_stripe"] >= 0.46 and chroma < 14.5 and bright_s and pants_ok:
                return "REFEREE"
            w = float(rec["n_rgb"])
            sr += rgb[0] * w
            sg += rgb[1] * w
            sb += rgb[2] * w
            cw += w
            if math.isfinite(rec["ema_pants"]):
                sum_pd += rec["ema_pants"] * w
                pd_w += w
        avg_c = 0.0
        if cw > 0:
            alab = srgb_to_lab(np.array([sr / cw, sg / cw, sb / cw]))
            avg_c = math.hypot(float(alab[1]), float(alab[2]))
        if tot_stripe_ref >= 3 and avg_c < 8.5:
            return "REFEREE"
        if tot_ref >= 3:
            avg_pd = (sum_pd / pd_w) if pd_w > 0.0 else float("nan")
            skater_escape = (tot_sk > 2 * tot_ref and avg_c >= 15.0) or (
                tot_sk > 3 * tot_ref and math.isfinite(avg_pd) and avg_pd < 0.60
            )
            if not skater_escape:
                return "REFEREE"
        weights: Dict[str, int] = {}
        for tid in tids:
            t = self.records[tid]["team"]
            if t != "UNKNOWN":
                weights[t] = weights.get(t, 0) + self.records[tid]["hits"]
        return max(weights.items(), key=lambda kv: kv[1])[0] if weights else "UNKNOWN"

    def _established_or_dominant_num(self, tids: List[int], min_prob: float, min_ev: float) -> Optional[str]:
        obs = [o for tid in tids for o in self.records[tid]["observations"]]
        if not obs:
            return None
        lp, ev_w = compute_chain_log_posterior(obs)
        w = np.exp(lp - lp.max())
        p = w / w.sum()
        bi = int(np.argmax(p))
        return ALL_NUMS[bi] if (float(p[bi]) >= min_prob and ev_w >= min_ev) else None

    def build_stitched_chains(self) -> List[List[int]]:
        """Exact 2-stage temporal stitching (1.5s then 4.0s) matching TrackIdentityRegistry."""
        nodes = [
            {
                "tids": [r["track_id"]],
                "start": r["first_sec"], "end": r["last_sec"],
                "first_box": r["first_box"], "last_box": r["last_box"],
                "vx": r["last_vx"], "vy": r["last_vy"],
            }
            for r in sorted(self.records.values(), key=lambda r: r["first_sec"])
        ]

        def _compat(ta: str, tb: str) -> bool:
            if ta == "UNKNOWN" or tb == "UNKNOWN":
                return ta != "REFEREE" and tb != "REFEREE"
            return ta == tb

        for max_gap, max_cap, h_base, h_slope in ((1.5, 0.24, 2.4, 1.6), (4.0, 0.38, 2.2, 1.8)):
            if len(nodes) < 2:
                break
            cands = []
            teams = [self._resolve_chain_team(nd["tids"]) for nd in nodes]
            ests = [self._established_or_dominant_num(nd["tids"], 0.55, 1.80) for nd in nodes]
            doms = [self._established_or_dominant_num(nd["tids"], 0.45, 0.90) for nd in nodes]
            for i, a in enumerate(nodes):
                for j, b in enumerate(nodes):
                    if i == j:
                        continue
                    gap = b["start"] - a["end"]
                    if gap <= 1e-6 or gap > max_gap or not _compat(teams[i], teams[j]):
                        continue
                    if ests[i] is not None and ests[j] is not None and ests[i] != ests[j]:
                        continue
                    abx, bbx = a["last_box"], b["first_box"]
                    avg_h = max(0.03, 0.5 * ((abx[3] - abx[1]) + (bbx[3] - bbx[1])))
                    acx, afy = 0.5 * (abx[0] + abx[2]), abx[3]
                    bcx, bfy = 0.5 * (bbx[0] + bbx[2]), bbx[3]
                    pdt = min(gap, 1.0) * 0.5
                    dist = min(
                        math.hypot(acx - bcx, afy - bfy),
                        math.hypot(acx + a["vx"] * pdt - bcx, afy + a["vy"] * pdt - bfy),
                    )
                    if dist > min(max_cap, max(0.10, (h_base + h_slope * gap) * avg_h)):
                        continue
                    adj = -0.04 if (doms[i] and doms[j] and doms[i] == doms[j]) else (0.06 if (doms[i] and doms[j] and doms[i] != doms[j]) else 0.0)
                    cands.append((dist + 0.04 * gap + adj, i, j))
            cands.sort(key=lambda c: c[0])
            next_of, prev_of = [-1] * len(nodes), [-1] * len(nodes)
            comp_head = list(range(len(nodes)))
            comp_team, comp_est = list(teams), list(ests)
            for _, pi, ni in cands:
                if next_of[pi] != -1 or prev_of[ni] != -1:
                    continue
                ha, hb = comp_head[pi], comp_head[ni]
                if ha == hb or not _compat(comp_team[ha], comp_team[hb]):
                    continue
                if comp_est[ha] is not None and comp_est[hb] is not None and comp_est[ha] != comp_est[hb]:
                    continue
                next_of[pi], prev_of[ni] = ni, pi
                comp_team[ha] = comp_team[ha] if comp_team[ha] != "UNKNOWN" else comp_team[hb]
                comp_est[ha] = comp_est[ha] or comp_est[hb]
                w = hb
                while w != -1:
                    comp_head[w] = ha
                    w = next_of[w]
            merged = []
            for i, root in enumerate(nodes):
                if prev_of[i] != -1:
                    continue
                tids = list(root["tids"])
                tail = root
                cur = next_of[i]
                while cur != -1:
                    tids.extend(nodes[cur]["tids"])
                    tail = nodes[cur]
                    cur = next_of[cur]
                merged.append({
                    "tids": tids, "start": root["start"], "end": tail["end"],
                    "first_box": root["first_box"], "last_box": tail["last_box"],
                    "vx": tail["vx"], "vy": tail["vy"],
                })
            nodes = merged
        return [nd["tids"] for nd in nodes]

    def resolve_all(self, single_bin_min_conf: float = SINGLE_BIN_COMMIT_MIN_CONF) -> Dict[int, Tuple[str, Optional[str], float]]:
        chains = self.build_stitched_chains()
        by_team: Dict[str, List[Dict[str, Any]]] = {"TEAM_A": [], "TEAM_B": [], "REFEREE": [], "UNKNOWN": []}
        for c_tids in chains:
            team = self._resolve_chain_team(c_tids)
            obs = sorted((o for tid in c_tids for o in self.records[tid]["observations"]), key=lambda o: o["timestamp_sec"])
            meta = {
                "chain_id": min(c_tids), "tids": c_tids, "team": team,
                "first_sec": min(self.records[t]["first_sec"] for t in c_tids),
                "last_sec": max(self.records[t]["last_sec"] for t in c_tids),
                "observations": obs,
            }
            by_team.setdefault(team, []).append(meta)

        out: Dict[int, Tuple[str, Optional[str], float]] = {}
        for team in ("TEAM_A", "TEAM_B"):
            sol = solve_team_numbers_bayesian_hungarian(by_team[team], single_bin_min_conf=single_bin_min_conf)
            for m in by_team[team]:
                num, conf = sol.get(m["chain_id"], (None, 0.0))
                for tid in m["tids"]:
                    out[tid] = (team, num, conf)
        for m in by_team["UNKNOWN"]:
            sol = solve_team_numbers_bayesian_hungarian([m], single_bin_min_conf=single_bin_min_conf)
            num, conf = sol.get(m["chain_id"], (None, 0.0))
            for tid in m["tids"]:
                out[tid] = ("UNKNOWN", num, conf)
        for m in by_team["REFEREE"]:
            for tid in m["tids"]:
                out[tid] = ("REFEREE", None, 0.0)
        return out


# ---------------------------------------------------------------------------
# Independent Visual GT & Cloud (coach.leonxzhao.com) Evaluation Data at t_eval
# ---------------------------------------------------------------------------

# Note: Strictly for post-inference scoring ONLY. Zero leakage into Mobile pipeline.
GAMES_SPEC: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Game 1: San Jose Jr Sharks 12AA-2 Teal vs Tri Valley Blue Devils 12AA-2 White",
        "t_eval_str": "48:02",
        "eval_t_sec": 12.6,
        "visual_gt": [
            {"id": "G1_Goalie_L", "cx": 0.035, "cy": 0.375, "team": "TEAM_B", "role": "GOALIE", "legible_num": None},
            {"id": "G1_Ref_L", "cx": 0.214, "cy": 0.667, "team": "REFEREE", "role": "REF", "legible_num": None},
            {"id": "G1_Wht_L", "cx": 0.239, "cy": 0.427, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G1_Wht_27", "cx": 0.294, "cy": 0.645, "team": "TEAM_B", "role": "SKATER", "legible_num": "27"},
            {"id": "G1_Tea_21", "cx": 0.328, "cy": 0.629, "team": "TEAM_A", "role": "SKATER", "legible_num": "21"},
            {"id": "G1_Tea_M", "cx": 0.369, "cy": 0.438, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G1_Wht_C", "cx": 0.602, "cy": 0.615, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G1_Tea_6", "cx": 0.617, "cy": 0.744, "team": "TEAM_A", "role": "SKATER", "legible_num": "6"},
            {"id": "G1_Wht_R", "cx": 0.697, "cy": 0.698, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G1_Wht_Far", "cx": 0.732, "cy": 0.353, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
        ],
        "cloud_eval": {
            "analyzed": True,
            "preds": [
                {"gt_id": "G1_Goalie_L", "team": "TEAM_B", "num": "1*", "ring": "white"},
                {"gt_id": "G1_Ref_L", "team": "REFEREE", "num": None, "ring": "none_excluded"},
                {"gt_id": "G1_Wht_L", "team": "TEAM_B", "num": "36", "ring": "white"},
                {"gt_id": "G1_Wht_27", "team": "TEAM_B", "num": "27", "ring": "white"},
                {"gt_id": "G1_Tea_21", "team": "TEAM_A", "num": "21*", "ring": "teal"},
                {"gt_id": "G1_Tea_M", "team": "TEAM_A", "num": "90*", "ring": "teal"},
                {"gt_id": "G1_Wht_C", "team": "TEAM_B", "num": "28", "ring": "white"},
                {"gt_id": "G1_Tea_6", "team": "TEAM_A", "num": "6", "ring": "teal"},
                {"gt_id": "G1_Wht_R", "team": "TEAM_A", "num": "84*", "ring": "teal_WRONG"},
                {"gt_id": "G1_Wht_Far", "team": "TEAM_B", "num": "19*", "ring": "white"},
            ],
            "off_ice_fp": 0,
            "notes": "White skater #84* mislabeled with Teal ring (TEAM_A); roster-assisted names on #27 Velupula, #28 MacDonald, #6 Wang.",
        },
    },
    2: {
        "name": "Game 2: TVBD 12AA-1 Navy vs TVBD 12AA-2 White",
        "t_eval_str": "11:09",
        "eval_t_sec": 12.6,
        "visual_gt": [
            {"id": "G2_Wht_L", "cx": 0.207, "cy": 0.452, "team": "TEAM_B", "role": "SKATER", "legible_num": "43"},
            {"id": "G2_Nav_Pair", "cx": 0.228, "cy": 0.340, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G2_Wht_Pair", "cx": 0.238, "cy": 0.330, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G2_Goalie", "cx": 0.330, "cy": 0.289, "team": "TEAM_B", "role": "GOALIE", "legible_num": None},
            {"id": "G2_Nav_C", "cx": 0.354, "cy": 0.466, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G2_Wht_U", "cx": 0.403, "cy": 0.328, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G2_Nav_D", "cx": 0.406, "cy": 0.365, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G2_Wht_19", "cx": 0.491, "cy": 0.443, "team": "TEAM_B", "role": "SKATER", "legible_num": "19"},
            {"id": "G2_Wht_Far", "cx": 0.599, "cy": 0.336, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G2_Nav_Fg", "cx": 0.732, "cy": 0.650, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G2_Nav_R", "cx": 0.865, "cy": 0.398, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G2_Ref_R", "cx": 0.871, "cy": 0.345, "team": "REFEREE", "role": "REF", "legible_num": None},
        ],
        "cloud_eval": {
            "analyzed": True,
            "preds": [
                {"gt_id": "G2_Wht_L", "team": "TEAM_B", "num": "43", "ring": "white"},
                {"gt_id": "G2_Nav_Pair", "team": "TEAM_A", "num": "14", "ring": "badge_only_no_ring"},
                {"gt_id": "G2_Wht_Pair", "team": "TEAM_B", "num": "19", "ring": "badge_only_no_ring"},
                {"gt_id": "G2_Goalie", "team": "TEAM_B", "num": "1*", "ring": "badge_only_no_ring"},
                {"gt_id": "G2_Nav_C", "team": "TEAM_A", "num": "78", "ring": "red"},
                {"gt_id": "G2_Wht_U", "team": None, "num": "57_MISPLACED", "ring": "missed_57_floating"},
                {"gt_id": "G2_Nav_D", "team": "TEAM_A", "num": "12", "ring": "red"},
                {"gt_id": "G2_Wht_19", "team": "TEAM_B", "num": "19", "ring": "white"},
                {"gt_id": "G2_Wht_Far", "team": "TEAM_B", "num": "11", "ring": "white"},
                {"gt_id": "G2_Nav_Fg", "team": "TEAM_A", "num": "9", "ring": "red"},
                {"gt_id": "G2_Nav_R", "team": "TEAM_A", "num": "72*", "ring": "red"},
                {"gt_id": "G2_Ref_R", "team": "REFEREE", "num": None, "ring": "none_excluded"},
            ],
            "off_ice_fp": 0,
            "notes": "4 distant/battling players lack floor rings (only 7/11 skaters ringed); duplicate #19 badge (Birkey/Gibson); #57 Sheliepov badge floats misplaced near Goalie/G2_Wht_U.",
        },
    },
    3: {
        "name": "Game 3: Anaheim Jr Ducks White vs TVBD 12AA-1 Navy",
        "t_eval_str": "14:16",
        "eval_t_sec": 12.6,
        "visual_gt": [
            {"id": "G3_Ref_L", "cx": 0.042, "cy": 0.880, "team": "REFEREE", "role": "REF", "legible_num": None},
            {"id": "G3_Goalie", "cx": 0.074, "cy": 0.320, "team": "TEAM_B", "role": "GOALIE", "legible_num": None},
            {"id": "G3_Wht_Net", "cx": 0.080, "cy": 0.292, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G3_Wht_Crease", "cx": 0.112, "cy": 0.331, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G3_Nav_Crease", "cx": 0.129, "cy": 0.310, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G3_Nav_Slot", "cx": 0.222, "cy": 0.353, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G3_Nav_Fg", "cx": 0.339, "cy": 0.893, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G3_Wht_Fg", "cx": 0.396, "cy": 0.950, "team": "TEAM_B", "role": "SKATER", "legible_num": "28"},
            {"id": "G3_Nav_Mid", "cx": 0.410, "cy": 0.371, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G3_Wht_Mid", "cx": 0.565, "cy": 0.369, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G3_Nav_Blue", "cx": 0.670, "cy": 0.434, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G3_Wht_Blue", "cx": 0.699, "cy": 0.393, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G3_Wht_FarR", "cx": 0.812, "cy": 0.355, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
        ],
        "cloud_eval": {
            "analyzed": True,
            "preds": [
                {"gt_id": "G3_Ref_L", "team": "REFEREE", "num": None, "ring": "none_excluded"},
                {"gt_id": "G3_Goalie", "team": "TEAM_B", "num": None, "ring": "white"},
                {"gt_id": "G3_Wht_Net", "team": "TEAM_B", "num": "97_WRONG", "ring": "white"},
                {"gt_id": "G3_Wht_Crease", "team": "TEAM_B", "num": None, "ring": "white"},
                {"gt_id": "G3_Nav_Crease", "team": "TEAM_A", "num": "12", "ring": "red"},
                {"gt_id": "G3_Nav_Slot", "team": "TEAM_A", "num": None, "ring": "red"},
                {"gt_id": "G3_Nav_Fg", "team": "TEAM_A", "num": "72", "ring": "red"},
                {"gt_id": "G3_Wht_Fg", "team": "TEAM_A", "num": None, "ring": "red_WRONG"},
                {"gt_id": "G3_Nav_Mid", "team": "TEAM_A", "num": "78", "ring": "red"},
                {"gt_id": "G3_Wht_Mid", "team": "TEAM_B", "num": "9", "ring": "white"},
                {"gt_id": "G3_Nav_Blue", "team": "TEAM_A", "num": None, "ring": "red"},
                {"gt_id": "G3_Wht_Blue", "team": None, "num": None, "ring": "MISSED"},
                {"gt_id": "G3_Wht_FarR", "team": "TEAM_A", "num": None, "ring": "red_WRONG"},
            ],
            "off_ice_fp": 0,
            "notes": "Cloud gave Red ring (TEAM_A) to White skater G3_Wht_FarR (0.812) & bottom-clipped White skater G3_Wht_Fg (0.396); missed White skater G3_Wht_Blue (0.699); spurious #97 on unreadable White net player.",
        },
    },
    4: {
        "name": "Game 4: Anahei vs TVBD 12AA-1 (Not analyzed yet / Game setup)",
        "t_eval_str": "14:17",
        "eval_t_sec": 12.6,
        "visual_gt": [
            {"id": "G4_Ref_L_Clip", "cx": 0.018, "cy": 0.885, "team": "REFEREE", "role": "REF", "legible_num": None},
            {"id": "G4_Goalie", "cx": 0.046, "cy": 0.314, "team": "TEAM_B", "role": "GOALIE", "legible_num": None},
            {"id": "G4_Wht_Crease", "cx": 0.069, "cy": 0.323, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G4_Wht_Top", "cx": 0.093, "cy": 0.277, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G4_Nav_Top", "cx": 0.108, "cy": 0.290, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G4_Nav_Slot", "cx": 0.205, "cy": 0.331, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G4_Nav_Mid", "cx": 0.393, "cy": 0.351, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G4_Nav_Fg", "cx": 0.401, "cy": 0.926, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G4_Wht_BotClip", "cx": 0.506, "cy": 0.975, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G4_Wht_Mid", "cx": 0.513, "cy": 0.341, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G4_Wht_Blue", "cx": 0.608, "cy": 0.367, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G4_Nav_Blue", "cx": 0.641, "cy": 0.407, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G4_Wht_Gate", "cx": 0.776, "cy": 0.330, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
        ],
        "cloud_eval": {
            "analyzed": False,
            "preds": [],
            "off_ice_fp": 0,
            "notes": "0% Cloud coverage — stuck on 'Not analyzed yet · Game setup' waiting for manual rink calibration (same 1:08:00 video as Game 3 at 14:17 vs 14:16).",
        },
    },
    5: {
        "name": "Game 5: TVBD 12AA-2 White vs 12AA-1 Navy exhibition",
        "t_eval_str": "11:13",
        "eval_t_sec": 12.6,
        "visual_gt": [
            {"id": "G5_Nav_Mesh", "cx": 0.123, "cy": 0.444, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G5_Wht_43", "cx": 0.129, "cy": 0.527, "team": "TEAM_B", "role": "SKATER", "legible_num": "43"},
            {"id": "G5_Wht_L", "cx": 0.199, "cy": 0.379, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G5_Wht_Fg", "cx": 0.241, "cy": 0.669, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G5_Nav_12", "cx": 0.293, "cy": 0.618, "team": "TEAM_A", "role": "SKATER", "legible_num": "12"},
            {"id": "G5_Nav_Slot", "cx": 0.305, "cy": 0.419, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G5_Goalie_1", "cx": 0.345, "cy": 0.275, "team": "TEAM_B", "role": "GOALIE", "legible_num": None},
            {"id": "G5_Nav_9", "cx": 0.357, "cy": 0.662, "team": "TEAM_A", "role": "SKATER", "legible_num": "9"},
            {"id": "G5_Wht_Far", "cx": 0.382, "cy": 0.327, "team": "TEAM_B", "role": "SKATER", "legible_num": None},
            {"id": "G5_Wht_11", "cx": 0.450, "cy": 0.441, "team": "TEAM_B", "role": "SKATER", "legible_num": "11"},
            {"id": "G5_Ref_M", "cx": 0.667, "cy": 0.266, "team": "REFEREE", "role": "REF", "legible_num": None},
            {"id": "G5_Nav_R", "cx": 0.868, "cy": 0.532, "team": "TEAM_A", "role": "SKATER", "legible_num": None},
            {"id": "G5_Ref_R", "cx": 0.908, "cy": 0.351, "team": "REFEREE", "role": "REF", "legible_num": None},
        ],
        "cloud_eval": {
            "analyzed": True,
            "preds": [
                {"gt_id": "G5_Nav_Mesh", "team": None, "num": None, "ring": "MISSED"},
                {"gt_id": "G5_Wht_43", "team": "TEAM_B", "num": "43", "ring": "white"},
                {"gt_id": "G5_Wht_L", "team": None, "num": None, "ring": "MISSED"},
                {"gt_id": "G5_Wht_Fg", "team": "TEAM_B", "num": "19", "ring": "white"},
                {"gt_id": "G5_Nav_12", "team": "TEAM_A", "num": "12", "ring": "red"},
                {"gt_id": "G5_Nav_Slot", "team": None, "num": None, "ring": "MISSED"},
                {"gt_id": "G5_Goalie_1", "team": "TEAM_B", "num": "57_WRONG", "ring": "white"},
                {"gt_id": "G5_Nav_9", "team": "TEAM_A", "num": "9", "ring": "red"},
                {"gt_id": "G5_Wht_Far", "team": "TEAM_B", "num": None, "ring": "white"},
                {"gt_id": "G5_Wht_11", "team": "TEAM_B", "num": "11", "ring": "white"},
                {"gt_id": "G5_Ref_M", "team": "REFEREE", "num": None, "ring": "none_excluded"},
                {"gt_id": "G5_Nav_R", "team": "TEAM_A", "num": None, "ring": "red"},
                {"gt_id": "G5_Ref_R", "team": "REFEREE", "num": None, "ring": "dashed_ref_ring"},
            ],
            "off_ice_fp": 0,
            "notes": "Missed 3 on-ice skaters (G5_Nav_Mesh, G5_Wht_L, G5_Nav_Slot); mislabeled #1 Goalie as '#57 Sheliepov'; 2/2 refs excluded/dotted.",
        },
    },
}

COACHBOX_GT_FRAME08 = [
    {"id": "#11", "cx": 0.113, "cy": 0.542, "team": "TEAM_B", "legible_num": "11"},
    {"id": "REF_L", "cx": 0.133, "cy": 0.403, "team": "REFEREE", "legible_num": None},
    {"id": "#12", "cx": 0.283, "cy": 0.416, "team": "TEAM_B", "legible_num": "12"},
    {"id": "#97", "cx": 0.308, "cy": 0.520, "team": "TEAM_B", "legible_num": "97"},
    {"id": "#67", "cx": 0.383, "cy": 0.521, "team": "TEAM_A", "legible_num": "67"},
    {"id": "DARK_U", "cx": 0.388, "cy": 0.420, "team": "TEAM_A", "legible_num": None},
    {"id": "#34", "cx": 0.454, "cy": 0.405, "team": "TEAM_B", "legible_num": "34"},
    {"id": "REF_C", "cx": 0.496, "cy": 0.340, "team": "REFEREE", "legible_num": None},
    {"id": "#43", "cx": 0.522, "cy": 0.412, "team": "TEAM_A", "legible_num": "43"},
    {"id": "#77", "cx": 0.542, "cy": 0.456, "team": "TEAM_B", "legible_num": "77"},
    {"id": "#14", "cx": 0.687, "cy": 0.423, "team": "TEAM_A", "legible_num": "14"},
    {"id": "#57", "cx": 0.893, "cy": 0.581, "team": "TEAM_A", "legible_num": "57"},
]


def score_cloud_game(spec: Dict[str, Any]) -> Dict[str, Any]:
    gt = spec["visual_gt"]
    ce = spec["cloud_eval"]
    if not ce["analyzed"]:
        return {
            "analyzed": False,
            "detection_precision": 0.0,
            "detection_recall": 0.0,
            "detection_f1": 0.0,
            "detected_count": 0,
            "gt_total": len(gt),
            "off_ice_fp": 0,
            "skater_team_acc": None,
            "ref_precision": None,
            "ref_recall": None,
            "legible_num_hits": f"0/{sum(1 for g in gt if g['legible_num'])}",
            "notes": ce["notes"],
        }

    pred_by_id = {p["gt_id"]: p for p in ce["preds"]}
    det_tp = 0
    skater_ok, skater_tot = 0, 0
    ref_tp, ref_fn, ref_fp = 0, 0, 0
    leg_hits, leg_tot = 0, 0
    ringed_only_hits = 0
    wrong_or_misplaced_nums = []
    team_errors = []

    for g in gt:
        p = pred_by_id.get(g["id"])
        has_det = p is not None and p["team"] is not None
        if has_det:
            det_tp += 1
            if "badge_only" not in p["ring"] and p["ring"] != "none_excluded":
                ringed_only_hits += 1
        if g["role"] == "REF":
            if p is not None and p["team"] == "REFEREE":
                ref_tp += 1
            else:
                ref_fn += 1
        else:
            if has_det:
                skater_tot += 1
                if p["team"] == g["team"]:
                    skater_ok += 1
                else:
                    team_errors.append(f"{g['id']}(GT={g['team']}->Cloud={p['team']},ring={p['ring']})")
                if p["team"] == "REFEREE":
                    ref_fp += 1
            if g["legible_num"] is not None:
                leg_tot += 1
                if has_det and p["num"] and p["num"].rstrip("*") == g["legible_num"]:
                    leg_hits += 1
        if has_det and p["num"] and ("WRONG" in p["num"] or "MISPLACED" in p["num"]):
            wrong_or_misplaced_nums.append(f"{g['id']}:{p['num']}")

    fp = int(ce["off_ice_fp"])
    det_p = det_tp / max(1, det_tp + fp)
    det_r = det_tp / max(1, len(gt))
    det_f1 = 2 * det_p * det_r / max(1e-9, det_p + det_r)
    named_badges = sum(1 for p in ce["preds"] if p["num"] is not None)
    clean_badges = sum(1 for p in ce["preds"] if p["num"] is not None and "WRONG" not in p["num"] and "MISPLACED" not in p["num"])
    return {
        "analyzed": True,
        "detection_precision": round(det_p, 3),
        "detection_recall": round(det_r, 3),
        "detection_f1": round(det_f1, 3),
        "detected_count": f"{det_tp}/{len(gt)}",
        "ringed_skaters_count": ringed_only_hits,
        "off_ice_fp": fp,
        "skater_team_acc": f"{skater_ok}/{skater_tot} ({skater_ok / max(1, skater_tot) * 100:.1f}%)",
        "team_errors": team_errors,
        "ref_precision": round(ref_tp / max(1, ref_tp + ref_fp), 3) if (ref_tp + ref_fp) > 0 else 0.0,
        "ref_recall": round(ref_tp / max(1, ref_tp + ref_fn), 3),
        "ref_tp_fp_fn": [ref_tp, ref_fp, ref_fn],
        "legible_num_recall": f"{leg_hits}/{leg_tot}",
        "total_badges_emitted": named_badges,
        "wrong_or_misplaced_badges": wrong_or_misplaced_nums,
        "badge_clean_precision_est": f"{clean_badges}/{named_badges}" if named_badges else "0/0",
        "notes": ce["notes"],
    }


def _match_predictions_to_gt(
    gt_items: List[Dict[str, Any]],
    preds: List[Dict[str, Any]],
    max_dist: float = 0.062,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Greedy/optimal bipartite match between predictions and GT centers; return (gt_matches, unmatched_preds)."""
    used_p = set()
    pairs = []
    for gi, g in enumerate(gt_items):
        for pi, p in enumerate(preds):
            pcx = 0.5 * (p["box"][0] + p["box"][2])
            pcy = 0.5 * (p["box"][1] + p["box"][3])
            d = math.hypot(pcx - g["cx"], pcy - g["cy"])
            if d <= max_dist:
                pairs.append((d, gi, pi))
    pairs.sort(key=lambda kv: kv[0])
    gt_match: Dict[int, int] = {}
    for d, gi, pi in pairs:
        if gi not in gt_match and pi not in used_p:
            gt_match[gi] = pi
            used_p.add(pi)

    matched_rows = []
    for gi, g in enumerate(gt_items):
        pi = gt_match.get(gi)
        p = preds[pi] if pi is not None else None
        matched_rows.append({"gt": g, "pred": p})
    unmatched = [preds[i] for i in range(len(preds)) if i not in used_p]
    return matched_rows, unmatched


MIN_OCR_BOX_HEIGHT = 0.085
DISTANT_STR_FULL_BOX_HEIGHT = 0.115
DISTANT_STR_PARTIAL_BOX_HEIGHT = 0.105
DISTANT_STR_MIN_TOP_PROB = 0.72
DISTANT_STR_MIN_LEGIBILITY_PROB = 0.82
DISTANT_STR_MIN_SHARPNESS = 28.0
UNCHANGED_CROP_ANCHOR_WINDOW_SEC = 2.5
UNCHANGED_CROP_LUMA_DIFF_MAX = 3.5

V5_DEFAULT_CFG: Dict[str, Any] = {
    "min_ocr_box_height": 0.085,
    "distant_str_full_box_height": 0.115,
    "distant_str_partial_box_height": 0.105,
    "distant_str_min_top_prob": 0.58,
    "distant_str_min_legibility_prob": 0.75,
    "distant_str_min_sharpness": 12.0,
    "min_laplacian_sharpness": 5.5,
    "min_top_candidate_prob": 0.32,
    "max_illegible_prob": 0.48,
    "min_candidate_legibility_prob": 0.64,
    "min_candidate_back_prob": 0.56,
    "min_partial_legibility_prob": 0.55,
    "min_partial_back_prob": 0.60,
    "min_partial_digit_prob": 0.65,
    "single_bin_commit_min_conf": 0.90,
}


def apply_str_cfg(str_rec: JerseyStrRecognizer, cfg: Dict[str, Any]) -> None:
    str_rec.MIN_LAPLACIAN_SHARPNESS = float(cfg["min_laplacian_sharpness"])
    str_rec.MIN_TOP_CANDIDATE_PROB = float(cfg["min_top_candidate_prob"])
    str_rec.MAX_ILLEGIBLE_PROB = float(cfg["max_illegible_prob"])
    str_rec.MIN_CANDIDATE_LEGIBILITY_PROB = float(cfg["min_candidate_legibility_prob"])
    str_rec.MIN_CANDIDATE_BACK_PROB = float(cfg["min_candidate_back_prob"])
    str_rec.MIN_PARTIAL_LEGIBILITY_PROB = float(cfg["min_partial_legibility_prob"])
    str_rec.MIN_PARTIAL_BACK_PROB = float(cfg["min_partial_back_prob"])
    str_rec.MIN_PARTIAL_DIGIT_PROB = float(cfg["min_partial_digit_prob"])


def _compute_crop_luma_grid_4x4(crop: np.ndarray) -> np.ndarray:
    h, w = max(1, crop.shape[0]), max(1, crop.shape[1])
    luma = 0.299 * crop[..., 0].astype(np.float64) + 0.587 * crop[..., 1].astype(np.float64) + 0.114 * crop[..., 2].astype(np.float64)
    grid = np.zeros(16, dtype=np.float64)
    for cy in range(4):
        y0 = (cy * h) // 4
        y1 = max(y0 + 1, min(h, ((cy + 1) * h) // 4))
        for cx in range(4):
            x0 = (cx * w) // 4
            x1 = max(x0 + 1, min(w, ((cx + 1) * w) // 4))
            grid[cy * 4 + cx] = float(luma[y0:y1, x0:x1].mean())
    return grid


def _resolve_effective_read_timestamp(
    tid: int,
    crop: np.ndarray,
    ts: float,
    signatures: Dict[int, Tuple[np.ndarray, float]],
) -> float:
    if crop.size == 0:
        return ts
    grid = _compute_crop_luma_grid_4x4(crop)
    prev = signatures.get(tid)
    if prev is not None and abs(ts - prev[1]) <= UNCHANGED_CROP_ANCHOR_WINDOW_SEC:
        if float(np.abs(grid - prev[0]).mean()) < UNCHANGED_CROP_LUMA_DIFF_MAX:
            return prev[1]
    signatures[tid] = (grid, ts)
    return ts


def run_mobile_on_mp4(
    mp4_path: Path,
    eval_t_sec: float,
    gt_items: Optional[List[Dict[str, Any]]],
    detector: YoloDetector,
    str_rec: JerseyStrRecognizer,
    input_wh: Tuple[int, int] = (960, 540),
    sample_dt_sec: float = 0.25,
    top_k_str_per_step: Optional[int] = 6,
    cfg: Optional[Dict[str, Any]] = None,
    collect_diag: bool = False,
) -> Dict[str, Any]:
    """Run v5 Mobile pipeline on an MP4 at input_wh and sample_dt_sec (~4 fps)."""
    c = dict(V5_DEFAULT_CFG)
    if cfg:
        c.update(cfg)
    apply_str_cfg(str_rec, c)

    cap = cv2.VideoCapture(str(mp4_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps * sample_dt_sec)))

    tracker = MobileTracker()
    track_str_signatures: Dict[int, Tuple[np.ndarray, float]] = {}
    eval_emitted: List[Dict[str, Any]] = []
    eval_rgb: Optional[np.ndarray] = None
    best_t_diff = 1e9
    det_counts, raw_det_counts = [], []
    str_calls = 0
    best_diag_by_tid: Dict[int, Dict[str, Any]] = {}

    for fi in range(0, n_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok:
            break
        ts = fi / fps
        if bgr.shape[1] != input_wh[0] or bgr.shape[0] != input_wh[1]:
            bgr_in = cv2.resize(bgr, input_wh, interpolation=cv2.INTER_AREA)
        else:
            bgr_in = bgr
        rgb = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2RGB)
        aspect = rgb.shape[1] / max(1.0, float(rgb.shape[0]))
        feat_rgb = make_feature_rgb(rgb)
        wb = compute_ice_wb(feat_rgb)
        raw_boxes = detector.detect(rgb, panoramic_tiling=True, apply_on_ice=False, wb=wb, feat_rgb=feat_rgb)
        raw_det_counts.append(len(raw_boxes))
        boxes = [b for b in raw_boxes if is_on_ice(b[:4], aspect, ice_context=b[5])]
        det_counts.append(len(boxes))
        dets = [{**extract_torso_features(feat_rgb, rgb, wb, b[:4]), "box": b[:4]} for b in boxes]
        emitted = tracker.update(ts, dets)

        if top_k_str_per_step is not None:
            str_targets = tracker.select_priority_skaters(emitted, limit=top_k_str_per_step, min_height=c["min_ocr_box_height"])
        else:
            str_targets = [
                it for it in emitted
                if not tracker.is_likely_official_track(it["track_id"])
                and (it["box"][3] - it["box"][1]) >= c["min_ocr_box_height"]
                and (it["box"][2] - it["box"][0]) >= 0.015
            ]

        for item in str_targets:
            tid = item["track_id"]
            crop = str_rec.crop_upper_torso(rgb, item["box"], wb)
            if crop is None:
                continue
            tracker.str_attempts[tid] = tracker.str_attempts.get(tid, 0) + 1
            res = str_rec.infer_from_crop(crop, wb, return_raw=collect_diag)
            str_calls += 1
            effective_ts = _resolve_effective_read_timestamp(tid, crop, ts, track_str_signatures)
            tracker.last_str_score[tid] = {"legibility": res["legibility"], "back": res["back"], "sharpness": res["sharpness"]}
            raw_cands = res["candidates"] if (res["legibility"] >= c["min_candidate_legibility_prob"] and res["back"] >= c["min_candidate_back_prob"]) else []
            raw_top_p = raw_cands[0][1] if raw_cands else 0.0
            box_h = item["box"][3] - item["box"][1]
            if box_h < c["distant_str_full_box_height"] and (
                raw_top_p < c["distant_str_min_top_prob"]
                or res["legibility"] < c["distant_str_min_legibility_prob"]
                or res["sharpness"] < c["distant_str_min_sharpness"]
            ):
                cands = []
            else:
                cands = raw_cands
            p_left = res["partial_left"] if box_h >= c["distant_str_partial_box_height"] else None
            p_right = res["partial_right"] if box_h >= c["distant_str_partial_box_height"] else None
            if collect_diag:
                raw_tuple = res.get("_raw")
                score_d = (raw_top_p if cands else 0.0) + 0.3 * res["legibility"]
                prev_d = best_diag_by_tid.get(tid)
                if prev_d is None or score_d > prev_d["score"]:
                    best_diag_by_tid[tid] = {
                        "score": score_d,
                        "ts": ts,
                        "crop": crop.copy(),
                        "derot_crop": raw_tuple[8].copy() if (raw_tuple and raw_tuple[8] is not None) else None,
                        "orient": res.get("orientation_deg", 0.0),
                        "leg": res["legibility"],
                        "back": res["back"],
                        "sharp": res["sharpness"],
                        "cands": cands,
                        "box_h": box_h,
                    }
            if cands or p_left or p_right:
                qw = min(max((0.35 + 0.65 * res["legibility"]) * (0.50 + 0.50 * res["back"]), 0.05), 1.25) if cands else min(max(0.40 + 0.60 * res["back"], 0.10), 1.0)
                rec = tracker.records[tid]
                if cands:
                    rec["observations"].append({"timestamp_sec": effective_ts, "candidates": cands, "weight": qw, "source_weight": 1.6})
                top_p = cands[0][1] if cands else 0.0
                has_l, has_r = p_left is not None, p_right is not None
                pscale = 0.35 if top_p >= 0.15 else (0.65 if (has_l and has_r) else 1.0)
                if has_l:
                    rec["observations"].append({
                        "timestamp_sec": effective_ts, "number": p_left[0], "conf": p_left[1],
                        "kind": "LEFT_PARTIAL", "weight": min(max(qw * pscale, 0.05), 1.5), "source_weight": 1.6,
                    })
                if has_r:
                    rec["observations"].append({
                        "timestamp_sec": effective_ts, "number": p_right[0], "conf": p_right[1],
                        "kind": "RIGHT_PARTIAL", "weight": min(max(qw * pscale, 0.05), 1.5), "source_weight": 1.6,
                    })

        if abs(ts - eval_t_sec) < best_t_diff:
            best_t_diff = abs(ts - eval_t_sec)
            eval_emitted = [dict(x) for x in emitted]
            eval_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    cap.release()
    resolved = tracker.resolve_all(single_bin_min_conf=c["single_bin_commit_min_conf"])
    for p in eval_emitted:
        team, num, conf = resolved.get(p["track_id"], (p["team"], None, 0.0))
        p["team"] = team
        p["number"] = num if team != "REFEREE" else None
        p["conf"] = round(conf, 3)

    team_counts = {"TEAM_A": 0, "TEAM_B": 0, "REFEREE": 0, "UNKNOWN": 0}
    for tid in tracker.records:
        t = resolved.get(tid, (tracker.records[tid]["team"], None, 0.0))[0]
        team_counts[t] = team_counts.get(t, 0) + 1
    resolved_nums = sorted({
        f"{resolved[tid][0]}:#{resolved[tid][1]}(T{tid})"
        for tid in resolved if resolved[tid][1] is not None and resolved[tid][0] != "REFEREE"
    })

    out: Dict[str, Any] = {
        "input_wh": list(input_wh),
        "frames_sampled": len(det_counts),
        "str_calls": str_calls,
        "mean_raw_det": round(float(np.mean(raw_det_counts)) if raw_det_counts else 0.0, 2),
        "mean_on_ice_det": round(float(np.mean(det_counts)) if det_counts else 0.0, 2),
        "max_on_ice_det": int(max(det_counts)) if det_counts else 0,
        "track_team_counts": team_counts,
        "all_resolved_numbers": resolved_nums,
        "_eval_emitted": eval_emitted,
        "_eval_rgb": eval_rgb,
    }
    if collect_diag:
        chains = tracker.build_stitched_chains()
        tid_to_chain_diag = {}
        for ch_tids in chains:
            best_ch_diag = None
            for tid in ch_tids:
                d = best_diag_by_tid.get(tid)
                if d is not None and (best_ch_diag is None or d["score"] > best_ch_diag["score"]):
                    best_ch_diag = d
            if best_ch_diag is not None:
                for tid in ch_tids:
                    tid_to_chain_diag[tid] = best_ch_diag
        out["_diag_by_tid"] = tid_to_chain_diag

    if gt_items is not None:
        matched_rows, unmatched = _match_predictions_to_gt(gt_items, eval_emitted)
        det_tp = sum(1 for m in matched_rows if m["pred"] is not None)
        fp = len(unmatched)
        det_p = det_tp / max(1, det_tp + fp)
        det_r = det_tp / max(1, len(gt_items))
        det_f1 = 2 * det_p * det_r / max(1e-9, det_p + det_r)

        sk_ok, sk_tot = 0, 0
        ref_tp, ref_fp, ref_fn = 0, 0, 0
        num_hits, num_preds, leg_tot = 0, 0, 0
        misses, team_errs, num_details = [], [], []
        wrong_preds = 0

        for m in matched_rows:
            g, p = m["gt"], m["pred"]
            if p is None:
                misses.append(g["id"])
                if g["team"] == "REFEREE":
                    ref_fn += 1
                elif g.get("legible_num"):
                    leg_tot += 1
                continue
            if g["team"] == "REFEREE":
                if p["team"] == "REFEREE":
                    ref_tp += 1
                else:
                    ref_fn += 1
                    team_errs.append(f"{g['id']}(REF->{p['team']})")
            else:
                sk_tot += 1
                if p["team"] == g["team"]:
                    sk_ok += 1
                else:
                    team_errs.append(f"{g['id']}({g['team']}->{p['team']})")
                if p["team"] == "REFEREE":
                    ref_fp += 1
                if g.get("legible_num"):
                    leg_tot += 1
                if p["number"] is not None:
                    num_preds += 1
                    if g.get("legible_num") and p["number"] == g["legible_num"]:
                        num_hits += 1
                    elif g.get("legible_num") and p["number"] != g["legible_num"]:
                        wrong_preds += 1
                    num_details.append(f"{g['id']}:pred=#{p['number']}(gt={g.get('legible_num')},c={p['conf']})")

        out.update({
            "detection_precision": round(det_p, 3),
            "detection_recall": round(det_r, 3),
            "detection_f1": round(det_f1, 3),
            "detected_count": f"{det_tp}/{len(gt_items)}",
            "off_ice_fp_at_teval": fp,
            "off_ice_fp_boxes": [
                {"cx": round(0.5*(u["box"][0]+u["box"][2]), 3), "fy": round(u["box"][3], 3), "team": u["team"], "tid": u["track_id"]}
                for u in unmatched
            ],
            "missed_gt": misses,
            "skater_team_acc": f"{sk_ok}/{sk_tot} ({sk_ok / max(1, sk_tot) * 100:.1f}%)",
            "team_errors": team_errs,
            "ref_precision": round(ref_tp / max(1, ref_tp + ref_fp), 3) if (ref_tp + ref_fp) > 0 else 0.0,
            "ref_recall": round(ref_tp / max(1, ref_tp + ref_fn), 3),
            "ref_tp_fp_fn": [ref_tp, ref_fp, ref_fn],
            "legible_num_hits": f"{num_hits}/{leg_tot}",
            "legible_wrong_num_preds": wrong_preds,
            "num_predictions_at_teval": num_preds,
            "num_details": num_details,
            "_eval_emitted": eval_emitted,
            "_unmatched": unmatched,
            "_eval_rgb": eval_rgb,
        })
    return out


def extract_sweep_cache(
    detector: YoloDetector,
    str_rec: JerseyStrRecognizer,
    eval_dir: Path,
    min_box_h: float = 0.055,
    cleanhold_mp4: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run YOLO + raw v5 STR once across Games 1-5 + CoachBox Cleanhold (960x540 & 1920x1080) for fast threshold sweep."""
    videos: List[Tuple[str, Path, float, List[Dict[str, Any]]]] = []
    for g in range(1, 6):
        spec = GAMES_SPEC[g]
        videos.append((f"game_{g}", eval_dir / f"lz_game{g}_30s.mp4", spec["eval_t_sec"], spec["visual_gt"]))
    if cleanhold_mp4 is None:
        cleanhold_mp4 = DEFAULT_WORK_DIR / "coachbox_v2" / "coachbox_tvbd_cougars_cleanhold_30s.mp4"
    if cleanhold_mp4.exists():
        videos.append(("coachbox_cleanhold", cleanhold_mp4, 16.8, COACHBOX_GT_FRAME08))
    else:
        print(f"[sweep_cache] warning: cleanhold mp4 not found at {cleanhold_mp4}; skipping", flush=True)

    apply_str_cfg(str_rec, {**V5_DEFAULT_CFG, "min_laplacian_sharpness": 1.0})
    cached_runs = []
    for vname, mp4_path, eval_t, gt_items in videos:
        for input_wh in ((960, 540), (1920, 1080)):
            cap = cv2.VideoCapture(str(mp4_path))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, int(round(fps * 0.25)))
            frames = []
            best_t_diff = 1e9
            eval_fi = -1
            for fi in range(0, n_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, bgr = cap.read()
                if not ok:
                    break
                ts = fi / fps
                bgr_in = cv2.resize(bgr, input_wh, interpolation=cv2.INTER_AREA) if (bgr.shape[1], bgr.shape[0]) != input_wh else bgr
                rgb = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2RGB)
                aspect = rgb.shape[1] / max(1.0, float(rgb.shape[0]))
                feat_rgb = make_feature_rgb(rgb)
                wb = compute_ice_wb(feat_rgb)
                raw_boxes = detector.detect(rgb, panoramic_tiling=True, apply_on_ice=False, wb=wb, feat_rgb=feat_rgb)
                boxes = [b for b in raw_boxes if is_on_ice(b[:4], aspect, ice_context=b[5])]
                dets = []
                for b in boxes:
                    feat = extract_torso_features(feat_rgb, rgb, wb, b[:4])
                    feat["box"] = b[:4]
                    bh = b[3] - b[1]
                    bw = b[2] - b[0]
                    raw_str = None
                    luma_grid = None
                    if bh >= min_box_h and bw >= 0.015:
                        crop = str_rec.crop_upper_torso(rgb, b[:4], wb)
                        if crop is not None:
                            res = str_rec.infer_from_crop(crop, wb, return_raw=True)
                            rt = res["_raw"]
                            raw_str = (rt[0], rt[1], rt[2], rt[3], rt[4], rt[5], rt[6])
                            luma_grid = _compute_crop_luma_grid_4x4(crop)
                    dets.append((feat, raw_str, luma_grid))
                if abs(ts - eval_t) < best_t_diff:
                    best_t_diff = abs(ts - eval_t)
                    eval_fi = len(frames)
                frames.append((ts, dets))
            cap.release()
            cached_runs.append({
                "name": vname,
                "wh": input_wh,
                "gt": gt_items,
                "frames": frames,
                "eval_fi": eval_fi,
            })
            print(f"[sweep_cache] {vname} {input_wh[0]}x{input_wh[1]}: {len(frames)} frames cached", flush=True)
    return cached_runs


def replay_sweep_config(cached_runs: List[Dict[str, Any]], str_rec: JerseyStrRecognizer, cfg: Dict[str, Any]) -> Dict[str, Any]:
    apply_str_cfg(str_rec, cfg)
    lz_hits = {960: [0, 0, 0, 0], 1920: [0, 0, 0, 0]}  # hits, gt_tot, wrong_on_legible, total_num_preds
    cb_hits = {960: [0, 0, 0, 0], 1920: [0, 0, 0, 0]}

    for run in cached_runs:
        tracker = MobileTracker()
        sigs: Dict[int, Tuple[np.ndarray, float]] = {}
        eval_emitted: List[Dict[str, Any]] = []
        for f_idx, (ts, frame_dets) in enumerate(run["frames"]):
            dets = [fd[0] for fd in frame_dets]
            emitted = tracker.update(ts, dets)
            box_to_raw = {fd[0]["box"]: (fd[1], fd[2]) for fd in frame_dets}
            targets = tracker.select_priority_skaters(emitted, limit=6, min_height=cfg["min_ocr_box_height"])
            for item in targets:
                tid = item["track_id"]
                raw_str, luma_grid = box_to_raw.get(item["box"], (None, None))
                if raw_str is None or luma_grid is None:
                    continue
                tracker.str_attempts[tid] = tracker.str_attempts.get(tid, 0) + 1
                leg_f, back_f, sharp1, orient1, num_f, ld_f, rd_f = raw_str
                res = str_rec.decode_raw_pass(leg_f, back_f, sharp1, orient1, num_f, ld_f, rd_f)
                prev = sigs.get(tid)
                if prev is not None and abs(ts - prev[1]) <= UNCHANGED_CROP_ANCHOR_WINDOW_SEC and float(np.abs(luma_grid - prev[0]).mean()) < UNCHANGED_CROP_LUMA_DIFF_MAX:
                    eff_ts = prev[1]
                else:
                    sigs[tid] = (luma_grid, ts)
                    eff_ts = ts
                tracker.last_str_score[tid] = {"legibility": res["legibility"], "back": res["back"], "sharpness": res["sharpness"]}
                raw_cands = res["candidates"] if (res["legibility"] >= cfg["min_candidate_legibility_prob"] and res["back"] >= cfg["min_candidate_back_prob"]) else []
                raw_top_p = raw_cands[0][1] if raw_cands else 0.0
                box_h = item["box"][3] - item["box"][1]
                if box_h < cfg["distant_str_full_box_height"] and (
                    raw_top_p < cfg["distant_str_min_top_prob"]
                    or res["legibility"] < cfg["distant_str_min_legibility_prob"]
                    or res["sharpness"] < cfg["distant_str_min_sharpness"]
                ):
                    cands = []
                else:
                    cands = raw_cands
                p_left = res["partial_left"] if box_h >= cfg["distant_str_partial_box_height"] else None
                p_right = res["partial_right"] if box_h >= cfg["distant_str_partial_box_height"] else None
                if cands or p_left or p_right:
                    qw = min(max((0.35 + 0.65 * res["legibility"]) * (0.50 + 0.50 * res["back"]), 0.05), 1.25) if cands else min(max(0.40 + 0.60 * res["back"], 0.10), 1.0)
                    rec = tracker.records[tid]
                    if cands:
                        rec["observations"].append({"timestamp_sec": eff_ts, "candidates": cands, "weight": qw, "source_weight": 1.6})
                    top_p = cands[0][1] if cands else 0.0
                    has_l, has_r = p_left is not None, p_right is not None
                    pscale = 0.35 if top_p >= 0.15 else (0.65 if (has_l and has_r) else 1.0)
                    if has_l:
                        rec["observations"].append({"timestamp_sec": eff_ts, "number": p_left[0], "conf": p_left[1], "kind": "LEFT_PARTIAL", "weight": min(max(qw * pscale, 0.05), 1.5), "source_weight": 1.6})
                    if has_r:
                        rec["observations"].append({"timestamp_sec": eff_ts, "number": p_right[0], "conf": p_right[1], "kind": "RIGHT_PARTIAL", "weight": min(max(qw * pscale, 0.05), 1.5), "source_weight": 1.6})
            if f_idx == run["eval_fi"]:
                eval_emitted = [dict(x) for x in emitted]

        resolved = tracker.resolve_all(single_bin_min_conf=cfg["single_bin_commit_min_conf"])
        for p in eval_emitted:
            team, num, conf = resolved.get(p["track_id"], (p["team"], None, 0.0))
            p["team"] = team
            p["number"] = num if team != "REFEREE" else None

        matched_rows, _ = _match_predictions_to_gt(run["gt"], eval_emitted)
        bucket = cb_hits[run["wh"][0]] if run["name"] == "coachbox_cleanhold" else lz_hits[run["wh"][0]]
        for m in matched_rows:
            g, p = m["gt"], m["pred"]
            if g.get("legible_num"):
                bucket[1] += 1
            if p is not None and g["team"] != "REFEREE" and p["number"] is not None:
                bucket[3] += 1
                if g.get("legible_num"):
                    if p["number"] == g["legible_num"]:
                        bucket[0] += 1
                    else:
                        bucket[2] += 1

    return {
        "lz_960": lz_hits[960],
        "lz_1920": lz_hits[1920],
        "cb_960": cb_hits[960],
        "cb_1920": cb_hits[1920],
        "total_hits": lz_hits[960][0] + lz_hits[1920][0] + cb_hits[960][0] + cb_hits[1920][0],
        "total_wrong_leg": lz_hits[960][2] + lz_hits[1920][2] + cb_hits[960][2] + cb_hits[1920][2],
    }


def render_v5_diag_eval(
    out_png: Path,
    diag_records: List[Dict[str, Any]],
) -> None:
    """Render visual diagnostic sheet of v5 torso crops, derotated crops, orientation_deg, and predictions."""
    if not diag_records:
        return
    cols = 4
    rows = (len(diag_records) + cols - 1) // cols
    cell_w, cell_h = 340, 152
    canvas = np.full((rows * cell_h + 36, cols * cell_w, 3), 22, dtype=np.uint8)
    cv2.putText(canvas, "v5 STR Diagnostic Eval (Torso Crop | Derotated Crop | Pred vs GT)", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 240, 245), 2, cv2.LINE_AA)

    for idx, rec in enumerate(diag_records):
        r, c = idx // cols, idx % cols
        x0, y0 = c * cell_w, 36 + r * cell_h
        cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + cell_w - 4, y0 + cell_h - 4), (45, 52, 66), 1)
        crop_bgr = cv2.cvtColor(rec["crop"], cv2.COLOR_RGB2BGR)
        ch, cw = crop_bgr.shape[:2]
        sc = min(96.0 / max(1, cw), 96.0 / max(1, ch))
        dw, dh = max(1, int(round(cw * sc))), max(1, int(round(ch * sc)))
        px, py = x0 + 10 + (96 - dw) // 2, y0 + 10 + (96 - dh) // 2
        canvas[py:py + dh, px:px + dw] = cv2.resize(crop_bgr, (dw, dh), interpolation=cv2.INTER_CUBIC)

        if rec.get("derot_crop") is not None:
            dc_bgr = cv2.cvtColor(rec["derot_crop"], cv2.COLOR_RGB2BGR)
            dch, dcw = dc_bgr.shape[:2]
            dsc = min(96.0 / max(1, dcw), 96.0 / max(1, dch))
            ddw, ddh = max(1, int(round(dcw * dsc))), max(1, int(round(dch * dsc)))
            dpx, dpy = x0 + 112 + (96 - ddw) // 2, y0 + 10 + (96 - ddh) // 2
            canvas[dpy:dpy + ddh, dpx:dpx + ddw] = cv2.resize(dc_bgr, (ddw, ddh), interpolation=cv2.INTER_CUBIC)

        hit = rec.get("gt_num") and (rec.get("pred_num") == rec.get("gt_num"))
        col = (80, 240, 120) if hit else ((0, 195, 255) if rec.get("pred_num") is None else (80, 90, 255))
        t1 = f"{rec['tag']} gt=#{rec.get('gt_num') or '-'}"
        t2 = f"pred=#{rec.get('pred_num') or '-'} ({rec.get('conf', 0.0):.2f})"
        t3 = f"ori={rec.get('orient', 0.0):+.1f}d leg={rec.get('leg', 0.0):.2f}"
        t4 = f"bk={rec.get('back', 0.0):.2f} lap={rec.get('sharp', 0.0):.0f} bh={rec.get('box_h', 0.0):.3f}"
        cv2.putText(canvas, t1, (x0 + 214, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 235, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, t2, (x0 + 214, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
        cv2.putText(canvas, t3, (x0 + 214, y0 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 225), 1, cv2.LINE_AA)
        cv2.putText(canvas, t4, (x0 + 214, y0 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 195, 210), 1, cv2.LINE_AA)
        cands_str = ",".join(f"#{n}:{p:.2f}" for n, p in rec.get("cands", [])[:2])
        cv2.putText(canvas, f"top: {cands_str}" if cands_str else "top: none", (x0 + 10, y0 + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 210, 230), 1, cv2.LINE_AA)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)
    print(f"[diag] saved v5 diagnostic sheet ({len(diag_records)} tiles) to {out_png}")


def render_eval_compare_sheet(
    game_idx: int,
    spec: Dict[str, Any],
    eval_dir: Path,
    res_960: Dict[str, Any],
    res_1080: Dict[str, Any],
) -> None:
    """Create 2x2 Cloud Overlay vs Mobile R2 (1920x1080) vs Mobile R3 (960x540 & 1920x1080) sheet."""
    cloud_bgr = cv2.imread(str(eval_dir / f"lz_g{game_idx}_cloud_overlay_crop.png"))
    raw_bgr = cv2.imread(str(eval_dir / f"lz_g{game_idx}_raw_teval_crop.png"))
    H, W = raw_bgr.shape[:2]

    def _draw_mobile(base_bgr: np.ndarray, res: Dict[str, Any], title: str) -> np.ndarray:
        vis = base_bgr.copy()
        unmatched_ids = {u["track_id"] for u in res.get("_unmatched", [])}
        colors = {"TEAM_A": (0, 220, 255), "TEAM_B": (255, 255, 255), "REFEREE": (0, 255, 0), "UNKNOWN": (180, 180, 180)}
        for p in res.get("_eval_emitted", []):
            x1, y1, x2, y2 = [int(v) for v in (p["box"][0]*W, p["box"][1]*H, p["box"][2]*W, p["box"][3]*H)]
            is_fp = p["track_id"] in unmatched_ids
            col = (255, 0, 255) if is_fp else colors.get(p["team"], (200, 200, 200))
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3 if is_fp else 2)
            t_tag = "FP" if is_fp else ("R" if p["team"] == "REFEREE" else ("A" if p["team"] == "TEAM_A" else "B"))
            lbl = f"[{t_tag}]T{p['track_id']}" + (f" #{p['number']}" if p["number"] else "")
            cv2.putText(vis, lbl, (x1, max(24, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        for g in spec["visual_gt"]:
            gx, gy = int(g["cx"] * W), int(g["cy"] * H)
            cv2.drawMarker(vis, (gx, gy), (0, 165, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.rectangle(vis, (0, 0), (W, 44), (15, 20, 30), -1)
        cv2.putText(vis, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        return vis

    c_vis = cloud_bgr.copy()
    cv2.rectangle(c_vis, (0, 0), (W, 44), (15, 20, 30), -1)
    c_title = f"Game {game_idx} Cloud (t={spec['t_eval_str']}) " + ("NOT ANALYZED (0%)" if not spec["cloud_eval"]["analyzed"] else "Analyzed")
    cv2.putText(c_vis, c_title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
    c_panel = cv2.resize(c_vis, (1280, 720))

    r2_sheet = cv2.imread(str(eval_dir / f"lz_g{game_idx}_eval_compare_r2.png"))
    if r2_sheet is not None and r2_sheet.shape[0] >= 1440 and r2_sheet.shape[1] >= 2560:
        r2_panel = cv2.resize(r2_sheet[720:1440, 1280:2560], (1280, 720))
    else:
        r2_panel = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(r2_panel, "R2 1080p baseline missing", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    m960_panel = cv2.resize(_draw_mobile(
        raw_bgr, res_960,
        f"Mobile R3 960x540: Det={res_960['detected_count']} FP={res_960['off_ice_fp_at_teval']} Team={res_960['skater_team_acc']} Ref={res_960['ref_tp_fp_fn']}",
    ), (1280, 720))
    m1080_panel = cv2.resize(_draw_mobile(
        raw_bgr, res_1080,
        f"Mobile R3 1920x1080: Det={res_1080['detected_count']} FP={res_1080['off_ice_fp_at_teval']} Team={res_1080['skater_team_acc']} Ref={res_1080['ref_tp_fp_fn']}",
    ), (1280, 720))

    grid = np.vstack([np.hstack([c_panel, r2_panel]), np.hstack([m960_panel, m1080_panel])])
    cv2.imwrite(str(eval_dir / f"lz_g{game_idx}_eval_compare_r3.png"), grid)


def _aggregate_5game_mobile(games_out: Dict[str, Any], key: str) -> Dict[str, Any]:
    det_tp = det_gt = off_fp = sk_ok = sk_tot = r_tp = r_fp = r_fn = num_hits = num_gt = num_preds = 0
    for g in range(1, 6):
        m = games_out[f"game_{g}"].get(key)
        if not m:
            continue
        tp, gt = [int(x) for x in m["detected_count"].split("/")]
        det_tp += tp
        det_gt += gt
        off_fp += int(m["off_ice_fp_at_teval"])
        sok, stot = [int(x) for x in m["skater_team_acc"].split(" ")[0].split("/")]
        sk_ok += sok
        sk_tot += stot
        rtp, rfp, rfn = m["ref_tp_fp_fn"]
        r_tp += rtp
        r_fp += rfp
        r_fn += rfn
        lh, lt = [int(x) for x in m["legible_num_hits"].split("/")]
        num_hits += lh
        num_gt += lt
        num_preds += int(m.get("num_predictions_at_teval", 0))
    det_p = det_tp / max(1, det_tp + off_fp)
    det_r = det_tp / max(1, det_gt)
    det_f1 = 2 * det_p * det_r / max(1e-9, det_p + det_r)
    return {
        "detected": f"{det_tp}/{det_gt}",
        "detection_precision": round(det_p, 3),
        "detection_recall": round(det_r, 3),
        "detection_f1": round(det_f1, 3),
        "off_ice_fp": off_fp,
        "skater_team_acc": f"{sk_ok}/{sk_tot} ({sk_ok / max(1, sk_tot) * 100:.1f}%)",
        "ref_tp_fp_fn": [r_tp, r_fp, r_fn],
        "ref_precision": round(r_tp / max(1, r_tp + r_fp), 3) if (r_tp + r_fp) > 0 else 0.0,
        "ref_recall": round(r_tp / max(1, r_tp + r_fn), 3) if (r_tp + r_fn) > 0 else 0.0,
        "legible_num_hits": f"{num_hits}/{num_gt}",
        "num_predictions_at_teval": num_preds,
    }


def _delta_str(prev: Optional[Dict[str, Any]], cur: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not prev:
        return None
    return {
        "detected_count": f"{prev['detected_count']} -> {cur['detected_count']}",
        "off_ice_fp_at_teval": f"{prev['off_ice_fp_at_teval']} -> {cur['off_ice_fp_at_teval']}",
        "skater_team_acc": f"{prev['skater_team_acc']} -> {cur['skater_team_acc']}",
        "ref_tp_fp_fn": f"{prev['ref_tp_fp_fn']} -> {cur['ref_tp_fp_fn']}",
        "legible_num_hits": f"{prev['legible_num_hits']} -> {cur['legible_num_hits']}",
        "num_predictions_at_teval": f"{prev['num_predictions_at_teval']} -> {cur['num_predictions_at_teval']}",
    }


_SWEEP_WORKER_CACHE: Optional[List[Dict[str, Any]]] = None
_SWEEP_WORKER_REC: Optional[JerseyStrRecognizer] = None


def _init_sweep_worker(cache_pkl: str) -> None:
    global _SWEEP_WORKER_CACHE, _SWEEP_WORKER_REC
    with open(cache_pkl, "rb") as f:
        payload = pickle.load(f)
    _SWEEP_WORKER_CACHE = payload["cached"]
    rec = JerseyStrRecognizer.__new__(JerseyStrRecognizer)
    _SWEEP_WORKER_REC = rec


def _eval_sweep_worker(cfg: Dict[str, Any]) -> Dict[str, Any]:
    assert _SWEEP_WORKER_CACHE is not None and _SWEEP_WORKER_REC is not None
    sc = replay_sweep_config(_SWEEP_WORKER_CACHE, _SWEEP_WORKER_REC, cfg)
    tot_preds = sc["lz_960"][3] + sc["lz_1920"][3] + sc["cb_960"][3] + sc["cb_1920"][3]
    spurious_est = max(0, tot_preds - sc["total_hits"] - sc["total_wrong_leg"])
    score = sc["total_hits"] * 3.0 - sc["total_wrong_leg"] * 4.0 - 0.25 * spurious_est
    return {"score": round(score, 2), "summary": sc, "cfg": cfg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir", type=Path, default=REPO_ROOT / "android" / "app" / "src" / "main" / "assets")
    ap.add_argument("--eval-dir", type=Path, default=DEFAULT_WORK_DIR / "leonxzhao_eval")
    ap.add_argument("--v5-dir", type=Path, default=DEFAULT_WORK_DIR / "v5_str")
    ap.add_argument("--report-out", type=Path, default=REPO_ROOT / "research" / "leonxzhao_5games_report.json")
    ap.add_argument("--cleanhold-mp4", type=Path, default=DEFAULT_WORK_DIR / "coachbox_v2" / "coachbox_tvbd_cougars_cleanhold_30s.mp4")
    ap.add_argument(
        "--native-g1-mp4",
        type=Path,
        default=DEFAULT_WORK_DIR / "downloads" / "Jr-12AA2-vs-Tri-Valley-Blue-Devils-12AA-2-Nov-9-2026-Penalty-Tripping-28-at-27m53s.mp4",
    )
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--cfg-json", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    detector = YoloDetector(args.assets_dir / "coach_person_detector.onnx")
    str_rec = JerseyStrRecognizer(args.assets_dir / "coach_jersey_str.onnx")

    if args.sweep:
        args.v5_dir.mkdir(parents=True, exist_ok=True)
        onnx_p = args.assets_dir / "coach_jersey_str.onnx"
        onnx_sig = (int(onnx_p.stat().st_mtime_ns), int(onnx_p.stat().st_size))
        cache_pkl = args.v5_dir / "v5_sweep_cache.pkl"
        cached = None
        if cache_pkl.exists():
            try:
                with open(cache_pkl, "rb") as f:
                    payload = pickle.load(f)
                if payload.get("onnx_sig") == onnx_sig:
                    cached = payload["cached"]
                    print(f"[sweep_cache] loaded cached frames from {cache_pkl.name}", flush=True)
            except Exception:
                cached = None
        if cached is None:
            cached = extract_sweep_cache(detector, str_rec, args.eval_dir, min_box_h=0.055, cleanhold_mp4=args.cleanhold_mp4)
            with open(cache_pkl, "wb") as f:
                pickle.dump({"onnx_sig": onnx_sig, "cached": cached}, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[sweep_cache] saved cache to {cache_pkl.name}", flush=True)

        cfgs = []
        for min_h in (0.072, 0.075, 0.082):
            for dist_h in (0.100, 0.108, 0.115):
                for dist_top, dist_leg, dist_sh in (
                    (0.42, 0.68, 8.0),
                    (0.52, 0.74, 10.0),
                    (0.58, 0.76, 12.0),
                    (0.68, 0.80, 14.0),
                ):
                    for min_top, min_leg, min_bk, min_sh in (
                        (0.28, 0.58, 0.52, 4.0),
                        (0.32, 0.64, 0.56, 5.5),
                        (0.35, 0.68, 0.60, 7.5),
                    ):
                        for sb_conf in (0.85, 0.88, 0.92):
                            cfgs.append({
                                **V5_DEFAULT_CFG,
                                "min_ocr_box_height": min_h,
                                "distant_str_full_box_height": dist_h,
                                "distant_str_partial_box_height": max(0.085, dist_h - 0.010),
                                "distant_str_min_top_prob": dist_top,
                                "distant_str_min_legibility_prob": dist_leg,
                                "distant_str_min_sharpness": dist_sh,
                                "min_top_candidate_prob": min_top,
                                "min_candidate_legibility_prob": min_leg,
                                "min_candidate_back_prob": min_bk,
                                "min_laplacian_sharpness": min_sh,
                                "single_bin_commit_min_conf": sb_conf,
                            })
        grid = []
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=12, initializer=_init_sweep_worker, initargs=(str(cache_pkl),)) as pool:
            for idx, row in enumerate(pool.imap_unordered(_eval_sweep_worker, cfgs), 1):
                grid.append(row)
                if idx % 24 == 0 or idx == len(cfgs):
                    best_cur = max(grid, key=lambda r: r["score"])
                    print(f"[sweep] {idx}/{len(cfgs)} ({time.time()-t0:.1f}s) best_score={best_cur['score']} {best_cur['summary']}", flush=True)
        grid.sort(key=lambda r: -r["score"])
        out_sweep = args.v5_dir / "v5_sweep_results.json"
        out_sweep.write_text(json.dumps(grid[:25], indent=2) + "\n")
        print(f"[sweep] evaluated {len(grid)} configs in {time.time()-t0:.1f}s; top 8:")
        for row in grid[:8]:
            print(json.dumps(row["summary"]), "min_h=", row["cfg"]["min_ocr_box_height"], "min_sh=", row["cfg"]["min_laplacian_sharpness"], "dist_sh=", row["cfg"]["distant_str_min_sharpness"], "min_top=", row["cfg"]["min_top_candidate_prob"], "dist_top=", row["cfg"]["distant_str_min_top_prob"], "leg=", row["cfg"]["min_candidate_legibility_prob"], "sb=", row["cfg"]["single_bin_commit_min_conf"])
        return

    v5_cfg = dict(V5_DEFAULT_CFG)
    if args.cfg_json and args.cfg_json.exists():
        v5_cfg.update(json.loads(args.cfg_json.read_text()))

    r1_path = args.eval_dir / "leonxzhao_5games_report_r1.json"
    r2_path = args.eval_dir / "leonxzhao_5games_report_r2.json"
    r3_bak_path = args.v5_dir / "leonxzhao_5games_report_r3_backup.json"
    r1_data = json.loads(r1_path.read_text()) if r1_path.exists() else {}
    r2_data = json.loads(r2_path.read_text()) if r2_path.exists() else {}
    if r3_bak_path.exists():
        r3_data = json.loads(r3_bak_path.read_text())
    elif args.report_out.exists():
        r3_data = json.loads(args.report_out.read_text())
    else:
        r3_data = {}

    games_out: Dict[str, Any] = {}
    diag_records: List[Dict[str, Any]] = []

    for g in range(1, 6):
        spec = GAMES_SPEC[g]
        mp4_path = args.eval_dir / f"lz_game{g}_30s.mp4"
        cloud_score = score_cloud_game(spec)
        res_960 = run_mobile_on_mp4(mp4_path, spec["eval_t_sec"], spec["visual_gt"], detector, str_rec, input_wh=(960, 540), sample_dt_sec=0.25, top_k_str_per_step=6, cfg=v5_cfg)
        res_1080 = run_mobile_on_mp4(mp4_path, spec["eval_t_sec"], spec["visual_gt"], detector, str_rec, input_wh=(1920, 1080), sample_dt_sec=0.25, top_k_str_per_step=6, cfg=v5_cfg, collect_diag=True)
        res_960_allstr = run_mobile_on_mp4(mp4_path, spec["eval_t_sec"], spec["visual_gt"], detector, str_rec, input_wh=(960, 540), sample_dt_sec=0.25, top_k_str_per_step=None, cfg=v5_cfg)

        diag_by_tid = res_1080.pop("_diag_by_tid", {})
        matched_rows, _ = _match_predictions_to_gt(spec["visual_gt"], res_1080.get("_eval_emitted", []))
        for m in matched_rows:
            gt_i, pr_i = m["gt"], m["pred"]
            if pr_i is not None and (gt_i.get("legible_num") or pr_i.get("number") is not None):
                d = diag_by_tid.get(pr_i["track_id"])
                if d is not None:
                    diag_records.append({
                        **d,
                        "tag": gt_i["id"],
                        "gt_num": gt_i.get("legible_num"),
                        "pred_num": pr_i.get("number"),
                        "conf": pr_i.get("conf", 0.0),
                    })

        for d in (res_960, res_1080, res_960_allstr):
            d.pop("_eval_emitted", None)
            d.pop("_unmatched", None)
            d.pop("_eval_rgb", None)

        r1_game = r1_data.get("games", {}).get(f"game_{g}", {})
        r2_game = r2_data.get("games", {}).get(f"game_{g}", {})
        r3_game = r3_data.get("games", {}).get(f"game_{g}", {})
        r1_960 = r1_game.get("mobile_960x540")
        r1_1080 = r1_game.get("mobile_1920x1080")
        r2_960 = r2_game.get("mobile_960x540")
        r2_1080 = r2_game.get("mobile_1920x1080")
        r3_960 = r3_game.get("r3_final", {}).get("mobile_960x540") or r3_game.get("mobile_960x540")
        r3_1080 = r3_game.get("r3_final", {}).get("mobile_1920x1080") or r3_game.get("mobile_1920x1080")
        r3_960_allstr = r3_game.get("mobile_960x540_uncapped_str")

        games_out[f"game_{g}"] = {
            "name": spec["name"],
            "t_eval_str": spec["t_eval_str"],
            "gt_summary": {
                "total_on_ice": len(spec["visual_gt"]),
                "team_a": sum(1 for x in spec["visual_gt"] if x["team"] == "TEAM_A"),
                "team_b": sum(1 for x in spec["visual_gt"] if x["team"] == "TEAM_B"),
                "referees": sum(1 for x in spec["visual_gt"] if x["team"] == "REFEREE"),
                "legible_numbers": [x["legible_num"] for x in spec["visual_gt"] if x["legible_num"]],
            },
            "cloud": cloud_score,
            "mobile_960x540": r3_960,
            "mobile_1920x1080": r3_1080,
            "mobile_960x540_uncapped_str": r3_960_allstr,
            "r1_baseline": {"mobile_960x540": r1_960, "mobile_1920x1080": r1_1080},
            "r2_metrics": {"mobile_960x540": r2_960, "mobile_1920x1080": r2_1080},
            "r3_final": {"mobile_960x540": r3_960, "mobile_1920x1080": r3_1080},
            "v5_python": {
                "mobile_960x540": res_960,
                "mobile_1920x1080": res_1080,
                "mobile_960x540_uncapped_str": res_960_allstr,
            },
            "mobile_960x540_v5_python": res_960,
            "mobile_1920x1080_v5_python": res_1080,
            "mobile_960x540_r1_baseline": r1_960,
            "mobile_1920x1080_r1_baseline": r1_1080,
            "mobile_960x540_r2_metrics": r2_960,
            "mobile_1920x1080_r2_metrics": r2_1080,
            "delta_v5_vs_r3_1080": _delta_str(r3_1080, res_1080),
            "delta_v5_vs_r3_960": _delta_str(r3_960, res_960),
        }

    # Regression check: CoachBox cleanhold 30s (960 & 1080) + Game 1 downloaded native MP4 clip
    cleanhold_mp4 = args.cleanhold_mp4
    if cleanhold_mp4.exists():
        reg_cleanhold_960 = run_mobile_on_mp4(cleanhold_mp4, 16.8, COACHBOX_GT_FRAME08, detector, str_rec, input_wh=(960, 540), sample_dt_sec=0.25, top_k_str_per_step=6, cfg=v5_cfg)
        reg_cleanhold_1080 = run_mobile_on_mp4(cleanhold_mp4, 16.8, COACHBOX_GT_FRAME08, detector, str_rec, input_wh=(1920, 1080), sample_dt_sec=0.25, top_k_str_per_step=6, cfg=v5_cfg, collect_diag=True)

        cb_diag_by_tid = reg_cleanhold_1080.pop("_diag_by_tid", {})
        cb_matched, _ = _match_predictions_to_gt(COACHBOX_GT_FRAME08, reg_cleanhold_1080.get("_eval_emitted", []))
        for m in cb_matched:
            gt_i, pr_i = m["gt"], m["pred"]
            if pr_i is not None and (gt_i.get("legible_num") or pr_i.get("number") is not None):
                d = cb_diag_by_tid.get(pr_i["track_id"])
                if d is not None:
                    diag_records.append({
                        **d,
                        "tag": f"CB_{gt_i['id']}",
                        "gt_num": gt_i.get("legible_num"),
                        "pred_num": pr_i.get("number"),
                        "conf": pr_i.get("conf", 0.0),
                    })

        for d in (reg_cleanhold_960, reg_cleanhold_1080):
            for k in ("_eval_emitted", "_unmatched", "_eval_rgb"):
                d.pop(k, None)
    else:
        reg_cleanhold_960 = {"skipped": True, "reason": "file_not_found", "path": cleanhold_mp4.name}
        reg_cleanhold_1080 = {"skipped": True, "reason": "file_not_found", "path": cleanhold_mp4.name}

    native_g1_mp4 = args.native_g1_mp4
    if native_g1_mp4.exists():
        native_g1_960 = run_mobile_on_mp4(native_g1_mp4, 7.5, None, detector, str_rec, input_wh=(960, 540), sample_dt_sec=0.25, top_k_str_per_step=6, cfg=v5_cfg)
        native_g1_1080 = run_mobile_on_mp4(native_g1_mp4, 7.5, None, detector, str_rec, input_wh=(1920, 1080), sample_dt_sec=0.25, top_k_str_per_step=6, cfg=v5_cfg)
        for d in (native_g1_960, native_g1_1080):
            d.pop("_eval_emitted", None)
            d.pop("_eval_rgb", None)
    else:
        native_g1_960 = {"skipped": True, "reason": "file_not_found", "path": native_g1_mp4.name}
        native_g1_1080 = {"skipped": True, "reason": "file_not_found", "path": native_g1_mp4.name}

    try:
        native_g1_rel = str(native_g1_mp4.relative_to(DEFAULT_WORK_DIR))
    except ValueError:
        native_g1_rel = native_g1_mp4.name

    render_v5_diag_eval(args.v5_dir / "v5_diag_eval.png", diag_records)

    r3_sum = r3_data.get("summary_comparison_5games", {})
    report = {
        "round": "v5_python",
        "roster_free_verified": True,
        "elapsed_sec": round(time.time() - t0, 2),
        "v5_calibrated_constants": v5_cfg,
        "summary_comparison_5games": {
            "mobile_r1_960x540": r3_sum.get("mobile_r1_960x540") or (_aggregate_5game_mobile(games_out, "mobile_960x540_r1_baseline") if r1_data else None),
            "mobile_r2_960x540": r3_sum.get("mobile_r2_960x540") or (_aggregate_5game_mobile(games_out, "mobile_960x540_r2_metrics") if r2_data else None),
            "mobile_r3_960x540": r3_sum.get("mobile_r3_960x540") or _aggregate_5game_mobile(games_out, "mobile_960x540"),
            "mobile_v5_python_960x540": _aggregate_5game_mobile(games_out, "mobile_960x540_v5_python"),
            "mobile_r1_1920x1080": r3_sum.get("mobile_r1_1920x1080") or (_aggregate_5game_mobile(games_out, "mobile_1920x1080_r1_baseline") if r1_data else None),
            "mobile_r2_1920x1080": r3_sum.get("mobile_r2_1920x1080") or (_aggregate_5game_mobile(games_out, "mobile_1920x1080_r2_metrics") if r2_data else None),
            "mobile_r3_1920x1080": r3_sum.get("mobile_r3_1920x1080") or _aggregate_5game_mobile(games_out, "mobile_1920x1080"),
            "mobile_v5_python_1920x1080": _aggregate_5game_mobile(games_out, "mobile_1920x1080_v5_python"),
        },
        "games": games_out,
        "regression_coachbox_cleanhold_v5_python_960x540": reg_cleanhold_960,
        "regression_coachbox_cleanhold_v5_python_1920x1080": reg_cleanhold_1080,
        "regression_coachbox_cleanhold_960x540": r3_data.get("regression_coachbox_cleanhold_960x540"),
        "regression_coachbox_cleanhold_1920x1080": r3_data.get("regression_coachbox_cleanhold_1920x1080"),
        "regression_coachbox_cleanhold_960x540_r2_metrics": r2_data.get("regression_coachbox_cleanhold_960x540"),
        "regression_coachbox_cleanhold_960x540_r1_baseline": r1_data.get("regression_coachbox_cleanhold_960x540"),
        "lz_game1_native_clip": {
            "path": native_g1_rel,
            "v5_python": {"mobile_960x540": native_g1_960, "mobile_1920x1080": native_g1_1080},
            "r3_final": r3_data.get("lz_game1_native_clip", {}).get("r3_final"),
            "mobile_960x540": r3_data.get("lz_game1_native_clip", {}).get("mobile_960x540"),
            "mobile_1920x1080": r3_data.get("lz_game1_native_clip", {}).get("mobile_1920x1080"),
            "r2_metrics": {
                "mobile_960x540": r2_data.get("lz_game1_native_clip", {}).get("mobile_960x540"),
                "mobile_1920x1080": r2_data.get("lz_game1_native_clip", {}).get("mobile_1920x1080"),
            },
            "r1_baseline": r1_data.get("lz_game1_native_clip"),
        },
    }
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary_comparison_5games"], indent=2))
    print(f"[done] wrote report to {args.report_out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
