"""Train & export roster-free v5 hockey jersey STR + Orientation model (`coach_jersey_str.onnx`) on RTX 5090.

Architecture (`JerseyStrNetV5`, 128x128 RGB input):
  - Pretrained MobileNetV3-Large backbone (`stage4` 32x32, `stage8` 16x16, `stage16` 8x8, `stage32` 4x4)
  - Lightweight top-down FPN (`f32 -> f16 -> f8`) fused with stride-2 `stage4` edges -> `f8_fpn [B, 128, 16, 16]`
  - Training-only Auxiliary Super-Resolution Decoder (`SR-Aux`, BCE + Dice weight 0.60 vs `sr_target` [32x32]),
    omitted completely from `CalibratedStrWrapper.forward()` at ONNX export (0 runtime cost on phone)
  - 6-query 2D-positional cross-attention (`left, right, center, legible, back, orient`) over 256 tokens
  - Compositional digit-to-number fusion + `orientation_deg = 60.0 * tanh(orient_head)` (`+` = CCW in image space)
  - 100% XNNPACK-safe ONNX ops only (no GridSample), embedded ImageNet normalization & temperature scaling.

Zero test leakage: trains & validates strictly on synthetic shards from `jersey_synth.py`.
Uniform 1..99 + "00" class prior (100% roster-free).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from jersey_synth import (
    DIGIT_BLANK,
    LETTERBOX_PAD,
    NUM_CLASSES,
    PARTIAL_LEFT_ONLY,
    PARTIAL_NONE,
    PARTIAL_RIGHT_ONLY,
    TARGET_SIZE,
    _apply_cylindrical_warp_and_folds,
    _render_jersey_number_patch,
)


def build_compositional_maps() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (L_map [11, 101], R_map [11, 101]) mapping digit log-probs to 101 number classes."""
    l_map = torch.zeros(11, NUM_CLASSES, dtype=torch.float32)
    r_map = torch.zeros(11, NUM_CLASSES, dtype=torch.float32)
    l_map[DIGIT_BLANK, 0] = 1.0
    r_map[DIGIT_BLANK, 0] = 1.0
    for d in range(1, 10):
        l_map[d, d] = 1.0
        r_map[DIGIT_BLANK, d] = 1.0
    for n in range(10, 100):
        l_map[n // 10, n] = 1.0
        r_map[n % 10, n] = 1.0
    l_map[0, 100] = 1.0
    r_map[0, 100] = 1.0
    return l_map, r_map


def build_swapped_number_lut() -> torch.Tensor:
    """Return lookup tensor [101] mapping 2-digit number class to its digit-swapped class (else -1)."""
    lut = torch.full((NUM_CLASSES,), -1, dtype=torch.long)
    for n in range(10, 100):
        a, b = n // 10, n % 10
        if a != b and b != 0:
            lut[n] = b * 10 + a
    return lut


def _make_2d_sincos_pos_emb(gh: int, gw: int, dim: int) -> torch.Tensor:
    """Create (1, gh*gw, dim) 2D sin-cos positional embedding."""
    assert dim % 4 == 0
    quarter = dim // 4
    omega = torch.arange(quarter, dtype=torch.float32) / float(quarter)
    omega = 1.0 / (10000.0 ** omega)
    ys = torch.arange(gh, dtype=torch.float32) / max(1.0, float(gh - 1)) * 2.0 - 1.0
    xs = torch.arange(gw, dtype=torch.float32) / max(1.0, float(gw - 1)) * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    out_y = grid_y.reshape(-1, 1) * omega.unsqueeze(0) * math.pi
    out_x = grid_x.reshape(-1, 1) * omega.unsqueeze(0) * math.pi
    emb = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=-1)
    return emb.unsqueeze(0)


class ConvBnHswish(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 1, stride: int = 1, pad: int = 0, groups: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=stride, padding=pad, groups=groups, bias=False),
            nn.BatchNorm2d(c_out),
            nn.Hardswish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class JerseyStrNetV5(nn.Module):
    """High-res FPN + Training-Only SR-Aux + 6-Query 2D Cross-Attention + Orientation STR network."""

    def __init__(self, ckpt_path: Path) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        base = torchvision.models.mobilenet_v3_large(weights=None)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            base.load_state_dict(state)
        else:
            print(f"[warn] checkpoint not found at {ckpt_path}, using random init")

        self.stage4 = nn.Sequential(*list(base.features[:4]))     # -> (B, 24, 32, 32)
        self.stage8 = nn.Sequential(*list(base.features[4:7]))    # -> (B, 40, 16, 16)
        self.stage16 = nn.Sequential(*list(base.features[7:13]))  # -> (B, 112, 8, 8)
        self.stage32 = nn.Sequential(*list(base.features[13:15])) # -> (B, 160, 4, 4)

        # Global & column semantic branch (4x4)
        self.proj32 = ConvBnHswish(160, 192, k=1)

        # Top-down FPN -> f8_fpn [B, 128, 16, 16]
        self.lat32 = ConvBnHswish(192, 128, k=1)
        self.lat16 = ConvBnHswish(112, 128, k=1)
        self.smooth16 = ConvBnHswish(128, 128, k=3, pad=1, groups=128)
        self.lat8 = ConvBnHswish(40, 128, k=1)
        self.edge4_to_8 = ConvBnHswish(24, 128, k=3, stride=2, pad=1)
        self.smooth8 = nn.Sequential(
            ConvBnHswish(128, 128, k=3, pad=1, groups=128),
            ConvBnHswish(128, 128, k=1),
        )
        self.fpn_drop = nn.Dropout2d(0.24)
        self.drop16 = nn.Dropout2d(0.12)
        self.drop32 = nn.Dropout2d(0.25)

        # Training-only auxiliary SR decoder (excluded from export forward!)
        self.sr_proj8 = ConvBnHswish(128, 64, k=1)
        self.sr_proj4 = ConvBnHswish(24, 64, k=1)
        self.sr_head = nn.Sequential(
            ConvBnHswish(64, 64, k=3, pad=1, groups=64),
            ConvBnHswish(64, 32, k=1),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # 6-Query 2D-Positional Cross-Attention over 16x16 = 256 tokens
        self.pos_emb = nn.Parameter(_make_2d_sincos_pos_emb(16, 16, 128) * 0.15 + torch.randn(1, 256, 128) * 0.01)
        self.key_proj = nn.Linear(128, 128)
        self.queries = nn.Parameter(torch.randn(1, 6, 128) * 0.02)  # left, right, center, leg, back, orient
        self.attn_scale = 1.0 / math.sqrt(128.0)

        l_map, r_map = build_compositional_maps()
        self.register_buffer("l_map", l_map)
        self.register_buffer("r_map", r_map)
        self.comp_alpha = nn.Parameter(torch.tensor(0.70))

        # Digit heads
        self.left_head = nn.Sequential(
            nn.Linear(128 * 3 + 192, 192),
            nn.Hardswish(),
            nn.Dropout(0.32),
            nn.Linear(192, 11),
        )
        self.right_head = nn.Sequential(
            nn.Linear(128 * 3 + 192, 192),
            nn.Hardswish(),
            nn.Dropout(0.32),
            nn.Linear(192, 11),
        )

        # Orientation head: 60.0 * tanh(...) -> [-60, +60] deg (+ = CCW)
        self.orient_head = nn.Sequential(
            nn.Linear(128 * 4 + 192, 160),
            nn.Hardswish(),
            nn.Dropout(0.20),
            nn.Linear(160, 1),
        )

        # Shared multi-task trunk: g32(192) + 4 horiz cols(768) + 6 attn feats(768) = 1728
        shared_in = 192 + 192 * 4 + 128 * 6
        self.shared_fc = nn.Sequential(
            nn.Linear(shared_in, 320),
            nn.Hardswish(),
            nn.Dropout(0.40),
        )
        self.leg_head = nn.Linear(320, 1)
        self.back_head = nn.Linear(320, 1)
        self.num_head = nn.Linear(320, NUM_CLASSES)

    def _forward_core(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = (images - self.mean) / self.std
        f4_raw = self.stage4(x)      # (B, 24, 32, 32)
        f8_raw = self.stage8(f4_raw) # (B, 40, 16, 16)
        f16_raw = self.drop16(self.stage16(f8_raw))  # (B, 112, 8, 8)
        f32 = self.proj32(self.stage32(f16_raw))  # (B, 192, 4, 4)

        p32 = self.lat32(f32)
        p16 = self.smooth16(self.lat16(f16_raw) + F.interpolate(p32, scale_factor=2.0, mode="bilinear", align_corners=False))
        f8_fpn = self.smooth8(
            self.lat8(f8_raw)
            + F.interpolate(p16, scale_factor=2.0, mode="bilinear", align_corners=False)
            + self.edge4_to_8(f4_raw)
        )  # (B, 128, 16, 16)

        b = f8_fpn.shape[0]
        f8_tok = self.fpn_drop(f8_fpn)
        tokens = f8_tok.flatten(2).transpose(1, 2) + self.pos_emb  # (B, 256, 128)
        keys = self.key_proj(tokens)                               # (B, 256, 128)
        scores = torch.matmul(self.queries.expand(b, -1, -1), keys.transpose(1, 2)) * self.attn_scale
        attn = F.softmax(scores, dim=-1)                           # (B, 6, 256)
        q_feats = torch.matmul(attn, tokens)                       # (B, 6, 128)

        q_left = q_feats[:, 0, :]
        q_right = q_feats[:, 1, :]
        q_center = q_feats[:, 2, :]
        q_leg = q_feats[:, 3, :]
        q_back = q_feats[:, 4, :]
        q_orient = q_feats[:, 5, :]

        f32_head = self.drop32(f32)
        g32 = f32_head.mean(dim=(2, 3))                  # (B, 192)
        h_cols = f32_head.mean(dim=2).reshape(b, 192 * 4)  # (B, 768)

        left_logits = self.left_head(torch.cat([q_left, q_center, q_orient, g32], dim=1))
        right_logits = self.right_head(torch.cat([q_right, q_center, q_orient, g32], dim=1))
        orientation_deg = 60.0 * torch.tanh(
            self.orient_head(torch.cat([q_orient, q_center, q_left, q_right, g32], dim=1))
        )

        shared = self.shared_fc(
            torch.cat([g32, h_cols, q_left, q_right, q_center, q_leg, q_back, q_orient], dim=1)
        )
        leg_logit = self.leg_head(shared)
        back_logit = self.back_head(shared)
        holistic_num = self.num_head(shared)

        logp_l = F.log_softmax(left_logits, dim=-1)
        logp_r = F.log_softmax(right_logits, dim=-1)
        comp_num = torch.matmul(logp_l, self.l_map) + torch.matmul(logp_r, self.r_map)
        num_logits = holistic_num + self.comp_alpha * comp_num

        return leg_logit, back_logit, num_logits, left_logits, right_logits, orientation_deg, f8_fpn, f4_raw

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference & ONNX-export forward (omits training-only SR-Aux decoder)."""
        leg_logit, back_logit, num_logits, left_logits, right_logits, orientation_deg, _, _ = self._forward_core(images)
        return leg_logit, back_logit, num_logits, left_logits, right_logits, orientation_deg

    def forward_train(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward including SR-Aux 32x32 stroke mask logits."""
        leg_logit, back_logit, num_logits, left_logits, right_logits, orientation_deg, f8_fpn, f4_raw = self._forward_core(images)
        sr_feat = self.sr_proj4(f4_raw) + F.interpolate(self.sr_proj8(f8_fpn), scale_factor=2.0, mode="bilinear", align_corners=False)
        sr_logits = self.sr_head(sr_feat)
        return leg_logit, back_logit, num_logits, left_logits, right_logits, orientation_deg, sr_logits


class CalibratedStrWrapper(nn.Module):
    """Wraps trained JerseyStrNetV5 with fitted scalar temperature scaling (6 outputs, no SR-Aux)."""

    def __init__(self, net: JerseyStrNetV5, t_leg: float, t_back: float, t_num: float, t_digit: float) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("inv_t_leg", torch.tensor(1.0 / max(0.05, t_leg), dtype=torch.float32))
        self.register_buffer("inv_t_back", torch.tensor(1.0 / max(0.05, t_back), dtype=torch.float32))
        self.register_buffer("inv_t_num", torch.tensor(1.0 / max(0.05, t_num), dtype=torch.float32))
        self.register_buffer("inv_t_digit", torch.tensor(1.0 / max(0.05, t_digit), dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        leg, back, num, left, right, orient = self.net(images)
        return (
            leg * self.inv_t_leg,
            back * self.inv_t_back,
            num * self.inv_t_num,
            left * self.inv_t_digit,
            right * self.inv_t_digit,
            orient,
        )


def load_shards(shard_dir: Path, prefix: str) -> Dict[str, np.ndarray]:
    paths = sorted(shard_dir.glob(f"{prefix}_shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shards matching {prefix}_shard_*.npz in {shard_dir}")
    buckets: Dict[str, List[np.ndarray]] = {}
    for p in paths:
        with np.load(p) as z:
            for k in z.files:
                buckets.setdefault(k, []).append(z[k])
    data = {k: np.concatenate(v, axis=0) for k, v in buckets.items()}
    print(f"[data] loaded {prefix}: {len(data['images'])} samples from {len(paths)} shards", flush=True)
    return data


def build_partial_soft_targets(
    left_d: torch.Tensor,
    right_d: torch.Tensor,
    partial_f: torch.Tensor,
) -> torch.Tensor:
    """Soft targets for partial-occlusion crops: 0.5 mass on class 0 (none) + 0.5 mass on consistent 2-digit candidates."""
    b = partial_f.shape[0]
    tgt = torch.zeros((b, NUM_CLASSES), dtype=torch.float32, device=partial_f.device)
    tgt[:, 0] = 0.50

    left_mask = partial_f == PARTIAL_LEFT_ONLY
    if left_mask.any():
        idx = torch.nonzero(left_mask, as_tuple=False).squeeze(1)
        a = left_d[idx].long()
        for row_i, d_a in zip(idx.tolist(), a.tolist()):
            if 1 <= d_a <= 9:
                cands = [d_a * 10 + d_b for d_b in range(10)]
                w = 0.50 / len(cands)
                tgt[row_i, cands] = w
            elif d_a == 0:
                tgt[row_i, 100] = 0.50

    right_mask = partial_f == PARTIAL_RIGHT_ONLY
    if right_mask.any():
        idx = torch.nonzero(right_mask, as_tuple=False).squeeze(1)
        b_d = right_d[idx].long()
        for row_i, d_b in zip(idx.tolist(), b_d.tolist()):
            cands = [d_a * 10 + d_b for d_a in range(1, 10)]
            if d_b == 0:
                cands.append(100)
            w = 0.50 / len(cands)
            tgt[row_i, cands] = w
    return tgt


def gpu_batch_augment(
    x: torch.Tensor,
    sr_t: torch.Tensor,
    y_tilt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """On-GPU per-sample affine/rot jitter, blur5/blur3/sharpen mix, DropPatch & photometric jitter."""
    b, _, h, w = x.shape
    sr_4d = sr_t.unsqueeze(1)
    # 1. Per-sample affine scale (0.90..1.10), shift (+/-4.5 px) & rotation jitter (+/-6.0 deg when |tilt|<46)
    inv_s = 1.0 / torch.empty((b,), device=x.device).uniform_(0.90, 1.10)
    tx = torch.empty((b,), device=x.device).uniform_(-0.068, 0.068)
    ty = torch.empty((b,), device=x.device).uniform_(-0.068, 0.068)
    allow_rot = (y_tilt.abs() < 46.0).float()
    delta_deg = torch.empty((b,), device=x.device).uniform_(-6.0, 6.0) * allow_rot
    rad = delta_deg * (math.pi / 180.0)
    c = torch.cos(rad)
    s = torch.sin(rad)
    theta = torch.zeros((b, 2, 3), device=x.device, dtype=torch.float32)
    theta[:, 0, 0] = inv_s * c
    theta[:, 0, 1] = -inv_s * s
    theta[:, 1, 0] = inv_s * s
    theta[:, 1, 1] = inv_s * c
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid_x = F.affine_grid(theta, x.shape, align_corners=False)
    grid_sr = F.affine_grid(theta, sr_4d.shape, align_corners=False)
    x = F.grid_sample(x, grid_x, mode="bilinear", padding_mode="border", align_corners=False)
    sr_4d = F.grid_sample(sr_4d, grid_sr, mode="bilinear", padding_mode="zeros", align_corners=False)
    y_tilt = torch.clamp(y_tilt + delta_deg, -55.0, 55.0)

    # 2. Per-sample continuous soft-telephoto blur blend / 3x3 blur / unsharp / clean mix
    blur3_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    blur5_x = F.avg_pool2d(blur3_x, kernel_size=3, stride=1, padding=1)
    alpha_b = torch.empty((b, 1, 1, 1), device=x.device).uniform_(0.12, 0.60)
    soft_mix = alpha_b * blur5_x + (1.0 - alpha_b) * blur3_x
    sharp_x = torch.clamp(1.30 * x - 0.30 * blur3_x, 0.0, 1.0)
    r_k = torch.rand((b, 1, 1, 1), device=x.device)
    x = torch.where(r_k < 0.24, soft_mix, torch.where(r_k < 0.34, blur3_x, torch.where(r_k < 0.45, sharp_x, x)))

    # 3. Per-sample random small DropPatch cutout (5..9 px) on 22% of images
    pw = torch.randint(5, 10, (b, 1, 1), device=x.device)
    ph = torch.randint(5, 10, (b, 1, 1), device=x.device)
    px0 = torch.randint(10, w - 24, (b, 1, 1), device=x.device)
    py0 = torch.randint(10, h - 24, (b, 1, 1), device=x.device)
    ys = torch.arange(h, device=x.device).view(1, h, 1)
    xs = torch.arange(w, device=x.device).view(1, 1, w)
    box = ((ys >= py0) & (ys < py0 + ph) & (xs >= px0) & (xs < px0 + pw))
    keep = (torch.rand((b, 1, 1), device=x.device) < 0.22) & box
    pad_val = LETTERBOX_PAD / 255.0
    keep_4d = keep.unsqueeze(1)
    x = torch.where(keep_4d, torch.full_like(x, pad_val), x)

    # 4. Per-channel photometric gain + contrast + bias + sensor noise
    ch_gain = torch.empty((b, 3, 1, 1), device=x.device).uniform_(0.84, 1.16)
    contrast = torch.empty((b, 1, 1, 1), device=x.device).uniform_(0.84, 1.18)
    bias = torch.empty((b, 1, 1, 1), device=x.device).uniform_(-0.055, 0.055)
    x_m = x.mean(dim=(2, 3), keepdim=True)
    x = torch.clamp((x - x_m) * contrast + x_m, 0.0, 1.0)
    x = torch.clamp(x * ch_gain + bias, 0.0, 1.0)
    noise_sigma = torch.empty((b, 1, 1, 1), device=x.device).uniform_(0.0, 0.022)
    noise_on = (torch.rand((b, 1, 1, 1), device=x.device) < 0.42).float()
    x = torch.clamp(x + torch.randn_like(x) * noise_sigma * noise_on, 0.0, 1.0)
    return x, sr_4d.squeeze(1), y_tilt


def sr_bce_dice_loss(sr_logits: torch.Tensor, sr_target: torch.Tensor) -> torch.Tensor:
    """Compute float32 BCE + Soft Dice loss against [0..1] 32x32 stroke mask."""
    logits_f = sr_logits.squeeze(1).float()
    tgt_f = sr_target.float()
    bce = F.binary_cross_entropy_with_logits(logits_f, tgt_f)
    probs = torch.sigmoid(logits_f)
    inter = (probs * tgt_f).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + tgt_f.sum(dim=(1, 2))
    dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
    return bce + dice


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    val_data: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    model.eval()
    n = len(val_data["images"])
    all_leg, all_back, all_num, all_left, all_right, all_orient = [], [], [], [], [], []
    all_sr_dice = []
    has_train_sr = hasattr(model, "forward_train") and ("sr_target" in val_data)

    for i in range(0, n, batch_size):
        xb = torch.from_numpy(val_data["images"][i:i + batch_size]).to(device=device, dtype=torch.float32) / 255.0
        xb = xb.permute(0, 3, 1, 2).contiguous()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            if has_train_sr:
                lg, bk, num, ld, rd, ori, sr_l = model.forward_train(xb)  # type: ignore[operator]
                sr_t = torch.from_numpy(val_data["sr_target"][i:i + batch_size]).to(device=device, dtype=torch.float32) / 255.0
                pos_m = sr_t.sum(dim=(1, 2)) > 0.5
                if pos_m.any():
                    sp = (torch.sigmoid(sr_l.squeeze(1)[pos_m].float()) > 0.5).float()
                    st = (sr_t[pos_m] > 0.3).float()
                    d = (2.0 * (sp * st).sum(dim=(1, 2)) + 1e-4) / (sp.sum(dim=(1, 2)) + st.sum(dim=(1, 2)) + 1e-4)
                    all_sr_dice.append(d.cpu())
            else:
                lg, bk, num, ld, rd, ori = model(xb)
        all_leg.append(lg.float().cpu())
        all_back.append(bk.float().cpu())
        all_num.append(num.float().cpu())
        all_left.append(ld.float().cpu())
        all_right.append(rd.float().cpu())
        all_orient.append(ori.float().cpu())

    logits = {
        "leg": torch.cat(all_leg, dim=0).squeeze(1),
        "back": torch.cat(all_back, dim=0).squeeze(1),
        "num": torch.cat(all_num, dim=0),
        "left": torch.cat(all_left, dim=0),
        "right": torch.cat(all_right, dim=0),
        "orient": torch.cat(all_orient, dim=0).squeeze(1),
    }

    y_num = torch.from_numpy(val_data["number_cls"]).long()
    y_left = torch.from_numpy(val_data["left_digit"]).long()
    y_right = torch.from_numpy(val_data["right_digit"]).long()
    y_leg = torch.from_numpy(val_data["legible"]).float()
    y_back = torch.from_numpy(val_data["is_back"]).float()
    y_part = torch.from_numpy(val_data["partial_flag"]).long()
    y_tilt = torch.from_numpy(val_data["tilt_deg"]).float()

    leg_prob = torch.sigmoid(logits["leg"])
    back_prob = torch.sigmoid(logits["back"])
    num_prob = F.softmax(logits["num"], dim=-1)
    num_pred = num_prob.argmax(dim=-1)
    top3 = logits["num"].topk(3, dim=-1).indices

    leg_mask = y_leg > 0.5
    pos_orient_mask = leg_mask | (y_part > 0)
    neg_mask = (y_num == 0)

    swap_lut = build_swapped_number_lut()
    swapped_gt = swap_lut[y_num]
    order_swap_mask = leg_mask & (swapped_gt >= 0)

    legible_top1 = float((num_pred[leg_mask] == y_num[leg_mask]).float().mean().item())
    legible_top3 = float((top3[leg_mask] == y_num[leg_mask].unsqueeze(1)).any(dim=-1).float().mean().item())
    order_swap_top1 = float((num_pred[order_swap_mask] == y_num[order_swap_mask]).float().mean().item())

    # Conditional digit-order accuracy among {gt, swap(gt)} (measures flip errors like #27->#72, #21->#12, #43->#34)
    n_exact = int((num_pred[order_swap_mask] == y_num[order_swap_mask]).sum().item())
    n_swapped = int((num_pred[order_swap_mask] == swapped_gt[order_swap_mask]).sum().item())
    order_swap_acc = float(n_exact / max(1, n_exact + n_swapped))

    neg_fp_rate = float(((leg_prob[neg_mask] >= 0.50) | ((num_pred[neg_mask] != 0) & (num_prob[neg_mask].max(dim=-1).values >= 0.35))).float().mean().item())
    neg_strict_fp = float((num_pred[neg_mask] != 0).float().mean().item())
    leg_acc = float(((leg_prob >= 0.50).float() == y_leg).float().mean().item())
    back_acc = float(((back_prob >= 0.50).float() == y_back).float().mean().item())

    tilt_mae_deg = float((logits["orient"][pos_orient_mask] - y_tilt[pos_orient_mask]).abs().mean().item())
    tilt_wide_m = pos_orient_mask & (y_tilt.abs() >= 18.0)
    tilt_wide_mae_deg = float((logits["orient"][tilt_wide_m] - y_tilt[tilt_wide_m]).abs().mean().item()) if tilt_wide_m.any() else 0.0

    lp_m = y_part == PARTIAL_LEFT_ONLY
    rp_m = y_part == PARTIAL_RIGHT_ONLY
    left_part_acc = float((logits["left"][lp_m].argmax(dim=-1) == y_left[lp_m]).float().mean().item()) if lp_m.any() else 0.0
    right_part_acc = float((logits["right"][rp_m].argmax(dim=-1) == y_right[rp_m]).float().mean().item()) if rp_m.any() else 0.0
    left_all_acc = float((logits["left"].argmax(dim=-1) == y_left).float().mean().item())
    right_all_acc = float((logits["right"].argmax(dim=-1) == y_right).float().mean().item())
    sr_dice = float(torch.cat(all_sr_dice, dim=0).mean().item()) if all_sr_dice else 0.0

    metrics = {
        "legible_top1": legible_top1,
        "legible_top3": legible_top3,
        "order_swap_top1": order_swap_top1,
        "order_swap_acc": order_swap_acc,
        "swap_error_count": float(n_swapped),
        "hard_neg_fp_rate": neg_fp_rate,
        "hard_neg_argmax_nonzero": neg_strict_fp,
        "legibility_acc": leg_acc,
        "back_acc": back_acc,
        "tilt_mae_deg": tilt_mae_deg,
        "tilt_wide_mae_deg": tilt_wide_mae_deg,
        "sr_dice": sr_dice,
        "left_partial_digit_acc": left_part_acc,
        "right_partial_digit_acc": right_part_acc,
        "left_digit_acc": left_all_acc,
        "right_digit_acc": right_all_acc,
    }
    return metrics, logits


def compute_ece(probs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if m.any():
            ece += (m.sum() / n) * abs(float(correct[m].mean()) - float(probs[m].mean()))
    return float(ece)


def fit_temperature_ce(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float, float, float]:
    """Fit scalar T > 0 minimizing NLL on multi-class logits; return (T, nll_before, nll_after, ece_before, ece_after)."""
    with torch.no_grad():
        p0 = F.softmax(logits, dim=-1)
        conf0, pred0 = p0.max(dim=-1)
        corr = (pred0 == targets).numpy().astype(np.float32)
        nll0 = float(F.cross_entropy(logits, targets).item())
        ece0 = compute_ece(conf0.numpy(), corr)

    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(0.1, 10.0)
        loss = F.cross_entropy(logits / t, targets)
        loss.backward()
        return loss

    opt.step(closure)
    t_val = float(torch.exp(log_t).clamp(0.1, 10.0).item())
    with torch.no_grad():
        p1 = F.softmax(logits / t_val, dim=-1)
        conf1 = p1.max(dim=-1).values
        nll1 = float(F.cross_entropy(logits / t_val, targets).item())
        ece1 = compute_ece(conf1.numpy(), corr)
    return t_val, nll0, nll1, ece0, ece1


def fit_temperature_bce(logits: torch.Tensor, targets: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=40)

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(0.1, 10.0)
        loss = F.binary_cross_entropy_with_logits(logits / t, targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).clamp(0.1, 10.0).item())


def train_model(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    ckpt_path: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    max_train_samples: Optional[int] = None,
) -> Tuple[JerseyStrNetV5, Dict[str, float]]:
    model = JerseyStrNetV5(ckpt_path).to(device)
    backbone_prefixes = ("stage4.", "stage8.", "stage16.", "stage32.")
    backbone_params = [p for n, p in model.named_parameters() if n.startswith(backbone_prefixes)]
    head_params = [p for n, p in model.named_parameters() if not n.startswith(backbone_prefixes)]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.62},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=7e-3,
    )
    n_train = len(train_data["images"]) if max_train_samples is None else min(len(train_data["images"]), max_train_samples)
    steps_per_epoch = max(1, n_train // batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr * 0.62, lr], epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=0.10
    )
    rng = np.random.default_rng(1337)
    best_score = -1e9
    best_state = None
    best_metrics: Dict[str, float] = {}
    ema_state: Optional[Dict[str, torch.Tensor]] = None
    ema_decay = 0.9985

    imgs_np = train_data["images"]
    sr_np = train_data["sr_target"]
    tilt_np = train_data["tilt_deg"]
    num_np = train_data["number_cls"]
    left_np = train_data["left_digit"]
    right_np = train_data["right_digit"]
    leg_np = train_data["legible"]
    back_np = train_data["is_back"]
    part_np = train_data["partial_flag"]

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        running_loss = 0.0
        n_batches = 0

        for i in range(0, n_train - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.from_numpy(imgs_np[idx]).to(device=device, dtype=torch.float32) / 255.0
            xb = xb.permute(0, 3, 1, 2).contiguous()
            yb_sr = torch.from_numpy(sr_np[idx]).to(device=device, dtype=torch.float32) / 255.0
            yb_tilt = torch.from_numpy(tilt_np[idx]).to(device=device, dtype=torch.float32)
            xb, yb_sr, yb_tilt = gpu_batch_augment(xb, yb_sr, yb_tilt)
            yb_num = torch.from_numpy(num_np[idx]).to(device=device, dtype=torch.long)
            yb_left = torch.from_numpy(left_np[idx]).to(device=device, dtype=torch.long)
            yb_right = torch.from_numpy(right_np[idx]).to(device=device, dtype=torch.long)
            yb_leg = torch.from_numpy(leg_np[idx]).to(device=device, dtype=torch.float32)
            yb_back = torch.from_numpy(back_np[idx]).to(device=device, dtype=torch.float32)
            yb_part = torch.from_numpy(part_np[idx]).to(device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                leg_l, back_l, num_l, ld_l, rd_l, orient_p, sr_l = model.forward_train(xb)

                loss_leg = F.binary_cross_entropy_with_logits(leg_l.squeeze(1).float(), yb_leg * 0.98 + 0.01)
                loss_back = F.binary_cross_entropy_with_logits(back_l.squeeze(1).float(), yb_back * 0.98 + 0.01)
                loss_ld = F.cross_entropy(ld_l.float(), yb_left, label_smoothing=0.055)
                loss_rd = F.cross_entropy(rd_l.float(), yb_right, label_smoothing=0.055)

                full_mask = yb_part == PARTIAL_NONE
                part_mask = yb_part > PARTIAL_NONE
                loss_num = torch.tensor(0.0, device=device)
                if full_mask.any():
                    loss_num = loss_num + F.cross_entropy(num_l[full_mask].float(), yb_num[full_mask], label_smoothing=0.055) * (
                        full_mask.float().mean()
                    )
                if part_mask.any():
                    soft_tgt = build_partial_soft_targets(yb_left[part_mask], yb_right[part_mask], yb_part[part_mask])
                    log_p = F.log_softmax(num_l[part_mask].float(), dim=-1)
                    loss_num = loss_num + F.kl_div(log_p, soft_tgt, reduction="batchmean") * (
                        part_mask.float().mean()
                    )

                loss_sr = sr_bce_dice_loss(sr_l, yb_sr)
                ori_mask = (yb_leg > 0.5) | part_mask
                if ori_mask.any():
                    loss_ori = F.smooth_l1_loss(orient_p.squeeze(1)[ori_mask].float(), yb_tilt[ori_mask], beta=3.0)
                else:
                    loss_ori = torch.tensor(0.0, device=device)

                loss = (
                    loss_num
                    + 0.65 * (loss_ld + loss_rd)
                    + 0.55 * loss_leg
                    + 0.30 * loss_back
                    + 0.60 * loss_sr
                    + 0.04 * loss_ori
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            if epoch >= 4:
                with torch.no_grad():
                    cur_sd = model.state_dict()
                    if ema_state is None:
                        ema_state = {k: v.detach().clone() for k, v in cur_sd.items()}
                    else:
                        for k, v in cur_sd.items():
                            if v.is_floating_point():
                                ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
                            else:
                                ema_state[k].copy_(v.detach())
            running_loss += float(loss.item())
            n_batches += 1

        metrics, _ = evaluate_split(model, val_data, device=device)
        def _calc_score(m: Dict[str, float]) -> float:
            return (
                m["legible_top1"]
                - 0.8 * m["hard_neg_fp_rate"]
                + 0.20 * 0.5 * (m["left_partial_digit_acc"] + m["right_partial_digit_acc"])
                - 0.012 * m["tilt_mae_deg"]
            )
        score = _calc_score(metrics)
        use_ema_tag = ""
        eval_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ema_state is not None and epoch >= 12:
            raw_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state)
            ema_metrics, _ = evaluate_split(model, val_data, device=device)
            ema_score = _calc_score(ema_metrics)
            model.load_state_dict(raw_backup)
            if ema_score > score:
                score = ema_score
                metrics = ema_metrics
                eval_state = {k: v.detach().cpu().clone() for k, v in ema_state.items()}
                use_ema_tag = " [EMA]"

        dt = time.time() - t0
        print(
            f"[v5] ep {epoch:02d}/{epochs:02d} ({dt:.1f}s) loss={running_loss/max(1,n_batches):.4f}{use_ema_tag} "
            f"top1={metrics['legible_top1']*100:.2f}% top3={metrics['legible_top3']*100:.2f}% "
            f"swapAcc={metrics['order_swap_acc']*100:.2f}% negFP={metrics['hard_neg_fp_rate']*100:.2f}% "
            f"tiltMAE={metrics['tilt_mae_deg']:.2f}d(wide={metrics['tilt_wide_mae_deg']:.2f}d) "
            f"srDice={metrics['sr_dice']:.3f} legAcc={metrics['legibility_acc']*100:.2f}% "
            f"partL/R={metrics['left_partial_digit_acc']*100:.1f}/{metrics['right_partial_digit_acc']*100:.1f}%",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_state = eval_state

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def verify_ccw_derotation_sanity(onnx_path: Path) -> Dict[str, object]:
    """Verify canonical rotation sign (+35 deg CCW '27' crop -> derotate via -orientation_deg CW -> #27)."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(777)
    H, W = 160, 160
    jersey = np.full((H, W, 3), (26, 36, 66), dtype=np.uint8)
    jersey[:28, :] = (198, 152, 54)
    patch, _ = _render_jersey_number_patch(rng, "27", W, 74, (224, 228, 234), (198, 152, 54), (26, 36, 66))
    py0 = 38
    pa = patch[:, :, 3:4].astype(np.float32) / 255.0
    jersey[py0:py0 + 74, :] = np.clip(patch[:, :, :3].astype(np.float32) * pa + jersey[py0:py0 + 74, :].astype(np.float32) * (1.0 - pa), 0, 255).astype(np.uint8)
    j_rgba = np.dstack([jersey, np.full((H, W), 255, dtype=np.uint8)])
    warped, _ = _apply_cylindrical_warp_and_folds(rng, j_rgba, 0.0, 0.80, 0.0)
    bg = np.full((H, W, 3), 192, dtype=np.uint8)
    wa = warped[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.clip(warped[:, :, :3].astype(np.float32) * wa + bg.astype(np.float32) * (1.0 - wa), 0, 255).astype(np.uint8)
    M_ccw = cv2.getRotationMatrix2D((W * 0.5, H * 0.5), 35.0, 1.0)
    tilted_35_ccw = cv2.warpAffine(bg, M_ccw, (W, H), flags=cv2.INTER_CUBIC, borderValue=(192, 192, 192))

    def _run_crop(img_rgb: np.ndarray, interp: int = cv2.INTER_CUBIC) -> Tuple[int, float, float, float]:
        inp_128 = cv2.resize(img_rgb, (TARGET_SIZE, TARGET_SIZE), interpolation=interp).astype(np.float32) / 255.0
        outs = sess.run(None, {"images": inp_128.transpose(2, 0, 1)[None]})
        leg_p = float(1.0 / (1.0 + math.exp(-float(outs[0][0, 0]))))
        num_v = outs[2][0]
        e = np.exp(num_v - num_v.max())
        p = e / e.sum()
        pred_cls = int(np.argmax(p))
        pred_prob = float(p[pred_cls])
        ori_deg = float(outs[5][0, 0])
        return pred_cls, pred_prob, leg_p, ori_deg

    res_by_interp = {}
    for iname, interp in (("cubic", cv2.INTER_CUBIC), ("linear", cv2.INTER_LINEAR)):
        p1_cls, p1_prob, p1_leg, p1_ori = _run_crop(tilted_35_ccw, interp=interp)
        M_derot = cv2.getRotationMatrix2D((W * 0.5, H * 0.5), -p1_ori, 1.0)
        derotated = cv2.warpAffine(tilted_35_ccw, M_derot, (W, H), flags=interp, borderValue=(LETTERBOX_PAD, LETTERBOX_PAD, LETTERBOX_PAD))
        p2_cls, p2_prob, p2_leg, p2_ori = _run_crop(derotated, interp=interp)
        res_by_interp[iname] = {
            "pass1_pred_cls": p1_cls,
            "pass1_prob": round(p1_prob, 4),
            "pass1_leg": round(p1_leg, 4),
            "pass1_orientation_deg": round(p1_ori, 2),
            "pass2_derot_angle_opencv": round(-p1_ori, 2),
            "pass2_pred_cls": p2_cls,
            "pass2_prob": round(p2_prob, 4),
            "pass2_leg": round(p2_leg, 4),
            "pass2_residual_orientation_deg": round(p2_ori, 2),
        }
        assert p1_ori > 18.0, f"[{iname}] Expected positive CCW orientation_deg near +35, got {p1_ori:.2f}"
        assert p2_cls == 27, f"[{iname}] Expected derotated prediction == 27, got {p2_cls}"
        assert abs(p2_ori) < 12.0, f"[{iname}] Expected residual orientation after derotation near 0, got {p2_ori:.2f}"

    res = {"input_true_tilt_deg": 35.0, **res_by_interp["cubic"], "linear_check": res_by_interp["linear"]}
    print(f"[sanity_ccw35] {json.dumps(res)}")
    return res


def export_and_verify_onnx(
    model: JerseyStrNetV5,
    val_data: Dict[str, np.ndarray],
    device: torch.device,
    onnx_path: Path,
    json_path: Path,
    summary_metrics: Dict[str, object],
) -> None:
    model.eval()
    _, val_logits = evaluate_split(model, val_data, device=device)

    y_num = torch.from_numpy(val_data["number_cls"]).long()
    y_left = torch.from_numpy(val_data["left_digit"]).long()
    y_right = torch.from_numpy(val_data["right_digit"]).long()
    y_leg = torch.from_numpy(val_data["legible"]).float()
    y_back = torch.from_numpy(val_data["is_back"]).float()
    y_part = torch.from_numpy(val_data["partial_flag"]).long()

    full_m = y_part == PARTIAL_NONE
    t_num_fit, nll0, nll1, ece0, ece1 = fit_temperature_ce(val_logits["num"][full_m], y_num[full_m])
    t_leg_fit = fit_temperature_bce(val_logits["leg"], y_leg)
    t_back_fit = fit_temperature_bce(val_logits["back"], y_back)
    t_l, _, _, _, _ = fit_temperature_ce(val_logits["left"], y_left)
    t_r, _, _, _, _ = fit_temperature_ce(val_logits["right"], y_right)
    t_digit_fit = 0.5 * (t_l + t_r)

    t_num = float(np.clip(t_num_fit, 0.94, 1.30))
    t_digit = float(np.clip(t_digit_fit, 0.94, 1.30))
    t_leg = float(np.clip(t_leg_fit, 0.94, 1.25))
    t_back = float(np.clip(t_back_fit, 0.94, 1.25))
    print(
        f"[calib] fit T_num={t_num_fit:.4f}->exp {t_num:.4f} (NLL {nll0:.4f}->{nll1:.4f}, ECE {ece0:.4f}->{ece1:.4f}) "
        f"T_leg={t_leg_fit:.4f}->exp {t_leg:.4f} T_back={t_back_fit:.4f}->exp {t_back:.4f} T_digit={t_digit_fit:.4f}->exp {t_digit:.4f}"
    )

    wrapper = CalibratedStrWrapper(model.cpu(), t_leg=t_leg, t_back=t_back, t_num=t_num, t_digit=t_digit).eval()
    dummy = torch.rand(1, 3, TARGET_SIZE, TARGET_SIZE, dtype=torch.float32)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        opset_version=17,
        dynamo=False,
        input_names=["images"],
        output_names=[
            "legibility_logit",
            "back_logit",
            "number_logits",
            "left_digit_logits",
            "right_digit_logits",
            "orientation_deg",
        ],
        do_constant_folding=True,
    )
    om = onnx.load(str(onnx_path))
    onnx.checker.check_model(om)
    op_types = sorted({n.op_type for n in om.graph.node})
    assert "GridSample" not in op_types, f"Forbidden GridSample found in ONNX graph: {op_types}"
    print(f"[onnx] ops verified (no GridSample): {op_types}")

    size_bytes = onnx_path.stat().st_size
    assert size_bytes <= 15 * 1024 * 1024, f"ONNX size {size_bytes} exceeds 15 MB!"

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    with torch.no_grad():
        test_x = torch.from_numpy(val_data["images"][:1]).float() / 255.0
        test_x = test_x.permute(0, 3, 1, 2).contiguous()
        pt_outs = [o.numpy() for o in wrapper(test_x)]
    ort_outs = sess.run(None, {"images": test_x.numpy()})
    max_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(pt_outs, ort_outs))

    inp = test_x.numpy()
    for _ in range(30):
        sess.run(None, {"images": inp})
    t0 = time.perf_counter()
    for _ in range(300):
        sess.run(None, {"images": inp})
    cpu_ms = (time.perf_counter() - t0) / 300.0 * 1000.0
    print(f"[onnx] exported {onnx_path.name}: {size_bytes / 1e6:.2f} MB ({size_bytes} B), opset={om.opset_import[0].version}, max_diff={max_diff:.2e}, ORT_CPU_2thread={cpu_ms:.3f} ms/crop")

    final_metrics, _ = evaluate_split(wrapper.to(device), val_data, device=device)
    sanity_ccw = verify_ccw_derotation_sanity(onnx_path)

    meta = {
        "version": 5,
        "model_name": "coach_jersey_str",
        "architecture": "MobileNetV3Large_FPN_SR_Aux_TrainOnly_6Query2DCrossAttn_Compositional_Orientation",
        "input_size": TARGET_SIZE,
        "opset": int(om.opset_import[0].version),
        "size_bytes": int(size_bytes),
        "size_mb": round(size_bytes / 1e6, 3),
        "xnnpack_safe_ops": op_types,
        "input": {
            "name": "images",
            "dtype": "float32",
            "shape": [1, 3, TARGET_SIZE, TARGET_SIZE],
            "layout": "NCHW_RGB_0_TO_1",
            "normalization_embedded": True,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
            "letterbox_pad_rgb": [LETTERBOX_PAD, LETTERBOX_PAD, LETTERBOX_PAD],
            "recommended_torso_crop_rel": [-0.06, 0.04, 1.06, 0.58],
        },
        "outputs": {
            "legibility_logit": [1, 1],
            "back_logit": [1, 1],
            "number_logits": [1, 101],
            "left_digit_logits": [1, 11],
            "right_digit_logits": [1, 11],
            "orientation_deg": [1, 1],
        },
        "orientation_convention": {
            "sign": "+theta > 0 means CCW rotation in image space (cv2.getRotationMatrix2D(center, +theta, 1.0))",
            "python_derotate": "cv2.getRotationMatrix2D(center, -orientation_deg, 1.0)",
            "android_derotate": "matrix.postRotate(+orientationDeg, cx, cy)",
            "range_deg": [-60.0, 60.0],
        },
        "class_encoding": {
            "number_logits": "0=illegible/none, 1..99=jersey 1..99, 100=\"00\"",
            "left_digit_logits": "0..9=tens_or_only_digit, 10=blank",
            "right_digit_logits": "0..9=ones_digit, 10=blank_or_single",
            "prior": "uniform_1_to_99_plus_00_roster_free",
        },
        "calibration": {
            "temperature_baked_in_onnx": True,
            "effective_temperature": 1.0,
            "fitted_raw_temperatures": {
                "number_logits": round(t_num, 4),
                "legibility_logit": round(t_leg, 4),
                "back_logit": round(t_back, 4),
                "digit_logits": round(t_digit, 4),
            },
            "val_nll_before": round(nll0, 4),
            "val_nll_after": round(nll1, 4),
            "val_ece_before": round(ece0, 4),
            "val_ece_after": round(ece1, 4),
        },
        "synthetic_validation_metrics": {k: round(v, 4) for k, v in final_metrics.items()},
        "benchmark": {
            "onnxruntime_cpu_2thread_ms": round(cpu_ms, 3),
            "pytorch_onnx_max_abs_diff": float(f"{max_diff:.2e}"),
        },
        "ccw35_derotation_sanity": sanity_ccw,
        "summary": summary_metrics,
    }
    json_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[onnx] wrote metadata to {json_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, default=Path("/home/windgone/.gemini/antigravity/brain/d4f6c85d-b213-4653-af5c-c83e4f23795d/scratch/v5_str/shards"))
    ap.add_argument("--ckpt", type=Path, default=Path.home() / ".cache/torch/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth")
    ap.add_argument("--onnx-out", type=Path, default=Path("/home/windgone/workspace/eng-test-bench-pinkseven97/android/app/src/main/assets/coach_jersey_str.onnx"))
    ap.add_argument("--json-out", type=Path, default=Path("/home/windgone/workspace/eng-test-bench-pinkseven97/android/app/src/main/assets/coach_jersey_str.json"))
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    torch.manual_seed(20261120)
    np.random.seed(20261120)

    if args.smoke_test:
        m = JerseyStrNetV5(args.ckpt).to(device)
        x = torch.rand(4, 3, TARGET_SIZE, TARGET_SIZE, device=device)
        outs = m.forward_train(x)
        print("[smoke] train shapes:", [tuple(o.shape) for o in outs])
        inf_outs = m(x)
        print("[smoke] infer shapes:", [tuple(o.shape) for o in inf_outs])
        tmp_onnx = Path("/home/windgone/.gemini/antigravity/brain/d4f6c85d-b213-4653-af5c-c83e4f23795d/scratch/v5_str/smoke.onnx")
        tmp_onnx.parent.mkdir(parents=True, exist_ok=True)
        w = CalibratedStrWrapper(m.cpu(), 1.1, 1.1, 1.1, 1.1).eval()
        torch.onnx.export(
            w, torch.rand(1, 3, TARGET_SIZE, TARGET_SIZE), str(tmp_onnx),
            opset_version=17, dynamo=False, input_names=["images"],
            output_names=["legibility_logit", "back_logit", "number_logits", "left_digit_logits", "right_digit_logits", "orientation_deg"],
            do_constant_folding=True,
        )
        om = onnx.load(str(tmp_onnx))
        onnx.checker.check_model(om)
        ops = sorted({n.op_type for n in om.graph.node})
        assert "GridSample" not in ops
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        sess = ort.InferenceSession(str(tmp_onnx), opts, providers=["CPUExecutionProvider"])
        inp = np.random.rand(1, 3, TARGET_SIZE, TARGET_SIZE).astype(np.float32)
        for _ in range(20):
            sess.run(None, {"images": inp})
        t0 = time.perf_counter()
        for _ in range(200):
            sess.run(None, {"images": inp})
        ms = (time.perf_counter() - t0) / 200.0 * 1000.0
        print(f"[smoke] size={tmp_onnx.stat().st_size/1e6:.2f} MB, ops={ops}, ORT_CPU_2thread={ms:.3f} ms/crop")
        tmp_onnx.unlink(missing_ok=True)
        return

    train_data = load_shards(args.shard_dir, "train")
    val_data = load_shards(args.shard_dir, "val")

    print("\n=== Training JerseyStrNetV5 (128x128 FPN + SR-Aux + 6-Query 2D Cross-Attn + Orientation) ===")
    model, best_metrics = train_model(
        train_data, val_data, args.ckpt, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        max_train_samples=args.max_train_samples,
    )
    summary = {"best_v5_val": {k: round(v, 4) for k, v in best_metrics.items()}}
    export_and_verify_onnx(model, val_data, device, args.onnx_out, args.json_out, summary)


if __name__ == "__main__":
    main()
