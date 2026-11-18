"""Train & export roster-free hockey jersey STR model (`coach_jersey_str.onnx`) on RTX 5090.

Architecture (`JerseyStrNet`):
  - Pretrained MobileNetV3-Large backbone (`features[0..14]`, loaded from offline cache)
  - Multi-scale spatial representation: stride-16 (6x6x112) tokens + stride-32 (3x3x160) context
  - 4 learned spatial-attention queries (left, right, center, legible) over 36 positional tokens
  - Compositional digit-to-number fusion (left_digit + right_digit log-probs + holistic residual)
  - Embedded RGB [0..1] -> ImageNet normalization and calibrated temperature scaling inside ONNX

Zero test leakage: trains & validates strictly on synthetic shards from `jersey_synth.py`.
Uniform 1..99 + "00" class prior (100% roster-free).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from jersey_synth import DIGIT_BLANK, NUM_CLASSES, PARTIAL_LEFT_ONLY, PARTIAL_NONE, PARTIAL_RIGHT_ONLY


def build_compositional_maps() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (L_map [11, 101], R_map [11, 101]) mapping digit log-probs to 101 number classes."""
    l_map = torch.zeros(11, NUM_CLASSES, dtype=torch.float32)
    r_map = torch.zeros(11, NUM_CLASSES, dtype=torch.float32)
    # 0 = none/illegible -> (blank, blank)
    l_map[DIGIT_BLANK, 0] = 1.0
    r_map[DIGIT_BLANK, 0] = 1.0
    # 1..9 -> (d, blank)
    for d in range(1, 10):
        l_map[d, d] = 1.0
        r_map[DIGIT_BLANK, d] = 1.0
    # 10..99 -> (tens, ones)
    for n in range(10, 100):
        l_map[n // 10, n] = 1.0
        r_map[n % 10, n] = 1.0
    # 100 = "00" -> (0, 0)
    l_map[0, 100] = 1.0
    r_map[0, 100] = 1.0
    return l_map, r_map


class JerseyStrNet(nn.Module):
    """Multi-scale spatial-attention + compositional jersey STR network."""

    def __init__(self, ckpt_path: Path, use_spatial_comp: bool = True) -> None:
        super().__init__()
        self.use_spatial_comp = use_spatial_comp
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        base = torchvision.models.mobilenet_v3_large(weights=None)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            base.load_state_dict(state)
        else:
            print(f"[warn] checkpoint not found at {ckpt_path}, using random init")

        self.stage16 = nn.Sequential(*list(base.features[:13]))  # -> (B, 112, 6, 6)
        self.stage32 = nn.Sequential(*list(base.features[13:15]))  # -> (B, 160, 3, 3)

        self.proj32 = nn.Sequential(
            nn.Conv2d(160, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.Hardswish(),
        )

        if use_spatial_comp:
            self.proj16 = nn.Sequential(
                nn.Conv2d(112, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.Hardswish(),
            )
            self.pos_emb = nn.Parameter(torch.randn(1, 36, 128) * 0.02)
            self.key_proj = nn.Linear(128, 128)
            self.queries = nn.Parameter(torch.randn(1, 4, 128) * 0.02)  # left, right, center, legible
            self.attn_scale = 1.0 / math.sqrt(128.0)

            l_map, r_map = build_compositional_maps()
            self.register_buffer("l_map", l_map)
            self.register_buffer("r_map", r_map)
            self.comp_alpha = nn.Parameter(torch.tensor(0.65))

            self.left_head = nn.Sequential(
                nn.Linear(128 * 2 + 192, 160),
                nn.Hardswish(),
                nn.Dropout(0.15),
                nn.Linear(160, 11),
            )
            self.right_head = nn.Sequential(
                nn.Linear(128 * 2 + 192, 160),
                nn.Hardswish(),
                nn.Dropout(0.15),
                nn.Linear(160, 11),
            )
            shared_in = 192 * 4 + 128 * 4  # g32 + 3 horiz cols + 4 attn tokens = 1280
        else:
            shared_in = 192
            self.left_head = nn.Sequential(nn.Linear(192, 128), nn.Hardswish(), nn.Linear(128, 11))
            self.right_head = nn.Sequential(nn.Linear(192, 128), nn.Hardswish(), nn.Linear(128, 11))

        self.shared_fc = nn.Sequential(
            nn.Linear(shared_in, 288),
            nn.Hardswish(),
            nn.Dropout(0.20),
        )
        self.leg_head = nn.Linear(288, 1)
        self.back_head = nn.Linear(288, 1)
        self.num_head = nn.Linear(288, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = (images - self.mean) / self.std
        f16_raw = self.stage16(x)            # (B, 112, 6, 6)
        f32 = self.proj32(self.stage32(f16_raw))  # (B, 192, 3, 3)
        g32 = f32.mean(dim=(2, 3))           # (B, 192)

        if not self.use_spatial_comp:
            feat = self.shared_fc(g32)
            return (
                self.leg_head(feat),
                self.back_head(feat),
                self.num_head(feat),
                self.left_head(g32),
                self.right_head(g32),
            )

        f16 = self.proj16(f16_raw)  # (B, 128, 6, 6)
        b = f16.shape[0]
        tokens = f16.flatten(2).transpose(1, 2) + self.pos_emb  # (B, 36, 128)
        keys = self.key_proj(tokens)                            # (B, 36, 128)
        scores = torch.matmul(self.queries.expand(b, -1, -1), keys.transpose(1, 2)) * self.attn_scale
        attn = F.softmax(scores, dim=-1)                        # (B, 4, 36)
        q_feats = torch.matmul(attn, tokens)                    # (B, 4, 128)

        q_left = q_feats[:, 0, :]
        q_right = q_feats[:, 1, :]
        q_center = q_feats[:, 2, :]
        q_leg = q_feats[:, 3, :]

        left_logits = self.left_head(torch.cat([q_left, q_center, g32], dim=1))
        right_logits = self.right_head(torch.cat([q_right, q_center, g32], dim=1))

        h_cols = f32.mean(dim=2).reshape(b, 192 * 3)  # (B, 576)
        shared = self.shared_fc(torch.cat([g32, h_cols, q_left, q_right, q_center, q_leg], dim=1))

        leg_logit = self.leg_head(shared)
        back_logit = self.back_head(shared)
        holistic_num = self.num_head(shared)

        logp_l = F.log_softmax(left_logits, dim=-1)
        logp_r = F.log_softmax(right_logits, dim=-1)
        comp_num = torch.matmul(logp_l, self.l_map) + torch.matmul(logp_r, self.r_map)
        num_logits = holistic_num + self.comp_alpha * comp_num

        return leg_logit, back_logit, num_logits, left_logits, right_logits


class CalibratedStrWrapper(nn.Module):
    """Wraps trained JerseyStrNet with fitted scalar temperature scaling for ONNX export."""

    def __init__(self, net: JerseyStrNet, t_leg: float, t_back: float, t_num: float, t_digit: float) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("inv_t_leg", torch.tensor(1.0 / max(0.05, t_leg), dtype=torch.float32))
        self.register_buffer("inv_t_back", torch.tensor(1.0 / max(0.05, t_back), dtype=torch.float32))
        self.register_buffer("inv_t_num", torch.tensor(1.0 / max(0.05, t_num), dtype=torch.float32))
        self.register_buffer("inv_t_digit", torch.tensor(1.0 / max(0.05, t_digit), dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        leg, back, num, left, right = self.net(images)
        return (
            leg * self.inv_t_leg,
            back * self.inv_t_back,
            num * self.inv_t_num,
            left * self.inv_t_digit,
            right * self.inv_t_digit,
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
                tgt[row_i, 100] = 0.50  # "00"

    right_mask = partial_f == PARTIAL_RIGHT_ONLY
    if right_mask.any():
        idx = torch.nonzero(right_mask, as_tuple=False).squeeze(1)
        b_d = right_d[idx].long()
        for row_i, d_b in zip(idx.tolist(), b_d.tolist()):
            cands = [d_a * 10 + d_b for d_a in range(1, 10)]
            if d_b == 0:
                cands.append(100)  # "00"
            w = 0.50 / len(cands)
            tgt[row_i, cands] = w
    return tgt


def gpu_batch_augment(x: torch.Tensor) -> torch.Tensor:
    """Fast mild on-GPU photometric jitter (preserves left/right digit orientation)."""
    b = x.shape[0]
    scale = torch.empty((b, 1, 1, 1), device=x.device).uniform_(0.90, 1.10)
    bias = torch.empty((b, 1, 1, 1), device=x.device).uniform_(-0.04, 0.04)
    x = torch.clamp(x * scale + bias, 0.0, 1.0)
    if torch.rand(1).item() < 0.35:
        noise = torch.randn_like(x) * 0.015
        x = torch.clamp(x + noise, 0.0, 1.0)
    return x


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    val_data: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    model.eval()
    n = len(val_data["images"])
    all_leg, all_back, all_num, all_left, all_right = [], [], [], [], []
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(val_data["images"][i:i + batch_size]).to(device=device, dtype=torch.float32) / 255.0
        xb = xb.permute(0, 3, 1, 2).contiguous()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            lg, bk, num, ld, rd = model(xb)
        all_leg.append(lg.float().cpu())
        all_back.append(bk.float().cpu())
        all_num.append(num.float().cpu())
        all_left.append(ld.float().cpu())
        all_right.append(rd.float().cpu())

    logits = {
        "leg": torch.cat(all_leg, dim=0).squeeze(1),
        "back": torch.cat(all_back, dim=0).squeeze(1),
        "num": torch.cat(all_num, dim=0),
        "left": torch.cat(all_left, dim=0),
        "right": torch.cat(all_right, dim=0),
    }

    y_num = torch.from_numpy(val_data["number_cls"]).long()
    y_left = torch.from_numpy(val_data["left_digit"]).long()
    y_right = torch.from_numpy(val_data["right_digit"]).long()
    y_leg = torch.from_numpy(val_data["legible"]).float()
    y_back = torch.from_numpy(val_data["is_back"]).float()
    y_part = torch.from_numpy(val_data["partial_flag"]).long()

    leg_prob = torch.sigmoid(logits["leg"])
    back_prob = torch.sigmoid(logits["back"])
    num_prob = F.softmax(logits["num"], dim=-1)
    num_pred = num_prob.argmax(dim=-1)
    top3 = logits["num"].topk(3, dim=-1).indices

    leg_mask = y_leg > 0.5
    neg_mask = (y_num == 0)
    part_mask = y_part > 0
    order_swap_mask = leg_mask & (y_num >= 10) & (y_num <= 99) & ((y_num // 10) != (y_num % 10))

    legible_top1 = float((num_pred[leg_mask] == y_num[leg_mask]).float().mean().item())
    legible_top3 = float((top3[leg_mask] == y_num[leg_mask].unsqueeze(1)).any(dim=-1).float().mean().item())
    order_swap_acc = float((num_pred[order_swap_mask] == y_num[order_swap_mask]).float().mean().item())
    # Hard-negative FP: predicts a jersey number != 0 with >= 0.35 confidence OR legibility >= 0.50
    neg_fp_rate = float(((leg_prob[neg_mask] >= 0.50) | ((num_pred[neg_mask] != 0) & (num_prob[neg_mask].max(dim=-1).values >= 0.35))).float().mean().item())
    neg_strict_fp = float((num_pred[neg_mask] != 0).float().mean().item())
    leg_acc = float(((leg_prob >= 0.50).float() == y_leg).float().mean().item())
    back_acc = float(((back_prob >= 0.50).float() == y_back).float().mean().item())

    # Partial visible digit recall
    lp_m = y_part == PARTIAL_LEFT_ONLY
    rp_m = y_part == PARTIAL_RIGHT_ONLY
    left_part_acc = float((logits["left"][lp_m].argmax(dim=-1) == y_left[lp_m]).float().mean().item()) if lp_m.any() else 0.0
    right_part_acc = float((logits["right"][rp_m].argmax(dim=-1) == y_right[rp_m]).float().mean().item()) if rp_m.any() else 0.0
    left_all_acc = float((logits["left"].argmax(dim=-1) == y_left).float().mean().item())
    right_all_acc = float((logits["right"].argmax(dim=-1) == y_right).float().mean().item())

    metrics = {
        "legible_top1": legible_top1,
        "legible_top3": legible_top3,
        "order_swap_top1": order_swap_acc,
        "hard_neg_fp_rate": neg_fp_rate,
        "hard_neg_argmax_nonzero": neg_strict_fp,
        "legibility_acc": leg_acc,
        "back_acc": back_acc,
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
    use_spatial_comp: bool = True,
    max_train_samples: int | None = None,
) -> Tuple[JerseyStrNet, Dict[str, float]]:
    model = JerseyStrNet(ckpt_path, use_spatial_comp=use_spatial_comp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    n_train = len(train_data["images"]) if max_train_samples is None else min(len(train_data["images"]), max_train_samples)
    steps_per_epoch = max(1, n_train // batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=0.12
    )
    rng = np.random.default_rng(1337)
    best_score = -1.0
    best_state = None
    best_metrics: Dict[str, float] = {}

    imgs_np = train_data["images"]
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
            xb = gpu_batch_augment(xb)

            yb_num = torch.from_numpy(num_np[idx]).to(device=device, dtype=torch.long)
            yb_left = torch.from_numpy(left_np[idx]).to(device=device, dtype=torch.long)
            yb_right = torch.from_numpy(right_np[idx]).to(device=device, dtype=torch.long)
            yb_leg = torch.from_numpy(leg_np[idx]).to(device=device, dtype=torch.float32)
            yb_back = torch.from_numpy(back_np[idx]).to(device=device, dtype=torch.float32)
            yb_part = torch.from_numpy(part_np[idx]).to(device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                leg_l, back_l, num_l, ld_l, rd_l = model(xb)

                loss_leg = F.binary_cross_entropy_with_logits(leg_l.squeeze(1), yb_leg * 0.98 + 0.01)
                loss_back = F.binary_cross_entropy_with_logits(back_l.squeeze(1), yb_back * 0.98 + 0.01)
                loss_ld = F.cross_entropy(ld_l, yb_left, label_smoothing=0.05)
                loss_rd = F.cross_entropy(rd_l, yb_right, label_smoothing=0.05)

                full_mask = yb_part == PARTIAL_NONE
                part_mask = yb_part > PARTIAL_NONE
                loss_num = torch.tensor(0.0, device=device)
                if full_mask.any():
                    loss_num = loss_num + F.cross_entropy(num_l[full_mask], yb_num[full_mask], label_smoothing=0.05) * (
                        full_mask.float().mean()
                    )
                if part_mask.any():
                    soft_tgt = build_partial_soft_targets(yb_left[part_mask], yb_right[part_mask], yb_part[part_mask])
                    log_p = F.log_softmax(num_l[part_mask].float(), dim=-1)
                    loss_num = loss_num + F.kl_div(log_p, soft_tgt, reduction="batchmean") * (
                        part_mask.float().mean()
                    )

                loss = loss_num + 0.60 * (loss_ld + loss_rd) + 0.55 * loss_leg + 0.30 * loss_back

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            n_batches += 1

        metrics, _ = evaluate_split(model, val_data, device=device)
        score = metrics["legible_top1"] - 0.8 * metrics["hard_neg_fp_rate"] + 0.25 * (
            0.5 * (metrics["left_partial_digit_acc"] + metrics["right_partial_digit_acc"])
        )
        dt = time.time() - t0
        tag = "spatial_comp" if use_spatial_comp else "gap_baseline"
        print(
            f"[{tag}] ep {epoch:02d}/{epochs:02d} ({dt:.1f}s) loss={running_loss/max(1,n_batches):.4f} "
            f"top1={metrics['legible_top1']*100:.2f}% top3={metrics['legible_top3']*100:.2f}% "
            f"negFP={metrics['hard_neg_fp_rate']*100:.2f}% legAcc={metrics['legibility_acc']*100:.2f}% "
            f"partL/R={metrics['left_partial_digit_acc']*100:.1f}/{metrics['right_partial_digit_acc']*100:.1f}%",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def export_and_verify_onnx(
    model: JerseyStrNet,
    val_data: Dict[str, np.ndarray],
    device: torch.device,
    onnx_path: Path,
    json_path: Path,
    ablation_summary: Dict[str, object],
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
    # Enforce conservative OOD calibration floor T >= 1.08 (digits/number) and >= 1.05 (leg/back)
    # so label-smoothing T<1 fits never sharpen blurry far-ice crops into 0.99 false positives.
    t_num = max(1.08, t_num_fit)
    t_digit = max(1.08, t_digit_fit)
    t_leg = max(1.05, t_leg_fit)
    t_back = max(1.05, t_back_fit)
    print(f"[calib] fit T_num={t_num_fit:.4f}->exp {t_num:.4f} (NLL {nll0:.4f}->{nll1:.4f}, ECE {ece0:.4f}->{ece1:.4f}) T_leg={t_leg_fit:.4f}->exp {t_leg:.4f} T_back={t_back_fit:.4f}->exp {t_back:.4f} T_digit={t_digit_fit:.4f}->exp {t_digit:.4f}")

    wrapper = CalibratedStrWrapper(model.cpu(), t_leg=t_leg, t_back=t_back, t_num=t_num, t_digit=t_digit).eval()
    dummy = torch.rand(1, 3, 96, 96, dtype=torch.float32)

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
        ],
        do_constant_folding=True,
    )
    om = onnx.load(str(onnx_path))
    onnx.checker.check_model(om)
    size_bytes = onnx_path.stat().st_size
    assert size_bytes <= 12 * 1024 * 1024, f"ONNX size {size_bytes} exceeds 12 MB!"

    # Verify numerical parity and CPU latency via onnxruntime
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
    meta = {
        "model_name": "coach_jersey_str",
        "architecture": "MobileNetV3Large_0_14_MultiScale_SpatialAttention_Compositional",
        "opset": int(om.opset_import[0].version),
        "size_bytes": int(size_bytes),
        "size_mb": round(size_bytes / 1e6, 3),
        "input": {
            "name": "images",
            "dtype": "float32",
            "shape": [1, 3, 96, 96],
            "layout": "NCHW_RGB_0_TO_1",
            "normalization_embedded": True,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
            "letterbox_pad_rgb": [114, 114, 114],
            "recommended_torso_crop_rel": [-0.06, 0.10, 1.06, 0.62],
        },
        "outputs": {
            "legibility_logit": [1, 1],
            "back_logit": [1, 1],
            "number_logits": [1, 101],
            "left_digit_logits": [1, 11],
            "right_digit_logits": [1, 11],
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
        "ablations": ablation_summary,
    }
    json_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[onnx] wrote metadata to {json_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, default=Path("/home/windgone/.gemini/antigravity/brain/d4f6c85d-b213-4653-af5c-c83e4f23795d/scratch/coachbox_v2/synth_shards"))
    ap.add_argument("--ckpt", type=Path, default=Path.home() / ".cache/torch/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth")
    ap.add_argument("--onnx-out", type=Path, default=Path("/home/windgone/workspace/eng-test-bench-pinkseven97/android/app/src/main/assets/coach_jersey_str.onnx"))
    ap.add_argument("--json-out", type=Path, default=Path("/home/windgone/workspace/eng-test-bench-pinkseven97/android/app/src/main/assets/coach_jersey_str.json"))
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--ablation-epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.4e-3)
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    torch.manual_seed(20261118)
    np.random.seed(20261118)

    train_data = load_shards(args.shard_dir, "train")
    val_data = load_shards(args.shard_dir, "val")

    ablation_summary: Dict[str, object] = {}
    if not args.skip_ablation:
        print("\n=== Ablation A1: Global-Avg-Pool Baseline (160k, 10 epochs) ===")
        _, base_metrics = train_model(
            train_data, val_data, args.ckpt, device,
            epochs=args.ablation_epochs, batch_size=args.batch_size, lr=args.lr,
            use_spatial_comp=False, max_train_samples=None,
        )
        ablation_summary["gap_baseline_160k"] = {k: round(v, 4) for k, v in base_metrics.items()}

    print("\n=== Full Proposed Model: Multi-Scale 6x6 Spatial Attention + Compositional STR (160k) ===")
    model, best_metrics = train_model(
        train_data, val_data, args.ckpt, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        use_spatial_comp=True, max_train_samples=None,
    )
    ablation_summary["proposed_spatial_comp_160k"] = {k: round(v, 4) for k, v in best_metrics.items()}

    export_and_verify_onnx(model, val_data, device, args.onnx_out, args.json_out, ablation_summary)


if __name__ == "__main__":
    main()
