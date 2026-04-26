"""DA3-Metric finetune training loop.

Single-GPU (1xH200 bf16) training. No PyTorch-Lightning — just a clean
PyTorch loop that's easy to read and modify.

Stages:
  1. Phase A (frozen backbone, decoder warmup): few epochs at higher LR
  2. Phase B (full unfreeze, lower backbone LR): rest of training

Usage:
  python train/train.py --config configs/metric_kitti.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# Make sure repo root is on sys.path so 'eval' / 'train' both import.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.datasets import KITTIEigenDataset, NYUv2Dataset, collate_pad
from train.losses import MetricDepthLoss
from train.model_wrapper import DA3MetricTrainable
from eval.metrics import compute_depth_metrics, kitti_eigen_crop


# ---------------------------------------------------------------------------
def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_lr(step: int, total: int, base_lr: float, warmup: int = 500, min_lr: float = 0.0) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
@torch.no_grad()
def quick_eval_kitti(model: DA3MetricTrainable, ds, n: int, device: str, max_depth: float) -> dict:
    """Mini eval on first n KITTI samples (no Garg crop for speed; use full)."""
    model.eval()
    agg = {k: 0.0 for k in ("abs_rel", "d1", "rmse")}
    n_pix = 0
    for i in range(n):
        b = ds[i]
        img = b["image"].unsqueeze(0).to(device)
        K = b["K"].unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                             dtype=torch.bfloat16):
            pred, _ = model.train_forward(img, K)
        pred = pred[0].float().cpu().numpy()
        gt = b["depth"].numpy()
        mask = b["mask"].numpy()
        H, W = gt.shape
        # apply Garg crop on KITTI to compare against eval
        if b["src"] == "kitti":
            sy, sx = kitti_eigen_crop(H, W)
            crop = np.zeros_like(mask, dtype=bool)
            crop[sy, sx] = True
            mask = mask & crop
        m = compute_depth_metrics(pred, gt, mask)
        if m["n"] == 0:
            continue
        for k in agg:
            agg[k] += m[k] * m["n"]
        n_pix += m["n"]
    if n_pix == 0:
        return {k: float("nan") for k in agg}
    for k in agg:
        agg[k] /= n_pix
    model.train()
    return agg


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="50 steps + early stop")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpts").mkdir(exist_ok=True)
    print(f"[cfg] -> {out_dir}")
    print(json.dumps(cfg, indent=2))

    device = cfg.get("device", "cuda")
    process_res = int(cfg.get("process_res", 504))

    # ---- data ----
    print("[data] building datasets ...")
    ds_list = []
    if "kitti" in cfg.get("data", {}):
        kc = cfg["data"]["kitti"]
        ds_list.append(KITTIEigenDataset(
            data_root=kc["data_root"],
            split_file=kc["split_file"],
            process_res=process_res,
            min_depth=kc.get("min_depth", 1e-3),
            max_depth=kc.get("max_depth", 80.0),
            augment=True,
        ))
        print(f"  KITTI train: {len(ds_list[-1])} samples")
    if "nyu" in cfg.get("data", {}):
        nc = cfg["data"]["nyu"]
        ds_list.append(NYUv2Dataset(
            data_root=nc["data_root"],
            split_file=nc.get("split_file"),
            process_res=process_res,
            min_depth=nc.get("min_depth", 1e-3),
            max_depth=nc.get("max_depth", 10.0),
            augment=True,
        ))
        print(f"  NYU   train: {len(ds_list[-1])} samples")
    assert ds_list, "no datasets configured"

    # For now, train on a single dataset at a time (avoids variable-shape
    # batching headaches). Pick the first one in cfg.
    ds = ds_list[0]
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg["train"].get("num_workers", 4) > 0,
    )

    # ---- model ----
    print("[model] loading DA3-LARGE init ...")
    t0 = time.time()
    model = DA3MetricTrainable(
        ckpt_dir=cfg["model"]["init_ckpt"],
        freeze_backbone=cfg["model"].get("freeze_backbone", True),
    ).to(device)
    print(f"[model] ready in {time.time()-t0:.1f}s")

    # ---- loss ----
    lc = cfg["loss"]
    criterion = MetricDepthLoss(
        w_silog=lc.get("w_silog", 1.0),
        w_l1=lc.get("w_l1", 0.1),
        w_grad=lc.get("w_grad", 0.5),
        silog_variance_focus=lc.get("silog_variance_focus", 0.85),
    ).to(device)

    # ---- optim ----
    tc = cfg["train"]
    base_lr = float(tc["lr"])
    bb_lr_factor = float(tc.get("backbone_lr_factor", 0.1))
    weight_decay = float(tc.get("weight_decay", 0.01))
    decoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "backbone" not in n]
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "backbone" in n]
    param_groups = [{"params": decoder_params, "lr": base_lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": base_lr * bb_lr_factor})
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
    print(f"[optim] decoder={sum(p.numel() for p in decoder_params)/1e6:.1f}M "
          f"backbone={sum(p.numel() for p in backbone_params)/1e6:.1f}M")

    # ---- training loop ----
    max_epochs = int(tc["max_epochs"])
    steps_per_epoch = len(loader)
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = int(tc.get("warmup_steps", 500))
    grad_clip = float(tc.get("grad_clip", 1.0))
    log_every = int(tc.get("log_every", 20))
    eval_every = int(tc.get("eval_every", 500))
    save_every = int(tc.get("save_every", 1000))

    use_amp = cfg.get("amp", "bf16")
    amp_dtype = torch.bfloat16 if use_amp == "bf16" else torch.float16

    print(f"[train] {max_epochs} epochs * {steps_per_epoch} steps/ep = {total_steps} total")
    if args.smoke:
        total_steps = min(total_steps, 50)
        print(f"[smoke] capping at {total_steps} steps")

    step = 0
    t_loop = time.time()
    for ep in range(max_epochs):
        for batch in loader:
            if "per_sample" in batch:
                # mixed-shape batch: degrade to per-sample loop
                batches = [{k: v.unsqueeze(0) if torch.is_tensor(v) else [v]
                            for k, v in s.items()} for s in batch["per_sample"]]
            else:
                batches = [batch]

            losses_acc = {}
            for b in batches:
                img = b["image"].to(device, non_blocking=True)
                gt = b["depth"].to(device, non_blocking=True)
                mask = b["mask"].to(device, non_blocking=True)
                K = b["K"].to(device, non_blocking=True)
                # set per-step LR
                lr_now = cosine_lr(step, total_steps, base_lr, warmup=warmup_steps)
                for i, g in enumerate(optimizer.param_groups):
                    g["lr"] = lr_now * (bb_lr_factor if i == 1 else 1.0)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                     dtype=amp_dtype):
                    pred, _ = model.train_forward(img, K)
                # Cast pred back to float32 for loss numerical stability
                pred_f = pred.float()
                loss_dict = criterion(pred_f, gt, mask)
                total = loss_dict["total"]
                if not torch.isfinite(total):
                    print(f"  [warn] non-finite loss at step {step}, skipping")
                    continue
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                    max_norm=grad_clip,
                )
                optimizer.step()
                for k, v in loss_dict.items():
                    losses_acc[k] = losses_acc.get(k, 0.0) + float(v.detach().cpu())

            for k in losses_acc:
                losses_acc[k] /= max(len(batches), 1)

            if step % log_every == 0:
                el = time.time() - t_loop
                print(f"  [ep{ep:02d} step {step:>5d}/{total_steps}] "
                      f"loss={losses_acc.get('total', 0):.4f}  "
                      f"silog={losses_acc.get('silog', 0):.4f}  "
                      f"l1={losses_acc.get('l1', 0):.4f}  "
                      f"grad={losses_acc.get('grad', 0):.4f}  "
                      f"lr={lr_now:.2e}  ({el:.1f}s)")

            if step > 0 and step % eval_every == 0:
                print(f"  [eval @ step {step}]")
                m = quick_eval_kitti(model, ds, n=64, device=device, max_depth=80.0)
                print(f"    quick-eval (64 train samples): {m}")

            if step > 0 and step % save_every == 0:
                ck = out_dir / "ckpts" / f"step_{step:06d}.pt"
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "cfg": cfg,
                }, ck)
                print(f"    [ckpt] -> {ck}")

            step += 1
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    # final ckpt
    ck = out_dir / "ckpts" / "final.pt"
    torch.save({"step": step, "model": model.state_dict(), "cfg": cfg}, ck)
    print(f"[done] final ckpt -> {ck}")


if __name__ == "__main__":
    main()
