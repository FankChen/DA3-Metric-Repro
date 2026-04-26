"""DA3-Metric trainable wrapper.

Bypasses the official ``DepthAnything3.forward`` (which is decorated with
``@torch.inference_mode()`` + ``torch.no_grad()`` and is unusable for training).
Loads the inner ``DepthAnything3Net`` and exposes a clean ``train_forward``
that returns metric depth in metres (using the canonical formula
``metric = focal_proc / 300 * raw_output``).

Init is from the official ``DA3-LARGE`` (relative-depth) checkpoint; the
metric finetune adapts the DPT head to predict canonical metric depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DA3MetricTrainable(nn.Module):
    """Wraps DA3 inner network so it's training-capable.

    Args:
        ckpt_dir: HF-style folder containing config.json + model.safetensors.
                  We use ``DA3-LARGE`` (relative-depth) as init. The metric
                  finetune is performed by training with metric supervision.
        freeze_backbone: if True, DINOv2 backbone params are frozen
                         (they are not updated by the optimizer).
    """

    def __init__(
        self,
        ckpt_dir: str = "checkpoints/DA3-LARGE",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        # Defer the import so we don't trigger DA3 init unless this class is used.
        from depth_anything_3.api import DepthAnything3

        ckpt_dir = str(Path(ckpt_dir))
        # ``from_pretrained`` is exposed via PyTorchModelHubMixin.
        self.api = DepthAnything3.from_pretrained(ckpt_dir)
        # Inner network where we will run forward in training mode.
        self.net = self.api.model
        self.net.train()

        if freeze_backbone:
            self.freeze_backbone()

    # ------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        # In DA3, the DINOv2 lives at self.net.backbone.* (it may be nested).
        # We freeze every parameter under any submodule named 'backbone'
        # (DINOv2). Decoder/head params stay trainable.
        n_frozen = 0
        n_trainable = 0
        for name, p in self.net.named_parameters():
            # 'backbone' here matches DA3 cross-view backbone which contains DINOv2.
            if "backbone" in name and "head" not in name:
                p.requires_grad_(False)
                n_frozen += p.numel()
            else:
                n_trainable += p.numel()
        print(f"[DA3MetricTrainable] frozen={n_frozen/1e6:.1f}M  "
              f"trainable={n_trainable/1e6:.1f}M")

    def unfreeze_all(self) -> None:
        for p in self.net.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    def train_forward(
        self,
        image: torch.Tensor,
        K: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass for training.

        Args:
            image: (B, 3, H, W) ImageNet-normalized tensor on the model device.
                   H, W must be multiples of 14 (we already enforced this in
                   the dataset).
            K    : (B, 3, 3) intrinsics at the same H, W.

        Returns:
            metric_depth: (B, H, W) metres.
            output_dict : raw model output (depth, conf, ...) for diagnostics.
        """
        B, C, H, W = image.shape
        # DA3 expects (B, N, 3, H, W) with N = number of views; for monocular N=1.
        x = image.unsqueeze(1)
        # We do not pass extrinsics/intrinsics into the network — it's a single
        # view, no cross-view geometry needed. K is only used for the focal/300
        # rescaling at the OUTPUT side (consistent with eval).
        out = self.net(x, extrinsics=None, intrinsics=None,
                        export_feat_layers=[], infer_gs=False,
                        use_ray_pose=False, ref_view_strategy="first")
        # Output 'depth' has shape (B, N=1, H, W) or (B, N, 1, H, W).
        d = out["depth"] if "depth" in out else out.depth
        if d.dim() == 5:
            d = d.squeeze(2)
        if d.dim() == 4:
            d = d.squeeze(1)            # -> (B, H, W)
        # Resize to image res if needed (DPT usually outputs at input res).
        if d.shape[-2:] != (H, W):
            d = F.interpolate(d.unsqueeze(1), size=(H, W),
                              mode="bilinear", align_corners=False).squeeze(1)

        # Apply canonical metric scaling: metric = (focal/300) * raw
        f = (K[:, 0, 0] + K[:, 1, 1]) * 0.5  # (B,)
        metric = (f.view(B, 1, 1) / 300.0) * d
        # Strictly positive (training stability)
        metric = metric.clamp(min=1e-4)
        return metric, out
