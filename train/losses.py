"""SiLog + L1(log) + multi-scale gradient loss for metric depth.

Adapted to DA3-Metric: predictions and GT are in metres on valid pixels.
Same combo used by BTS / Adabins / DA2-metric.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _flatten(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and x.size(1) == 1:
        return x.squeeze(1)
    return x


class SiLogLoss(nn.Module):
    """BTS-style scale-invariant log loss.

    L = 10 * sqrt( mean(d^2) - lambda * mean(d)^2 )    where d_i = log(pred_i) - log(gt_i) on valid pixels.
    """

    def __init__(self, variance_focus: float = 0.85, eps: float = 1e-7):
        super().__init__()
        self.variance_focus = variance_focus
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = _flatten(pred)
        gt = _flatten(gt)
        mask = _flatten(mask).bool()

        if mask.sum() < 16:
            return (pred * 0).sum()

        p = pred[mask].clamp(min=self.eps)
        g = gt[mask].clamp(min=self.eps)
        d = torch.log(p) - torch.log(g)
        var = (d ** 2).mean() - self.variance_focus * (d.mean() ** 2)
        var = var.clamp(min=self.eps)
        return 10.0 * torch.sqrt(var)


class LogL1Loss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = _flatten(pred)
        gt = _flatten(gt)
        mask = _flatten(mask).bool()
        if mask.sum() < 16:
            return (pred * 0).sum()
        p = torch.log(pred[mask].clamp(min=self.eps))
        g = torch.log(gt[mask].clamp(min=self.eps))
        return (p - g).abs().mean()


class GradMatchLoss(nn.Module):
    """Multi-scale L1 on x/y gradients of log depth (only valid neighbours).

    Following Ranftl et al. MiDaS, scales = {1, 2, 4, 8}.
    """

    def __init__(self, scales: tuple = (1, 2, 4, 8), eps: float = 1e-7):
        super().__init__()
        self.scales = scales
        self.eps = eps

    @staticmethod
    def _grad(x: torch.Tensor):
        gx = x[:, :, 1:] - x[:, :, :-1]
        gy = x[:, 1:, :] - x[:, :-1, :]
        return gx, gy

    def _step(self, log_p, log_g, mask) -> torch.Tensor:
        px, py = self._grad(log_p)
        gx, gy = self._grad(log_g)
        mx = mask[:, :, 1:] * mask[:, :, :-1]
        my = mask[:, 1:, :] * mask[:, :-1, :]
        loss = log_p.new_zeros(())
        n = 0.0
        if mx.sum() > 0:
            loss = loss + ((px - gx).abs() * mx).sum() / mx.sum().clamp(min=1.0)
            n += 1
        if my.sum() > 0:
            loss = loss + ((py - gy).abs() * my).sum() / my.sum().clamp(min=1.0)
            n += 1
        return loss / max(n, 1.0)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = _flatten(pred)
        gt = _flatten(gt)
        mask = _flatten(mask).float()
        if mask.sum() < 16:
            return (pred * 0).sum()

        log_p = torch.log(pred.clamp(min=self.eps))
        log_g = torch.log(gt.clamp(min=self.eps))

        total = log_p.new_zeros(())
        n = 0.0
        for s in self.scales:
            if s == 1:
                lp, lg, mk = log_p, log_g, mask
            else:
                lp = log_p[:, ::s, ::s]
                lg = log_g[:, ::s, ::s]
                mk = mask[:, ::s, ::s]
            if mk.sum() > 0:
                total = total + self._step(lp, lg, mk)
                n += 1
        return total / max(n, 1.0)


class MetricDepthLoss(nn.Module):
    """w_silog * SiLog + w_l1 * LogL1 + w_grad * GradMatch."""

    def __init__(
        self,
        w_silog: float = 1.0,
        w_l1: float = 0.1,
        w_grad: float = 0.5,
        silog_variance_focus: float = 0.85,
    ):
        super().__init__()
        self.w_silog = w_silog
        self.w_l1 = w_l1
        self.w_grad = w_grad
        self.silog = SiLogLoss(variance_focus=silog_variance_focus)
        self.l1 = LogL1Loss()
        self.grad = GradMatchLoss()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict:
        l_silog = self.silog(pred, gt, mask)
        l_l1 = self.l1(pred, gt, mask)
        l_grad = self.grad(pred, gt, mask)
        total = self.w_silog * l_silog + self.w_l1 * l_l1 + self.w_grad * l_grad
        return {
            "total": total,
            "silog": l_silog.detach(),
            "l1": l_l1.detach(),
            "grad": l_grad.detach(),
        }
