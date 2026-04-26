"""Standard depth metrics used by DA3 / DA2 / BTS papers.

References:
- Eigen et al. 2014 (NIPS): SiLog, abs_rel, sq_rel, rmse, rmse_log, δ1/δ2/δ3
- Depth Anything 2 / 3 evaluation protocol uses the same set, with δ1
  threshold = 1.25.
"""
from __future__ import annotations

import numpy as np


def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray | None = None,
    median_align: bool = False,
) -> dict[str, float]:
    """Compute standard depth metrics.

    Args:
        pred: predicted depth, shape (H, W) or (N, H, W), in meters.
        gt:   ground-truth depth, same shape, in meters.
        valid: optional bool mask of same shape; if None we use gt > 0.
        median_align: if True, scale pred by median(gt) / median(pred) on
            valid pixels before metrics (used for relative-depth eval; for
            METRIC depth we keep this False).

    Returns:
        dict with abs_rel, sq_rel, rmse, rmse_log, log10, d1, d2, d3, silog.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if valid is None:
        valid = gt > 0
    valid = valid & np.isfinite(pred) & np.isfinite(gt) & (pred > 0)

    p = pred[valid]
    g = gt[valid]
    if p.size == 0:
        return {k: float("nan") for k in
                ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10",
                 "d1", "d2", "d3", "silog", "n"]}

    if median_align:
        p = p * (np.median(g) / np.median(p))

    thresh = np.maximum(g / p, p / g)
    d1 = float((thresh < 1.25).mean())
    d2 = float((thresh < 1.25 ** 2).mean())
    d3 = float((thresh < 1.25 ** 3).mean())

    abs_rel = float(np.mean(np.abs(g - p) / g))
    sq_rel = float(np.mean(((g - p) ** 2) / g))
    rmse = float(np.sqrt(np.mean((g - p) ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(g) - np.log(p)) ** 2)))
    log10 = float(np.mean(np.abs(np.log10(g) - np.log10(p))))
    silog = float(
        np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)
                - np.mean(np.log(p) - np.log(g)) ** 2) * 100.0
    )

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "log10": log10,
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "silog": silog,
        "n": int(g.size),
    }


def kitti_eigen_crop(h: int, w: int) -> tuple[slice, slice]:
    """Garg/Eigen KITTI crop (used by DA2/DA3 KITTI eval).

    Crops to top=int(0.40810811*h), bot=int(0.99189189*h),
              left=int(0.03594771*w), right=int(0.96405229*w).
    """
    t = int(0.40810811 * h)
    b = int(0.99189189 * h)
    l = int(0.03594771 * w)
    r = int(0.96405229 * w)
    return slice(t, b), slice(l, r)
