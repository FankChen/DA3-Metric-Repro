"""Thin wrapper around the official `depth_anything_3.api.DepthAnything3` to
load DA3METRIC-LARGE once and run forward on a single RGB image with known
intrinsics, returning metric depth in meters.

Usage:
    from infer import DA3MetricInfer
    engine = DA3MetricInfer(model_id="depth-anything/DA3METRIC-LARGE",
                            device="cuda")
    depth_m = engine.predict(rgb_uint8_HW3, K_3x3, eval_size=518)

We follow the FAQ in the official README:
    metric_depth_meters = focal * raw_output / 300.0,  focal = (fx + fy) / 2

Notes:
- For the standalone DA3METRIC-LARGE model, `inference()` returns RAW
  depth (not metric scaled). That is the value we apply the focal/300
  rule to.
- Input images are auto-resized internally by the API to a multiple of
  patch_size=14 close to `process_res`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def _ensure_repo_on_path(repo_path: str | Path) -> None:
    p = str(Path(repo_path).expanduser().resolve() / "src")
    if p not in sys.path:
        sys.path.insert(0, p)


class DA3MetricInfer:
    def __init__(
        self,
        model_id: str | None = None,
        repo_path: str | Path | None = None,
        device: str = "cuda",
        process_res: int = 504,
    ) -> None:
        if model_id is None:
            # Prefer the locally downloaded checkpoint to avoid hitting HF
            # from compute nodes that may have no internet.
            local = (
                Path(__file__).resolve().parents[1]
                / "checkpoints" / "DA3METRIC-LARGE"
            )
            model_id = str(local) if local.is_dir() else "depth-anything/DA3METRIC-LARGE"
        if repo_path is None:
            repo_path = Path(__file__).resolve().parents[1] / "third_party" / "Depth-Anything-3"
        _ensure_repo_on_path(repo_path)
        from depth_anything_3.api import DepthAnything3  # noqa: E402

        self.device = torch.device(device)
        self.process_res = int(process_res)
        self.model = DepthAnything3.from_pretrained(model_id)
        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(
        self,
        rgb_uint8: np.ndarray,
        K: np.ndarray,
        process_res: int | None = None,
    ) -> np.ndarray:
        """Run DA3-Metric on a single RGB image and return metric depth (H, W).

        Args:
            rgb_uint8: (H, W, 3) uint8 RGB image.
            K: (3, 3) camera intrinsics matrix corresponding to the input
                resolution (i.e. the original H, W of `rgb_uint8`).
            process_res: optional resize target (default: self.process_res).

        Returns:
            depth_m: (H, W) float32 metric depth in meters, resampled to
                input resolution.

        Important: the model forwards on a resized image of shape
        (h_proc, w_proc) close to ``process_res`` on its longest side.
        Per the official FAQ ``metric_depth = focal / 300 * raw_output``,
        with `focal` measured **at the model's input resolution** because
        ``raw_output`` is conditioned on the canonical focal of 300 px in
        that same resolution.  We therefore scale K by (sx, sy) before
        computing focal, then upsample the metric depth to the original
        resolution (depth in meters is invariant to spatial resize).
        """
        assert rgb_uint8.ndim == 3 and rgb_uint8.shape[2] == 3
        assert rgb_uint8.dtype == np.uint8
        H, W = rgb_uint8.shape[:2]

        prediction = self.model.inference(
            [rgb_uint8],
            intrinsics=None,  # we apply our own focal/300 scaling
            process_res=process_res or self.process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        depth_raw = np.asarray(prediction.depth)  # (1, h_proc, w_proc) or (h_proc, w_proc)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[0]
        h_proc, w_proc = depth_raw.shape

        # --- correct metric scaling ---
        # Use intrinsics at the model-input resolution.
        sx = w_proc / float(W)
        sy = h_proc / float(H)
        f_proc = 0.5 * (float(K[0, 0]) * sx + float(K[1, 1]) * sy)
        metric_proc = (f_proc / 300.0) * depth_raw.astype(np.float32)

        # Upsample metric depth back to input resolution.
        d_t = torch.from_numpy(metric_proc)[None, None]
        d_t = torch.nn.functional.interpolate(
            d_t, size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        return d_t.astype(np.float32)
