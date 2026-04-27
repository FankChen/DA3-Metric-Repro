"""Thin wrapper around the official `depth_anything_3.api.DepthAnything3` to
load DA3METRIC-LARGE once and run forward on a single RGB image with known
intrinsics, returning metric depth in meters.

Usage:
    from infer import DA3MetricInfer
    engine = DA3MetricInfer(model_id="depth-anything/DA3METRIC-LARGE",
                            device="cuda")
    depth_m = engine.predict(rgb_uint8_HW3, K_3x3)

The metric scaling rule is the official one (see DA3 paper Sec 4.3 / FAQ):

    metric_depth_meters = depth_raw * focal / 300,  focal = (fx + fy) / 2

with ``focal`` measured at the model's process resolution. We **reuse the
official utility** ``depth_anything_3.utils.alignment.apply_metric_scaling``
instead of re-implementing the formula, to guarantee bit-exact agreement
with how DA3-NESTED applies it internally
(see model/da3.py::_apply_metric_scaling).

Preprocessing (resize to multiple of patch_size=14, ImageNet normalize, K
resize) is **entirely handled inside** ``model.inference()`` via the
official ``InputProcessor`` (utils/io/input_processor.py). We do not
implement any of that ourselves.
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
        from depth_anything_3.utils.alignment import apply_metric_scaling  # noqa: E402

        self.device = torch.device(device)
        self.process_res = int(process_res)
        self.model = DepthAnything3.from_pretrained(model_id)
        self.model = self.model.to(self.device).eval()
        # Reference to official metric scaling utility (depth * focal / 300).
        # Same function DA3-NESTED uses internally (model/da3.py L368).
        self._apply_metric_scaling = apply_metric_scaling

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

        Implementation:
            1. ``model.inference([rgb_uint8])`` — official API; the official
               ``InputProcessor`` handles resize-to-multiple-of-14,
               ImageNet normalization, and (if intrinsics passed) K
               rescaling. We pass ``intrinsics=None`` because for the
               standalone DA3METRIC-LARGE the model output is raw depth
               (not metric); we apply the metric scaling ourselves below.
            2. Build the process-resolution K by the same rule the
               official ``_resize_ixt`` applies: ``fx *= w_proc/W;
               fy *= h_proc/H``.
            3. Call the official ``apply_metric_scaling(depth, K_proc)``
               (from ``depth_anything_3.utils.alignment``) — exactly the
               function DA3-NESTED uses internally to convert raw depth
               to metric meters.
            4. Upsample metric depth to original resolution (depth in
               meters is invariant to spatial resampling).
        """
        assert rgb_uint8.ndim == 3 and rgb_uint8.shape[2] == 3
        assert rgb_uint8.dtype == np.uint8
        H, W = rgb_uint8.shape[:2]

        prediction = self.model.inference(
            [rgb_uint8],
            intrinsics=None,
            process_res=process_res or self.process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        depth_raw = np.asarray(prediction.depth)  # (1, h_proc, w_proc) or (h_proc, w_proc)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[0]
        h_proc, w_proc = depth_raw.shape

        # --- official metric scaling (depth * focal / 300) ---
        # Build K_proc following official _resize_ixt rule (input_processor.py L262).
        K_proc = K.astype(np.float32).copy()
        K_proc[0] *= w_proc / float(W)   # fx, cx
        K_proc[1] *= h_proc / float(H)   # fy, cy
        depth_t = torch.from_numpy(depth_raw.astype(np.float32))[None, None]   # (1,1,h,w)
        ixt_t = torch.from_numpy(K_proc)[None, None]                            # (1,1,3,3)
        metric_t = self._apply_metric_scaling(depth_t, ixt_t)                  # (1,1,h,w)

        # Upsample metric depth back to input resolution.
        d_t = torch.nn.functional.interpolate(
            metric_t, size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        return d_t.astype(np.float32)
