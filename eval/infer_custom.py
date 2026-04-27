"""Variant of eval/infer.py that loads our self-trained DA3-Metric ckpt
on top of DA3-LARGE backbone weights. Mirrors DA3MetricInfer's predict()
exactly so we can swap engines transparently in the runners.
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


class DA3MetricCustomInfer:
    """Loads DA3-LARGE then overrides with a self-trained state_dict."""

    def __init__(
        self,
        train_ckpt: str | Path,
        base_dir: str | Path | None = None,
        repo_path: str | Path | None = None,
        device: str = "cuda",
        process_res: int = 504,
    ) -> None:
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "DA3-LARGE"
        if repo_path is None:
            repo_path = Path(__file__).resolve().parents[1] / "third_party" / "Depth-Anything-3"
        _ensure_repo_on_path(repo_path)
        from depth_anything_3.api import DepthAnything3  # noqa: E402
        from depth_anything_3.utils.alignment import apply_metric_scaling  # noqa: E402

        self.device = torch.device(device)
        self.process_res = int(process_res)
        self._apply_metric_scaling = apply_metric_scaling

        print(f"[infer-custom] base = {base_dir}")
        self.model = DepthAnything3.from_pretrained(str(base_dir))

        print(f"[infer-custom] loading trained ckpt = {train_ckpt}")
        ckpt = torch.load(str(train_ckpt), map_location="cpu", weights_only=False)
        sd_full = ckpt["model"]
        # The trained wrapper saved both `api.*` and `net.*` aliases.
        # Filter to api.* and strip the prefix to match self.model.
        sd = {k[len("api."):]: v for k, v in sd_full.items() if k.startswith("api.")}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"[infer-custom] missing={len(missing)}  unexpected={len(unexpected)}")
        if missing:
            print("  first missing:", missing[:3])
        if unexpected:
            print("  first unexpected:", unexpected[:3])

        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(
        self,
        rgb_uint8: np.ndarray,
        K: np.ndarray,
        process_res: int | None = None,
    ) -> np.ndarray:
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
        depth_raw = np.asarray(prediction.depth)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[0]
        h_proc, w_proc = depth_raw.shape

        # Official metric scaling (depth * focal / 300), with K rescaled
        # to process resolution per official _resize_ixt rule.
        K_proc = K.astype(np.float32).copy()
        K_proc[0] *= w_proc / float(W)
        K_proc[1] *= h_proc / float(H)
        depth_t = torch.from_numpy(depth_raw.astype(np.float32))[None, None]
        ixt_t = torch.from_numpy(K_proc)[None, None]
        metric_t = self._apply_metric_scaling(depth_t, ixt_t)

        d_t = torch.nn.functional.interpolate(
            metric_t, size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        return d_t.astype(np.float32)
