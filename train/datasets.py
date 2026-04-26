"""Training Datasets for DA3-Metric finetune.

Two datasets, both returning the same dict:
    image: (3, H, W) float32, ImageNet-normalized
    depth: (H, W) float32 metres
    mask : (H, W) bool, valid GT pixels
    K    : (3, 3) float32, intrinsics at the *image's H,W* (after resize)

We resize each image so longest side == process_res (default 504), preserving
aspect ratio, then round to the nearest multiple of 14 (DINOv2 patch size),
mirroring the official InputProcessor "upper_bound_resize" used in eval.
The intrinsics K are scaled by (sx, sy) accordingly so that the on-the-fly
formula  metric = (focal/300) * raw  is numerically consistent between train
and eval.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 14


def _round_to_patch(x: int) -> int:
    return max(PATCH, int(round(x / PATCH)) * PATCH)


def upper_bound_resize_size(W: int, H: int, target: int = 504) -> Tuple[int, int]:
    longest = max(W, H)
    scale = target / float(longest)
    new_w = _round_to_patch(int(round(W * scale)))
    new_h = _round_to_patch(int(round(H * scale)))
    return new_w, new_h


def normalize_image_np(rgb_np: np.ndarray) -> torch.Tensor:
    """rgb_np: (H, W, 3) uint8 -> normalized (3, H, W) float."""
    t = TF.to_tensor(rgb_np)  # (3,H,W) in [0,1]
    return TF.normalize(t, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))


# ---------------------------------------------------------------------------
# KITTI Eigen training split (BTS format)
# ---------------------------------------------------------------------------
def parse_kitti_split(split_path: Path) -> List[Tuple[str, str, float]]:
    out: List[Tuple[str, str, float]] = []
    with open(split_path) as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 3 or parts[1] in ("None", "none"):
                continue
            out.append((parts[0], parts[1], float(parts[2])))
    return out


def resolve_kitti_depth(data_root: Path, dep_rel: str) -> Optional[Path]:
    for sub in ("train", "val"):
        p = data_root / "depth" / sub / dep_rel
        if p.is_file():
            return p
    return None


class KITTIEigenDataset(Dataset):
    """KITTI raw + projected depth GT (BTS Eigen split format)."""

    def __init__(
        self,
        data_root: str,
        split_file: str,
        process_res: int = 504,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        augment: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.process_res = process_res
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.augment = augment

        self.samples = parse_kitti_split(Path(split_file))
        if not self.samples:
            raise RuntimeError(f"empty KITTI split: {split_file}")

        # Per-drive K cache: drive -> (fx, fy, cx, cy) at native 1242x375.
        self._K_cache: dict = {}

        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def _load_K(self, rgb_rel: str, focal_fallback: float) -> np.ndarray:
        """Load P_rect_02 from KITTI calib file. Fallback: f only, principal at centre.

        rgb_rel example: 2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png
        calib at: data_root/raw/<date>/calib_cam_to_cam.txt
        """
        date = rgb_rel.split("/")[0]
        if date in self._K_cache:
            return self._K_cache[date].copy()
        calib = self.data_root / "raw" / date / "calib_cam_to_cam.txt"
        K = None
        if calib.is_file():
            for ln in calib.read_text().splitlines():
                if ln.startswith("P_rect_02:"):
                    vals = [float(v) for v in ln.split()[1:]]
                    if len(vals) >= 12:
                        fx, fy, cx, cy = vals[0], vals[5], vals[2], vals[6]
                        K = np.array(
                            [[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float32
                        )
                        break
        if K is None:
            # focal-only fallback (cx, cy will be set after resize knows H, W)
            K = np.array(
                [[focal_fallback, 0, -1.0], [0, focal_fallback, -1.0], [0, 0, 1.0]],
                dtype=np.float32,
            )
        self._K_cache[date] = K
        return K.copy()

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        rgb_rel, dep_rel, focal = self.samples[idx]
        rgb_p = self.data_root / "raw" / rgb_rel
        dep_p = resolve_kitti_depth(self.data_root, dep_rel)
        if dep_p is None:
            # skip by returning the next valid one
            return self.__getitem__((idx + 1) % len(self))

        rgb_pil = Image.open(rgb_p).convert("RGB")
        dep_arr = np.asarray(Image.open(dep_p), dtype=np.int32).astype(np.float32) / 256.0

        Worig, Horig = rgb_pil.size
        K_orig = self._load_K(rgb_rel, focal_fallback=focal)
        # if cx,cy were unknown, default to centre on ORIGINAL image
        if K_orig[0, 2] < 0:
            K_orig[0, 2] = Worig / 2.0
            K_orig[1, 2] = Horig / 2.0

        # match depth shape to RGB if needed
        if dep_arr.shape[1] != Worig or dep_arr.shape[0] != Horig:
            dep_arr = np.asarray(
                Image.fromarray(dep_arr).resize((Worig, Horig), Image.NEAREST), dtype=np.float32
            )

        # ---- resize to model input ----
        new_w, new_h = upper_bound_resize_size(Worig, Horig, self.process_res)
        rgb_pil = rgb_pil.resize((new_w, new_h), Image.BICUBIC)
        # Depth: use NEAREST so we keep validity sharp; also rescale magnitudes are unchanged (z is invariant to spatial resize).
        dep_arr = np.asarray(
            Image.fromarray(dep_arr).resize((new_w, new_h), Image.NEAREST), dtype=np.float32
        )
        sx = new_w / Worig
        sy = new_h / Horig
        K = K_orig.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

        # ---- augmentation ----
        if self.augment and random.random() < 0.5:
            rgb_pil = TF.hflip(rgb_pil)
            dep_arr = dep_arr[:, ::-1].copy()
            # K_x is reflected: cx -> W - cx. fx unchanged.
            K[0, 2] = new_w - K[0, 2]
        if self.augment:
            rgb_pil = self.color_jitter(rgb_pil)

        rgb_np = np.asarray(rgb_pil)
        img_t = normalize_image_np(rgb_np)

        depth_t = torch.from_numpy(np.ascontiguousarray(dep_arr)).float()
        mask = (depth_t > self.min_depth) & (depth_t < self.max_depth)

        return {
            "image": img_t,
            "depth": depth_t,
            "mask": mask,
            "K": torch.from_numpy(K).float(),
            "src": "kitti",
        }


# ---------------------------------------------------------------------------
# NYUv2 training split (raw 480x640 PNG)
# ---------------------------------------------------------------------------
NYU_FX, NYU_FY = 518.8579, 519.4696
NYU_CX, NYU_CY = 325.5824, 253.7362


class NYUv2Dataset(Dataset):
    """NYUv2 train split: nyu_depth_v2/nyuv2/train/{rgb,depth}/{id}.png.

    Depth is uint16 millimetres; converted to metres on load.
    """

    def __init__(
        self,
        data_root: str,
        split_file: Optional[str] = None,
        process_res: int = 504,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        augment: bool = True,
    ):
        super().__init__()
        self.root = Path(data_root)
        self.process_res = process_res
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.augment = augment
        if split_file is None:
            split_file = str(self.root / "train.txt")
        self.ids = [
            ln.strip() for ln in Path(split_file).read_text().splitlines() if ln.strip()
        ]
        if not self.ids:
            raise RuntimeError(f"empty NYU split: {split_file}")

        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        idn = self.ids[idx]
        rgb_p = self.root / "train" / "rgb" / f"{idn}.png"
        dep_p = self.root / "train" / "depth" / f"{idn}.png"
        if not (rgb_p.is_file() and dep_p.is_file()):
            return self.__getitem__((idx + 1) % len(self))

        rgb_pil = Image.open(rgb_p).convert("RGB")
        dep_arr = np.asarray(Image.open(dep_p), dtype=np.uint16).astype(np.float32) / 1000.0
        Worig, Horig = rgb_pil.size  # 640, 480

        K_orig = np.array(
            [[NYU_FX, 0, NYU_CX], [0, NYU_FY, NYU_CY], [0, 0, 1.0]], dtype=np.float32
        )

        new_w, new_h = upper_bound_resize_size(Worig, Horig, self.process_res)
        rgb_pil = rgb_pil.resize((new_w, new_h), Image.BICUBIC)
        dep_arr = np.asarray(
            Image.fromarray(dep_arr).resize((new_w, new_h), Image.NEAREST), dtype=np.float32
        )
        sx, sy = new_w / Worig, new_h / Horig
        K = K_orig.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

        if self.augment and random.random() < 0.5:
            rgb_pil = TF.hflip(rgb_pil)
            dep_arr = dep_arr[:, ::-1].copy()
            K[0, 2] = new_w - K[0, 2]
        if self.augment:
            rgb_pil = self.color_jitter(rgb_pil)

        img_t = normalize_image_np(np.asarray(rgb_pil))
        depth_t = torch.from_numpy(np.ascontiguousarray(dep_arr)).float()
        mask = (depth_t > self.min_depth) & (depth_t < self.max_depth)

        return {
            "image": img_t,
            "depth": depth_t,
            "mask": mask,
            "K": torch.from_numpy(K).float(),
            "src": "nyu",
        }


# ---------------------------------------------------------------------------
# Mixed sampler: alternate between two datasets with given probability
# ---------------------------------------------------------------------------
class MixedSampler(torch.utils.data.IterableDataset):
    """Wraps multiple Datasets; each step samples one with given probability."""

    def __init__(self, datasets: List[Dataset], probs: List[float], seed: int = 0):
        super().__init__()
        assert len(datasets) == len(probs) and abs(sum(probs) - 1.0) < 1e-6
        self.datasets = datasets
        self.probs = probs
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed + (torch.utils.data.get_worker_info().id
                                                  if torch.utils.data.get_worker_info() else 0))
        while True:
            di = rng.choice(len(self.datasets), p=self.probs)
            ds = self.datasets[di]
            i = int(rng.integers(0, len(ds)))
            yield ds[i]


def collate_pad(batch: List[dict]) -> dict:
    """Variable-size images need padding to common HxW within batch.

    For simplicity we currently require single-source batches (KITTI-only or
    NYU-only) where all samples share size. The MixedSampler should produce
    batches of size 1 or use grad-accum across heterogeneous shapes.
    """
    # Check if all images share same shape
    shapes = {tuple(b["image"].shape) for b in batch}
    if len(shapes) > 1:
        # fall back to per-sample list: training loop must iterate
        return {"per_sample": batch}
    out = {}
    for k in ("image", "depth", "mask", "K"):
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["src"] = [b["src"] for b in batch]
    return out
