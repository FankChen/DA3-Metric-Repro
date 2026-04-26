"""DA3-Metric eval on NYUv2 official 654-image Eigen test split (full res).

Data layout (cluster):
  /fs/scratch/datasets/cr_dlp_open_permissive/nyuv2/nyu_depth_v2/nyuv2/test/
      rgb/{####}.png         (480, 640, 3) uint8
      depth/{####}.png       (480, 640) uint16, millimetres (inpainted)
      depth_raw/{####}.png   (480, 640) uint16, mm, 0 = invalid
  test.txt                   654 frame ids (one per line)

NYUv2 evaluation protocol (Eigen et al. 2014, used by DA2/DA3):
- Cap depth at 10 m, min 1e-3.
- Eigen crop rows [45:471] cols [41:601].
- Use the inpainted depth/ as GT (matches BTS/Adabins/DA2/DA3).

Intrinsics: official Kinect intrinsics on 640x480:
  fx=518.8579, fy=519.4696, cx=325.5824, cy=253.7362

Paper Table 11 NYUv2 target: d1=0.963, AbsRel=0.070.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from infer import DA3MetricInfer  # noqa: E402
from metrics import compute_depth_metrics  # noqa: E402


# Official Kinect intrinsics on 640x480
NYU_FX, NYU_FY = 518.8579, 519.4696
NYU_CX, NYU_CY = 325.5824, 253.7362
NYU_BASE_W, NYU_BASE_H = 640, 480


def nyu_intrinsics(h: int, w: int) -> np.ndarray:
    sx = w / NYU_BASE_W
    sy = h / NYU_BASE_H
    K = np.array([[NYU_FX * sx, 0, NYU_CX * sx],
                  [0, NYU_FY * sy, NYU_CY * sy],
                  [0, 0, 1.0]], dtype=np.float64)
    return K


def nyu_eigen_crop(h: int, w: int) -> tuple[slice, slice]:
    """Eigen crop on 480x640: rows [45:471], cols [41:601]. Rescaled."""
    t = int(round(45 / 480 * h))
    b = int(round(471 / 480 * h))
    l = int(round(41 / 640 * w))
    r = int(round(601 / 640 * w))
    return slice(t, b), slice(l, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/fs/scratch/datasets/cr_dlp_open_permissive/nyuv2/nyu_depth_v2/nyuv2")
    ap.add_argument("--split_file", default=None,
                    help="defaults to <root>/test.txt")
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--max_depth", type=float, default=10.0)
    ap.add_argument("--min_depth", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument("--output", default=str(ROOT / "results" / "nyuv2_val.csv"))
    args = ap.parse_args()

    root = Path(args.root)
    split = Path(args.split_file) if args.split_file else (root / "test.txt")
    ids = [ln.strip() for ln in split.read_text().splitlines() if ln.strip()]
    if args.max_samples > 0:
        ids = ids[: args.max_samples]
    print(f"[nyu] root={root}")
    print(f"[nyu] split={split}  N={len(ids)}")

    print("[nyu] loading DA3METRIC-LARGE ...")
    t0 = time.time()
    engine = DA3MetricInfer(device=args.device, process_res=args.process_res)
    print(f"[nyu] model ready in {time.time() - t0:.1f}s")

    aggregated = {k: 0.0 for k in
                  ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10",
                   "d1", "d2", "d3", "silog"]}
    n_pix = 0
    n_img = 0
    rows = []
    t_loop = time.time()

    for i, idx in enumerate(ids):
        rgb_p = root / "test" / "rgb" / f"{idx}.png"
        gt_p = root / "test" / "depth" / f"{idx}.png"
        if not (rgb_p.is_file() and gt_p.is_file()):
            print(f"  [skip] missing {idx}")
            continue

        rgb = np.array(Image.open(rgb_p))                # (480,640,3) uint8
        gt = np.array(Image.open(gt_p)).astype(np.float32) / 1000.0  # mm -> m
        H, W = rgb.shape[:2]
        K = nyu_intrinsics(H, W)

        depth_pred = engine.predict(rgb, K)

        ys, xs = nyu_eigen_crop(H, W)
        gt_c = gt[ys, xs]
        pr_c = depth_pred[ys, xs]
        valid = (gt_c > args.min_depth) & (gt_c < args.max_depth)
        pr_c = np.clip(pr_c, args.min_depth, args.max_depth)

        m = compute_depth_metrics(pr_c, gt_c, valid=valid, median_align=False)
        for k in aggregated:
            aggregated[k] += m[k] * m["n"]
        n_pix += m["n"]
        n_img += 1
        rows.append((idx, m["abs_rel"], m["d1"], m["rmse"]))
        if i < 3 or (i + 1) % 50 == 0:
            elapsed = time.time() - t_loop
            print(f"  [{i+1:>4}/{len(ids)}] idx={idx}  "
                  f"abs_rel={m['abs_rel']:.4f}  d1={m['d1']:.4f}  "
                  f"({elapsed:.1f}s)")

    if n_pix == 0:
        print("[nyu] no valid samples; aborting.")
        return
    avg = {k: v / n_pix for k, v in aggregated.items()}
    print("\n=== NYUv2 (DA3METRIC-LARGE)  paper: d1=0.963  AbsRel=0.070 ===")
    print(f"  N images : {n_img}")
    print(f"  abs_rel  : {avg['abs_rel']:.4f}")
    print(f"  sq_rel   : {avg['sq_rel']:.4f}")
    print(f"  rmse     : {avg['rmse']:.4f}")
    print(f"  rmse_log : {avg['rmse_log']:.4f}")
    print(f"  log10    : {avg['log10']:.4f}")
    print(f"  d1       : {avg['d1']:.4f}")
    print(f"  d2       : {avg['d2']:.4f}")
    print(f"  d3       : {avg['d3']:.4f}")
    print(f"  silog    : {avg['silog']:.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("idx,abs_rel,d1,rmse\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f}\n")
    print(f"[nyu] per-image rows -> {out}")


if __name__ == "__main__":
    main()
