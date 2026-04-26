"""Initial test: tiny KITTI Eigen eval (small subset) to confirm pipeline.

Reuses kitti_root/splits and KITTI raw + GT layout from the existing
training-for-depth-anything-v3 sandbox so we don't re-download anything.

KITTI eval protocol (DA2 / Eigen):
- Use the standard 697 Eigen test images.
- GT depth: KITTI raw projected lidar (from BTS) or `data_depth_annotated`
  validation set if available; we accept whichever the existing
  `eigen_test_files.txt` points at.
- Crop with the Garg/Eigen crop before metric computation.
- Cap at 80 m; report abs_rel and δ1 to compare with paper Table 11
  (DA3-metric KITTI: δ1=0.953, AbsRel=0.086).

This script is intentionally simple and runs eagerly; once the pipeline
is verified we'll wrap it into a config-driven runner.
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
from metrics import compute_depth_metrics, kitti_eigen_crop  # noqa: E402


# Default KITTI intrinsics (Eigen split uses cam_02 of various drives; the
# original BTS evaluator approximates with these). DA2/DA3 papers use the
# per-drive calibration; for our first pass we will read fx,fy from the
# kitti raw `calib_cam_to_cam.txt` per-drive.
def load_kitti_intrinsics(date_dir: Path, cam_idx: int = 2) -> np.ndarray | None:
    """Parse calib_cam_to_cam.txt and return P_rect_0{cam}'s intrinsics."""
    calib_p = date_dir / "calib_cam_to_cam.txt"
    if not calib_p.is_file():
        return None
    K = None
    with open(calib_p) as f:
        for line in f:
            if line.startswith(f"P_rect_0{cam_idx}:"):
                vals = [float(v) for v in line.split()[1:]]
                K = np.array(vals, dtype=np.float64).reshape(3, 4)[:, :3]
                break
    return K


def parse_split(split_file: Path) -> list[tuple[str, str | None]]:
    """Each line: '<rgb_rel_path> <depth_rel_path or None> <focal>'."""
    items = []
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rgb_rel = parts[0]
            depth_rel = parts[1] if len(parts) > 1 and parts[1] != "None" else None
            items.append((rgb_rel, depth_rel))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",
                    default="/home/izi2sgh/MYDATA/quanjie/liren/depth-v3/training-for-depth-anything-v3/kitti_root")
    ap.add_argument("--split_file",
                    default="/home/izi2sgh/MYDATA/quanjie/liren/depth-v3/training-for-depth-anything-v3/splits/eigen_test_files.txt")
    ap.add_argument("--max_samples", type=int, default=20,
                    help="for the smoke run; set to 697 for full Eigen test.")
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--min_depth", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument("--output", default=str(ROOT / "results" / "kitti_eigen_smoke.csv"))
    args = ap.parse_args()

    data_root = Path(args.data_root)
    split = parse_split(Path(args.split_file))
    if args.max_samples > 0:
        split = split[: args.max_samples]
    print(f"[kitti] data_root={data_root}")
    print(f"[kitti] num samples={len(split)}")

    print("[kitti] loading DA3METRIC-LARGE ...")
    t0 = time.time()
    engine = DA3MetricInfer(device=args.device, process_res=args.process_res)
    print(f"[kitti] model ready in {time.time() - t0:.1f}s")

    aggregated = {k: 0.0 for k in
                  ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10",
                   "d1", "d2", "d3", "silog"]}
    n_pix = 0
    n_img = 0
    rows = []

    for i, (rgb_rel, depth_rel) in enumerate(split):
        rgb_p = data_root / "raw" / rgb_rel
        if not rgb_p.is_file():
            print(f"  [skip] missing rgb: {rgb_p}")
            continue
        if depth_rel is None:
            continue
        depth_p = data_root / "depth" / depth_rel
        if not depth_p.is_file():
            # Try BTS layout fallback
            alt = data_root / depth_rel
            if alt.is_file():
                depth_p = alt
            else:
                print(f"  [skip] missing gt: {depth_p}")
                continue

        # KITTI date dir lives at  raw/<date>/<date>_drive_xxxx_sync/...
        date_dir = data_root / "raw" / rgb_rel.split("/")[0]
        K = load_kitti_intrinsics(date_dir)
        if K is None:
            print(f"  [skip] missing calib for {date_dir}")
            continue

        rgb = np.asarray(Image.open(rgb_p).convert("RGB"))
        gt_raw = np.asarray(Image.open(depth_p), dtype=np.uint16).astype(np.float32) / 256.0

        # Resize K if RGB and GT differ in scale (KITTI: usually identical)
        H, W = rgb.shape[:2]
        gh, gw = gt_raw.shape
        if (gh, gw) != (H, W):
            # rescale GT depth grid coords -> use simple cropping protocol:
            # take min H,W intersection. KITTI Eigen GT is already aligned
            # to RGB so this branch should rarely trigger.
            min_h, min_w = min(H, gh), min(W, gw)
            rgb = rgb[:min_h, :min_w]
            gt_raw = gt_raw[:min_h, :min_w]
            H, W = min_h, min_w

        depth_pred = engine.predict(rgb, K)

        # Garg/Eigen crop + depth cap
        ys, xs = kitti_eigen_crop(H, W)
        gt = gt_raw[ys, xs]
        pr = depth_pred[ys, xs]
        valid = (gt > args.min_depth) & (gt < args.max_depth)
        pr = np.clip(pr, args.min_depth, args.max_depth)

        m = compute_depth_metrics(pr, gt, valid=valid, median_align=False)
        for k in aggregated:
            aggregated[k] += m[k] * m["n"]
        n_pix += m["n"]
        n_img += 1
        rows.append((rgb_rel, m["abs_rel"], m["d1"], m["rmse"]))
        if i < 3 or i % 50 == 0:
            print(f"  [{i+1:>4}/{len(split)}] {rgb_rel}  "
                  f"abs_rel={m['abs_rel']:.4f}  d1={m['d1']:.4f}")

    if n_pix == 0:
        print("[kitti] no valid samples; aborting.")
        return
    avg = {k: v / n_pix for k, v in aggregated.items()}
    print("\n=== KITTI Eigen (DA3METRIC-LARGE)  paper: d1=0.953  AbsRel=0.086 ===")
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
        f.write("rgb,abs_rel,d1,rmse\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f}\n")
    print(f"[kitti] per-image rows -> {out}")


if __name__ == "__main__":
    main()
