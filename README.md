# DA3-Metric-Repro

> Depth Anything 3 — Metric Depth (Table 11) 复现项目

## 0. 写在最前

师兄要求：先复现 DA3 论文 Table 11（DA3-Metric 在 5 个 metric depth benchmark 上的指标），暂不训练。

参考：
- 论文：[Depth Anything 3 (arXiv:2511.10647)](https://arxiv.org/abs/2511.10647) — ICLR 2026 Oral
- 官方 repo：[ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- 模型：HuggingFace `depth-anything/DA3METRIC-LARGE`（0.35B，DINOv2-L backbone + Dual-DPT head，单帧 metric）

## 1. 目标 — 论文 Table 11

| Methods | NYUv2 | KITTI | ETH3D | SUN-RGBD | DIODE |
|---|---|---|---|---|---|
|  | δ1 / AbsRel | δ1 / AbsRel | δ1 / AbsRel | δ1 / AbsRel | δ1 / AbsRel |
| DA3-metric (paper) | 0.963 / 0.070 | 0.953 / 0.086 | 0.917 / 0.104 | 0.973 / 0.105 | 0.838 / 0.128 |
| **本项目复现** | 0.720 / 0.171 ⚠️ | **0.926 / 0.094** ✅ | TBD | TBD | TBD |

**KITTI 复现成功**：AbsRel 差 +0.008（相对 +9%），d1 差 -0.027（相对 -2.8%）。
完整 7 个指标见 [results/kitti_eigen_full.csv](results/kitti_eigen_full.csv)。

**NYUv2 偏差较大**：单张诊断显示 pred 与 GT 接近（例 idx=0: pred=2.91 m, GT=3.07 m），
但 654 张统计 abs_rel=0.17。怀疑 `/fs/scratch/.../nyuv2/val/` 不是官方 Eigen
test split，或 npy 经过非标准预处理。下一步从 `nyu_depth_v2_labeled.mat` +
`splits.mat` 重做。

## 2. 关键转换公式（论文 FAQ）

DA3-Metric 输出的不是直接米单位深度。要得到米单位：

```
metric_depth_meters = focal * net_output / 300.0
```

其中 `focal = (fx + fy) / 2`（来自相机内参 K），`300` 是论文中的 canonical focal length。

## 3. 项目结构

```
DA3-Metric-Repro/
├── README.md                 # 本文档
├── third_party/
│   └── Depth-Anything-3/     # 官方 repo（git clone）
├── checkpoints/
│   └── DA3METRIC-LARGE/      # HF 下载的官方权重
├── data_prep/                # 各数据集 list 准备脚本
├── eval/
│   ├── infer.py              # 调用官方 API forward
│   ├── metrics.py            # AbsRel, δ1
│   └── run_*.py              # 单个 benchmark 的 eval 入口
├── configs/                  # 评测配置 yaml
├── results/                  # 复现数字 csv
├── scripts/                  # LSF .bsub
└── docs/                     # 复现笔记、坑
```

## 4. 进度

- [x] 项目骨架 + README
- [x] 拉官方 repo + 下载 ckpt（1.3 GB at `checkpoints/DA3METRIC-LARGE/`）
- [x] conda env `liren/envs/da3`（torch2.4.1+cu121 + xformers + DA3 -e .）
- [x] 端到端 CPU smoke 跑通
- [x] KITTI Eigen runner + 提交（job 12222224 PEND on batch_h200）
- [x] NYUv2 runner + 提交（共享 scratch 654 张 val 已就位）
- [ ] ETH3D 评测（数据未就位，需下载）
- [ ] SUN-RGBD 评测（数据未就位，需下载）
- [ ] DIODE 评测（数据未就位，需下载）
- [ ] 五列复现数字 vs 论文对比表

### 数据集状态

| Dataset | 状态 | 路径 |
|---|---|---|
| KITTI Eigen | ✅ | `liren/depth-v3/training-for-depth-anything-v3/kitti_root` |
| NYUv2 (654 val) | ✅ | `/fs/scratch/datasets/cr_dlp_open_permissive/nyuv2/val/` |
| ETH3D | ❌ | TODO 下载（约 6 GB） |
| SUN-RGBD | ❌ | TODO 下载（约 6 GB） |
| DIODE indoor | ❌ | TODO 下载（约 3 GB） |

## 5. 与 DenseGRU-v1.0 项目的关系

完全独立，DenseGRU 的代码不动；它本身仍然是个有效的研究项目，只是命名上要避免和"DA3"挂钩。

## 6. Changelog

- **2026-04-26**: 项目初始化。明确走 Path A（评测复现）而不是 Path B（训练复现）。
