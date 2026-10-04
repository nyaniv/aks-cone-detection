# Cone obstacle detection — Autonomous Karting Series

Detect traffic cones as **mid-track obstacles to steer around**.

The track is unbounded (road bleeds into grass), so cones here are *not*
boundary markers. That is the one assumption that shapes everything else:
a cone is an object to avoid, its color carries no meaning, and the cone that
matters most is the small, far-away one we have not reached yet.

- **Framework:** Ultralytics YOLO26n (fallback YOLO11n if JetPack dependency issues)
- **Target:** Jetson, TensorRT engine
- **Classes:** one — `cone`. No color distinction.

## What we optimize

**Recall, bucketed by bbox height in pixels.** A missed cone is a collision;
a false positive is a downstream problem we filter temporally. Overall mAP is
dominated by big near cones that were never in danger of being missed, so it is
logged for context and otherwise ignored.

Buckets: `<10px`, `10–25px`, `25–60px`, `>60px`.

Corollary: we do **not** buy precision with a high confidence threshold.
False positives get handled by temporal filtering in the planner, not here.

## Tasks

| # | Task | Script | Status |
|---|------|--------|--------|
| 0 | Env + readiness check | `scripts/env_check.py` | **done** |
| 1 | Inventory both datasets | `scripts/inventory.py`, `scripts/fsoco_to_yolo.py` | **done** — see `notes/findings.md` |
| 2 | Inspect the AKS `1` class | — | **done — it is real cones, merge it** |
| 3 | Merge: collapse to 1 class, phash dedupe, split by prefix | `scripts/merge.py` | **done** — 13,847 imgs / 227,317 boxes |
| 4 | Baseline: **yolo26n-p2 @ 1280**, 100 epochs, close_mosaic 10 | `scripts/train.py` | script ready — **needs a CUDA box** |
| 5 | Eval: recall by bbox-height bucket | `scripts/eval_buckets.py` | **script ready + self-tested** |
| 6 | Export TensorRT, benchmark **end-to-end** latency on Jetson | `scripts/export_trt.py` | after 5 |
| 7 | Quantify loss at 640 / 960 vs the 1280 baseline | `scripts/sweep.py` | after 6 |

Scripts for tasks 2–7 are written once the previous task tells us what the data
actually looks like — guessing at a merge before seeing the class histogram is
how you silently drop half a dataset.

## Datasets

Drop the downloads in place, **in YOLO format**:

```
data/raw/fsoco/     # FSOCO — 5 classes, all collapse to `cone`
data/raw/aks/       # AKS cone set (Roboflow Universe) — 2 classes: `cone` + junk class `1`
data/merged/        # written by scripts/merge.py — never edit by hand
```

**Overlap is much bigger than expected:** 2,997 AKS images (56.5%) stem-match an
FSOCO image — prefixes `amz`, `bme`, `gfr`, `ulm`, `mms`, `frt`, `ka`, `tuwr`,
`pwrrt` are all FSOCO teams. Roboflow re-encodes and renames
(`bme_00625_jpg.rf.<hash>.jpg`), so phash is still needed to catch the rest.

**And the overlapping copies disagree:** AKS is missing 66,238 boxes relative to
FSOCO on those same images — it labels only the large near cones. See
`notes/findings.md`. The merge must prefer FSOCO labels, not just deduplicate.

**Splitting:** by **lowercased filename prefix** (= source team), never randomly
and never by folder — folder is the *labeling* team, and `amz_*` alone spans 20
folders. Both sets are video-derived; adjacent frames are near-duplicates, so a
random split leaks val into train and reports a recall we do not have.
`amz` is 34% of FSOCO, which constrains how the prefixes can be balanced.
Folder `prom` holds all 98 issue-tagged images → train only, never val/test.

## Setup

**Already done.** The `cone` env exists and is verified:

```bash
conda activate cone
python scripts/env_check.py      # verifies the whole stack in one shot
```

| | |
|---|---|
| env | `cone` — Python 3.11.16 (base miniforge is 3.14, which has no torch/ultralytics wheels) |
| torch | 2.14.0, **MPS** on the M4 |
| ultralytics | 8.4.155 |
| weights | `weights/yolo26n.pt`, `weights/yolo11n.pt` — cached locally |

`yolo26n` resolves and loads (2.57M params), so the YOLO11n fallback is
insurance, not the plan. `yolo26n-p2.yaml` builds too (2.66M params), so the
`n-p2@960` sweep entry is good.

Verified end-to-end on a synthetic cone set: **train → val → ONNX export** all
run clean on MPS. Task 1 itself needs none of this — `inventory.py` is
stdlib-only.

## imgsz — decided from the data, not a sweep

Post-letterbox box-height distribution, share clearing the ~10px labeling floor:

| imgsz | usable boxes |
|---|---|
| 640 | 30.6% |
| 960 | 47.1% |
| **1280** | **60.7%** |

→ **Train at imgsz 1280 with the P2 head.** 640/960 are run only to quantify
what they cost, not as candidates. Budget VRAM accordingly — 1280 + P2 is
memory-hungry.

Queued experiments once a baseline exists:
- filter boxes `<6px` out of training (14.2% of labels at 1280)
- `rect=True` — the most common source resolution is 2872x920, a 3.1:1 ultrawide

The ~10px floor is FSOCO's labeling policy ("100 square pixels"), so **recall
below 10px is not measurable** — absence of a label there does not mean absence
of a cone. The `<10px` bucket stays in the eval as a diagnostic, not a target.

## Negative results — do not retry

- **`yolo26n-p2 @ imgsz 1280` does not train on M4 MPS.** A probe on a 411-image
  slice (4% of train, batch 4) did not finish a single epoch in 20+ minutes.
  Extrapolated, one full epoch is measured in hours and the 100-epoch baseline
  is weeks. The P2 head adds a stride-4 feature map — at 1280 that is a
  320×320 grid, and MPS has no efficient path for it.
  → **Training happens on a cluster GPU.** `scripts/train.py` takes `--device`
  and `--batch` so the same script runs both places; the Mac is for data prep,
  eval and export only.

## Gotchas already paid for

- **`path:` in a dataset yaml must be absolute.** Ultralytics resolves a
  relative `path:` against the *current working directory* (then against
  `datasets_dir`), **not** against the yaml's own location. A relative path
  silently points somewhere else depending on where you launch from.
- **`runs_dir` was pointing at `/opt/homebrew/runs`.** Now set to
  `runs/` inside the project, so runs stay with the code they came from.
  Same for `datasets_dir` and `weights_dir` (`yolo settings`).
- **AKS `names: ['1', 'cone']` — id 0 is `'1'`, id 1 is `cone`.** Reversed from
  intuition; assuming `0 == cone` inverts the dataset.
- **TensorRT export cannot happen on this Mac.** Task 6 builds the engine
  *on the Jetson*; the Mac side only gets as far as ONNX. Engines are not
  portable across devices or TensorRT versions anyway.

## Training

`scripts/train.py` hardcodes nothing about the machine — same script locally and
on a cluster.

```bash
# see the resolved config without training
python scripts/train.py --dry-run

# the baseline, on a cluster GPU
python scripts/train.py --device 0 --batch -1 --name base_p2_1280

# multi-GPU
python scripts/train.py --device 0,1,2,3 --batch 32 --name base_p2_1280

# quick plumbing check anywhere
python scripts/train.py --model yolo26n.pt --imgsz 640 --batch 8 \
    --epochs 1 --fraction 0.02 --name smoke
```

Defaults are the decided policy: `yolo26n-p2.yaml`, `imgsz 1280`,
`close_mosaic 10`, `fliplr 0.5`, `hsv_h 0.0`. `--device auto` resolves
cuda > mps > cpu; `--batch -1` is AutoBatch and falls back to 4 off CUDA.
The script refuses to start a P2 run at imgsz≥1280 on MPS rather than let you
discover the negative result again.

Queued experiments have flags already: `--min-box-px` (in `merge.py`) and
`--rect`.

## Eval

`scripts/eval_buckets.py` — recall per size bucket, which is the number that
decides things. Overall mAP is logged for context and otherwise ignored.

```bash
python scripts/eval_buckets.py --weights runs/base_p2_1280/weights/best.pt
python scripts/eval_buckets.py --weights ... --conf 0.05,0.1,0.25,0.5 --split test
python scripts/eval_buckets.py --self-test        # no model, no data needed
```

**Buckets: `10-16` · `16-32` · `32-64` · `>=64` px**, measured in
**post-letterbox pixels at the eval imgsz** — source resolutions run 2872x920
to 6304x4304, so a source-pixel bucket means nothing.

**Buckets floor at 10px.** FSOCO's labeling policy is ~100 square pixels, so
below ~10px a missing label does not mean a missing cone. Sub-floor boxes are
counted and reported but **not scored** — and a prediction landing on one is
neither credited nor penalised, since that region is effectively unlabelled.

How much of the val set is even scorable depends entirely on imgsz:

| imgsz | scorable | sub-floor (unscorable) |
|---|---|---|
| 640 | 34.7% | 65.3% |
| 960 | 52.9% | 47.1% |
| **1280** | **68.9%** | 31.1% |

`--conf` sweeps confidence so the recall/confidence trade-off is visible per
bucket; `--log-conf` picks which one is written to `results/runs.csv`. Full
sweep goes to `results/buckets_<run>_<split>.csv`.

## Run log

Every run appends config + bucketed recall to `results/runs.csv`
(`scripts/runlog.py`), starting from run one. Per-dataset inventories land in
`results/inventory_<name>.json`.

## Training defaults

- `model=yolo26n-p2.yaml`, `imgsz=1280` — see the imgsz section above
- keep `fliplr` — a cone is symmetric, the flip is free data
- `hsv_h` ≈ 0 — cone hue is the signal; do not rotate it away
- `close_mosaic 10`
- recall over precision; no high confidence threshold — false positives are
  filtered temporally downstream
