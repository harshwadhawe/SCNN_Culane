# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Pytorch re-implementation of SCNN (Spatial CNN) lane detection, ported from the official lua-torch repo. Supports two datasets: CULane and Tusimple. `README.md` has dataset download links, expected directory layouts, and pretrained-model links.

## Paths that must be edited before anything runs

Absolute paths are hardcoded in two places and are almost certainly wrong on a fresh checkout:

- `config.py` — `Dataset_Path['CULane']` and `Dataset_Path['Tusimple']`
- `utils/lane_evaluation/CULane/Run.sh` — `root` (absolute project path) and `data_dir` (CULane path)

## Commands

Local env is a uv venv at `.venv/` (Python 3.12, torch 2.13 + torchvision 0.28, MPS available).
Prefix with `.venv/bin/python` or `source .venv/bin/activate` first.

`scnn_notebook.ipynb` walks the whole pipeline (data → model → train → checkpoint → test →
prob2lines → evaluation) in small cells with inline figures. It runs against a real dataset when
`config.py` points at one, and otherwise writes a toy dataset in CULane's on-disk format under
`toy_data/` so every cell still exercises the real code path. It uses `experiments/exp_notebook/`
as its exp_dir, leaving `exp0`/`exp10` alone.

`scnn_tusimple_colab.ipynb` is the Colab/A100 path: it clones the repo, fetches TuSimple to local
disk (never a mounted Drive — per-epoch JPEG reads dominate otherwise), and drives
`scnn_tusimple.py` rather than reimplementing the loop. All hardware branching is confined to the `RUNTIME`
dict in section 0 (batch size, workers, `pin_memory`, AMP, DataParallel); CUDA and MPS/CPU differ
only there. Note `Subset` does not forward `.collate`, so its loaders use `Dataset_Type.collate`.

The three README checkpoints are downloaded (`gdown`, ids in the notebook's `PUBLISHED` dict) and
all load with `strict=True`: `experiments/vgg_SCNN_DULR_w9/vgg_SCNN_DULR_w9.pth` (official CULane
conversion, 800×288), `experiments/exp10/exp10_best.pth` (CULane, 800×288),
`experiments/exp0/exp0_best.pth` (Tusimple, 512×288). Each must be built at its own cfg's
`resize_shape` — `fc_input_feature` depends on it. `vgg_SCNN_DULR_w9` reproduces
`demo/demo_result.jpg` exactly.

```bash
# TuSimple: fetch (Kaggle; the old S3 links are 404) then train under tmux
python download_tusimple.py --dest /data/tusimple      # or --from-zip <already-downloaded.zip>
python download_tusimple.py --dest /data/tusimple --verify-only
python scnn_tusimple.py --data /data/tusimple --exp-dir experiments/tusimple [--resume|--eval-only]

# fetch CULane (~55GB unpacked, resumable; ids verified against the official Drive folder)
python download_culane.py --list                          # plan only
python download_culane.py --dest /path/to/CULane          # everything SCNN needs

# train (reads experiments/<exp>/cfg.json, writes checkpoint + tensorboard there)
python train.py --exp_dir ./experiments/exp0 [--resume/-r]
tensorboard --logdir='experiments/exp0'

# single-image demo (NB: overwrites the committed demo/demo_result.jpg reference)
python demo_test.py -i demo/demo.jpg -w experiments/vgg_SCNN_DULR_w9/vgg_SCNN_DULR_w9.pth [-v]

# full inference + evaluation
python test_tusimple.py --exp_dir ./experiments/exp0
python test_CULane.py  --exp_dir ./experiments/exp10   # requires the C++ evaluator built first

# build the CULane C++ evaluator (needs OpenCV, cmake); binary lands in utils/lane_evaluation/CULane/
cd utils/lane_evaluation/CULane && mkdir build && cd build && cmake .. && make
```

There is no test suite and no linter. `test_*.py` at the repo root are inference+evaluation scripts, not unit tests.

## Architecture

**Experiment-directory convention.** Everything is keyed off `--exp_dir`; the directory basename is the experiment name and also the checkpoint filename. A run produces, inside `exp_dir`: `cfg.json` (input), `<exp_name>.pth` (latest), `<exp_name>_best.pth` (best val loss — this is what the test scripts load), `coord_output/` (per-image predicted lane coords), `evaluate/` (evaluator output), plus tensorboard event files. Adding a new experiment = new directory with a `cfg.json`.

**Config → dataset dispatch.** `cfg.json`'s `dataset.dataset_name` is used as `getattr(dataset, dataset_name)` to pick the `Dataset` class, and as the key into `Dataset_Path` in `config.py`. The remaining `optim` / `lr_scheduler` sub-dicts are splatted directly into `optim.SGD(...)` and `PolyLR(...)`, so their keys must match those constructors.

**Sample-dict pipeline.** Datasets yield `{'img', 'segLabel', 'exist', 'img_name'}` and every transform in `utils/transforms/` takes and returns that dict (`CustomTransform` base, composed with `Compose`). Batching goes through each dataset's static `collate`, which handles the `test` split where `segLabel`/`exist` are `None`.

**Model (`model.py`).** VGG16-BN backbone from torchvision, surgically modified in `net_init`: modules `34/37/40` are replaced with dilation-2 convs and `33`/`43` (maxpools) are popped — these are positional indices into the torchvision module list, so a torchvision version bump can silently break them. Spatial message passing (`message_passing_once`) slices the feature map row-by-row / column-by-column in a Python loop across four directions; that loop is the runtime bottleneck. `fc_input_feature` is derived from `input_size`, so a checkpoint is only loadable at the `resize_shape` it was trained at.

`forward()` computes the loss internally and returns `(seg_pred, exist_pred, loss_seg, loss_exist, loss)`. This is deliberate — it lets `DataParallel` shard the loss — which is why callers do `loss.sum()` when the net is wrapped. Passing no `seg_gt`/`exist_gt` returns zero losses (inference path).

**Evaluation is two-stage and file-based.** The net produces per-class segmentation probabilities; `utils/prob2lines/getLane.py` converts those to discrete lane point lists at the *original* image resolution (CULane 590×1640, Tusimple 720×1280); those are written as `.lines.txt` files under `coord_output/`; then an external evaluator scores them. CULane uses the ported C++ binary via `os.system("sh .../Run.sh <exp_name>")`; Tusimple uses `utils/lane_evaluation/tusimple/lane.py` (`LaneEval.bench_one_submit`) on a dumped `predict_test.json`.

## Gotchas

- `MAX_EPOCHES` in `cfg.json` is **ignored**. `train.py:main` overwrites it with `ceil(lr_scheduler.max_iter / len(train_loader))`. Control training length via `lr_scheduler.max_iter`.
- `PolyLR` steps **per batch**, not per epoch, so `max_iter` and `warmup` are in iterations.
- Both `train.py` and the test scripts do `exp_cfg['dataset'].pop('dataset_name')` at import time — the config dict is mutated as a side effect of module-level code. These scripts run their setup at import, not inside `main()`.
- The lane colour arrays in `train.py`, `test_*.py` and `demo_test.py` are **BGR**, since they end at `cv2.imwrite`. Rendering them as RGB (matplotlib) silently swaps lanes 1/3 and turns lane 4 cyan instead of yellow.
- `demo_test.py` writes over `demo/demo_result.jpg`, which is committed as the reference output — `git checkout demo/demo_result.jpg` to restore.
- `demo_test.py` normalizes with CULane mean/std; `train.py` and the test scripts use ImageNet mean/std. Not interchangeable across checkpoints.
- Tusimple generates `seg_label/` and list files on first `Tusimple(...)` instantiation — slow, and it silently skips regeneration if the directory already exists.
- `utils/prob2lines/` and `utils/lane_evaluation/tusimple/` have no `__init__.py` and rely on Python 3 namespace packages; scripts must be run from the repo root.
- `utils/lane_evaluation/CULane/src/` is the multithreaded evaluator and is what `CMakeLists.txt` builds; `src_origin/` is the unmodified upstream single-threaded version, kept for reference and not compiled.
- `utils/tensorboard.py` imports `tensorflow` and `scipy.misc` (removed from scipy years ago) — it does **not** run on a modern stack. `train.py` imports it at module scope, so training as written needs that file replaced or stubbed.
- `scnn_tusimple.py` is the unattended path (CUDA/MPS/CPU picked at runtime, VRAM-scaled batch, AMP on CUDA, signal-safe checkpointing, resume). It deliberately avoids `utils/tensorboard.py`. `utils/lane_evaluation/tusimple/lane.py` needs `scikit-learn`, which is not in requirements.txt.
- `_LRScheduler.state_dict()` persists **every** attribute, so `load_state_dict` restores the old run's `max_iter`/`warmup`/`base_lrs`. Resuming with a longer schedule needs them reapplied afterwards or the LR sits pinned at `min_lrs`.
- `dataset/Tusimple.py:_gen_label_for_json` assumes `raw_file` is exactly `clips/<date>/<clip>/20.jpg` (4 components) and runs for train/val/**test**, so `test_label.json` must be present or instantiation crashes.
- Training hardcodes `torch.nn.DataParallel` and `num_workers=8`; adjust for single-GPU or CPU machines.
