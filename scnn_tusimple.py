#!/usr/bin/env python
"""Train SCNN on TuSimple, unattended.

    tmux new -s scnn
    python scnn_tusimple.py --data /data/tusimple --exp-dir experiments/tusimple_4060
    # ctrl-b d to detach; tmux attach -t scnn to come back
    tail -f experiments/tusimple_4060/train.log

Resume after a kill (SIGINT/SIGTERM checkpoint before exiting):
    python scnn_tusimple.py --data /data/tusimple --exp-dir experiments/tusimple_4060 --resume

Differs from train.py deliberately: no tensorboard (utils/tensorboard.py imports
tensorflow and the long-removed scipy.misc), CUDA/MPS/CPU autodetect, optional AMP,
periodic line-oriented logging rather than a progress bar, and signal-safe exit.
"""
import argparse, json, os, signal, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import dataset
from model import SCNN
from utils.lr_scheduler import PolyLR
from utils.transforms import Compose, Resize, Rotation, ToTensor, Normalize
from utils.prob2lines import getLane

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet, as train.py uses
stop_requested = False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="TuSimple root (see download_tusimple.py)")
    p.add_argument("--exp-dir", type=Path, default=REPO / "experiments" / "tusimple")
    p.add_argument("--batch-size", type=int, default=None, help="default: 32 on CUDA, 4 otherwise")
    p.add_argument("--resize", type=int, nargs=2, default=[512, 288], metavar=("W", "H"))
    p.add_argument("--lr", type=float, default=None, help="default: 0.15 scaled linearly from batch 32")
    p.add_argument("--max-iter", type=int, default=1500, help="exp0 reference; PolyLR counts iterations")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--log-every", type=int, default=20, help="iterations between progress lines")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="skip training, score the best checkpoint")
    p.add_argument("--no-eval", action="store_true", help="train only, skip the final LaneEval")
    return p.parse_args()


def pick_device():
    if torch.cuda.is_available():
        # input size is fixed all run, so let cuDNN pick algorithms once and reuse them
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe(device):
    if device.type == "cuda":
        pr = torch.cuda.get_device_properties(0)
        n = torch.cuda.device_count()
        return f"cuda: {pr.name}, {pr.total_memory/2**30:.1f} GiB" + (f" x{n}" if n > 1 else "")
    return f"{device.type}: no dedicated VRAM limit tracked"


def autoscale_batch(device, resize, amp):
    """Fit the batch to VRAM.

    Calibrated against an RTX 4060 (7799 MiB): batch 26 at 512x288 with autocast
    OOMs, implying ~287 MB per image. Autocast saves roughly 30% over fp32 rather
    than half -- fp32 master weights and cuDNN workspace do not shrink. Scale
    linearly with pixel count and keep headroom for context, weights, SGD momentum
    and cudnn.benchmark's algorithm search.
    """
    if device.type != "cuda":
        return 4
    total_mb = torch.cuda.get_device_properties(0).total_memory / 2**20
    px_ratio = (resize[0] * resize[1]) / (512 * 288)
    per_img = (300 if amp else 430) * px_ratio
    usable = total_mb * 0.85 - 800           # context + weights + optimizer + cudnn workspace
    return int(max(2, min(32, usable // per_img)))


class Log:
    """stdout + file, line-buffered so tail -f and tmux capture-pane both work."""

    def __init__(self, path):
        self.fh = open(path, "a", buffering=1)

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")


def on_signal(signum, _frame):
    global stop_requested
    stop_requested = True
    print(f"\nsignal {signum} received — finishing this iteration, then checkpointing", flush=True)


def build_loaders(args, log):
    resize = tuple(args.resize)
    t_train = Compose(Resize(resize), Rotation(2), ToTensor(), Normalize(MEAN, STD))
    t_eval = Compose(Resize(resize), ToTensor(), Normalize(MEAN, STD))

    log("building datasets (first run generates seg_label/, which takes a while)")
    Ds = dataset.Tusimple
    splits = {s: Ds(str(args.data), s, t_train if s == "train" else t_eval)
              for s in ("train", "val", "test")}
    log("  " + " | ".join(f"{k} {len(v)}" for k, v in splits.items()))

    kw = dict(collate_fn=Ds.collate, num_workers=args.workers,
              pin_memory=args.pin_memory, persistent_workers=args.workers > 0)
    return (DataLoader(splits["train"], batch_size=args.batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(splits["val"], batch_size=args.batch_size, **kw),
            DataLoader(splits["test"], batch_size=args.batch_size, **kw))


def save(path, core, optimizer, sched, scaler, epoch, best):
    torch.save({"epoch": epoch, "net": core.state_dict(), "optim": optimizer.state_dict(),
                "lr_scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "best_val_loss": best}, path)


@torch.no_grad()
def validate(net, loader, device, amp):
    net.eval()
    total, n = 0.0, 0
    for s in loader:
        with torch.autocast(device.type, enabled=amp):
            loss = net(s["img"].to(device), s["segLabel"].to(device), s["exist"].to(device))[4]
        total += loss.sum().item() if loss.dim() else loss.item()
        n += 1
    return total / max(n, 1)


def run_eval(net, loader, device, amp, exp_dir, data_root, log):
    """Inference -> predict_test.json -> LaneEval, mirroring test_tusimple.py."""
    from utils.lane_evaluation.tusimple.lane import LaneEval   # needs scikit-learn

    net.eval()
    out_dir = exp_dir / "coord_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    dumped, t0 = [], time.time()

    with torch.no_grad():
        for i, s in enumerate(loader):
            with torch.autocast(device.type, enabled=amp):
                seg, exist = net(s["img"].to(device))[:2]
            seg = torch.softmax(seg.float(), dim=1).cpu().numpy()
            exist = exist.float().cpu().numpy()

            for b in range(len(seg)):
                flags = [int(exist[b, k] > 0.5) for k in range(4)]
                lanes = getLane.prob2lines_tusimple(seg[b], flags, resize_shape=(720, 1280),
                                                    y_px_gap=10, pts=56)
                lanes = [sorted(l, key=lambda p: p[1]) for l in lanes]
                parts = Path(s["img_name"][b]).parts[-4:]
                rec = {"raw_file": "/".join(parts), "run_time": 0,
                       "lanes": [[int(x) for x, _ in l] for l in lanes if l],
                       "h_sample": [y for _, y in lanes[0]] if lanes else []}
                dumped.append(json.dumps(rec))
            if i % 20 == 0:
                log(f"  eval batch {i}/{len(loader)}")

    pred = out_dir / "predict_test.json"
    pred.write_text("\n".join(dumped) + "\n")
    log(f"wrote {len(dumped)} predictions in {time.time()-t0:.0f}s -> {pred}")

    result = LaneEval.bench_one_submit(str(pred), str(data_root / "test_label.json"))
    (exp_dir / "evaluation_result.txt").write_text(result)
    log("LaneEval:\n" + result)
    return result


def main():
    args = parse_args()
    device = pick_device()
    is_cuda = device.type == "cuda"

    # hardware-dependent defaults, all in one place
    if args.batch_size is None:
        args.batch_size = autoscale_batch(device, tuple(args.resize),
                                          is_cuda if args.amp is None else args.amp)
    if args.workers is None:
        args.workers = 8 if is_cuda else 0
    if args.amp is None:
        args.amp = is_cuda
    args.pin_memory = is_cuda
    if args.lr is None:
        # exp0 used lr 0.15 at batch 32; linear scaling keeps the step size sane elsewhere
        args.lr = 0.15 * args.batch_size / 32

    if args.warmup >= args.max_iter:          # PolyLR would spend the whole run warming up
        args.warmup = max(1, args.max_iter // 4)

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    log = Log(args.exp_dir / "train.log")
    log("=" * 72)
    log(describe(device))
    log(f"batch {args.batch_size} | lr {args.lr:.4f} | amp {args.amp} | workers {args.workers}")
    log(f"resize {tuple(args.resize)} | max_iter {args.max_iter} | exp_dir {args.exp_dir}")
    (args.exp_dir / "cfg.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2))

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    train_loader, val_loader, test_loader = build_loaders(args, log)

    core = SCNN(input_size=tuple(args.resize), pretrained=True).to(device)
    net = core
    if is_cuda and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(core)
        log(f"DataParallel over {torch.cuda.device_count()} GPUs")

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=1e-4, nesterov=True)
    sched = PolyLR(optimizer, 0.9, max_iter=args.max_iter, warmup=args.warmup, min_lrs=1e-10)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)

    latest, best_path = args.exp_dir / "latest.pth", args.exp_dir / "best.pth"
    start_epoch, best_val = 0, float("inf")

    if args.resume or args.eval_only:
        ckpt_path = best_path if args.eval_only else latest
        if not ckpt_path.exists():
            log(f"nothing to load at {ckpt_path}")
            sys.exit(1)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        core.load_state_dict(ck["net"])
        if not args.eval_only:
            optimizer.load_state_dict(ck["optim"])
            sched.load_state_dict(ck["lr_scheduler"])
            # _LRScheduler.state_dict() stores every attribute, so the load above silently
            # restores the previous run's max_iter/warmup/base_lrs and a longer --max-iter
            # would be ignored (leaving lr pinned at min_lr). CLI flags win.
            sched.max_iter, sched.warmup = args.max_iter, args.warmup
            sched.base_lrs = [args.lr] * len(optimizer.param_groups)
            scaler.load_state_dict(ck["scaler"])
            start_epoch, best_val = ck["epoch"] + 1, ck.get("best_val_loss", float("inf"))
        log(f"loaded {ckpt_path.name} (epoch {ck['epoch']}, best val {ck.get('best_val_loss', float('nan')):.4f})")

    if args.eval_only:
        run_eval(net, test_loader, device, args.amp, args.exp_dir, args.data, log)
        return

    iters_per_epoch = len(train_loader)
    epochs = int(np.ceil(args.max_iter / max(iters_per_epoch, 1)))
    log(f"{iters_per_epoch} iters/epoch -> {epochs} epochs for max_iter={args.max_iter}")

    it = start_epoch * iters_per_epoch
    t_start = time.time()
    csv = open(args.exp_dir / "history.csv", "a", buffering=1)
    if csv.tell() == 0:
        csv.write("epoch,iter,train_loss,val_loss,lr,elapsed_s\n")

    for epoch in range(start_epoch, epochs):
        net.train()
        run, seen = 0.0, 0
        t_epoch = time.time()

        for i, s in enumerate(train_loader):
            img, seg, ex = (s["img"].to(device, non_blocking=True),
                            s["segLabel"].to(device, non_blocking=True),
                            s["exist"].to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device.type, enabled=args.amp):
                    loss = net(img, seg, ex)[4]
                if loss.dim():                   # DataParallel returns one loss per shard
                    loss = loss.sum()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            except torch.cuda.OutOfMemoryError:
                b = max(2, int(args.batch_size * 0.7))
                log(f"CUDA OOM at batch {args.batch_size}. Rerun with:\n"
                    f"  --batch-size {b} --max-iter {int(args.max_iter * args.batch_size / b)}"
                    + ("  --amp" if not args.amp else "")
                    + "\n(the larger --max-iter keeps the same number of images seen)")
                sys.exit(2)
            sched.step()

            if not torch.isfinite(loss):
                log(f"loss went non-finite at iteration {it}. Training diverged — lower "
                    f"--lr (now {args.lr:.4f}) or raise --warmup (now {args.warmup}). "
                    f"latest.pth still holds epoch {epoch-1}.")
                sys.exit(3)

            run += loss.item(); seen += 1; it += 1
            if i % args.log_every == 0:
                done = it / max(args.max_iter, 1)
                eta = (time.time() - t_start) / max(done, 1e-9) * (1 - done)
                log(f"ep {epoch}/{epochs-1} it {it}/{args.max_iter} "
                    f"loss {run/seen:.4f} lr {optimizer.param_groups[0]['lr']:.5f} "
                    f"eta {eta/60:.0f}m")
            if stop_requested:
                break

        train_loss = run / max(seen, 1)
        val_loss = validate(net, val_loader, device, args.amp)
        log(f"epoch {epoch} done in {time.time()-t_epoch:.0f}s | train {train_loss:.4f} | val {val_loss:.4f}")
        csv.write(f"{epoch},{it},{train_loss:.6f},{val_loss:.6f},"
                  f"{optimizer.param_groups[0]['lr']:.8f},{time.time()-t_start:.0f}\n")

        save(latest, core, optimizer, sched, scaler, epoch, best_val)
        if val_loss < best_val:
            best_val = val_loss
            save(best_path, core, optimizer, sched, scaler, epoch, best_val)
            log(f"  new best ({best_val:.4f}) -> {best_path.name}")

        if stop_requested:
            log("stopping early on signal; checkpoint written, rerun with --resume")
            csv.close()
            return

    csv.close()
    log(f"training finished in {(time.time()-t_start)/60:.1f} min, best val {best_val:.4f}")

    if not args.no_eval:
        ckpt = best_path if best_path.exists() else latest
        if ckpt is not best_path:
            log(f"no {best_path.name} (validation never beat inf); scoring {ckpt.name} instead")
        core.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["net"])
        run_eval(net, test_loader, device, args.amp, args.exp_dir, args.data, log)


if __name__ == "__main__":
    main()
