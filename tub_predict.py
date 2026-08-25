#!/usr/bin/env python
"""Render lane predictions from a checkpoint over tub frames.

    python tub_predict.py --ckpt experiments/tub/best.pth --data ../tub_dataset
    python tub_predict.py --ckpt experiments/tub/best.pth --data ../tub_dataset -n 24 --split test
    python tub_predict.py --ckpt experiments/tub/best.pth --tub ../tub_generated_track --stride 40

Defaults to the held-out test split, so what you are looking at is frames the model
never trained on. Each row is: original | pseudo-label | model prediction, which
also shows whether the model is cleaner than the labels it learned from.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from model import SCNN
from utils.transforms import Compose, Resize, ToTensor, Normalize

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# class 1 left edge, 2 centre, 3 right edge -- BGR, since these go through imwrite.
# Deliberately not yellow or white: the centre line is already yellow and the edges
# already white, so matching colours would render the overlay invisible.
COLORS = [(255, 255, 0), (255, 0, 255), (0, 255, 0)]      # cyan, magenta, green


def overlay(bgr, mask, alpha=0.75):
    lane = np.zeros_like(bgr)
    for cls, col in enumerate(COLORS, start=1):
        lane[mask == cls] = col
    return cv2.addWeighted(lane, alpha, bgr, 1.0, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=None,
                    help="dataset built by tub_to_tusimple.py; uses its held-out split")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--tub", type=Path, default=None, help="raw tub, if --data is not given")
    ap.add_argument("--stride", type=int, default=None, help="frame spacing (default: even spread)")
    ap.add_argument("-n", "--num", type=int, default=12)
    ap.add_argument("--resize", type=int, nargs=2, default=[512, 288], metavar=("W", "H"),
                    help="must match what the checkpoint was trained at")
    ap.add_argument("--out", type=Path, default=Path("tub_predictions.png"))
    ap.add_argument("--video", type=Path, default=None,
                    help="write an mp4 instead of a grid. Frames are taken in order, so "
                         "with --data the split is a contiguous slice of the drive.")
    ap.add_argument("--fps", type=int, default=20, help="tub frames are ~50ms apart")
    ap.add_argument("--thresh", type=float, default=0.5, help="existence threshold")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    resize = tuple(args.resize)

    net = SCNN(input_size=resize, pretrained=False)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(sd.get("net", sd))
    net = net.to(device).eval()
    print(f"{args.ckpt} (epoch {sd.get('epoch', '?')}) on {device}, input {resize[0]}x{resize[1]}")

    # frames, with their pseudo-label mask when we have one. For video take them in
    # order and keep every one; for a grid, sample across the range.
    pairs = []
    if args.data is not None:
        listing = args.data / "seg_label" / "list" / f"{args.split}_gt.txt"
        rows = [l.split() for l in listing.read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: r[0])
        keep = rows if args.video else rows[::(args.stride or max(1, len(rows) // args.num))][:args.num]
        pairs = [(args.data / r[0][1:], args.data / r[1][1:]) for r in keep]
        print(f"{len(pairs)} frames from the {args.split} split ({len(rows)} available)")
    else:
        frames = sorted(args.tub.glob("*.jpg"))
        keep = frames[::args.stride] if args.stride else (
            frames if args.video else frames[::max(1, len(frames) // args.num)][:args.num])
        pairs = [(p, None) for p in keep]
        print(f"{len(pairs)} frames from {args.tub}")

    to_net = Compose(ToTensor(), Normalize(MEAN, STD))
    rows_img, writer = [], None
    with torch.no_grad():
        for img_path, lab_path in pairs:
            bgr = cv2.imread(str(img_path))
            shown = Resize(resize)({"img": bgr})["img"]
            x = to_net({"img": cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)})["img"][None].to(device)
            seg, exist = net(x)[:2]
            pred = torch.softmax(seg.float(), 1)[0].cpu().numpy().argmax(0)
            e = exist.float()[0].cpu().numpy()
            for cls in range(1, 4):
                if e[cls - 1] <= args.thresh:
                    pred[pred == cls] = 0

            tiles = [shown]
            if lab_path is not None and lab_path.exists() and not args.video:
                lab = cv2.resize(cv2.imread(str(lab_path))[:, :, 0], resize,
                                 interpolation=cv2.INTER_NEAREST)
                tiles.append(overlay(shown, lab))
            tiles.append(overlay(shown, pred))

            if args.video:
                frame = np.hstack(tiles)
                if frame.shape[1] < 768:        # a 128px model would otherwise be unwatchable
                    k = int(np.ceil(768 / frame.shape[1]))
                    frame = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
                cv2.putText(frame, "input", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"prediction  exist {(e > args.thresh).astype(int)[:3]}",
                            (resize[0] + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if writer is None:
                    writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise SystemExit(f"could not open {args.video} for writing")
                writer.write(frame)
                continue

            row = np.hstack([cv2.resize(t, (320, 180)) for t in tiles])
            cv2.putText(row, f"{img_path.parent.name}  exist {(e > args.thresh).astype(int)[:3]}",
                        (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            rows_img.append(cv2.copyMakeBorder(row, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(60, 60, 60)))

    if writer is not None:
        writer.release()
        print(f"wrote {args.video}  ({len(pairs)} frames at {args.fps} fps, "
              f"{len(pairs)/args.fps:.0f}s, input | prediction)")
    else:
        cv2.imwrite(str(args.out), np.vstack(rows_img))
        cols = "original | pseudo-label | prediction" if pairs[0][1] else "original | prediction"
        print(f"wrote {args.out}  ({cols})")


if __name__ == "__main__":
    main()
