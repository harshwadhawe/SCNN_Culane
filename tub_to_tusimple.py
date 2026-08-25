#!/usr/bin/env python
"""Generate SCNN-ready lane masks for a donkeycar tub, with no hand labelling.

    python tub_to_tusimple.py --tub tub_generated_track --dest tub_dataset
    python tub_to_tusimple.py --tub tub_generated_track --dest tub_dataset --limit 500 --preview

Geometry and thresholds come from lane_detection_tub.ipynb (bird's-eye warp,
HSV yellow, top-hat white, sliding windows). Two deliberate differences:

* Only the yellow centre line is labelled. The white edges are picked up by the
  top-hat but land on gravel and shadows often enough to poison training; the
  yellow line is found on ~96% of frames and tracks cleanly.
* seg_label/ and the list files are written directly rather than via
  Tusimple.generate_label(). That helper assigns a lane to class 2 or 3 by its
  slope, which would flip a single centre line between classes as the track
  curves. Writing them here pins it to class 1 always.

Output is what dataset/Tusimple.py reads, so scnn_tusimple.py trains on it
unchanged. Episodes are split whole -- consecutive frames are ~50ms apart and a
random split would leak near-duplicates into validation.
"""
import argparse, csv, json, shutil, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

FRAME_W, FRAME_H = 128, 120
SRC = np.float32([(42, 60), (-10, 104), (82, 60), (134, 104)])
WARP_W = WARP_H = 160
DST = np.float32([[0, 0], [0, WARP_H], [WARP_W, 0], [WARP_W, WARP_H]])
M = cv2.getPerspectiveTransform(SRC, DST)
M_INV = cv2.getPerspectiveTransform(DST, SRC)

YELLOW_LOWER, YELLOW_UPPER = np.array([20, 60, 80]), np.array([40, 255, 255])
WIN_H, WIN_MARGIN, N_WINDOWS = 16, 25, 10
MIN_POINTS = 4              # fewer than this and the fit is not trustworthy
SEG_WIDTH = 4               # drawn line thickness in source pixels
MAX_RESID = 6.0             # px; a worse polynomial fit means the tracker wandered


def yellow_points(bgr):
    """Track the yellow centre line in bird's-eye space; returns source-space points."""
    warped = cv2.warpPerspective(bgr, M, (WARP_W, WARP_H))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = cv2.medianBlur(cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER), 5)

    col = mask[WARP_H // 2:, :].sum(axis=0)
    if col.max() < 255 * 3:
        return None
    base, xs, ys = int(np.argmax(col)), [], []

    y = WARP_H
    for _ in range(N_WINDOWS):
        y0, y1 = max(0, y - WIN_H), y
        if y0 >= y1:
            break
        x0, x1 = max(0, base - WIN_MARGIN), min(WARP_W, base + WIN_MARGIN)
        cs, _ = cv2.findContours(mask[y0:y1, x0:x1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            m = cv2.moments(max(cs, key=cv2.contourArea))
            if m["m00"]:
                base = x0 + int(m["m10"] / m["m00"])
                xs.append(base); ys.append((y0 + y1) // 2)
        y -= WIN_H

    if len(xs) < MIN_POINTS:
        return None

    # Fit in bird's-eye space, where the line is close to straight, then sample densely.
    # Joining the raw window centroids leaves kinks, and stops at whichever window last
    # found something rather than spanning the visible line.
    xs, ys = np.array(xs, float), np.array(ys, float)
    fit = np.polyfit(ys, xs, 2 if len(xs) >= 5 else 1)
    if np.sqrt(np.mean((np.polyval(fit, ys) - xs) ** 2)) > MAX_RESID:
        return None

    yy = np.linspace(ys.min(), min(ys.max() + WIN_H, WARP_H), 40)
    pts = cv2.perspectiveTransform(
        np.float32([[[x, y]] for x, y in zip(np.polyval(fit, yy), yy)]), M_INV).reshape(-1, 2)
    pts = pts[(pts[:, 0] > -FRAME_W) & (pts[:, 0] < 2 * FRAME_W)
              & (pts[:, 1] >= 0) & (pts[:, 1] < FRAME_H)]
    return pts if len(pts) >= 5 else None


def draw_mask(pts):
    """Source-resolution segmentation mask, centre line as class 1."""
    seg = np.zeros((FRAME_H, FRAME_W), np.uint8)
    order = pts[np.argsort(-pts[:, 1])]                               # bottom of frame upward
    ipts = np.round(order).astype(np.int32)
    cv2.polylines(seg, [ipts], False, 1, SEG_WIDTH)
    return seg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tub", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames (for a trial run)")
    ap.add_argument("--val-episodes", type=int, default=2, help="whole episodes held out for val")
    ap.add_argument("--test-episodes", type=int, default=1)
    ap.add_argument("--preview", action="store_true", help="write preview.png and stop")
    args = ap.parse_args()

    tub, dest = args.tub.expanduser().resolve(), args.dest.expanduser().resolve()
    telem = {r["frame"]: r for r in csv.DictReader(open(tub / "telemetry.csv"))}
    frames = sorted(p for p in tub.glob("*.jpg"))
    if args.limit:
        frames = frames[:args.limit]
    print(f"{len(frames)} frames, {len({r['episode_id'] for r in telem.values()})} episodes")

    if args.preview:
        picks = frames[::max(1, len(frames) // 6)][:6]
        tiles = []
        for p in picks:
            bgr = cv2.imread(str(p)); vis = bgr.copy()
            pts = yellow_points(bgr)
            if pts is not None:
                vis[draw_mask(pts) == 1] = (0, 255, 255)
            tiles.append(np.hstack([cv2.resize(bgr, (256, 240), interpolation=cv2.INTER_NEAREST),
                                    cv2.resize(vis, (256, 240), interpolation=cv2.INTER_NEAREST)]))
        cv2.imwrite("preview.png", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
        print("wrote preview.png")
        return

    by_ep = defaultdict(list)
    for p in frames:
        row = telem.get(p.name)
        if row:
            by_ep[row["episode_id"]].append(p)
    # Hold out whole episodes, smallest first, so the bulk of the data stays in train.
    eps = sorted(by_ep, key=lambda e: len(by_ep[e]))
    test_eps = set(eps[:args.test_episodes])
    val_eps = set(eps[args.test_episodes:args.test_episodes + args.val_episodes])
    eps = sorted(by_ep, key=lambda e: -len(by_ep[e]))
    split_of = lambda e: "test" if e in test_eps else "val" if e in val_eps else "train"
    print("episodes ->", {e: split_of(e) for e in eps})

    (dest / "seg_label" / "list").mkdir(parents=True, exist_ok=True)
    lists, kept, dropped = defaultdict(list), 0, 0

    for ep in eps:
        for p in by_ep[ep]:
            bgr = cv2.imread(str(p))
            pts = None if bgr is None else yellow_points(bgr)
            if pts is None:
                dropped += 1
                continue
            stem = p.stem
            rel_img = f"clips/tub/{stem}/20.jpg"
            rel_seg = f"seg_label/tub/{stem}/20.png"
            for rel in (rel_img, rel_seg):
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            if not (dest / rel_img).exists():
                shutil.copyfile(p, dest / rel_img)
            cv2.imwrite(str(dest / rel_seg), draw_mask(pts))
            lists[split_of(ep)].append(f"/{rel_img} /{rel_seg} 1 0 0 0")
            kept += 1

    for split, rows in lists.items():
        (dest / "seg_label" / "list" / f"{split}_gt.txt").write_text("\n".join(rows) + "\n")
        print(f"  {split:<6} {len(rows)} frames")

    print(f"\nkept {kept}, dropped {dropped} ({100*dropped/max(kept+dropped,1):.1f}% no centre line)")
    print(f"train with:\n  python scnn_tusimple.py --data {dest} --resize 128 128")


if __name__ == "__main__":
    main()
