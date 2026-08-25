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

# Saturation and value separate the centre line from the shoulder, not hue: on this
# track the line sits at H~25 S>120 V~250 while grass and dirt sit at H~30 V~139.
# The notebook's [20,60,80] floor lets the shoulder through, and "widest run per row"
# then locks onto grass instead of the line.
YELLOW_LOWER = np.array([18, 80, 170])
YELLOW_UPPER = np.array([42, 255, 255])

ROAD_TOP = 50               # first row below the horizon
MIN_ROWS = 6                # fewer rows than this and the line is too short to trust
MAX_RESID = 4.0             # px RMS; a worse fit means the tracker wandered off the line
SEG_WIDTH = 4               # drawn line thickness in source pixels


def yellow_points(bgr):
    """Row-wise centroid of the yellow centre line, fitted and resampled.

    Works directly in image space. The bird's-eye warp the notebook uses is tuned
    for a different tub: on this one it pushes the line outside the warp quad on
    ~21% of frames, which shows up as an empty mask even though the line is plainly
    visible in the source.
    """
    mask = cv2.medianBlur(cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
                                      YELLOW_LOWER, YELLOW_UPPER), 3)
    xs, ys = [], []
    for y in range(ROAD_TOP, FRAME_H):
        idx = np.flatnonzero(mask[y] > 0)
        if idx.size == 0:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 3) + 1)
        widest = max(runs, key=len)
        xs.append(widest.mean()); ys.append(y)

    if len(xs) < MIN_ROWS:
        return None
    xs, ys = np.array(xs, float), np.array(ys, float)
    fit = np.polyfit(ys, xs, 2 if len(xs) >= 5 else 1)
    if np.sqrt(np.mean((np.polyval(fit, ys) - xs) ** 2)) > MAX_RESID:
        return None

    yy = np.arange(ys.min(), FRAME_H)                 # extend the fit down to the bumper
    pts = np.stack([np.polyval(fit, yy), yy], axis=1)
    return pts[(pts[:, 0] >= 0) & (pts[:, 0] < FRAME_W)]


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
    need = args.val_episodes + args.test_episodes + 1
    if len(by_ep) >= need:
        # Hold out whole episodes, smallest first, so the bulk stays in train.
        order = sorted(by_ep, key=lambda e: len(by_ep[e]))
        test_eps, val_eps = set(order[:args.test_episodes]), \
                            set(order[args.test_episodes:args.test_episodes + args.val_episodes])
        eps = sorted(by_ep, key=lambda e: -len(by_ep[e]))
        split_of = lambda p, e: "test" if e in test_eps else "val" if e in val_eps else "train"
        print("episodes ->", {e: split_of(None, e) for e in eps})
    else:
        # Too few episodes to hold one out. Split the timeline contiguously instead and
        # drop GAP frames either side of each boundary: frames are ~50ms apart, so
        # neighbours are near-duplicates and would leak across the split.
        eps = sorted(by_ep)
        ordered = [p for e in eps for p in sorted(by_ep[e])]
        n = len(ordered)
        i_val, i_test = int(n * 0.70), int(n * 0.85)
        GAP = 15
        pos = {p: i for i, p in enumerate(ordered)}

        def split_of(p, e):
            i = pos[p]
            if abs(i - i_val) < GAP or abs(i - i_test) < GAP:
                return None                      # boundary buffer, discarded
            return "train" if i < i_val else "val" if i < i_test else "test"

        print(f"{len(by_ep)} episode(s) -- contiguous 70/15/15 split of {n} frames, "
              f"{GAP}-frame buffer at each boundary")

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
            split = split_of(p, ep)
            if split is None:
                dropped += 1
                continue
            lists[split].append(f"/{rel_img} /{rel_seg} 1 0 0 0")
            kept += 1

    for split, rows in lists.items():
        (dest / "seg_label" / "list" / f"{split}_gt.txt").write_text("\n".join(rows) + "\n")
        print(f"  {split:<6} {len(rows)} frames")

    print(f"\nkept {kept}, dropped {dropped} ({100*dropped/max(kept+dropped,1):.1f}% no centre line)")
    print(f"train with:\n  python scnn_tusimple.py --data {dest} --resize 128 128")


if __name__ == "__main__":
    main()
