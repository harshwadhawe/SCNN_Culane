#!/usr/bin/env python
"""Fetch TuSimple and arrange it the way dataset/Tusimple.py expects.

    python download_tusimple.py --dest /data/tusimple                    # via Kaggle
    python download_tusimple.py --dest /data/tusimple --from-zip a.zip b.zip
    python download_tusimple.py --dest /data/tusimple --verify-only

The original S3 links (s3.us-east-2.amazonaws.com/benchmark-frontend/...) are dead
-- all three return 404 -- and the TuSimple issue tracker now points at Kaggle.
The public HuggingFace mirrors are incomplete: they carry the label files but 404
on the clip images those labels reference, so they are not usable either.

Kaggle needs an API token once:
    https://www.kaggle.com/settings -> "Create New Token" -> ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    pip install kaggle

Target layout:
    <dest>/clips/<date>/<clip id>/20.jpg     (only frame 20 is ever labelled)
    <dest>/label_data_0313.json              train
    <dest>/label_data_0601.json              train
    <dest>/label_data_0531.json              val
    <dest>/test_label.json                   test
"""
import argparse, json, shutil, subprocess, sys, zipfile
from pathlib import Path

KAGGLE_DS = "manideep1108/tusimple"
LABELS = ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json", "test_label.json"]
# Kaggle ships the test ground truth as test_label_new.json; dataset/Tusimple.py hardcodes
# TEST_SET = ['test_label.json'], so whichever we find gets normalised to that name.
TEST_ALIASES = ["test_label.json", "test_label_new.json", "test_tasks_0627.json"]


def merge_into(src: Path, dst: Path, stats=None):
    """Move src's entries into dst, merging directories instead of clobbering.

    Returns the number of files left behind because the destination already had
    that name -- nothing is ever overwritten.
    """
    stats = {"kept": 0} if stats is None else stats
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir() and target.exists():
            merge_into(item, target, stats)
            if not any(item.iterdir()):
                item.rmdir()
        elif not target.exists():
            shutil.move(str(item), str(target))
        else:
            stats["kept"] += 1          # same name on both sides; leave the original alone
    return stats["kept"]


def normalise(staging: Path, dest: Path):
    """Pull clips/ and the label json files out of however Kaggle nested them."""
    for clips in sorted(staging.rglob("clips")):
        if clips.is_dir() and clips.resolve() != (dest / "clips").resolve():
            collisions = merge_into(clips, dest / "clips")
            print(f"  merge {clips.relative_to(staging)} -> clips/"
                  + (f"  ({collisions} files already present, originals kept)" if collisions else ""))
    for seg in sorted(staging.rglob("seg_label")):
        if not seg.is_dir() or seg.resolve() == (dest / "seg_label").resolve():
            continue
        if (dest / "seg_label").exists():
            break
        if (seg / "list" / "train_gt.txt").exists():
            print(f"  keep  {seg.relative_to(staging)} -> seg_label/  (skips label generation)")
            shutil.move(str(seg), str(dest / "seg_label"))
        else:
            alt = dest / "seg_label_prebuilt"
            print(f"  note  {seg.relative_to(staging)} has no list/train_gt.txt; parking it at "
                  f"{alt.name}/ so dataset/Tusimple.py regenerates its own")
            shutil.move(str(seg), str(alt))
        break

    for name in LABELS[:3]:
        for found in staging.rglob(name):
            if not (dest / name).exists():
                print(f"  take  {found.relative_to(staging)}")
                shutil.move(str(found), str(dest / name))
            break
    if not (dest / "test_label.json").exists():
        for alias in TEST_ALIASES:
            found = next(staging.rglob(alias), None)
            if found is not None:
                print(f"  take  {found.relative_to(staging)} -> test_label.json")
                shutil.move(str(found), str(dest / "test_label.json"))
                break


def link_from(src: Path, dest: Path):
    """Build the expected tree out of symlinks into a read-only source.

    Kaggle mounts datasets at /kaggle/input read-only, so nothing can be moved or
    written there -- but dataset/Tusimple.py writes seg_label/ into the dataset root.
    Symlinking each clip directory gives a writable root without copying the images.
    Links are per-clip rather than per-date because train_set and test_set both
    contain 0531 and 0601.
    """
    (dest / "clips").mkdir(parents=True, exist_ok=True)
    linked = 0
    for clips in sorted(src.rglob("clips")):
        if not clips.is_dir():
            continue
        for date_dir in sorted(d for d in clips.iterdir() if d.is_dir()):
            out_date = dest / "clips" / date_dir.name
            out_date.mkdir(parents=True, exist_ok=True)
            for clip in date_dir.iterdir():
                target = out_date / clip.name
                if not target.exists():
                    target.symlink_to(clip.resolve())
                    linked += 1
        print(f"  linked {clips.relative_to(src)}")
    for name in LABELS[:3]:
        found = next(src.rglob(name), None)
        if found is not None and not (dest / name).exists():
            shutil.copyfile(found, dest / name)
            print(f"  copied {name}")
    if not (dest / "test_label.json").exists():
        for alias in TEST_ALIASES:
            found = next(src.rglob(alias), None)
            if found is not None:
                shutil.copyfile(found, dest / "test_label.json")
                print(f"  copied {found.name} -> test_label.json")
                break
    print(f"  {linked} clip directories linked")


def export_slim(dest: Path, out: Path):
    """Copy only what SCNN reads: the frames the labels reference, plus the labels.

    The Kaggle archive is ~13 GB because it ships all 20 frames of every clip, but
    only frame 20 is annotated -- 6408 images, ~1.8 GB. Small enough to park on
    Drive and re-copy into each Colab session instead of re-downloading 13 GB.
    seg_label/ comes along when present so the slow first-run generation is skipped.
    """
    out.mkdir(parents=True, exist_ok=True)
    copied = missing = 0
    for name in LABELS:
        src_label = dest / name
        if not src_label.exists():
            continue
        shutil.copyfile(src_label, out / name)
        for line in src_label.read_text().splitlines():
            if not line.strip():
                continue
            rel = json.loads(line)["raw_file"]
            src, dst = dest / rel, out / rel
            if not src.exists():
                missing += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(src, dst)      # resolves symlinks, giving real files
            copied += 1
        print(f"  {name}: {copied} frames so far")

    seg = dest / "seg_label"
    if seg.is_dir():
        print("  copying seg_label/ (skips regeneration on the other machine)")
        shutil.copytree(seg, out / "seg_label", dirs_exist_ok=True)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nexported {copied} frames ({missing} missing) -> {out}  [{size/2**30:.2f} GiB]")
    return missing == 0


def count_frames(dest: Path) -> int:
    """Frames the labels actually reference. rglob("20.jpg") would undercount to zero
    on a symlinked tree, since Path.rglob does not descend into symlinked directories."""
    n = 0
    for name in LABELS:
        path = dest / name
        if path.exists():
            n += sum(1 for l in path.read_text().splitlines() if l.strip())
    return n


def verify(dest: Path) -> bool:
    """Every raw_file a label references must exist. This is what catches bad mirrors."""
    ok = True
    for name in LABELS:
        path = dest / name
        if not path.exists():
            print(f"  MISSING  {name}")
            ok = False
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        missing = [r["raw_file"] for r in rows if not (dest / r["raw_file"]).exists()]
        flag = "ok " if not missing else "BAD"
        print(f"  {flag} {name:<24} {len(rows):>5} entries, {len(missing)} images missing")
        if missing:
            print(f"       e.g. {missing[0]}")
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--from-zip", nargs="*", type=Path, default=None,
                    help="use these local zips instead of downloading from Kaggle")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--export", type=Path, default=None,
                    help="write a slim copy holding only the frames the labels "
                         "reference (~1.8 GB vs ~13 GB), for moving between machines")
    ap.add_argument("--link-from", type=Path, default=None,
                    help="build the tree as symlinks into a read-only source "
                         "(Kaggle's /kaggle/input); --dest must be writable")
    ap.add_argument("--arrange", action="store_true",
                    help="flatten an already-extracted tree in place (Kaggle nests the data "
                         "under train_set/ and test_set/; the repo needs one clips/ at the root)")
    ap.add_argument("--inspect", action="store_true",
                    help="show what is already under --dest and exit, changing nothing")
    ap.add_argument("--test-label", type=Path, default=None,
                    help="copy this file in as test_label.json (Kaggle names it test_label_new.json "
                         "and may leave it outside the dataset directory)")
    ap.add_argument("--keep-staging", action="store_true")
    args = ap.parse_args()

    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if args.inspect:
        print(f"{dest}")
        for child in sorted(dest.iterdir())[:20]:
            kind = "dir " if child.is_dir() else f"{child.stat().st_size/1e6:.1f}MB"
            print(f"  {kind:>8}  {child.name}")
            if child.is_dir() and child.name != "clips":
                for g in sorted(child.iterdir())[:10]:
                    gk = "dir " if g.is_dir() else f"{g.stat().st_size/1e6:.1f}MB"
                    print(f"      {gk:>8}  {child.name}/{g.name}")
        for sub_ in sorted(dest.rglob("clips")):
            dates = sorted(d.name for d in sub_.iterdir() if d.is_dir())
            print(f"  clips at {sub_.relative_to(dest)}: {dates[:6]}{' ...' if len(dates) > 6 else ''}")
        for alias in TEST_ALIASES:
            for hit in list(dest.rglob(alias))[:3]:
                print(f"  test label: {hit.relative_to(dest)}")
        sys.exit(0)

    if args.export is not None:
        sys.exit(0 if export_slim(dest, args.export.expanduser().resolve()) else 1)

    if args.link_from is not None:
        src = args.link_from.expanduser().resolve()
        print(f"linking {src} -> {dest}")
        link_from(src, dest)
        print("verifying")
        ok = verify(dest)
        if ok:
            print(f"\nready: {count_frames(dest)} labelled frames linked under {dest}")
            print(f"train with:\n  python scnn_tusimple.py --data {dest}")
        sys.exit(0 if ok else 1)

    if args.arrange:
        print(f"arranging {dest} in place")
        normalise(dest, dest)
        for leftover in ("train_set", "test_set", "_staging"):
            d = dest / leftover
            if not d.is_dir():
                continue
            files = [f for f in d.rglob("*") if f.is_file()]
            if not files:
                shutil.rmtree(d, ignore_errors=True)
                print(f"  removed empty {leftover}/")
            else:
                print(f"  left {leftover}/ in place ({len(files)} files remain, e.g. {files[0].name})")
        print("verifying")
        ok = verify(dest)
        if ok:
            print(f"\nready: {count_frames(dest)} labelled frames under {dest}")
        sys.exit(0 if ok else 1)

    if args.test_label is not None:
        src = args.test_label.expanduser().resolve()
        print(f"install {src.name} -> {dest/'test_label.json'}")
        shutil.copyfile(src, dest / "test_label.json")

    if args.verify_only:
        sys.exit(0 if verify(dest) else 1)

    staging = dest / "_staging"
    staging.mkdir(exist_ok=True)

    zips = args.from_zip
    if zips is None:
        print(f"downloading {KAGGLE_DS} from Kaggle (~13 GB)")
        r = subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DS, "-p", str(staging)])
        if r.returncode != 0:
            print("\nkaggle CLI failed. Install and authenticate it, or download the dataset\n"
                  f"manually from https://www.kaggle.com/datasets/{KAGGLE_DS} and re-run with\n"
                  f"  python {Path(__file__).name} --dest {dest} --from-zip <file.zip>")
            sys.exit(1)
        zips = sorted(staging.glob("*.zip"))

    for z in zips:
        print(f"unzip {z}")
        with zipfile.ZipFile(z) as f:
            f.extractall(staging)

    print("arranging tree")
    normalise(staging, dest)

    print("verifying")
    ok = verify(dest)

    if not args.keep_staging and staging.exists():
        shutil.rmtree(staging, ignore_errors=True)

    if ok:
        print(f"\nready: {count_frames(dest)} labelled frames under {dest}")
        print(f"train with:\n  python scnn_tusimple.py --data {dest}")
    else:
        print("\nincomplete — see the BAD/MISSING lines above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
