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


def merge_into(src: Path, dst: Path):
    """Move src's entries into dst, merging directories instead of clobbering."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir() and target.exists():
            merge_into(item, target)
            item.rmdir() if not any(item.iterdir()) else None
        elif not target.exists():
            shutil.move(str(item), str(target))


def normalise(staging: Path, dest: Path):
    """Pull clips/ and the label json files out of however Kaggle nested them."""
    for clips in sorted(staging.rglob("clips")):
        if clips.is_dir():
            print(f"  merge {clips.relative_to(staging)} -> clips/")
            merge_into(clips, dest / "clips")
    for name in LABELS:
        for found in staging.rglob(name):
            if not (dest / name).exists():
                print(f"  take  {found.relative_to(staging)}")
                shutil.move(str(found), str(dest / name))
            break


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
    ap.add_argument("--keep-staging", action="store_true")
    args = ap.parse_args()

    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

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
        n = sum(1 for _ in (dest / "clips").rglob("20.jpg"))
        print(f"\nready: {n} labelled frames under {dest}")
        print(f"train with:\n  python scnn_tusimple.py --data {dest}")
    else:
        print("\nincomplete — see the BAD/MISSING lines above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
