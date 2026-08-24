#!/usr/bin/env python
"""Download and unpack CULane into the layout dataset/CULane.py expects.

    python download_culane.py --dest /path/to/CULane            # everything
    python download_culane.py --dest /path/to/CULane --only list laneseg_label_w16
    python download_culane.py --dest /path/to/CULane --list     # show files, download nothing

Resumable: an archive already unpacked (its target directory exists) is skipped.
File ids are from the official Drive folder linked at
https://xingangpan.github.io/projects/CULane.html (folder 1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu).
"""
import argparse, sys, tarfile, zipfile
from pathlib import Path

# archive name -> (drive file id, directory it unpacks to)
FILES = {
    "driver_23_30frame":      ("14Gi1AXbgkqvSysuoLyq1CsjFSypvoLVL", "driver_23_30frame"),
    "driver_161_90frame":     ("1AQjQZwOAkeBTSG_1I9fYn8KBcxBBbYyk", "driver_161_90frame"),
    "driver_182_30frame":     ("1PH7UdmtZOK3Qi3SBqtYOkWSH2dpbfmkL", "driver_182_30frame"),
    "driver_37_30frame":      ("1Z6a463FQ3pfP54HMwF3QS5h9p2Ch3An7", "driver_37_30frame"),
    "driver_100_30frame":     ("1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8", "driver_100_30frame"),
    "driver_193_90frame":     ("1daWl7XVzH06GwcZtF4WD8Xpvci5SZiUV", "driver_193_90frame"),
    "laneseg_label_w16":      ("1MlL1oSiRu6ZRU-62E39OZ7izljagPycH", "laneseg_label_w16"),
    "laneseg_label_w16_test": ("1yCOXaaNcyoVrHDR0_A_gXH-thg-7QDv8", "laneseg_label_w16_test"),
    "list":                   ("18alVEPAMBA9Hpr3RDAAchqSj5IxZNRKd", "list"),
    "annotations_new":        ("1QbB1TOk9Fy6Sk0CoOsR3V8R56_eG6Xnu", "annotations_new"),
}
# what SCNN actually needs; annotations_new is the corrected-label extra
DEFAULT = [k for k in FILES if k != "annotations_new"]


def unpack(archive: Path, dest: Path):
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as f:
            f.extractall(dest)
    else:
        with tarfile.open(archive) as f:
            f.extractall(dest, filter="data")   # 3.12+ default in 3.14; set it explicitly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, help="required unless --list")
    ap.add_argument("--only", nargs="*", choices=list(FILES), default=DEFAULT)
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--keep", action="store_true", help="keep archives after unpacking")
    args = ap.parse_args()

    if not args.list and args.dest is None:
        ap.error("--dest is required")

    if args.list:
        for k, (fid, d) in FILES.items():
            print(f"{k:<24} {fid}  -> {d}/{'  (default)' if k in DEFAULT else ''}")
        return

    import gdown  # pip install gdown

    dest = args.dest.expanduser().resolve()
    (dest / "_archives").mkdir(parents=True, exist_ok=True)

    for name in args.only:
        fid, outdir = FILES[name]
        if (dest / outdir).is_dir():
            print(f"[skip] {outdir}/ already present")
            continue
        suffix = ".zip" if name == "laneseg_label_w16_test" else ".tar.gz"
        archive = dest / "_archives" / (name + suffix)
        if not archive.exists():
            print(f"[get ] {name}{suffix}")
            gdown.download(id=fid, output=str(archive), quiet=False)
        print(f"[open] {archive.name} -> {dest}")
        unpack(archive, dest)
        if not args.keep:
            archive.unlink()

    missing = [d for d in ("list", "laneseg_label_w16") if not (dest / d).is_dir()]
    print("\ndone." if not missing else f"\ndone, but missing: {missing}")
    print(f"now set  Dataset_Path['CULane'] = \"{dest}\"  in config.py")
    print(f"and      data_dir={dest}/  in utils/lane_evaluation/CULane/Run.sh")


if __name__ == "__main__":
    sys.exit(main())
