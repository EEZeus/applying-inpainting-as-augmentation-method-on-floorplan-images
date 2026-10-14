#!/usr/bin/env python3
"""Write data/splits_heldout21: frozen train/val, test = all 21 held-out plans.

Used only through FP_SPLIT_DIR for evaluation of the epoch-25 checkpoint
(last.pth), which is never selected on validation data, so the 10 validation
plans are legitimately held out for it. Training never reads this directory.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.coco_dataset import read_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--out", type=Path, default=ROOT / "data/splits_heldout21")
    args = parser.parse_args()
    train, val, test = (read_split(args.splits / f"{name}.txt") for name in ("train", "val", "test"))
    if set(train) & (set(val) | set(test)) or set(val) & set(test):
        raise ValueError("frozen splits overlap")
    heldout = sorted(set(val) | set(test))
    if len(heldout) != 21 or len(train) != 35:
        raise ValueError(f"unexpected split sizes: train={len(train)} heldout={len(heldout)}")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, ids in (("train", train), ("val", val), ("test", heldout)):
        (args.out / f"{name}.txt").write_text("".join(f"{i}\n" for i in sorted(ids)), encoding="utf-8")
    print(f"wrote {args.out}: train={len(train)} val={len(val)} test(heldout)={len(heldout)}")


if __name__ == "__main__":
    main()
