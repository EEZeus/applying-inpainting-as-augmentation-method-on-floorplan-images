#!/usr/bin/env python3
"""Rotated held-out evaluation splits: image-level K-fold over the 21 held-out images.

Training stays the frozen 35-image split for every fold, so the augmented datasets and grid
results remain comparable; only test.txt rotates. The held-out images span just 3 projects,
so a project-aware split with K>3 is impossible; folds are assigned round-robin over sorted
IDs and each held-out image is tested once. The CV eval (`run_experiment.py --eval-checkpoint`)
reads only test.txt and never trains.

Writes data/splits_cv/fold{k}/{train,val,test}.txt and manifest.json.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.grouping import image_to_project

PRIMARY_RARE = [11, 12, 13, 14, 15, 16]


def _read_ids(path):
    return [int(v) for v in path.read_text(encoding="utf-8").split()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    parser.add_argument(
        "--annotations", type=Path, default=ROOT / "data/processed/instances_deduplicated.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/splits_cv")
    args = parser.parse_args()

    train_ids = sorted(_read_ids(args.splits / "train.txt"))
    val_ids = sorted(_read_ids(args.splits / "val.txt"))
    heldout = sorted(set(val_ids) | set(_read_ids(args.splits / "test.txt")))
    if set(train_ids) & set(heldout):
        raise ValueError("frozen train overlaps val/test")
    if args.folds < 2 or args.folds > len(heldout):
        raise ValueError(f"--folds {args.folds} invalid for {len(heldout)} held-out images")

    fold_test = [
        sorted(heldout[i] for i in range(len(heldout)) if i % args.folds == k) for k in range(args.folds)
    ]
    if sorted(i for f in fold_test for i in f) != heldout:
        raise ValueError("fold union != held-out set")

    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    ann_by_image = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(int(ann["image_id"]), []).append(int(ann["category_id"]))
    proj_of = image_to_project()

    def rare_support(ids):
        counts = {c: 0 for c in PRIMARY_RARE}
        for image_id in ids:
            for cid in ann_by_image.get(image_id, []):
                if cid in counts:
                    counts[cid] += 1
        return counts

    args.out.mkdir(parents=True, exist_ok=True)
    train_txt = "".join(f"{i}\n" for i in train_ids)
    val_txt = "".join(f"{i}\n" for i in val_ids)
    manifest = {
        "folds": args.folds,
        "mode": "rotated_heldout_eval_imagelevel",
        "train": "frozen data/splits/train.txt (unchanged)",
        "pooled_test_images": len(heldout),
        "fold": [],
    }
    for k in range(args.folds):
        fold_dir = args.out / f"fold{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "train.txt").write_text(train_txt, encoding="utf-8")
        (fold_dir / "val.txt").write_text(val_txt, encoding="utf-8")
        (fold_dir / "test.txt").write_text("".join(f"{i}\n" for i in fold_test[k]), encoding="utf-8")
        manifest["fold"].append(
            {
                "k": k,
                "test_ids": fold_test[k],
                "test_projects": sorted({proj_of[i] for i in fold_test[k]}),
                "n_test": len(fold_test[k]),
                "test_rare_support": rare_support(fold_test[k]),
            }
        )

    manifest["pooled_test_rare_support"] = {
        c: sum(f["test_rare_support"][c] for f in manifest["fold"]) for c in PRIMARY_RARE
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
