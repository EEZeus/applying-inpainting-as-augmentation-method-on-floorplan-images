#!/usr/bin/env python3
"""Extract frozen DINOv2 features for room-region crops.

Two tasks:
  --task gt        : one crop per annotation of a COCO file (optionally limited
                     to a split file and/or a category whitelist). Labels kept.
  --task proposals : one crop per detection in a COCO-results JSON (e.g. the SAM3
                     class-agnostic test dump). No labels; detection score and
                     the raw RLE are carried through for the two-stage eval.

Feature per crop = [CLS token ; mean of patch tokens] (1536-d for dinov2-base).
Crop = annotation/detection bbox expanded by --context * max(w,h) each side,
clamped, padded to a white square, then the HF image processor resizes to 224.

Output .npz: features[N,D] float32, labels[N] int (-1 for proposals),
image_ids[N] int, keys[N] int (annotation id, or detection index),
scores[N] float32 (1.0 for gt). For --task proposals a companion
<out>.dets.json holds the kept detections in npz-row order.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _load_dino(model_name, device):
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return proc, model


def _square_pad(im):
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


def _crop(im, bbox, context):
    x, y, w, h = bbox
    if w <= 1 or h <= 1:
        return None
    pad = context * max(w, h)
    left = max(0, int(round(x - pad)))
    top = max(0, int(round(y - pad)))
    right = min(im.width, int(round(x + w + pad)))
    bottom = min(im.height, int(round(y + h + pad)))
    if right - left < 2 or bottom - top < 2:
        return None
    return _square_pad(im.crop((left, top, right, bottom)))


@torch.no_grad()
def _embed(crops, proc, model, device):
    inputs = proc(images=crops, return_tensors="pt").to(device)
    out = model(**inputs)
    hs = out.last_hidden_state  # [B, 1+P, D]
    feat = torch.cat([hs[:, 0], hs[:, 1:].mean(dim=1)], dim=-1)
    return feat.float().cpu().numpy()


def _run(records, image_root, proc, model, device, context, batch):
    """records: list of (image_path, bbox, label, image_id, key, score)."""
    feats, keep = [], []
    buf_crops, buf_idx = [], []

    def flush():
        if not buf_crops:
            return
        feats.append(_embed(buf_crops, proc, model, device))
        keep.extend(buf_idx)
        buf_crops.clear()
        buf_idx.clear()

    cache_path, cache_im = None, None
    for i, (img_path, bbox, *_rest) in enumerate(records):
        if img_path != cache_path:
            cache_im = Image.open(image_root / img_path).convert("RGB")
            cache_path = img_path
        c = _crop(cache_im, bbox, context)
        if c is None:
            continue
        buf_crops.append(c)
        buf_idx.append(i)
        if len(buf_crops) >= batch:
            flush()
    flush()
    if not keep:
        raise RuntimeError("no valid crops produced")
    features = np.concatenate(feats, axis=0).astype(np.float32)
    kept = [records[i] for i in keep]
    return features, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gt", "proposals"])
    ap.add_argument("--coco", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-file")
    ap.add_argument("--only-cats", help="comma list, gt task only")
    ap.add_argument("--pred-json", help="proposals task: COCO-results JSON")
    ap.add_argument("--dets-out", help="proposals task: companion dets JSON")
    ap.add_argument("--score-min", type=float, default=0.0)
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--context", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap crops")
    args = ap.parse_args()

    coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
    img_by_id = {int(i["id"]): i for i in coco["images"]}
    image_root = Path(args.image_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = []
    if args.task == "gt":
        allowed = None
        if args.split_file:
            allowed = {int(x) for x in Path(args.split_file).read_text().split()}
        cats = None
        if args.only_cats:
            cats = {int(c) for c in args.only_cats.split(",")}
        for a in coco["annotations"]:
            iid = int(a["image_id"])
            if allowed is not None and iid not in allowed:
                continue
            if cats is not None and int(a["category_id"]) not in cats:
                continue
            if iid not in img_by_id:
                continue
            records.append(
                (
                    img_by_id[iid]["file_name"],
                    [float(v) for v in a["bbox"]],
                    int(a["category_id"]),
                    iid,
                    int(a["id"]),
                    1.0,
                )
            )
    else:
        dets = json.loads(Path(args.pred_json).read_text(encoding="utf-8"))
        for k, d in enumerate(dets):
            if float(d.get("score", 1.0)) < args.score_min:
                continue
            iid = int(d["image_id"])
            if iid not in img_by_id:
                continue
            records.append(
                (
                    img_by_id[iid]["file_name"],
                    [float(v) for v in d["bbox"]],
                    -1,
                    iid,
                    k,
                    float(d.get("score", 1.0)),
                )
            )
    # sort by image so _run can reuse the open image
    records.sort(key=lambda r: (r[0], r[4]))
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("no records selected")

    proc, model = _load_dino(args.model, device)
    features, kept = _run(records, image_root, proc, model, device, args.context, args.batch)

    labels = np.array([r[2] for r in kept], dtype=np.int64)
    image_ids = np.array([r[3] for r in kept], dtype=np.int64)
    keys = np.array([r[4] for r in kept], dtype=np.int64)
    scores = np.array([r[5] for r in kept], dtype=np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        features=features,
        labels=labels,
        image_ids=image_ids,
        keys=keys,
        scores=scores,
        model=args.model,
        context=args.context,
    )

    if args.task == "proposals":
        dets = json.loads(Path(args.pred_json).read_text(encoding="utf-8"))
        ordered = [dets[int(k)] for k in keys]
        Path(args.dets_out).write_text(json.dumps(ordered), encoding="utf-8")

    print(
        json.dumps(
            {
                "out": str(out),
                "task": args.task,
                "n": int(features.shape[0]),
                "dim": int(features.shape[1]),
                "label_dist": {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
