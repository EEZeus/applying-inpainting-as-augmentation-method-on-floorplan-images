#!/usr/bin/env python3
"""OCR every image of a COCO file at full resolution and cache the tokens.

Output JSON: {"metadata": {...}, "images": {<basename>: {"width", "height",
"tokens": [{"box": [x0, y0, x1, y1], "text", "conf"}]}}}. Keys are file
basenames. Resumable: an image is skipped when its cached SHA-256 matches the
file, and re-read when the file changed (e.g. a rebuilt synthetic plan).

    python scripts/ocr_cache.py --coco data/processed/instances_deduplicated.json \
        --image-dir ../area_segmentation_coco/images --out data/ocr/original_tokens.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ocr.engine import EasyOcrEngine
from src.utils.config import resolve_path

Image.MAX_IMAGE_PIXELS = None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-dir", default=os.environ.get("EASYOCR_MODEL_DIR"))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--tile", type=int, default=1600)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="only images flagged synthetic (real plans are in the original cache)",
    )
    args = parser.parse_args()

    coco = json.loads(resolve_path(args.coco).read_text(encoding="utf-8"))
    image_dir, out = resolve_path(args.image_dir), resolve_path(args.out)
    cache = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {"metadata": {}, "images": {}}
    engine = EasyOcrEngine(
        gpu=not args.cpu,
        model_dir=args.model_dir,
        allow_download=args.allow_download,
        tile=args.tile,
        overlap=args.overlap,
    )
    if cache["metadata"] and cache["metadata"].get("engine_config") != engine.metadata:
        raise ValueError(f"{out} was produced with a different OCR configuration; use a new path")
    cache["metadata"] = {"engine_config": engine.metadata, "coco": str(args.coco)}
    images = sorted(
        (item for item in coco["images"] if item.get("synthetic") or not args.synthetic_only),
        key=lambda item: int(item["id"]),
    )[: args.limit]
    names = [Path(item["file_name"]).name for item in images]
    if len(set(names)) != len(names):
        raise ValueError("Image basenames are not unique; the cache key would be ambiguous")
    wanted = set(names)
    for stale in [name for name in cache["images"] if name not in wanted]:
        del cache["images"][stale]
    for info, name in zip(images, names):
        path = image_dir / info["file_name"]
        digest = sha256(path)
        if cache["images"].get(name, {}).get("sha256") == digest:
            continue
        started = time.time()
        with Image.open(path) as handle:
            rgb = np.asarray(handle.convert("RGB"))
        tokens = engine.read_array(rgb)
        cache["images"][name] = {
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "sha256": digest,
            "tokens": tokens,
        }
        atomic_json(out, cache)
        print(
            f"{name}: {rgb.shape[1]}x{rgb.shape[0]} -> {len(tokens)} tokens in {time.time() - started:.1f}s",
            flush=True,
        )
    print(f"cached {len(cache['images'])} images -> {out}")


if __name__ == "__main__":
    main()
