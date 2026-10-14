#!/usr/bin/env python3
"""Compute text features for every label string of one or more OCR caches.

    python scripts/ocr_embed.py --cache data/ocr/original_tokens.json \
        --out data/ocr/original_features.npz
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ocr.features import E5_MODEL, E5Encoder, build_text_features, label_texts, save_features
from src.utils.config import resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--categories-from", default="data/processed/instances_deduplicated.json")
    parser.add_argument("--model", default=E5_MODEL)
    args = parser.parse_args()
    texts = set()
    for path in args.cache:
        texts.update(label_texts(json.loads(resolve_path(path).read_text(encoding="utf-8"))))
    texts = sorted(texts)
    categories = json.loads(resolve_path(args.categories_from).read_text(encoding="utf-8"))["categories"]
    class_names = [c["name"] for c in sorted(categories, key=lambda c: int(c["id"]))]
    encoder = E5Encoder(args.model)
    features, layout = build_text_features(texts, class_names, encoder)
    save_features(resolve_path(args.out), texts, features, layout)
    print(f"{len(texts)} label strings -> {features.shape} -> {args.out}")


if __name__ == "__main__":
    main()
