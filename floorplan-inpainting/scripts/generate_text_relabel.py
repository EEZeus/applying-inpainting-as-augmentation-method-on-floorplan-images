#!/usr/bin/env python3
"""Generate OCR-guided label-rewrite room edits (inpainting env, GPU).

Each planned (frequent room -> rare class) pair is edited ``variants_per_room``
times. Every edit is stored as a patch (pixels + mask inside the room) plus a
room crop for the OCR verification pass. Resumable by sample id.

    python scripts/generate_text_relabel.py --config configs/augmentation_text.yaml [--limit 4]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation.pipeline import _append_jsonl, _atomic_json, latest_metadata, read_metadata  # noqa: E402
from src.augmentation.strategy_b import plan_strategy_b  # noqa: E402
from src.augmentation.text_relabel import convert_room, make_patch, rare_vocabulary, sample_word  # noqa: E402
from src.data.coco_dataset import read_split  # noqa: E402
from src.utils.config import load_yaml, resolve_path  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/augmentation_text.yaml")
    parser.add_argument("--limit", type=int, help="only the first N planned rooms (smoke)")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--model-cache")
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    if args.model_cache:
        config["generation"]["cache_dir"] = args.model_cache
    coco = json.loads(resolve_path(config["dataset"]["annotation_file"]).read_text(encoding="utf-8"))
    train_ids = set(read_split(resolve_path(config["dataset"]["train_split"])))
    image_dir = resolve_path(config["dataset"]["image_dir"])
    ocr = json.loads(resolve_path(config["dataset"]["ocr_cache"]).read_text(encoding="utf-8"))["images"]
    out = resolve_path(config["output"]["generation_dir"])
    (out / "patches").mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    file_names = {int(i["id"]): Path(i["file_name"]).name for i in coco["images"]}
    missing = sorted(i for i in train_ids if file_names[i] not in ocr)
    if missing:
        raise ValueError(f"OCR cache lacks training plans {missing}; run scripts/ocr_cache.py first")
    tokens_by_name = {name: entry["tokens"] for name, entry in ocr.items()}
    targets = [int(c) for c in config["classes"]["target_ids"]]
    vocab = rare_vocabulary(coco, train_ids, tokens_by_name, targets, file_names)
    (plan, _), diagnostics = plan_strategy_b(coco, train_ids, config)
    plan = plan[:args.limit] if args.limit else plan
    _atomic_json(out / "plan.json", {
        "diagnostics": diagnostics, "vocabulary": {str(k): dict(v) for k, v in vocab.items()},
        "rooms": [{"annotation_id": c.target_annotation_id, "source_image_id": c.source_image_id,
                   "target_category_id": c.category_id, "seed": c.seed} for c in plan]})

    records_path = out / "records.jsonl"
    done = latest_metadata(read_metadata(records_path))
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
    by_id = {int(a["id"]): a for a in coco["annotations"]}
    infos = {int(i["id"]): i for i in coco["images"]}

    backend = None
    if config["edit"].get("eraser", "diffusion") == "diffusion":
        from src.augmentation.backends import StableDiffusionInpaintingBackend
        backend = StableDiffusionInpaintingBackend(config["generation"])
    image_cache = {}
    variants = int(config["variants_per_room"])
    for candidate in plan:
        for variant in range(variants):
            sample_id = f"t-ann{candidate.target_annotation_id}-to{candidate.category_id}-v{variant}"
            previous = done.get(sample_id)
            if previous and (previous["status"] != "error" or not args.retry_errors):
                continue
            seed = int(config["seed"]) * 1000 + candidate.candidate_index * 10 + variant
            rng = np.random.default_rng(seed)
            word = sample_word(vocab[candidate.category_id], rng)
            image_id = candidate.source_image_id
            if image_id not in image_cache:
                image_cache.clear()
                with Image.open(image_dir / infos[image_id]["file_name"]) as handle:
                    image_cache[image_id] = np.asarray(handle.convert("RGB"))
            image = image_cache[image_id]
            target = by_id[candidate.target_annotation_id]
            others = [a for a in annotations[image_id] if a["id"] != target["id"]]
            started = time.time()
            record = {"sample_id": sample_id, "source_image_id": image_id, "variant": variant,
                      "annotation_id": target["id"], "source_category_id": int(target["category_id"]),
                      "target_category_id": candidate.category_id, "seed": seed}
            try:
                result = convert_room(image, target, others, tokens_by_name[file_names[image_id]], word,
                                      config["edit"], seed, backend)
            except Exception as error:  # recorded and retried with --retry-errors
                _append_jsonl(records_path, {**record, "status": "error", "reason": repr(error)})
                print(f"{sample_id}: error {error!r}", flush=True)
                continue
            edited, edit_mask = result.pop("image", None), result.pop("edit_mask", None)
            if result["status"] == "generated":
                patch = make_patch(edited, edit_mask)
                np.savez_compressed(out / "patches" / f"{sample_id}.npz", **patch)
                x, y, w, h = target["bbox"]
                margin = config["verify"]["crop_margin_fraction"] * max(w, h)
                cx0, cy0 = max(0, int(x - margin)), max(0, int(y - margin))
                cx1, cy1 = min(image.shape[1], int(x + w + margin)), min(image.shape[0], int(y + h + margin))
                Image.fromarray(edited[cy0:cy1, cx0:cx1]).save(out / "crops" / f"{sample_id}.png")
                Image.fromarray(image[cy0:cy1, cx0:cx1]).save(out / "crops" / f"{sample_id}_before.png")
                result["crop_box"] = [cx0, cy0, cx1, cy1]
            record.update(result)
            record["seconds"] = time.time() - started
            _append_jsonl(records_path, record)
            print(f"{sample_id}: {record['status']} {record.get('reason', '')} word={word!r} "
                  f"old={record.get('old_texts')} eraser={record.get('eraser')}", flush=True)
    final = latest_metadata(read_metadata(records_path))
    counts = defaultdict(int)
    for item in final.values():
        counts[item["status"]] += 1
    print(json.dumps(dict(counts)))


if __name__ == "__main__":
    main()
