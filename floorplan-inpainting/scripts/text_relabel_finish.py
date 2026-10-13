#!/usr/bin/env python3
"""OCR-verify the label-rewrite edits and build the training COCO (ocr env, GPU).

A room edit is admitted only if OCR reads the rendered rare-class name inside
the room and no longer reads the erased name. Variant v of a source plan
applies the admitted v-th edits of all its planned rooms; rejected rooms stay
original (pixels and category). Output:
  <dataset_dir>/instances_train_augmented.json  (35 real plans + synthetic plans)
  <dataset_dir>/manifest.json, verification.jsonl, qa_sheet.png

    python scripts/text_relabel_finish.py --config configs/augmentation_text.yaml
"""
import argparse
import copy
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation.masks import rasterize_annotation  # noqa: E402
from src.augmentation.pipeline import (_append_jsonl, _atomic_json, _copy_original_train_images,  # noqa: E402
                                       latest_metadata, read_metadata, validate_augmented_coco)
from src.augmentation.text_relabel import apply_patch, verify_reading  # noqa: E402
from src.data.coco_dataset import read_split  # noqa: E402
from src.ocr.text import tokens_in_mask  # noqa: E402
from src.utils.config import load_yaml, resolve_path  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def save_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.png")
    Image.fromarray(array).save(temporary)
    os.replace(temporary, path)


def qa_sheet(path, rows, crops_dir, max_rows=24, width=320):
    tiles = []
    for record, verdict in rows[:max_rows]:
        before = Image.open(crops_dir / f"{record['sample_id']}_before.png").convert("RGB")
        after = Image.open(crops_dir / f"{record['sample_id']}.png").convert("RGB")
        scale = width / max(before.width, 1)
        size = (width, max(1, int(before.height * scale)))
        tile = Image.new("RGB", (2 * width + 10, size[1] + 34), "white")
        tile.paste(before.resize(size), (0, 34)); tile.paste(after.resize(size), (width + 10, 34))
        label = (f"{record['old_texts']} -> {record['word']!r}  cat {record['source_category_id']}->"
                 f"{record['target_category_id']}  {'OK' if verdict['passed'] else 'REJECT'} "
                 f"new={verdict['new_word_similarity']:.2f} old={verdict['old_word_similarity']:.2f}")
        ImageDraw.Draw(tile).text((4, 4), label[:110], fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return
    sheet = Image.new("RGB", (max(t.width for t in tiles), sum(t.height + 6 for t in tiles)), "white")
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y)); y += tile.height + 6
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/augmentation_text.yaml")
    parser.add_argument("--model-dir", default=os.environ.get("EASYOCR_MODEL_DIR"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    coco = json.loads(resolve_path(config["dataset"]["annotation_file"]).read_text(encoding="utf-8"))
    train_ids = set(read_split(resolve_path(config["dataset"]["train_split"])))
    image_dir = resolve_path(config["dataset"]["image_dir"])
    gen_dir, out = resolve_path(config["output"]["generation_dir"]), resolve_path(config["output"]["dataset_dir"])
    out.mkdir(parents=True, exist_ok=True)
    records = latest_metadata(read_metadata(gen_dir / "records.jsonl"))
    by_id = {int(a["id"]): a for a in coco["annotations"]}
    infos = {int(i["id"]): i for i in coco["images"]}
    threshold = float(config["verify"]["similarity_threshold"])

    # ---- OCR verification of every generated edit (resumable)
    verification_path = out / "verification.jsonl"
    verified = latest_metadata(read_metadata(verification_path))
    pending = [r for r in records.values() if r["status"] == "generated" and r["sample_id"] not in verified]
    if pending:
        from src.ocr.engine import EasyOcrEngine
        engine = EasyOcrEngine(gpu=not args.cpu, model_dir=args.model_dir)
        for record in sorted(pending, key=lambda r: r["sample_id"]):
            crop = np.asarray(Image.open(gen_dir / "crops" / f"{record['sample_id']}.png").convert("RGB"))
            tokens = engine.read_array(crop)
            info = infos[record["source_image_id"]]
            cx0, cy0, cx1, cy1 = record["crop_box"]
            room = rasterize_annotation(by_id[record["annotation_id"]], int(info["height"]),
                                        int(info["width"]))[cy0:cy1, cx0:cx1]
            inside = [tokens[i] for i in tokens_in_mask([t["box"] for t in tokens], room)]
            verdict = verify_reading(inside, record["word"], record["old_texts"], threshold)
            verdict["sample_id"] = record["sample_id"]
            _append_jsonl(verification_path, verdict)
            verified[record["sample_id"]] = verdict
            print(f"{record['sample_id']}: {'OK' if verdict['passed'] else 'reject'} read={verdict['read_texts']} "
                  f"word={record['word']!r}", flush=True)

    # ---- compose plan variants from admitted edits
    admitted = defaultdict(list)
    for sample_id, verdict in verified.items():
        record = records.get(sample_id)
        if record and record["status"] == "generated" and verdict["passed"]:
            admitted[(record["source_image_id"], record["variant"])].append(record)
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
    stride = max(len(v) for v in annotations.values())
    output = {key: copy.deepcopy(value) for key, value in coco.items() if key not in {"images", "annotations"}}
    output["images"] = _copy_original_train_images(coco, train_ids, image_dir, out)
    output["annotations"] = [copy.deepcopy(a) for a in coco["annotations"] if int(a["image_id"]) in train_ids]
    converted = Counter()
    for (source_id, variant), rooms in sorted(admitted.items()):
        if source_id not in train_ids:
            raise ValueError("Non-training source plan in text relabel edits")
        info = infos[source_id]
        with Image.open(image_dir / info["file_name"]) as handle:
            plan = np.asarray(handle.convert("RGB"))
        for record in sorted(rooms, key=lambda r: r["annotation_id"]):
            patch = dict(np.load(gen_dir / "patches" / f"{record['sample_id']}.npz"))
            plan = apply_patch(plan, patch)
        synthetic_id = 2_000_000 + source_id * 10 + variant
        file_name = f"images/plan{source_id}_v{variant}.png"
        save_png(out / file_name, plan)
        targets = {r["annotation_id"]: r["target_category_id"] for r in rooms}
        record = copy.deepcopy(info)
        record.update({"id": synthetic_id, "file_name": file_name, "synthetic": True, "strategy": "text_relabel",
                       "source_image_id": source_id, "variant": variant,
                       "converted_annotation_ids": sorted(targets),
                       "rendered_words": {str(r["annotation_id"]): r["word"] for r in rooms}})
        output["images"].append(record)
        for ordinal, original in enumerate(sorted(annotations[source_id], key=lambda a: int(a["id"]))):
            duplicate = copy.deepcopy(original)
            duplicate["id"] = 3_000_000_000 + (source_id * 10 + variant) * stride + ordinal
            duplicate["image_id"] = synthetic_id
            if original["id"] in targets:
                duplicate["category_id"] = targets[original["id"]]
                converted[targets[original["id"]]] += 1
            output["annotations"].append(duplicate)
    output["images"].sort(key=lambda item: int(item["id"]))
    output["annotations"].sort(key=lambda item: int(item["id"]))
    validate_augmented_coco(output, out, train_ids)
    _atomic_json(out / "instances_train_augmented.json", output)

    statuses = Counter(r["status"] for r in records.values())
    reasons = Counter(r.get("reason") for r in records.values() if r["status"] == "rejected")
    erasers = Counter(r.get("eraser") for r in records.values() if r["status"] == "generated")
    manifest = {
        "edits_attempted": len(records), "edit_status": dict(statuses), "rejection_reasons": dict(reasons),
        "erasers_used": dict(erasers),
        "ocr_verified": sum(v["passed"] for v in verified.values()), "ocr_checked": len(verified),
        "synthetic_plans": len(admitted), "source_plans_with_variants": len({k[0] for k in admitted}),
        "converted_rooms_by_class": {str(k): v for k, v in sorted(converted.items())},
        "converted_rooms_total": int(sum(converted.values())),
    }
    _atomic_json(out / "manifest.json", manifest)
    rows = [(records[s], v) for s, v in sorted(verified.items()) if s in records]
    qa_sheet(out / "qa_sheet.png", rows, gen_dir / "crops")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
