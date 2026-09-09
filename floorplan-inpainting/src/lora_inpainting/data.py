"""Build and load the room-centred dataset used by the shared inpainting LoRA."""
from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from src.augmentation.crops import CropTransform
from src.augmentation.masks import build_safe_mask
from src.augmentation.pipeline import _atomic_json, file_digest
from src.augmentation.strategy_b import room_geometry


def _png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")


def _variant(image, mask, rng, settings):
    """Apply conservative scan/photometric transforms to image and geometry-aligned mask."""
    out = Image.fromarray(image)
    meta = {}
    brightness = float(rng.uniform(*settings["brightness_range"]))
    contrast = float(rng.uniform(*settings["contrast_range"]))
    out = ImageEnhance.Brightness(out).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    meta.update(brightness=brightness, contrast=contrast)
    if rng.random() < settings["grayscale_probability"]:
        out = out.convert("L").convert("RGB"); meta["grayscale"] = True
    if rng.random() < settings["blur_probability"]:
        sigma = float(rng.uniform(*settings["blur_sigma_range"]))
        out = out.filter(ImageFilter.GaussianBlur(sigma)); meta["blur_sigma"] = sigma
    arr = np.asarray(out, dtype=np.uint8)
    if rng.random() < settings["line_thickness_probability"]:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        dark = gray < 150
        operation = "dilate" if rng.random() < .5 else "erode"
        changed = cv2.dilate(dark.astype(np.uint8), np.ones((2, 2), np.uint8)) if operation == "dilate" else cv2.erode(dark.astype(np.uint8), np.ones((2, 2), np.uint8))
        delta = changed.astype(bool) & ~dark if operation == "dilate" else dark & ~changed.astype(bool)
        arr = arr.copy(); arr[delta] = 0 if operation == "dilate" else 255
        meta["line_thickness"] = operation
    if rng.random() < settings["affine_probability"]:
        angle = float(rng.uniform(-settings["rotation_degrees"], settings["rotation_degrees"]))
        scale = float(rng.uniform(*settings["scale_range"]))
        shift = int(settings["translation_pixels"])
        tx, ty = (float(rng.uniform(-shift, shift)) for _ in range(2))
        matrix = cv2.getRotationMatrix2D((arr.shape[1] / 2, arr.shape[0] / 2), angle, scale)
        matrix[:, 2] += (tx, ty)
        arr = cv2.warpAffine(arr, matrix, (arr.shape[1], arr.shape[0]), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        mask = cv2.warpAffine(mask.astype(np.uint8), matrix, (mask.shape[1], mask.shape[0]),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        meta["affine"] = {"angle": angle, "scale": scale, "translation": [tx, ty]}
    if rng.random() < settings["noise_probability"]:
        sigma = float(rng.uniform(*settings["noise_sigma_range"]))
        arr = np.clip(arr.astype(np.float32) + rng.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
        meta["noise_sigma"] = sigma
    if rng.random() < settings["jpeg_probability"]:
        quality = int(rng.integers(settings["jpeg_quality_range"][0], settings["jpeg_quality_range"][1] + 1))
        buffer = io.BytesIO(); Image.fromarray(arr).save(buffer, "JPEG", quality=quality)
        arr = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"), dtype=np.uint8)
        meta["jpeg_quality"] = quality
    return arr, mask, meta


def _validation_cases(records, annotations, category_names, target_ids, output_dir, prompt_template):
    """Pick held-as-validation prompts on common geometries, never rare augmented copies."""
    base = [r for r in records if not r["is_classical_variant"]]
    by_ann = {int(a["id"]): a for a in annotations}
    common = [r for r in base if int(r["category_id"]) not in target_ids]
    target_geometry = defaultdict(list)
    for r in base:
        if int(r["category_id"]) in target_ids:
            target_geometry[int(r["category_id"])].append(room_geometry(by_ann[int(r["source_annotation_id"])]))
    used = set(); result = []
    for target in target_ids:
        examples = target_geometry.get(target, [])
        if not examples or not common:
            continue
        center = {k: float(np.median([g[k] for g in examples])) for k in examples[0]}
        def distance(record):
            g = room_geometry(by_ann[int(record["source_annotation_id"])])
            return sum(abs(np.log((g[k] + 1e-6) / (center[k] + 1e-6))) for k in ("area", "aspect_ratio", "width", "height", "solidity"))
        candidates = [r for r in common if r["source_annotation_id"] not in used]
        chosen = min(candidates or common, key=distance)
        used.add(chosen["source_annotation_id"])
        result.append({
            "case_id": f"target-{target}-source-{chosen['source_annotation_id']}",
            "target_category_id": target, "target_category_name": category_names[target],
            "source_annotation_id": chosen["source_annotation_id"], "source_image_id": chosen["source_image_id"],
            "image_file": chosen["image_file"], "mask_file": chosen["mask_file"],
            "prompt": prompt_template.format(category_name=category_names[target]),
            "selection": "nearest common-room geometry to median real target geometry",
        })
    _atomic_json(output_dir / "validation_cases.json", {"cases": result})
    return result


def build_training_crops(config, coco, train_ids, image_dir, output_dir):
    output_dir = Path(output_dir)
    images_dir, masks_dir = output_dir / "images", output_dir / "masks"
    image_map = {int(i["id"]): i for i in coco["images"]}
    annotations_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[int(ann["image_id"])].append(ann)
    train_annotations = sorted((a for a in coco["annotations"] if int(a["image_id"]) in train_ids), key=lambda a: int(a["id"]))
    if any(int(a["image_id"]) not in train_ids for a in train_annotations):
        raise ValueError("Non-training annotation selected")
    names = {int(c["id"]): c["name"] for c in coco["categories"]}
    primary = set(map(int, config["rare_augmentation"]["primary_ids"]))
    optional = set(map(int, config["rare_augmentation"]["optional_ids"])) if config["rare_augmentation"].get("include_optional") else set()
    records, skipped = [], []
    for ann in train_annotations:
        info = image_map[int(ann["image_id"])]
        source_path = Path(image_dir) / info["file_name"]
        original = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)
        others = [a for a in annotations_by_image[int(ann["image_id"])] if int(a["id"]) != int(ann["id"])]
        safe = build_safe_mask(original, ann, others, config["mask"])
        if safe.mutable_fraction < float(config["mask"]["min_mutable_fraction"]) or not safe.safe_mask.any():
            skipped.append({"annotation_id": ann["id"], "reason": "safe_mask_too_small", "mutable_fraction": safe.mutable_fraction}); continue
        transform = CropTransform.from_bbox(ann["bbox"], original.shape[1], original.shape[0], config["crop"]["context_fraction"], config["crop"]["output_size"])
        image, mask = transform.forward_image(original), transform.forward_mask(safe.safe_mask)
        category = int(ann["category_id"]); count = (config["rare_augmentation"]["variants_per_primary_instance"] if category in primary else config["rare_augmentation"]["variants_per_optional_instance"] if category in optional else 0)
        for variant_index in range(int(count) + 1):
            sample_id = f"ann{int(ann['id'])}-v{variant_index}"
            variant_image, variant_mask, transform_meta = image, mask, {}
            if variant_index:
                seed = int(config["seed"]) + int(ann["id"]) * 1009 + variant_index
                variant_image, variant_mask, transform_meta = _variant(image, mask, np.random.default_rng(seed), config["rare_augmentation"]["transforms"])
            image_file, mask_file = f"images/{sample_id}.png", f"masks/{sample_id}.png"
            _png(output_dir / image_file, variant_image); _png(output_dir / mask_file, variant_mask.astype(np.uint8) * 255)
            records.append({
                "sample_id": sample_id, "source_image_id": int(ann["image_id"]), "source_annotation_id": int(ann["id"]),
                "geometry_identity": f"annotation-{int(ann['id'])}", "independent_geometry": False,
                "category_id": category, "category_name": names[category],
                "prompt": config["prompt"]["template"].format(category_name=names[category]),
                "image_file": image_file, "mask_file": mask_file,
                "is_classical_variant": bool(variant_index), "variant_index": variant_index,
                "classical_transform": transform_meta, "mutable_fraction": safe.mutable_fraction,
                "erosion_pixels": safe.erosion_pixels, "crop_transform": transform.to_dict(),
                "source_image_sha256": file_digest(source_path),
            })
    real_counts = Counter(r["category_name"] for r in records if not r["is_classical_variant"])
    variant_counts = Counter(r["category_name"] for r in records if r["is_classical_variant"])
    geometry_counts = Counter(r["category_name"] for r in records if not r["is_classical_variant"])
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    manifest = {"schema_version": 1, "train_image_ids": sorted(train_ids), "records": records,
                "manifest_sha256": digest, "scientific_distinction": {
                    "real_geometry_counts": dict(sorted(geometry_counts.items())),
                    "classical_variant_counts": dict(sorted(variant_counts.items())),
                    "classical_variants_are_independent_geometries": False}, "skipped": skipped}
    _atomic_json(output_dir / "manifest.json", manifest)
    validation = _validation_cases(records, train_annotations, names, primary | optional, output_dir,
                                   config["prompt"]["template"])
    summary = {"records": len(records), "real_crops": sum(real_counts.values()), "classical_variants": sum(variant_counts.values()),
               "real_geometry_counts": dict(sorted(geometry_counts.items())), "classical_variant_counts": dict(sorted(variant_counts.items())),
               "skipped": len(skipped), "validation_cases": len(validation), "train_only": True}
    _atomic_json(output_dir / "augmentation_counts.json", summary)
    return summary


class LoraCropDataset(Dataset):
    def __init__(self, manifest_file, tokenizer):
        self.root = Path(manifest_file).parent
        self.records = json.loads(Path(manifest_file).read_text(encoding="utf-8"))["records"]
        self.tokenizer = tokenizer

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        import torch
        record = self.records[index]
        image = np.asarray(Image.open(self.root / record["image_file"]).convert("RGB"), dtype=np.float32) / 127.5 - 1
        mask = np.asarray(Image.open(self.root / record["mask_file"]).convert("L"), dtype=np.float32) / 255
        pixel = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy((mask >= .5).astype(np.float32))[None]
        ids = self.tokenizer(record["prompt"], max_length=self.tokenizer.model_max_length,
                             padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
        return {"pixel_values": pixel, "mask": mask, "masked_pixel_values": pixel * (mask < .5),
                "input_ids": ids, "category_id": int(record["category_id"])}
