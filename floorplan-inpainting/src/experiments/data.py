"""Training datasets and samplers for the controlled Phase 3 matrix."""
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, RandomSampler, Sampler, WeightedRandomSampler

from src.data.coco_dataset import FloorplanCocoDataset, read_split
from src.data.transforms import DetectionTransform
from src.utils.config import resolve_path, split_dir


SOURCE_FILES = {
    "original": ("data/processed/instances_deduplicated.json", "../area_segmentation_coco/images", "full_plan"),
    "strategy_a": ("data/augmented_a/instances_train_aug_a.json", "data/augmented_a", "full_plan"),
    "strategy_a_crops": ("data/augmented_a/crops/instances_train_aug_a_crops.json", "data/augmented_a/crops", "rare_crop"),
    "strategy_b": ("data/augmented_b/instances_train_aug_b.json", "data/augmented_b", "full_plan"),
    "strategy_c_copy_paste": ("data/augmented_c/copy_paste/instances_train_aug_c.json",
                               "data/augmented_c/copy_paste", "full_plan"),
    "strategy_c_copy_paste_multi": ("data/augmented_c/copy_paste_multi/instances_train_augmented.json",
                                     "data/augmented_c/copy_paste_multi", "full_plan"),
    "strategy_c_feather_blend": ("data/augmented_c/feather_blend/instances_train_aug_c.json",
                                  "data/augmented_c/feather_blend", "full_plan"),
    "strategy_c_diffusion_inpaint": ("data/augmented_c/diffusion_inpaint/instances_train_aug_c.json",
                                      "data/augmented_c/diffusion_inpaint", "full_plan"),
    "strategy_lora_b": ("data/augmented_lora_b/instances_train_augmented.json",
                         "data/augmented_lora_b", "full_plan"),
    # Confirmatory study (e16/e17): two multi-room datasets composed from the
    # SAME accepted (target, donor) pairs with the same builder seed, so they
    # differ only in how the rare exemplar pixels were integrated.
    "strategy_c_copy_paste_matched": ("data/augmented_c_matched/copy_paste_multi/instances_train_augmented.json",
                                      "data/augmented_c_matched/copy_paste_multi", "full_plan"),
    "strategy_c_wide_matched": ("data/augmented_c_matched/inpaint_wide_multi/instances_train_augmented.json",
                                "data/augmented_c_matched/inpaint_wide_multi", "full_plan"),
}

MATCHED_SOURCE_SUBDIRS = {
    "strategy_c_copy_paste_matched": "copy_paste_multi",
    "strategy_c_wide_matched": "inpaint_wide_multi",
}


def _source_annotations(original):
    grouped = defaultdict(list)
    for annotation in original["annotations"]:
        grouped[annotation["image_id"]].append(annotation)
    for annotations in grouped.values():
        annotations.sort(key=lambda item: item["id"])
    return grouped


def _target_annotation_ids(coco, original):
    """Resolve the deliberately modified annotation in A/B/C images and crops."""
    source = _source_annotations(original)
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    targets = set()
    for info in coco["images"]:
        if info.get("strategy") not in {"a", "b", "c", "c_multi", "lora_b", "lora_multi"}:
            continue
        annotations = by_image[info["id"]]
        if info.get("strategy") in {"c_multi", "lora_multi"}:
            originals = source.get(info.get("source_image_id"), [])
            positions = {a["id"]: index for index, a in enumerate(originals)}
            ordered = sorted(annotations, key=lambda item: item["id"])
            for source_id in info["converted_annotation_ids"]:
                if source_id not in positions or len(ordered) != len(originals):
                    raise ValueError(f"Cannot trace multi-room target for image {info['id']}")
                targets.add(ordered[positions[source_id]]["id"])
            continue
        target_id = info["target_annotation_id"]
        crop_matches = [a for a in annotations if a.get("source_annotation_id") == target_id]
        if crop_matches:
            if len(crop_matches) != 1:
                raise ValueError(f"Ambiguous crop target for image {info['id']}")
            targets.add(crop_matches[0]["id"])
            continue
        originals = source.get(info.get("source_image_id"), [])
        positions = [index for index, a in enumerate(originals) if a["id"] == target_id]
        ordered = sorted(annotations, key=lambda item: item["id"])
        if len(positions) != 1 or len(ordered) != len(originals):
            raise ValueError(f"Cannot trace synthetic target for image {info['id']}")
        targets.add(ordered[positions[0]]["id"])
    return targets


class ExperimentCocoDataset(Dataset):
    """COCO dataset retaining multiclass provenance under agnostic labels."""
    def __init__(self, source_name, image_ids, transform, class_agnostic=False, source_spec=None):
        if source_name not in SOURCE_FILES and source_spec is None:
            raise ValueError(f"Unknown experiment source: {source_name}")
        annotation_file, image_dir, sample_kind = source_spec or SOURCE_FILES[source_name]
        self.source_name = source_name
        self.sample_kind = sample_kind
        self.image_dir = resolve_path(image_dir)
        self.coco = json.loads(resolve_path(annotation_file).read_text(encoding="utf-8"))
        original = json.loads(resolve_path(SOURCE_FILES["original"][0]).read_text(encoding="utf-8"))
        self.target_annotation_ids = _target_annotation_ids(self.coco, original)
        frozen_train = set(map(int, image_ids))
        # Cross-validation: the synthetic sets were generated from the original
        # 70/15/15 train split, so under a CV fold some synthetic images derive
        # from images now in this fold's val/test. Dropping them (rather than
        # raising) is the leakage-safe behaviour: train only on synthetic whose
        # source image is in the current train split. Unset -> original strict
        # behaviour, so every non-CV run is byte-identical.
        drop_out_of_fold = bool(os.environ.get("FP_CV_DROP_OOF_SYNTHETIC"))
        selected = []
        for info in self.coco["images"]:
            source_id = int(info.get("source_image_id", info["id"]))
            if source_id in frozen_train:
                selected.append(info)
            elif info.get("synthetic") or source_name != "original":
                if drop_out_of_fold:
                    continue
                raise ValueError(f"{source_name} contains image derived outside frozen train split: {source_id}")
        self.images = sorted(selected, key=lambda item: item["id"])
        allowed = {info["id"] for info in self.images}
        self.annotations = defaultdict(list)
        for annotation in self.coco["annotations"]:
            if annotation["image_id"] in allowed:
                self.annotations[annotation["image_id"]].append(annotation)
        self.transform = transform
        self.class_agnostic = bool(class_agnostic)
        self.records = []
        for info in self.images:
            occurrences = []
            for annotation in self.annotations[info["id"]]:
                source = "real"
                if annotation["id"] in self.target_annotation_ids:
                    source = {"a": "synthetic_A", "b": "synthetic_B", "c": "synthetic_C", "c_multi": "synthetic_C_MULTI", "lora_b": "synthetic_LORA", "lora_multi": "synthetic_LORA_MULTI"}[
                        info.get("strategy")]
                occurrences.append({"category_id": int(annotation["category_id"]), "source": source})
            self.records.append({"image_id": info["id"], "sample_kind": sample_kind,
                                 "source_name": source_name, "occurrences": occurrences,
                                 "source_image_id": int(info.get("source_image_id", info["id"])),
                                 "synthetic": bool(info.get("synthetic", False))})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        image = Image.open(self.image_dir / info["file_name"]).convert("RGB")
        width, height = image.size
        masks, labels, exposure = [], [], []
        for annotation in self.annotations[info["id"]]:
            if not isinstance(annotation["segmentation"], list):
                raise ValueError(f"Annotation {annotation['id']} is not polygonal")
            mask = FloorplanCocoDataset.rasterize(annotation["segmentation"], height, width)
            if not mask.any():
                continue
            masks.append(mask)
            labels.append(1 if self.class_agnostic else annotation["category_id"])
            source = "real"
            if annotation["id"] in self.target_annotation_ids:
                source = {"a": "synthetic_A", "b": "synthetic_B", "c": "synthetic_C", "c_multi": "synthetic_C_MULTI", "lora_b": "synthetic_LORA", "lora_multi": "synthetic_LORA_MULTI"}[
                    info.get("strategy")]
            exposure.append({"category_id": int(annotation["category_id"]), "source": source})
        if masks:
            masks_tensor = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes = []
            for mask in masks:
                ys, xs = np.where(mask)
                boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            masks_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        target = {"boxes": boxes, "labels": torch.as_tensor(labels, dtype=torch.int64),
                  "masks": masks_tensor, "image_id": torch.tensor(info["id"], dtype=torch.int64),
                  "area": masks_tensor.flatten(1).sum(1).to(torch.float32),
                  "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
                  "_exposure": exposure, "_sample_kind": self.sample_kind}
        return self.transform(image, target) if self.transform else (image, target)


def build_training_dataset(experiment_cfg, model_cfg, class_agnostic=False):
    train_ids = read_split(split_dir() / "train.txt")
    transform = DetectionTransform(True, model_cfg["horizontal_flip_probability"],
                                   model_cfg["brightness"], model_cfg["contrast"])
    sources = experiment_cfg["dataset"]["sources"]
    overrides = experiment_cfg["dataset"].get("source_overrides", {})
    datasets = [ExperimentCocoDataset(name, train_ids, transform, class_agnostic,
                tuple(overrides[name][key] for key in ("annotation_file", "image_dir", "sample_kind")) if name in overrides else None)
                for name in sources]
    if not datasets or any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("Every configured training source must be non-empty")
    combined = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    records = [record for dataset in datasets for record in dataset.records]
    return combined, records


def _bounded_rare_weights(records, original_counts, rare_ids, sampling_cfg):
    result = []
    multiplier = float(sampling_cfg.get("lambda", 2.0))
    cap = float(sampling_cfg.get("max_image_weight", 5.0))
    for record in records:
        counts = Counter(o["category_id"] for o in record["occurrences"] if o["category_id"] in rare_ids)
        score = sum(count / np.sqrt(original_counts[category]) for category, count in counts.items())
        result.append(min(cap, 1.0 + multiplier * score))
    return np.asarray(result, dtype=np.float64)


class FixedOrderSampler(Sampler):
    """Yield a precomputed index list (used by the paired-variant design)."""

    def __init__(self, indices):
        self.indices = [int(index) for index in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def paired_variant_indices(records, seed, samples_per_epoch, variant_probability):
    """Exposure-parity sampling: e0's plan order, synthetic variants swapped in.

    Real plans are the non-synthetic records. Their per-epoch order is
    ``torch.randperm(n_real)`` from a generator seeded with ``seed``, which is
    exactly the order RandomSampler(replacement=False) gives e0 for the same
    seed (e0's records are the same real plans sorted by image id). Each drawn
    real plan is then replaced, with probability ``variant_probability``, by
    one of its own synthetic variants (uniform choice), using an independent
    generator. Every real plan therefore occupies exactly one step per epoch,
    the step budget is unchanged, and two arms that share a seed see the same
    plan sequence; they differ only in the pixels of the swapped draws.
    """
    real = sorted((index for index, record in enumerate(records) if not record.get("synthetic")),
                  key=lambda index: int(records[index]["image_id"]))
    if len(real) != int(samples_per_epoch):
        raise ValueError(f"paired_variant needs one step per real plan: {len(real)} real plans, "
                         f"{samples_per_epoch} steps per epoch")
    if not 0.0 <= float(variant_probability) <= 1.0:
        raise ValueError("variant_probability must lie in [0, 1]")
    variants = {}
    for index, record in enumerate(records):
        if record.get("synthetic"):
            variants.setdefault(int(record["source_image_id"]), []).append(index)
    real_ids = {int(records[index]["image_id"]) for index in real}
    unknown = sorted(set(variants) - real_ids)
    if unknown:
        raise ValueError(f"Synthetic variants trace to plans outside the real pool: {unknown}")
    for key in variants:
        variants[key].sort(key=lambda index: int(records[index]["image_id"]))
    order = torch.randperm(len(real), generator=torch.Generator().manual_seed(int(seed))).tolist()
    # Separate stream for the swap decisions so the plan order never depends on them.
    swap_rng = np.random.default_rng([int(seed), 7919])
    indices = []
    for position in order:
        index = real[position]
        options = variants.get(int(records[index]["image_id"]), [])
        draw = swap_rng.random()
        pick = int(swap_rng.integers(len(options))) if options else -1
        if options and draw < float(variant_probability):
            index = options[pick]
        indices.append(index)
    return indices


def build_epoch_sampler(records, experiment_cfg, seed, samples_per_epoch, original_counts, rare_ids):
    """Build a deterministic fixed-budget sampler with an optional crop mixture."""
    sampling = experiment_cfg["sampling"]
    if sampling["mode"] == "paired_variant":
        return FixedOrderSampler(paired_variant_indices(
            records, seed, samples_per_epoch, float(sampling.get("variant_probability", 0.5))))
    weighted = sampling["mode"] == "rare_weighted"
    has_crops = any(record["sample_kind"] == "rare_crop" for record in records)
    if not weighted and not has_crops:
        return RandomSampler(range(len(records)), replacement=len(records) < samples_per_epoch,
                             num_samples=samples_per_epoch,
                             generator=torch.Generator().manual_seed(seed))
    raw = (_bounded_rare_weights(records, original_counts, rare_ids, sampling)
           if weighted else np.ones(len(records), dtype=np.float64))
    if has_crops:
        crop_probability = float(experiment_cfg["dataset"]["rare_crop_probability"])
        if not 0 < crop_probability < 1:
            raise ValueError("rare_crop_probability must be strictly between zero and one")
        crop = np.asarray([r["sample_kind"] == "rare_crop" for r in records])
        weights = np.zeros(len(records), dtype=np.float64)
        weights[crop] = crop_probability * raw[crop] / raw[crop].sum()
        weights[~crop] = (1.0 - crop_probability) * raw[~crop] / raw[~crop].sum()
    else:
        weights = raw / raw.sum()
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), samples_per_epoch,
                                 replacement=True, generator=torch.Generator().manual_seed(seed))


def original_train_counts():
    coco = json.loads(resolve_path(SOURCE_FILES["original"][0]).read_text(encoding="utf-8"))
    train_ids = set(read_split(split_dir() / "train.txt"))
    return Counter(a["category_id"] for a in coco["annotations"] if a["image_id"] in train_ids)
