import copy
import fcntl
import functools
import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.augmentation.backends import StableDiffusionInpaintingBackend
from src.augmentation.crops import CropTransform, composite_masked
from src.augmentation.masks import build_safe_mask
from src.augmentation.qa import build_contact_sheets, save_qa_row
from src.augmentation.quality import evaluate_quality


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    candidate_index: int
    source_image_id: int
    target_annotation_id: int
    category_id: int
    category_name: str
    variant_index: int
    seed: int
    synthetic_image_id: int
    annotation_id_start: int
    strategy: str = "a"
    source_geometry: dict | None = None
    target_geometry_range: dict | None = None
    donor_image_id: int | None = None
    donor_annotation_id: int | None = None
    donor_category_id: int | None = None
    integration_mode: str | None = None


@dataclass
class PreparedCandidate:
    candidate: Candidate
    source_info: dict
    target_annotation: dict
    source_annotations: list
    original: np.ndarray
    mask_result: object
    transform: CropTransform
    source_crop: np.ndarray
    model_image: np.ndarray
    model_mask: np.ndarray


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve a crash-truncated last line as evidence, but delimit the next record.
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                handle.seek(0, 2)
                handle.write(b"\n")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_metadata(path):
    path = Path(path)
    records = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Torn writes are not completed samples; the stable key will be retried.
            continue
    return records


def latest_metadata(records):
    latest = {}
    for record in records:
        latest[record["sample_id"]] = record
    return latest


def plan_candidates(coco, train_ids, rare_config, augmentation_config):
    include_roles = set(augmentation_config["classes"]["include_roles"])
    configured = {int(key): value for key, value in rare_config["categories"].items()}
    categories = {int(item["id"]): item["name"] for item in coco["categories"]}
    eligible_ids = {key for key, value in configured.items() if value["role"] in include_roles}
    targets = sorted(
        (
            annotation
            for annotation in coco["annotations"]
            if int(annotation["image_id"]) in train_ids and int(annotation["category_id"]) in eligible_ids
        ),
        key=lambda annotation: int(annotation["id"]),
    )
    variants = int(augmentation_config["generation"]["variants_per_instance"])
    max_image_id = max(int(image["id"]) for image in coco["images"])
    max_annotation_id = max(int(annotation["id"]) for annotation in coco["annotations"])
    per_image = Counter(int(annotation["image_id"]) for annotation in coco["annotations"])
    annotation_stride = max(per_image.values())
    candidates = []
    for target in targets:
        for variant in range(variants):
            index = len(candidates)
            candidates.append(
                Candidate(
                    sample_id=f"a-ann{int(target['id'])}-v{variant}",
                    candidate_index=index,
                    source_image_id=int(target["image_id"]),
                    target_annotation_id=int(target["id"]),
                    category_id=int(target["category_id"]),
                    category_name=categories[int(target["category_id"])],
                    variant_index=variant,
                    seed=int(augmentation_config["seed"]) + index,
                    synthetic_image_id=max_image_id + 1 + index,
                    annotation_id_start=max_annotation_id + 1 + index * annotation_stride,
                )
            )
    return candidates, annotation_stride


def run_fingerprint(coco, candidates, annotation_stride, config):
    # Omit unset Strategy C fields so fingerprints of earlier A/B runs stay valid.
    strategy_c_fields = {"donor_image_id", "donor_annotation_id", "donor_category_id", "integration_mode"}
    serialized_candidates = [
        {
            key: value
            for key, value in candidate.__dict__.items()
            if key not in strategy_c_fields or value is not None
        }
        for candidate in candidates
    ]
    relevant = {
        "source": coco,
        "candidates": serialized_candidates,
        "annotation_stride": annotation_stride,
        "crop": config["crop"],
        "mask": config["mask"],
        "generation": config["generation"],
        "quality": config["quality"],
        "prompt": config["prompt"],
        "input_hashes": config.get("input_hashes"),
    }
    encoded = json.dumps(relevant, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _minimum_mutable_fraction(config, category_id):
    overrides = config.get("min_mutable_fraction_by_category", {})
    return float(overrides.get(category_id, overrides.get(str(category_id), config["min_mutable_fraction"])))


def prepare_candidate(candidate, coco_indices, image_dir, config):
    source_info = coco_indices["images"][candidate.source_image_id]
    source_annotations = coco_indices["annotations"][candidate.source_image_id]
    target = coco_indices["annotation_by_id"][candidate.target_annotation_id]
    image_path = Path(image_dir) / source_info["file_name"]
    original = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    if original.shape[:2] != (source_info["height"], source_info["width"]):
        raise ValueError(f"Source dimensions mismatch: {image_path}")
    others = [item for item in source_annotations if int(item["id"]) != candidate.target_annotation_id]
    result = build_safe_mask(original, target, others, config["mask"])
    transform = CropTransform.from_bbox(
        target["bbox"],
        original.shape[1],
        original.shape[0],
        config["crop"]["context_fraction"],
        config["crop"]["output_size"],
    )
    source_crop = original[transform.y0 : transform.y1, transform.x0 : transform.x1].copy()
    return PreparedCandidate(
        candidate=candidate,
        source_info=source_info,
        target_annotation=target,
        source_annotations=source_annotations,
        original=original,
        mask_result=result,
        transform=transform,
        source_crop=source_crop,
        model_image=transform.forward_image(original),
        model_mask=transform.forward_mask(result.safe_mask),
    )


def _ordered_prefetch(candidates, prepare, workers):
    if workers <= 0:
        for candidate in candidates:
            yield prepare(candidate)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        iterator = iter(candidates)
        for _ in range(max(1, workers * 2)):
            try:
                pending.append(executor.submit(prepare, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(prepare, next(iterator)))
            except StopIteration:
                pass


def _record_base(prepared, fingerprint, config, backend_metadata):
    candidate, result = prepared.candidate, prepared.mask_result
    return {
        "sample_id": candidate.sample_id,
        "candidate_index": candidate.candidate_index,
        "synthetic_image_id": candidate.synthetic_image_id,
        "source_image_id": candidate.source_image_id,
        "target_annotation_id": candidate.target_annotation_id,
        "category_id": candidate.category_id,
        "category_name": candidate.category_name,
        "variant_index": candidate.variant_index,
        "seed": candidate.seed,
        "crop_bbox": prepared.transform.crop_bbox,
        "crop_transform": prepared.transform.to_dict(),
        "source_bbox": copy.deepcopy(prepared.target_annotation["bbox"]),
        "erosion_fraction": float(config["mask"]["erosion_fraction"]),
        "erosion_pixels": result.erosion_pixels,
        "mutable_fraction": result.mutable_fraction,
        "dark_threshold": result.dark_threshold,
        "model_name": backend_metadata.get("model_name", config["generation"]["model_name"]),
        "model_revision": backend_metadata.get(
            "model_revision", config["generation"].get("model_revision") or "main"
        ),
        "device": backend_metadata.get("device", config["generation"].get("device", "cuda")),
        "dtype": backend_metadata.get("dtype", config["generation"].get("dtype", "fp16")),
        "gpu": backend_metadata.get("gpu", "unknown"),
        "prompt": config["prompt"]["template"].format(category_name=candidate.category_name),
        "negative_prompt": config["prompt"].get("negative_prompt"),
        "generation_steps": int(config["generation"]["num_inference_steps"]),
        "guidance_scale": float(config["generation"]["guidance_scale"]),
        "run_fingerprint": fingerprint,
        "strategy": candidate.strategy,
        "source_category_id": prepared.target_annotation["category_id"],
        "source_geometry": candidate.source_geometry,
        "target_geometry_range": candidate.target_geometry_range,
        "decision_method": (
            "automatic_geometry_pixel_and_structural_qc"
            if candidate.strategy in {"lora_b", "lora_multi"}
            else "automatic_geometry_and_pixel_qc"
            if candidate.strategy == "b"
            else "automatic_pixel_qc"
        ),
        "semantic_correctness": (
            "unverified" if candidate.strategy in {"b", "lora_b", "lora_multi"} else "source_label_preserved"
        ),
    }


def _save_png_atomic(path, image_rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(image_rgb).save(temporary, format="PNG")
    reopened = np.asarray(Image.open(temporary).convert("RGB"), dtype=np.uint8)
    if reopened.shape != image_rgb.shape or not np.array_equal(reopened, image_rgb):
        temporary.unlink(missing_ok=True)
        raise ValueError("output_file_invalid")
    os.replace(temporary, path)
    return reopened


def _copy_original_train_images(coco, train_ids, image_dir, output_dir):
    copied = []
    destination_dir = Path(output_dir) / "full_images"
    destination_dir.mkdir(parents=True, exist_ok=True)
    for image in coco["images"]:
        if int(image["id"]) not in train_ids:
            continue
        source = Path(image_dir) / image["file_name"]
        destination = destination_dir / f"real_{image['id']}{source.suffix}"
        if not destination.is_file() or file_digest(destination) != file_digest(source):
            temporary = destination.with_name(destination.name + ".tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        record = copy.deepcopy(image)
        record["file_name"] = f"full_images/{destination.name}"
        copied.append(record)
    return copied


def build_augmented_coco(
    coco,
    train_ids,
    candidates,
    records,
    image_dir,
    output_dir,
    annotation_stride,
    filename="instances_train_aug_a.json",
    include_originals=True,
):
    accepted = {item["sample_id"]: item for item in records if item["status"] == "accepted"}
    images_by_id = {int(item["id"]): item for item in coco["images"]}
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    output = {
        key: copy.deepcopy(value) for key, value in coco.items() if key not in {"images", "annotations"}
    }
    output["images"] = (
        _copy_original_train_images(coco, train_ids, image_dir, output_dir) if include_originals else []
    )
    output["annotations"] = (
        [copy.deepcopy(item) for item in coco["annotations"] if int(item["image_id"]) in train_ids]
        if include_originals
        else []
    )
    for candidate in candidates:
        record = accepted.get(candidate.sample_id)
        if record is None:
            continue
        if candidate.source_image_id not in train_ids:
            raise ValueError("Non-training source in candidate export")
        source = copy.deepcopy(images_by_id[candidate.source_image_id])
        source.update(
            {
                "id": candidate.synthetic_image_id,
                "file_name": record["output_file"],
                "source_image_id": candidate.source_image_id,
                "target_annotation_id": candidate.target_annotation_id,
                "synthetic": True,
                "strategy": candidate.strategy,
                "sample_id": candidate.sample_id,
            }
        )
        if candidate.donor_annotation_id is not None:
            source.update(
                {
                    "donor_image_id": candidate.donor_image_id,
                    "donor_annotation_id": candidate.donor_annotation_id,
                    "donor_category_id": candidate.donor_category_id,
                    "integration_mode": candidate.integration_mode,
                }
            )
        output["images"].append(source)
        source_annotations = sorted(
            annotations_by_image[candidate.source_image_id], key=lambda item: int(item["id"])
        )
        if len(source_annotations) > annotation_stride:
            raise ValueError("Annotation stride is smaller than source annotation count")
        for ordinal, original in enumerate(source_annotations):
            duplicate = copy.deepcopy(original)
            duplicate["id"] = candidate.annotation_id_start + ordinal
            duplicate["image_id"] = candidate.synthetic_image_id
            if (
                candidate.strategy in {"b", "c", "lora_b", "lora_multi"}
                and original["id"] == candidate.target_annotation_id
            ):
                duplicate["category_id"] = candidate.category_id
            output["annotations"].append(duplicate)
    output["images"].sort(key=lambda item: int(item["id"]))
    output["annotations"].sort(key=lambda item: int(item["id"]))
    path = Path(output_dir) / filename
    validate_augmented_coco(output, Path(output_dir))
    _atomic_json(path, output)
    return output


def validate_augmented_coco(coco, output_dir, train_ids=None):
    output_dir = Path(output_dir)
    image_ids = [int(item["id"]) for item in coco["images"]]
    annotation_ids = [int(item["id"]) for item in coco["annotations"]]
    category_ids = {int(item["id"]) for item in coco["categories"]}
    if len(image_ids) != len(set(image_ids)) or len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("COCO IDs are not unique")
    if any(int(item["image_id"]) not in set(image_ids) for item in coco["annotations"]):
        raise ValueError("Annotation references a missing image")
    if any(int(item["category_id"]) not in category_ids for item in coco["annotations"]):
        raise ValueError("Annotation references a missing category")
    if any(not isinstance(item.get("segmentation"), list) for item in coco["annotations"]):
        raise ValueError("Augmented COCO contains non-polygon segmentation")
    for image in coco["images"]:
        if train_ids is not None and image.get("source_image_id", image["id"]) not in train_ids:
            raise ValueError("Validation/test contamination in dataset")
        path = output_dir / image["file_name"]
        with Image.open(path) as handle:
            handle.load()
            if handle.size != (int(image["width"]), int(image["height"])):
                raise ValueError(f"Image dimensions mismatch for {path}")
    from pycocotools.coco import COCO

    index = COCO()
    index.dataset = coco
    index.createIndex()
    for annotation in coco["annotations"]:
        polygons = annotation["segmentation"]
        if not polygons or any(len(p) < 6 or len(p) % 2 or not np.isfinite(p).all() for p in polygons):
            raise ValueError("Invalid polygon")
        if (
            len(annotation["bbox"]) != 4
            or not np.isfinite(annotation["bbox"]).all()
            or min(annotation["bbox"][2:]) <= 0
        ):
            raise ValueError("Invalid bbox")
        if not np.isfinite(annotation["area"]) or annotation["area"] <= 0:
            raise ValueError("Invalid area")
        if not index.annToRLE(annotation):
            raise ValueError("Invalid COCO mask")
        info = index.imgs[annotation["image_id"]]
        if "crop_transform" in info:
            for polygon in annotation["segmentation"]:
                points = np.asarray(polygon).reshape(-1, 2)
                if (
                    (points < -1e-6).any()
                    or (points[:, 0] > info["width"] + 1e-6).any()
                    or (points[:, 1] > info["height"] + 1e-6).any()
                ):
                    raise ValueError("Crop polygon exceeds image bounds")


def summarize(records, candidate_count, source_instance_count):
    latest = latest_metadata(records)
    values = list(latest.values())
    accepted = [item for item in values if item["status"] == "accepted"]
    rejected = [item for item in values if item["status"] == "rejected"]
    errors = [item for item in values if item["status"] == "error"]
    times = [
        float(item["generation_time_seconds"])
        for item in values
        if item.get("generation_time_seconds") is not None
    ]
    mutables = [
        float(item["mutable_fraction"]) for item in values if item.get("mutable_fraction") is not None
    ]
    outside = [
        float(item["outside_change_ratio"]) for item in values if item.get("outside_change_ratio") is not None
    ]
    boundary = [
        float(item["protected_boundary_change_ratio"])
        for item in values
        if item.get("protected_boundary_change_ratio") is not None
    ]
    return {
        "source_rare_instances": source_instance_count,
        "planned_candidates": candidate_count,
        "attempted": len(values),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "errors": len(errors),
        "acceptance_rate": len(accepted) / len(values) if values else 0.0,
        "per_class_generated_count": dict(
            sorted(Counter(item["category_name"] for item in accepted).items())
        ),
        "mean_mutable_fraction": float(np.mean(mutables)) if mutables else None,
        "max_outside_change_ratio": max(outside, default=None),
        "max_boundary_change_ratio": max(boundary, default=None),
        "mean_generation_time_seconds": float(np.mean(times)) if times else None,
        "total_generation_time_seconds": float(sum(times)),
        "model_name": values[-1].get("model_name") if values else None,
        "model_revision": values[-1].get("model_revision") if values else None,
        "gpu": values[-1].get("gpu") if values else None,
        "common_failure_modes": dict(Counter(item.get("rejection_reason") for item in rejected + errors)),
    }


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_run(function):
    @functools.wraps(function)
    def wrapped(config, coco, rare_config, train_ids, image_dir, output_dir, qa_dir, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with (Path(output_dir) / ".generation.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another generation process owns this output directory") from error
            return function(config, coco, rare_config, train_ids, image_dir, output_dir, qa_dir, **kwargs)

    return wrapped


@exclusive_run
def run_augmentation(
    config,
    coco,
    rare_config,
    train_ids,
    image_dir,
    output_dir,
    qa_dir,
    backend=None,
    limit=None,
    num_workers=None,
    retry_errors=False,
    candidate_plan=None,
    strategy="a",
):
    validate_config(config)
    output_dir, qa_dir = Path(output_dir), Path(qa_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "full_images").mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    candidates, stride = candidate_plan or plan_candidates(coco, train_ids, rare_config, config)
    if any(c.source_image_id not in train_ids for c in candidates):
        raise ValueError("Non-training generation source")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    config = copy.deepcopy(config)
    lora_path = config.get("generation", {}).get("lora_path")
    if lora_path:
        weight = Path(lora_path) / config["generation"].get(
            "lora_weight_name", "pytorch_lora_weights.safetensors"
        )
        if not weight.is_file():
            raise FileNotFoundError(f"LoRA weights not found: {weight}")
        config["generation"]["lora_sha256"] = file_digest(weight)
    config["input_hashes"] = {
        str(im["id"]): file_digest(Path(image_dir) / im["file_name"])
        for im in coco["images"]
        if im["id"] in train_ids
    }
    fingerprint = run_fingerprint(coco, candidates, stride, config)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Source/configuration changed; use a new output directory")
    _atomic_json(
        manifest_path,
        {
            "fingerprint": fingerprint,
            "config": config,
            "candidates": [c.__dict__ for c in candidates],
            "annotation_stride": stride,
        },
    )
    records = read_metadata(metadata_path)
    fingerprints = {item.get("run_fingerprint") for item in records}
    if fingerprints and fingerprints != {fingerprint}:
        raise ValueError("Existing metadata was created with a different source/configuration")
    previous = latest_metadata(records)
    selected = candidates if limit is None else candidates[: int(limit)]
    pending = []
    for candidate in selected:
        prior = previous.get(candidate.sample_id)
        accepted_file_missing = (
            prior is not None
            and prior["status"] == "accepted"
            and (
                not (output_dir / prior["output_file"]).is_file()
                or file_digest(output_dir / prior["output_file"]) != prior.get("output_sha256")
            )
        )
        if prior is None or accepted_file_missing or (prior["status"] == "error" and retry_errors):
            pending.append(candidate)

    images = {int(item["id"]): item for item in coco["images"]}
    annotations = defaultdict(list)
    annotation_by_id = {}
    for annotation in coco["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
        annotation_by_id[int(annotation["id"])] = annotation
    indices = {"images": images, "annotations": annotations, "annotation_by_id": annotation_by_id}
    workers = int(config.get("runtime", {}).get("num_workers", 0) if num_workers is None else num_workers)

    def prepare(candidate):
        return prepare_candidate(candidate, indices, image_dir, config)

    active_backend = backend

    for prepared in _ordered_prefetch(pending, prepare, workers):
        candidate, mask_result = prepared.candidate, prepared.mask_result
        minimum = _minimum_mutable_fraction(config["mask"], candidate.category_id)
        backend_info = active_backend.metadata if active_backend is not None else {}
        record = _record_base(prepared, fingerprint, config, backend_info)
        record["attempt"] = sum(item["sample_id"] == candidate.sample_id for item in records) + 1
        record["output_file"] = f"full_images/{candidate.sample_id}.png"
        record["generation_time_seconds"] = None
        if mask_result.mutable_fraction < minimum or not prepared.model_mask.any():
            record.update(status="rejected", accepted=False, rejection_reason="safe_mask_too_small")
            _append_jsonl(metadata_path, record)
            records.append(record)
            continue
        try:
            if active_backend is None:
                generation_config = dict(config["generation"])
                revision_path = output_dir / "backend.json"
                if revision_path.exists():
                    generation_config["model_revision"] = json.loads(revision_path.read_text())[
                        "model_revision"
                    ]
                active_backend = StableDiffusionInpaintingBackend(
                    generation_config | {"negative_prompt": config["prompt"].get("negative_prompt")}
                )
                record.update(active_backend.metadata)
                _atomic_json(revision_path, active_backend.metadata)
            start = time.perf_counter()
            generated_model = np.asarray(
                active_backend.generate(
                    Image.fromarray(prepared.model_image),
                    Image.fromarray(prepared.model_mask.astype(np.uint8) * 255),
                    record["prompt"],
                    candidate.seed,
                ).convert("RGB"),
                dtype=np.uint8,
            )
            record["generation_time_seconds"] = time.perf_counter() - start
            if generated_model.shape[:2] != prepared.model_image.shape[:2]:
                raise ValueError("invalid_generated_crop_dimensions")
            generated_crop = prepared.transform.inverse_image(generated_model)
            final = composite_masked(
                prepared.original, generated_crop, mask_result.safe_mask, prepared.transform
            )
            boundary = mask_result.room_mask & ~mask_result.eroded_mask
            quality = evaluate_quality(
                prepared.original,
                final,
                mask_result.safe_mask,
                boundary,
                mask_result.dark_threshold,
                config["quality"],
            )
            record.update(quality.metrics)
            record.update(
                status="accepted" if quality.accepted else "rejected",
                accepted=quality.accepted,
                rejection_reason=quality.rejection_reason,
            )
            if quality.accepted:
                saved = _save_png_atomic(output_dir / record["output_file"], final)
                persisted_quality = evaluate_quality(
                    prepared.original,
                    saved,
                    mask_result.safe_mask,
                    boundary,
                    mask_result.dark_threshold,
                    config["quality"],
                )
                if not persisted_quality.accepted:
                    (output_dir / record["output_file"]).unlink(missing_ok=True)
                    record.update(
                        status="rejected", accepted=False, rejection_reason="persisted_output_failed_qc"
                    )
                else:
                    record["output_sha256"] = file_digest(output_dir / record["output_file"])
            raw_path = qa_dir / "raw_generated" / f"{candidate.sample_id}.png"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(generated_model).save(raw_path)
            row_path = qa_dir / "rows" / f"{candidate.sample_id}.png"
            record["raw_generated_path"] = str(raw_path.resolve())
            record["qa_row_path"] = str(row_path.resolve())
            save_qa_row(
                row_path,
                prepared.original,
                prepared.source_crop,
                mask_result.room_mask[
                    prepared.transform.y0 : prepared.transform.y1,
                    prepared.transform.x0 : prepared.transform.x1,
                ]
                * 255,
                mask_result.safe_mask[
                    prepared.transform.y0 : prepared.transform.y1,
                    prepared.transform.x0 : prepared.transform.x1,
                ]
                * 255,
                generated_model,
                final,
                record,
            )
        except ValueError as error:
            record.update(status="rejected", accepted=False, rejection_reason=str(error))
        except Exception as error:
            record.update(
                status="error",
                accepted=False,
                rejection_reason=f"{type(error).__name__}: {error}",
            )
        _append_jsonl(metadata_path, record)
        records.append(record)
        print(f"{candidate.sample_id}: {record['status']} {record.get('rejection_reason') or ''}", flush=True)

    latest = list(latest_metadata(records).values())
    diagnostic_filename = (
        "instances_train_aug_a.json"
        if strategy == "a"
        else "candidates_qc_passed.json"
        if strategy in {"lora_b", "lora_multi"}
        else "candidates_review_only.json"
    )
    build_augmented_coco(
        coco,
        train_ids,
        candidates,
        latest,
        image_dir,
        output_dir,
        stride,
        filename=diagnostic_filename,
        include_originals=strategy == "a",
    )
    build_contact_sheets(
        latest,
        qa_dir / "contact_sheets",
        int(config.get("qa", {}).get("max_examples", 30)),
        int(config.get("qa", {}).get("rows_per_sheet", 5)),
    )
    source_instances = len({candidate.target_annotation_id for candidate in candidates})
    summary = summarize(records, len(candidates), source_instances)
    _atomic_json(qa_dir / "summary.json", summary)
    return summary


def validate_config(config):
    mask, crop = config["mask"], config["crop"]
    numeric = [
        mask["erosion_fraction"],
        mask["min_erosion_px"],
        mask["max_erosion_px"],
        mask["structure_dilation_px"],
        mask["min_mutable_fraction"],
        crop["context_fraction"],
    ]
    if not np.isfinite(numeric).all() or min(numeric) < 0:
        raise ValueError("Mask/crop settings must be finite and nonnegative")
    if mask["min_erosion_px"] < 1 or mask["max_erosion_px"] < mask["min_erosion_px"]:
        raise ValueError("Erosion bounds must satisfy 1 <= min <= max")
    if not 0 < mask["min_mutable_fraction"] <= 1:
        raise ValueError("min_mutable_fraction must be in (0, 1]")
    if crop["output_size"] < 64 or crop["output_size"] % 8:
        raise ValueError("Model resolution must be >=64 and divisible by 8")
    if config["generation"].get("variants_per_instance", 1) < 1:
        raise ValueError("variants_per_instance must be positive")
    if not config["quality"].get("require_zero_outside_changes", True) or not config["quality"].get(
        "require_zero_boundary_changes", True
    ):
        raise ValueError("Exact structural invariance cannot be disabled")
