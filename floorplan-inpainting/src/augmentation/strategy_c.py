"""Strategy C: exemplar-guided common-to-rare augmentation.

Pixels from a real train-split rare room are mapped into a geometry-compatible frequent room and
integrated by copy-paste, feather blending, or diffusion inpainting of the seams and uncovered interior.
"""

import copy
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .backends import StableDiffusionInpaintingBackend
from .crops import CropTransform, composite_masked
from .masks import build_safe_mask, rasterize_annotation
from .pipeline import (
    Candidate,
    _append_jsonl,
    _atomic_json,
    _minimum_mutable_fraction,
    _ordered_prefetch,
    _save_png_atomic,
    build_augmented_coco,
    exclusive_run,
    file_digest,
    latest_metadata,
    read_metadata,
    run_fingerprint,
    summarize,
    validate_config,
)
from .qa import build_contact_sheets
from .quality import evaluate_quality
from .strategy_b import room_geometry

MODES = {"copy_paste": 2_000_000, "feather_blend": 3_000_000, "diffusion_inpaint": 4_000_000}


def _geometry_distance(first, second):
    ratio_features = ("area", "aspect_ratio", "width", "height")
    distance = sum(abs(np.log(max(first[key], 1e-6) / max(second[key], 1e-6))) for key in ratio_features)
    return float(distance + 2.0 * abs(first["solidity"] - second["solidity"]))


def plan_strategy_c(coco, train_ids, config, mode):
    """Pair real rare donors with unique, geometry-compatible frequent targets."""
    if mode not in MODES:
        raise ValueError(f"Unknown Strategy C mode: {mode}")
    train = [a for a in coco["annotations"] if int(a["image_id"]) in train_ids]
    names = {int(c["id"]): c["name"] for c in coco["categories"]}
    counts = Counter(int(a["category_id"]) for a in train)
    geometry = {int(a["id"]): room_geometry(a) for a in train}
    sources = [a for a in train if int(a["category_id"]) in config["classes"]["source_ids"]]
    tolerance = float(config["geometry"]["tolerance"])
    maximum_reuse = int(config["selection"]["max_donor_reuse"])
    rng = np.random.default_rng(int(config["seed"]))
    used_targets, donor_uses, selected, diagnostics = set(), Counter(), [], {}

    # Scarcest categories first, so abundant ones cannot use up all compatible targets.
    target_ids = sorted(config["classes"]["target_ids"], key=lambda cid: (counts[int(cid)], int(cid)))
    for category_id in target_ids:
        category_id = int(category_id)
        donors = sorted(
            [a for a in train if int(a["category_id"]) == category_id], key=lambda a: int(a["id"])
        )
        if not donors:
            diagnostics[str(category_id)] = {
                "category_name": names[category_id],
                "real_count": 0,
                "planned": 0,
                "reason": "no_real_donor",
            }
            continue
        donor_geometries = [geometry[int(a["id"])] for a in donors]
        bounds = {}
        for feature in donor_geometries[0]:
            low, high = np.quantile([g[feature] for g in donor_geometries], [0.05, 0.95])
            bounds[feature] = [max(0.0, float(low * (1.0 - tolerance))), float(high * (1.0 + tolerance))]
        eligible = [
            a
            for a in sources
            if int(a["id"]) not in used_targets
            and all(low <= geometry[int(a["id"])][feature] <= high for feature, (low, high) in bounds.items())
        ]
        rng.shuffle(eligible)
        eligible.sort(
            key=lambda target: min(
                _geometry_distance(geometry[int(target["id"])], geometry[int(donor["id"])])
                for donor in donors
            )
        )
        target_total = min(
            int(np.ceil(config["balancing"]["multiplier"] * counts[category_id])),
            int(config["balancing"]["max_target_instances"]),
        )
        budget = max(0, target_total - counts[category_id])
        pairs = []
        for target in eligible:
            available = [donor for donor in donors if donor_uses[int(donor["id"])] < maximum_reuse]
            if not available or len(pairs) >= budget:
                break
            donor = min(
                available,
                key=lambda item: (
                    donor_uses[int(item["id"])],
                    _geometry_distance(geometry[int(target["id"])], geometry[int(item["id"])]),
                    int(item["id"]),
                ),
            )
            pairs.append((target, donor))
            used_targets.add(int(target["id"]))
            donor_uses[int(donor["id"])] += 1
        diagnostics[str(category_id)] = {
            "category_name": names[category_id],
            "real_count": counts[category_id],
            "target_total": target_total,
            "candidate_budget": budget,
            "geometry_eligible": len(eligible),
            "planned": len(pairs),
            "real_donors": len(donors),
            "geometry_range": bounds,
        }
        selected.extend((target, donor, category_id) for target, donor in pairs)

    selected.sort(key=lambda value: (value[2], int(value[0]["id"]), int(value[1]["id"])))
    stride = max(Counter(int(a["image_id"]) for a in coco["annotations"]).values())
    max_image_id = max(int(i["id"]) for i in coco["images"]) + MODES[mode]
    max_annotation_id = max(int(a["id"]) for a in coco["annotations"]) + MODES[mode]
    candidates = []
    for index, (target, donor, category_id) in enumerate(selected):
        candidates.append(
            Candidate(
                sample_id=f"c-{mode}-target{int(target['id'])}-donor{int(donor['id'])}",
                candidate_index=index,
                source_image_id=int(target["image_id"]),
                target_annotation_id=int(target["id"]),
                category_id=category_id,
                category_name=names[category_id],
                variant_index=0,
                seed=int(config["seed"]) + index,
                synthetic_image_id=max_image_id + index,
                annotation_id_start=max_annotation_id + index * stride,
                strategy="c",
                source_geometry=geometry[int(target["id"])],
                target_geometry_range=diagnostics[str(category_id)]["geometry_range"],
                donor_image_id=int(donor["image_id"]),
                donor_annotation_id=int(donor["id"]),
                donor_category_id=category_id,
                integration_mode=mode,
            )
        )
    return (candidates, stride), diagnostics


def _warp_donor(donor_image, donor_annotation, target_annotation, output_shape):
    dx, dy, dw, dh = map(float, donor_annotation["bbox"])
    tx, ty, tw, th = map(float, target_annotation["bbox"])
    if min(dw, dh, tw, th) <= 0:
        raise ValueError("invalid_donor_or_target_bbox")
    matrix = np.asarray(
        [[tw / dw, 0.0, tx - dx * tw / dw], [0.0, th / dh, ty - dy * th / dh]], dtype=np.float32
    )
    height, width = output_shape
    warped_image = cv2.warpAffine(
        donor_image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    donor_mask = rasterize_annotation(donor_annotation, donor_image.shape[0], donor_image.shape[1])
    warped_mask = cv2.warpAffine(
        donor_mask.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    return warped_image, warped_mask, matrix


def _compose_seed(original, warped_donor, editable, paste_mask, mode, feather_pixels):
    seed = original.copy()
    # Blank the old common-room content before adding the rare exemplar.
    seed[editable] = 255
    if mode == "copy_paste":
        seed[paste_mask] = warped_donor[paste_mask]
        retention = paste_mask.copy()
    else:
        distance = cv2.distanceTransform(paste_mask.astype(np.uint8), cv2.DIST_L2, 5)
        alpha = np.clip(distance / max(float(feather_pixels), 1.0), 0.0, 1.0)
        alpha = alpha[..., None]
        mixed = alpha * warped_donor.astype(np.float32) + (1.0 - alpha) * 255.0
        seed[paste_mask] = np.rint(mixed[paste_mask]).astype(np.uint8)
        retention = paste_mask & (alpha[..., 0] >= 0.999)
    return seed, retention


def seam_width_in_original_pixels(integration, transform):
    """Seam width in original pixels.

    ``seam_model_pixels`` (if set) fixes the seam in model space, so the diffusion band has the same
    width in latent cells regardless of room size; otherwise ``seam_width_pixels`` is used.
    """
    model_pixels = integration.get("seam_model_pixels")
    if not model_pixels:
        return int(integration["seam_width_pixels"])
    scale = float(transform.output_size) / float(transform.padded_side)
    return max(1, int(np.ceil(float(model_pixels) / scale)))


def donor_stroke_mask(warped_donor, paste_mask, dark_threshold, dilation_px=3):
    gray = cv2.cvtColor(warped_donor, cv2.COLOR_RGB2GRAY)
    strokes = paste_mask & (gray <= float(dark_threshold))
    if dilation_px > 0 and strokes.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1))
        strokes = cv2.dilate(strokes.astype(np.uint8), kernel).astype(bool)
    return strokes & paste_mask


def _inpaint_mask(editable, paste_mask, seam_width):
    width = max(1, int(seam_width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    eroded = cv2.erode(paste_mask.astype(np.uint8), kernel).astype(bool)
    seam = paste_mask & ~eroded
    holes = editable & ~paste_mask
    return (seam | holes) & editable


def _save_strategy_c_qa(path, prepared, donor_crop, seed_crop, model_mask, generated_model, final, record):
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = prepared["transform"]
    x0, y0, x1, y1 = t.crop_bbox
    difference = np.clip(
        np.abs(prepared["original"][y0:y1, x0:x1].astype(np.int16) - final[y0:y1, x0:x1].astype(np.int16))
        * 4,
        0,
        255,
    ).astype(np.uint8)
    panels = [
        (prepared["original"], "Original target plan", None),
        (prepared["target_crop"], "Frequent target crop", None),
        (donor_crop, "Real rare donor crop", None),
        (prepared["editable_crop"] * 255, "Editable target interior", "gray"),
        (seed_crop, "Warped exemplar seed", None),
        (model_mask.astype(np.uint8) * 255, "Inpainting mask", "gray"),
        (generated_model, "Generated/model crop", None),
        (difference, "Final difference (4x)", None),
        (final, "Final full plan", None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(27, 4.2))
    for axis, (value, title, cmap) in zip(axes, panels):
        axis.imshow(value, cmap=cmap, vmin=0, vmax=255)
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    fig.suptitle(
        f"mode={record['integration_mode']} target_ann={record['target_annotation_id']} "
        f"donor_ann={record['donor_annotation_id']} class={record['category_name']} "
        f"coverage={record['donor_coverage_fraction']:.3f} status={record['status']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def _prepare(candidate, indices, image_dir, config):
    source_info = indices["images"][candidate.source_image_id]
    donor_info = indices["images"][candidate.donor_image_id]
    target = indices["annotation_by_id"][candidate.target_annotation_id]
    donor = indices["annotation_by_id"][candidate.donor_annotation_id]
    original = np.asarray(
        Image.open(Path(image_dir) / source_info["file_name"]).convert("RGB"), dtype=np.uint8
    )
    donor_image = np.asarray(
        Image.open(Path(image_dir) / donor_info["file_name"]).convert("RGB"), dtype=np.uint8
    )
    if original.shape[:2] != (int(source_info["height"]), int(source_info["width"])):
        raise ValueError("target_source_dimensions_mismatch")
    if donor_image.shape[:2] != (int(donor_info["height"]), int(donor_info["width"])):
        raise ValueError("donor_source_dimensions_mismatch")
    others = [
        a
        for a in indices["annotations"][candidate.source_image_id]
        if int(a["id"]) != candidate.target_annotation_id
    ]
    mask_cfg = copy.deepcopy(config["mask"])
    mask_cfg["structure_protection"] = False
    mask_result = build_safe_mask(original, target, others, mask_cfg)
    transform = CropTransform.from_bbox(
        target["bbox"],
        original.shape[1],
        original.shape[0],
        config["crop"]["context_fraction"],
        config["crop"]["output_size"],
    )
    warped, warped_mask, matrix = _warp_donor(donor_image, donor, target, original.shape[:2])
    paste_mask = mask_result.safe_mask & warped_mask
    mode = candidate.integration_mode
    compose_mode = "copy_paste" if mode == "copy_paste" else "feather_blend"
    seed_image, retention_mask = _compose_seed(
        original,
        warped,
        mask_result.safe_mask,
        paste_mask,
        compose_mode,
        config["integration"]["feather_pixels"],
    )
    integration = config["integration"]
    seam_pixels = seam_width_in_original_pixels(integration, transform)
    generation_mask = (
        _inpaint_mask(mask_result.safe_mask, paste_mask, seam_pixels)
        if mode == "diffusion_inpaint"
        else np.zeros_like(paste_mask)
    )
    if mode == "diffusion_inpaint" and integration.get("protect_donor_strokes", False):
        # Keep donor strokes (label text, symbols) out of the diffusion mask; semantic QC still checks them.
        generation_mask &= ~donor_stroke_mask(
            warped,
            paste_mask,
            mask_result.dark_threshold,
            int(integration.get("donor_stroke_dilation_px", 3)),
        )
    if mode == "diffusion_inpaint":
        retention_mask &= ~generation_mask
    donor_transform = CropTransform.from_bbox(
        donor["bbox"],
        donor_image.shape[1],
        donor_image.shape[0],
        config["crop"]["context_fraction"],
        config["crop"]["output_size"],
    )
    return {
        "candidate": candidate,
        "source_info": source_info,
        "donor_info": donor_info,
        "target": target,
        "donor": donor,
        "original": original,
        "donor_image": donor_image,
        "mask_result": mask_result,
        "transform": transform,
        "warped_donor": warped,
        "paste_mask": paste_mask,
        "retention_mask": retention_mask,
        "generation_mask": generation_mask,
        "seed_image": seed_image,
        "model_image": transform.forward_image(seed_image),
        "model_mask": transform.forward_mask(generation_mask),
        "target_crop": original[transform.y0 : transform.y1, transform.x0 : transform.x1].copy(),
        "editable_crop": mask_result.safe_mask[transform.y0 : transform.y1, transform.x0 : transform.x1],
        "donor_crop": donor_transform.forward_image(donor_image),
        "affine_matrix": matrix.tolist(),
        "seam_original_pixels": int(seam_pixels),
        "model_scale": float(transform.output_size) / float(transform.padded_side),
    }


def _semantic_metrics(prepared, final, config):
    editable, paste = prepared["mask_result"].safe_mask, prepared["paste_mask"]
    retention = prepared["retention_mask"]
    coverage = float(paste.sum() / editable.sum()) if editable.any() else 0.0
    threshold = float(prepared["mask_result"].dark_threshold)
    donor_gray = cv2.cvtColor(prepared["warped_donor"], cv2.COLOR_RGB2GRAY)
    final_gray = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)
    donor_dark = retention & (donor_gray <= threshold)
    retained = donor_dark & (
        final_gray <= min(255.0, threshold + float(config["semantic_quality"]["dark_tolerance"]))
    )
    fraction = float(retained.sum() / donor_dark.sum()) if donor_dark.any() else 0.0
    metrics = {
        "donor_coverage_fraction": coverage,
        "donor_dark_pixels_evaluated": int(donor_dark.sum()),
        "donor_dark_retention_fraction": fraction,
    }
    reason = None
    if coverage < float(config["semantic_quality"]["min_donor_coverage_fraction"]):
        reason = "insufficient_donor_coverage"
    elif int(donor_dark.sum()) < int(config["semantic_quality"]["min_donor_dark_pixels"]):
        reason = "insufficient_donor_semantic_strokes"
    elif fraction < float(config["semantic_quality"]["min_dark_retention_fraction"]):
        reason = "donor_semantic_strokes_lost"
    return metrics, reason


@exclusive_run
def _run_strategy_c(
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
    mode=None,
):
    del rare_config
    validate_config(config)
    if mode not in MODES:
        raise ValueError(f"Unknown Strategy C mode: {mode}")
    output_dir, qa_dir = Path(output_dir), Path(qa_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "full_images").mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    (candidates, stride), diagnostics = plan_strategy_c(coco, train_ids, config, mode)
    _atomic_json(qa_dir / "pairing_diagnostics.json", diagnostics)
    if any(c.source_image_id not in train_ids or c.donor_image_id not in train_ids for c in candidates):
        raise ValueError("Strategy C source or donor is outside the frozen training split")
    if limit is not None and int(limit) < 0:
        raise ValueError("limit must be nonnegative")
    config = copy.deepcopy(config)
    config["integration_mode"] = mode
    config["input_hashes"] = {
        str(im["id"]): file_digest(Path(image_dir) / im["file_name"])
        for im in coco["images"]
        if int(im["id"]) in train_ids
    }
    fingerprint = run_fingerprint(coco, candidates, stride, config)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Source/configuration changed; use a new Strategy C output directory")
    _atomic_json(
        manifest_path,
        {
            "fingerprint": fingerprint,
            "config": config,
            "mode": mode,
            "candidates": [c.__dict__ for c in candidates],
            "annotation_stride": stride,
        },
    )
    records = read_metadata(metadata_path)
    if records and {r.get("run_fingerprint") for r in records} != {fingerprint}:
        raise ValueError("Existing Strategy C metadata has a different fingerprint")
    previous = latest_metadata(records)
    selected = candidates if limit is None else candidates[: int(limit)]
    pending = []
    for candidate in selected:
        prior = previous.get(candidate.sample_id)
        missing = (
            prior is not None
            and prior["status"] == "accepted"
            and (
                not (output_dir / prior["output_file"]).is_file()
                or file_digest(output_dir / prior["output_file"]) != prior.get("output_sha256")
            )
        )
        if prior is None or missing or (prior["status"] == "error" and retry_errors):
            pending.append(candidate)

    images = {int(i["id"]): i for i in coco["images"]}
    annotations, annotation_by_id = defaultdict(list), {}
    for annotation in coco["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
        annotation_by_id[int(annotation["id"])] = annotation
    indices = {"images": images, "annotations": annotations, "annotation_by_id": annotation_by_id}
    workers = int(config.get("runtime", {}).get("num_workers", 0) if num_workers is None else num_workers)
    active_backend = backend

    def prepare(candidate):
        return _prepare(candidate, indices, image_dir, config)

    for prepared in _ordered_prefetch(pending, prepare, workers):
        candidate, mask_result = prepared["candidate"], prepared["mask_result"]
        record = {
            "sample_id": candidate.sample_id,
            "candidate_index": candidate.candidate_index,
            "synthetic_image_id": candidate.synthetic_image_id,
            "source_image_id": candidate.source_image_id,
            "target_annotation_id": candidate.target_annotation_id,
            "source_category_id": int(prepared["target"]["category_id"]),
            "donor_image_id": candidate.donor_image_id,
            "donor_annotation_id": candidate.donor_annotation_id,
            "donor_category_id": candidate.donor_category_id,
            "category_id": candidate.category_id,
            "category_name": candidate.category_name,
            "integration_mode": mode,
            "strategy": "c",
            "seed": candidate.seed,
            "crop_bbox": prepared["transform"].crop_bbox,
            "crop_transform": prepared["transform"].to_dict(),
            "source_bbox": copy.deepcopy(prepared["target"]["bbox"]),
            "donor_bbox": copy.deepcopy(prepared["donor"]["bbox"]),
            "donor_to_target_affine": prepared["affine_matrix"],
            "erosion_pixels": mask_result.erosion_pixels,
            "mutable_fraction": mask_result.mutable_fraction,
            "dark_threshold": mask_result.dark_threshold,
            "source_geometry": candidate.source_geometry,
            "target_geometry_range": candidate.target_geometry_range,
            "prompt": config["prompt"]["template"].format(category_name=candidate.category_name),
            "negative_prompt": config["prompt"].get("negative_prompt"),
            "generation_steps": int(config["generation"]["num_inference_steps"]),
            "guidance_scale": float(config["generation"]["guidance_scale"]),
            "strength": float(config["generation"].get("strength", 1.0)),
            "run_fingerprint": fingerprint,
            "decision_method": "automatic_pixel_and_exemplar_qc",
            "semantic_correctness": "real_train_exemplar_preserved",
            "attempt": sum(r["sample_id"] == candidate.sample_id for r in records) + 1,
            "output_file": f"full_images/{candidate.sample_id}.png",
            "generation_time_seconds": 0.0,
            "seam_original_pixels": prepared["seam_original_pixels"],
            "model_scale": prepared["model_scale"],
            "model_mask_pixels": int(prepared["model_mask"].sum()),
        }
        minimum = _minimum_mutable_fraction(config["mask"], candidate.category_id)
        if mask_result.mutable_fraction < minimum or not prepared["paste_mask"].any():
            record.update(status="rejected", accepted=False, rejection_reason="editable_region_too_small")
            _append_jsonl(metadata_path, record)
            records.append(record)
            continue
        try:
            generated_model = prepared["model_image"]
            final = prepared["seed_image"].copy()
            if mode == "diffusion_inpaint":
                if not prepared["model_mask"].any():
                    raise ValueError("empty_inpainting_mask")
                if active_backend is None:
                    generation = dict(config["generation"])
                    revision_path = output_dir / "backend.json"
                    if revision_path.exists():
                        generation["model_revision"] = json.loads(revision_path.read_text())["model_revision"]
                    active_backend = StableDiffusionInpaintingBackend(
                        generation | {"negative_prompt": config["prompt"].get("negative_prompt")}
                    )
                    _atomic_json(revision_path, active_backend.metadata)
                record.update(active_backend.metadata)
                started = time.perf_counter()
                generated_model = np.asarray(
                    active_backend.generate(
                        Image.fromarray(prepared["model_image"]),
                        Image.fromarray(prepared["model_mask"].astype(np.uint8) * 255),
                        record["prompt"],
                        candidate.seed,
                        strength=float(config["generation"].get("strength", 1.0)),
                    ).convert("RGB"),
                    dtype=np.uint8,
                )
                record["generation_time_seconds"] = time.perf_counter() - started
                if generated_model.shape[:2] != prepared["model_image"].shape[:2]:
                    raise ValueError("invalid_generated_crop_dimensions")
                generated_crop = prepared["transform"].inverse_image(generated_model)
                final = composite_masked(
                    prepared["seed_image"], generated_crop, prepared["generation_mask"], prepared["transform"]
                )
            boundary = mask_result.room_mask & ~mask_result.eroded_mask
            quality = evaluate_quality(
                prepared["original"],
                final,
                mask_result.safe_mask,
                boundary,
                mask_result.dark_threshold,
                config["quality"],
            )
            semantic, semantic_reason = _semantic_metrics(prepared, final, config)
            record.update(quality.metrics)
            record.update(semantic)
            reason = quality.rejection_reason or semantic_reason
            record.update(
                status="accepted" if reason is None else "rejected",
                accepted=reason is None,
                rejection_reason=reason,
            )
            if reason is None:
                saved = _save_png_atomic(output_dir / record["output_file"], final)
                persisted = evaluate_quality(
                    prepared["original"],
                    saved,
                    mask_result.safe_mask,
                    boundary,
                    mask_result.dark_threshold,
                    config["quality"],
                )
                persisted_semantic, persisted_reason = _semantic_metrics(prepared, saved, config)
                if not persisted.accepted or persisted_reason:
                    (output_dir / record["output_file"]).unlink(missing_ok=True)
                    record.update(
                        status="rejected", accepted=False, rejection_reason="persisted_output_failed_qc"
                    )
                else:
                    record.update(persisted_semantic)
                    record["output_sha256"] = file_digest(output_dir / record["output_file"])
            raw_path = qa_dir / "raw_generated" / f"{candidate.sample_id}.png"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(generated_model).save(raw_path)
            row_path = qa_dir / "rows" / f"{candidate.sample_id}.png"
            record["raw_generated_path"] = str(raw_path.resolve())
            record["qa_row_path"] = str(row_path.resolve())
            _save_strategy_c_qa(
                row_path,
                prepared,
                prepared["donor_crop"],
                prepared["model_image"],
                prepared["model_mask"],
                generated_model,
                final,
                record,
            )
        except ValueError as error:
            record.update(status="rejected", accepted=False, rejection_reason=str(error))
        except Exception as error:
            record.update(status="error", accepted=False, rejection_reason=f"{type(error).__name__}: {error}")
        _append_jsonl(metadata_path, record)
        records.append(record)
        print(f"{candidate.sample_id}: {record['status']} {record.get('rejection_reason') or ''}", flush=True)

    latest = list(latest_metadata(records).values())
    build_augmented_coco(
        coco,
        train_ids,
        candidates,
        latest,
        image_dir,
        output_dir,
        stride,
        filename="instances_train_aug_c.json",
        include_originals=True,
    )
    build_contact_sheets(
        latest,
        qa_dir / "contact_sheets",
        int(config.get("qa", {}).get("max_examples", 30)),
        int(config.get("qa", {}).get("rows_per_sheet", 5)),
    )
    source_rare_instances = sum(value.get("real_count", 0) for value in diagnostics.values())
    summary = summarize(records, len(candidates), source_rare_instances)
    summary.update(
        {
            "strategy": "C",
            "integration_mode": mode,
            "semantic_correctness": "real_train_exemplar_preserved",
            "planned_candidates_per_target": {
                value["category_name"]: value.get("planned", 0) for value in diagnostics.values()
            },
            "unique_donor_annotations": len({c.donor_annotation_id for c in candidates}),
        }
    )
    _atomic_json(qa_dir / "summary.json", summary)
    return summary


def run_strategy_c(config, coco, rare_config, train_ids, image_dir, output_dir, qa_dir, mode, **kwargs):
    return _run_strategy_c(
        config, coco, rare_config, train_ids, image_dir, output_dir, qa_dir, mode=mode, **kwargs
    )
