import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.augmentation.backends import InpaintingBackend
from src.augmentation.crops import CropTransform, composite_masked
from src.augmentation.masks import build_safe_mask, relative_erosion_pixels
from src.augmentation.pipeline import (
    _append_jsonl,
    read_metadata,
    run_augmentation,
    validate_augmented_coco,
)
from src.augmentation.quality import evaluate_quality
from src.augmentation.rare_crops import build_rare_crops, clip_annotation
from src.augmentation.rare_sampling import make_sampler, sampling_weights
from src.augmentation.review import apply_review, build_review
from src.augmentation.strategy_b import plan_strategy_b, run_strategy_b
from src.augmentation.strategy_c import plan_strategy_c, run_strategy_c
from src.augmentation.validation import validate_full_provenance
from src.utils.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def annotation(aid, image=1, category=12, bbox=(20, 20, 60, 60)):
    x, y, w, h = bbox
    return {
        "id": aid,
        "image_id": image,
        "category_id": category,
        "bbox": list(bbox),
        "area": w * h,
        "iscrowd": 0,
        "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
        "custom_field": "preserve",
    }


@pytest.fixture
def fixture_data(tmp_path):
    config = load_yaml(ROOT / "configs/augmentation_a.yaml")
    config["crop"]["output_size"] = 128
    config["generation"]["variants_per_instance"] = 2
    config["runtime"]["num_workers"] = 0
    config["qa"]["max_examples"] = 2
    rare = {"categories": {12: {"name": "Abstellraum", "role": "primary_rare"}}}
    image = np.full((100, 120, 3), 255, np.uint8)
    image[42:45, 30:65] = 0
    images = tmp_path / "source"
    images.mkdir()
    Image.fromarray(image).save(images / "train.png")
    # Deliberately absent validation image: generation must never open it.
    coco = {
        "images": [
            {"id": 1, "file_name": "train.png", "width": 120, "height": 100},
            {"id": 2, "file_name": "MUST_NOT_READ.png", "width": 120, "height": 100},
        ],
        "annotations": [
            annotation(10),
            annotation(11, category=1, bbox=(70, 10, 40, 70)),
            annotation(12, image=2),
        ],
        "categories": [{"id": i, "name": str(i)} for i in range(1, 20)],
    }
    return config, coco, rare, images, tmp_path / "output", tmp_path / "qa"


class FakeBackend(InpaintingBackend):
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    @property
    def metadata(self):
        return {"model_name": "test-only-fake", "model_revision": "test", "device": "cpu", "gpu": "CPU"}

    def generate(self, image, mask, prompt, seed, **kwargs):
        self.calls.append((prompt, seed))
        if self.fail:
            raise RuntimeError("temporary failure")
        # Adversarially change the whole crop, including every protected pixel.
        y, x = np.indices((image.height, image.width))
        values = (195 + (x + y) % 40).astype(np.uint8)
        return Image.fromarray(np.repeat(values[..., None], 3, axis=2))


def test_relative_erosion_and_mask_protection():
    assert relative_erosion_pixels([0, 0, 130, 132]) == 3
    assert relative_erosion_pixels([0, 0, 20, 20]) == 2
    assert relative_erosion_pixels([0, 0, 1000, 2000]) == 12
    image = np.full((100, 120, 3), 255, np.uint8)
    image[40:43, 30:65] = 0
    result = build_safe_mask(
        image,
        annotation(1),
        [annotation(2, bbox=(60, 0, 40, 90))],
        {"dark_threshold": "auto", "structure_dilation_px": 2},
    )
    assert np.all(~result.safe_mask | result.room_mask)
    assert not result.safe_mask[result.structure_mask].any()
    assert not result.safe_mask[result.neighbor_mask].any()
    assert not result.safe_mask[20:22].any()
    assert not result.safe_mask[38:45, 28:67].any()
    assert result.safe_mask[30, 30]


@pytest.mark.parametrize("bbox", [[0, 0, 30, 50], [80, 0, 40, 50], [0, 60, 30, 40], [80, 60, 40, 40]])
def test_crop_padding_inverse_and_exact_composite(bbox, tmp_path):
    original = np.random.default_rng(4).integers(0, 256, (100, 120, 3), dtype=np.uint8)
    transform = CropTransform.from_bbox(bbox, 120, 100, 1.0, 128)
    mask = np.zeros((100, 120), bool)
    x, y, w, h = bbox
    mask[y : y + h, x : x + w] = True
    model_mask = transform.forward_mask(mask)
    assert model_mask.dtype == bool
    assert set(np.unique(model_mask)) <= {False, True}
    generated = np.full((128, 128, 3), 87, np.uint8)
    mapped = transform.inverse_image(generated)
    assert mapped.shape[:2] == (transform.crop_size[1], transform.crop_size[0])
    final = composite_masked(original, mapped, mask, transform)
    path = tmp_path / "test.png"
    Image.fromarray(final).save(path)
    saved = np.asarray(Image.open(path))
    assert np.array_equal(saved[~mask], original[~mask])
    assert (saved[mask] == 87).all()


def test_qc_detects_violations_and_blank():
    original = np.full((20, 20, 3), 255, np.uint8)
    safe = np.zeros((20, 20), bool)
    safe[5:15, 5:15] = True
    bad = original.copy()
    bad[0, 0] = 0
    assert evaluate_quality(original, bad, safe, ~safe, 100, {}).rejection_reason == "outside_pixels_changed"
    assert not evaluate_quality(original, original, safe, ~safe, 100, {}).accepted
    bad = original.copy()
    bad[safe] = 0
    assert not evaluate_quality(original, bad, safe, ~safe, 100, {}).accepted


def test_a_end_to_end_resume_annotations_and_png(fixture_data):
    config, coco, rare, images, output, qa = fixture_data
    baseline = copy.deepcopy(coco)
    backend = FakeBackend()
    result = run_augmentation(config, coco, rare, {1}, images, output, qa, backend=backend, limit=1)
    assert result["accepted"] == 1
    result = run_augmentation(config, coco, rare, {1}, images, output, qa, backend=backend)
    assert result["accepted"] == 2 and len(backend.calls) == 2
    run_augmentation(config, coco, rare, {1}, images, output, qa, backend=backend)
    assert len(backend.calls) == 2
    records = read_metadata(output / "metadata.jsonl")
    assert [r["seed"] for r in records] == [42, 43]
    assert all(r["outside_change_ratio"] == r["protected_boundary_change_ratio"] == 0 for r in records)
    data = json.loads((output / "instances_train_aug_a.json").read_text())
    assert len(data["images"]) == 3 and len(data["annotations"]) == 6
    validate_augmented_coco(data, output, {1})
    assert validate_full_provenance(data, coco, images, output, {1})["verified_synthetic_images"] == 2
    for im in data["images"]:
        if im.get("synthetic"):
            anns = sorted(
                [a for a in data["annotations"] if a["image_id"] == im["id"]], key=lambda a: a["id"]
            )
            for source, duplicate in zip(coco["annotations"][:2], anns):
                assert {k: v for k, v in source.items() if k not in {"id", "image_id"}} == {
                    k: v for k, v in duplicate.items() if k not in {"id", "image_id"}
                }
    assert coco == baseline
    changed = copy.deepcopy(config)
    changed["prompt"]["template"] = "changed {category_name}"
    with pytest.raises(ValueError, match="changed"):
        run_augmentation(changed, coco, rare, {1}, images, output, qa, backend=backend)
    assert list((qa / "contact_sheets").glob("*.jpg"))


def test_retry_error_and_missing_output(fixture_data):
    config, coco, rare, images, output, qa = fixture_data
    backend = FakeBackend(fail=True)
    result = run_augmentation(config, coco, rare, {1}, images, output, qa, backend=backend, limit=1)
    assert result["errors"] == 1
    backend.fail = False
    result = run_augmentation(
        config, coco, rare, {1}, images, output, qa, backend=backend, limit=1, retry_errors=True
    )
    assert result["accepted"] == 1
    records = read_metadata(output / "metadata.jsonl")
    assert records[0]["synthetic_image_id"] == records[1]["synthetic_image_id"]
    (output / records[-1]["output_file"]).unlink()
    result = run_augmentation(config, coco, rare, {1}, images, output, qa, backend=backend, limit=1)
    assert result["accepted"] == 1


def test_torn_jsonl_resumable(tmp_path):
    path = tmp_path / "metadata.jsonl"
    path.write_text('{"sample_id":"torn')
    _append_jsonl(path, {"sample_id": "complete"})
    assert read_metadata(path) == [{"sample_id": "complete"}]


def test_clipped_crops_and_target_coverage(fixture_data):
    config, coco, rare, images, output, qa = fixture_data
    train = {**coco, "images": coco["images"][:1], "annotations": coco["annotations"][:2]}
    crops = build_rare_crops(train, images, output, {1}, {12}, {"output_size": 128})
    assert len(crops["images"]) == 1 and len(crops["annotations"]) == 2
    target = crops["annotations"][0]
    assert target["category_id"] == 12
    assert target["area"] == pytest.approx(3600 * (128 / 120) ** 2)
    for ann in crops["annotations"]:
        assert ann["area"] > 0
        assert all(0 <= coord <= 128 for p in ann["segmentation"] for coord in p)
    validate_augmented_coco(crops, output, {1})
    with pytest.raises(ValueError, match="Non-training"):
        build_rare_crops(coco, images, output, {1}, {12}, {"output_size": 128})


def test_bounded_weighted_sampler(fixture_data):
    _, coco, _, _, _, _ = fixture_data
    train = {**coco, "images": coco["images"][:1], "annotations": coco["annotations"][:2]}
    weights, rows = sampling_weights(coco, train, {12}, {"lambda": 2.0, "max_image_weight": 5.0})
    assert np.isfinite(weights).all() and (weights >= 1).all() and (weights <= 5).all()
    assert max(r["probability"] for r in rows) <= 5 / (len(weights) - 1 + 5)
    assert list(make_sampler(weights, 42)) == list(make_sampler(weights, 42))
    with pytest.raises(ValueError):
        sampling_weights(coco, train, {12}, {"lambda": float("nan")})


def test_b_automatic_acceptance_and_only_target_category_changes(fixture_data):
    _, coco, rare, images, output, qa = fixture_data
    config = load_yaml(ROOT / "configs/augmentation_b.yaml")
    config["classes"]["target_ids"] = [12]
    config["crop"]["output_size"] = 128
    config["runtime"]["num_workers"] = 0
    config["geometry"]["tolerance"] = 0.9
    plan, diagnostic = plan_strategy_b(coco, {1}, config)
    assert len(plan[0]) == 1 and plan[0][0].target_annotation_id == 11
    result = run_strategy_b(config, coco, rare, {1}, images, output, qa, backend=FakeBackend())
    assert result["accepted"] == 1
    path = qa / "review.csv"
    rows = build_review(output, path)
    assert rows[0]["status"] == "approved"
    assert rows[0]["semantic_correctness"] == "unverified"
    pending = apply_review(output, path, coco, {1}, images)
    assert len(pending["images"]) == 2  # Original plus automatically accepted B image.
    rows[0]["status"] = "approved"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    approved = apply_review(output, path, coco, {1}, images)
    synthetic = [a for a in approved["annotations"] if a["image_id"] != 1]
    for original, duplicate in zip(coco["annotations"][:2], synthetic):
        assert duplicate["segmentation"] == original["segmentation"]
        assert duplicate["category_id"] == (12 if original["id"] == 11 else original["category_id"])
        assert duplicate["bbox"] == original["bbox"]
    assert build_review(output, path)[0]["status"] == "approved"
    assert validate_full_provenance(approved, coco, images, output, {1})["verified_synthetic_images"] == 1


def test_tampered_annotation_is_rejected(fixture_data):
    config, coco, rare, images, output, qa = fixture_data
    run_augmentation(config, coco, rare, {1}, images, output, qa, backend=FakeBackend(), limit=1)
    data = json.loads((output / "instances_train_aug_a.json").read_text())
    data["annotations"][-1]["category_id"] = 16
    with pytest.raises(ValueError, match="annotations differ"):
        validate_full_provenance(data, coco, images, output, {1})


def test_empty_safe_region_skips_backend(fixture_data):
    config, coco, rare, images, output, qa = fixture_data
    config["mask"]["dark_threshold"] = 255
    backend = FakeBackend()
    result = run_augmentation(config, coco, rare, {1}, images, output, qa, backend=backend)
    assert result["rejected"] == 2 and not backend.calls


def test_clipping_can_split_concave_polygon():
    ann = annotation(1)
    ann["segmentation"] = [[0, 0, 10, 0, 10, 10, 8, 10, 8, 2, 2, 2, 2, 10, 0, 10]]
    transform = CropTransform.from_bbox([0, 4, 10, 6], 20, 20, 0, 128)
    clipped = clip_annotation(ann, transform)
    assert len(clipped["segmentation"]) == 2
    assert clipped["area"] == pytest.approx(24 * 12.8**2)


def test_source_pixel_change_invalidates_resume(fixture_data):
    config, coco, rare, images, output, qa = fixture_data
    run_augmentation(config, coco, rare, {1}, images, output, qa, backend=FakeBackend(), limit=1)
    Image.new("RGB", (120, 100), "gray").save(images / "train.png")
    with pytest.raises(ValueError, match="changed"):
        run_augmentation(config, coco, rare, {1}, images, output, qa, backend=FakeBackend())


def test_split_authority_rejects_validation_path():
    from src.augmentation.inputs import load_inputs

    config = load_yaml(ROOT / "configs/augmentation_a.yaml")
    config["dataset"]["train_split"] = "data/splits/val.txt"
    with pytest.raises(ValueError, match="frozen training split"):
        load_inputs(config)


def test_crop_inverse_preserves_coordinates_without_resampling():
    image = np.random.default_rng(8).integers(0, 256, (100, 120, 3), dtype=np.uint8)
    transform = CropTransform.from_bbox([13, 7, 20, 40], 120, 100, 0, 40)
    assert transform.pad_left == 10 and transform.pad_top == 0
    assert np.array_equal(transform.inverse_image(transform.forward_image(image)), image[7:47, 13:33])


def test_safety_checker_flag_is_terminal_rejection(fixture_data):
    class Flagged(FakeBackend):
        def generate(self, *args, **kwargs):
            self.calls.append(1)
            raise ValueError("safety_checker_flagged")

    config, coco, rare, images, output, qa = fixture_data
    backend = Flagged()
    result = run_augmentation(config, coco, rare, {1}, images, output, qa, backend=backend, limit=1)
    assert result["rejected"] == 1 and result["errors"] == 0
    run_augmentation(config, coco, rare, {1}, images, output, qa, backend=backend, limit=1, retry_errors=True)
    assert len(backend.calls) == 1


def test_b_audit_never_waits_for_human_input(fixture_data):
    _, coco, rare, images, output, qa = fixture_data
    config = load_yaml(ROOT / "configs/augmentation_b.yaml")
    config["classes"]["target_ids"] = [12]
    config["crop"]["output_size"] = 128
    config["runtime"]["num_workers"] = 0
    config["geometry"]["tolerance"] = 0.9
    run_strategy_b(config, coco, rare, {1}, images, output, qa, backend=FakeBackend())
    audit = qa / "review.csv"
    audit.write_text("an obsolete or incomplete human review sheet")
    result = apply_review(output, audit, coco, {1}, images)
    assert len(result["images"]) == 2
    rows = build_review(output, audit)
    assert rows[0]["status"] == "approved" and rows[0]["decision_method"].startswith("automatic")


@pytest.mark.parametrize("mode", ["copy_paste", "feather_blend", "diffusion_inpaint"])
def test_c_exemplar_modes_preserve_provenance_and_only_relabel_target(fixture_data, mode):
    _, coco, rare, images, output, qa = fixture_data
    config = load_yaml(ROOT / "configs/augmentation_c.yaml")
    config["classes"]["source_ids"] = [1]
    config["classes"]["target_ids"] = [12]
    config["crop"]["output_size"] = 128
    config["geometry"]["tolerance"] = 0.9
    config["runtime"]["num_workers"] = 0
    config["semantic_quality"]["min_donor_dark_pixels"] = 1
    plan, diagnostics = plan_strategy_c(coco, {1}, config, mode)
    assert len(plan[0]) == 1
    candidate = plan[0][0]
    assert candidate.target_annotation_id == 11
    assert candidate.donor_annotation_id == 10
    assert diagnostics["12"]["real_donors"] == 1

    backend = FakeBackend()
    result = run_strategy_c(config, coco, rare, {1}, images, output / mode, qa / mode, mode, backend=backend)
    assert result["accepted"] == 1 and result["semantic_correctness"] == "real_train_exemplar_preserved"
    assert len(backend.calls) == (1 if mode == "diffusion_inpaint" else 0)
    generated = json.loads((output / mode / "instances_train_aug_c.json").read_text())
    validate_augmented_coco(generated, output / mode, {1})
    assert (
        validate_full_provenance(generated, coco, images, output / mode, {1})["verified_synthetic_images"]
        == 1
    )
    synthetic_image = next(image for image in generated["images"] if image.get("synthetic"))
    assert synthetic_image["strategy"] == "c"
    assert synthetic_image["donor_annotation_id"] == 10
    synthetic = sorted(
        (a for a in generated["annotations"] if a["image_id"] == synthetic_image["id"]), key=lambda a: a["id"]
    )
    for original, duplicate in zip(coco["annotations"][:2], synthetic):
        assert duplicate["segmentation"] == original["segmentation"]
        assert duplicate["bbox"] == original["bbox"]
        assert duplicate["category_id"] == (12 if original["id"] == 11 else original["category_id"])
    record = read_metadata(output / mode / "metadata.jsonl")[-1]
    assert record["donor_dark_retention_fraction"] >= 0.95
    assert record["outside_change_ratio"] == record["protected_boundary_change_ratio"] == 0


def test_c_pairing_rejects_non_train_donor(fixture_data):
    _, coco, _, _, _, _ = fixture_data
    config = load_yaml(ROOT / "configs/augmentation_c.yaml")
    config["classes"]["source_ids"] = [12]
    config["classes"]["target_ids"] = [12]
    # The only category-12 target outside image 1 must never become a donor or target.
    plan, _ = plan_strategy_c(coco, {1}, config, "copy_paste")
    assert all(c.source_image_id == 1 and c.donor_image_id == 1 for c in plan[0])
