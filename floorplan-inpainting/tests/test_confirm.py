"""Tests for the confirmatory study code (sampler, subset AP, statistics, seam)."""
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler

from src.augmentation.crops import CropTransform
from src.evaluation.confirm_stats import holm, permutation_test, t_sf, welch
from src.experiments.data import FixedOrderSampler, build_epoch_sampler, paired_variant_indices
from src.experiments.train import _validate_experiment
from src.utils.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def make_records(n_real=35, variants=None):
    records = [{"image_id": 100 + i, "source_image_id": 100 + i, "synthetic": False,
                "sample_kind": "full_plan", "occurrences": []} for i in range(n_real)]
    next_id = 9_000_000
    for source, count in (variants or {}).items():
        for _ in range(count):
            records.append({"image_id": next_id, "source_image_id": 100 + source, "synthetic": True,
                            "sample_kind": "full_plan", "occurrences": []})
            next_id += 1
    return records


@pytest.mark.parametrize("seed", [43, 1001 + 7, 2026 + 25])
def test_paired_variant_without_variants_reproduces_e0_order(seed):
    records = make_records()
    expected = list(RandomSampler(range(35), replacement=False, num_samples=35,
                                  generator=torch.Generator().manual_seed(seed)))
    assert paired_variant_indices(records, seed, 35, 0.5) == expected


def test_paired_variant_keeps_plan_order_and_swaps_only_own_variants():
    variants = {0: 2, 3: 1, 10: 4, 34: 1}
    records = make_records(variants=variants)
    seed = 1005
    plain = paired_variant_indices(make_records(), seed, 35, 0.5)
    mixed = paired_variant_indices(records, seed, 35, 0.5)
    assert len(mixed) == 35
    # Same plan sequence: position i draws the same source plan in both arms.
    assert [records[i]["source_image_id"] for i in mixed] == [records[i]["source_image_id"] for i in plain]
    sources = [records[i]["source_image_id"] for i in mixed]
    assert sorted(sources) == [100 + i for i in range(35)]
    for index in mixed:
        if records[index]["synthetic"]:
            assert records[index]["source_image_id"] - 100 in variants
    assert paired_variant_indices(records, seed, 35, 0.0) == plain
    always = paired_variant_indices(records, seed, 35, 1.0)
    assert sum(records[i]["synthetic"] for i in always) == len(variants)
    assert paired_variant_indices(records, seed, 35, 0.5) == mixed  # deterministic


def test_paired_variant_swap_rate_is_close_to_p():
    records = make_records(variants={i: 1 for i in range(35)})
    swapped = [sum(records[i]["synthetic"] for i in paired_variant_indices(records, s, 35, 0.5))
               for s in range(200)]
    assert abs(np.mean(swapped) / 35 - 0.5) < 0.03


def test_paired_variant_rejects_budget_mismatch_and_orphans():
    with pytest.raises(ValueError):
        paired_variant_indices(make_records(n_real=34), 1, 35, 0.5)
    orphan = make_records()
    orphan.append({"image_id": 9_999_999, "source_image_id": 5, "synthetic": True})
    with pytest.raises(ValueError):
        paired_variant_indices(orphan, 1, 35, 0.5)


def test_confirm_configs_pass_whitelist_and_dispatch_sampler():
    for name in ("e16_c_copy_paste_matched.yaml", "e17_c_inpaint_wide_matched.yaml"):
        cfg = load_yaml(ROOT / "configs/experiments_confirm" / name)
        _validate_experiment(cfg)
        sampler = build_epoch_sampler(make_records(variants={1: 1}), cfg, 1001, 35, {}, set())
        assert isinstance(sampler, FixedOrderSampler) and len(list(sampler)) == 35
    # The e0 and e8 configs used by the confirm array are the unchanged grid configs.
    _validate_experiment(load_yaml(ROOT / "configs/experiments/e0_original.yaml"))
    _validate_experiment(load_yaml(ROOT / "configs/experiments_c/e8_strategy_c_copy_paste.yaml"))


def _toy_coco(rng, n_images=12, categories=(1, 2)):
    from pycocotools.coco import COCO
    images, annotations, results, ann_id = [], [], [], 1
    for image_id in range(1, n_images + 1):
        images.append({"id": image_id, "width": 200, "height": 200, "file_name": f"{image_id}.png"})
        for _ in range(int(rng.integers(1, 5))):
            x, y = rng.uniform(0, 150, 2)
            w, h = rng.uniform(10, 50, 2)
            cat = int(rng.choice(categories))
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": cat, "bbox": [x, y, w, h],
                                "area": w * h, "iscrowd": 0})
            ann_id += 1
            if rng.random() < 0.8:
                jitter = rng.normal(0, 6, 4)
                results.append({"image_id": image_id, "category_id": cat, "score": float(rng.random()),
                                "bbox": [x + jitter[0], y + jitter[1], max(1, w + jitter[2]), max(1, h + jitter[3])]})
        for _ in range(int(rng.integers(0, 3))):
            results.append({"image_id": image_id, "category_id": int(rng.choice(categories)),
                            "score": float(rng.random()), "bbox": list(rng.uniform(0, 150, 2)) + [20.0, 20.0]})
    coco = COCO()
    coco.dataset = {"images": images, "annotations": annotations,
                    "categories": [{"id": c, "name": str(c)} for c in categories]}
    coco.createIndex()
    return coco, results


@pytest.mark.parametrize("categories", [(1,), (1, 2, 3)])
def test_subset_ap_replicates_cocoeval(categories):
    import contextlib
    import io
    from pycocotools.cocoeval import COCOeval
    from src.evaluation.subset_ap import PerImageEval
    rng = np.random.default_rng(len(categories))
    coco, results = _toy_coco(rng, categories=categories)
    ids = sorted(coco.getImgIds())
    per_image = PerImageEval(coco, results, ids, "bbox")
    assert per_image.check_against_cocoeval()
    subset = ids[:5]
    reference = COCOeval(coco, coco.loadRes(results), "bbox")
    reference.params.imgIds = subset
    with contextlib.redirect_stdout(io.StringIO()):
        reference.evaluate(); reference.accumulate(); reference.summarize()
    mine = per_image.ap(subset)
    assert abs(mine["AP"] - reference.stats[0]) < 1e-9
    assert abs(mine["AP50"] - reference.stats[1]) < 1e-9
    # Resamples with duplicates are accepted (bootstrap use).
    assert 0.0 <= per_image.ap(ids[:3] * 2 + ids[3:4])["AP"] <= 1.0


def test_statistics_against_t_tables():
    assert abs(t_sf(2.228, 10) - 0.025) < 5e-4
    assert abs(t_sf(1.812, 10) - 0.05) < 5e-4
    assert abs(t_sf(-1.0, 7) + t_sf(1.0, 7) - 1.0) < 1e-12
    result = welch([0.58, 0.59, 0.60, 0.57], [0.62, 0.63, 0.61, 0.64])
    assert result["delta"] > 0 and result["p_one_sided"] < 0.01
    assert holm({"H1": 0.01, "H2": 0.04}) == {"H1": (0.02, True), "H2": (0.04, True)}
    assert holm({"H1": 0.03, "H2": 0.04})["H1"] == (0.06, False)
    assert permutation_test([0, 0, 0], [1, 1, 1], 20_000) < 0.1


def test_seam_is_fixed_in_model_space():
    from src.augmentation.strategy_c import seam_width_in_original_pixels
    small = CropTransform.from_bbox([100, 100, 50, 50], 2000, 2000, 1.0, 512)   # side 150 -> scale 3.41
    large = CropTransform.from_bbox([100, 100, 300, 300], 2000, 2000, 1.0, 512)  # side 700 -> scale 0.73
    integration = {"seam_width_pixels": 6, "seam_model_pixels": 24}
    assert seam_width_in_original_pixels(integration, small) == 8
    assert seam_width_in_original_pixels(integration, large) == 33
    assert seam_width_in_original_pixels({"seam_width_pixels": 6}, large) == 6


def test_donor_stroke_mask_is_excluded_from_generation():
    from src.augmentation.strategy_c import _inpaint_mask, donor_stroke_mask
    image = np.full((60, 60, 3), 255, np.uint8)
    image[28:32, 10:50] = 0  # a "label" stroke crossing the seam band
    paste = np.zeros((60, 60), bool)
    paste[5:55, 5:55] = True
    generation = _inpaint_mask(paste, paste, 8)
    strokes = donor_stroke_mask(image, paste, 100, 2)
    assert (generation & strokes).any()
    assert not ((generation & ~strokes) & strokes).any()
    assert strokes[30, 12] and not strokes[10, 30]
