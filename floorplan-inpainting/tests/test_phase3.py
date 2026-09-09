from collections import Counter
from pathlib import Path

import pytest

from src.experiments.data import (build_epoch_sampler, build_training_dataset,
                                  original_train_counts)
from src.experiments.train import _validate_experiment
from src.utils.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]
requires_phase3_artifacts = pytest.mark.skipif(
    not all(
        (ROOT / path).exists()
        for path in (
            "data/splits/train.txt",
            "data/augmented_a/instances_train_aug_a.json",
            "data/augmented_b/instances_train_aug_b.json",
            "outputs/rare_crops/real/instances_train_aug_a_crops.json",
        )
    ),
    reason="cluster Phase 3 artifacts are not available",
)


def configs():
    return [load_yaml(path) for path in sorted((ROOT / "configs/experiments").glob("*.yaml"))]


def test_all_required_experiments_and_both_tracks_are_configured():
    loaded = configs()
    assert [item["id"] for item in loaded] == [f"e{i}" for i in range(8)]
    for item in loaded:
        _validate_experiment(item)
        assert item["tracks"] == ["multiclass", "agnostic"]
        assert item["budget"]["steps_per_epoch"] == 35


@requires_phase3_artifacts
def test_phase3_sources_are_train_only_and_agnostic_keeps_provenance():
    model = load_yaml(ROOT / "configs/baseline.yaml")
    experiment = load_yaml(ROOT / "configs/experiments/e5_strategy_a_weighted_crops.yaml")
    dataset, records = build_training_dataset(experiment, model, class_agnostic=True)
    assert len(dataset) == 109 + 104
    assert all(record["source_name"] in {"strategy_a", "strategy_a_crops"} for record in records)
    sources = Counter(o["source"] for record in records for o in record["occurrences"])
    assert sources["synthetic_A"] == 148  # 74 full-plan targets + their 74 crop targets.
    # The model sees category 1, while exposure remains attributable to original categories.
    crop_dataset = dataset.datasets[1]
    synthetic_index = next(i for i, info in enumerate(crop_dataset.images) if info.get("synthetic"))
    _, target = crop_dataset[synthetic_index]
    assert set(target["labels"].tolist()) == {1}
    assert any(item["source"] == "synthetic_A" and 11 <= item["category_id"] <= 16
               for item in target["_exposure"])


@requires_phase3_artifacts
def test_fixed_budget_sampler_is_deterministic_and_crop_mix_is_bounded():
    model = load_yaml(ROOT / "configs/baseline.yaml")
    experiment = load_yaml(ROOT / "configs/experiments/e5_strategy_a_weighted_crops.yaml")
    _, records = build_training_dataset(experiment, model)
    counts, rare = original_train_counts(), set(range(11, 17))
    first = list(build_epoch_sampler(records, experiment, 43, 35, counts, rare))
    second = list(build_epoch_sampler(records, experiment, 43, 35, counts, rare))
    assert first == second and len(first) == 35
    crop_count = sum(records[index]["sample_kind"] == "rare_crop" for index in first)
    assert 5 <= crop_count <= 25


@requires_phase3_artifacts
def test_strategy_b_training_uses_only_automatically_accepted_outputs():
    model = load_yaml(ROOT / "configs/baseline.yaml")
    experiment = load_yaml(ROOT / "configs/experiments/e6_strategy_b.yaml")
    _, records = build_training_dataset(experiment, model)
    assert len(records) == 35 + 27
    assert sum(o["source"] == "synthetic_B" for record in records for o in record["occurrences"]) == 27


def test_strategy_c_experiment_configs_are_valid_and_isolated():
    loaded = [load_yaml(path) for path in sorted(
        (ROOT / "configs/experiments_c").glob("*.yaml"),
        key=lambda path: int(path.name.split("_")[0][1:]))]
    assert [item["id"] for item in loaded] == ["e8", "e9", "e10", "e11"]
    for item in loaded:
        _validate_experiment(item)
        assert item["output_root"] == "outputs/experiments_c_multi"
