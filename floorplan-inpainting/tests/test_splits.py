import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "outputs/splits/split_manifest.json").exists(),
    reason="cluster dataset artifacts are not available",
)


def test_exact_duplicates_removed():
    coco = json.loads((ROOT / "data/processed/instances_deduplicated.json").read_text())
    ids = {im["id"] for im in coco["images"]}
    assert len(ids) == 56
    assert not ({22, 24, 25, 26} & ids)
    assert len(coco["annotations"]) == 1633


def test_splits_are_disjoint_and_complete():
    splits = {
        name: {int(x) for x in (ROOT / f"data/splits/{name}.txt").read_text().split()}
        for name in ("train", "val", "test")
    }
    assert not (
        splits["train"] & splits["val"] or splits["train"] & splits["test"] or splits["val"] & splits["test"]
    )
    assert len(set.union(*splits.values())) == 56


def test_projects_never_cross_splits():
    manifest = json.loads((ROOT / "outputs/splits/split_manifest.json").read_text())
    groups = [set(manifest["project_groups"][name]) for name in ("train", "val", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_all_labels_present_in_training():
    manifest = json.loads((ROOT / "outputs/splits/split_manifest.json").read_text())
    assert all(value > 0 for value in manifest["per_class_instance_counts"]["train"].values())
