import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .grouping import PROJECTS, image_to_project

SPLITS = ("train", "val", "test")


def _counts(coco, ids):
    ids = set(ids)
    instances, images = Counter(), defaultdict(set)
    for ann in coco["annotations"]:
        if ann["image_id"] in ids:
            instances[ann["category_id"]] += 1
            images[ann["category_id"]].add(ann["image_id"])
    return instances, {key: len(value) for key, value in images.items()}


def _score(assignment, group_ids, coco, cfg):
    n = len(coco["images"])
    ratios = cfg["ratios"]
    score = 0.0
    split_counts = {}
    for split in SPLITS:
        ids = [i for group, values in group_ids.items() if assignment[group] == split for i in values]
        split_counts[split], _ = _counts(coco, ids)
        score += cfg["size_weight"] * abs(len(ids) / n - ratios[split])
        # Missing common classes matter, but impossible coverage must not dominate size.
        score += cfg["coverage_weight"] * sum(split_counts[split][c] == 0 for c in range(1, 17)) / 16
    for category in cfg["primary_rare_ids"]:
        score += cfg["rare_coverage_weight"] * (
            split_counts["train"][category] == 0 or split_counts["test"][category] == 0
        )
    # Werkstatt is fragile but it can be represented on both sides if P01 and
    # P07 are separated; reward that uniquely feasible assignment.
    score += 10.0 * (split_counts["train"][17] == 0 or split_counts["test"][17] == 0)
    # A detector cannot learn a label absent from training. This also keeps the
    # single-project Labor class on the learning side of the leakage boundary.
    score += 50.0 * sum(split_counts["train"][c] == 0 for c in range(1, 20))
    return score


def search_group_split(coco, cfg):
    valid = {im["id"] for im in coco["images"]}
    group_ids = {p: sorted(set(v["image_ids"]) & valid) for p, v in PROJECTS.items()}
    rng = random.Random(cfg["seed"])
    projects = sorted(group_ids)
    best, candidates = None, []
    for _ in range(cfg["search_iterations"]):
        shuffled = projects[:]
        rng.shuffle(shuffled)
        assignment = {}
        # Seed all partitions, then sample near the desired project-level proportions.
        assignment[shuffled[0]], assignment[shuffled[1]], assignment[shuffled[2]] = SPLITS
        for project in shuffled[3:]:
            assignment[project] = rng.choices(SPLITS, weights=[.70, .15, .15])[0]
        score = _score(assignment, group_ids, coco, cfg)
        candidate = (score, tuple(assignment[p] for p in projects), assignment.copy())
        candidates.append(candidate)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assignment = best[2]
    # Refine only the best sampled assignment; doing this for every candidate
    # makes the small search needlessly expensive.
    improved = True
    while improved:
        improved = False
        current = _score(assignment, group_ids, coco, cfg)
        for project in projects:
            original = assignment[project]
            for destination in SPLITS:
                if destination == original or list(assignment.values()).count(original) == 1:
                    continue
                trial = assignment.copy(); trial[project] = destination
                trial_score = _score(trial, group_ids, coco, cfg)
                if trial_score < current - 1e-12:
                    assignment, current, improved = trial, trial_score, True
                    break
            if improved: break
    candidates.append((_score(assignment, group_ids, coco, cfg), tuple(assignment[p] for p in projects), assignment.copy()))
    candidates.sort(key=lambda item: item[:2])
    return assignment, [(x[0], x[2]) for x in candidates[:25]]


def write_split_outputs(coco, assignment, candidates, cfg, split_dir, output_dir, removed_ids):
    split_dir, output_dir = Path(split_dir), Path(output_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = image_to_project()
    ids_by_split = {s: sorted(im["id"] for im in coco["images"] if assignment[mapping[im["id"]]] == s) for s in SPLITS}
    for split, ids in ids_by_split.items():
        (split_dir / f"{split}.txt").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    instance_counts, image_counts = {}, {}
    for split, ids in ids_by_split.items():
        instance_counts[split], image_counts[split] = _counts(coco, ids)
    manifest = {
        "random_seed": cfg["seed"], "ratios_target": cfg["ratios"],
        "project_groups": {s: sorted(p for p, x in assignment.items() if x == s) for s in SPLITS},
        "image_ids": ids_by_split, "exact_duplicates_removed": removed_ids,
        "per_class_instance_counts": {s: {str(c): instance_counts[s][c] for c in cat_names} for s in SPLITS},
        "per_class_image_counts": {s: {str(c): image_counts[s].get(c, 0) for c in cat_names} for s in SPLITS},
        "not_fairly_evaluable": {
            str(c): "absent from train or test" for c in cat_names
            if instance_counts["train"][c] == 0 or instance_counts["test"][c] == 0
        },
        "top_candidate_scores": [score for score, _ in candidates],
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output_dir / "class_distribution_by_split.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["category_id", "category_name"] + [f"{s}_instances" for s in SPLITS] + [f"{s}_images" for s in SPLITS]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for c, name in cat_names.items():
            row = {"category_id": c, "category_name": name}
            row.update({f"{s}_instances": instance_counts[s][c] for s in SPLITS})
            row.update({f"{s}_images": image_counts[s].get(c, 0) for s in SPLITS})
            writer.writerow(row)
    with (output_dir / "group_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["project", "project_name", "split", "images"]); writer.writeheader()
        for p in sorted(assignment):
            writer.writerow({"project": p, "project_name": PROJECTS[p]["name"], "split": assignment[p],
                             "images": len(set(PROJECTS[p]["image_ids"]) & set(ids_by_split[assignment[p]]))})
    with (output_dir / "split_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "images", "annotations", "projects", "ratio"]); writer.writeheader()
        for s in SPLITS:
            writer.writerow({"split": s, "images": len(ids_by_split[s]), "annotations": sum(instance_counts[s].values()),
                             "projects": len(manifest["project_groups"][s]), "ratio": len(ids_by_split[s]) / len(coco["images"])})
    with (output_dir / "candidate_splits.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["rank", "score"] + list(PROJECTS)
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for rank, (score, candidate) in enumerate(candidates, 1):
            writer.writerow({"rank": rank, "score": score, **candidate})
    return manifest
