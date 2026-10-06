#!/usr/bin/env python3
"""Analyse the leakage-free re-score and the pre-registered confirmatory study.

Inputs (all produced by the confirm chain):
  <rescore-root>/<track>/<arm>/seed_<s>/heldout21_last/{result.json,metrics/coco_results.json.gz}
  <confirm-root>/agnostic/<arm>/seed_<s>/{result.json,training_summary.json,class_exposure_by_epoch.csv}
  <confirm-root>/agnostic/<arm>/seed_<s>/heldout21_last/{result.json,metrics/coco_results.json.gz}
  <confirm-root>/decisions.json, strength_selection.json
Outputs (<out>): summary.json plus CSV tables used by make_confirm_report.py.

The analysis follows docs/preregistration_confirm.md exactly:
  endpoint  class-agnostic mask AP@[.50:.95] of last.pth on the 21 held-out plans
  H1 e8 > e0, H2 e17 > e0: one-sided Welch, Holm over the active family (alpha .05)
  a hypothesis is supported only if Holm rejects AND the two-level (seed x image)
  bootstrap 95% CI of the difference excludes 0
  H3 e17 > e16 (alpha .05, same CI rule) is tested only if H2 is supported.
Everything else is descriptive.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pycocotools.coco import COCO  # noqa: E402

from src.data.coco_dataset import read_split  # noqa: E402
from src.data.grouping import image_to_project  # noqa: E402
from src.evaluation.confirm_stats import (hedges_g, holm, percentile_ci,  # noqa: E402
                                          permutation_test, welch)
from src.evaluation.subset_ap import PerImageEval, load_dump  # noqa: E402
from src.utils.config import resolve_path  # noqa: E402

GRID_SEEDS = (42, 123, 2026)
CONFIRM_SEEDS = tuple(range(1001, 1013))
GRID_ARMS = [
    ("e0", "Original (no aug)"), ("e1", "Rare-weighted sampling"), ("e2", "Strategy A"),
    ("e3", "Strategy A + w"), ("e4", "Strategy A + crops"), ("e5", "Strategy A + crops + w"),
    ("e6", "Strategy B"), ("e7", "Strategy B + w"), ("e8", "C copy-paste (multi-room)"),
    ("e9", "C feather-blend"), ("e10", "C diffusion inpaint (6 px seam)"),
    ("e11", "C diffusion inpaint + w"), ("e12", "LoRA Strategy B"), ("e13", "LoRA Strategy B + w"),
]
CONFIRM_ARMS = [
    ("e0", "Original (no aug)"),
    ("e8", "Copy-paste (e8 replication)"),
    ("e16", "Matched copy-paste"),
    ("e17", "Matched wide-seam inpainting"),
]
ANNOTATIONS = {"agnostic": "data/processed/instances_deduplicated_agnostic.json",
               "multiclass": "data/processed/instances_deduplicated.json"}
TAG = "heldout21_last"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class RunScores:
    """Per-image evaluation of one checkpoint, from its prediction dump."""

    def __init__(self, coco_gt, eval_dir, heldout):
        self.eval_dir = Path(eval_dir)
        result = read_json(self.eval_dir / "result.json")
        self.result_ap = float(result["test_metrics"]["mask"]["AP"])
        self.metrics = result["test_metrics"]
        self.checkpoint_epoch = result.get("checkpoint_epoch")
        dump = load_dump(self.eval_dir / "metrics" / "coco_results.json.gz")
        if sorted(dump["image_ids"]) != sorted(heldout):
            raise ValueError(f"{eval_dir}: dump was not made on the 21 held-out plans")
        self.per_image = PerImageEval(coco_gt, dump["segm"], heldout, "segm")
        self.per_image.check_against_cocoeval()
        full = self.per_image.ap(heldout)
        if abs(full["AP"] - self.result_ap) > 1e-6:
            raise AssertionError(f"{eval_dir}: offline AP {full['AP']} != recorded {self.result_ap}")

    def ap(self, image_ids):
        return self.per_image.ap(image_ids)


def collect_runs(root, track, arms, seeds, coco_gt, heldout, nested_seed=True):
    runs = {}
    for arm, _ in arms:
        for seed in seeds:
            eval_dir = Path(root) / track / arm / f"seed_{seed}" / TAG
            if not (eval_dir / "result.json").is_file():
                continue
            runs[(arm, seed)] = RunScores(coco_gt, eval_dir, heldout)
    return runs


def bootstrap_table(runs, arms, image_pool, n_boot, rng):
    """AP of every run on n_boot shared image resamples: dict (arm, seed) -> array."""
    resamples = [rng.choice(image_pool, size=len(image_pool), replace=True) for _ in range(n_boot)]
    return {key: np.array([run.ap(sample)["AP"] for sample in resamples]) for key, run in runs.items()}


def two_level_ci(boot, runs, control, treatment, rng, level=0.95):
    """Seed x image bootstrap CI of mean(treatment) - mean(control)."""
    ctrl = [key for key in runs if key[0] == control]
    trt = [key for key in runs if key[0] == treatment]
    if not ctrl or not trt:
        return None
    n_boot = len(next(iter(boot.values())))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        c = [boot[ctrl[i]][b] for i in rng.integers(len(ctrl), size=len(ctrl))]
        t = [boot[trt[i]][b] for i in rng.integers(len(trt), size=len(trt))]
        diffs[b] = np.mean(t) - np.mean(c)
    return percentile_ci(diffs, level)


def arm_values(runs, arm, image_ids, metric="AP"):
    keys = sorted(key for key in runs if key[0] == arm)
    return [runs[key].ap(image_ids)[metric] for key in keys], [key[1] for key in keys]


def describe(values):
    if not values:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {"n": len(values), "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def project_subsets(heldout):
    mapping = image_to_project()
    groups = {}
    for image_id in heldout:
        groups.setdefault(mapping[image_id], []).append(image_id)
    return dict(sorted(groups.items()))


def analyse_rescore(args, heldout, test11, val10, rng):
    out = {}
    for track in ("agnostic", "multiclass"):
        coco_gt = COCO(str(resolve_path(ANNOTATIONS[track])))
        runs = collect_runs(resolve_path(args.rescore_root), track, GRID_ARMS, GRID_SEEDS, coco_gt, heldout)
        if not runs:
            continue
        boot = bootstrap_table(runs, GRID_ARMS, np.array(heldout), args.rescore_boot, rng)
        projects = project_subsets(heldout)
        e0_vals, e0_seeds = arm_values(runs, "e0", heldout)
        rows, pvals = [], {}
        for arm, label in GRID_ARMS:
            values, seeds = arm_values(runs, arm, heldout)
            if not values:
                continue
            row = {"arm": arm, "label": label, "seeds": seeds, "heldout21": describe(values),
                   "heldout21_AP50": describe(arm_values(runs, arm, heldout, "AP50")[0]),
                   "test11": describe(arm_values(runs, arm, test11)[0]),
                   "val10": describe(arm_values(runs, arm, val10)[0]),
                   "per_project": {p: describe(arm_values(runs, arm, ids)[0]) for p, ids in projects.items()},
                   "checkpoint_epochs": sorted({runs[(arm, s)].checkpoint_epoch for s in seeds})}
            if track == "multiclass":
                rare = [runs[(arm, s)].metrics.get("rare_mask_AP") for s in seeds]
                row["rare_mask_AP_heldout21"] = describe([r for r in rare if r is not None])
            if arm != "e0" and e0_vals:
                test = welch(e0_vals, values)
                row["delta_vs_e0"] = test["delta"]
                row["welch_p_one_sided"] = test["p_one_sided"]
                row["per_seed_delta"] = {s: runs[(arm, s)].ap(heldout)["AP"] - runs[("e0", s)].ap(heldout)["AP"]
                                         for s in seeds if ("e0", s) in runs}
                row["ci95_seed_x_image"] = two_level_ci(boot, runs, "e0", arm, rng)
                row["delta_test11"] = (row["test11"]["mean"] - statistics.fmean(arm_values(runs, "e0", test11)[0]))
                pvals[arm] = test["p_one_sided"]
            rows.append(row)
        adjusted = holm(pvals)
        for row in rows:
            if row["arm"] in adjusted:
                row["holm_p"], row["holm_reject"] = adjusted[row["arm"]]
        out[track] = {"rows": rows, "n_runs": len(runs)}
    return out


def exposure_totals(run_dir):
    path = Path(run_dir) / "class_exposure_by_epoch.csv"
    if not path.is_file():
        return None
    totals = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                if key.endswith("_occurrences_seen"):
                    totals[key] = totals.get(key, 0) + int(value)
    return totals


def analyse_confirm(args, heldout, test11, val10, rng):
    confirm_root = resolve_path(args.confirm_root)
    decisions = read_json(confirm_root / "decisions.json") if (confirm_root / "decisions.json").is_file() else None
    coco_gt = COCO(str(resolve_path(ANNOTATIONS["agnostic"])))
    runs = collect_runs(confirm_root, "agnostic", CONFIRM_ARMS, CONFIRM_SEEDS, coco_gt, heldout)
    result = {"decisions": decisions, "arms": [], "tests": {}, "descriptive": {}, "n_runs": len(runs)}
    if not runs:
        return result
    boot = bootstrap_table(runs, CONFIRM_ARMS, np.array(heldout), args.boot, rng)
    projects = project_subsets(heldout)
    values = {arm: arm_values(runs, arm, heldout)[0] for arm, _ in CONFIRM_ARMS}
    for arm, label in CONFIRM_ARMS:
        vals, seeds = arm_values(runs, arm, heldout)
        if not vals:
            continue
        run_rows = []
        for seed in seeds:
            seed_dir = confirm_root / "agnostic" / arm / f"seed_{seed}"
            summary = read_json(seed_dir / "training_summary.json") if (seed_dir / "training_summary.json").is_file() else {}
            best = read_json(seed_dir / "result.json") if (seed_dir / "result.json").is_file() else None
            run_rows.append({"seed": seed, "heldout21_AP": runs[(arm, seed)].ap(heldout)["AP"],
                             "heldout21_AP50": runs[(arm, seed)].ap(heldout)["AP50"],
                             "test11_AP": runs[(arm, seed)].ap(test11)["AP"],
                             "val10_AP": runs[(arm, seed)].ap(val10)["AP"],
                             "best_pth_test11_AP": best["test_metrics"]["mask"]["AP"] if best else None,
                             "checkpoint_epoch": runs[(arm, seed)].checkpoint_epoch,
                             "runtime_hours": summary.get("runtime_seconds", 0) / 3600 if summary else None,
                             "gpu": summary.get("gpu"), "total_optimizer_steps": summary.get("total_optimizer_steps"),
                             "exposure": exposure_totals(seed_dir)})
        result["arms"].append({
            "arm": arm, "label": label, "seeds": seeds, "heldout21": describe(vals),
            "heldout21_AP50": describe(arm_values(runs, arm, heldout, "AP50")[0]),
            "test11": describe(arm_values(runs, arm, test11)[0]),
            "val10": describe(arm_values(runs, arm, val10)[0]),
            "per_project": {p: describe(arm_values(runs, arm, ids)[0]) for p, ids in projects.items()},
            "runs": run_rows})

    def contrast(control, treatment):
        a, b = values.get(control, []), values.get(treatment, [])
        if not a or not b:
            return None
        test = welch(a, b)
        ci = two_level_ci(boot, runs, control, treatment, rng)
        common = sorted(set(arm_values(runs, control, heldout)[1]) & set(arm_values(runs, treatment, heldout)[1]))
        paired = [runs[(treatment, s)].ap(heldout)["AP"] - runs[(control, s)].ap(heldout)["AP"] for s in common]
        return {"control": control, "treatment": treatment, "n_control": len(a), "n_treatment": len(b),
                "delta": test["delta"], "welch_t": test["t"], "welch_df": test["df"],
                "welch_p_one_sided": test["p_one_sided"],
                "permutation_p_one_sided": permutation_test(a, b, args.permutations),
                "ci95_seed_x_image": ci, "hedges_g": hedges_g(a, b),
                "seed_paired_mean": statistics.fmean(paired) if paired else None,
                "seed_paired_positive": sum(d > 0 for d in paired), "seed_paired_n": len(paired),
                "delta_test11": (statistics.fmean(arm_values(runs, treatment, test11)[0])
                                 - statistics.fmean(arm_values(runs, control, test11)[0])),
                "delta_val10": (statistics.fmean(arm_values(runs, treatment, val10)[0])
                                - statistics.fmean(arm_values(runs, control, val10)[0])),
                "delta_AP50": (statistics.fmean(arm_values(runs, treatment, heldout, "AP50")[0])
                               - statistics.fmean(arm_values(runs, control, heldout, "AP50")[0])),
                "delta_per_project": {p: statistics.fmean(arm_values(runs, treatment, ids)[0])
                                      - statistics.fmean(arm_values(runs, control, ids)[0])
                                      for p, ids in projects.items()}}

    family = {"H1": contrast("e0", "e8"), "H2": contrast("e0", "e17")}
    family = {name: c for name, c in family.items() if c is not None}
    adjusted = holm({name: c["welch_p_one_sided"] for name, c in family.items()})
    for name, c in family.items():
        p_adj, reject = adjusted[name]
        c["holm_p"], c["holm_reject"] = p_adj, reject
        c["ci_excludes_zero"] = c["ci95_seed_x_image"] is not None and c["ci95_seed_x_image"][0] > 0
        c["supported"] = bool(reject and c["ci_excludes_zero"])
    result["tests"] = family
    h3 = contrast("e16", "e17")
    if h3 is not None:
        tested = bool(family.get("H2", {}).get("supported"))
        h3["tested"] = tested
        h3["ci_excludes_zero"] = h3["ci95_seed_x_image"] is not None and h3["ci95_seed_x_image"][0] > 0
        h3["supported"] = bool(tested and h3["welch_p_one_sided"] is not None
                               and h3["welch_p_one_sided"] < 0.05 and h3["ci_excludes_zero"])
        result["tests"]["H3"] = h3
    for name, (control, treatment) in {"e16_vs_e0": ("e0", "e16"), "e8_vs_e16": ("e16", "e8"),
                                       "e17_vs_e16": ("e16", "e17")}.items():
        c = contrast(control, treatment)
        if c is not None:
            result["descriptive"][name] = c
    return result


def write_tables(summary, out):
    with (out / "rescore_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track", "arm", "label", "n", "heldout21_mean", "heldout21_sd", "test11_mean",
                         "val10_mean", "delta_vs_e0", "ci95_lo", "ci95_hi", "welch_p", "holm_p"])
        for track, block in summary.get("rescore", {}).items():
            for row in block["rows"]:
                ci = row.get("ci95_seed_x_image") or (None, None)
                writer.writerow([track, row["arm"], row["label"], row["heldout21"]["n"], row["heldout21"]["mean"],
                                 row["heldout21"]["sd"], row["test11"]["mean"], row["val10"]["mean"],
                                 row.get("delta_vs_e0"), ci[0], ci[1], row.get("welch_p_one_sided"),
                                 row.get("holm_p")])
    confirm = summary.get("confirm", {})
    with (out / "confirm_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", "heldout21_AP", "heldout21_AP50", "test11_AP", "val10_AP",
                         "best_pth_test11_AP", "checkpoint_epoch", "runtime_hours", "gpu"])
        for arm in confirm.get("arms", []):
            for run in arm["runs"]:
                writer.writerow([arm["arm"], run["seed"], run["heldout21_AP"], run["heldout21_AP50"],
                                 run["test11_AP"], run["val10_AP"], run["best_pth_test11_AP"],
                                 run["checkpoint_epoch"], run["runtime_hours"], run["gpu"]])
    with (out / "confirm_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test", "control", "treatment", "n_c", "n_t", "delta", "ci95_lo", "ci95_hi",
                         "welch_p", "perm_p", "holm_p", "supported", "hedges_g"])
        for name, c in {**confirm.get("tests", {}), **confirm.get("descriptive", {})}.items():
            ci = c.get("ci95_seed_x_image") or (None, None)
            writer.writerow([name, c["control"], c["treatment"], c["n_control"], c["n_treatment"], c["delta"],
                             ci[0], ci[1], c["welch_p_one_sided"], c["permutation_p_one_sided"],
                             c.get("holm_p"), c.get("supported"), c["hedges_g"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore-root", default="outputs/rescore")
    parser.add_argument("--confirm-root", default="outputs/confirm")
    parser.add_argument("--out", default="outputs/final/confirm")
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--rescore-boot", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    heldout = read_split(resolve_path("data/splits_heldout21/test.txt"))
    test11 = read_split(resolve_path("data/splits/test.txt"))
    val10 = read_split(resolve_path("data/splits/val.txt"))
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"endpoint": "class-agnostic mask AP@[.50:.95], last.pth (epoch 25), 21 held-out plans",
               "heldout_ids": heldout, "test11_ids": test11, "val10_ids": val10,
               "bootstrap": {"confirm": args.boot, "rescore": args.rescore_boot, "seed": args.seed},
               "permutations": args.permutations}
    summary["rescore"] = analyse_rescore(args, heldout, test11, val10, rng)
    summary["confirm"] = analyse_confirm(args, heldout, test11, val10, rng)
    for name in ("strength_selection.json", "decisions.json"):
        path = resolve_path(args.confirm_root) / name
        summary[name.split(".")[0]] = read_json(path) if path.is_file() else None
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    write_tables(summary, out)
    print(f"wrote {out}/summary.json, rescore_table.csv, confirm_runs.csv, confirm_tests.csv")
    for name, c in summary["confirm"].get("tests", {}).items():
        print(name, {k: c.get(k) for k in ("delta", "ci95_seed_x_image", "welch_p_one_sided", "holm_p", "supported")})


if __name__ == "__main__":
    main()
