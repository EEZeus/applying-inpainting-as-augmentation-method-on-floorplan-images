#!/usr/bin/env python3
"""Aggregate the rotated held-out CV eval.

Reads outputs/cv/fold{k}/<track>/<arm>/eval/result.json for k in 0..K-1, written
by `run_experiment.py --eval-checkpoint`. Across folds every held-out image is
tested once, so per-class test support pools to the full held-out set.

Per (track, arm) it reports:
  - overall mask AP: mean / std / min / max over the K folds (K disjoint eval sets)
  - per-class mask AP: support-weighted mean over folds  (weight = that fold's
    test_instances for the class), and pooled test_instances
  - rare-macro (11-16) and head-macro (1-10) from the pooled per-class table
  - paired comparison vs e0 over the K folds (Wilcoxon, exact sign-test fallback)

Writes outputs/final/cv/summary.{json,csv}.
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARMS = [
    ("e0", "outputs/experiments", "Original (no aug)"),
    ("e1", "outputs/experiments", "Rare-weighted sampling"),
    ("e6", "outputs/experiments", "Strategy B"),
    ("e8", "outputs/experiments_c_multi", "Strategy C copy-paste"),
    ("e10", "outputs/experiments_c_multi", "Strategy C diffusion inpaint"),
    ("e12", "outputs/experiments_lora", "LoRA Strategy B"),
]
BASELINE = "e0"
RARE_IDS = {11, 12, 13, 14, 15, 16}
HEAD_IDS = set(range(1, 11))


def _wilcoxon(arm_vals, base_vals):
    diffs = [a - b for a, b in zip(arm_vals, base_vals) if a is not None and b is not None]
    if not diffs or all(d == 0 for d in diffs):
        return {"test": "none", "n_pairs": len(diffs),
                "mean_delta": statistics.fmean(diffs) if diffs else None, "p_value": None}
    res = {"n_pairs": len(diffs), "mean_delta": statistics.fmean(diffs)}
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(diffs, alternative="two-sided", mode="auto")
        res.update({"test": "wilcoxon", "statistic": float(stat), "p_value": float(p)})
    except Exception:
        pos = sum(1 for d in diffs if d > 0)
        n = sum(1 for d in diffs if d != 0)
        tail = sum(math.comb(n, k) for k in range(0, min(pos, n - pos) + 1)) / (2 ** n)
        res.update({"test": "sign", "n_positive": pos, "n_nonzero": n, "p_value": min(1.0, 2 * tail)})
    return res


def _fold_path(root, track, arm, k):
    return ROOT / root / f"fold{k}" / track / arm / "eval" / "result.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tracks", nargs="+", default=["multiclass", "agnostic"])
    parser.add_argument("--out", default="outputs/final/cv")
    args = parser.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"folds": args.folds, "tracks": args.tracks, "mode": "rotated_heldout_eval", "arms": []}
    csv_rows = []

    for track in args.tracks:
        base_overall = {}
        for k in range(args.folds):
            p = _fold_path(dict((a, r) for a, r, _ in ARMS)[BASELINE], track, BASELINE, k)
            if p.is_file():
                base_overall[k] = json.loads(p.read_text())["test_metrics"]["mask"]["AP"]

        for arm, root, label in ARMS:
            folds = {}
            for k in range(args.folds):
                p = _fold_path(root, track, arm, k)
                if p.is_file():
                    folds[k] = json.loads(p.read_text())

            overall = [folds[k]["test_metrics"]["mask"]["AP"] for k in sorted(folds)]
            # pooled per-class: support-weighted mean AP over folds
            per_class = {}
            for k, payload in folds.items():
                for r in payload.get("per_class", []):
                    cid = int(r["category_id"])
                    ap = r.get("mask_AP")
                    n = r.get("test_instances") or 0
                    d = per_class.setdefault(cid, {"name": r.get("category_name"), "support": 0,
                                                   "wsum": 0.0, "wn": 0})
                    d["support"] += n
                    if ap is not None and n > 0:
                        d["wsum"] += ap * n
                        d["wn"] += n
            pooled = {cid: {"name": d["name"], "support": d["support"],
                            "mask_AP": (d["wsum"] / d["wn"]) if d["wn"] else None}
                      for cid, d in per_class.items()}

            def macro(ids):
                vals = [pooled[c]["mask_AP"] for c in ids if c in pooled and pooled[c]["mask_AP"] is not None]
                return statistics.fmean(vals) if vals else None

            entry = {
                "arm": arm, "track": track, "label": label,
                "folds_done": sorted(folds),
                "overall_mask_AP": {
                    "mean": statistics.fmean(overall) if overall else None,
                    "std": statistics.pstdev(overall) if len(overall) > 1 else 0.0,
                    "min": min(overall) if overall else None,
                    "max": max(overall) if overall else None,
                    "per_fold": overall,
                },
                "pooled_per_class": {str(c): pooled[c] for c in sorted(pooled)},
            }
            if track == "multiclass":
                entry["rare_macro_mask_AP"] = macro(RARE_IDS)
                entry["head_macro_mask_AP"] = macro(HEAD_IDS)
            if arm != BASELINE and base_overall:
                shared = [k for k in sorted(folds) if k in base_overall]
                entry["paired_vs_e0_overall"] = {
                    "folds": shared,
                    **_wilcoxon([folds[k]["test_metrics"]["mask"]["AP"] for k in shared],
                                [base_overall[k] for k in shared]),
                }
            report["arms"].append(entry)

            row = {"track": track, "arm": arm, "label": label,
                   "folds_done": ";".join(map(str, sorted(folds))),
                   "overall_AP_mean": entry["overall_mask_AP"]["mean"],
                   "overall_AP_std": entry["overall_mask_AP"]["std"]}
            if track == "multiclass":
                row["rare_macro_AP"] = entry.get("rare_macro_mask_AP")
                row["head_macro_AP"] = entry.get("head_macro_mask_AP")
            if "paired_vs_e0_overall" in entry:
                row["delta_vs_e0"] = entry["paired_vs_e0_overall"].get("mean_delta")
                row["p_vs_e0"] = entry["paired_vs_e0_overall"].get("p_value")
                row["test_vs_e0"] = entry["paired_vs_e0_overall"].get("test")
            csv_rows.append(row)

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fieldnames = sorted({k for r in csv_rows for k in r})
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    done = sum(len(a["folds_done"]) for a in report["arms"])
    total = len(report["arms"]) * args.folds
    print("cv analysis: %d/%d (arm x fold) evals found across %s" % (done, total, args.tracks))
    print(json.dumps(report, indent=2))
    if done < total:
        print("WARNING: %d evals missing" % (total - done), file=sys.stderr)


if __name__ == "__main__":
    main()
