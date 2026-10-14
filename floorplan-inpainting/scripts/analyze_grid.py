#!/usr/bin/env python3
"""Aggregate the Round-1 grid: per-arm mean/std over seeds and a paired test vs e0.

Reads result.json written by scripts/run_experiment.py for every
(config id, track, seed) that finished. Writes:
  outputs/final/grid_r1/summary.json   machine-readable
  outputs/final/grid_r1/summary.csv    one row per (track, arm)

Image-level bootstrap CIs are produced by a later stage (scripts/eval_bootstrap.py),
which re-loads checkpoints; this script only needs the finished result.json files.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEEDS = (42, 123, 2026)

# arm id -> (output_root, human label). Order is the report order.
ARMS = [
    ("e0", "outputs/experiments", "Original (no aug)"),
    ("e1", "outputs/experiments", "Rare-weighted sampling"),
    ("e2", "outputs/experiments", "Strategy A"),
    ("e3", "outputs/experiments", "Strategy A + weighting"),
    ("e4", "outputs/experiments", "Strategy A + crops"),
    ("e5", "outputs/experiments", "Strategy A + crops + weighting"),
    ("e6", "outputs/experiments", "Strategy B"),
    ("e7", "outputs/experiments", "Strategy B + weighting"),
    ("e8", "outputs/experiments_c_multi", "Strategy C copy-paste"),
    ("e9", "outputs/experiments_c_multi", "Strategy C feather-blend"),
    ("e10", "outputs/experiments_c_multi", "Strategy C diffusion inpaint"),
    ("e11", "outputs/experiments_c_multi", "Strategy C diffusion inpaint + weighting"),
    ("e12", "outputs/experiments_lora", "LoRA Strategy B"),
    ("e13", "outputs/experiments_lora", "LoRA Strategy B + weighting"),
]

BASELINE = "e0"


def _result_path(output_root, track, arm, seed):
    return ROOT / output_root / track / arm / f"seed_{seed}" / "result.json"


def _metrics(payload, track):
    m = payload["test_metrics"]
    out = {
        "overall_mask_AP": m["mask"]["AP"],
        "overall_bbox_AP": m["bbox"]["AP"],
    }
    if track == "multiclass":
        out["rare_mask_AP"] = m.get("rare_mask_AP")
    return out


def _agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "values": []}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "values": vals,
    }


def _wilcoxon(paired_arm, paired_base):
    """Paired Wilcoxon signed-rank vs baseline. Falls back to a sign test."""
    diffs = [a - b for a, b in zip(paired_arm, paired_base) if a is not None and b is not None]
    if len(diffs) < 1 or all(d == 0 for d in diffs):
        return {
            "test": "none",
            "n_pairs": len(diffs),
            "p_value": None,
            "mean_delta": statistics.fmean(diffs) if diffs else None,
        }
    result = {"n_pairs": len(diffs), "mean_delta": statistics.fmean(diffs)}
    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", mode="auto")
        result.update({"test": "wilcoxon", "statistic": float(stat), "p_value": float(p)})
    except Exception:
        pos = sum(1 for d in diffs if d > 0)
        n = sum(1 for d in diffs if d != 0)
        # two-sided exact sign test
        from math import comb

        tail = sum(comb(n, k) for k in range(0, min(pos, n - pos) + 1)) / (2**n)
        result.update({"test": "sign", "n_positive": pos, "p_value": min(1.0, 2 * tail)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", nargs="+", default=["multiclass", "agnostic"])
    parser.add_argument("--out", default="outputs/final/grid_r1")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"seeds": list(SEEDS), "tracks": args.tracks, "arms": []}
    csv_rows = []

    for track in args.tracks:
        # collect baseline per-seed metrics for paired tests
        base_by_seed = {}
        base_root = dict((a, r) for a, r, _ in ARMS)[BASELINE]
        for seed in SEEDS:
            p = _result_path(base_root, track, BASELINE, seed)
            if p.is_file():
                base_by_seed[seed] = _metrics(json.loads(p.read_text()), track)

        for arm, output_root, label in ARMS:
            per_seed = {}
            for seed in SEEDS:
                p = _result_path(output_root, track, arm, seed)
                if p.is_file():
                    per_seed[seed] = _metrics(json.loads(p.read_text()), track)

            keys = ["overall_mask_AP", "overall_bbox_AP"] + (
                ["rare_mask_AP"] if track == "multiclass" else []
            )
            agg = {k: _agg([per_seed.get(s, {}).get(k) for s in SEEDS]) for k in keys}

            entry = {
                "arm": arm,
                "track": track,
                "label": label,
                "seeds_done": sorted(per_seed),
                "missing_seeds": [s for s in SEEDS if s not in per_seed],
                "aggregate": agg,
            }
            if arm != BASELINE and base_by_seed:
                primary = "rare_mask_AP" if track == "multiclass" else "overall_mask_AP"
                shared = [s for s in SEEDS if s in per_seed and s in base_by_seed]
                entry["paired_vs_e0"] = {
                    "metric": primary,
                    "seeds": shared,
                    **_wilcoxon(
                        [per_seed[s][primary] for s in shared], [base_by_seed[s][primary] for s in shared]
                    ),
                }
            report["arms"].append(entry)

            row = {
                "track": track,
                "arm": arm,
                "label": label,
                "seeds_done": ";".join(map(str, sorted(per_seed))),
            }
            for k in keys:
                row[f"{k}_mean"] = agg[k]["mean"]
                row[f"{k}_std"] = agg[k]["std"]
            if "paired_vs_e0" in entry:
                row["delta_vs_e0"] = entry["paired_vs_e0"].get("mean_delta")
                row["p_vs_e0"] = entry["paired_vs_e0"].get("p_value")
                row["test_vs_e0"] = entry["paired_vs_e0"].get("test")
            csv_rows.append(row)

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fieldnames = sorted({k for r in csv_rows for k in r})
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    done = sum(len(a["seeds_done"]) for a in report["arms"])
    total = len(report["arms"]) * len(SEEDS)
    print(f"grid_r1 analysis: {done}/{total} (arm x seed) results found across {args.tracks}")
    print(json.dumps(report, indent=2))
    if done < total:
        print(f"WARNING: {total - done} runs still missing", file=sys.stderr)


if __name__ == "__main__":
    main()
