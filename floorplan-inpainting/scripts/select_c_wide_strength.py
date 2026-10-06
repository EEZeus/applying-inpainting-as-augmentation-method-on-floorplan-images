#!/usr/bin/env python3
"""Pick the C2-wide diffusion strength by a pre-registered, generation-only rule.

For each piloted strength the rule looks only at generation outcomes, never at
any detector metric:
  acceptance rate (accepted / attempted)              >= --min-acceptance
  mean donor dark-stroke retention (accepted samples) >= --min-retention
  median diffusion-mask area in SD latent cells        >= --min-latent-cells
The highest strength meeting all three is chosen. If none does, no strength is
chosen and the inpainting arms are dropped by the gate.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation.pipeline import latest_metadata, read_metadata
from src.utils.config import resolve_path


def strength_tag(value):
    """0.5 -> s050, 0.75 -> s075 (directory name of one pilot run)."""
    return f"s{int(round(float(value) * 100)):03d}"


def pilot_stats(metadata_path):
    records = list(latest_metadata(read_metadata(metadata_path)).values())
    accepted = [r for r in records if r["status"] == "accepted"]
    retention = [float(r["donor_dark_retention_fraction"]) for r in accepted
                 if r.get("donor_dark_retention_fraction") is not None]
    cells = [int(r["model_mask_pixels"]) / 64.0 for r in records if r.get("model_mask_pixels") is not None]
    reasons = {}
    for r in records:
        if r["status"] != "accepted":
            reasons[r.get("rejection_reason") or r["status"]] = reasons.get(r.get("rejection_reason") or r["status"], 0) + 1
    return {"attempted": len(records), "accepted": len(accepted),
            "acceptance_rate": len(accepted) / len(records) if records else 0.0,
            "mean_donor_retention": statistics.fmean(retention) if retention else 0.0,
            "median_latent_cells": statistics.median(cells) if cells else 0.0,
            "median_seam_original_pixels": statistics.median(
                [r["seam_original_pixels"] for r in records if "seam_original_pixels" in r] or [0]),
            "rejection_reasons": reasons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", default="data/augmented_c_wide")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.50, 0.75])
    parser.add_argument("--min-acceptance", type=float, default=0.80)
    parser.add_argument("--min-retention", type=float, default=0.95)
    parser.add_argument("--min-latent-cells", type=float, default=30.0)
    parser.add_argument("--force-strength", type=float,
                        help="smoke tests only: choose this strength regardless of the rule")
    parser.add_argument("--out", default="outputs/confirm/strength_selection.json")
    args = parser.parse_args()
    root = resolve_path(args.pilot_root)
    table, chosen = {}, None
    for strength in sorted(args.strengths):
        path = root / strength_tag(strength) / "diffusion_inpaint" / "metadata.jsonl"
        if not path.is_file():
            table[str(strength)] = {"missing": str(path), "passes": False}
            continue
        stats = pilot_stats(path)
        stats["passes"] = (stats["acceptance_rate"] >= args.min_acceptance
                           and stats["mean_donor_retention"] >= args.min_retention
                           and stats["median_latent_cells"] >= args.min_latent_cells)
        table[str(strength)] = stats
        if stats["passes"]:
            chosen = strength
    forced = args.force_strength is not None
    if forced:
        chosen = float(args.force_strength)
    result = {"rule": {"min_acceptance": args.min_acceptance, "min_retention": args.min_retention,
                       "min_latent_cells": args.min_latent_cells, "pick": "highest passing strength"},
              "pilot": table, "chosen_strength": chosen, "forced": forced,
              "chosen_dir": str(root / strength_tag(chosen)) if chosen is not None else None}
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
