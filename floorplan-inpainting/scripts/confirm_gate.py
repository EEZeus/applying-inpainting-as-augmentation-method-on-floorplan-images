#!/usr/bin/env python3
"""Pre-registered gates that decide which confirmatory arms run.

Gates can only cancel work; they never add arms or change the analysis.
  e8  (exact copy-paste replication): dropped if the leakage-free re-score of
      the existing 3 grid seeds gives mean(e8 - e0) <= 0 on class-agnostic mask
      AP of last.pth over the 21 held-out plans (futility). If the re-score is
      incomplete the gate cannot be evaluated and e8 is kept.
  e16/e17 (matched copy-paste / wide-seam inpainting): run only if a strength
      passed the generation-only rule and >= min_pairs matched pairs exist.
  e0: runs if any other arm runs.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import resolve_path

GRID_SEEDS = (42, 123, 2026)


def rescore_ap(rescore_root, arm, seed, track="agnostic", tag="heldout21_last"):
    path = rescore_root / track / arm / f"seed_{seed}" / tag / "result.json"
    if not path.is_file():
        return None
    return float(json.loads(path.read_text())["test_metrics"]["mask"]["AP"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore-root", default="outputs/rescore")
    parser.add_argument("--matched-root", default="data/augmented_c_matched")
    parser.add_argument("--out", default="outputs/confirm/decisions.json")
    parser.add_argument("--force-all", action="store_true", help="smoke tests only")
    args = parser.parse_args()
    rescore_root = resolve_path(args.rescore_root)
    decisions, notes = {}, {}

    e0 = [rescore_ap(rescore_root, "e0", s) for s in GRID_SEEDS]
    e8 = [rescore_ap(rescore_root, "e8", s) for s in GRID_SEEDS]
    if None in e0 or None in e8:
        decisions["e8"] = True
        notes["e8"] = "futility gate not evaluable (re-score incomplete); arm kept"
        futility = None
    else:
        futility = statistics.fmean(b - a for a, b in zip(e0, e8))
        decisions["e8"] = futility > 0
        notes["e8"] = f"mean(e8 - e0) over grid seeds on heldout21 last.pth = {futility:+.4f}"

    manifest_path = resolve_path(args.matched_root) / "matched_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"status": "missing"}
    matched_ok = manifest.get("status") == "built"
    decisions["e16"] = decisions["e17"] = matched_ok
    notes["e16"] = notes["e17"] = (
        f"matched datasets {manifest.get('status')}; "
        f"pairs={len(manifest.get('pairs', []))}, "
        f"strength={manifest.get('chosen_strength')}"
    )
    decisions["e0"] = any(decisions.values())
    notes["e0"] = (
        "baseline runs because at least one comparison arm runs"
        if decisions["e0"]
        else "no comparison arm left"
    )
    if args.force_all:
        decisions = {arm: True for arm in ("e0", "e8", "e16", "e17")}
        notes["forced"] = "smoke test: all arms forced on"
    result = {
        "run": decisions,
        "notes": notes,
        "futility_mean_delta": futility,
        "rescore_e0": e0,
        "rescore_e8": e8,
        "matched_status": manifest.get("status"),
    }
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
