#!/usr/bin/env python3
"""Build the matched e16 (copy-paste) and e17 (wide-seam inpainting) datasets.

Both are composed by the existing multi-room builder from the intersection of
(target, donor) pairs accepted in BOTH candidate directories, with the same
config seed, the same replacement policy and a canonical pool order. The two
datasets therefore contain the same synthetic plans, converting the same rooms
with the same donors; only the integrated pixels differ (checked before writing).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation.inputs import load_inputs
from src.augmentation.lora_multiroom import build_multiroom_dataset, pair_key
from src.augmentation.pipeline import latest_metadata, read_metadata
from src.utils.config import load_yaml, resolve_path


def accepted_pairs(candidate_dir):
    records = latest_metadata(read_metadata(Path(candidate_dir) / "metadata.jsonl")).values()
    return {pair_key(r) for r in records if r["status"] == "accepted"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/augmentation_c.yaml")
    parser.add_argument("--copy-paste-dir", default="data/augmented_c/copy_paste")
    parser.add_argument("--selection", default="outputs/confirm/strength_selection.json")
    parser.add_argument("--output-root", default="data/augmented_c_matched")
    parser.add_argument("--min-pairs", type=int, default=30)
    args = parser.parse_args()
    out_root = resolve_path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "matched_manifest.json"
    selection = json.loads(resolve_path(args.selection).read_text())
    if selection.get("chosen_strength") is None:
        manifest = {"status": "no_strength_passed", "pairs": []}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2))
        return
    wide_dir = Path(selection["chosen_dir"]) / "diffusion_inpaint"
    copy_dir = resolve_path(args.copy_paste_dir)
    copy_pairs, wide_pairs = accepted_pairs(copy_dir), accepted_pairs(wide_dir)
    pairs = sorted(copy_pairs & wide_pairs)
    manifest = {
        "copy_paste_dir": str(copy_dir),
        "inpaint_wide_dir": str(wide_dir),
        "chosen_strength": selection["chosen_strength"],
        "accepted_copy_paste": len(copy_pairs),
        "accepted_inpaint_wide": len(wide_pairs),
        "pairs": [list(p) for p in pairs],
        "min_pairs": args.min_pairs,
    }
    if len(pairs) < args.min_pairs:
        manifest["status"] = "insufficient_pairs"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2))
        return
    config = load_yaml(args.config)
    coco, _, train_ids, image_dir = load_inputs(config)
    built = {}
    for name, candidate_dir, prefix in (
        ("copy_paste_multi", copy_dir, "c-matched-copy-paste"),
        ("inpaint_wide_multi", wide_dir, "c-matched-inpaint-wide"),
    ):
        built[name] = build_multiroom_dataset(
            config,
            coco,
            train_ids,
            image_dir,
            candidate_dir,
            out_root / name,
            strategy="c_multi",
            sample_prefix=prefix,
            allowed_pairs=pairs,
            canonical_order=True,
        )

    def signature(m):
        return [(g["source_image_id"], g["pair_keys"], g["replacement_count"]) for g in m["groups"]]

    if signature(built["copy_paste_multi"]) != signature(built["inpaint_wide_multi"]):
        raise ValueError("Matched datasets differ in composition; they must differ only in pixels")
    manifest.update(
        status="built",
        synthetic_plans=built["copy_paste_multi"]["synthetic_plans"],
        converted_rooms=sum(g["replacement_count"] for g in built["copy_paste_multi"]["groups"]),
        plans_with_variants=len({g["source_image_id"] for g in built["copy_paste_multi"]["groups"]}),
        replacement_fraction=built["copy_paste_multi"]["replacement_fraction"],
        fraction_jitter=built["copy_paste_multi"]["fraction_jitter"],
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
