#!/usr/bin/env python3
"""Train and test one Phase 3 method under one label track."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import evaluate_checkpoint
from src.data.coco_dataset import FloorplanCocoDataset, read_split
from src.data.transforms import DetectionTransform
from src.experiments.train import train_experiment
from src.utils.config import load_yaml
from src.utils.config import resolve_path, split_dir
from src.visualization.predictions import render_prediction_comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--track", required=True, choices=["multiclass", "agnostic"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root",
                        help="override experiment output_root (e.g. outputs/cv/fold3); "
                             "used by the CV array together with FP_SPLIT_DIR")
    parser.add_argument("--eval-checkpoint",
                        help="evaluate this best.pth on the (FP_SPLIT_DIR) split and exit; "
                             "no training. Used by the rotated held-out CV eval.")
    parser.add_argument("--eval-split", default="test", choices=["val", "test"])
    parser.add_argument("--eval-tag", default="eval",
                        help="subdir name under output_root/<track>/<id>/ for eval-only results")
    parser.add_argument("--replacement-percentage", type=float,
                        help="E14/E15 target percentage of common rooms changed per synthetic plan")
    parser.add_argument("--percentage-jitter", type=float,
                        help="E14/E15 uniform ± percentage-point randomness")
    args = parser.parse_args()
    experiment = load_yaml(args.config)
    if args.output_root:
        experiment["output_root"] = args.output_root

    if args.eval_checkpoint:
        checkpoint = Path(args.eval_checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"--eval-checkpoint does not exist: {checkpoint}")
        model_cfg = load_yaml(resolve_path(experiment.get("base_model_config", "configs/baseline.yaml")))
        dataset_cfg = load_yaml(resolve_path(
            "configs/dataset_agnostic.yaml" if args.track == "agnostic" else "configs/dataset.yaml"))
        rare_cfg = None if args.track == "agnostic" else load_yaml(ROOT / "configs/rare_classes.yaml")
        out_dir = resolve_path(experiment["output_root"]) / args.track / experiment["id"] / args.eval_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics, rows, _ = evaluate_checkpoint(
            dataset_cfg, model_cfg, rare_cfg, checkpoint, args.eval_split, out_dir / "metrics")
        result = {"mode": "eval_only", "eval_checkpoint": str(checkpoint),
                  "eval_split": args.eval_split, "test_metrics": metrics, "per_class": rows}
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"eval_checkpoint": str(checkpoint), "eval_split": args.eval_split,
                          "test_metrics": metrics}, ensure_ascii=False, indent=2))
        return

    if experiment["id"] in {"e14", "e15"}:
        policy = experiment["multi_room"]
        percentage = float(policy["replacement_percentage"] if args.replacement_percentage is None else args.replacement_percentage)
        jitter = float(policy["percentage_jitter"] if args.percentage_jitter is None else args.percentage_jitter)
        slug = lambda value: ("%g" % value).replace(".", "p")
        variant = f"pct_{slug(percentage)}_jitter_{slug(jitter)}"
        dataset_dir = ROOT / "data/augmented_lora_multi_semantic_v2" / variant
        manifest_path = dataset_dir / "multiroom_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Build the parameterized multi-room dataset first: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if abs(100 * manifest["replacement_fraction"] - percentage) > 1e-9 or abs(100 * manifest["fraction_jitter"] - jitter) > 1e-9:
            raise ValueError("Multi-room dataset policy does not match experiment parameters")
        experiment["multi_room"] = {"replacement_percentage": percentage, "percentage_jitter": jitter,
                                    "manifest": str(manifest_path), "source_candidate_count": manifest["source_candidate_count"]}
        experiment["dataset"]["source_overrides"] = {"strategy_lora_multi": {
            "annotation_file": str(dataset_dir / "instances_train_augmented.json"),
            "image_dir": str(dataset_dir), "sample_kind": "full_plan"}}
        experiment["output_root"] = str(ROOT / experiment["output_root"] / variant)
    elif args.replacement_percentage is not None or args.percentage_jitter is not None:
        raise ValueError("Multi-room percentage parameters are only valid for E14/E15")
    training, output_dir, dataset_cfg, model_cfg = train_experiment(
        experiment, args.track, args.seed, args.epochs, args.resume)
    checkpoint = output_dir / "checkpoints/best.pth"
    rare_cfg = None if args.track == "agnostic" else load_yaml(ROOT / "configs/rare_classes.yaml")
    metrics, rows, predictions = evaluate_checkpoint(
        dataset_cfg, model_cfg, rare_cfg, checkpoint, "test", output_dir / "metrics")
    test_ids = read_split(split_dir(dataset_cfg["split_dir"]) / "test.txt")
    test_dataset = FloorplanCocoDataset(resolve_path(dataset_cfg["image_dir"]),
                                        resolve_path(dataset_cfg["annotation_file"]), test_ids,
                                        DetectionTransform())
    render_prediction_comparisons(test_dataset, predictions, output_dir / "predictions",
                                  model_cfg["score_threshold"], 20)
    result = {"training": training, "test_metrics": metrics, "per_class": rows}
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
