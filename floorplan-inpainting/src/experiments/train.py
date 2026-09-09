"""Fair-budget training for Phase 3 multiclass and class-agnostic tracks."""
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.coco_dataset import FloorplanCocoDataset, read_split
from src.data.transforms import DetectionTransform, collate_fn
from src.evaluation.coco_eval import evaluate_coco_results
from src.experiments.data import (build_epoch_sampler, build_training_dataset,
                                  original_train_counts)
from src.models.mask_rcnn import build_mask_rcnn
from src.training.engine import predict_coco, train_one_epoch
from src.utils.config import load_yaml, resolve_path
from src.utils.reproducibility import seed_everything


def _stable_hash(experiment_cfg, model_cfg, track, seed):
    payload = json.dumps({"experiment": experiment_cfg, "model": model_cfg,
                          "track": track, "seed": seed}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_experiment(experiment_cfg):
    expected = {
        "e0": (["original"], "uniform", False),
        "e1": (["original"], "rare_weighted", False),
        "e2": (["strategy_a"], "uniform", False),
        "e3": (["strategy_a"], "rare_weighted", False),
        "e4": (["strategy_a", "strategy_a_crops"], "uniform", True),
        "e5": (["strategy_a", "strategy_a_crops"], "rare_weighted", True),
        "e6": (["strategy_b"], "uniform", False),
        "e7": (["strategy_b"], "rare_weighted", False),
        "e8": (["strategy_c_copy_paste_multi"], "uniform", False),
        "e9": (["strategy_c_feather_blend"], "uniform", False),
        "e10": (["strategy_c_diffusion_inpaint"], "uniform", False),
        "e11": (["strategy_c_diffusion_inpaint"], "rare_weighted", False),
        "e12": (["strategy_lora_b"], "uniform", False),
        "e13": (["strategy_lora_b"], "rare_weighted", False),
        "e14": (["strategy_lora_multi"], "uniform", False),
        "e15": (["strategy_lora_multi"], "rare_weighted", False),
    }
    experiment_id = experiment_cfg["id"]
    if experiment_id not in expected:
        raise ValueError(f"Unknown required experiment ID: {experiment_id}")
    sources, mode, crops = expected[experiment_id]
    if experiment_cfg["dataset"]["sources"] != sources or experiment_cfg["sampling"]["mode"] != mode:
        raise ValueError(f"{experiment_id} configuration does not match the fixed Phase 3 matrix")
    if crops != ("strategy_a_crops" in sources):
        raise ValueError("Crop configuration mismatch")
    if int(experiment_cfg["budget"]["steps_per_epoch"]) <= 0:
        raise ValueError("steps_per_epoch must be positive")


def _append_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def train_experiment(experiment_cfg, track="multiclass", seed=None, epochs=None, resume=False):
    _validate_experiment(experiment_cfg)
    if track not in {"multiclass", "agnostic"}:
        raise ValueError("track must be multiclass or agnostic")
    model_cfg = load_yaml(resolve_path(experiment_cfg.get("base_model_config", "configs/baseline.yaml")))
    seed = int(model_cfg["seed"] if seed is None else seed)
    model_cfg["seed"] = seed
    epoch_budget = int(epochs or model_cfg["epochs"])
    steps_per_epoch = int(experiment_cfg["budget"]["steps_per_epoch"])
    samples_per_epoch = steps_per_epoch * int(model_cfg["batch_size"])
    output_dir = resolve_path(experiment_cfg["output_root"]) / track / experiment_cfg["id"] / f"seed_{seed}"
    checkpoint_dir, log_dir = output_dir / "checkpoints", output_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    signature = _stable_hash(experiment_cfg, model_cfg, track, seed)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("signature") != signature:
            raise ValueError("Existing experiment output was created with a different configuration")
        if not resume:
            raise FileExistsError(f"Experiment output already exists; pass --resume: {output_dir}")
    else:
        manifest_path.write_text(json.dumps({"signature": signature, "experiment": experiment_cfg,
                                             "model": model_cfg, "track": track, "seed": seed}, indent=2),
                                 encoding="utf-8")

    seed_everything(seed)
    device = torch.device(model_cfg["device"] if torch.cuda.is_available() else "cpu")
    class_agnostic = track == "agnostic"
    train_dataset, records = build_training_dataset(experiment_cfg, model_cfg, class_agnostic)
    original_counts = original_train_counts()
    rare_cfg = load_yaml(resolve_path("configs/rare_classes.yaml"))
    rare_ids = {int(cid) for cid, value in rare_cfg["categories"].items()
                if value["role"] == "primary_rare"}

    evaluation_dataset_cfg = load_yaml(resolve_path(
        "configs/dataset_agnostic.yaml" if class_agnostic else "configs/dataset.yaml"))
    val_ids = read_split(resolve_path(evaluation_dataset_cfg["split_dir"]) / "val.txt")
    val_dataset = FloorplanCocoDataset(resolve_path(evaluation_dataset_cfg["image_dir"]),
                                       resolve_path(evaluation_dataset_cfg["annotation_file"]), val_ids,
                                       DetectionTransform())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=model_cfg["num_workers"], collate_fn=collate_fn)
    num_classes = 2 if class_agnostic else 20
    last_path = checkpoint_dir / "last.pth"
    restoring = resume and last_path.exists()
    model = build_mask_rcnn(num_classes, model_cfg["min_size"], model_cfg["max_size"],
                            pretrained=not restoring).to(device)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if model_cfg["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(params, lr=model_cfg["learning_rate"],
                                      weight_decay=model_cfg["weight_decay"])
    else:
        optimizer = torch.optim.SGD(params, lr=model_cfg["learning_rate"],
                                    momentum=model_cfg["momentum"], weight_decay=model_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, model_cfg["lr_step_size"],
                                                gamma=model_cfg["lr_gamma"])
    scaler = torch.cuda.amp.GradScaler(enabled=model_cfg["amp"] and device.type == "cuda")
    start_epoch, best_ap, total_steps, total_samples, prior_runtime = 1, -1.0, 0, 0, 0.0
    if restoring:
        state = torch.load(last_path, map_location="cpu")
        if state.get("signature") != signature or state.get("num_classes") != num_classes:
            raise ValueError("Checkpoint provenance does not match this run")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_ap, total_steps = float(state["best_val_mask_AP"]), int(state["total_optimizer_steps"])
        total_samples, prior_runtime = int(state["samples_seen"]), float(state["runtime_seconds"])
    writer = SummaryWriter(log_dir)
    category_names = {int(cid): value["name"] for cid, value in rare_cfg["categories"].items()}
    original = json.loads(resolve_path("data/processed/instances_deduplicated.json").read_text())
    category_names.update({c["id"]: c["name"] for c in original["categories"]})
    exposure_path = output_dir / "class_exposure_by_epoch.csv"
    history_path = output_dir / "training_history.csv"
    started = time.time()
    for epoch in range(start_epoch, epoch_budget + 1):
        sampler = build_epoch_sampler(records, experiment_cfg, seed + epoch, samples_per_epoch,
                                      original_counts, rare_ids)
        loader = DataLoader(train_dataset, batch_size=model_cfg["batch_size"], sampler=sampler,
                            num_workers=model_cfg["num_workers"], collate_fn=collate_fn,
                            pin_memory=device.type == "cuda", drop_last=True)
        report = train_one_epoch(model, loader, optimizer, device, scaler, epoch)
        if report["optimizer_steps"] != steps_per_epoch:
            raise RuntimeError("Fixed optimizer-step budget was not respected")
        scheduler.step()
        bbox_results, mask_results = predict_coco(model, val_loader, device)
        metrics, _, _ = evaluate_coco_results(resolve_path(evaluation_dataset_cfg["annotation_file"]),
                                               bbox_results, mask_results, val_ids)
        mask_ap = metrics["mask"]["AP"]
        total_steps += report["optimizer_steps"]; total_samples += report["samples_seen"]
        writer.add_scalar("train/loss", report["loss"], epoch)
        writer.add_scalar("val/mask_AP", mask_ap, epoch)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        exposure_rows = []
        for category_id in sorted(category_names):
            real_seen = report["class_exposure"].get((category_id, "real"), 0)
            a_seen = report["class_exposure"].get((category_id, "synthetic_A"), 0)
            b_seen = report["class_exposure"].get((category_id, "synthetic_B"), 0)
            c_seen = report["class_exposure"].get((category_id, "synthetic_C"), 0)
            c_multi_seen = report["class_exposure"].get((category_id, "synthetic_C_MULTI"), 0)
            lora_seen = report["class_exposure"].get((category_id, "synthetic_LORA"), 0)
            lora_multi_seen = report["class_exposure"].get((category_id, "synthetic_LORA_MULTI"), 0)
            exposure_rows.append({"epoch": epoch, "category_id": category_id,
                                  "category_name": category_names[category_id],
                                  "real_occurrences_seen": real_seen,
                                  "synthetic_A_occurrences_seen": a_seen,
                                  "synthetic_B_occurrences_seen": b_seen,
                                  "synthetic_C_occurrences_seen": c_seen,
                                  "synthetic_C_MULTI_occurrences_seen": c_multi_seen,
                                  "synthetic_LORA_occurrences_seen": lora_seen,
                                  "synthetic_LORA_MULTI_occurrences_seen": lora_multi_seen,
                                  "total_occurrences_seen": real_seen + a_seen + b_seen + c_seen + c_multi_seen + lora_seen + lora_multi_seen})
        _append_rows(exposure_path, exposure_rows, list(exposure_rows[0]))
        history = {"epoch": epoch, "training_loss": report["loss"], "validation_mask_AP": mask_ap,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "optimizer_steps": report["optimizer_steps"], "samples_seen": report["samples_seen"],
                   "full_plan_samples_seen": report["full_plan_samples_seen"],
                   "rare_crop_samples_seen": report["rare_crop_samples_seen"],
                   "epoch_seconds": report["seconds"]}
        _append_rows(history_path, [history], list(history))
        elapsed = prior_runtime + time.time() - started
        if mask_ap > best_ap:
            best_ap = mask_ap
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch, "val_mask_AP": mask_ap,
                 "best_val_mask_AP": best_ap, "runtime_seconds": elapsed,
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                 "num_classes": num_classes, "class_agnostic": class_agnostic,
                 "total_optimizer_steps": total_steps, "samples_seen": total_samples,
                 "signature": signature, "experiment_id": experiment_cfg["id"]}
        torch.save(state, last_path)
        if mask_ap >= best_ap:
            torch.save(state, checkpoint_dir / "best.pth")
    writer.close()
    result = {"experiment_id": experiment_cfg["id"], "track": track, "seed": seed,
              "best_val_mask_AP": best_ap, "total_optimizer_steps": total_steps,
              "samples_seen": total_samples, "steps_per_epoch": steps_per_epoch,
              "epochs": epoch_budget, "runtime_seconds": prior_runtime + time.time() - started,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
    (output_dir / "training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, output_dir, evaluation_dataset_cfg, model_cfg
