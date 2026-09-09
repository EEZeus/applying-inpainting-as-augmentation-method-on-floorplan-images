import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.coco_dataset import FloorplanCocoDataset, read_split
from src.data.transforms import DetectionTransform, collate_fn
from src.evaluation.coco_eval import evaluate_coco_results
from src.models.mask_rcnn import build_mask_rcnn
from src.training.engine import predict_coco, train_one_epoch
from src.utils.config import resolve_path
from src.utils.reproducibility import seed_everything


def train(dataset_cfg, cfg, epochs=None):
    seed_everything(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    annotation_file, image_dir = resolve_path(dataset_cfg["annotation_file"]), resolve_path(dataset_cfg["image_dir"])
    split_dir = resolve_path(dataset_cfg["split_dir"])
    train_tf = DetectionTransform(True, cfg["horizontal_flip_probability"], cfg["brightness"], cfg["contrast"])
    train_ds = FloorplanCocoDataset(image_dir, annotation_file, read_split(split_dir / "train.txt"), train_tf)
    val_ids = read_split(split_dir / "val.txt")
    val_ds = FloorplanCocoDataset(image_dir, annotation_file, val_ids, DetectionTransform())
    category_ids = sorted(category["id"] for category in train_ds.coco_data["categories"])
    if category_ids != dataset_cfg["category_ids"] or dataset_cfg["num_classes"] != len(category_ids) + 1:
        raise ValueError("Dataset category IDs and configured num_classes disagree")
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                              collate_fn=collate_fn, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate_fn)
    model = build_mask_rcnn(dataset_cfg["num_classes"], cfg["min_size"], cfg["max_size"], pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.SGD(params, lr=cfg["learning_rate"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg["lr_step_size"], gamma=cfg["lr_gamma"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")
    checkpoint_dir, log_dir = resolve_path(cfg["checkpoint_dir"]), resolve_path(cfg["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    writer, best_ap, total_start = SummaryWriter(log_dir), -1.0, time.time()
    for epoch in range(1, (epochs or cfg["epochs"]) + 1):
        report = train_one_epoch(model, train_loader, optimizer, device, scaler, epoch)
        scheduler.step()
        bbox_results, mask_results = predict_coco(model, val_loader, device)
        metrics, _, _ = evaluate_coco_results(annotation_file, bbox_results, mask_results, val_ids)
        mask_ap = metrics["mask"]["AP"]
        writer.add_scalar("train/loss", report["loss"], epoch)
        writer.add_scalar("val/mask_AP", mask_ap, epoch)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "val_mask_AP": mask_ap, "runtime_seconds": time.time() - total_start,
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                 "num_classes": dataset_cfg["num_classes"],
                 "class_agnostic": bool(dataset_cfg.get("class_agnostic", False)),
                 "annotation_file": str(annotation_file)}
        torch.save(state, checkpoint_dir / "last.pth")
        if mask_ap > best_ap:
            best_ap = mask_ap; torch.save(state, checkpoint_dir / "best.pth")
    writer.close()
    return {"best_val_mask_AP": best_ap, "runtime_seconds": time.time() - total_start,
            "device": str(device), "class_agnostic": bool(dataset_cfg.get("class_agnostic", False))}
