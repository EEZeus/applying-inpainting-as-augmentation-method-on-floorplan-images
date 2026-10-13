import time
from collections import Counter

import torch
from tqdm import tqdm

from src.evaluation.coco_eval import output_to_coco
from src.models.text_head import is_text_aware, set_text_tokens


def move_targets(targets, device):
    return [{key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()} for target in targets]


def train_one_epoch(model, loader, optimizer, device, scaler, epoch):
    optimizers = list(optimizer) if isinstance(optimizer, (list, tuple)) else [optimizer]
    """Train once and return loss, budget, source mix, and class exposure.

    Experiment datasets may attach private ``_exposure`` and ``_sample_kind``
    fields.  They are removed before torchvision sees the target and are used
    only for Phase 3 diagnostics.  Baseline datasets remain fully compatible.
    """
    model.train()
    running, started = 0.0, time.time()
    exposure = Counter()
    samples_seen = optimizer_steps = full_plans_seen = rare_crops_seen = 0
    progress = tqdm(loader, desc=f"epoch {epoch}")
    for images, targets in progress:
        samples_seen += len(targets)
        for target in targets:
            kind = target.get("_sample_kind", "full_plan")
            full_plans_seen += kind == "full_plan"
            rare_crops_seen += kind == "rare_crop"
            for occurrence in target.get("_exposure", []):
                exposure[(int(occurrence["category_id"]), occurrence["source"])] += 1
        if is_text_aware(model):
            set_text_tokens(model, targets, device)
        targets = [
            {key: value for key, value in target.items() if not key.startswith("_")}
            for target in targets
        ]
        images = [image.to(device) for image in images]
        targets = move_targets(targets, device)
        for item in optimizers:
            item.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            losses = model(images, targets)
            loss = sum(losses.values())
        scaler.scale(loss).backward()
        for item in optimizers:
            scaler.step(item)
        scaler.update()
        optimizer_steps += 1
        running += loss.item()
        progress.set_postfix(loss=f"{loss.item():.3f}")
    return {"loss": running / max(optimizer_steps, 1), "seconds": time.time() - started,
            "optimizer_steps": optimizer_steps, "samples_seen": samples_seen,
            "full_plan_samples_seen": full_plans_seen, "rare_crop_samples_seen": rare_crops_seen,
            "class_exposure": exposure}


@torch.inference_mode()
def predict_coco(model, loader, device):
    """Run inference while retaining only compact COCO boxes and mask RLEs."""
    model.eval()
    bbox_results, mask_results = [], []
    for images, targets in tqdm(loader, desc="inference"):
        if is_text_aware(model):
            set_text_tokens(model, targets, device)
        outputs = model([image.to(device) for image in images])
        for target, output in zip(targets, outputs):
            boxes, masks = output_to_coco(int(target["image_id"]), output)
            bbox_results.extend(boxes)
            mask_results.extend(masks)
        del outputs
    return bbox_results, mask_results
