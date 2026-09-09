"""Fixed-seed side-by-side validation of base and floor-plan LoRA inpainting."""
import csv
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.augmentation.backends import StableDiffusionInpaintingBackend
from src.augmentation.pipeline import _atomic_json, file_digest


COMPARISON_FIELDS = ["case_id", "target_category", "source_annotation_id", "seed", "base_file", "lora_file",
                     "base_outside_change_ratio", "lora_outside_change_ratio", "lora_inside_changed_fraction",
                     "structural_qc_passed", "human_approval_used"]


def _panel(values, titles, width=512):
    canvas = Image.new("RGB", (width * len(values), width + 38), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (value, title) in enumerate(zip(values, titles)):
        image = value.convert("RGB").resize((width, width)); canvas.paste(image, (i * width, 38)); draw.text((i * width + 8, 10), title, fill="black")
    return canvas


def validate_generator(config, manifest_file, output_dir):
    root = Path(manifest_file).parent; output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads((root / "validation_cases.json").read_text(encoding="utf-8"))["cases"]
    generation = {"model_name": config["model"]["name"], "model_revision": config["model"].get("revision"),
                  "cache_dir": config["model"].get("cache_dir"), "device": "cuda", "dtype": config["training"]["mixed_precision"],
                  "num_inference_steps": config["validation"]["num_inference_steps"], "guidance_scale": config["validation"]["guidance_scale"],
                  "negative_prompt": config["prompt"].get("negative_prompt")}
    base = StableDiffusionInpaintingBackend(generation)
    rows, metadata = [], []
    for index, case in enumerate(cases):
        image = Image.open(root / case["image_file"]).convert("RGB"); mask = Image.open(root / case["mask_file"]).convert("L")
        seed = int(config["validation"]["seed"]) + index
        generated = base.generate(image, mask, case["prompt"], seed)
        safe = np.asarray(mask) > 0; source = np.asarray(image); arr = source.copy(); arr[safe] = np.asarray(generated)[safe]
        path = output_dir / "base" / f"{case['case_id']}.png"; path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(arr).save(path)
        case["seed"] = seed; case["base_file"] = str(path.relative_to(output_dir))
    del base; gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError: pass
    lora = StableDiffusionInpaintingBackend(generation | {"lora_path": str(Path(config["model"]["output_dir"]).resolve()),
                                                          "lora_weight_name": "pytorch_lora_weights.safetensors"})
    for case in cases:
        image = Image.open(root / case["image_file"]).convert("RGB"); mask = Image.open(root / case["mask_file"]).convert("L")
        generated = lora.generate(image, mask, case["prompt"], case["seed"])
        safe = np.asarray(mask) > 0; source = np.asarray(image); arr = source.copy(); arr[safe] = np.asarray(generated)[safe]
        path = output_dir / "lora" / f"{case['case_id']}.png"; path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(arr).save(path)
        base_path = output_dir / case["base_file"]
        outside = ~safe
        base_outside = float(np.any(np.asarray(Image.open(base_path)) != source, axis=2)[outside].mean()) if outside.any() else 0.0
        lora_outside = float(np.any(arr != source, axis=2)[outside].mean()) if outside.any() else 0.0
        inside_changed = float(np.any(arr != source, axis=2)[safe].mean()) if safe.any() else 0.0
        contact = output_dir / "contact_sheets" / f"{case['case_id']}.jpg"; contact.parent.mkdir(parents=True, exist_ok=True)
        panel = _panel([image, mask.convert("RGB"), Image.open(base_path), Image.fromarray(arr)], ["source", "safe mask", "pretrained SD", "floor-plan LoRA"])
        panel.save(contact, quality=92)
        model_example = Path(config["model"]["output_dir"]) / "example_generations" / contact.name
        model_example.parent.mkdir(parents=True, exist_ok=True)
        panel.save(model_example, quality=92)
        record = {**case, "lora_file": str(path.relative_to(output_dir)), "contact_sheet": str(contact.relative_to(output_dir)),
                  "base_outside_change_ratio": base_outside, "lora_outside_change_ratio": lora_outside,
                  "lora_inside_changed_fraction": inside_changed,
                  "lora_sha256": lora.metadata["lora_sha256"]}
        metadata.append(record)
        rows.append({"case_id": case["case_id"], "target_category": case["target_category_name"], "source_annotation_id": case["source_annotation_id"],
                     "seed": case["seed"], "base_file": case["base_file"], "lora_file": record["lora_file"],
                     "base_outside_change_ratio": base_outside, "lora_outside_change_ratio": lora_outside,
                     "lora_inside_changed_fraction": inside_changed, "structural_qc_passed": lora_outside == 0,
                     "human_approval_used": False})
    with (output_dir / "automatic_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS); writer.writeheader(); writer.writerows(rows)
    _atomic_json(output_dir / "generation_metadata.json", metadata)
    summary = {"cases": len(cases), "base_model": config["model"]["name"], "lora_sha256": lora.metadata["lora_sha256"],
               "fixed_seeds": True, "outside_change_ratio_max": max((r["lora_outside_change_ratio"] for r in metadata), default=0),
               "structural_qc_passed": all(r["lora_outside_change_ratio"] == 0 for r in metadata),
               "human_approval_used": False}
    _atomic_json(output_dir / "summary.json", summary); return summary
