"""Minimal Diffusers/PEFT trainer for a single class-conditioned SD-inpainting LoRA."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.augmentation.pipeline import _atomic_json
from src.lora_inpainting.data import LoraCropDataset


def make_inpainting_unet_input(noisy_latents, mask, masked_latents):
    mask = F.interpolate(mask, size=noisy_latents.shape[-2:], mode="nearest")
    result = torch.cat([noisy_latents, mask, masked_latents], dim=1)
    if result.shape[1] != 9:
        raise ValueError(f"SD inpainting UNet requires 9 channels, got {result.shape[1]}")
    return result


def _save_lora(unet, output_dir):
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionInpaintPipeline.save_lora_weights(
        output_dir, unet_lora_layers=state, safe_serialization=True
    )


def train_lora(config, manifest_file, output_dir):
    from accelerate import Accelerator
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    cfg, model_cfg = config["training"], config["model"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        mixed_precision=cfg["mixed_precision"],
    )
    common = {
        "pretrained_model_name_or_path": model_cfg["name"],
        "revision": model_cfg.get("revision"),
        "cache_dir": model_cfg.get("cache_dir"),
        "local_files_only": True,
    }
    tokenizer = CLIPTokenizer.from_pretrained(**common, subfolder="tokenizer")
    model_common = {**common, "variant": "fp16", "use_safetensors": True}
    text_encoder = CLIPTextModel.from_pretrained(**model_common, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(**model_common, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(**model_common, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(**common, subfolder="scheduler")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=int(cfg["rank"]),
            lora_alpha=int(cfg["rank"]),
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    if accelerator.mixed_precision == "fp16":
        unet.to(dtype=torch.float32)  # trainable LoRA parameters must remain fp32
    manifest_payload = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    dataset = LoraCropDataset(manifest_file, tokenizer)
    primary = set(map(int, config["rare_augmentation"]["primary_ids"]))
    optional = set(map(int, config["rare_augmentation"]["optional_ids"]))
    weights = [
        float(
            cfg["rare_sampling_multiplier"]
            if r["category_id"] in primary
            else cfg["optional_sampling_multiplier"]
            if r["category_id"] in optional
            else 1.0
        )
        for r in dataset.records
    ]
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train_batch_size"]),
        sampler=sampler,
        num_workers=int(cfg["dataloader_workers"]),
        pin_memory=True,
    )
    params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg["learning_rate"]))
    updates_per_epoch = math.ceil(len(loader) / int(cfg["gradient_accumulation_steps"]))
    epochs = math.ceil(int(cfg["max_train_steps"]) / updates_per_epoch)
    lr_scheduler = get_scheduler(
        cfg["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=int(cfg["lr_warmup_steps"]),
        num_training_steps=int(cfg["max_train_steps"]),
    )
    unet, optimizer, loader, lr_scheduler = accelerator.prepare(unet, optimizer, loader, lr_scheduler)
    dtype = (
        torch.float16
        if accelerator.mixed_precision == "fp16"
        else torch.bfloat16
        if accelerator.mixed_precision == "bf16"
        else torch.float32
    )
    vae.to(accelerator.device, dtype=dtype)
    text_encoder.to(accelerator.device, dtype=dtype)
    losses, global_step = [], 0
    for _ in range(epochs):
        unet.train()
        for batch in loader:
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    latents = (
                        vae.encode(batch["pixel_values"].to(dtype=dtype)).latent_dist.sample()
                        * vae.config.scaling_factor
                    )
                    masked = (
                        vae.encode(batch["masked_pixel_values"].to(dtype=dtype)).latent_dist.sample()
                        * vae.config.scaling_factor
                    )
                    conditioning = text_encoder(batch["input_ids"])[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy = scheduler.add_noise(latents, noise, timesteps)
                model_input = make_inpainting_unet_input(noisy, batch["mask"], masked)
                prediction = unet(model_input, timesteps, conditioning).sample
                target = (
                    noise
                    if scheduler.config.prediction_type == "epsilon"
                    else scheduler.get_velocity(latents, noise, timesteps)
                )
                loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                value = accelerator.gather(loss.detach().repeat(latents.shape[0])).mean().item()
                losses.append(
                    {"step": global_step, "loss": value, "learning_rate": lr_scheduler.get_last_lr()[0]}
                )
                if accelerator.is_main_process and global_step % int(cfg["logging_steps"]) == 0:
                    print(f"step={global_step} loss={value:.6f}", flush=True)
                if global_step % int(cfg["checkpointing_steps"]) == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        checkpoint = output_dir / "checkpoints" / f"checkpoint-{global_step}"
                        checkpoint.mkdir(parents=True, exist_ok=True)
                        _save_lora(accelerator.unwrap_model(unet), checkpoint)
                        _atomic_json(
                            checkpoint / "metadata.json",
                            {
                                "global_step": global_step,
                                "base_model": model_cfg["name"],
                                "rank": cfg["rank"],
                            },
                        )
            if global_step >= int(cfg["max_train_steps"]):
                break
        if global_step >= int(cfg["max_train_steps"]):
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_lora(accelerator.unwrap_model(unet), output_dir)
        with (output_dir / "loss.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "loss", "learning_rate"])
            writer.writeheader()
            writer.writerows(losses)
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 4))
            plt.plot([x["step"] for x in losses], [x["loss"] for x in losses])
            plt.xlabel("optimization step")
            plt.ylabel("MSE loss")
            plt.tight_layout()
            plt.savefig(output_dir / "loss_curve.png", dpi=150)
            plt.close()
        except ImportError:
            pass
        _atomic_json(output_dir / "training_config.json", config)
        _atomic_json(
            output_dir / "training_summary.json",
            {
                "completed_steps": global_step,
                "base_model": model_cfg["name"],
                "trainable_parameters": sum(p.numel() for p in params),
                "full_model_finetuned": False,
                "shared_class_conditioned_lora": True,
                "training_records": len(dataset),
                "training_manifest_sha256": manifest_payload["manifest_sha256"],
                "sampled_category_records": dict(Counter(r["category_name"] for r in dataset.records)),
            },
        )
    return {"completed_steps": global_step, "output_dir": str(output_dir)}
