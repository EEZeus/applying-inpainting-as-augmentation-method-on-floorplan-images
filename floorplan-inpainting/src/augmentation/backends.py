import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


class InpaintingBackend(ABC):
    @abstractmethod
    def generate(self, image, mask, prompt, seed, **kwargs):
        """Return an RGB PIL image with the same dimensions as image."""

    @property
    def metadata(self):
        return {}


class StableDiffusionInpaintingBackend(InpaintingBackend):
    def __init__(self, config):
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline
        except ImportError as error:
            raise RuntimeError("Diffusers dependencies are missing; install requirements.txt") from error

        requested_device = config.get("device", "cuda")
        self.device = requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
        requested_dtype = config.get("dtype", "fp16")
        self.torch_dtype = torch.float32
        if self.device.startswith("cuda"):
            self.torch_dtype = (
                torch.bfloat16
                if requested_dtype == "bf16" and torch.cuda.is_bf16_supported()
                else torch.float16
            )
        self.model_name = config["model_name"]
        self.model_revision = config.get("model_revision") or None
        self.cache_dir = config.get("cache_dir") or None
        self.negative_prompt = config.get("negative_prompt")
        self.steps = int(config.get("num_inference_steps", 25))
        self.guidance_scale = float(config.get("guidance_scale", 6.0))

        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "cache_dir": self.cache_dir,
        }
        if Path(self.model_name).is_dir():
            model_path = self.model_name
            self.resolved_revision = "local"
        else:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                self.model_name,
                revision=self.model_revision,
                cache_dir=self.cache_dir,
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
                ignore_patterns=["*.non_ema.*"],
            )
            self.resolved_revision = Path(model_path).name
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            variant=config.get("variant", "fp16"),
            use_safetensors=True,
            local_files_only=True,
            **load_kwargs,
        )
        self.pipeline = self.pipeline.to(self.device)
        self.lora_path = config.get("lora_path")
        self.lora_weight_name = config.get("lora_weight_name", "pytorch_lora_weights.safetensors")
        self.lora_sha256 = None
        if self.lora_path:
            lora_path = Path(self.lora_path)
            weight_path = lora_path / self.lora_weight_name
            if not weight_path.is_file():
                raise FileNotFoundError(f"LoRA weights not found: {weight_path}")
            self.pipeline.load_lora_weights(str(lora_path), weight_name=self.lora_weight_name)
            digest = hashlib.sha256()
            with weight_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.lora_sha256 = digest.hexdigest()
        self.pipeline.enable_attention_slicing()
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
        self._torch = torch

    def generate(self, image, mask, prompt, seed, **kwargs):
        image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
        mask = mask if isinstance(mask, Image.Image) else Image.fromarray(np.asarray(mask))
        image = image.convert("RGB")
        mask = mask.convert("L")
        generator = self._torch.Generator(device=self.device).manual_seed(int(seed))
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            height=image.height,
            width=image.width,
            **kwargs,
        )
        flags = result.nsfw_content_detected
        if flags and flags[0]:
            raise ValueError("safety_checker_flagged")
        return result.images[0].convert("RGB")

    @property
    def metadata(self):
        return {
            "model_name": self.model_name,
            "model_revision": self.resolved_revision,
            "device": self.device,
            "dtype": str(self.torch_dtype).replace("torch.", ""),
            "gpu": self.gpu_name,
            "lora_path": str(Path(self.lora_path).resolve()) if self.lora_path else None,
            "lora_sha256": self.lora_sha256,
        }
