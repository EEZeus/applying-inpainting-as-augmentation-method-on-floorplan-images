"""Structure-preserving floor-plan augmentation utilities."""

from .backends import InpaintingBackend, StableDiffusionInpaintingBackend
from .crops import CropTransform, composite_masked
from .masks import SafeMaskResult, build_safe_mask, relative_erosion_pixels

__all__ = [
    "CropTransform",
    "InpaintingBackend",
    "SafeMaskResult",
    "StableDiffusionInpaintingBackend",
    "build_safe_mask",
    "composite_masked",
    "relative_erosion_pixels",
]
