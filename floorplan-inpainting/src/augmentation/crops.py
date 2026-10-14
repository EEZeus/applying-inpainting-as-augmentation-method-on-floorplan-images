from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CropTransform:
    """Invertible crop -> square padding -> resize transform."""

    x0: int
    y0: int
    x1: int
    y1: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    output_size: int
    image_width: int
    image_height: int

    @classmethod
    def from_bbox(cls, bbox, image_width, image_height, context_fraction, output_size):
        x, y, width, height = (float(value) for value in bbox)
        x0 = max(0, int(np.floor(x - context_fraction * width)))
        y0 = max(0, int(np.floor(y - context_fraction * height)))
        x1 = min(image_width, int(np.ceil(x + width + context_fraction * width)))
        y1 = min(image_height, int(np.ceil(y + height + context_fraction * height)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Empty crop derived from bbox {bbox}")
        crop_width, crop_height = x1 - x0, y1 - y0
        side = max(crop_width, crop_height)
        pad_x, pad_y = side - crop_width, side - crop_height
        return cls(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            pad_left=pad_x // 2,
            pad_top=pad_y // 2,
            pad_right=pad_x - pad_x // 2,
            pad_bottom=pad_y - pad_y // 2,
            output_size=int(output_size),
            image_width=int(image_width),
            image_height=int(image_height),
        )

    @property
    def crop_bbox(self):
        return [self.x0, self.y0, self.x1, self.y1]

    @property
    def crop_size(self):
        return self.x1 - self.x0, self.y1 - self.y0

    @property
    def padded_side(self):
        width, height = self.crop_size
        return max(width, height)

    def to_dict(self):
        return asdict(self)

    def forward_image(self, image_rgb):
        crop = image_rgb[self.y0 : self.y1, self.x0 : self.x1]
        padded = np.full((self.padded_side, self.padded_side, 3), 255, dtype=np.uint8)
        height, width = crop.shape[:2]
        padded[
            self.pad_top : self.pad_top + height,
            self.pad_left : self.pad_left + width,
        ] = crop
        return np.asarray(
            Image.fromarray(padded).resize((self.output_size, self.output_size), Image.Resampling.BICUBIC)
        )

    def forward_mask(self, mask):
        crop = mask[self.y0 : self.y1, self.x0 : self.x1].astype(np.uint8) * 255
        padded = np.zeros((self.padded_side, self.padded_side), dtype=np.uint8)
        height, width = crop.shape
        padded[
            self.pad_top : self.pad_top + height,
            self.pad_left : self.pad_left + width,
        ] = crop
        resized = Image.fromarray(padded).resize(
            (self.output_size, self.output_size), Image.Resampling.NEAREST
        )
        return np.asarray(resized) > 0

    def inverse_image(self, generated_rgb):
        square = np.asarray(
            Image.fromarray(np.asarray(generated_rgb, dtype=np.uint8)).resize(
                (self.padded_side, self.padded_side), Image.Resampling.BICUBIC
            )
        )
        width, height = self.crop_size
        return square[
            self.pad_top : self.pad_top + height,
            self.pad_left : self.pad_left + width,
        ]


def composite_masked(original_rgb, generated_crop_rgb, safe_mask, transform):
    """Copy generated pixels into only the true pixels of an original-size mask."""
    if original_rgb.shape[:2] != safe_mask.shape:
        raise ValueError("Image and safe mask dimensions differ")
    crop_width, crop_height = transform.crop_size
    if generated_crop_rgb.shape[:2] != (crop_height, crop_width):
        raise ValueError("Generated crop dimensions do not match crop transform")
    output = original_rgb.copy()
    local_mask = safe_mask[transform.y0 : transform.y1, transform.x0 : transform.x1]
    output_crop = output[transform.y0 : transform.y1, transform.x0 : transform.x1]
    output_crop[local_mask] = generated_crop_rgb[local_mask]
    return output
