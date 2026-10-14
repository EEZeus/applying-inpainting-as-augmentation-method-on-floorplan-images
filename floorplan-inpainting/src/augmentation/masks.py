from dataclasses import dataclass

import cv2
import numpy as np

from src.data.coco_dataset import FloorplanCocoDataset


@dataclass(frozen=True)
class SafeMaskResult:
    room_mask: np.ndarray
    eroded_mask: np.ndarray
    structure_mask: np.ndarray
    neighbor_mask: np.ndarray
    safe_mask: np.ndarray
    erosion_pixels: int
    mutable_fraction: float
    dark_threshold: float


def relative_erosion_pixels(bbox, fraction=0.02, minimum=2, maximum=12):
    """Return a clamped relative erosion radius for a COCO xywh bbox."""
    width, height = float(bbox[2]), float(bbox[3])
    value = min(max(fraction * min(width, height), minimum), maximum)
    return int(round(value))


def rasterize_annotation(annotation, height, width):
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list):
        raise ValueError(f"Annotation {annotation.get('id')} is not polygonal")
    return FloorplanCocoDataset.rasterize(segmentation, height, width).astype(bool)


def _resolve_dark_threshold(gray, room_mask, configured):
    if configured != "auto":
        value = float(configured)
        if not 0 <= value <= 255:
            raise ValueError("dark_threshold must be 'auto' or a value in [0, 255]")
        return value
    values = gray[room_mask].astype(np.uint8)
    if values.size == 0:
        return 0.0
    threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold)


def build_safe_mask(image_rgb, target_annotation, other_annotations, config):
    """Construct an original-resolution mutable mask for one target room."""
    height, width = image_rgb.shape[:2]
    room = rasterize_annotation(target_annotation, height, width)
    erosion_px = relative_erosion_pixels(
        target_annotation["bbox"],
        config.get("erosion_fraction", 0.02),
        config.get("min_erosion_px", 2),
        config.get("max_erosion_px", 12),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_px + 1, 2 * erosion_px + 1))
    eroded = cv2.erode(
        room.astype(np.uint8), kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dark_threshold = _resolve_dark_threshold(gray, room, config.get("dark_threshold", "auto"))
    structure = np.zeros_like(room)
    if config.get("structure_protection", True):
        dark = room & (gray <= dark_threshold)
        mode = config.get("structure_protection_mode", "all_dark")
        if mode == "all_dark":
            structure = dark
        elif mode == "long_lines":
            # Protect wall strokes but leave compact labels/symbols editable for class conversion.
            bbox_width, bbox_height = (max(1, int(round(v))) for v in target_annotation["bbox"][2:])
            horizontal = cv2.morphologyEx(
                dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, max(5, int(0.20 * bbox_width))), np.uint8)
            ).astype(bool)
            vertical = cv2.morphologyEx(
                dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((max(5, int(0.20 * bbox_height)), 1), np.uint8)
            ).astype(bool)
            structure = horizontal | vertical
        else:
            raise ValueError("structure_protection_mode must be all_dark or long_lines")
        dilation = int(config.get("structure_dilation_px", 2))
        if dilation > 0:
            dilation_kernel = np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)
            structure = cv2.dilate(structure.astype(np.uint8), dilation_kernel, iterations=1).astype(bool)

    neighbors = np.zeros_like(room)
    for annotation in other_annotations:
        neighbors |= rasterize_annotation(annotation, height, width)

    safe = eroded & ~structure & ~neighbors
    room_area = int(room.sum())
    mutable_fraction = float(safe.sum() / room_area) if room_area else 0.0
    return SafeMaskResult(
        room_mask=room,
        eroded_mask=eroded,
        structure_mask=structure,
        neighbor_mask=neighbors,
        safe_mask=safe,
        erosion_pixels=erosion_px,
        mutable_fraction=mutable_fraction,
        dark_threshold=dark_threshold,
    )
