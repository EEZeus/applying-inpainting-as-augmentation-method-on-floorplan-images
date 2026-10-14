import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FloorplanCocoDataset(Dataset):
    def __init__(self, image_dir, annotation_file, image_ids=None, transforms=None, ocr=None):
        self.image_dir = Path(image_dir)
        self.ocr = ocr
        self.annotation_file = str(annotation_file)
        self.transforms = transforms
        with Path(annotation_file).open(encoding="utf-8") as handle:
            self.coco_data = json.load(handle)
        allowed = None if image_ids is None else {int(i) for i in image_ids}
        self.images = sorted(
            (im for im in self.coco_data["images"] if allowed is None or im["id"] in allowed),
            key=lambda im: im["id"],
        )
        self.annotations = defaultdict(list)
        for ann in self.coco_data["annotations"]:
            if allowed is None or ann["image_id"] in allowed:
                self.annotations[ann["image_id"]].append(ann)
        category_ids = sorted(c["id"] for c in self.coco_data["categories"])
        if not category_ids:
            raise ValueError("COCO dataset must define at least one foreground category")
        expected = list(range(1, len(category_ids) + 1))
        if category_ids != expected:
            raise ValueError(
                f"Expected contiguous foreground category IDs 1..{len(category_ids)}, found {category_ids}"
            )

    def __len__(self):
        return len(self.images)

    @staticmethod
    def rasterize(segmentation, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in segmentation:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) >= 3:
                points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        return mask

    def __getitem__(self, index):
        info = self.images[index]
        image = Image.open(self.image_dir / info["file_name"]).convert("RGB")
        width, height = image.size
        masks, labels = [], []
        for ann in self.annotations[info["id"]]:
            if not isinstance(ann["segmentation"], list):
                raise ValueError(f"Annotation {ann['id']} is not polygonal")
            mask = self.rasterize(ann["segmentation"], height, width)
            if mask.any():
                masks.append(mask)
                labels.append(ann["category_id"])
        if masks:
            masks_np = np.stack(masks)
            boxes = []
            for mask in masks_np:
                ys, xs = np.where(mask)
                boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks_tensor = torch.as_tensor(masks_np, dtype=torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
        target = {
            "boxes": boxes,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": masks_tensor,
            "image_id": torch.tensor(info["id"], dtype=torch.int64),
            "area": masks_tensor.flatten(1).sum(1).to(torch.float32),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        if self.ocr is not None:
            target.update(ocr_target_fields(self.ocr, info["file_name"], width, height))
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target


def read_split(path):
    return [int(line) for line in Path(path).read_text().splitlines() if line.strip()]


def ocr_target_fields(provider, file_name, width, height):
    """Private target keys holding the OCR label tokens of one plan."""
    boxes, feats, conf = provider.get(file_name, width, height)
    return {
        "_ocr_boxes": torch.from_numpy(boxes),
        "_ocr_feats": torch.from_numpy(feats),
        "_ocr_conf": torch.from_numpy(conf),
    }
