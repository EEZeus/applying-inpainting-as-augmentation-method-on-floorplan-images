import random

from torchvision.transforms import functional as F


class DetectionTransform:
    def __init__(self, train=False, flip_probability=0.5, brightness=0.0, contrast=0.0):
        self.train = train
        self.flip_probability = flip_probability
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image, target):
        if self.train and random.random() < self.flip_probability:
            image = F.hflip(image)
            width = image.width
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes
            target["masks"] = target["masks"].flip(-1)
            if "_ocr_boxes" in target:
                # OCR token boxes are normalised to [0, 1].
                ocr = target["_ocr_boxes"].clone()
                ocr[:, [0, 2]] = 1.0 - target["_ocr_boxes"][:, [2, 0]]
                target["_ocr_boxes"] = ocr
        if self.train and self.brightness:
            image = F.adjust_brightness(image, 1 + random.uniform(-self.brightness, self.brightness))
        if self.train and self.contrast:
            image = F.adjust_contrast(image, 1 + random.uniform(-self.contrast, self.contrast))
        return F.pil_to_tensor(image).float().div(255), target


def collate_fn(batch):
    return tuple(zip(*batch))
