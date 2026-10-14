from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

from .annotations import PALETTE, draw_annotations


def draw_predictions(image, prediction, category_names, threshold=0.5):
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    segmentations = prediction.get("segmentations")
    masks = prediction.get("masks")
    mask_values = segmentations if segmentations is not None else masks
    for box, label, score, mask in zip(
        prediction["boxes"], prediction["labels"], prediction["scores"], mask_values
    ):
        if float(score) < threshold:
            continue
        label = int(label)
        color = PALETTE[(label - 1) % len(PALETTE)]
        if segmentations is not None:
            encoded = dict(mask)
            if isinstance(encoded["counts"], str):
                encoded["counts"] = encoded["counts"].encode("ascii")
            binary = mask_utils.decode(encoded)
        else:
            binary = (mask[0].numpy() >= 0.5).astype(np.uint8)
        mask_image = Image.fromarray(binary * 100, mode="L")
        color_layer = Image.new("RGBA", image.size, (*color, 0))
        color_layer.putalpha(mask_image)
        overlay = Image.alpha_composite(overlay, color_layer)
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(tuple(map(float, box)), outline=(*color, 255), width=3)
        draw.text(
            (float(box[0]), float(box[1])), f"{category_names[label]} {float(score):.2f}", fill=(*color, 255)
        )
    return Image.alpha_composite(image, overlay).convert("RGB")


def render_prediction_comparisons(dataset, predictions, output_dir, threshold=0.5, limit=20):
    output_dir = Path(output_dir)
    comparison_dir, prediction_dir = output_dir / "comparisons", output_dir / "predictions_only"
    rare_dir, failure_dir = output_dir / "rare_classes", output_dir / "failures"
    for directory in (comparison_dir, prediction_dir, rare_dir, failure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    names = {c["id"]: c["name"] for c in dataset.coco_data["categories"]}
    pred_by_id = dict(predictions)
    index_by_id = {im["id"]: i for i, im in enumerate(dataset.images)}
    rendered = {}
    for image_id in list(pred_by_id)[:limit]:
        index = index_by_id[image_id]
        info = dataset.images[index]
        original = Image.open(dataset.image_dir / info["file_name"]).convert("RGB")
        gt = draw_annotations(original, dataset.annotations[image_id], names)
        pred = draw_predictions(original, pred_by_id[image_id], names, threshold)
        canvas = Image.new("RGB", (original.width * 2, original.height))
        canvas.paste(gt, (0, 0))
        canvas.paste(pred, (original.width, 0))
        pred.save(prediction_dir / f"prediction_{image_id:03d}.jpg", quality=88)
        canvas.save(comparison_dir / f"comparison_{image_id:03d}.jpg", quality=88)
        rendered[image_id] = canvas
    rare_ids = [
        image_id
        for image_id in pred_by_id
        if any(11 <= a["category_id"] <= 18 for a in dataset.annotations[image_id])
    ]
    for image_id in rare_ids[:10]:
        if image_id in rendered:
            rendered[image_id].save(rare_dir / f"rare_{image_id:03d}.jpg", quality=88)

    def mismatch(image_id):
        predicted = sum(float(score) >= threshold for score in pred_by_id[image_id]["scores"])
        return abs(predicted - len(dataset.annotations[image_id]))

    for image_id in sorted(rendered, key=mismatch, reverse=True)[:10]:
        rendered[image_id].save(failure_dir / f"failure_candidate_{image_id:03d}.jpg", quality=88)
