import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
]


def draw_annotations(image, annotations, category_names, alpha=80):
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    for ann in annotations:
        color = PALETTE[(ann["category_id"] - 1) % len(PALETTE)]
        for polygon in ann["segmentation"]:
            points = list(zip(polygon[::2], polygon[1::2]))
            draw.polygon(points, fill=(*color, alpha), outline=(*color, 255), width=3)
        x, y, _, _ = ann["bbox"]
        draw.text((x, y), category_names[ann["category_id"]], fill=(*color, 255))
    return Image.alpha_composite(image, overlay).convert("RGB")


def render_dataset_annotations(annotation_file, image_dir, output_dir, image_ids=None, limit=20):
    coco = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    allowed = None if image_ids is None else set(image_ids)
    anns = defaultdict(list)
    for ann in coco["annotations"]:
        anns[ann["image_id"]].append(ann)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for info in [im for im in coco["images"] if allowed is None or im["id"] in allowed][:limit]:
        image = Image.open(Path(image_dir) / info["file_name"])
        draw_annotations(image, anns[info["id"]], names).save(
            output_dir / f"gt_{info['id']:03d}.jpg", quality=90
        )
