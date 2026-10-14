from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_qa_row(
    path,
    original_full,
    source_crop,
    room_mask_crop,
    safe_mask_crop,
    generated_model_crop,
    final_full,
    record,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = record["crop_bbox"]
    difference = np.abs(
        original_full[y0:y1, x0:x1].astype(np.int16) - final_full[y0:y1, x0:x1].astype(np.int16)
    )
    difference = np.clip(difference * 4, 0, 255).astype(np.uint8)
    panels = [
        (original_full, "Original full plan", None),
        (source_crop, "Source room crop", None),
        (room_mask_crop, "Original room mask", "gray"),
        (safe_mask_crop, "Safe mutable mask", "gray"),
        (generated_model_crop, "Generated crop", None),
        (difference, "Difference (4×)", None),
        (final_full, "Final full plan", None),
    ]
    fig, axes = plt.subplots(1, 7, figsize=(21, 4.2))
    for axis, (value, title, cmap) in zip(axes, panels):
        axis.imshow(value, cmap=cmap, vmin=0, vmax=255)
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    fig.suptitle(
        f"image={record['source_image_id']} ann={record['target_annotation_id']} "
        f"class={record['category_name']} seed={record['seed']} "
        f"mutable={record['mutable_fraction']:.3f} status={record['status']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def _stratified_rows(records, maximum):
    groups = defaultdict(list)
    for record in records:
        row = record.get("qa_row_path")
        if row and Path(row).is_file():
            groups[int(record["category_id"])].append(record)
    queues = {}
    for category, values in groups.items():
        values.sort(key=lambda item: (item["status"] != "rejected", item["sample_id"]))
        queues[category] = deque(values)
    chosen = []
    while len(chosen) < maximum and any(queues.values()):
        for category in sorted(queues):
            if queues[category] and len(chosen) < maximum:
                chosen.append(queues[category].popleft())
    return chosen


def build_contact_sheets(records, output_dir, maximum=30, rows_per_sheet=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _stratified_rows(records, maximum)
    paths = []
    for sheet_index in range(0, len(selected), rows_per_sheet):
        batch = selected[sheet_index : sheet_index + rows_per_sheet]
        images = [Image.open(record["qa_row_path"]).convert("RGB") for record in batch]
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        sheet = Image.new("RGB", (width, height), "white")
        y = 0
        for image in images:
            sheet.paste(image, (0, y))
            y += image.height
        path = output_dir / f"contact_sheet_{sheet_index // rows_per_sheet + 1:02d}.jpg"
        sheet.save(path, quality=92)
        paths.append(path)
        for image in images:
            image.close()
    return paths
