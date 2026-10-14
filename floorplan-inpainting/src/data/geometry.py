import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def polygon_area(points):
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) / 2


def polygon_perimeter(points):
    return sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)


def shape_statistics(annotation_file, output_file):
    coco = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    per_class = defaultdict(lambda: defaultdict(list))
    for ann in coco["annotations"]:
        polygons = [[(p[i], p[i + 1]) for i in range(0, len(p), 2)] for p in ann["segmentation"]]
        points = [point for polygon in polygons for point in polygon]
        area = sum(polygon_area(polygon) for polygon in polygons)
        perimeter = sum(polygon_perimeter(polygon) for polygon in polygons)
        hull_area = polygon_area(convex_hull(points))
        width, height = ann["bbox"][2:]
        values = per_class[ann["category_id"]]
        values["area"].append(area)
        values["solidity"].append(area / hull_area if hull_area else 0.0)
        values["normalized_perimeter"].append(perimeter / math.sqrt(area) if area else 0.0)
        values["bbox_aspect_ratio"].append(width / height if height else 0.0)
    rows = []
    for category_id, name in names.items():
        values = per_class[category_id]
        rows.append(
            {
                "category_id": category_id,
                "category_name": name,
                "instances": len(values["area"]),
                "median_mask_area": percentile(values["area"], 0.5),
                "median_solidity": percentile(values["solidity"], 0.5),
                "p10_solidity": percentile(values["solidity"], 0.1),
                "median_normalized_perimeter": percentile(values["normalized_perimeter"], 0.5),
                "median_bbox_aspect_ratio": percentile(values["bbox_aspect_ratio"], 0.5),
            }
        )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return rows
