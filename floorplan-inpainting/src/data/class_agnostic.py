import copy
import json
import os
from pathlib import Path


def make_class_agnostic_coco(coco, category_id=1, category_name="room"):
    """Return a COCO copy whose instance geometry and IDs are unchanged."""
    category_id = int(category_id)
    if category_id != 1:
        raise ValueError("The class-agnostic foreground category must use ID 1")
    output = copy.deepcopy(coco)
    output["categories"] = [{"id": category_id, "name": category_name, "supercategory": "room"}]
    for annotation in output["annotations"]:
        annotation["category_id"] = category_id
    output["info"] = {
        **output.get("info", {}),
        "class_agnostic": True,
        "source_categories": copy.deepcopy(coco["categories"]),
        "description": (
            "Class-agnostic room-instance segmentation derived from the deduplicated Phase 1 COCO dataset"
        ),
    }
    return output


def validate_class_agnostic(source, converted):
    if converted["categories"] != [{"id": 1, "name": "room", "supercategory": "room"}]:
        raise ValueError("Expected one class-agnostic foreground category")
    if source["images"] != converted["images"]:
        raise ValueError("Image records changed during class-agnostic conversion")
    if len(source["annotations"]) != len(converted["annotations"]):
        raise ValueError("Annotation count changed during class-agnostic conversion")
    for original, agnostic in zip(source["annotations"], converted["annotations"]):
        expected = copy.deepcopy(original)
        expected["category_id"] = 1
        if agnostic != expected:
            raise ValueError(f"Annotation {original['id']} changed beyond category_id")
    if set(converted) - set(source) - {"info"}:
        raise ValueError("Unexpected top-level fields in class-agnostic COCO")
    return True


def write_class_agnostic_coco(source_path, output_path):
    source_path, output_path = Path(source_path), Path(output_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    converted = make_class_agnostic_coco(source)
    validate_class_agnostic(source, converted)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    return converted
