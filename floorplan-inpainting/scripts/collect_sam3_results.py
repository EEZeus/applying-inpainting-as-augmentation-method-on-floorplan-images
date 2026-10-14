#!/usr/bin/env python3
"""Score every finished SAM 3 run on its test split and tabulate.

For each run it re-runs pycocotools COCOeval on the dumped test predictions
(outputs/sam3_<name>/dumps/test/{bbox,mask}/coco_predictions_*.json) against that
run's test GT (data/sam3_<name>/test/_annotations.coco.json), so the numbers are
reproducible from artefacts and independent of log parsing.

Writes outputs/final/sam3/summary.{json,csv}.
"""

import csv
import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name -> (dataset dir under data/, output dir under outputs/)
RUNS = [
    ("baseline / multiclass", "sam3_floorplan", "sam3_baseline"),
    ("baseline / agnostic", "sam3_floorplan_agnostic", "sam3_baseline_agnostic"),
    ("Strategy C inpaint / multiclass", "sam3_strategy_c_inpaint", "sam3_strategy_c_inpaint"),
    ("Strategy C inpaint / agnostic", "sam3_strategy_c_inpaint_agnostic", "sam3_strategy_c_inpaint_agnostic"),
    ("LoRA-b / multiclass", "sam3_lora_b", "sam3_lora_b"),
    ("LoRA-b / agnostic", "sam3_lora_b_agnostic", "sam3_lora_b_agnostic"),
    ("copy-paste / multiclass", "sam3_copy_paste", "sam3_copy_paste"),
    ("copy-paste / agnostic", "sam3_copy_paste_agnostic", "sam3_copy_paste_agnostic"),
]


def _ap(gt_path, pred_path, iou_type):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    buf = StringIO()
    with redirect_stdout(buf):
        coco_gt = COCO(str(gt_path))
        if pred_path.stat().st_size == 0:
            return None
        coco_dt = coco_gt.loadRes(str(pred_path))
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP75": float(ev.stats[2])}


def main():
    out_dir = ROOT / "outputs/final/sam3"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, report = [], []
    for label, data_name, out_name in RUNS:
        gt = ROOT / "data" / data_name / "test" / "_annotations.coco.json"
        bbox_pred = ROOT / "outputs" / out_name / "dumps/test/bbox/coco_predictions_bbox.json"
        mask_pred = ROOT / "outputs" / out_name / "dumps/test/mask/coco_predictions_segm.json"
        entry = {
            "run": label,
            "data": data_name,
            "outputs": out_name,
            "status": "ok",
            "bbox": None,
            "mask": None,
        }
        try:
            if not gt.is_file():
                entry["status"] = "no test GT"
            elif not mask_pred.is_file():
                entry["status"] = "no predictions (run not finished?)"
            else:
                entry["bbox"] = _ap(gt, bbox_pred, "bbox") if bbox_pred.is_file() else None
                entry["mask"] = _ap(gt, mask_pred, "segm")
        except Exception as exc:  # record the failure and keep going
            entry["status"] = f"error: {exc}"
        report.append(entry)
        rows.append(
            {
                "run": label,
                "bbox_AP": entry["bbox"]["AP"] if entry["bbox"] else None,
                "mask_AP": entry["mask"]["AP"] if entry["mask"] else None,
                "mask_AP50": entry["mask"]["AP50"] if entry["mask"] else None,
                "status": entry["status"],
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "bbox_AP", "mask_AP", "mask_AP50", "status"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(report, indent=2))
    missing = [r["run"] for r in report if r["status"] != "ok"]
    if missing:
        print("not scored: " + "; ".join(missing), file=sys.stderr)


if __name__ == "__main__":
    main()
