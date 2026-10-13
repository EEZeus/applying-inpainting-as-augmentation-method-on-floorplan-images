#!/usr/bin/env python3
"""OCR study, Stages 1-2: does the printed text carry the room class? (CPU)

Stage 1 (GT rooms, no detector error): pool the OCR label tokens inside every
annotated room and classify the room with fixed-hyperparameter heads trained
on the 35 training plans only. Evaluated on the 21 held-out plans:
  zero_shot  argmax class-name similarity of the room's tokens (no training)
  text       logistic regression on pooled token features
  visual     logistic regression on frozen DINOv2 crop features (if present)
  fused      logistic regression on [text ; visual]

Stage 2 (two-stage instance segmentation): the 12 class-agnostic e0 last.pth
runs of the confirmatory study (seeds 1001-1012, predictions already dumped on
the 21 plans) supply masks; the text head labels each mask (top-3 classes,
score x probability). Multiclass mask AP is compared per seed with the 12
multiclass e0 runs of the multiclass follow-up on the same plans.

Descriptive; nothing here selects an arm or a hyperparameter.
"""
import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation.masks import rasterize_annotation  # noqa: E402
from src.data.coco_dataset import read_split  # noqa: E402
from src.evaluation.coco_eval import evaluate_coco_results  # noqa: E402
from src.evaluation.confirm_stats import paired_t  # noqa: E402
from src.evaluation.per_class_metrics import per_class_mask_metrics, rare_aggregate_metrics  # noqa: E402
from src.evaluation.subset_ap import load_dump  # noqa: E402
from src.ocr.features import OcrProvider  # noqa: E402
from src.ocr.text import tokens_in_mask  # noqa: E402
from src.utils.config import load_yaml, resolve_path  # noqa: E402

CATS = list(range(1, 20))
RARE = [11, 12, 13, 14, 15, 16]
HEAD = list(range(1, 11))
SEEDS = list(range(1001, 1013))


def pooled_feature(boxes, feats, conf, mask, sim_slice):
    """Same pooling as the text-aware RoI head, over a pixel mask instead of a box."""
    dim, n_sim = feats.shape[1], sim_slice[1] - sim_slice[0]
    height, width = mask.shape
    if len(boxes):
        pixel_boxes = boxes * np.array([width, height, width, height], dtype=np.float32)
        inside = tokens_in_mask(pixel_boxes, mask)
    else:
        inside = np.zeros(0, dtype=np.int64)
    if inside.size == 0:
        return np.zeros(dim + n_sim + 2, dtype=np.float32), 0
    weights = conf[inside][:, None]
    mean = (weights * feats[inside]).sum(0) / max(weights.sum(), 1e-6)
    max_sim = feats[inside, sim_slice[0]:sim_slice[1]].max(0)
    return np.concatenate([mean, max_sim, [1.0, np.log1p(inside.size) / 3.0]]).astype(np.float32), int(inside.size)


class LogReg:
    """Multinomial logistic regression, fixed hyperparameters, 1/sqrt(n) class weights."""

    def __init__(self, epochs=500, lr=0.01, weight_decay=1e-3, seed=0):
        self.epochs, self.lr, self.weight_decay, self.seed = epochs, lr, weight_decay, seed

    def fit(self, x, y):
        torch.manual_seed(self.seed)
        self.mu, self.sd = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-6
        xt = torch.from_numpy((x - self.mu) / self.sd).float()
        yt = torch.as_tensor([CATS.index(int(v)) for v in y])
        counts = Counter(int(v) for v in y)
        weights = torch.tensor([1 / np.sqrt(counts[c]) if counts[c] else 0.0 for c in CATS], dtype=torch.float32)
        self.layer = torch.nn.Linear(x.shape[1], len(CATS))
        optimizer = torch.optim.Adam(self.layer.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(self.layer(xt), yt, weight=weights).backward()
            optimizer.step()
        return self

    def proba(self, x):
        with torch.no_grad():
            logits = self.layer(torch.from_numpy((x - self.mu) / self.sd).float())
            return torch.softmax(logits, 1).numpy()


def classification_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recall = {c: float((y_pred[y_true == c] == c).mean()) for c in CATS if (y_true == c).any()}
    rare = np.isin(y_true, RARE)
    return {"n": int(len(y_true)), "top1": float((y_true == y_pred).mean()),
            "macro_recall": float(np.mean(list(recall.values()))),
            "head_macro_recall": float(np.mean([recall[c] for c in HEAD if c in recall])),
            "rare_macro_recall": float(np.mean([recall[c] for c in RARE if c in recall])),
            "rare_pooled_recall": float((y_pred[rare] == y_true[rare]).mean()) if rare.any() else None,
            "per_class_recall": {str(c): v for c, v in recall.items()}}


def room_table(coco, image_ids, provider, sim_slice):
    infos = {int(i["id"]): i for i in coco["images"]}
    rows = []
    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in image_ids:
            continue
        info = infos[image_id]
        boxes, feats, conf = provider.get(info["file_name"], info["width"], info["height"])
        mask = rasterize_annotation(annotation, int(info["height"]), int(info["width"]))
        feature, count = pooled_feature(boxes, feats, conf, mask, sim_slice)
        rows.append({"ann_id": int(annotation["id"]), "image_id": image_id, "label": int(annotation["category_id"]),
                     "feature": feature, "tokens": count})
    return rows


def load_dino(paths):
    lookup = {}
    for path in paths:
        if not Path(path).is_file():
            return None
        data = np.load(path, allow_pickle=True)
        for key, feature in zip(data["keys"], data["features"]):
            lookup[int(key)] = feature.astype(np.float32)
    return lookup


def two_stage(dump, coco_gt, infos, provider, sim_slice, model, heldout, top_k=3, min_prob=0.05):
    masks, bboxes = [], []
    cache = {}
    for detection in dump["segm"]:
        image_id = int(detection["image_id"])
        info = infos[image_id]
        if image_id not in cache:
            cache[image_id] = provider.get(info["file_name"], info["width"], info["height"])
        boxes, feats, conf = cache[image_id]
        binary = mask_utils.decode(detection["segmentation"]).astype(bool)
        feature, _ = pooled_feature(boxes, feats, conf, binary, sim_slice)
        probability = model(feature[None])[0]
        for index in np.argsort(-probability)[:top_k]:
            if probability[index] < min_prob:
                continue
            score = float(detection["score"]) * float(probability[index])
            common = {"image_id": image_id, "category_id": CATS[index], "score": score}
            masks.append({**common, "segmentation": detection["segmentation"]})
            bboxes.append({**common, "bbox": [float(v) for v in mask_utils.toBbox(detection["segmentation"])]})
    metrics, _, coco, evaluators = evaluate_coco_results(resolve_path("data/processed/instances_deduplicated.json"),
                                                         bboxes, masks, heldout, return_evaluators=True)
    rows = per_class_mask_metrics(coco, evaluators["mask"], heldout)
    rare = rare_aggregate_metrics(rows, load_yaml(resolve_path("configs/rare_classes.yaml")))
    return {"mask_AP": metrics["mask"]["AP"], "mask_AP50": metrics["mask"]["AP50"],
            "rare_mask_AP": rare["rare_mask_AP"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-cache", default="data/ocr/original_tokens.json")
    parser.add_argument("--ocr-features", default="data/ocr/original_features.npz")
    parser.add_argument("--dino-root", default="outputs/dino")
    parser.add_argument("--agnostic-root", default="outputs/confirm/agnostic/e0")
    parser.add_argument("--multiclass-root", default="outputs/confirm_mc/multiclass/e0")
    parser.add_argument("--out", default="outputs/final/ocr_study/diagnostics")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = parser.parse_args()
    out = resolve_path(args.out); out.mkdir(parents=True, exist_ok=True)
    coco = json.loads(resolve_path("data/processed/instances_deduplicated.json").read_text(encoding="utf-8"))
    coco_gt = COCO(str(resolve_path("data/processed/instances_deduplicated.json")))
    infos = {int(i["id"]): i for i in coco["images"]}
    train = set(read_split(resolve_path("data/splits/train.txt")))
    heldout = read_split(resolve_path("data/splits_heldout21/test.txt"))
    provider = OcrProvider([resolve_path(args.ocr_cache)], [resolve_path(args.ocr_features)])
    sim_slice = tuple(provider.layout["class_sim"])
    names = {int(c["id"]): c["name"] for c in coco["categories"]}

    train_rows = room_table(coco, train, provider, sim_slice)
    test_rows = room_table(coco, set(heldout), provider, sim_slice)
    summary = {"ocr_coverage": {}, "stage1": {}, "stage2": {}}
    for split, rows in (("train", train_rows), ("heldout21", test_rows)):
        by_group = defaultdict(list)
        for row in rows:
            group = "rare" if row["label"] in RARE else "head" if row["label"] in HEAD else "other"
            by_group[group].append(row["tokens"] > 0)
            by_group["all"].append(row["tokens"] > 0)
        summary["ocr_coverage"][split] = {g: float(np.mean(v)) for g, v in by_group.items()}
    tokens_total = sum(len(e["tokens"]) for e in provider.images.values())
    summary["ocr_tokens_total"] = tokens_total

    x_train = np.stack([r["feature"] for r in train_rows]); y_train = [r["label"] for r in train_rows]
    x_test = np.stack([r["feature"] for r in test_rows]); y_test = [r["label"] for r in test_rows]
    s0, s1 = sim_slice
    n_feat = provider.dim
    # zero-shot: max class-name similarity of the room's tokens; rooms without text get the majority class.
    majority = Counter(y_train).most_common(1)[0][0]
    zero = [CATS[int(np.argmax(f[n_feat:n_feat + (s1 - s0)]))] if f[-2] > 0 else majority for f in x_test]
    summary["stage1"]["zero_shot"] = classification_metrics(y_test, zero)
    text_model = LogReg().fit(x_train, y_train)
    summary["stage1"]["text"] = classification_metrics(y_test, [CATS[i] for i in text_model.proba(x_test).argmax(1)])
    dino = load_dino([resolve_path(args.dino_root) / f"gt_{s}.npz" for s in ("train", "val", "test")])
    if dino is not None:
        v_train = np.stack([dino[r["ann_id"]] for r in train_rows])
        v_test = np.stack([dino[r["ann_id"]] for r in test_rows])
        visual = LogReg().fit(v_train, y_train)
        summary["stage1"]["visual"] = classification_metrics(y_test, [CATS[i] for i in visual.proba(v_test).argmax(1)])
        fused = LogReg().fit(np.hstack([x_train, v_train]), y_train)
        summary["stage1"]["fused"] = classification_metrics(
            y_test, [CATS[i] for i in fused.proba(np.hstack([x_test, v_test])).argmax(1)])

    # ---- Stage 2
    runs = []
    for seed in args.seeds:
        dump_path = resolve_path(args.agnostic_root) / f"seed_{seed}/heldout21_last/metrics/coco_results.json.gz"
        mc_path = resolve_path(args.multiclass_root) / f"seed_{seed}/heldout21_last/result.json"
        if not dump_path.is_file():
            continue
        dump = load_dump(dump_path)
        if sorted(dump["image_ids"]) != sorted(heldout):
            raise ValueError(f"{dump_path} was not made on the 21 held-out plans")
        result = two_stage(dump, coco_gt, infos, provider, sim_slice, text_model.proba, heldout)
        result["seed"] = seed
        if mc_path.is_file():
            metrics = json.loads(mc_path.read_text())["test_metrics"]
            result["e0_multiclass_mask_AP"] = metrics["mask"]["AP"]
            result["e0_multiclass_rare_mask_AP"] = metrics.get("rare_mask_AP")
        runs.append(result)
        print(f"seed {seed}: two-stage {result['mask_AP']:.4f} vs e0 multiclass "
              f"{result.get('e0_multiclass_mask_AP', float('nan')):.4f}", flush=True)
    summary["stage2"]["runs"] = runs
    paired = [(r["e0_multiclass_mask_AP"], r["mask_AP"]) for r in runs if "e0_multiclass_mask_AP" in r]
    if len(paired) >= 2:
        control, treatment = zip(*paired)
        summary["stage2"]["two_stage_vs_e0"] = {
            "n": len(paired), "two_stage_mean": statistics.fmean(treatment), "e0_mean": statistics.fmean(control),
            **paired_t(control, treatment)}
    (out / "diagnostics.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    lines = ["# OCR study diagnostics (Stages 1-2, descriptive)", "",
             f"OCR tokens cached: {tokens_total}. Rooms with at least one label token: "
             + ", ".join(f"{s} {v['all']:.2f} (rare {v.get('rare', float('nan')):.2f})"
                         for s, v in summary["ocr_coverage"].items()), "",
             "## Stage 1: classify ground-truth rooms of the 21 held-out plans", "",
             "| Head | top-1 | macro recall | head recall | rare recall (macro) | rare recall (pooled) |",
             "|---|---|---|---|---|---|"]
    for name, m in summary["stage1"].items():
        lines.append(f"| {name} | {m['top1']:.3f} | {m['macro_recall']:.3f} | {m['head_macro_recall']:.3f} | "
                     f"{m['rare_macro_recall']:.3f} | {m['rare_pooled_recall']:.3f} |")
    lines += ["", "Per-class recall (text head): " + ", ".join(
        f"{names[int(c)]} {v:.2f}" for c, v in summary["stage1"]["text"]["per_class_recall"].items()), ""]
    if runs:
        lines += ["## Stage 2: agnostic e0 masks + text head vs multiclass e0 (21 plans, last.pth)", "",
                  "| seed | two-stage AP | two-stage rare AP | e0 multiclass AP | e0 rare AP |", "|---|---|---|---|---|"]
        fmt = lambda v: "NA" if v is None else f"{v:.4f}"
        for r in runs:
            lines.append(f"| {r['seed']} | {fmt(r['mask_AP'])} | {fmt(r['rare_mask_AP'])} | "
                         f"{fmt(r.get('e0_multiclass_mask_AP'))} | {fmt(r.get('e0_multiclass_rare_mask_AP'))} |")
        c = summary["stage2"].get("two_stage_vs_e0")
        if c:
            lines += ["", f"Paired difference (two-stage - e0): {c['delta']:+.4f}, t = {c['t']:.2f}, "
                          f"one-sided p = {c['p_one_sided']:.4g} (n = {c['n']} seeds)."]
    (out / "diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
