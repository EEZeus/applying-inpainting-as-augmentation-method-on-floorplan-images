#!/usr/bin/env python3
"""Two-stage room typing: frozen DINOv2 crop features -> lightweight head.

Stage 1 (localisation) is the class-agnostic detector, already trained. This
script is stage 2: a linear head and a cosine-kNN head over the DINOv2 features
from dino_extract_crops.py.

Arms:
  real            head trained on real train-split GT crops
  real_plus_synth head trained on real + synthetic rare (cats 11-16) crops

Evaluations:
  A) GT-region top-1 classification on the real test crops (isolates the
     classification bottleneck; no detector error)
  B) two-stage mask AP: label the class-agnostic SAM3 test proposals with the
     head, then COCOeval(segm) vs the multiclass test GT

Writes outputs/final/dino/summary.{json,csv} and report.md. CPU only.
"""

import argparse
import csv
import io
import json
import statistics
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CATS = list(range(1, 20))  # room category ids
RARE = [11, 12, 13, 14, 15, 16]
HEAD = list(range(1, 11))
LOWSUP = [17, 18]
IDX = {c: i for i, c in enumerate(CATS)}


def _load(p):
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _standardize_fit(x):
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True) + 1e-6
    return mu, sd


def _prep(x, mu, sd):
    x = (x - mu) / sd
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return (x / n).astype(np.float32)


def _class_weights(labels):
    w = np.ones(len(CATS), dtype=np.float32)
    uniq, cnt = np.unique(labels, return_counts=True)
    cmap = dict(zip(uniq.tolist(), cnt.tolist()))
    for c in CATS:
        n = cmap.get(c, 0)
        w[IDX[c]] = 0.0 if n == 0 else (1.0 / np.sqrt(n))
    if w.sum() > 0:
        w = w / w[w > 0].mean()
    return torch.tensor(w)


def train_linear(feat, labels, seed=0, epochs=300, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed)
    x = torch.tensor(feat)
    y = torch.tensor([IDX[int(c)] for c in labels], dtype=torch.long)
    net = torch.nn.Linear(x.shape[1], len(CATS))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.CrossEntropyLoss(weight=_class_weights(labels))
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(net(x), y).backward()
        opt.step()
    net.eval()
    return net


def linear_predict(net, feat):
    with torch.no_grad():
        logits = net(torch.tensor(feat))
        prob = torch.softmax(logits, dim=1).numpy()
    idx = prob.argmax(1)
    return np.array([CATS[i] for i in idx]), prob.max(1)


def knn_predict(train_feat, train_labels, feat, k=5):
    sim = feat @ train_feat.T  # cosine, features are L2-normed
    order = np.argsort(-sim, axis=1)[:, :k]
    preds, conf = [], []
    for r, idxs in enumerate(order):
        votes = {}
        for j in idxs:
            votes[int(train_labels[j])] = votes.get(int(train_labels[j]), 0.0) + float(sim[r, j])
        best = max(votes, key=votes.get)
        preds.append(best)
        conf.append(votes[best] / (sum(votes.values()) + 1e-8))
    return np.array(preds), np.array(conf, dtype=np.float32)


def recall_report(y_true, y_pred):
    out = {"top1_acc": float((y_true == y_pred).mean())}
    per = {}
    for c in CATS:
        m = y_true == c
        per[c] = {"support": int(m.sum()), "recall": float((y_pred[m] == c).mean()) if m.any() else None}
    out["per_class"] = per

    def macro(ids):
        v = [per[c]["recall"] for c in ids if per[c]["recall"] is not None]
        return statistics.fmean(v) if v else None

    out["head_macro_recall"] = macro(HEAD)
    out["rare_macro_recall"] = macro(RARE)
    out["lowsup_macro_recall"] = macro(LOWSUP)
    return out


def two_stage_ap(pred_cats, pred_conf, dets, prop_scores, gt_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    results = []
    for cat, conf, det, s in zip(pred_cats, pred_conf, dets, prop_scores):
        results.append(
            {
                "image_id": int(det["image_id"]),
                "category_id": int(cat),
                "segmentation": det["segmentation"],
                "score": float(s) * float(conf),
            }
        )
    buf = io.StringIO()
    with redirect_stdout(buf):
        gt = COCO(str(gt_path))
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, "segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        overall = {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP75": float(ev.stats[2])}
        gt_cats = set(int(c) for c in gt.getCatIds())
        per_cat = {}
        for c in CATS:
            if c not in gt_cats:
                continue
            e = COCOeval(gt, dt, "segm")
            e.params.catIds = [c]
            e.evaluate()
            e.accumulate()
            e.summarize()
            per_cat[c] = float(e.stats[0])

    def macro(ids):
        v = [per_cat[c] for c in ids if c in per_cat and per_cat[c] >= 0]
        return statistics.fmean(v) if v else None

    return {
        "overall": overall,
        "per_class_AP": per_cat,
        "head_macro_AP": macro(HEAD),
        "rare_macro_AP": macro(RARE),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/dino", help="dir holding the .npz from extraction")
    ap.add_argument("--gt-multiclass", default="data/sam3_floorplan/test/_annotations.coco.json")
    ap.add_argument("--out", default="outputs/final/dino")
    ap.add_argument("--knn-k", type=int, default=5)
    args = ap.parse_args()
    d = ROOT / args.dir
    gt_tr = _load(d / "gt_train.npz")
    gt_va = _load(d / "gt_val.npz")
    gt_te = _load(d / "gt_test.npz")
    synth = _load(d / "synth_rare.npz")
    prop = _load(d / "test_proposals.npz")
    dets = json.loads((d / "test_proposals.dets.json").read_text())
    assert len(dets) == len(prop["features"]), "proposal dets/npz length mismatch"

    mu, sd = _standardize_fit(gt_tr["features"])
    Xtr = _prep(gt_tr["features"], mu, sd)
    Xva = _prep(gt_va["features"], mu, sd)
    Xte = _prep(gt_te["features"], mu, sd)
    Xsy = _prep(synth["features"], mu, sd)
    Xpr = _prep(prop["features"], mu, sd)

    arms = {
        "real": (Xtr, gt_tr["labels"]),
        "real_plus_synth": (np.concatenate([Xtr, Xsy]), np.concatenate([gt_tr["labels"], synth["labels"]])),
    }

    report = {
        "knn_k": args.knn_k,
        "n_train_real": int(len(Xtr)),
        "n_synth_rare": int(len(Xsy)),
        "n_test_gt": int(len(Xte)),
        "n_test_proposals": int(len(Xpr)),
        "arms": {},
    }
    rows = []
    for arm, (Xa, ya) in arms.items():
        net = train_linear(Xa, ya)
        lin_va = recall_report(gt_va["labels"], linear_predict(net, Xva)[0])
        knn_va = recall_report(gt_va["labels"], knn_predict(Xa, ya, Xva, args.knn_k)[0])

        lin_te_pred, _ = linear_predict(net, Xte)
        knn_te_pred, _ = knn_predict(Xa, ya, Xte, args.knn_k)
        lin_te = recall_report(gt_te["labels"], lin_te_pred)
        knn_te = recall_report(gt_te["labels"], knn_te_pred)

        lin_pr_pred, lin_pr_conf = linear_predict(net, Xpr)
        knn_pr_pred, knn_pr_conf = knn_predict(Xa, ya, Xpr, args.knn_k)
        lin_ap = two_stage_ap(lin_pr_pred, lin_pr_conf, dets, prop["scores"], ROOT / args.gt_multiclass)
        knn_ap = two_stage_ap(knn_pr_pred, knn_pr_conf, dets, prop["scores"], ROOT / args.gt_multiclass)

        report["arms"][arm] = {
            "val": {"linear": lin_va, "knn": knn_va},
            "test_gt_classification": {"linear": lin_te, "knn": knn_te},
            "two_stage_detection_AP": {"linear": lin_ap, "knn": knn_ap},
        }
        for head, te, apr in (("linear", lin_te, lin_ap), ("knn", knn_te, knn_ap)):
            rows.append(
                {
                    "arm": arm,
                    "head": head,
                    "test_top1_acc": round(te["top1_acc"], 4),
                    "head_macro_recall": round(te["head_macro_recall"], 4)
                    if te["head_macro_recall"] is not None
                    else None,
                    "rare_macro_recall": round(te["rare_macro_recall"], 4)
                    if te["rare_macro_recall"] is not None
                    else None,
                    "twostage_mask_AP": round(apr["overall"]["AP"], 4),
                    "twostage_rare_macro_AP": round(apr["rare_macro_AP"], 4)
                    if apr["rare_macro_AP"] is not None
                    else None,
                }
            )

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# DINOv2 two-stage room typing",
        "",
        f"- real train crops: {report['n_train_real']}  |  synthetic rare crops: {report['n_synth_rare']}",
        f"- test GT crops: {report['n_test_gt']}  |  test agnostic proposals: {report['n_test_proposals']}",
        "",
        "| arm | head | test top-1 | head recall | rare recall | two-stage mask AP | two-stage rare AP |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {arm} | {head} | {test_top1_acc} | {head_macro_recall} | {rare_macro_recall} | "
            "{twostage_mask_AP} | {twostage_rare_macro_AP} |".format(**r)
        )
    lines += [
        "",
        "Reference (fine-tuned multiclass, same test split): "
        "SAM3 0.115 mask AP; best Mask R-CNN ~0.10; Mask R-CNN e0 0.081.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
