#!/usr/bin/env python3
"""Build every generated figure of the project report as a vector PDF.

Inputs are the result files fetched from the cluster into ``report/assets``:

    assets/final/grid_r1/summary.csv     Mask R-CNN 3-seed grid (14 arms x 2 tracks)
    assets/final/cv/summary.{csv,json}   rotated held-out CV (6 arms x 2 tracks x 5 folds)
    assets/final/sam3/summary.csv        SAM 3 baseline + 3 augmented runs
    assets/final/dino/summary.{csv,json} DINOv2 two-stage classifier
    assets/splits_class_distribution.csv per-class train/val/test instance counts

Outputs land in ``report/images`` and are referenced from the .tex sources.

    cd report && python scripts/make_figures.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FINAL = ASSETS / "final"
IMAGES = ROOT / "images"

# Okabe-Ito colour-blind-safe qualitative palette.
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#9E9E9E"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    }
)

ARM_ORDER = [f"e{i}" for i in range(14)]
# Two-line labels for the 6-bar CV axis; the 14-bar axes use the bare id.
ARM_SHORT = {
    "e0": "e0\noriginal",
    "e1": "e1\nrare-weight",
    "e6": "e6\nB",
    "e8": "e8\nC paste",
    "e10": "e10\nC inpaint",
    "e12": "e12\nLoRA-B",
}
# Family colour: control / sampling / A / B / C / LoRA.
ARM_COLOR = {
    "e0": BLACK,
    "e1": GREY,
    "e2": SKY,
    "e3": SKY,
    "e4": SKY,
    "e5": SKY,
    "e6": GREEN,
    "e7": GREEN,
    "e8": ORANGE,
    "e9": ORANGE,
    "e10": ORANGE,
    "e11": ORANGE,
    "e12": PURPLE,
    "e13": PURPLE,
}
CLASS_NAMES = {
    1: "Büroraum",
    2: "WC",
    3: "Treppe",
    4: "Lager",
    5: "Technikraum",
    6: "Aufzug",
    7: "Andere",
    8: "Küche",
    9: "Flur, Halle",
    10: "Besprechungsraum",
    11: "Allg. Verkehrsfläche",
    12: "Abstellraum",
    13: "Versammlungsraum",
    14: "Schacht",
    15: "Garderobe",
    16: "Sanitärraum",
    17: "Werkstatt",
    18: "Speiseraum",
    19: "Labor",
}
RARE = [11, 12, 13, 14, 15, 16]
LOWSUP = [17, 18]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value):
    return None if value in (None, "") else float(value)


def save(fig, name):
    IMAGES.mkdir(parents=True, exist_ok=True)
    out = IMAGES / name
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def family_legend(fig):
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=BLACK),
        plt.Rectangle((0, 0), 1, 1, fc=GREY),
        plt.Rectangle((0, 0), 1, 1, fc=SKY),
        plt.Rectangle((0, 0), 1, 1, fc=GREEN),
        plt.Rectangle((0, 0), 1, 1, fc=ORANGE),
        plt.Rectangle((0, 0), 1, 1, fc=PURPLE),
    ]
    labels = [
        "no augmentation",
        "sampling only",
        "Strategy A (prompt)",
        "Strategy B (prompt)",
        "Strategy C (exemplar)",
        "LoRA (Strategy B)",
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=6,
        frameon=False,
        handlelength=1.1,
        columnspacing=1.4,
    )


def fig_class_distribution():
    rows = read_csv(ASSETS / "splits_class_distribution.csv")
    ids = [int(r["category_id"]) for r in rows]
    tr = [int(r["train_instances"]) for r in rows]
    va = [int(r["val_instances"]) for r in rows]
    te = [int(r["test_instances"]) for r in rows]
    labels = [f"{i} {CLASS_NAMES[i]}" for i in ids]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ypos = range(len(ids))
    ax.barh(ypos, tr, color=BLUE, label="train (35 images)")
    ax.barh(ypos, va, left=tr, color=SKY, label="val (10 images)")
    ax.barh(ypos, te, left=[a + b for a, b in zip(tr, va)], color=ORANGE, label="test (11 images)")
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("polygon instances")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    for pos, cid, total in zip(ypos, ids, [a + b + c for a, b, c in zip(tr, va, te)]):
        if cid in RARE:
            ax.get_yticklabels()[pos].set_color(VERMILLION)
            ax.get_yticklabels()[pos].set_fontweight("bold")
        elif cid in LOWSUP:
            ax.get_yticklabels()[pos].set_color(GREY)
        ax.text(total + 4, pos, str(total), va="center", fontsize=6.5, color="#444444")

    ax.axhspan(9.5, 15.5, color=VERMILLION, alpha=0.07, zorder=0)
    ax.text(
        ax.get_xlim()[1] * 0.72,
        12.5,
        "primary-rare\nclasses 11–16",
        color=VERMILLION,
        fontsize=7,
        ha="center",
        va="center",
    )
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Room-class instance counts per split (56 images, 1633 annotations)")
    save(fig, "fig_class_distribution.pdf")


def fig_rare_support():
    rows = {int(r["category_id"]): r for r in read_csv(ASSETS / "splits_class_distribution.csv")}
    cats = RARE + LOWSUP
    labels = [f"{c} {CLASS_NAMES[c]}" for c in cats]
    tr = [int(rows[c]["train_instances"]) for c in cats]
    va = [int(rows[c]["val_instances"]) for c in cats]
    te = [int(rows[c]["test_instances"]) for c in cats]
    top = max(tr + va + te)

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    width = 0.27
    xs = range(len(cats))
    for offset, vals, colour, name in (
        (-width, tr, BLUE, "train"),
        (0.0, va, SKY, "val"),
        (width, te, ORANGE, "test"),
    ):
        bars = ax.bar([x + offset for x in xs], vals, width, color=colour, label=name)
        ax.bar_label(bars, fontsize=6.5, padding=1)
    ax.set_xticks(list(xs))
    # Rotated: the German class names collide with their neighbours when horizontal.
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right", rotation_mode="anchor")
    for pos, c in enumerate(cats):
        ax.get_xticklabels()[pos].set_color(VERMILLION if c in RARE else GREY)
    ax.set_ylabel("instances")
    ax.set_ylim(0, top * 1.30)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.axvline(5.5, color=GREY, lw=0.8, ls=":")
    ax.text(6.5, top * 1.13, "low support", color=GREY, fontsize=7, ha="center")
    ax.legend(loc="upper left", frameon=False, ncol=3, columnspacing=1.2)
    ax.set_title("Support of the rare and low-support classes; 17 rare test instances in total")
    save(fig, "fig_rare_support.pdf")


def fig_grid_ap():
    rows = read_csv(FINAL / "grid_r1" / "summary.csv")
    data = {(r["track"], r["arm"]): r for r in rows}

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.8), sharex=True)
    panels = [("multiclass", "multiclass mask AP", axes[0]), ("agnostic", "class-agnostic mask AP", axes[1])]
    for track, ylabel, ax in panels:
        means = [num(data[(track, a)]["overall_mask_AP_mean"]) for a in ARM_ORDER]
        stds = [num(data[(track, a)]["overall_mask_AP_std"]) for a in ARM_ORDER]
        colours = [ARM_COLOR[a] for a in ARM_ORDER]
        ax.bar(
            range(14),
            means,
            yerr=stds,
            capsize=2.5,
            color=colours,
            error_kw={"elinewidth": 0.9, "ecolor": "#333333"},
        )
        base = means[0]
        ax.axhline(base, color=BLACK, ls="--", lw=0.9)
        ax.text(13.62, base, f"e0\n{base:.3f}", va="center", ha="left", fontsize=6.5)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.7, 15.1)
        ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.14)
    axes[0].set_title(
        "Mask R-CNN grid: test mask AP per arm (mean of seeds 42/123/2026, error bars $=$ std, $n=3$)"
    )
    family_legend(fig)
    axes[1].set_xticks(range(14))
    axes[1].set_xticklabels(ARM_ORDER, fontsize=7)
    fig.align_ylabels(axes)
    save(fig, "fig_grid_ap.pdf")


def fig_cv_agnostic():
    summary = json.loads((FINAL / "cv" / "summary.json").read_text(encoding="utf-8"))
    arms = [a for a in summary["arms"] if a["track"] == "agnostic"]
    order = ["e0", "e1", "e6", "e8", "e10", "e12"]
    arms = sorted(arms, key=lambda a: order.index(a["arm"]))

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    xs = range(len(arms))
    means = [a["overall_mask_AP"]["mean"] for a in arms]
    stds = [a["overall_mask_AP"]["std"] for a in arms]
    colours = [VERMILLION if a["arm"] == "e8" else ARM_COLOR[a["arm"]] for a in arms]
    ax.bar(
        xs,
        means,
        yerr=stds,
        capsize=2.5,
        color=colours,
        alpha=0.85,
        error_kw={"elinewidth": 0.9, "ecolor": "#333333"},
    )
    for x, arm in zip(xs, arms):
        folds = arm["overall_mask_AP"]["per_fold"]
        ax.scatter(
            [x + (i - 2) * 0.055 for i in range(len(folds))],
            folds,
            s=9,
            facecolor="white",
            edgecolor=BLACK,
            linewidth=0.6,
            zorder=3,
        )
    ax.axhline(means[0], color=BLACK, ls="--", lw=0.9)
    ax.text(len(arms) - 0.4, means[0], " e0", va="center", fontsize=7)
    ax.annotate(
        "e8: $+0.045$ vs e0,\npositive in 5/5 folds,\nsign test $p=0.0625$",
        xy=(3, means[3] + stds[3]),
        xytext=(3.6, 0.74),
        fontsize=7,
        color=VERMILLION,
        arrowprops={"arrowstyle": "->", "color": VERMILLION, "lw": 0.8},
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([ARM_SHORT[a["arm"]] for a in arms], fontsize=7)
    ax.set_ylabel("class-agnostic mask AP")
    ax.set_ylim(0, 0.92)
    ax.set_title(
        "Rotated held-out CV: agnostic AP per arm (bar $=$ mean of $K=5$ folds,\n"
        "error bar $=$ std, open circles $=$ the individual folds)"
    )
    save(fig, "fig_cv_agnostic.pdf")


def fig_rare_macro():
    grid = {(r["track"], r["arm"]): r for r in read_csv(FINAL / "grid_r1" / "summary.csv")}
    cv = {(r["track"], r["arm"]): r for r in read_csv(FINAL / "cv" / "summary.csv")}
    cv_arms = ["e0", "e1", "e6", "e8", "e10", "e12"]

    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    width = 0.4
    gvals = [num(grid[("multiclass", a)]["rare_mask_AP_mean"]) or 0.0 for a in ARM_ORDER]
    gerr = [num(grid[("multiclass", a)]["rare_mask_AP_std"]) or 0.0 for a in ARM_ORDER]
    ax.bar(
        [x - width / 2 for x in range(14)],
        gvals,
        width,
        yerr=gerr,
        capsize=2,
        color=BLUE,
        label="grid, 11 test images ($n=3$ seeds)",
        error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
    )
    cvals = [(num(cv[("multiclass", a)]["rare_macro_AP"]) or 0.0) if a in cv_arms else 0.0 for a in ARM_ORDER]
    ax.bar(
        [x + width / 2 for x in range(14)],
        [v if ARM_ORDER[i] in cv_arms else 0.0 for i, v in enumerate(cvals)],
        width,
        color=ORANGE,
        label="CV, 21 pooled held-out images ($K=5$ folds)",
    )
    ax.set_xticks(range(14))
    ax.set_xticklabels(ARM_ORDER, fontsize=7)
    ax.set_ylabel("rare-macro mask AP\n(classes 11–16)")
    ax.set_ylim(0, 0.66)
    ax.axhline(0.588, color=GREY, ls=":", lw=0.9)
    ax.text(6.5, 0.545, "for scale: class-agnostic AP of e0 $=0.588$", fontsize=7, color=GREY, ha="center")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), frameon=False, ncol=2)
    ax.set_title("Rare-class macro AP stays at the floor for every arm and both evaluations", pad=18)
    save(fig, "fig_rare_macro.pdf")


def fig_sam3_vs_mrcnn():
    sam = {r["run"]: r for r in read_csv(FINAL / "sam3" / "summary.csv")}
    grid = {(r["track"], r["arm"]): r for r in read_csv(FINAL / "grid_r1" / "summary.csv")}

    # SAM 3 runs and the matching Mask R-CNN grid arms.
    pairs = [
        ("baseline", "e0", "e0\nnone"),
        ("Strategy C inpaint", "e10", "e10\nC inpaint"),
        ("LoRA-b", "e12", "e12\nLoRA-B"),
        ("copy-paste", "e8", "e8\nC paste"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, track, title in ((axes[0], "multiclass", "multiclass"), (axes[1], "agnostic", "class-agnostic")):
        xs = range(len(pairs))
        width = 0.38
        mr = [num(grid[(track, arm)]["overall_mask_AP_mean"]) for _, arm, _ in pairs]
        mrs = [num(grid[(track, arm)]["overall_mask_AP_std"]) for _, arm, _ in pairs]
        s3 = [float(sam[f"{run} / {track}"]["mask_AP"]) for run, _, _ in pairs]
        ax.bar(
            [x - width / 2 for x in xs],
            mr,
            width,
            yerr=mrs,
            capsize=2,
            color=SKY,
            label="Mask R-CNN ($n=3$ seeds)",
            error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
        )
        ax.bar([x + width / 2 for x in xs], s3, width, color=BLUE, label="SAM 3 ($n=1$ seed)")
        ax.axhline(s3[0], color=BLUE, ls="--", lw=0.8)
        ax.axhline(mr[0], color=SKY, ls="--", lw=0.8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([lbl for _, _, lbl in pairs], fontsize=7)
        ax.set_ylabel("test mask AP")
        ax.set_title(title)
        ax.set_ylim(0, (0.20 if track == "multiclass" else 0.86))
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle(
        "Backbone effect dominates the augmentation effect "
        "(dashed lines $=$ the two no-augmentation baselines)",
        y=1.03,
    )
    save(fig, "fig_sam3_vs_mrcnn.pdf")


def fig_dino_recall():
    summary = json.loads((FINAL / "dino" / "summary.json").read_text(encoding="utf-8"))
    per_class = summary["arms"]["real"]["test_gt_classification"]["linear"]["per_class"]
    cats = [c for c in range(1, 19) if per_class[str(c)]["recall"] is not None]
    recalls = [per_class[str(c)]["recall"] for c in cats]
    support = [per_class[str(c)]["support"] for c in cats]
    colours = [VERMILLION if c in RARE else (GREY if c in LOWSUP else BLUE) for c in cats]

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    bars = ax.bar(range(len(cats)), recalls, color=colours)
    ax.bar_label(bars, labels=[f"n={s}" for s in support], fontsize=6, padding=2)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{c} {CLASS_NAMES[c]}" for c in cats], rotation=45, ha="right", fontsize=6.6)
    for pos, c in enumerate(cats):
        if c in RARE:
            ax.get_xticklabels()[pos].set_color(VERMILLION)
    ax.set_ylabel("top-1 recall on GT crops")
    ax.set_ylim(0, 1.12)
    ax.axvspan(9.5, 15.5, color=VERMILLION, alpha=0.07, zorder=0)
    ax.text(
        12.5, 0.95, "rare classes 11–16:\nrecall $=0$ everywhere", color=VERMILLION, fontsize=7, ha="center"
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=BLUE),
        plt.Rectangle((0, 0), 1, 1, fc=VERMILLION),
        plt.Rectangle((0, 0), 1, 1, fc=GREY),
    ]
    ax.legend(
        handles,
        ["head classes 1–10", "primary-rare 11–16", "low-support 17–18"],
        loc="upper left",
        frameon=False,
    )
    ax.set_title(
        "Frozen DINOv2-base + linear head, real-crop arm: per-class recall on 368 ground-truth test crops"
    )
    save(fig, "fig_dino_recall.pdf")


def fig_dino_twostage():
    rows = read_csv(FINAL / "dino" / "summary.csv")
    sam = {r["run"]: r for r in read_csv(FINAL / "sam3" / "summary.csv")}
    grid = {(r["track"], r["arm"]): r for r in read_csv(FINAL / "grid_r1" / "summary.csv")}
    best_mrcnn = max(num(grid[("multiclass", a)]["overall_mask_AP_mean"]) for a in ARM_ORDER)
    sam3_mc = float(sam["baseline / multiclass"]["mask_AP"])
    e0_mc = num(grid[("multiclass", "e0")]["overall_mask_AP_mean"])

    names = {"real": "real crops", "real_plus_synth": "real $+$ synth"}
    labels = [f"{names[r['arm']]}\n{r['head']} head" for r in rows]
    vals = [float(r["twostage_mask_AP"]) for r in rows]
    colours = [BLUE if r["arm"] == "real" else PURPLE for r in rows]

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    bars = ax.bar(range(len(rows)), vals, color=colours, width=0.6)
    ax.bar_label(bars, fmt="%.4f", fontsize=6.8, padding=2)
    for value, colour, name, style in (
        (sam3_mc, BLUE, f"fine-tuned SAM 3 = {sam3_mc:.3f}", "-"),
        (best_mrcnn, GREEN, f"best Mask R-CNN arm = {best_mrcnn:.3f}", "--"),
        (e0_mc, GREY, f"Mask R-CNN e0 = {e0_mc:.3f}", ":"),
    ):
        ax.axhline(value, color=colour, ls=style, lw=1.0)
        ax.text(
            3.62,
            value,
            "  " + name,
            fontsize=6.8,
            color=colour,
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("two-stage multiclass mask AP")
    ax.set_xlim(-0.6, 6.9)
    ax.set_ylim(0, 0.145)
    ax.set_title(
        "Two-stage pipeline (SAM 3 agnostic proposals $+$ DINOv2 head)\n"
        "against the end-to-end fine-tuned detectors, same test split"
    )
    save(fig, "fig_dino_twostage.pdf")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    ax.axis("off")
    ax.grid(False)

    stages = [
        (16.0, "floor plan\n(test image)", "#FFFFFF", BLACK),
        (27.0, "stage 1: class-agnostic\ndetector (SAM 3)\n483 test proposals", "#EAF3FA", BLUE),
        (25.0, "crop: bbox $+$ context,\nwhite square padding,\nresized to $224^2$", "#FFFFFF", BLACK),
        (23.0, "frozen DINOv2-base\n[CLS ; mean-patch]\n1536-d feature", "#EAF7F2", GREEN),
        (23.0, "stage 2 head:\nlinear / cosine kNN\n19 room classes", "#FBEFF6", PURPLE),
    ]
    gap = 3.5
    total = sum(w for w, _, _, _ in stages) + gap * (len(stages) - 1)
    ax.set_xlim(-1, total + 1)
    ax.set_ylim(0, 30)

    x = 0.0
    centres = []
    for width, text, face, edge in stages:
        ax.add_patch(
            FancyBboxPatch(
                (x, 9),
                width,
                12,
                boxstyle="round,pad=0.0,rounding_size=1.2",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.1,
            )
        )
        ax.text(x + width / 2, 15, text, ha="center", va="center", fontsize=6.2)
        centres.append(x + width / 2)
        if x > 0:
            ax.add_patch(
                FancyArrowPatch(
                    (x - gap + 0.2, 15),
                    (x - 0.5, 15),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    color="#555555",
                    linewidth=1.0,
                )
            )
        x += width + gap

    ax.text(centres[1], 6.0, "trained once, frozen", ha="center", fontsize=6.5, color=BLUE)
    ax.text(centres[3], 6.0, "no gradient reaches the backbone", ha="center", fontsize=6.5, color=GREEN)
    ax.text(
        centres[4],
        24.6,
        "trained on 887 real crops,\n$\\pm$ 171 synthetic rare crops",
        ha="center",
        fontsize=6.5,
        color=PURPLE,
    )
    save(fig, "fig_twostage_pipeline.pdf")


def sign_test(diffs):
    """Exact two-sided sign test on paired differences (as in scripts/analyze_grid.py without scipy)."""
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return None
    pos = sum(1 for d in nz if d > 0)
    tail = sum(math.comb(n, k) for k in range(0, min(pos, n - pos) + 1)) / 2**n
    return min(1.0, 2 * tail), pos, n


def derived_stats():
    """Print the paired sign tests on overall mask AP quoted in the report's grid table.

    summary.csv's `p_vs_e0` uses rare_mask_AP on the multiclass track, so the test on overall
    multiclass mask AP is recomputed here from the per-seed values in summary.json.
    """
    payload = json.loads((FINAL / "grid_r1" / "summary.json").read_text(encoding="utf-8"))
    by_key = {(a["track"], a["arm"]): a for a in payload["arms"]}
    print("\n# derived paired sign tests on overall mask AP (n=3 seeds)")
    print("# track        arm   mean     std      delta_vs_e0  seeds_pos  p")
    for track in ("multiclass", "agnostic"):
        base = by_key[(track, "e0")]["aggregate"]["overall_mask_AP"]["values"]
        for arm in ARM_ORDER:
            agg = by_key[(track, arm)]["aggregate"]["overall_mask_AP"]
            diffs = [a - b for a, b in zip(agg["values"], base)]
            res = sign_test(diffs)
            delta = sum(diffs) / len(diffs)
            seeds_pos = f"{res[1]}/{res[2]}" if res else "-"
            p = f"{res[0]:.4f}" if res else "n/a"
            print(
                f"{track:<12} {arm:<5} {agg['mean']:.4f}  {agg['std']:.4f}  {delta:+.4f}"
                f"      {seeds_pos}        {p}"
            )


def main():
    fig_class_distribution()
    fig_rare_support()
    fig_grid_ap()
    fig_cv_agnostic()
    fig_rare_macro()
    fig_sam3_vs_mrcnn()
    fig_dino_recall()
    fig_dino_twostage()
    fig_pipeline()
    derived_stats()


if __name__ == "__main__":
    main()
