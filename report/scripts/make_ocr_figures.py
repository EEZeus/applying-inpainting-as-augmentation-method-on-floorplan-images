#!/usr/bin/env python3
"""Figures for the OCR study and the follow-up experiments (Section 5c).

Inputs (fetched from the cluster into report/assets/final):
    ocr_study/summary.json            e0, e18, e19, e20 per seed
    followup/summary.json             e0, e19, e21-e24 per seed and per class
    ocr_study/examples/               label replacement crops, one plan + its OCR tokens

    cd report && python scripts/make_ocr_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "assets" / "final"
IMAGES = ROOT / "images"
EXAMPLES = FINAL / "ocr_study" / "examples"

BLACK = "#000000"
SKY = "#56B4E9"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREY = "#9E9E9E"
ORANGE = "#E69F00"
plt.rcParams.update(
    {
        "font.size": 8,
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


def load(name):
    return json.loads((FINAL / name / "summary.json").read_text(encoding="utf-8"))


def arms_by_id(summary):
    return {a["arm"]: a for a in summary["arms"]}


def seed_values(arm):
    return [arm["per_seed"][k] for k in sorted(arm["per_seed"], key=int)]


def fig_text_runs(ocr, follow):
    arms = arms_by_id(ocr)
    arms["e21"] = arms_by_id(follow)["e21"]
    order = [
        ("e0", "e0\nplain model"),
        ("e18", "e18\nplain model,\nreplaced labels"),
        ("e19", "e19\ntext branch"),
        ("e20", "e20\ntext branch,\nreplaced labels"),
        ("e21", "e21\ntext branch\nwith AdamW"),
    ]
    colors = {"e0": BLACK, "e18": GREY, "e19": SKY, "e20": BLUE, "e21": VERMILLION}
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    for x, (arm, _) in enumerate(order):
        values = seed_values(arms[arm])
        offsets = [(i - (len(values) - 1) / 2) * 0.025 for i in range(len(values))]
        ax.scatter([x + o for o in offsets], values, s=11, color=colors[arm], alpha=0.75, zorder=3)
        mean = sum(values) / len(values)
        ax.plot([x - 0.22, x + 0.22], [mean, mean], color=colors[arm], lw=1.6, zorder=4)
    ax.axhline(
        sum(seed_values(arms["e0"])) / len(seed_values(arms["e0"])), color=BLACK, lw=0.7, ls="--", zorder=2
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label for _, label in order])
    ax.set_ylabel("multiclass mask AP\n(last.pth, 21 held-out plans)")
    fig.savefig(IMAGES / "fig_text_runs.pdf")
    plt.close(fig)


def fig_dose(follow):
    arms = arms_by_id(follow)
    order = ["e0", "e22", "e23", "e24"]
    rooms = [arms[a]["converted_rooms"] for a in order]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for axis, key, label in (
        (left, "per_seed", "multiclass mask AP"),
        (right, "rare_by_seed", "rare macro mask AP"),
    ):
        means = []
        for arm, x in zip(order, rooms):
            values = [
                v for k, v in sorted(arms[arm][key].items(), key=lambda kv: int(kv[0])) if v is not None
            ]
            axis.scatter([x] * len(values), values, s=9, color=GREY, alpha=0.6, zorder=2)
            means.append(sum(values) / len(values))
        axis.plot(rooms, means, "-o", color=VERMILLION, ms=3.5, zorder=3)
        axis.set_xlabel("converted rare rooms in the training plans")
        axis.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(IMAGES / "fig_dose_response.pdf")
    plt.close(fig)


def fig_perclass(follow):
    arms = arms_by_id(follow)
    names = follow["category_names"]
    e0, e21 = arms["e0"]["per_class_mask_AP"], arms["e21"]["per_class_mask_AP"]
    classes = [c for c in sorted(e0, key=int) if c in e21 and int(c) <= 16]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    y = list(range(len(classes)))[::-1]
    ax.barh([v + 0.2 for v in y], [e0[c] for c in classes], height=0.38, color=GREY, label="e0 plain model")
    ax.barh(
        [v - 0.2 for v in y],
        [e21[c] for c in classes],
        height=0.38,
        color=VERMILLION,
        label="e21 text branch with AdamW",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} {names[c]}" for c in classes])
    ax.set_xlabel("per-class mask AP (mean over 12 seeds, 21 held-out plans)")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(IMAGES / "fig_text_perclass.pdf")
    plt.close(fig)


def fig_examples():
    examples = [
        ("t-ann21-to13-v0", "Büro", "Konferenz", "13 Versammlungsraum"),
        ("t-ann636-to16-v0", "Büro", "Putzraum", "16 Sanitärraum"),
        ("t-ann639-to14-v0", "Büro", "Schacht", "14 Schacht"),
        ("t-ann1525-to11-v0", "Lager", "Foyer", "11 Allg. Verkehrsfläche"),
    ]
    fig, axes = plt.subplots(2, len(examples), figsize=(7.4, 4.0))
    for column, (sample, old, new, target) in enumerate(examples):
        for row, suffix in enumerate(("_before", "")):
            axis = axes[row][column]
            axis.imshow(Image.open(EXAMPLES / f"{sample}{suffix}.png").convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[0][column].set_title(f"original: “{old}”", fontsize=7.5)
        axes[1][column].set_xlabel(f"edited: “{new}”\nclass {target}", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(IMAGES / "fig_label_replacement.png", dpi=220)
    plt.close(fig)


def fig_tokens():
    cache = json.loads((EXAMPLES / "original_tokens.json").read_text(encoding="utf-8"))["images"]
    name = (
        "Alstria__Feuerwehrplan__01.05.2015_-_Zeichnung_-_Haus_1_-_"
        "Feuerwehrplan_-_2015_-_OG_09_-_Vorabzug__0.jpg"
    )
    tokens = cache[name]["tokens"]
    x0, y0, x1, y1 = 300, 420, 1700, 1300
    image = Image.open(EXAMPLES / "plan_001.jpg").convert("RGB").crop((x0, y0, x1, y1))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.imshow(image)
    sys.path.insert(0, str(ROOT.parent / "floorplan-inpainting"))
    from src.ocr.text import is_label_text

    for token in tokens:
        bx0, by0, bx1, by1 = token["box"]
        if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
            continue
        label = is_label_text(token["text"])
        ax.add_patch(
            Rectangle(
                (bx0 - x0, by0 - y0),
                bx1 - bx0,
                by1 - by0,
                fill=False,
                lw=1.1,
                ls="-" if label else ":",
                edgecolor=BLUE if label else GREY,
            )
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(IMAGES / "fig_ocr_tokens.png", dpi=220)
    plt.close(fig)


def main():
    ocr, follow = load("ocr_study"), load("followup")
    fig_text_runs(ocr, follow)
    fig_dose(follow)
    fig_perclass(follow)
    fig_examples()
    fig_tokens()
    print("figures written to", IMAGES)


if __name__ == "__main__":
    main()
