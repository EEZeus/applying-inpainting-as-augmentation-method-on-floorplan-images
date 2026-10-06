#!/usr/bin/env python3
"""Multiclass follow-up (docs/preregistration_confirm_mc.md): analysis + report subsection.

Arms e0 / e1 / e2 on the multiclass track, 12 fresh seeds, endpoint multiclass
mask AP@[.50:.95] of last.pth on the 21 held-out plans.
H4: e2 > e0, then (only if H4 is supported) H5: e2 > e1. One-sided Welch,
alpha 0.05, and the seed x image bootstrap 95% CI must exclude 0.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pycocotools.coco import COCO  # noqa: E402

from analyze_confirm import (ANNOTATIONS, CONFIRM_SEEDS, arm_values, bootstrap_table,  # noqa: E402
                             collect_runs, describe, project_subsets, two_level_ci)
from make_confirm_report import BLACK, GREY, SKY, fmt, fmt_ci, fmt_p, mean_sd, plt, tex  # noqa: E402
from src.data.coco_dataset import read_split  # noqa: E402
from src.evaluation.confirm_stats import hedges_g, permutation_test, welch  # noqa: E402
from src.utils.config import resolve_path  # noqa: E402

ARMS = [("e0", "Original (no aug)"), ("e1", "Rare-weighted sampling"), ("e2", "Strategy A (inpainting)")]
COLORS = {"e0": BLACK, "e1": GREY, "e2": SKY}
SESOI = 0.01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-root", default="outputs/confirm_mc")
    parser.add_argument("--out", default="outputs/final/confirm_mc")
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    heldout = read_split(resolve_path("data/splits_heldout21/test.txt"))
    test11 = read_split(resolve_path("data/splits/test.txt"))
    root, out = resolve_path(args.confirm_root), resolve_path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    coco_gt = COCO(str(resolve_path(ANNOTATIONS["multiclass"])))
    runs = collect_runs(root, "multiclass", ARMS, CONFIRM_SEEDS, coco_gt, heldout)
    boot = bootstrap_table(runs, ARMS, np.array(heldout), args.boot, rng)
    projects = project_subsets(heldout)
    values = {arm: arm_values(runs, arm, heldout)[0] for arm, _ in ARMS}

    arms = []
    for arm, label in ARMS:
        seeds = arm_values(runs, arm, heldout)[1]
        if not seeds:
            continue
        rare = [runs[(arm, s)].metrics.get("rare_mask_AP") for s in seeds]
        arms.append({"arm": arm, "label": label, "seeds": seeds, "heldout21": describe(values[arm]),
                     "heldout21_AP50": describe(arm_values(runs, arm, heldout, "AP50")[0]),
                     "test11": describe(arm_values(runs, arm, test11)[0]),
                     "rare_mask_AP": describe([r for r in rare if r is not None]),
                     "per_seed": dict(zip(seeds, values[arm]))})

    def contrast(control, treatment):
        a, b = values.get(control, []), values.get(treatment, [])
        if not a or not b:
            return None
        test = welch(a, b)
        ci = two_level_ci(boot, runs, control, treatment, rng)
        return {"control": control, "treatment": treatment, "n_control": len(a), "n_treatment": len(b),
                "delta": test["delta"], "welch_p_one_sided": test["p_one_sided"],
                "permutation_p_one_sided": permutation_test(a, b, args.permutations),
                "ci95_seed_x_image": ci, "hedges_g": hedges_g(a, b),
                "ci_excludes_zero": bool(ci and ci[0] > 0),
                "delta_test11": statistics.fmean(arm_values(runs, treatment, test11)[0])
                - statistics.fmean(arm_values(runs, control, test11)[0]),
                "delta_AP50": statistics.fmean(arm_values(runs, treatment, heldout, "AP50")[0])
                - statistics.fmean(arm_values(runs, control, heldout, "AP50")[0]),
                "delta_per_project": {p: statistics.fmean(arm_values(runs, treatment, ids)[0])
                                      - statistics.fmean(arm_values(runs, control, ids)[0])
                                      for p, ids in projects.items()}}

    tests = {"H4": contrast("e0", "e2"), "H5": contrast("e1", "e2"), "e1_vs_e0": contrast("e0", "e1")}
    h4 = tests["H4"]
    if h4:
        h4["supported"] = bool(h4["welch_p_one_sided"] is not None and h4["welch_p_one_sided"] < 0.05
                               and h4["ci_excludes_zero"])
    if tests["H5"]:
        tests["H5"]["tested"] = bool(h4 and h4["supported"])
        tests["H5"]["supported"] = bool(tests["H5"]["tested"] and tests["H5"]["welch_p_one_sided"] is not None
                                        and tests["H5"]["welch_p_one_sided"] < 0.05
                                        and tests["H5"]["ci_excludes_zero"])
    record = root / "preregistration_record.txt"
    summary = {"endpoint": "multiclass mask AP@[.50:.95], last.pth, 21 held-out plans", "arms": arms,
               "tests": tests, "preregistration_record": record.read_text() if record.is_file() else None}
    (out / "summary_mc.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    # ---- figure
    if arms:
        fig, ax = plt.subplots(figsize=(5.0, 2.6))
        for position, arm in enumerate(arms):
            vals = list(arm["per_seed"].values())
            offsets = [(i - (len(vals) - 1) / 2) * 0.03 for i in range(len(vals))]
            ax.scatter([position + o for o in offsets], vals, s=12, color=COLORS[arm["arm"]], alpha=0.75, zorder=3)
            sd = arm["heldout21"]["sd"] or 0.0
            ax.errorbar(position + 0.28, arm["heldout21"]["mean"], yerr=sd, fmt="D", ms=4, capsize=3,
                        color=COLORS[arm["arm"]], zorder=4)
        ax.axhline(arms[0]["heldout21"]["mean"], color=BLACK, lw=0.8, ls="--")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([f"{a['arm']}\n{a['label']}" for a in arms])
        ax.set_ylabel("multiclass mask AP\n(last.pth, 21 held-out plans)")
        fig.tight_layout()
        fig.savefig(out / "figures" / "fig_confirm_mc_runs.pdf")
        plt.close(fig)

    # ---- text
    md, tx = ["## Multiclass follow-up: Strategy A versus oversampling\n"], [
        "\\subsection{Multiclass follow-up: Strategy A versus oversampling}\n\\label{sec:confirmmc}\n"]
    intro = ("The exploratory re-score left one open question: Strategy A (e2) and rare-weighted sampling (e1) "
             "both improved multiclass AP, and e1 adds no generated pixels. A second pre-registered study "
             "(docs/preregistration_confirm_mc.md) trained e0, e1 and e2 on the multiclass track with 12 fresh "
             "seeds each and scored last.pth on the 21 held-out plans. H4 tests e2 > e0; H5, tested only if H4 "
             "is supported, tests e2 > e1, i.e. whether inpainted appearance adds anything beyond oversampling.")
    md.append(intro + "\n")
    tx.append(tex(intro).replace("docs/preregistration\\_confirm\\_mc.md", "\\code{docs/preregistration_confirm_mc.md}")
              + "\n")
    md.append("| Arm | Method | n | Mask AP (21 plans) | AP50 | AP (11 test plans) | Rare macro AP |")
    md.append("|---|---|---|---|---|---|---|")
    tx.append("\\begin{table}[H]\n\\centering\n\\caption{Multiclass follow-up: mask AP of \\code{last.pth}, mean "
              "$\\pm$ s.d. over 12 fresh seeds.}\n\\label{tab:confirmmc}\n\\small\n\\begin{tabular}{@{}llrrrrr@{}}\n"
              "\\toprule\nArm & Method & $n$ & AP (21) & AP50 (21) & AP (11 test) & Rare AP \\\\\n\\midrule")
    for arm in arms:
        cells = [mean_sd(arm["heldout21"]), mean_sd(arm["heldout21_AP50"]), mean_sd(arm["test11"]),
                 mean_sd(arm["rare_mask_AP"], 4)]
        md.append(f"| {arm['arm']} | {arm['label']} | {arm['heldout21']['n']} | " + " | ".join(cells) + " |")
        tx.append(f"{arm['arm']} & {tex(arm['label'])} & {arm['heldout21']['n']} & " + " & ".join(tex(c) for c in cells)
                  + " \\\\")
    md.append("")
    tx.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    md.append("| Contrast | Δ AP | 95% CI | Welch p | Perm. p | g | Supported |")
    md.append("|---|---|---|---|---|---|---|")
    tx.append("\\begin{table}[H]\n\\centering\n\\caption{Multiclass follow-up contrasts (one-sided $p$; CI resamples "
              "seeds and the 21 held-out plans).}\n\\label{tab:confirmmctests}\n\\small\n\\begin{tabular}{@{}lrrrrrl@{}}\n"
              "\\toprule\nContrast & $\\Delta$ & 95\\% CI & Welch $p$ & Perm.\\ $p$ & $g$ & Supported \\\\\n\\midrule")
    labels = {"H4": "H4: e2 - e0", "H5": "H5: e2 - e1", "e1_vs_e0": "e1 - e0 (descr.)"}
    for name, c in tests.items():
        if not c:
            continue
        supported = ("yes" if c.get("supported") else "no") if name in ("H4", "H5") else "--"
        if name == "H5" and not c.get("tested"):
            supported = "not tested"
        row = [fmt(c["delta"], 3, True), fmt_ci(c["ci95_seed_x_image"]), fmt_p(c["welch_p_one_sided"]),
               fmt_p(c["permutation_p_one_sided"]), fmt(c["hedges_g"], 2), supported]
        md.append(f"| {labels[name]} | " + " | ".join(row) + " |")
        tx.append(f"{tex(labels[name])} & " + " & ".join(tex(x) for x in row) + " \\\\")
    md.append("")
    tx.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    if arms:
        tx.append("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.75\\textwidth]{fig_confirm_mc_runs.pdf}\n"
                  "\\caption{Multiclass follow-up runs: one dot per seed, diamond = mean $\\pm$ s.d.}\n"
                  "\\label{fig:confirmmc}\n\\end{figure}\n")

    sentences = []
    h5, e1 = tests.get("H5"), tests.get("e1_vs_e0")
    if h4:
        if h4["supported"]:
            sentences.append(f"H4 is supported: Strategy A raises multiclass mask AP by {h4['delta']:+.3f} "
                             f"(95% CI {fmt_ci(h4['ci95_seed_x_image'])}, one-sided p = {fmt_p(h4['welch_p_one_sided'])}).")
        else:
            sentences.append(f"H4 is not supported: Strategy A changes multiclass mask AP by {h4['delta']:+.3f} "
                             f"(95% CI {fmt_ci(h4['ci95_seed_x_image'])}, one-sided p = {fmt_p(h4['welch_p_one_sided'])}).")
    if h5:
        if h5.get("supported"):
            sentences.append(f"H5 is supported: e2 exceeds rare-weighted sampling by {h5['delta']:+.3f} "
                             f"(95% CI {fmt_ci(h5['ci95_seed_x_image'])}), so the inpainted appearance contributes "
                             "beyond oversampling.")
        elif h5.get("tested"):
            sentences.append(f"H5 is not supported: e2 differs from rare-weighted sampling by {h5['delta']:+.3f} "
                             f"(95% CI {fmt_ci(h5['ci95_seed_x_image'])}), so the gain of Strategy A cannot be "
                             "attributed to the inpainted pixels rather than to oversampling rare-room plans.")
        else:
            sentences.append(f"H5 was not tested (fixed sequence). Descriptively, e2 - e1 = {h5['delta']:+.3f} "
                             f"(95% CI {fmt_ci(h5['ci95_seed_x_image'])}).")
    if e1:
        sentences.append(f"Descriptively, rare-weighted sampling alone gives {e1['delta']:+.3f} "
                         f"(95% CI {fmt_ci(e1['ci95_seed_x_image'])}, one-sided p = {fmt_p(e1['welch_p_one_sided'])}).")
    rare_means = {a["arm"]: a["rare_mask_AP"]["mean"] for a in arms}
    if rare_means:
        text = "Rare-class macro mask AP: " + ", ".join(f"{arm} {fmt(v, 4)}" for arm, v in rare_means.items()) + "."
        if all(v is not None and v < 0.01 for v in rare_means.values()):
            text += (" It stays near zero in every arm, so any change in overall AP comes from the frequent "
                     "classes, not from the rare classes the augmentation targets.")
        sentences.append(text)
    for line in sentences:
        md.append(line + "\n")
        tx.append(tex(line) + "\n")
    (out / "confirm_mc_section.tex").write_text("\n".join(tx), encoding="utf-8")
    (out / "report_mc.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(sentences))


if __name__ == "__main__":
    main()
