#!/usr/bin/env python3
"""Follow-up analysis (docs/preregistration_followup.md, exploratory).

Multiclass track, seeds 1001-1012, endpoint mask AP@[.50:.95] of last.pth on
the 21 held-out plans.
  X1 dose-response: e0 (reused) < e22 (20/class) < e23 (50) < e24 (100)
     F1 trend: one-sided t-test on the per-seed OLS slope of AP on the number
        of converted rooms; F2 each dose vs e0 (paired t, Holm over 3).
  X4 e21 (text branch with own AdamW, containment pooling):
     F3 e21 vs e0 and e21 vs e19 (paired t, Holm over 2).
  X2 balanced two-stage text head vs the original one (paired by seed).
  X3 SAM 3 proposals + text head vs SAM 3 multiclass (11 test plans, 1 seed).
Support rule as before: Holm rejects AND the paired seed x image bootstrap
95% CI excludes 0. Per-class AP of the frequent classes is reported per arm.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pycocotools.coco import COCO  # noqa: E402

from analyze_confirm import ANNOTATIONS, CONFIRM_SEEDS, TAG, collect_runs, describe  # noqa: E402
from analyze_ocr_study import ap_table, cluster_resamples, image_resamples, paired_ci, per_seed  # noqa: E402
from make_confirm_report import BLACK, GREY, SKY, VERMILLION, fmt, fmt_ci, fmt_p, mean_sd, plt, tex  # noqa: E402
from src.data.coco_dataset import read_split  # noqa: E402
from src.evaluation.confirm_stats import holm, paired_t  # noqa: E402
from src.utils.config import resolve_path  # noqa: E402

DOSE_ARMS = [("e22", "b20"), ("e23", "b50"), ("e24", "b100")]
LABELS = {"e0": "Plain, original plans", "e19": "Text-aware (OCR study)", "e21": "Text-aware, AdamW text branch",
          "e22": "Copy-paste, 20 per rare class", "e23": "Copy-paste, 50 per rare class",
          "e24": "Copy-paste, 100 per rare class"}
SESOI = 0.01


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--followup-root", default="outputs/followup")
    parser.add_argument("--ocr-root", default="outputs/ocr_study")
    parser.add_argument("--e0-root", default="outputs/confirm_mc")
    parser.add_argument("--balance-root", default="data/augmented_c_balance")
    parser.add_argument("--diag-base", default="outputs/final/ocr_study/diagnostics/diagnostics.json")
    parser.add_argument("--diag-balanced", default="outputs/final/followup/diagnostics_balanced/diagnostics.json")
    parser.add_argument("--out", default="outputs/final/followup")
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(CONFIRM_SEEDS))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    heldout = read_split(resolve_path("data/splits_heldout21/test.txt"))
    out = resolve_path(args.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    coco_gt = COCO(str(resolve_path(ANNOTATIONS["multiclass"])))
    roots = {"e0": args.e0_root, "e19": args.ocr_root, "e21": args.followup_root,
             "e22": args.followup_root, "e23": args.followup_root, "e24": args.followup_root}
    runs = {}
    for arm, root in roots.items():
        runs.update(collect_runs(resolve_path(root), "multiclass", [(arm, LABELS[arm])], args.seeds, coco_gt, heldout))
    image_boot = ap_table(runs, image_resamples(heldout, args.boot, rng))
    cluster_boot = ap_table(runs, cluster_resamples(heldout, args.boot, rng))

    doses = {"e0": 0}
    balance = {}
    for arm, tag in DOSE_ARMS:
        path = resolve_path(args.balance_root) / tag / "balance_summary.json"
        if path.is_file():
            balance[arm] = read_json(path)
            doses[arm] = balance[arm]["converted_rooms"]

    names = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}
    arms = []
    for arm in roots:
        values = per_seed(runs, arm, heldout)
        if not values:
            continue
        seeds = sorted(values)
        per_class = defaultdict(list)
        for s in seeds:
            root = resolve_path(roots[arm]) / "multiclass" / arm / f"seed_{s}" / TAG / "result.json"
            for row in read_json(root)["per_class"]:
                if row["mask_AP"] is not None:
                    per_class[int(row["category_id"])].append(row["mask_AP"])
        arms.append({"arm": arm, "label": LABELS[arm], "seeds": seeds, "per_seed": values,
                     "heldout21": describe([values[s] for s in seeds]),
                     "heldout21_AP50": describe(list(per_seed(runs, arm, heldout, "AP50").values())),
                     "rare_mask_AP": describe([runs[(arm, s)].metrics.get("rare_mask_AP") for s in seeds
                                               if runs[(arm, s)].metrics.get("rare_mask_AP") is not None]),
                     "rare_by_seed": {s: runs[(arm, s)].metrics.get("rare_mask_AP") for s in seeds},
                     "per_class_mask_AP": {str(c): statistics.fmean(v) for c, v in sorted(per_class.items())},
                     "converted_rooms": doses.get(arm)})
    by_arm = {a["arm"]: a for a in arms}

    def contrast(control, treatment):
        if control not in by_arm or treatment not in by_arm:
            return None
        common = sorted(set(by_arm[control]["seeds"]) & set(by_arm[treatment]["seeds"]))
        test = paired_t([by_arm[control]["per_seed"][s] for s in common], [by_arm[treatment]["per_seed"][s] for s in common])
        rare = [(by_arm[control]["rare_by_seed"][s], by_arm[treatment]["rare_by_seed"][s]) for s in common
                if by_arm[control]["rare_by_seed"][s] is not None and by_arm[treatment]["rare_by_seed"][s] is not None]
        ci = paired_ci(image_boot, control, treatment, rng)
        return {"control": control, "treatment": treatment, "n_pairs": len(common), "delta": test["delta"],
                "p_one_sided": test["p_one_sided"], "dz": test.get("dz"), "ci95_paired_seed_x_image": ci,
                "ci95_paired_seed_x_project_cluster": paired_ci(cluster_boot, control, treatment, rng),
                "ci_excludes_zero": bool(ci and ci[0] > 0),
                "rare_delta": statistics.fmean(t - c for c, t in rare) if rare else None}

    families = {"F2": {f"{arm}_vs_e0": contrast("e0", arm) for arm, _ in DOSE_ARMS},
                "F3": {"e21_vs_e0": contrast("e0", "e21"), "e21_vs_e19": contrast("e19", "e21")}}
    for family in families.values():
        adjusted = holm({k: c["p_one_sided"] for k, c in family.items() if c})
        for key, c in family.items():
            if c:
                c["p_holm"], c["holm_reject"] = adjusted[key]
                c["supported"] = bool(c["holm_reject"] and c["ci_excludes_zero"])

    trend = None
    dose_arms = [a for a in ["e0"] + [x for x, _ in DOSE_ARMS] if a in by_arm and a in doses]
    if len(dose_arms) >= 3:
        common = sorted(set.intersection(*(set(by_arm[a]["seeds"]) for a in dose_arms)))
        x = np.array([doses[a] for a in dose_arms], dtype=float)
        slopes = [float(np.polyfit(x, [by_arm[a]["per_seed"][s] for a in dose_arms], 1)[0]) for s in common]
        test = paired_t([0.0] * len(slopes), slopes)
        trend = {"arms": dose_arms, "converted_rooms": x.tolist(), "n_seeds": len(slopes),
                 "mean_slope_per_100_rooms": 100 * statistics.fmean(slopes), "t": test["t"],
                 "p_one_sided": test["p_one_sided"], "supported": bool(test["p_one_sided"] is not None
                                                                         and test["p_one_sided"] < 0.05)}

    x2 = None
    if resolve_path(args.diag_base).is_file() and resolve_path(args.diag_balanced).is_file():
        base = {r["seed"]: r for r in read_json(resolve_path(args.diag_base))["stage2"]["runs"]}
        bal_summary = read_json(resolve_path(args.diag_balanced))
        bal = {r["seed"]: r for r in bal_summary["stage2"]["runs"]}
        common = sorted(set(base) & set(bal))
        if len(common) >= 2:
            x2 = {"n": len(common), "base_mean": statistics.fmean(base[s]["mask_AP"] for s in common),
                  "balanced_mean": statistics.fmean(bal[s]["mask_AP"] for s in common),
                  "base_rare_mean": statistics.fmean(base[s]["rare_mask_AP"] or 0 for s in common),
                  "balanced_rare_mean": statistics.fmean(bal[s]["rare_mask_AP"] or 0 for s in common),
                  "test": paired_t([base[s]["mask_AP"] for s in common], [bal[s]["mask_AP"] for s in common]),
                  "rare_test": paired_t([base[s]["rare_mask_AP"] or 0 for s in common],
                                        [bal[s]["rare_mask_AP"] or 0 for s in common]),
                  "training_rooms": bal_summary.get("training_rooms"),
                  "stage1_base": read_json(resolve_path(args.diag_base))["stage1"].get("text"),
                  "stage1_balanced": bal_summary["stage1"].get("text"),
                  "sam3": bal_summary.get("stage3_sam3_test11")}

    summary = {"endpoint": "multiclass mask AP@[.50:.95], last.pth, 21 held-out plans (exploratory)",
               "arms": arms, "families": families, "dose_trend": trend, "balance": balance, "x2_x3": x2,
               "category_names": {str(k): v for k, v in names.items()}}
    record = resolve_path(args.followup_root) / "preregistration_record.txt"
    summary["preregistration_record"] = record.read_text() if record.is_file() else None
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    # ---- dose-response figure
    if trend:
        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        for arm in trend["arms"]:
            vals = list(by_arm[arm]["per_seed"].values())
            ax.scatter([doses[arm]] * len(vals), vals, s=10, color=GREY, alpha=0.6, zorder=2)
        means = [by_arm[a]["heldout21"]["mean"] for a in trend["arms"]]
        ax.plot(trend["converted_rooms"], means, "-D", color=VERMILLION, ms=4, zorder=3)
        ax.set_xlabel("converted rare rooms in the training data")
        ax.set_ylabel("multiclass mask AP\\n(last.pth, 21 plans)")
        fig.tight_layout(); fig.savefig(out / "figures" / "fig_dose_response.pdf"); plt.close(fig)

    # ---- report text
    md = ["## Follow-up (exploratory): class balance, text branch optimisation, two-stage OCR\n",
          "| Arm | Method | n | Converted rooms | Mask AP (21) | AP50 | Rare macro AP |", "|---|---|---|---|---|---|---|"]
    tx = ["\\subsection{Follow-up experiments (exploratory)}\n\\label{sec:followup}\n",
          "\\begin{table}[H]\n\\centering\n\\caption{Follow-up arms: multiclass mask AP of \\code{last.pth} on the 21 "
          "held-out plans, mean $\\pm$ s.d.\\ over seeds 1001--1012.}\n\\label{tab:followup}\n\\small\n"
          "\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\nArm & Method & $n$ & Converted & AP (21) & AP50 & Rare AP \\\\\n\\midrule"]
    for a in arms:
        conv = "--" if a["converted_rooms"] is None else str(a["converted_rooms"])
        cells = [mean_sd(a["heldout21"]), mean_sd(a["heldout21_AP50"]), mean_sd(a["rare_mask_AP"], 4)]
        md.append(f"| {a['arm']} | {a['label']} | {a['heldout21']['n']} | {conv} | " + " | ".join(cells) + " |")
        tx.append(f"{a['arm']} & {tex(a['label'])} & {a['heldout21']['n']} & {conv} & "
                  + " & ".join(tex(c) for c in cells) + " \\\\")
    md.append(""); tx.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    md += ["| Contrast | Δ AP | 95% CI (seed × image) | 95% CI (seed × project) | paired p | Holm p | Δ rare AP | Supported |",
           "|---|---|---|---|---|---|---|---|"]
    tx.append("\\begin{table}[H]\n\\centering\n\\caption{Follow-up contrasts (one-sided paired $t$; Holm within "
              "each family).}\n\\label{tab:followuptests}\n\\small\n\\begin{tabular}{@{}lrrrrl@{}}\n\\toprule\n"
              "Contrast & $\\Delta$ & 95\\% CI & paired $p$ & Holm $p$ & Supported \\\\\n\\midrule")
    for family in families.values():
        for key, c in family.items():
            if not c:
                continue
            label = f"{c['treatment']} - {c['control']}"
            row = [fmt(c["delta"], 3, True), fmt_ci(c["ci95_paired_seed_x_image"]),
                   fmt_ci(c["ci95_paired_seed_x_project_cluster"]), fmt_p(c["p_one_sided"]), fmt_p(c["p_holm"]),
                   fmt(c["rare_delta"], 4, True), "yes" if c["supported"] else "no"]
            md.append(f"| {label} | " + " | ".join(row) + " |")
            tx.append(f"{tex(label)} & " + " & ".join(tex(x) for x in (row[0], row[1], row[3], row[4], row[6])) + " \\\\")
    md.append(""); tx.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    if trend:
        tx.append("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.6\\textwidth]{fig_dose_response.pdf}\n"
                  "\\caption{Dose-response of balanced copy-paste: one grey dot per seed, line = mean.}\n"
                  "\\label{fig:dose}\n\\end{figure}\n")
    # per-class table for frequent and rare classes
    class_ids = sorted({int(c) for a in arms for c in a["per_class_mask_AP"]})
    md += ["Per-class mask AP (mean over seeds):", "",
           "| Class | " + " | ".join(a["arm"] for a in arms) + " |", "|---|" + "---|" * len(arms)]
    for c in class_ids:
        md.append(f"| {names[c]} | " + " | ".join(fmt(a["per_class_mask_AP"].get(str(c)), 3) for a in arms) + " |")
    md.append("")
    sentences = []
    if trend:
        sentences.append(f"Dose-response: the mean per-seed slope is {trend['mean_slope_per_100_rooms']:+.4f} AP per 100 "
                         f"converted rooms (one-sided p = {fmt_p(trend['p_one_sided'])}, n = {trend['n_seeds']}); the trend "
                         f"{'is' if trend['supported'] else 'is not'} supported.")
    for family in families.values():
        for c in family.values():
            if c:
                sentences.append(f"{c['treatment']} - {c['control']}: {c['delta']:+.3f} AP (95% CI "
                                 f"{fmt_ci(c['ci95_paired_seed_x_image'])}, Holm p = {fmt_p(c['p_holm'])}); "
                                 f"{'supported' if c['supported'] else 'not supported'}.")
    if x2:
        sentences.append(f"Two-stage OCR classification with the text head trained on real plus "
                         f"{(x2['training_rooms'] or {}).get('synthetic_converted', 0)} converted synthetic rare rooms: "
                         f"{x2['balanced_mean']:.3f} AP vs {x2['base_mean']:.3f} (paired p = {fmt_p(x2['test']['p_one_sided'])}); "
                         f"rare AP {x2['balanced_rare_mean']:.3f} vs {x2['base_rare_mean']:.3f}.")
        if x2.get("sam3"):
            s3 = x2["sam3"]
            two = s3.get("two_stage_sam3_agnostic_plus_text", {}); mc = s3.get("sam3_multiclass", {})
            sentences.append(f"On the 11 test plans, SAM 3 class-agnostic masks labelled by the OCR text head reach "
                             f"{fmt(two.get('mask_AP'), 3)} multiclass mask AP (rare {fmt(two.get('rare_mask_AP'), 3)}) "
                             f"against {fmt(mc.get('mask_AP'), 3)} (rare {fmt(mc.get('rare_mask_AP'), 3)}) for SAM 3 "
                             f"multiclass (single SAM 3 seed; descriptive).")
    for line in sentences:
        md.append(line + "\n"); tx.append(tex(line) + "\n")
    (out / "followup_section.tex").write_text("\n".join(tx), encoding="utf-8")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
