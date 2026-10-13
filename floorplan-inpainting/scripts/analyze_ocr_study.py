#!/usr/bin/env python3
"""OCR study (docs/preregistration_ocr.md): pre-registered analysis and report section.

2 x 2 design on the multiclass track, 12 seeds (1001-1012), endpoint multiclass
mask AP@[.50:.95] of last.pth on the 21 held-out plans:

              original plans        + label-rewrite inpainting
  plain       e0 (multiclass follow-up runs, reused)   e18
  text-aware  e19                   e20

H6 e19 > e0, H7 e20 > e19, H8 e18 > e0. One-sided paired t-test over seeds
(arms that share a seed share the per-epoch plan order), Holm over {H6, H7, H8}
at alpha .05; a hypothesis is supported only if Holm rejects AND the paired
seed x image bootstrap 95% CI of the difference excludes 0. Everything else
(rare AP, AP50, interaction, project-cluster bootstrap, Welch) is descriptive.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pycocotools.coco import COCO  # noqa: E402

from analyze_confirm import ANNOTATIONS, CONFIRM_SEEDS, collect_runs, describe, project_subsets  # noqa: E402
from make_confirm_report import BLACK, GREY, SKY, VERMILLION, fmt, fmt_ci, fmt_p, mean_sd, plt, tex  # noqa: E402
from src.data.coco_dataset import read_split  # noqa: E402
from src.evaluation.confirm_stats import hedges_g, holm, paired_t, percentile_ci, welch  # noqa: E402
from src.utils.config import resolve_path  # noqa: E402

ARMS = [("e0", "Plain, original plans"), ("e18", "Plain + label-rewrite inpainting"),
        ("e19", "Text-aware, original plans"), ("e20", "Text-aware + label-rewrite inpainting")]
COLORS = {"e0": BLACK, "e18": GREY, "e19": SKY, "e20": VERMILLION}
HYPOTHESES = {"H6": ("e0", "e19"), "H7": ("e19", "e20"), "H8": ("e0", "e18")}
SESOI = 0.01


def image_resamples(heldout, n_boot, rng):
    pool = np.asarray(heldout)
    return [rng.choice(pool, size=len(pool), replace=True) for _ in range(n_boot)]


def cluster_resamples(heldout, n_boot, rng):
    """Resample projects (buildings), then images within each drawn project."""
    groups = list(project_subsets(heldout).values())
    samples = []
    for _ in range(n_boot):
        drawn = [groups[i] for i in rng.integers(len(groups), size=len(groups))]
        samples.append(np.concatenate([rng.choice(g, size=len(g), replace=True) for g in drawn]))
    return samples


def ap_table(runs, resamples):
    """AP of every run on every resample, computed once: (arm, seed) -> array."""
    return {key: np.array([run.ap(sample)["AP"] for sample in resamples]) for key, run in runs.items()}


def paired_ci(table, control, treatment, rng, level=0.95):
    seeds = sorted({s for a, s in table if a == control} & {s for a, s in table if a == treatment})
    if len(seeds) < 2:
        return None
    ctrl = np.stack([table[(control, s)] for s in seeds])
    trt = np.stack([table[(treatment, s)] for s in seeds])
    diffs = np.empty(ctrl.shape[1])
    for b in range(ctrl.shape[1]):
        index = rng.integers(len(seeds), size=len(seeds))
        diffs[b] = (trt[index, b] - ctrl[index, b]).mean()
    return percentile_ci(diffs, level)


def per_seed(runs, arm, image_ids, metric="AP"):
    return {s: runs[(a, s)].ap(image_ids)[metric] for a, s in runs if a == arm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", default="outputs/ocr_study")
    parser.add_argument("--e0-root", default="outputs/confirm_mc")
    parser.add_argument("--out", default="outputs/final/ocr_study")
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(CONFIRM_SEEDS))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    heldout = read_split(resolve_path("data/splits_heldout21/test.txt"))
    test11 = read_split(resolve_path("data/splits/test.txt"))
    out = resolve_path(args.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    coco_gt = COCO(str(resolve_path(ANNOTATIONS["multiclass"])))
    runs = collect_runs(resolve_path(args.e0_root), "multiclass", ARMS[:1], args.seeds, coco_gt, heldout)
    runs.update(collect_runs(resolve_path(args.study_root), "multiclass", ARMS[1:], args.seeds, coco_gt, heldout))
    image_boot = ap_table(runs, image_resamples(heldout, args.boot, rng))
    cluster_boot = ap_table(runs, cluster_resamples(heldout, args.boot, rng))
    projects = project_subsets(heldout)

    arms = []
    for arm, label in ARMS:
        values = per_seed(runs, arm, heldout)
        if not values:
            continue
        seeds = sorted(values)
        rare = [runs[(arm, s)].metrics.get("rare_mask_AP") for s in seeds]
        exposure = None
        if arm in ("e18", "e20"):
            totals = []
            for s in seeds:
                path = resolve_path(args.study_root) / "multiclass" / arm / f"seed_{s}" / "class_exposure_by_epoch.csv"
                if path.is_file():
                    with path.open(encoding="utf-8") as handle:
                        totals.append(sum(int(r.get("synthetic_TEXT_occurrences_seen", 0) or 0)
                                          for r in csv.DictReader(handle)))
            exposure = describe(totals)
        arms.append({"arm": arm, "label": label, "seeds": seeds,
                     "heldout21": describe([values[s] for s in seeds]),
                     "heldout21_AP50": describe(list(per_seed(runs, arm, heldout, "AP50").values())),
                     "test11": describe(list(per_seed(runs, arm, test11).values())),
                     "rare_mask_AP": describe([r for r in rare if r is not None]),
                     "rare_by_seed": dict(zip(seeds, rare)), "per_seed": values,
                     "converted_room_exposure_per_run": exposure})
    by_arm = {a["arm"]: a for a in arms}

    def contrast(control, treatment):
        if control not in by_arm or treatment not in by_arm:
            return None
        common = sorted(set(by_arm[control]["seeds"]) & set(by_arm[treatment]["seeds"]))
        a = [by_arm[control]["per_seed"][s] for s in common]
        b = [by_arm[treatment]["per_seed"][s] for s in common]
        test = paired_t(a, b)
        rare_pairs = [(by_arm[control]["rare_by_seed"][s], by_arm[treatment]["rare_by_seed"][s]) for s in common
                      if by_arm[control]["rare_by_seed"][s] is not None
                      and by_arm[treatment]["rare_by_seed"][s] is not None]
        ci = paired_ci(image_boot, control, treatment, rng)
        return {"control": control, "treatment": treatment, "n_pairs": len(common), "delta": test["delta"],
                "paired_t": test["t"], "df": test["df"], "p_one_sided": test["p_one_sided"], "dz": test.get("dz"),
                "ci95_paired_seed_x_image": ci, "ci_excludes_zero": bool(ci and ci[0] > 0),
                "ci95_paired_seed_x_project_cluster": paired_ci(cluster_boot, control, treatment, rng),
                "welch_p_one_sided": welch(a, b)["p_one_sided"], "hedges_g": hedges_g(a, b),
                "delta_AP50": statistics.fmean(per_seed(runs, treatment, heldout, "AP50")[s] for s in common)
                - statistics.fmean(per_seed(runs, control, heldout, "AP50")[s] for s in common),
                "delta_test11": statistics.fmean(per_seed(runs, treatment, test11)[s] for s in common)
                - statistics.fmean(per_seed(runs, control, test11)[s] for s in common),
                "delta_per_project": {p: statistics.fmean(per_seed(runs, treatment, ids)[s] for s in common)
                                      - statistics.fmean(per_seed(runs, control, ids)[s] for s in common)
                                      for p, ids in projects.items()},
                "rare_delta": (statistics.fmean(t - c for c, t in rare_pairs) if rare_pairs else None),
                "rare_paired": paired_t(*zip(*rare_pairs)) if len(rare_pairs) >= 2 else None}

    tests = {name: contrast(*pair) for name, pair in HYPOTHESES.items()}
    adjusted = holm({name: c["p_one_sided"] for name, c in tests.items() if c})
    for name, c in tests.items():
        if c:
            c["p_holm"], c["holm_reject"] = adjusted[name]
            c["supported"] = bool(c["holm_reject"] and c["ci_excludes_zero"])
    descriptive = {"e20_vs_e0": contrast("e0", "e20"), "e20_vs_e18": contrast("e18", "e20")}
    if all(k in by_arm for k in ("e0", "e18", "e19", "e20")):
        common = sorted(set.intersection(*(set(by_arm[k]["seeds"]) for k in ("e0", "e18", "e19", "e20"))))
        inter = [(by_arm["e20"]["per_seed"][s] - by_arm["e19"]["per_seed"][s])
                 - (by_arm["e18"]["per_seed"][s] - by_arm["e0"]["per_seed"][s]) for s in common]
        descriptive["interaction"] = {"n": len(common), "mean": statistics.fmean(inter),
                                      **paired_t([0.0] * len(inter), inter)}
    extra = {}
    for name, path in (("generation", "data/augmented_text/relabel/manifest.json"),
                       ("diagnostics", "outputs/final/ocr_study/diagnostics/diagnostics.json")):
        if resolve_path(path).is_file():
            extra[name] = json.loads(resolve_path(path).read_text())
    record = resolve_path(args.study_root) / "preregistration_record.txt"
    summary = {"endpoint": "multiclass mask AP@[.50:.95], last.pth, 21 held-out plans", "arms": arms,
               "tests": tests, "descriptive": descriptive, "sesoi": SESOI,
               "preregistration_record": record.read_text() if record.is_file() else None, **extra}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    # ---- figure: one dot per seed, mean +- s.d.
    if arms:
        fig, ax = plt.subplots(figsize=(6.0, 2.8))
        for position, arm in enumerate(arms):
            vals = [arm["per_seed"][s] for s in arm["seeds"]]
            offsets = [(i - (len(vals) - 1) / 2) * 0.025 for i in range(len(vals))]
            ax.scatter([position + o for o in offsets], vals, s=12, color=COLORS[arm["arm"]], alpha=0.75, zorder=3)
            ax.errorbar(position + 0.3, arm["heldout21"]["mean"], yerr=arm["heldout21"]["sd"] or 0.0, fmt="D",
                        ms=4, capsize=3, color=COLORS[arm["arm"]], zorder=4)
        ax.axhline(arms[0]["heldout21"]["mean"], color=BLACK, lw=0.8, ls="--")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([f"{a['arm']}\n{a['label'].replace(', ', chr(10))}" for a in arms], fontsize=7)
        ax.set_ylabel("multiclass mask AP\n(last.pth, 21 held-out plans)")
        fig.tight_layout()
        fig.savefig(out / "figures" / "fig_ocr_runs.pdf")
        plt.close(fig)

    # ---- tables
    md = ["## OCR study: text-aware detection and label-rewrite inpainting\n"]
    tx = ["\\subsection{OCR study: text-aware detection and label-rewrite inpainting}\n\\label{sec:ocr}\n"]
    md += ["| Arm | Method | n | Mask AP (21) | AP50 (21) | AP (11 test) | Rare macro AP | Converted rooms seen |",
           "|---|---|---|---|---|---|---|---|"]
    tx.append("\\begin{table}[H]\n\\centering\n\\caption{OCR study: multiclass mask AP of \\code{last.pth}, mean "
              "$\\pm$ s.d. over seeds 1001--1012.}\n\\label{tab:ocr}\n\\small\n\\begin{tabular}{@{}llrrrrr@{}}\n"
              "\\toprule\nArm & Method & $n$ & AP (21) & AP50 (21) & AP (11 test) & Rare AP \\\\\n\\midrule")
    for arm in arms:
        cells = [mean_sd(arm["heldout21"]), mean_sd(arm["heldout21_AP50"]), mean_sd(arm["test11"]),
                 mean_sd(arm["rare_mask_AP"], 4)]
        dose = arm["converted_room_exposure_per_run"]
        md.append(f"| {arm['arm']} | {arm['label']} | {arm['heldout21']['n']} | " + " | ".join(cells)
                  + f" | {fmt(dose['mean'], 0) if dose and dose['n'] else '--'} |")
        tx.append(f"{arm['arm']} & {tex(arm['label'])} & {arm['heldout21']['n']} & "
                  + " & ".join(tex(c) for c in cells) + " \\\\")
    md.append("")
    tx.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    md += ["| Contrast | Δ AP | 95% CI (seed × image) | 95% CI (seed × project) | paired p | Holm p | Δ rare AP | Supported |",
           "|---|---|---|---|---|---|---|---|"]
    tx.append("\\begin{table}[H]\n\\centering\n\\caption{OCR study contrasts: one-sided paired $t$ over seeds, Holm "
              "over H6--H8; CI resamples seed pairs and the 21 held-out plans (or projects).}\n\\label{tab:ocrtests}\n"
              "\\small\n\\begin{tabular}{@{}lrrrrrl@{}}\n\\toprule\nContrast & $\\Delta$ & 95\\% CI & paired $p$ & "
              "Holm $p$ & $\\Delta$ rare AP & Supported \\\\\n\\midrule")
    labels = {"H6": "H6: e19 - e0", "H7": "H7: e20 - e19", "H8": "H8: e18 - e0",
              "e20_vs_e0": "e20 - e0 (descr.)", "e20_vs_e18": "e20 - e18 (descr.)"}
    for name, c in list(tests.items()) + [(k, v) for k, v in descriptive.items() if k != "interaction"]:
        if not c:
            continue
        supported = ("yes" if c.get("supported") else "no") if name in HYPOTHESES else "--"
        row = [fmt(c["delta"], 3, True), fmt_ci(c["ci95_paired_seed_x_image"]),
               fmt_ci(c["ci95_paired_seed_x_project_cluster"]), fmt_p(c["p_one_sided"]),
               fmt_p(c.get("p_holm")) if name in HYPOTHESES else "--", fmt(c["rare_delta"], 4, True), supported]
        md.append(f"| {labels[name]} | " + " | ".join(row) + " |")
        tex_row = [row[0], row[1], row[3], row[4], row[5], row[6]]
        tx.append(f"{tex(labels[name])} & " + " & ".join(tex(x) for x in tex_row) + " \\\\")
    md.append("")
    tx.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    if arms:
        tx.append("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.85\\textwidth]{fig_ocr_runs.pdf}\n"
                  "\\caption{OCR study runs: one dot per seed, diamond = mean $\\pm$ s.d.}\n\\label{fig:ocr}\n"
                  "\\end{figure}\n")
    sentences = []
    for name, c in tests.items():
        if not c:
            sentences.append(f"{name} could not be evaluated (missing runs).")
            continue
        verdict = "is supported" if c["supported"] else "is not supported"
        sentences.append(f"{labels[name]} {verdict}: {c['delta']:+.3f} AP (95% CI {fmt_ci(c['ci95_paired_seed_x_image'])}, "
                         f"paired one-sided p = {fmt_p(c['p_one_sided'])}, Holm p = {fmt_p(c['p_holm'])}).")
        ci = c["ci95_paired_seed_x_image"]
        if ci and ci[1] < SESOI:
            sentences.append(f"For {name} the CI upper bound is below the smallest effect of interest ({SESOI}).")
    inter = descriptive.get("interaction")
    if inter:
        sentences.append(f"Descriptively, the interaction (inpainting gain with text minus without) is "
                         f"{inter['mean']:+.3f} AP (n = {inter['n']}).")
    for line in sentences:
        md.append(line + "\n")
        tx.append(tex(line) + "\n")
    (out / "ocr_section.tex").write_text("\n".join(tx), encoding="utf-8")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
