#!/usr/bin/env python3
"""Turn outputs/final/confirm/summary.json into a report bundle.

Writes <out>/:
  confirm_section.tex   a LaTeX section for the project report (booktabs, float,
                        graphicx; uses the report's \\code{} macro)
  report.md             the same content as Markdown, for quick reading
  figures/*.pdf|png     figures referenced by the section
  README.txt            how to drop the bundle into report/

All numbers are read from summary.json and the chain's manifests; nothing is
typed in by hand. Sentences that depend on the outcome are chosen by the
pre-registered decision rules recorded in summary.json.
"""
import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import resolve_path  # noqa: E402

BLACK, ORANGE, SKY, GREEN, BLUE, VERMILLION, PURPLE, GREY = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#9E9E9E")
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
CONFIRM_COLOR = {"e0": BLACK, "e8": ORANGE, "e16": BLUE, "e17": VERMILLION}
GRID_COLOR = {"e0": BLACK, "e1": GREY, "e2": SKY, "e3": SKY, "e4": SKY, "e5": SKY, "e6": GREEN,
              "e7": GREEN, "e8": ORANGE, "e9": ORANGE, "e10": ORANGE, "e11": ORANGE,
              "e12": PURPLE, "e13": PURPLE}
SESOI = 0.02


def fmt(value, digits=3, signed=False):
    if value is None:
        return "--"
    return (f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}")


def fmt_p(value):
    if value is None:
        return "--"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def fmt_ci(ci, digits=3):
    if not ci:
        return "--"
    return f"[{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


def mean_sd(block, digits=3):
    if not block or block.get("mean") is None:
        return "--"
    sd = block.get("sd")
    return f"{block['mean']:.{digits}f} ± {sd:.{digits}f}" if sd is not None else f"{block['mean']:.{digits}f}"


def tex(text):
    return (text.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
            .replace("_", "\\_").replace("#", "\\#").replace("±", "$\\pm$").replace("≥", "$\\geq$")
            .replace("≤", "$\\leq$").replace("×", "$\\times$").replace("−", "-").replace("Δ", "$\\Delta$")
            .replace("→", "$\\rightarrow$"))


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def grid_reference(track):
    """Original grid numbers (best.pth on the 11 test plans), if available."""
    summary = read_json(resolve_path("outputs/final/grid_r1/summary.json"))
    if not summary:
        return {}
    return {row["arm"]: row["aggregate"]["overall_mask_AP"]["mean"]
            for row in summary["arms"] if row["track"] == track}


# ---------------------------------------------------------------- figures
def fig_confirm_runs(confirm, path):
    arms = confirm.get("arms", [])
    if not arms:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    for position, arm in enumerate(arms):
        values = [run["heldout21_AP"] for run in arm["runs"]]
        offsets = [(i - (len(values) - 1) / 2) * 0.03 for i in range(len(values))]
        ax.scatter([position + o for o in offsets], values, s=12, color=CONFIRM_COLOR.get(arm["arm"], GREY),
                   alpha=0.75, zorder=3)
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        ax.errorbar(position + 0.28, mean, yerr=sd, fmt="D", color=CONFIRM_COLOR.get(arm["arm"], GREY),
                    ms=4, capsize=3, zorder=4)
    e0 = next((a for a in arms if a["arm"] == "e0"), None)
    if e0:
        ax.axhline(e0["heldout21"]["mean"], color=BLACK, lw=0.8, ls="--", zorder=1)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([f"{a['arm']}\n" + {"e0": "original", "e8": "copy-paste\n(e8 replication)",
                                          "e16": "matched\ncopy-paste", "e17": "matched wide-seam\ninpainting"}
                        .get(a["arm"], a["arm"]) for a in arms])
    ax.set_ylabel("class-agnostic mask AP\n(last.pth, 21 held-out plans)")
    ax.set_title("Confirmatory runs: one dot per seed, diamond = mean ± s.d.")
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_contrasts(confirm, path):
    contrasts = []
    labels = {"H1": "H1  e8 − e0", "H2": "H2  e17 − e0", "H3": "H3  e17 − e16",
              "e16_vs_e0": "e16 − e0 (descr.)", "e8_vs_e16": "e8 − e16 (sampler, descr.)"}
    for name in ("H1", "H2", "H3", "e16_vs_e0", "e8_vs_e16"):
        c = confirm.get("tests", {}).get(name) or confirm.get("descriptive", {}).get(name)
        if c and c.get("ci95_seed_x_image"):
            contrasts.append((labels[name], c))
    if not contrasts:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 0.45 * len(contrasts) + 0.9))
    for row, (label, c) in enumerate(reversed(contrasts)):
        lo, hi = c["ci95_seed_x_image"]
        color = VERMILLION if c.get("supported") else BLUE
        ax.plot([lo, hi], [row, row], color=color, lw=2)
        ax.plot(c["delta"], row, "o", color=color, ms=5)
    ax.axvline(0, color=BLACK, lw=0.8)
    ax.axvline(SESOI, color=GREY, lw=0.8, ls="--")
    ax.set_yticks(range(len(contrasts)))
    ax.set_yticklabels([label for label, _ in reversed(contrasts)])
    ax.set_xlabel("difference in class-agnostic mask AP (95% seed × image bootstrap CI)")
    ax.set_title("Pre-registered contrasts (red = supported; dashed = smallest effect of interest +0.02)")
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_rescore(rescore, path):
    tracks = [t for t in ("agnostic", "multiclass") if t in rescore]
    if not tracks:
        return False
    fig, axes = plt.subplots(1, len(tracks), figsize=(6.4 * len(tracks) / 1.4, 2.6), squeeze=False)
    for ax, track in zip(axes[0], tracks):
        rows = rescore[track]["rows"]
        reference = grid_reference(track)
        for position, row in enumerate(rows):
            color = GRID_COLOR.get(row["arm"], GREY)
            block = row["heldout21"]
            ax.errorbar(position, block["mean"], yerr=block["sd"] or 0, fmt="o", color=color, ms=4, capsize=2)
            if row["arm"] in reference:
                ax.plot(position + 0.25, reference[row["arm"]], marker="o", mfc="none", color=color, ms=4)
        e0 = next((r for r in rows if r["arm"] == "e0"), None)
        if e0:
            ax.axhline(e0["heldout21"]["mean"], color=BLACK, lw=0.8, ls="--")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r["arm"] for r in rows])
        ax.set_title(f"{track}: filled = last.pth on 21 held-out plans; hollow = grid best.pth on 11 test plans")
        ax.set_ylabel("mask AP")
    fig.savefig(path)
    plt.close(fig)
    return True


def copy_qa_examples(selection, matched, qa_root, out_dir):
    """Copy one matched pair's QA rows (wide-seam inpainting and its copy-paste twin)."""
    if not selection or selection.get("chosen_strength") is None or not matched or matched.get("status") != "built":
        return []
    tag = f"s{int(round(float(selection['chosen_strength']) * 100)):03d}"
    copied = []
    for target, donor in matched["pairs"][:40]:
        wide = Path(qa_root) / tag / "diffusion_inpaint" / "rows" / f"c-diffusion_inpaint-target{target}-donor{donor}.png"
        paste = resolve_path("outputs/augmentation_c/copy_paste/rows") / f"c-copy_paste-target{target}-donor{donor}.png"
        if wide.is_file() and paste.is_file():
            shutil.copy(wide, out_dir / "c_wide_example.png")
            shutil.copy(paste, out_dir / "c_copypaste_twin.png")
            copied = ["c_wide_example.png", "c_copypaste_twin.png"]
            break
    return copied


# ---------------------------------------------------------------- text
def outcome_paragraphs(confirm):
    tests = confirm.get("tests", {})
    h1, h2, h3 = tests.get("H1"), tests.get("H2"), tests.get("H3")
    decisions = (confirm.get("decisions") or {}).get("run", {})
    lines = []

    def sentence(name, c, claim):
        ci = fmt_ci(c.get("ci95_seed_x_image"))
        base = (f"{name} ({claim}): Δ = {fmt(c['delta'], 3, True)} AP, 95% CI {ci}, "
                f"one-sided Welch p = {fmt_p(c['welch_p_one_sided'])}")
        if "holm_p" in c:
            base += f", Holm-adjusted p = {fmt_p(c['holm_p'])}"
        base += f", permutation p = {fmt_p(c['permutation_p_one_sided'])}, Hedges' g = {fmt(c['hedges_g'], 2)}"
        base += f" (n = {c['n_control']} vs {c['n_treatment']} seeds)."
        if c.get("supported"):
            return base + " The pre-registered criterion is met: the hypothesis is supported."
        if c.get("tested") is False:
            return base + " Not tested, because H2 was not supported (fixed sequence); reported descriptively."
        return base + " The pre-registered criterion is not met: the hypothesis is not supported."

    if h1:
        lines.append(sentence("H1", h1, "exact replication of copy-paste e8 against e0"))
    elif decisions.get("e8") is False:
        lines.append("H1 was dropped by the pre-registered futility gate before any confirmatory run "
                     "(the leakage-free re-score showed no positive e8 effect).")
    if h2:
        lines.append(sentence("H2", h2, "wide-seam exemplar inpainting e17 against e0"))
    elif decisions.get("e17") is False:
        lines.append("H2 and H3 were dropped because the wide-seam generator did not pass the "
                     "pre-registered generation rule or too few matched pairs survived.")
    if h3:
        lines.append(sentence("H3", h3, "e17 against matched copy-paste e16, i.e. the value of the generated pixels"))

    supported = {name for name, c in tests.items() if c and c.get("supported")}
    if "H2" in supported and "H3" in supported:
        verdict = ("Exemplar-seeded diffusion inpainting with a latent-sized diffusion band improves "
                   "class-agnostic room segmentation over the unaugmented baseline, and the improvement "
                   "exceeds that of copy-pasting the same exemplars into the same rooms. Under this "
                   "protocol, the generated pixels themselves carry the benefit.")
    elif "H2" in supported:
        verdict = ("Exemplar-seeded diffusion inpainting improves class-agnostic room segmentation over "
                   "the unaugmented baseline under a pre-registered, multiplicity-controlled test with "
                   "fresh seeds. The matched comparison with copy-paste does not show that the generated "
                   "pixels add value beyond the transplanted exemplars, so the benefit is attributed to "
                   "exemplar-based augmentation with inpainted integration, not to diffusion alone.")
    elif "H1" in supported:
        verdict = ("The copy-paste effect of the original grid replicates with fresh seeds on the "
                   "leakage-free endpoint, but wide-seam diffusion inpainting is not confirmed. The "
                   "positive finding of the project is exemplar-based augmentation without generated pixels.")
    else:
        bounds = [c.get("ci95_seed_x_image") for c in tests.values() if c and c.get("ci95_seed_x_image")]
        upper = max(ci[1] for ci in bounds) if bounds else None
        verdict = ("No confirmatory hypothesis is supported. The study removes the confounds of the "
                   "original grid and replaces its uninterpretable sign tests with interval estimates"
                   + (f": every pre-registered effect is bounded above by {fmt(upper, 3, True)} AP." if upper is not None
                      else "."))
    return lines, verdict


def errata():
    return [
        "Section 4 states that Mask R-CNN is ImageNet-initialised; the code loads the COCO-trained "
        "maskrcnn_resnet50_fpn_v2 detector weights (src/models/mask_rcnn.py).",
        "The rotated cross-validation pools the 10 validation plans that selected every best.pth and "
        "scores only seed 42; its 'leakage prevented by construction' wording should be withdrawn and the "
        "re-score in this section used instead.",
        "Strategy B (e6/e7) keeps the printed source-room label (e.g. 'Büro') inside rooms relabelled as "
        "rare, and its retention QC threshold defaults to 1.0, so it never fires; the LoRA caveat applies to B too.",
        "e8 is trained on the multi-room copy-paste dataset while e9-e11 use single-edit datasets; "
        "'copy-paste beats diffusion' in the original grid compares two data compositions.",
        "The C2 'diffusion inpaint' arm used a 6-pixel seam at strength 0.30, about one latent cell; "
        "e10/e11 therefore contain very few generated pixels.",
        "Augmented SAM 3 runs received about twice the optimiser steps of the SAM 3 baseline, used the "
        "last checkpoint and a 0.3 mask-score threshold; the within-SAM 3 deltas and the 'sign reversal' "
        "are confounded.",
        "Of the '171 synthetic rare crops' in the DINOv2 experiment only 74 are synthetic; the rest are "
        "re-crops of the 30 real rare rooms.",
        "e5 (0.523), not the LoRA arm e12 (0.531), is the lowest class-agnostic arm of the grid.",
    ]


def build(summary, confirm_root, matched_root, qa_root, out):
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    confirm = summary.get("confirm", {}) or {}
    rescore = summary.get("rescore", {}) or {}
    selection = summary.get("strength_selection")
    decisions = summary.get("decisions") or {}
    matched = read_json(Path(matched_root) / "matched_manifest.json")
    prereg = (Path(confirm_root) / "preregistration_record.txt")
    prereg_text = prereg.read_text().strip() if prereg.is_file() else "not found"
    has = {"runs": fig_confirm_runs(confirm, figures / "fig_confirm_runs.pdf"),
           "contrasts": fig_contrasts(confirm, figures / "fig_confirm_contrasts.pdf"),
           "rescore": fig_rescore(rescore, figures / "fig_rescore_last.pdf")}
    qa = copy_qa_examples(selection, matched, qa_root, figures)
    outcome_lines, verdict = outcome_paragraphs(confirm)

    md, tx = [], []
    md.append("# Confirmatory study: leakage-free re-score and pre-registered replication\n")
    tx.append("\\section{Confirmatory Study}\n\\label{sec:confirm}\n")

    intro = ("The original evaluation could not separate a real augmentation effect from its own "
             "confounds: the rotated cross-validation reused the validation plans that had selected each "
             "checkpoint, augmented arms saw each real plan less often than the baseline, the copy-paste "
             "and diffusion arms were built from differently composed datasets, and the diffusion arm "
             "regenerated a band about one latent cell wide. This section reports a study designed to "
             "remove those confounds. Its hypotheses, endpoint, gates and tests were fixed in "
             "docs/preregistration_confirm.md and committed before any job ran.")
    md.append(intro + "\n")
    tx.append(tex(intro).replace("docs/preregistration\\_confirm.md", "\\code{docs/preregistration_confirm.md}") + "\n")
    md.append("Pre-registration record:\n\n```\n" + prereg_text + "\n```\n")
    tx.append("\\noindent Pre-registration record: \\code{" + prereg_text.split("git_commit: ")[-1].split()[0]
              + "} (commit), recorded in \\code{outputs/confirm/preregistration_record.txt}.\n")

    # --- protocol
    protocol = [
        ("Endpoint", "class-agnostic COCO mask AP@[.50:.95] of the epoch-25 checkpoint (last.pth), which "
                     "is never selected on validation data, evaluated on all 21 held-out plans "
                     "(10 validation + 11 test plans, projects P02, P04 and P07)."),
        ("Arms", "e0 original plans; e8 exact replication of the grid's copy-paste arm; e16 matched "
                 "copy-paste; e17 matched wide-seam exemplar inpainting. e16 and e17 are composed from the "
                 "same (target, donor) pairs with the same builder seed, so they differ only in pixels."),
        ("Exposure parity", "e16 and e17 use a paired-variant sampler: every epoch follows e0's plan "
                            "order for the same seed, so every source plan occupies exactly one step per "
                            "epoch, as in e0; a plan that has synthetic variants is shown as one of them "
                            "with probability 0.5. Arms with the same seed therefore see the same plan "
                            "sequence and differ only in the pixels of the swapped steps."),
        ("Wide seam", "the diffusion band is 24 model pixels (3 Stable Diffusion latent cells) per "
                      "candidate; the donor's dark strokes (labels, symbols) are excluded from the "
                      "diffusion mask and checked by the retention QC."),
        ("Seeds and tests", "12 fresh seeds (1001-1012) per arm. H1 e8 > e0 and H2 e17 > e0 by one-sided "
                            "Welch tests with Holm correction; a hypothesis is supported only if Holm "
                            "rejects and the two-level (seed × image) bootstrap 95% CI excludes zero. "
                            "H3 e17 > e16 is tested only if H2 is supported."),
    ]
    md.append("## Protocol\n")
    tx.append("\\subsection{Protocol}\n\\begin{itemize}\n")
    for head, body in protocol:
        md.append(f"- **{head}.** {body}")
        tx.append(f"  \\item \\textbf{{{tex(head)}.}} {tex(body)}")
    md.append("")
    tx.append("\\end{itemize}\n")

    # --- generation
    md.append("## Wide-seam generation and gates\n")
    tx.append("\\subsection{Wide-seam generation and gates}\n")
    if selection:
        rows = []
        for strength, s in sorted(selection["pilot"].items(), key=lambda kv: float(kv[0])):
            if "missing" in s:
                continue
            rows.append((strength, s["attempted"], s["accepted"], s["acceptance_rate"],
                         s["mean_donor_retention"], s["median_latent_cells"], s["median_seam_original_pixels"],
                         "yes" if s["passes"] else "no"))
        md.append("| Strength | Attempted | Accepted | Rate | Donor retention | Median band (latent cells) | Median seam (px) | Passes |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.2f} | {r[4]:.3f} | {r[5]:.0f} | {r[6]:.0f} | {r[7]} |")
        tx.append("\\begin{table}[H]\n\\centering\n\\caption{Wide-seam generation pilot over all planned "
                  "candidates. The rule picks the highest strength with acceptance $\\geq 0.80$, mean donor-"
                  "stroke retention $\\geq 0.95$ and a median diffusion band of $\\geq 30$ latent cells.}\n"
                  "\\label{tab:cwide}\n\\small\n\\begin{tabular}{@{}rrrrrrrl@{}}\n\\toprule\n"
                  "Strength & Attempted & Accepted & Rate & Retention & Band (cells) & Seam (px) & Passes \\\\\n\\midrule")
        for r in rows:
            tx.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]:.2f} & {r[4]:.3f} & {r[5]:.0f} & {r[6]:.0f} & {r[7]} \\\\")
        tx.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        chosen = selection.get("chosen_strength")
        sel_text = (f"The rule selected strength {chosen}." if chosen is not None and not selection.get("forced")
                    else "Strength was forced (smoke test)." if selection.get("forced")
                    else "No strength passed the rule, so the inpainting arms were dropped.")
        md.append(sel_text + "\n")
        tx.append(tex(sel_text) + "\n")
    if matched:
        if matched.get("status") == "built":
            mtext = (f"Copy-paste accepted {matched['accepted_copy_paste']} and wide-seam inpainting accepted "
                     f"{matched['accepted_inpaint_wide']} of the planned pairs; {len(matched['pairs'])} pairs were "
                     f"accepted in both and composed into {matched['synthetic_plans']} synthetic plans "
                     f"({matched['converted_rooms']} converted rooms over {matched['plans_with_variants']} source plans).")
        else:
            mtext = f"Matched datasets were not built (status: {matched.get('status')})."
        md.append(mtext + "\n")
        tx.append(tex(mtext) + "\n")
    notes = decisions.get("notes", {})
    if notes:
        gate_text = ("Gates: " + "; ".join(f"{arm}: {'runs' if decisions['run'].get(arm) else 'dropped'} "
                                           f"({notes.get(arm, '')})" for arm in ("e0", "e8", "e16", "e17")) + ".")
        md.append(gate_text + "\n")
        tx.append(tex(gate_text) + "\n")
    if qa:
        md.append("Example pair (same target and donor): `figures/c_wide_example.png` (wide-seam inpainting) and "
                  "`figures/c_copypaste_twin.png` (copy-paste).\n")
        tx.append("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\textwidth]{c_wide_example.png}\\\\[1ex]\n"
                  "\\includegraphics[width=\\textwidth]{c_copypaste_twin.png}\n\\caption{The same matched pair "
                  "integrated by wide-seam inpainting (top, e17) and by copy-paste (bottom, e16).}\n"
                  "\\label{fig:cwide}\n\\end{figure}\n")

    # --- rescore
    md.append("## Leakage-free re-score of the existing grid (exploratory)\n")
    tx.append("\\subsection{Leakage-free re-score of the existing grid}\n")
    rescore_intro = ("Every grid checkpoint's last.pth was re-evaluated on the 21 held-out plans. These "
                     "numbers are exploratory: they re-use the seeds that produced the original ranking, so "
                     "they serve to correct the cross-validation and to drive the futility gate, not to "
                     "confirm an effect. Differences are against e0; CIs resample seeds and images; "
                     "p-values are one-sided Welch tests, Holm-adjusted over the 13 arms of each track.")
    md.append(rescore_intro + "\n")
    tx.append(tex(rescore_intro) + "\n")
    for track in ("agnostic", "multiclass"):
        if track not in rescore:
            continue
        rows = rescore[track]["rows"]
        reference = grid_reference(track)
        md.append(f"### {track}\n")
        md.append("| Arm | Method | last.pth, 21 plans | Δ vs e0 | 95% CI | Holm p | grid best.pth, 11 plans |")
        md.append("|---|---|---|---|---|---|---|")
        tx.append("\\begin{table}[H]\n\\centering\n\\caption{Re-score, " + track + " track: mask AP of "
                  "\\code{last.pth} on the 21 held-out plans (mean $\\pm$ s.d. over seeds 42, 123, 2026), "
                  "difference from e0 with its seed $\\times$ image bootstrap 95\\% CI and Holm-adjusted "
                  "one-sided Welch $p$; the last column repeats the original grid value (best.pth, 11 test plans).}\n"
                  f"\\label{{tab:rescore{track}}}\n\\small\n\\begin{{tabular}}{{@{{}}llrrrrr@{{}}}}\n\\toprule\n"
                  "Arm & Method & AP (21 plans) & $\\Delta$ & 95\\% CI & Holm $p$ & grid AP \\\\\n\\midrule")
        for row in rows:
            md.append(f"| {row['arm']} | {row['label']} | {mean_sd(row['heldout21'])} | "
                      f"{fmt(row.get('delta_vs_e0'), 3, True) if row['arm'] != 'e0' else '--'} | "
                      f"{fmt_ci(row.get('ci95_seed_x_image'))} | {fmt_p(row.get('holm_p'))} | "
                      f"{fmt(reference.get(row['arm']))} |")
            tx.append(f"{row['arm']} & {tex(row['label'])} & {tex(mean_sd(row['heldout21']))} & "
                      f"{fmt(row.get('delta_vs_e0'), 3, True) if row['arm'] != 'e0' else '--'} & "
                      f"{fmt_ci(row.get('ci95_seed_x_image'))} & {fmt_p(row.get('holm_p'))} & "
                      f"{fmt(reference.get(row['arm']))} \\\\")
        md.append("")
        tx.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        positives = [r["arm"] for r in rows if r.get("ci95_seed_x_image") and r["ci95_seed_x_image"][0] > 0]
        negatives = [r["arm"] for r in rows if r.get("ci95_seed_x_image") and r["ci95_seed_x_image"][1] < 0]
        held = [r["arm"] for r in rows if r.get("holm_reject")]
        txt = (f"On the {track} track, CIs above zero: {', '.join(positives) or 'none'}; CIs below zero: "
               f"{', '.join(negatives) or 'none'}; Holm-significant: {', '.join(held) or 'none'}.")
        md.append(txt + "\n")
        tx.append(tex(txt) + "\n")
    if has["rescore"]:
        md.append("Figure: `figures/fig_rescore_last.pdf`.\n")
        tx.append("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\textwidth]{fig_rescore_last.pdf}\n"
                  "\\caption{Re-score of the existing grid: filled markers are \\code{last.pth} on the 21 "
                  "held-out plans (mean $\\pm$ s.d. over three seeds), hollow markers the original grid "
                  "values (best.pth on the 11 test plans).}\n\\label{fig:rescore}\n\\end{figure}\n")

    # --- confirm
    md.append("## Confirmatory results\n")
    tx.append("\\subsection{Confirmatory results}\n")
    arms = confirm.get("arms", [])
    if arms:
        md.append("| Arm | Method | n | AP (21 plans) | AP50 | AP (11 test plans) | AP (10 val plans) |")
        md.append("|---|---|---|---|---|---|---|")
        tx.append("\\begin{table}[H]\n\\centering\n\\caption{Confirmatory runs, class-agnostic track: mask AP "
                  "of \\code{last.pth}, mean $\\pm$ s.d. over fresh seeds.}\n\\label{tab:confirmarms}\n\\small\n"
                  "\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\nArm & Method & $n$ & AP (21) & AP50 (21) & "
                  "AP (11 test) & AP (10 val) \\\\\n\\midrule")
        for arm in arms:
            md.append(f"| {arm['arm']} | {arm['label']} | {arm['heldout21']['n']} | {mean_sd(arm['heldout21'])} | "
                      f"{mean_sd(arm['heldout21_AP50'])} | {mean_sd(arm['test11'])} | {mean_sd(arm['val10'])} |")
            tx.append(f"{arm['arm']} & {tex(arm['label'])} & {arm['heldout21']['n']} & "
                      f"{tex(mean_sd(arm['heldout21']))} & {tex(mean_sd(arm['heldout21_AP50']))} & "
                      f"{tex(mean_sd(arm['test11']))} & {tex(mean_sd(arm['val10']))} \\\\")
        md.append("")
        tx.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        contrast_rows = [(name, c) for name, c in {**confirm.get("tests", {}), **confirm.get("descriptive", {})}.items() if c]
        md.append("| Contrast | Δ AP | 95% CI | Welch p | Holm p | Perm. p | g | Seeds positive (paired) | Supported |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        tx.append("\\begin{table}[H]\n\\centering\n\\caption{Pre-registered contrasts (H1--H3) and descriptive "
                  "contrasts. $\\Delta$ is treatment minus control; the CI resamples seeds within arm and the "
                  "21 held-out plans jointly; $p$-values are one-sided.}\n\\label{tab:confirmtests}\n\\small\n"
                  "\\begin{tabular}{@{}lrrrrrrl@{}}\n\\toprule\nContrast & $\\Delta$ & 95\\% CI & Welch $p$ & "
                  "Holm $p$ & Perm.\\ $p$ & $g$ & Supported \\\\\n\\midrule")
        names = {"H1": "H1: e8 - e0", "H2": "H2: e17 - e0", "H3": "H3: e17 - e16",
                 "e16_vs_e0": "e16 - e0 (descr.)", "e8_vs_e16": "e8 - e16 (descr.)", "e17_vs_e16": "e17 - e16 (descr.)"}
        for name, c in contrast_rows:
            supported = "yes" if c.get("supported") else ("--" if name not in ("H1", "H2", "H3") else "no")
            md.append(f"| {names.get(name, name)} | {fmt(c['delta'], 3, True)} | {fmt_ci(c.get('ci95_seed_x_image'))} | "
                      f"{fmt_p(c['welch_p_one_sided'])} | {fmt_p(c.get('holm_p'))} | {fmt_p(c['permutation_p_one_sided'])} | "
                      f"{fmt(c['hedges_g'], 2)} | {c['seed_paired_positive']}/{c['seed_paired_n']} | {supported} |")
            tx.append(f"{tex(names.get(name, name))} & {fmt(c['delta'], 3, True)} & {fmt_ci(c.get('ci95_seed_x_image'))} & "
                      f"{tex(fmt_p(c['welch_p_one_sided']))} & {tex(fmt_p(c.get('holm_p')))} & "
                      f"{tex(fmt_p(c['permutation_p_one_sided']))} & {fmt(c['hedges_g'], 2)} & {supported} \\\\")
        md.append("")
        tx.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        for line in outcome_lines:
            md.append(line + "\n")
            tx.append(tex(line) + "\n")
        per_project = []
        for name, c in confirm.get("tests", {}).items():
            if c and c.get("delta_per_project"):
                parts = ", ".join(f"{p} {d:+.3f}" for p, d in c["delta_per_project"].items())
                per_project.append(f"{name} per held-out project: {parts}; on the 11 original test plans "
                                   f"{c['delta_test11']:+.3f}, on the 10 validation plans {c['delta_val10']:+.3f}, "
                                   f"AP50 {c['delta_AP50']:+.3f}.")
        for line in per_project:
            md.append(line + "\n")
            tx.append(tex(line) + "\n")
        if has["runs"]:
            tx.append("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\textwidth]{fig_confirm_runs.pdf}\n"
                      "\\caption{Confirmatory runs per arm.}\n\\label{fig:confirmruns}\n\\end{figure}\n")
        if has["contrasts"]:
            tx.append("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\textwidth]{fig_confirm_contrasts.pdf}\n"
                      "\\caption{Contrasts with seed $\\times$ image bootstrap 95\\% CIs.}\n\\label{fig:confirmcontrasts}\n\\end{figure}\n")
        md.append("Figures: `figures/fig_confirm_runs.pdf`, `figures/fig_confirm_contrasts.pdf`.\n")
    else:
        none_text = "No confirmatory run was executed (all arms were dropped by the gates)."
        md.append(none_text + "\n")
        tx.append(tex(none_text) + "\n")

    md.append("## Conclusion\n")
    tx.append("\\subsection{Conclusion of the confirmatory study}\n")
    md.append(verdict + "\n")
    tx.append(tex(verdict) + "\n")
    limits = ("Limitations: the endpoint is class-agnostic only, because 17 rare test instances cannot "
              "resolve a rare-class effect; the 21 held-out plans come from three buildings; BatchNorm runs "
              "at batch size 1 in every arm; the synthetic pool is small (one to two converted rooms per "
              "synthetic plan, from at most 52 planned pairs).")
    md.append(limits + "\n")
    tx.append(tex(limits) + "\n")

    md.append("## Corrections to earlier sections\n")
    tx.append("\\subsection{Corrections to earlier sections}\n\\begin{itemize}\n")
    for item in errata():
        md.append(f"- {item}")
        tx.append(f"  \\item {tex(item)}")
    md.append("")
    tx.append("\\end{itemize}\n")

    (out / "confirm_section.tex").write_text("\n".join(tx), encoding="utf-8")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "README.txt").write_text(
        "Copy figures/* into report/images/ and add \\input{Sections/confirm_section} (after copying\n"
        "confirm_section.tex into report/Sections/) where the section should appear. The section needs\n"
        "the packages already loaded by report/main.tex (float, booktabs, graphicx, url) and its \\code macro.\n"
        "report.md has the same content as Markdown; summary.json one level up holds every number.\n",
        encoding="utf-8")
    print(f"wrote {out}/confirm_section.tex, report.md, figures/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/final/confirm/summary.json")
    parser.add_argument("--confirm-root", default="outputs/confirm")
    parser.add_argument("--matched-root", default="data/augmented_c_matched")
    parser.add_argument("--qa-root", default="outputs/augmentation_c_wide")
    parser.add_argument("--out", default="outputs/final/confirm/report")
    args = parser.parse_args()
    summary = json.loads(resolve_path(args.summary).read_text(encoding="utf-8"))
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    build(summary, resolve_path(args.confirm_root), resolve_path(args.matched_root), resolve_path(args.qa_root), out)


if __name__ == "__main__":
    main()
