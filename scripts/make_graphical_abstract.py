#!/usr/bin/env python3
"""Graphical abstract for the IEEE Access submission: one self-contained figure that
tells the whole story --- (A) task-irrelevant configuration moves the in-context model
but not the fine-tuned one or the controls; (B) that signal is verbatim memorization,
which shrinks with model scale. Vector PDF + 600-DPI PNG.

    /usr/bin/python3 scripts/make_graphical_abstract.py
"""
import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CB = {"icl": "#D55E00", "great": "#0072B2", "ctrl": "#999999", "accent": "#009E73"}
A = os.path.join(ROOT, "results_reseed", "analysis")

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "savefig.dpi": 600,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- data: per-config KS for icl & great (15-seed diabetes) ----
pm = defaultdict(lambda: defaultdict(dict))
for r in csv.DictReader(open(os.path.join(A, "per_metric_ci.csv"))):
    if r["metric"] == "ks_marginal":
        pm[r["generator"]][r["config_id"]] = (float(r["mean"]), float(r["ci_lo"]), float(r["ci_hi"]))
ORDER = ["baseline", "serialization=keyvalue", "serialization=compact",
         "column_order=perm1", "column_order=perm2", "column_order=perm3",
         "numeric_format=rounded"]
LAB = ["base", "key", "cmpct", "ord1", "ord2", "ord3", "round"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
fig.suptitle("Single-configuration comparisons of LLM tabular synthesizers can mislead",
             fontsize=14, fontweight="bold", y=1.02)

# ---- Panel A: config moves ICL, not GReaT ----
for gen, col, mk in [("icl", CB["icl"], "o"), ("great", CB["great"], "s")]:
    xs, ys, lo, hi = [], [], [], []
    for i, c in enumerate(ORDER):
        if c in pm[gen]:                      # GReaT lacks the serialization configs
            m, cl, ch = pm[gen][c]
            xs.append(i); ys.append(m); lo.append(m - cl); hi.append(ch - m)
    axA.errorbar(xs, ys, yerr=[lo, hi], marker=mk, color=col, capsize=3, lw=2,
                 label=("in-context (ICL)" if gen == "icl" else "fine-tuned (GReaT)"))
# control floor band (ctgan/tvae KS ~0.159-0.225, config-invariant)
axA.axhspan(0.159, 0.225, color=CB["ctrl"], alpha=0.18, label="text-blind controls (config-invariant)")
axA.set_xticks(range(len(ORDER))); axA.set_xticklabels(LAB, rotation=30, ha="right")
axA.set_ylabel("KS (lower = better)")
axA.set_title("Task-irrelevant configuration moves ICL,\nnot GReaT or the controls")
axA.legend(fontsize=8.5, frameon=False, loc="lower center", ncol=1)
axA.annotate("same data,\nonly formatting changes", xy=(1, pm["icl"][ORDER[1]][0]),
             xytext=(2.7, 0.305), fontsize=8.5, color=CB["icl"], ha="left",
             arrowprops=dict(arrowstyle="->", color=CB["icl"]))

# ---- Panel B: memorization, and it shrinks with scale ----
groups = ["ICL\n0.5B", "GReaT", "controls", "ICL 7B\nadult", "ICL 7B\ndiab.", "ICL 7B\nstudent"]
vals = [42.9, 0.0, 0.0, 0.5, 1.4, 13.5]
cols = [CB["icl"], CB["great"], CB["ctrl"], CB["accent"], CB["accent"], CB["accent"]]
bars = axB.bar(range(len(groups)), vals, color=cols, width=0.66)
for b, v in zip(bars, vals):
    axB.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:g}%", ha="center", fontsize=9)
axB.set_xticks(range(len(groups))); axB.set_xticklabels(groups, fontsize=8.5)
axB.set_ylabel("verbatim-copy rate (% of rows)")
axB.set_ylim(0, 50)
axB.set_title("The signal is verbatim memorization,\nand it shrinks with model scale")
axB.annotate("removing copies (size-matched)\nerases ICL's fidelity edge",
             xy=(0, 42.9), xytext=(1.3, 37), fontsize=9, color=CB["icl"],
             arrowprops=dict(arrowstyle="->", color=CB["icl"]))

fig.text(0.5, -0.06,
         "Protocol: report $\\geq$3 configurations with 95% CIs, include a text-blind "
         "control (noise floor), and — for in-context generators — report the "
         "verbatim-copy rate before trusting any fidelity number.",
         ha="center", fontsize=10, style="italic",
         bbox=dict(boxstyle="round,pad=0.5", fc="#F2F2F2", ec="#CCCCCC"))

fig.tight_layout(rect=[0, 0.02, 1, 1])
out_dir = os.path.join(A, "figures_hires")
os.makedirs(out_dir, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, f"graphical_abstract.{ext}"), bbox_inches="tight")
print("wrote graphical_abstract.pdf / .png to", out_dir)
