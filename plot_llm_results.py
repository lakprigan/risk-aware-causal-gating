#!/usr/bin/env python3
"""Regenerate the real-LLM validation figure from merged llm_results.json.

Usage:
    python llm_runner.py ...     # writes per-model JSON (and an aggregate)
    # merge per-model files into llm_results.json (see results_llm/), then:
    python plot_llm_results.py   # writes figures/llm_highrisk_by_method.png

The figure is the headline H5 result: the model-driven high-risk-call rate
under injection, per model, for the risk-agnostic causal frontier (CMTF) vs
RACG (lambda=2). RACG should be 0.00 for every model; CMTF should be nonzero.
Styling mirrors plot_results.py (default matplotlib palette, single-column
sizing, constrained layout).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "llm_results.json")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

C_CMTF = "C0"    # blue - risk-agnostic causal frontier (leaks)
C_RACG = "C1"    # orange - RACG (structural defense)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "figure.constrained_layout.use": True,
})

if not os.path.exists(RESULTS):
    raise SystemExit("llm_results.json not found -- merge per-model results first.")

with open(RESULTS) as f:
    R = json.load(f)

models = sorted(R["models"])


def _short(spec: str) -> str:
    s = spec.replace("bedrock:", "")
    # trim long ids for readable x labels
    s = s.replace("us.anthropic.claude-", "claude-").replace("us.amazon.", "")
    s = s.replace("-20251001-v1:0", "").replace("openai.", "")
    return s


labels = [_short(m) for m in models]
cmtf = [R["models"][m]["causal_frontier"]["highrisk_call_rate_under_injection"]
        for m in models]
racg = [R["models"][m]["racg_lam2"]["highrisk_call_rate_under_injection"]
        for m in models]

x = range(len(models))
w = 0.38
fig, ax = plt.subplots(figsize=(7.0, 3.2))
b1 = ax.bar([i - w / 2 for i in x], cmtf, w, label="Causal frontier (CMTF)",
            color=C_CMTF)
b2 = ax.bar([i + w / 2 for i in x], racg, w, label="RACG ($\\lambda=2$)",
            color=C_RACG)
ax.set_ylabel("High-risk-call rate under injection")
ax.set_title("Real-LLM validation: structural injection defense (H5)")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylim(0, max(0.4, max(cmtf) * 1.25))
ax.legend()

# annotate RACG zeros so the 0.00 is unmistakable in print
for i, v in zip(x, racg):
    if v == 0:
        ax.annotate("0.00", (i + w / 2, 0.005), ha="center", va="bottom",
                    fontsize=6, color=C_RACG)

path = os.path.join(OUT, "llm_highrisk_by_method.png")
fig.savefig(path)
print(f"wrote {path}")
print(f"models: {len(models)} | CMTF range {min(cmtf):.2f}-{max(cmtf):.2f} | "
      f"RACG all == {set(racg)}")
