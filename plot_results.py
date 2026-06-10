#!/usr/bin/env python3
"""Regenerate the paper's figures from measured results.json.

Usage:
    cd riskgate && python run.py            # produces results.json
    python plot_results.py                  # writes ../figures/*.png

Styling goals:
  * standard matplotlib palette (tab10 / default Cn colors)
  * compact figure sizes for a single-column layout
  * constrained_layout + explicit headroom so labels/annotations never
    overlap lines, markers, or each other
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results.json")
OUT = os.path.normpath(os.path.join(HERE, "..", "figures"))
os.makedirs(OUT, exist_ok=True)

# Standard matplotlib default colors
C_LINE = "C0"       # blue
C_ACCENT = "C1"     # orange (RACG / highlight)
BASELINE_COLORS = ["C0", "C2", "C4", "C5"]  # distinct standard colors

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "figure.constrained_layout.use": True,
})

if not os.path.exists(RESULTS):
    raise SystemExit("results.json not found -- run `python run.py` first.")

with open(RESULTS) as f:
    R = json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: success vs. risk-penalty lambda (the lambda-dagger crossover).
# UE and ISR are 0 for every lambda, so the discriminating axis is success:
# below the crossover RACG fails closed on high-risk-shortcut tasks.
# ---------------------------------------------------------------------------
pareto = sorted(R["pareto"], key=lambda d: d["lam"])
lams = [d["lam"] for d in pareto]
succ = [d["success"] for d in pareto]

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(lams, succ, "-o", color=C_LINE, lw=1.6, markersize=5, zorder=3,
        label="benign+stress success")

# Mark the predicted crossover lambda_dagger ~ 1.25.
ax.axvline(1.25, color=C_ACCENT, ls="--", lw=1.2, zorder=2,
           label=r"$\lambda^\dagger\!\approx\!1.25$")
ax.scatter([2.0], [dict(zip(lams, succ))[2.0]], s=70, color=C_ACCENT,
           edgecolor="black", linewidth=0.5, zorder=4,
           label=r"default $\lambda^\star=2$")

ax.set_xlabel(r"risk penalty $\lambda$")
ax.set_ylabel("Benign task success")
ax.set_ylim(min(succ) - 0.04, 1.02)
ax.set_title("Risk penalty vs. success")
ax.legend(loc="lower right", frameon=True)
ax.text(0.45, min(succ) + 0.005, "fails\nclosed", fontsize=7, ha="center",
        va="bottom", color="0.35")
fig.savefig(os.path.join(OUT, "pareto_frontier.png"))
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: injection success rate by method.
# ---------------------------------------------------------------------------
order = ["all_tools", "keyword_top10", "state_aware", "causal_frontier", "racg_lam2"]
labels = ["All\ntools", "Keyword\ntop-10", "State-\naware", "Causal\nfrontier", "RACG\n(ours)"]
isr = [R["methods"][m]["injection_success_rate"] for m in order]
colors = [C_LINE, C_LINE, C_LINE, C_LINE, C_ACCENT]

fig, ax = plt.subplots(figsize=(3.4, 2.6))
bars = ax.bar(labels, isr, color=colors, edgecolor="black", linewidth=0.5,
              width=0.7)
for b, v in zip(bars, isr):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
            ha="center", va="bottom", fontsize=7.5)
ax.set_ylabel("Injection success rate (lower = better)")
ax.set_ylim(0, 1.15)
ax.set_title("Structural injection resistance")
ax.margins(x=0.02)
fig.savefig(os.path.join(OUT, "injection_by_method.png"))
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: high-risk attack surface over trajectory steps.
# ---------------------------------------------------------------------------
trace = R["attack_surface_trace"]
style = [
    ("all_tools", "All tools", BASELINE_COLORS[0], "-o"),
    ("state_aware", "State-aware", BASELINE_COLORS[1], "-^"),
    ("causal_frontier", "Causal frontier", BASELINE_COLORS[2], "-s"),
    ("racg_lam2", "RACG (ours)", C_ACCENT, "-D"),
]

fig, ax = plt.subplots(figsize=(3.4, 2.6))
maxlen = max(len(v) for v in trace.values())
ymax = max((max(v) if v else 0) for v in trace.values())
for key, label, color, st in style:
    ys = trace.get(key, [])
    xs = list(range(1, len(ys) + 1))
    lw = 2.2 if key == "racg_lam2" else 1.4
    ax.plot(xs, ys, st, color=color, lw=lw, markersize=5, label=label,
            zorder=(4 if key == "racg_lam2" else 3))
ax.set_xlabel("Trajectory step")
ax.set_ylabel("Visible high-risk tools")
ax.set_xticks(range(1, maxlen + 1))
ax.set_ylim(-0.5, ymax * 1.28 + 0.5)   # headroom so legend clears the lines
ax.set_title("High-risk attack surface per step")
ax.legend(loc="upper center", frameon=True, ncol=2, fontsize=7.5)
fig.savefig(os.path.join(OUT, "attack_surface_by_step.png"))
plt.close(fig)

print("Wrote figures to", OUT)
print("  pareto_frontier.png, injection_by_method.png, attack_surface_by_step.png")
