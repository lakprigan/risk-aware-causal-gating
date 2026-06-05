#!/usr/bin/env python3
"""Regenerate the paper's figures from measured results.json.

Usage:
    cd riskgate && python run.py            # produces results.json
    python plot_results.py                  # writes ../figures/*.png

Falls back gracefully if results.json is missing by telling you to run first.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results.json")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

BLUE = (33/255, 68/255, 120/255)
BLUEF = (222/255, 235/255, 247/255)
ORANGE = (198/255, 113/255, 16/255)
RED = (170/255, 30/255, 30/255)

plt.rcParams.update({"font.size": 12, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 300})

if not os.path.exists(RESULTS):
    raise SystemExit("results.json not found -- run `python run.py` first.")

with open(RESULTS) as f:
    R = json.load(f)

# ---------------------------------------------------------------------------
# Figure 1: safety-success Pareto frontier from the lambda sweep.
# ---------------------------------------------------------------------------
pareto = sorted(R["pareto"], key=lambda d: d["lam"])
lams = [d["lam"] for d in pareto]
succ = [d["success"] for d in pareto]
ue = [d["unauthorized_exposure"] for d in pareto]

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(ue, succ, "-o", color=BLUE, lw=2, markersize=7,
        markerfacecolor=BLUEF, markeredgecolor=BLUE, zorder=3)
for x, y, lam in zip(ue, succ, lams):
    ax.annotate(rf"$\lambda={lam:g}$", (x, y), textcoords="offset points",
                xytext=(8, 6), fontsize=9)
# default operating point lambda*=2
d2 = next((d for d in pareto if d["lam"] == 2.0), None)
if d2:
    ax.scatter([d2["unauthorized_exposure"]], [d2["success"]], s=160,
               color=ORANGE, zorder=4, edgecolor="black", linewidth=0.6,
               label=r"default $\lambda^\star=2$")
    ax.legend(loc="lower right", frameon=True)
ax.set_xlabel("Unauthorized high-risk exposure (per task)  \u2190 safer")
ax.set_ylabel("Benign task success  \u2192 better")
ax.set_title("Safety\u2013success Pareto frontier (RACG)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pareto_frontier.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: injection success rate by method.
# ---------------------------------------------------------------------------
order = ["all_tools", "keyword_top10", "state_aware", "causal_frontier", "racg_lam2"]
labels = ["All tools", "Keyword\ntop-10", "State-aware", "Causal\nfrontier", "RACG\n(ours)"]
isr = [R["methods"][m]["injection_success_rate"] for m in order]
colors = [BLUE, BLUE, BLUE, BLUE, ORANGE]

fig, ax = plt.subplots(figsize=(5.6, 3.6))
bars = ax.bar(labels, isr, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, isr):
    ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Injection success rate  \u2190 lower is better")
ax.set_ylim(0, 1.05)
ax.set_title("Structural injection resistance by exposure method")
ax.axhline(0, color="black", lw=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "injection_by_method.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: high-risk attack surface over trajectory steps.
# ---------------------------------------------------------------------------
trace = R["attack_surface_trace"]
style = {"all_tools": ("All tools", BLUE, "-o"),
         "state_aware": ("State-aware", (0.45, 0.55, 0.70), "-^"),
         "causal_frontier": ("Causal frontier", (0.30, 0.45, 0.65), "-s"),
         "racg_lam2": ("RACG (ours)", ORANGE, "-D")}

fig, ax = plt.subplots(figsize=(5.6, 3.6))
maxlen = max(len(v) for v in trace.values())
for key, (label, color, st) in style.items():
    ys = trace.get(key, [])
    xs = list(range(1, len(ys) + 1))
    lw = 2.6 if key == "racg_lam2" else 1.8
    ax.plot(xs, ys, st, color=color, lw=lw, markersize=7,
            markerfacecolor=(color if key == "racg_lam2" else BLUEF),
            markeredgecolor=color, label=label,
            zorder=(4 if key == "racg_lam2" else 3))
ax.set_xlabel("Trajectory step")
ax.set_ylabel("Visible high-risk tools  \u2190 smaller surface")
ax.set_xticks(range(1, maxlen + 1))
ax.set_title("High-risk attack surface over a delete-event trajectory")
ax.legend(loc="upper right", frameon=True, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "attack_surface_by_step.png"))
plt.close(fig)

print("Wrote figures to", OUT)
print("  pareto_frontier.png, injection_by_method.png, attack_surface_by_step.png")
