#!/usr/bin/env python3
"""
Generate placeholder figures for the Risk-Aware Causal Gating (RACG) paper.

These use ILLUSTRATIVE numbers consistent with the paper's hypotheses so the
document compiles with real plots. Replace `DATA` blocks with measured results
once experiments are run (search for: TODO).

Outputs (PNG, 300 dpi) into ./figures:
  - pareto_frontier.png       : safety-success tradeoff as lambda varies
  - injection_by_method.png   : injection success rate per method
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# Paper palette (RGB 0-255 -> 0-1)
BLUE   = (33/255, 68/255, 120/255)     # boxedge
BLUEF  = (222/255, 235/255, 247/255)   # boxfill
ORANGE = (198/255, 113/255, 16/255)    # accentedge
RED    = (170/255, 30/255, 30/255)     # riskedge

plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
})

# ---------------------------------------------------------------------------
# Figure 1: Safety-success Pareto frontier as lambda sweeps.
# TODO: replace with measured (success, unauthorized_exposure) per lambda.
# ---------------------------------------------------------------------------
lambdas      = [0, 0.25, 0.5, 1, 2, 4]
success      = [0.99, 0.99, 0.99, 0.98, 0.98, 0.96]   # benign task success
unauth_expo  = [0.62, 0.41, 0.22, 0.08, 0.00, 0.00]   # avg unauthorized high-risk exposures / task

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(unauth_expo, success, "-o", color=BLUE, lw=2, markersize=7,
        markerfacecolor=BLUEF, markeredgecolor=BLUE, zorder=3)
for x, y, lam in zip(unauth_expo, success, lambdas):
    ax.annotate(rf"$\lambda={lam}$", (x, y),
                textcoords="offset points", xytext=(8, 6), fontsize=10)
# Highlight the default operating point lambda*=2
ax.scatter([0.0], [0.98], s=160, color=ORANGE, zorder=4,
           edgecolor="black", linewidth=0.6, label=r"default $\lambda^\star=2$")
ax.set_xlabel("Unauthorized high-risk exposure (per task)  \u2190 safer")
ax.set_ylabel("Benign task success  \u2192 better")
ax.set_xlim(-0.04, 0.7)
ax.set_ylim(0.93, 1.005)
ax.legend(loc="lower right", frameon=True)
ax.set_title("Safety\u2013success Pareto frontier (RACG)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pareto_frontier.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Injection success rate by method (adversarial track, lower better).
# TODO: replace with measured injection success rates.
# ---------------------------------------------------------------------------
methods = ["All tools", "Keyword\ntop-10", "State-aware",
           "Causal\nfrontier", "RACG\n(ours)"]
isr     = [0.71, 0.55, 0.58, 0.34, 0.00]   # injection success rate
colors  = [BLUE, BLUE, BLUE, BLUE, ORANGE]

fig, ax = plt.subplots(figsize=(5.6, 3.6))
bars = ax.bar(methods, isr, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, isr):
    ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Injection success rate  \u2190 lower is better")
ax.set_ylim(0, 0.8)
ax.set_title("Structural injection resistance by exposure method")
ax.axhline(0, color="black", lw=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "injection_by_method.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: High-risk attack surface over trajectory steps, by method.
# Visualizes capability minimization: RACG keeps the visible high-risk tool
# count at ~0 except at the single authorized step where the action is needed.
# TODO: replace with measured mean AS per step over the authorization-required
#       safety-stress tasks.
# ---------------------------------------------------------------------------
steps = [1, 2, 3, 4]   # search -> read(=authorize) -> draft -> send

series = {
    "All tools":       ([6, 6, 6, 6], BLUE,   "-o"),
    "State-aware":     ([2, 3, 3, 3], (0.45, 0.55, 0.70), "-^"),
    "Causal frontier": ([0, 1, 1, 1], (0.30, 0.45, 0.65), "-s"),
    "RACG (ours)":     ([0, 0, 0, 1], ORANGE, "-D"),
}

fig, ax = plt.subplots(figsize=(5.6, 3.6))
for label, (ys, color, style) in series.items():
    lw = 2.6 if "RACG" in label else 1.8
    ax.plot(steps, ys, style, color=color, lw=lw, markersize=7,
            markerfacecolor=(BLUEF if "RACG" not in label else color),
            markeredgecolor=color, label=label,
            zorder=(4 if "RACG" in label else 3))
# annotate the single authorized exposure for RACG
ax.annotate("authorized\nexposure", (4, 1), textcoords="offset points",
            xytext=(-6, 14), ha="center", fontsize=9, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1))
ax.set_xticks(steps)
ax.set_xticklabels(["1\nsearch", "2\nread\n(authorize)", "3\ndraft", "4\nsend"])
ax.set_xlabel("Trajectory step")
ax.set_ylabel("Visible high-risk tools  \u2190 smaller surface")
ax.set_ylim(-0.3, 6.6)
ax.set_title("High-risk attack surface over a send-email trajectory")
ax.legend(loc="center right", frameon=True, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "attack_surface_by_step.png"))
plt.close(fig)

print("Wrote:", os.listdir(OUT))
