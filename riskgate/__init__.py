"""RiskGate: a controlled benchmark for Risk-Aware Causal Gating (RACG).

Validates hypotheses H1-H5 from the paper
"Capability Minimization as a Safety Primitive".

The default agent is a deterministic mock (no LLM): this is intentional. H5 is
a structural claim about the action space, so it must hold for *any* agent,
including an adversarially-compliant one that always obeys injections. An
optional real-LLM validation track (riskgate.llm + llm_runner.py) drives actual
models through the same benchmark to confirm the prediction holds for real
agents; it complements rather than replaces the deterministic track.
"""

__version__ = "0.2.0"
