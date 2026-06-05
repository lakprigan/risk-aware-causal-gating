"""RiskGate: a controlled benchmark for Risk-Aware Causal Gating (RACG).

Validates hypotheses H1-H5 from the paper
"Capability Minimization as a Safety Primitive".

The agent is a deterministic mock (no LLM): this is intentional. H5 is a
structural claim about the action space, so it must hold for *any* agent,
including an adversarially-compliant one that always obeys injections.
"""

__version__ = "0.1.0"
