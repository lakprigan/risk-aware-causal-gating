# RiskGate

A controlled benchmark for **Risk-Aware Causal Gating (RACG)**, the method in
*"Capability Minimization as a Safety Primitive."* The **deterministic track**
validates hypotheses **H1–H5** with no LLM API calls; an optional **real-LLM
validation track** (paper Sec. 7.1) drives actual models through the same
benchmark to confirm the structural prediction holds for real agents.

## Why a deterministic agent (and why also a real LLM)?

The default agent is a deterministic, **adversarially-compliant** heuristic
policy, not an LLM. This is a deliberate scientific choice: **H5 (structural
injection defense) is a claim about the action space**, so it must hold for
*any* agent — including a worst-case one that always obeys injections when the
target tool is visible. If the dangerous tool is gated out, even a fully
compliant agent cannot call it. Demonstrating ISR=0 against this worst case
*upper-bounds* ISR for any real model.

The **real-LLM track** complements (does not replace) this. It swaps the
deterministic policy for a real model while holding the gating layer fixed, and
checks that the model-driven high-risk-call rate under injection behaves as
predicted: zero under RACG (the tool is absent from `V_t`), nonzero under
all-tools and CMTF. See "Real-LLM validation" below.

## Layout

```
riskgate/
  riskgate/
    model.py      # Tool contract, Risk tiers, risk() penalty map (Eq. 6), Task, Injection
    registry.py   # 100-tool registry: core workflow tools + distractors (Table II)
    filters.py    # AllTools, KeywordTopK, StateAware, CausalFrontier (CMTF), RACG
    env.py        # MockAgent + LLMAgent + deterministic mocked execution + bounded loop
    tasks.py      # benign + safety-stress + high-risk-shortcut + injections
    llm.py        # provider abstraction: Anthropic, Bedrock, OpenAI-compatible, Stub
    runner.py     # deterministic track: runs everything, aggregates, validates H1-H5
  tests/          # pytest: structural guarantees behind H1-H5
  run.py          # deterministic entry point -> results.json + console summary
  llm_runner.py   # real-LLM validation entry point -> llm_results.json
  plot_results.py # regenerate ../figures/*.png from results.json
```

## Run (deterministic track)

```bash
cd riskgate
pip install -r requirements.txt
python run.py            # prints the metrics table + H1-H5 PASS/FAIL, writes results.json
python -m pytest -q      # structural unit tests
python plot_results.py   # regenerate figures/*.png from measured data
```

## Real-LLM validation (paper Sec. 7.1)

Drives a real model as the policy over the filter-produced visible tool set,
holding the gating layer (all-tools / CMTF / RACG) fixed. Measures the
model-driven high-risk-call rate under injection, benign authorization-required
completion, and exposure-at-attack.

```bash
# Offline wiring test (no SDK or API key needed):
python llm_runner.py --models stub

# Real models (install the matching SDK and set the provider's credentials):
pip install anthropic            # then: export ANTHROPIC_API_KEY=...
python llm_runner.py --models anthropic:claude-3-5-sonnet-latest

pip install boto3                # then: configure AWS creds + region
python llm_runner.py --models bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0

pip install openai               # then: export OPENAI_API_KEY=...
python llm_runner.py --models openai_compat:gpt-4o-mini

# Local model via an OpenAI-compatible server (Ollama / vLLM), no API cost:
export OPENAI_BASE_URL=http://localhost:11434/v1
python llm_runner.py --models openai_compat:llama3.1

# Compare several at once; missing SDKs/keys are skipped with a logged reason:
python llm_runner.py --models anthropic:claude-3-5-sonnet-latest openai_compat:gpt-4o-mini --subset 20
```

Writes `llm_results.json`. The **prediction** (and what to look for): under
RACG, `highrisk_call_rate_under_injection` and `exposure_at_attack` should be
`0.00` for **every** model; under all-tools and CMTF they should be nonzero and
model-dependent. The offline `stub` reproduces the structural pattern
(all-tools 1.00, CMTF 0.25, RACG 0.00), mirroring the deterministic ISR column.

## Hypotheses (mapped to metrics)

| H  | Claim | Signal |
|----|-------|--------|
| H1 | relevance/executability ≠ safety | baselines: AS>0 and UE>0 |
| H2 | risk-agnostic causal filtering insufficient | causal_frontier: AS↓ but UE>0, ISR>0 |
| H3 | RACG drives UE→0 at near-ceiling success | racg_lam2: UE=0, success≈causal |
| H4 | RACG does not over-filter | racg auth-required success ≈ causal |
| H5 | gating is a structural injection defense | racg ISR=0; baselines ISR>0 |

## The λ result

The λ sweep reproduces the **λ† crossover** from Eq. 8: RACG succeeds on the
high-risk-shortcut tasks only once λ is large enough (here, between 0.5 and 1)
to prefer the longer *authorized* path over the dangerous one-step shortcut.
Small λ fails closed (success drops); λ ≥ 1 routes safely. The default λ\*=2
sits comfortably past the crossover.

## Boundary condition for H5

H5 holds **only if** an injection cannot forge the authorization variable.
`Injection(forges_authorization=True)` exercises this documented failure mode:
if attacker-controlled content can set, e.g., `recipient_confirmed`, the gate
opens. Authorization provenance is therefore an explicit assumption, not an
emergent property (see paper §6.1 and Limitations).

## Regenerating figures and metrics

`results.json` contains the measured metrics. `plot_results.py` writes the
figures into `figures/` from that file; `llm_runner.py` writes the real-LLM
validation metrics to `llm_results.json`.
