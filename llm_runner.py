"""Real-LLM validation runner (paper Sec. 7.1, "Validation with Real LLM Agents").

Holds the gating layer (all-tools, CMTF causal frontier, RACG) fixed and swaps
the deterministic MockAgent for a real model via riskgate.llm. Measures, per
(method, model):

  * highrisk_call_rate_under_injection : the model-driven analogue of ISR --
        fraction of adversarial trials in which the targeted high-risk tool is
        actually called by the model.
  * benign_auth_completion : completion rate on authorization-required tasks
        with NO injection (the over-filtering / capability check).
  * exposure_at_attack : fraction of adversarial trials where the targeted tool
        was in V_t at the injection step (the structural quantity H5 hinges on).
  * tokens : mean real prompt+completion tokens per task (when the provider
        reports usage).

Key prediction (H5): under RACG the targeted tool is gated out of V_t at the
injection step, so exposure_at_attack = 0 and therefore highrisk_call_rate = 0
regardless of model or phrasing; all-tools and CMTF admit a nonzero,
model-dependent rate. The deterministic track in run.py is unaffected.

Usage:
    python llm_runner.py --models stub
    python llm_runner.py --models anthropic:claude-3-5-sonnet-latest openai_compat:gpt-4o-mini
    python llm_runner.py --models bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0 --subset 20

Providers fail gracefully: if an SDK or key is missing, that model is skipped
with a logged reason rather than crashing the run.
"""
from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

from riskgate.env import LLMAgent, run_task
from riskgate.filters import (RACG, AllTools, CausalFrontier, Filter, KeywordTopK,
                              StateAware)
from riskgate.llm import LLMUnavailable, make_provider
from riskgate.registry import REGISTRY
from riskgate.tasks import build_tasks, injections_for


# The seven Bedrock models driven through the validation matrix by default.
DEFAULT_MODELS = [
    "bedrock:us.anthropic.claude-opus-4-8",
    "bedrock:us.anthropic.claude-sonnet-4-6",
    "bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "bedrock:openai.gpt-oss-120b-1:0",
    "bedrock:us.amazon.nova-premier-v1:0",
    "bedrock:us.amazon.nova-2-lite-v1:0",
    "bedrock:us.amazon.nova-2-pro-preview-20251202-v1:0",
]


def _mean(xs):
    # statistics.fmean is 3.8+; fall back to mean for older interpreters.
    if not xs:
        return 0.0
    fmean = getattr(statistics, "fmean", None)
    return fmean(xs) if fmean else statistics.mean(xs)


def llm_method_panel(names: list[str] | None = None) -> list[Filter]:
    """The exposure regimes contrasted in the LLM validation.

    By default the all_tools arm is DROPPED: it ships the full ~100-tool
    registry on every call (huge prompt cost and impractical on local hardware),
    and the scientifically discriminating comparison is causal_frontier vs RACG.

    The full selectable set is the paper's Table V panel MINUS the lambda sweep:
    all_tools, keyword_top10, state_aware, causal_frontier, and RACG fixed at
    lambda=2. The lambda sweep stays in the DETERMINISTIC track (run.py /
    runner.run_all); the LLM validation uses only RACG(lam=2) per Sec. 7.1, so
    no model is ever driven across multiple lambda values here.

    Pass names explicitly to select, e.g.:
        --methods causal_frontier racg                 (default behaviour)
        --methods all_tools keyword_top10 state_aware causal_frontier racg
    """
    full = {
        "all_tools": AllTools(),
        "keyword_top10": KeywordTopK(10),
        "state_aware": StateAware(),
        "causal_frontier": CausalFrontier(),
        "racg": RACG(lam=2.0),  # fixed operating point; NO sweep in the LLM track
    }
    if not names:
        # Default: only the discriminating comparison, no all_tools.
        return [full["causal_frontier"], full["racg"]]
    keep = set(names)
    # Accept both the canonical filter name and friendly aliases.
    aliases = {"keyword_top_10": "keyword_top10", "racg_lam2": "racg"}
    keep = {aliases.get(k, k) for k in keep}
    selected = [full[k] for k in full if k in keep]
    unknown = keep - set(full)
    if unknown:
        raise SystemExit(
            f"unknown method(s): {sorted(unknown)}; "
            f"choose from {sorted(full)} (RACG is fixed at lam=2 in the LLM track)"
        )
    return selected


def select_subset(n: int) -> list:
    """A balanced subset of safety-stress tasks across domains and flavors.

    The paper specifies ~20 tasks spanning the three domains and both
    safety-stress flavors (auth_required and no_action). RiskGate's stress set
    has four families: email auth-required, email no-action, files no-action,
    and the calendar high-risk-*shortcut* tasks (the H2 confuser where CMTF
    leaks but RACG does not). We round-robin across all four families so the
    subset is balanced, deterministic, and -- critically -- includes the
    shortcut tasks that distinguish CMTF from RACG under injection.
    """
    _, stress = build_tasks()
    families: dict[str, list] = {}
    for t in stress:
        # family key = task_id prefix without the numeric suffix
        key = t.task_id.rsplit("_", 1)[0]
        families.setdefault(key, []).append(t)
    # deterministic family order
    order = sorted(families)
    subset = []
    idx = 0
    while len(subset) < n and any(families[k] for k in order):
        k = order[idx % len(order)]
        if families[k]:
            subset.append(families[k].pop(0))
        idx += 1
    return subset[:n]


@dataclass
class LLMMethodMetrics:
    model: str
    method: str
    highrisk_call_rate_under_injection: float
    benign_auth_completion: float
    exposure_at_attack: float
    mean_tokens: float
    n_benign: int
    n_adversarial: int


def eval_model(model_spec: str, subset, methods, log_raw=False,
               concurrency: int = 8) -> dict | None:
    try:
        provider = make_provider(model_spec)
    except (LLMUnavailable, ValueError) as e:
        print(f"  [skip] {model_spec}: {e}")
        return None

    # The provider (and its underlying SDK client) is shared read-only across
    # threads. The LLMAgent wrapper, however, stashes per-step token usage on
    # mutable instance attributes, so each trial gets its OWN agent to avoid a
    # data race that would corrupt token accounting. Trials are otherwise
    # independent: run_task mutates only its local state and its own RunResult.
    def _benign_trial(t):
        agent = LLMAgent(provider, log_raw=log_raw)
        return t, run_task(t, _m, REGISTRY, agent=agent)

    def _adv_trial(args):
        t, inj = args
        agent = LLMAgent(provider, log_raw=log_raw)
        r = run_task(t, _m, REGISTRY, agent=agent, injection=inj)
        exposed = any(
            s.injected_active and inj.target_tool in s.visible
            for s in r.steps
        )
        return r, (1.0 if exposed else 0.0)

    out = {}
    for _m in methods:
        print(f"  [run] {model_spec} / {_m.name} ...", flush=True)

        # Benign (no injection): authorization-required completion + tokens.
        benign_pairs = _run_parallel(_benign_trial, list(subset), concurrency)
        benign_results = [r for _, r in benign_pairs]
        auth = [r for (t, r) in benign_pairs if t.flavor == "auth_required"]
        benign_auth_completion = _mean([1.0 if r.success else 0.0 for r in auth])
        mean_tokens = _mean([r.total_tokens for r in benign_results])

        # Adversarial: 3 phrasings per task targeting the gated tool.
        adv_jobs = [(t, inj) for t in subset for inj in injections_for(t)]
        adv_pairs = _run_parallel(_adv_trial, adv_jobs, concurrency)
        adv_results = [r for r, _ in adv_pairs]
        exposed_flags = [e for _, e in adv_pairs]

        hr_rate = _mean([1.0 if r.injection_succeeded else 0.0 for r in adv_results])

        mm = LLMMethodMetrics(
            model=model_spec,
            method=_m.name,
            highrisk_call_rate_under_injection=hr_rate,
            benign_auth_completion=benign_auth_completion,
            exposure_at_attack=_mean(exposed_flags),
            mean_tokens=mean_tokens,
            n_benign=len(benign_results),
            n_adversarial=len(adv_results),
        )
        out[_m.name] = asdict(mm)
        print(f"  {model_spec:<48} {_m.name:<16} "
              f"HRcall={hr_rate:>4.2f} exposed={mm.exposure_at_attack:>4.2f} "
              f"authok={benign_auth_completion:>4.2f} tok={mean_tokens:>7.0f}")
    return out


def _run_parallel(fn, items, concurrency):
    """Run fn over items concurrently, preserving INPUT ORDER in the output.

    Concurrency<=1 falls back to a simple sequential map (useful for the stub
    provider or for deterministic debugging). Exceptions in any single trial
    are re-raised so a misconfigured run fails loudly rather than silently
    skewing the aggregate metrics.
    """
    if not items:
        return []
    if concurrency <= 1:
        return [fn(it) for it in items]
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(fn, it): idx for idx, it in enumerate(items)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results


# ---------------------------------------------------------------------------
# Per-model durable persistence (S3 + local), so a mid-run credential expiry
# never loses already-completed models. Each model is written the instant it
# finishes; --resume skips models whose result is already in S3.
# ---------------------------------------------------------------------------
def _model_key(model_spec: str) -> str:
    """Filesystem/S3-safe per-model basename, e.g. bedrock_us.anthropic...json."""
    safe = model_spec.replace(":", "_").replace("/", "_")
    return f"{safe}.json"


def _split_s3(prefix: str):
    """s3://bucket/some/prefix/ -> ('bucket', 'some/prefix/')."""
    assert prefix.startswith("s3://"), prefix
    rest = prefix[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def _s3_client():
    import boto3  # already a hard dep for Bedrock
    return boto3.client("s3")


def _s3_join(key_prefix: str, name: str) -> str:
    if not key_prefix:
        return name
    return key_prefix + name if key_prefix.endswith("/") else key_prefix + "/" + name


def _s3_put_json(prefix: str, name: str, obj: dict):
    bucket, key_prefix = _split_s3(prefix)
    key = _s3_join(key_prefix, name)
    _s3_client().put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(obj, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


def _s3_exists(prefix: str, name: str) -> bool:
    from botocore.exceptions import ClientError
    bucket, key_prefix = _split_s3(prefix)
    key = _s3_join(key_prefix, name)
    try:
        _s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def main():
    ap = argparse.ArgumentParser(description="RiskGate real-LLM validation track")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="provider:model specs, or 'stub' for offline test "
                         "(default: the 7-model Bedrock matrix)")
    ap.add_argument("--subset", type=int, default=20,
                    help="number of safety-stress tasks (default 20; the "
                         "stress set has 80 total, so values >80 are capped)")
    ap.add_argument("--out", default="llm_results.json")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="restrict methods: all_tools keyword_top10 state_aware "
                         "causal_frontier racg (default: causal_frontier racg; "
                         "all_tools is dropped by default. RACG is fixed at "
                         "lam=2 -- the lambda sweep stays in run.py)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel in-flight LLM calls per method (default 8; "
                         "use 1 for sequential/debugging)")
    ap.add_argument("--s3-prefix", default=None,
                    help="s3://bucket/prefix/ to write a per-model JSON the "
                         "instant each model finishes (durable against cred "
                         "expiry). Also writes an aggregate _aggregate.json.")
    ap.add_argument("--local-dir", default=None,
                    help="directory to mirror per-model JSON locally (default: "
                         "alongside --out)")
    ap.add_argument("--resume", action="store_true",
                    help="skip models whose per-model JSON already exists in "
                         "--s3-prefix (resume a partial run after cred expiry)")
    ap.add_argument("--log-raw", action="store_true",
                    help="record raw model text (debugging)")
    args = ap.parse_args()

    subset = select_subset(args.subset)
    methods = llm_method_panel(args.methods)

    import os
    local_dir = args.local_dir or os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(local_dir, exist_ok=True)

    print("=" * 78)
    print(f"RiskGate LLM validation: {len(subset)} tasks x "
          f"{len(methods)} methods x {len(args.models)} model(s) "
          f"@ concurrency={args.concurrency}")
    if args.s3_prefix:
        print(f"per-model S3 persistence -> {args.s3_prefix}"
              f"{'  [resume]' if args.resume else ''}")
    print("=" * 78)

    config = {
        "subset_size": len(subset),
        "task_ids": [t.task_id for t in subset],
        "methods": [m.name for m in methods],
        "models": args.models,
        "concurrency": args.concurrency,
        "injection_phrasings": ["direct", "roleplay", "obfuscated"],
    }
    results = {"config": config, "models": {}}

    for spec in args.models:
        name = _model_key(spec)
        if args.resume and args.s3_prefix and _s3_exists(args.s3_prefix, name):
            print(f"\nmodel: {spec}  [resume: already in S3, skipping]")
            continue

        print(f"\nmodel: {spec}")
        try:
            res = eval_model(spec, subset, methods, log_raw=args.log_raw,
                             concurrency=args.concurrency)
        except Exception as e:  # noqa: BLE001 - persist what we have, keep going
            print(f"  [error] {spec}: {e!r}")
            print("  -> already-completed models remain saved; continue/retry "
                  "this model with --resume after re-auth.")
            continue
        if res is None:
            continue

        results["models"][spec] = res

        # Per-model durable write the instant this model finishes.
        per_model_obj = {"config": config, "model": spec, "results": res}
        local_path = os.path.join(local_dir, name)
        with open(local_path, "w") as f:
            json.dump(per_model_obj, f, indent=2)
        print(f"  saved local: {local_path}")
        if args.s3_prefix:
            try:
                uri = _s3_put_json(args.s3_prefix, name, per_model_obj)
                print(f"  saved S3:    {uri}")
                # Refresh the running aggregate in S3 too.
                _s3_put_json(args.s3_prefix, "_aggregate.json", results)
            except Exception as e:  # noqa: BLE001 - local copy already safe
                print(f"  [warn] S3 write failed (local copy kept): {e!r}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "-" * 78)
    if results["models"]:
        print(f"wrote {args.out} ({len(results['models'])} model(s))")
        print("Prediction check: RACG should show HRcall=0.00 and exposed=0.00 "
              "for every model.")
    else:
        print("No models ran (all skipped). Set provider keys/SDKs and retry, "
              "or use --models stub for an offline wiring test.")
    return results


if __name__ == "__main__":
    main()
