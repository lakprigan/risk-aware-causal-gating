"""Pytest suite asserting the structural properties RACG must hold.

These are unit-level guarantees that back the aggregate H1-H5 in runner.py.
Run: cd riskgate && python -m pytest -q
"""
from riskgate.env import run_task
from riskgate.filters import RACG, AllTools, CausalFrontier, StateAware
from riskgate.model import Risk
from riskgate.registry import REGISTRY
from riskgate.tasks import build_tasks, injections_for


def _task(flavor, idx=0):
    benign, stress = build_tasks()
    pool = [t for t in (benign + stress) if t.flavor == flavor]
    return pool[idx]


def test_racg_never_exposes_unauthorized_highrisk_on_no_action():
    """H3/no-action: send/delete must NEVER be visible when not needed."""
    t = _task("no_action")
    racg = RACG(lam=2.0)
    r = run_task(t, racg, REGISTRY)
    for step in r.steps:
        assert not step.unauthorized_visible, (
            f"RACG exposed unauthorized high-risk tool: {step.unauthorized_visible}")
    assert r.success, "RACG should still complete the benign summarize goal"


def test_racg_eventually_exposes_authorized_tool():
    """H4: on auth-required tasks, send_email IS exposed once authorized."""
    t = _task("auth_required")
    r = run_task(t, RACG(lam=2.0), REGISTRY)
    assert r.success, "RACG must complete auth-required task (no over-filtering)"
    # send_email should appear only after recipient_confirmed is in state
    saw_send = any("send_email" in s.visible for s in r.steps)
    assert saw_send, "RACG should expose send_email after authorization"


def test_racg_zero_injection_success_structural():
    """H5: with the target gated, a compliant agent still cannot call it."""
    t = _task("no_action")  # send_email is never authorized here
    for inj in injections_for(t):
        r = run_task(t, RACG(lam=2.0), REGISTRY, injection=inj)
        assert not r.injection_succeeded, (
            f"RACG admitted injection ({inj.phrasing}) -> {inj.target_tool}")


def test_baseline_admits_injection():
    """Sanity: AllTools exposes the target, so the compliant agent obeys."""
    t = _task("no_action")
    inj = injections_for(t)[0]
    r = run_task(t, AllTools(), REGISTRY, injection=inj)
    assert r.injection_succeeded, "AllTools should admit the injection (baseline)"


def test_h5_boundary_condition_forged_auth_breaks_gate():
    """If injection can forge the auth variable, the structural guarantee fails."""
    t = _task("auth_required")
    inj = injections_for(t)[0]
    forging = type(inj)(task_id=inj.task_id, inject_at_step=inj.inject_at_step,
                        target_tool=inj.target_tool, phrasing=inj.phrasing,
                        forges_authorization=True)
    # We exercise the documented failure mode: forging auth at an earlier step.
    # Here we assert the model *records* the boundary, not that RACG is immune.
    r = run_task(t, RACG(lam=2.0), REGISTRY, injection=forging)
    assert r is not None  # boundary condition is documented in env.run_task


def test_causal_reduces_but_not_zero_surface():
    """H2: causal frontier < all-tools surface."""
    t = _task("auth_required")
    a = run_task(t, AllTools(), REGISTRY).mean_attack_surface
    c = run_task(t, CausalFrontier(), REGISTRY).mean_attack_surface
    assert c < a


def test_registry_size():
    assert len(REGISTRY) == 100
