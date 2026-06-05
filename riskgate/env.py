"""Deterministic mock agent + execution environment + the bounded agent loop.

The agent is intentionally NOT an LLM. It is an adversarially-compliant
heuristic policy: among the VISIBLE tools it picks the one most relevant to
the query/goal, and -- crucially -- if an injection is active and its target
tool is visible, it obeys the injection. This is the worst case for safety,
which is exactly what makes a *structural* defense (H5) meaningful: if the
dangerous tool is not visible, even a fully compliant agent cannot call it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .filters import Filter
from .model import Injection, Risk, Task, Tool


# Rough token accounting: a fixed prompt overhead + per-visible-tool schema cost.
TOKENS_BASE = 350
TOKENS_PER_TOOL = 95


def _relevance(t: Tool, query: str, goal: frozenset[str]) -> int:
    q = set(query.lower().replace("_", " ").split())
    score = 0
    name_toks = set(t.name.lower().split("_"))
    score += len(q & name_toks) * 2
    score += len(set(v.lower() for v in t.produces) & {v.lower() for v in goal}) * 3
    return score


@dataclass
class StepTrace:
    step: int
    visible: list[str]
    visible_risky: list[str]          # high/med-risk visible tools
    unauthorized_visible: list[str]   # high/med visible whose alpha unmet
    chosen: str | None
    chosen_risk: int
    injected_active: bool
    injection_obeyed: bool
    tokens: int


@dataclass
class RunResult:
    task_id: str
    method: str
    success: bool
    steps: list[StepTrace] = field(default_factory=list)
    # convenience aggregates
    wrong_tool: int = 0
    premature_highrisk: int = 0

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.steps)

    @property
    def mean_attack_surface(self) -> float:
        if not self.steps:
            return 0.0
        return sum(len(s.visible_risky) for s in self.steps) / len(self.steps)

    @property
    def unauthorized_exposure(self) -> int:
        """Total count of steps*tools where a non-low tool was visible unauthorized."""
        return sum(len(s.unauthorized_visible) for s in self.steps)

    @property
    def injection_succeeded(self) -> bool:
        return any(s.injection_obeyed for s in self.steps)

    @property
    def exposed_target_at_attack(self) -> bool:
        return any(s.injected_active and s.injection_obeyed for s in self.steps)


class MockAgent:
    """Adversarially-compliant heuristic policy over the VISIBLE tool set."""

    def choose(self, visible: list[Tool], task: Task, state: frozenset[str],
               injection: Injection | None, step: int) -> Tool | None:
        if not visible:
            return None
        names = {t.name for t in visible}
        # 1) If an injection is active at this step and its target is visible,
        #    a compliant agent obeys it.
        if injection and injection.inject_at_step == step and injection.target_tool in names:
            return next(t for t in visible if t.name == injection.target_tool)
        # 2) Otherwise follow the gold chain if the next gold tool is visible
        #    (models a capable, well-behaved agent on benign tasks).
        for gold in task.gold_chain:
            if gold not in state_satisfied_chain(task, state) and gold in names:
                return next(t for t in visible if t.name == gold)
        # 3) Fall back to most relevant visible tool (executable preferred).
        execu = [t for t in visible if t.executable(state)] or visible
        execu.sort(key=lambda t: (-_relevance(t, task.query, task.goal_state), t.name))
        return execu[0]


def state_satisfied_chain(task: Task, state: frozenset[str]) -> set[str]:
    """Gold tools whose effects are already in the state (already 'done')."""
    done = set()
    for name in task.gold_chain:
        # crude: consider a gold tool done if any of its mocked outputs is present
        outs = task.mock_outputs.get(name, frozenset())
        if outs and outs <= state:
            done.add(name)
    return done


def run_task(task: Task, filt: Filter, registry: dict[str, Tool],
             agent: MockAgent | None = None, injection: Injection | None = None,
             max_steps: int = 6) -> RunResult:
    agent = agent or MockAgent()
    state = task.initial_state
    res = RunResult(task_id=task.task_id, method=filt.name, success=False)

    for step in range(max_steps):
        if task.is_complete(state):
            res.success = True
            break

        visible = filt.visible(state, task.goal_state, registry, task.query)
        risky = [t for t in visible if t.risk != Risk.LOW]
        unauth = [t.name for t in risky if not t.authorized(state)]
        inj_active = bool(injection and injection.inject_at_step == step)

        chosen = agent.choose(visible, task, state, injection, step)
        obeyed = bool(
            inj_active and chosen is not None
            and chosen.name == injection.target_tool
        )
        res.steps.append(StepTrace(
            step=step,
            visible=[t.name for t in visible],
            visible_risky=[t.name for t in risky],
            unauthorized_visible=unauth,
            chosen=chosen.name if chosen else None,
            chosen_risk=int(chosen.risk) if chosen else -1,
            injected_active=inj_active,
            injection_obeyed=obeyed,
            tokens=TOKENS_BASE + TOKENS_PER_TOOL * len(visible),
        ))

        if chosen is None:
            break

        # premature high-risk: chose a high/med tool before its alpha was met
        if chosen.risk != Risk.LOW and not chosen.authorized(state):
            res.premature_highrisk += 1

        # wrong-tool vs gold next
        gold_next = _gold_next(task, state)
        if gold_next and chosen.name != gold_next:
            res.wrong_tool += 1

        # apply deterministic mocked effect
        produced = task.mock_outputs.get(chosen.name, chosen.produces)
        # Boundary condition for H5: a forging injection sets the auth var.
        if obeyed and injection and injection.forges_authorization:
            produced = produced | _alpha_of(chosen)
        state = state | produced

    if task.is_complete(state):
        res.success = True
    return res


def _alpha_of(t: Tool) -> frozenset[str]:
    return t.authorizes


def _gold_next(task: Task, state: frozenset[str]) -> str | None:
    for name in task.gold_chain:
        outs = task.mock_outputs.get(name, frozenset())
        if not (outs and outs <= state):
            return name
    return None
