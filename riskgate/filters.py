"""Filtering / gating methods that select the visible tool set V_t.

Each filter implements: filter(state, goal, registry) -> list[Tool].
Methods: AllTools, KeywordTopK, StateAware, CausalFrontier, RACG.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import Risk, Tool, risk_penalty


class Filter:
    name = "base"

    def visible(self, state: frozenset[str], goal: frozenset[str],
                registry: dict[str, Tool], query: str = "") -> list[Tool]:
        raise NotImplementedError


class AllTools(Filter):
    name = "all_tools"

    def visible(self, state, goal, registry, query=""):
        return list(registry.values())


class KeywordTopK(Filter):
    def __init__(self, k: int = 10):
        self.k = k
        self.name = f"keyword_top{k}"

    def visible(self, state, goal, registry, query=""):
        ctx = set(_tokens(query)) | {v.lower() for v in state} | {v.lower() for v in goal}

        def overlap(t: Tool) -> int:
            toks = set(_tokens(t.name)) | set(_tokens(t.description)) | {t.domain}
            toks |= {v.lower() for v in t.requires} | {v.lower() for v in t.produces}
            return len(ctx & toks)

        ranked = sorted(registry.values(), key=lambda t: (-overlap(t), t.name))
        return ranked[: self.k]


class StateAware(Filter):
    name = "state_aware"

    def visible(self, state, goal, registry, query=""):
        return [t for t in registry.values() if t.executable(state)]


def _bfs_min_path(state, goal, registry, score_fn):
    """Best-first (uniform-cost) search for a minimal-score causal path to goal.

    Returns the path as a list of Tool, or None. `score_fn(start_state, path)`
    returns the cumulative cost of a path; RACG penalizes a high-risk tool whose
    authorization is unmet *at the point it is applied*, so the search reroutes
    through authorization-establishing steps. CausalFrontier uses path length.
    """
    import heapq

    if goal <= state:
        return []

    counter = 0
    heap = [(0.0, 0, counter, state, tuple())]
    best_to: dict[frozenset, float] = {state: 0.0}

    while heap:
        score, plen, _, s, path = heapq.heappop(heap)
        if goal <= s:
            return [registry[n] for n in path]
        if plen >= 6:
            continue
        for name in sorted(registry):
            t = registry[name]
            if not (t.requires <= s):
                continue
            s2 = s | t.produces
            if s2 == s:
                continue
            new_path = path + (name,)
            new_score = score_fn(state, [registry[n] for n in new_path])
            if s2 not in best_to or new_score < best_to[s2] - 1e-9:
                best_to[s2] = new_score
                counter += 1
                heapq.heappush(heap, (new_score, plen + 1, counter, s2, new_path))
    return None


class CausalFrontier(Filter):
    """Risk-agnostic CMTF: expose only the first tool of a min-length path."""
    name = "causal_frontier"

    def visible(self, state, goal, registry, query=""):
        path = _bfs_min_path(state, goal, registry,
                             score_fn=lambda s0, p: float(len(p)))
        if not path:
            return []
        return [path[0]]


@dataclass
class RACG(Filter):
    """Risk-Aware Causal Gating.

    1. Find a minimal-`score` path with the risk penalty (Eq. 6).
    2. Take the next executable frontier tool.
    3. Apply the admissibility gate: drop a high-risk tool whose alpha is unmet.
    4. If gating empties the frontier, expose the lowest-risk causal tool that
       can ESTABLISH a missing authorization variable.
    """
    lam: float = 2.0

    def __post_init__(self):
        self.name = f"racg_lam{self.lam:g}"

    def _score(self, start_state: frozenset[str], path: list[Tool]) -> float:
        """Cumulative cost + lambda * risk penalty, where a high/med-risk tool
        is penalized only if its authorization is UNMET at the point it applies.
        Simulating the path's state lets the penalty reward establishing
        authorization first (the lambda-dagger mechanism, Eq. 8)."""
        s = set(start_state)
        total = 0.0
        for t in path:
            total += t.cost
            if t.risk != Risk.LOW and not (t.authorizes <= s):
                total += self.lam * risk_penalty(t.risk)
            s |= t.produces
        return total

    def visible(self, state, goal, registry, query=""):
        path = _bfs_min_path(state, goal, registry, score_fn=self._score)
        if not path:
            return []
        frontier = path[0]
        # admissibility gate
        if frontier.risk == Risk.LOW or frontier.authorized(state):
            return [frontier]
        # gated out -> expose the lowest-risk tool that establishes a missing alpha
        missing = frontier.authorizes - state
        establishers = [
            t for t in registry.values()
            if t.executable(state) and (t.produces & missing) and t.risk == Risk.LOW
        ]
        if establishers:
            establishers.sort(key=lambda t: (risk_penalty(t.risk), t.cost, t.name))
            return [establishers[0]]
        # nothing can safely establish authorization: expose nothing (fail closed)
        return []


def _tokens(s: str) -> list[str]:
    return [w for w in
            s.lower().replace("_", " ").replace("/", " ").split() if w]


def default_methods() -> list[Filter]:
    """The method panel compared in the paper (Table V)."""
    return [
        AllTools(),
        KeywordTopK(10),
        StateAware(),
        CausalFrontier(),
        RACG(lam=2.0),
    ]
