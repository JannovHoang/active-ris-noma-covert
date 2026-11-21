"""Bounded, search-only proposal archive observed without changing the PSO.

The exact GNN initial particle is protected in warm pools. Cold pools receive
the same five-candidate cap. No independent-selection information enters here.
"""
from functools import cmp_to_key
import time

import numpy as np

from .graph_data import original_allocation
from .objective import decode_k2
from .pso import better, optimize


def position_key(position):
    return tuple(np.asarray(position, float))


def compare(a, b):
    if better(a["outcome"], b["outcome"]):
        return -1
    if better(b["outcome"], a["outcome"]):
        return 1
    return a["evaluation"] - b["evaluation"]


class CandidateArchive:
    def __init__(self, target, slacks, limit=5, protected_position=None):
        if limit != 5 or len(slacks) != 2:
            raise ValueError("The declared 6g archive uses five slots and two extra margins")
        self.target, self.slacks, self.limit = target, slacks, limit
        self.protected_key = None if protected_position is None else position_key(protected_position)
        self.protected = self.safest = None
        self.buffered = [None, None]
        self.top, self.count = [], 0

    def observe(self, position, outcome):
        self.count += 1
        item = {"position": np.asarray(position).copy(), "outcome": outcome, "evaluation": self.count,
                "key": position_key(position)}
        if self.protected_key == item["key"] and self.protected is None:
            self.protected = item
        existing = next((x for x in self.top if x["key"] == item["key"]), None)
        if existing is None:
            self.top = sorted([*self.top, item], key=cmp_to_key(compare))[:self.limit]
        violations = outcome.metrics["normalized_violations"]
        reliable = all(value == 0 for name, value in violations.items() if name != "covertness")
        dep = outcome.metrics["dep_min"]
        if reliable:
            if self.safest is None or (dep, outcome.score) > (self.safest["outcome"].metrics["dep_min"], self.safest["outcome"].score):
                self.safest = item
            for index, slack in enumerate(self.slacks):
                if dep >= self.target + slack and (self.buffered[index] is None or outcome.score > self.buffered[index]["outcome"].score):
                    self.buffered[index] = item

    def pool(self):
        if not self.top or self.protected_key is not None and self.protected is None:
            raise ValueError("Missing evaluated proposals or missing exact GNN initial particle")
        preferred = [self.protected, self.top[0], *self.buffered, self.safest, *self.top]
        selected, seen = [], set()
        for item in preferred:
            if item is not None and item["key"] not in seen:
                selected.append(item)
                seen.add(item["key"])
                if len(selected) == self.limit:
                    break
        return selected


def capture_candidates(objective, settings, random, initial, budgets, target, slacks, order, bounds,
                       prefix_seconds=0.):
    """Observe evaluated points only; each budget is a prefix of one trajectory."""
    protected = None if initial is None else initial[0]
    archive = CandidateArchive(target, slacks, 5, protected)
    snapshots = []
    started = time.perf_counter()
    marks = set(budgets)

    def evaluate(position):
        outcome = objective(position)
        archive.observe(position, outcome)
        if archive.count in marks:
            pool = []
            for item in archive.pool():
                mb, public, s = decode_k2(item["position"], bounds)
                allocation, _ = original_allocation({"mb": mb, "public": public, "s": s}, order, bounds)
                pool.append({"candidate_id": f"v{item['evaluation']:06d}", "allocation": allocation.tolist(),
                             "evaluation": item["evaluation"], "search_bob_rate": item["outcome"].score,
                             "search_dep_gamma": item["outcome"].metrics["dep_min"],
                             "search_feasible": item["outcome"].feasible})
            snapshots.append({"budget": archive.count, "candidates": pool,
                              "raw_candidate_id": f"v{archive.top[0]['evaluation']:06d}",
                              "search_wall_seconds": prefix_seconds + time.perf_counter() - started})
        return outcome

    result = optimize(evaluate, 3, settings, random, initial)
    if (result.evaluations != budgets[-1] or archive.count != result.evaluations
            or [r["budget"] for r in snapshots] != budgets
            or position_key(result.position) != archive.top[0]["key"]):
        raise ArithmeticError("Archived prefixes differ from the original PSO trajectory")
    for row in snapshots:
        history = result.history[row["budget"] // settings.swarm_size - 1]
        raw = next(c for c in row["candidates"] if c["candidate_id"] == row["raw_candidate_id"])
        if raw["search_bob_rate"] != history["gbest_score"] or raw["search_feasible"] != history["gbest_feasible"]:
            raise ArithmeticError("Raw PSO baseline differs from its archived global best")
    return snapshots
