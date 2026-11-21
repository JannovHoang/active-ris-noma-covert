"""PSO chuẩn hóa, ưu tiên nghiệm khả thi, cập nhật swarm đồng bộ.

Số sweeps bao gồm lần đánh giá khởi tạo: evaluations=swarm_size*sweeps.
Các hệ số mặc định là cấu hình pilot, không nhận là hệ số baseline gốc.
"""
from dataclasses import dataclass, field, fields
from pathlib import Path
import json
import numpy as np


@dataclass(frozen=True)
class Outcome:
    score: float  # Càng lớn càng tốt, thường là rate Bob.
    violation: float  # Tổng vi phạm chuẩn hóa, 0 nếu khả thi.
    feasible: bool
    metrics: dict = field(default_factory=dict)


def better(candidate, incumbent):
    if incumbent is None:
        return True
    if candidate.feasible != incumbent.feasible:
        return candidate.feasible
    if candidate.feasible:
        return candidate.score > incumbent.score
    if candidate.violation != incumbent.violation:
        return candidate.violation < incumbent.violation
    return candidate.score > incumbent.score


@dataclass(frozen=True)
class PSOSettings:
    swarm_size: int = 30
    sweeps: int = 60
    inertia_start: float = .90
    inertia_end: float = .40
    cognitive: float = 1.5
    social: float = 1.5
    velocity_max: float = .20
    initial_velocity_scale: float = 0.0

    def __post_init__(self):
        for name in ["swarm_size", "sweeps"]:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} phải là số nguyên dương")
        for name in ["inertia_start", "inertia_end", "cognitive", "social", "velocity_max", "initial_velocity_scale"]:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name == "velocity_max" and value == 0):
                raise ValueError(f"{name} ngoài miền hợp lệ")


def load_pso_settings(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {f.name for f in fields(PSOSettings)}
    unknown = set(data) - names - {"parameter_status", "notes"}
    missing = names - set(data) - {"initial_velocity_scale"}
    if unknown or missing:
        raise ValueError(f"PSO config: unknown={sorted(unknown)}, missing={sorted(missing)}")
    return PSOSettings(**{name: data[name] for name in names if name in data}), data


@dataclass(frozen=True)
class PSOResult:
    position: np.ndarray
    outcome: Outcome
    history: list
    evaluations: int
    feasible_evaluations: int


def optimize(objective, dimensions, settings, rng, initial_positions=None):
    if type(dimensions) is not int or dimensions < 1:
        raise ValueError("dimensions phải là số nguyên dương")
    S, T = settings.swarm_size, settings.sweeps
    positions = rng.uniform(0, 1, size=(S, dimensions))
    if initial_positions is not None:
        given = np.asarray(initial_positions, float)
        if given.ndim != 2 or given.shape[1] != dimensions or len(given) > S or not np.all(np.isfinite(given)) or np.any((given < 0) | (given > 1)):
            raise ValueError("initial_positions phải nằm trong [0,1] và đúng shape")
        positions[:len(given)] = given
    velocity = (rng.uniform(-settings.initial_velocity_scale, settings.initial_velocity_scale, size=positions.shape)
                if settings.initial_velocity_scale > 0 else np.zeros_like(positions))
    personal_positions = positions.copy()
    personal_outcomes = [None] * S
    global_position, global_outcome = None, None
    history, evaluations, feasible_count = [], 0, 0
    for sweep in range(T):
        inertia = None
        if sweep:
            # Có T-1 lần cập nhật; khi T>=3, w đi đúng từ start tới end.
            fraction = (sweep - 1) / max(1, T - 2)
            inertia = settings.inertia_start + fraction * (settings.inertia_end - settings.inertia_start)
            r1, r2 = rng.uniform(size=positions.shape), rng.uniform(size=positions.shape)
            velocity = (inertia * velocity
                        + settings.cognitive * r1 * (personal_positions - positions)
                        + settings.social * r2 * (global_position - positions))
            velocity = np.clip(velocity, -settings.velocity_max, settings.velocity_max)
            positions = np.clip(positions + velocity, 0, 1)
        current_feasible = 0
        for i in range(S):
            outcome = objective(positions[i].copy())
            if not isinstance(outcome, Outcome) or not np.isfinite(outcome.score) or not np.isfinite(outcome.violation) or outcome.violation < 0:
                raise ValueError("Objective phải trả Outcome hữu hạn")
            if bool(outcome.feasible) != (outcome.violation == 0):
                raise ValueError("feasible phải thống nhất với violation==0")
            evaluations += 1
            feasible_count += int(outcome.feasible)
            current_feasible += int(outcome.feasible)
            if better(outcome, personal_outcomes[i]):
                personal_positions[i] = positions[i]
                personal_outcomes[i] = outcome
        # Dùng cùng global best cũ cho toàn bộ lần cập nhật trước đó.
        for i, outcome in enumerate(personal_outcomes):
            if better(outcome, global_outcome):
                global_position = personal_positions[i].copy()
                global_outcome = outcome
        history.append({"sweep": sweep + 1, "evaluations": evaluations,
                        "inertia_used_for_update": inertia,
                        "gbest_feasible": bool(global_outcome.feasible),
                        "gbest_score": float(global_outcome.score),
                        "gbest_violation": float(global_outcome.violation),
                        "current_swarm_feasible_count": current_feasible})
    return PSOResult(global_position.copy(), global_outcome, history, evaluations, feasible_count)
