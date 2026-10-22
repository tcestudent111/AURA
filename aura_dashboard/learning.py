from __future__ import annotations

import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from .data import recommend_model
from .models import LOCAL_MODEL_NAMES, MODEL_NAMES, available_model_names


LEARNING_RATE = 0.2
FALLBACK_PENALTY = -0.25


class QTable:
    def __init__(self, values: dict[str, dict[str, float]] | None = None) -> None:
        self.values = values or {}

    def get(self, state: str, action: str) -> float:
        return self.values.get(state, {}).get(action, 0.0)

    def select(
        self,
        state: str,
        epsilon: float = 0.0,
        rng: random.Random | None = None,
        available_actions: tuple[str, ...] | None = None,
    ) -> str:
        if not 0 <= epsilon <= 1:
            raise ValueError("Exploration rate must be between zero and one.")
        actions = available_actions or available_model_names()
        if not actions or any(action not in MODEL_NAMES for action in actions):
            raise ValueError("At least one supported forecasting model must be available.")
        generator = rng or random.Random()
        if generator.random() < epsilon:
            return generator.choice(actions)
        preferred = recommend_model(state)
        scores = {model: self.get(state, model) for model in actions}
        best_score = max(scores.values())
        if best_score == 0 and all(score == 0 for score in scores.values()):
            return preferred if preferred in scores else actions[0]
        if preferred in scores and scores[preferred] == best_score:
            return preferred
        return next(model for model in actions if scores[model] == best_score)

    def update(self, state: str, action: str, reward: float) -> float:
        if action not in MODEL_NAMES:
            raise ValueError(f"Unknown model action: {action}")
        if not math.isfinite(reward):
            raise ValueError("Reward must be finite.")
        actions = self.values.setdefault(state, {})
        old_value = actions.get(action, 0.0)
        new_value = old_value + LEARNING_RATE * (reward - old_value)
        actions[action] = new_value
        return new_value

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {state: dict(actions) for state, actions in self.values.items()}

    @classmethod
    def from_dict(cls, payload: Any) -> "QTable":
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Q-table file has an unsupported or missing version.")
        states = payload.get("states")
        if not isinstance(states, dict):
            raise ValueError("Q-table file has an invalid states object.")
        values: dict[str, dict[str, float]] = {}
        for state, actions in states.items():
            if not isinstance(state, str) or not isinstance(actions, dict):
                raise ValueError("Q-table contains an invalid state.")
            values[state] = {}
            for action, score in actions.items():
                if action not in MODEL_NAMES or not isinstance(score, (float, int)):
                    raise ValueError(f"Q-table contains an invalid action or value for {state}.")
                if not math.isfinite(float(score)):
                    raise ValueError(f"Q-table contains a non-finite value for {state}/{action}.")
                values[state][action] = float(score)
        return cls(values)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"version": 1, "states": self.as_dict()}, indent=2)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as temporary:
                temporary.write(payload)
                temporary_path = temporary.name
            os.replace(temporary_path, path)
        except OSError:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise

    @classmethod
    def load(cls, path: Path) -> "QTable":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Q-table file is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)


def q_table_path() -> Path:
    configured = os.environ.get("AURA_Q_TABLE_PATH")
    return Path(configured).expanduser() if configured else Path.home() / ".aura_rl" / "q_table.json"


def calculate_reward(
    metrics: dict[str, float] | None,
    workload_scale: float,
    fallback: bool = False,
) -> float:
    if fallback:
        return FALLBACK_PENALTY
    if metrics is None or not math.isfinite(workload_scale) or workload_scale < 0:
        raise ValueError("Valid metrics and a finite, non-negative workload scale are required.")
    mae = metrics.get("MAE")
    if mae is None or not math.isfinite(mae) or mae < 0:
        raise ValueError("Metrics must include a finite, non-negative MAE.")
    normalized_mae = mae / max(workload_scale, 1e-9)
    return 1.0 / (1.0 + normalized_mae)
