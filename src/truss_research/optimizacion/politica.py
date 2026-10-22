"""Política Double DQN con puntuación compartida e invariancia al orden."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
import numpy.typing as npt
import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from truss_research.optimizacion.contratos import (
    BoolArray,
)

type Float32Array = npt.NDArray[np.float32]


@dataclass(frozen=True)
class DQNParameters:
    transitions: int = 50_000
    hidden_width: int = 64
    learning_rate: float = 0.001
    batch_size: int = 64
    replay_size: int = 50_000
    warmup: int = 1_000
    target_interval: int = 500
    epsilon_steps: int = 30_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    discount: float = 1.0
    validation_interval: int = 5_000
    gradient_clip: float = 10.0
    train_frequency: int = 4
    gradient_steps: int = 1

    def __post_init__(self) -> None:
        positive = (
            self.transitions,
            self.hidden_width,
            self.batch_size,
            self.replay_size,
            self.target_interval,
            self.epsilon_steps,
            self.validation_interval,
            self.train_frequency,
            self.gradient_steps,
        )
        if any(type(value) is not int or value < 1 for value in positive):
            raise ValueError(
                "Los tamaños y períodos de DQN deben ser enteros positivos"
            )
        if (
            type(self.warmup) is not int
            or self.warmup < 0
            or self.batch_size > self.replay_size
        ):
            raise ValueError("Warmup no negativo y replay suficiente para un lote")
        if (
            not 0 <= self.epsilon_end <= self.epsilon_start <= 1
            or not 0 <= self.discount <= 1
        ):
            raise ValueError("Epsilon y descuento deben estar entre cero y uno")
        if not all(
            isfinite(value) and value > 0
            for value in (self.learning_rate, self.gradient_clip)
        ):
            raise ValueError("Aprendizaje y norma de gradiente deben ser positivos")


@dataclass(frozen=True)
class EncodedState:
    values: Float32Array
    available: BoolArray
    budget: Float32Array
    front: BoolArray | None = None
    known: BoolArray | None = None
    valid: BoolArray | None = None


def tensor_states(states: list[EncodedState]) -> TensorDict:
    return TensorDict(
        {
            "values": torch.from_numpy(np.stack([state.values for state in states])),
            "available": torch.from_numpy(
                np.stack([state.available for state in states])
            ),
            "budget": torch.from_numpy(np.stack([state.budget for state in states])),
            "front": torch.from_numpy(
                np.stack(
                    [
                        state.front if state.front is not None else ~state.available
                        for state in states
                    ]
                )
            ),
            "known": torch.from_numpy(
                np.stack(
                    [
                        state.known if state.known is not None else ~state.available
                        for state in states
                    ]
                )
            ),
            "valid": torch.from_numpy(
                np.stack(
                    [
                        state.valid
                        if state.valid is not None
                        else np.ones_like(state.available)
                        for state in states
                    ]
                )
            ),
        },
        batch_size=[len(states)],
    )


class CandidateQNetwork(nn.Module):
    def __init__(
        self, dimensions: int, hidden_width: int, enhanced_state: bool = True
    ) -> None:
        super().__init__()
        self.enhanced_state = enhanced_state
        self.context_width = hidden_width * (3 if enhanced_state else 2) + 2
        self.encoder = nn.Sequential(
            nn.Linear(dimensions, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(self.context_width + hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1),
        )

    def representation(
        self, state: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor]:
        items = self.encoder(state["values"])
        available = state["available"]
        valid = state.get("valid", torch.ones_like(available))
        known_mask = state.get("known", ~available) & valid
        known = known_mask.unsqueeze(-1)
        remaining = (available & ~known_mask & valid).unsqueeze(-1)
        known_pool = (items * known).sum(dim=1) / known.sum(dim=1).clamp(min=1)
        remaining_pool = (items * remaining).sum(dim=1) / remaining.sum(dim=1).clamp(
            min=1
        )
        context = torch.cat((known_pool, remaining_pool, state["budget"]), dim=-1)
        if self.enhanced_state:
            front_mask = (state.get("front", known_mask) & valid).unsqueeze(-1)
            front_pool = (
                items.masked_fill(~front_mask, torch.finfo(items.dtype).min)
                .max(dim=1)
                .values
            )
            front_pool = torch.where(front_mask.any(dim=1), front_pool, 0.0)
            context = torch.cat(
                (known_pool, remaining_pool, front_pool, state["budget"]), dim=-1
            )
        combined = torch.cat(
            (items, context.unsqueeze(1).expand(-1, items.shape[1], -1)), dim=-1
        )
        return combined, context

    def forward(self, state: TensorDictBase) -> torch.Tensor:
        combined, _ = self.representation(state)
        scores = self.scorer(combined).squeeze(-1)
        return scores.masked_fill(~state["available"], torch.finfo(scores.dtype).min)
