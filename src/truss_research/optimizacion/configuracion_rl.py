"""Parámetros del aprendizaje Double DQN y del bandido contextual."""

from dataclasses import dataclass, field, fields
from math import isfinite

from truss_research.datos.procesados import JsonValue


@dataclass(frozen=True)
class DqnSettings:
    learning_rate: float = 0.001
    replay_size: int = 50_000
    target_interval: int = 500
    epsilon_steps: int = 30_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    discount: float = 1.0
    gradient_clip: float = 10.0
    train_frequency: int = 4
    gradient_steps: int = 1

    def __post_init__(self) -> None:
        for name in (
            "replay_size",
            "target_interval",
            "epsilon_steps",
            "train_frequency",
            "gradient_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} debe ser un entero positivo")
        validate_floats(self, ("learning_rate", "gradient_clip"))
        validate_floats(self, ("epsilon_start", "epsilon_end", "discount"), True)
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1 or self.discount != 1:
            raise ValueError("Exploración inválida o descuento distinto de 1 para AUC")


def validate_floats(
    settings: DqnSettings,
    names: tuple[str, ...],
    allow_zero: bool = False,
) -> None:
    for name in names:
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} debe ser numérico")
        if not isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"{name} debe ser finito y tener un valor permitido")


@dataclass(frozen=True)
class RLParameters:
    dqn: DqnSettings = field(default_factory=DqnSettings)


def read_rl_parameters(payload: JsonValue) -> RLParameters:
    if not isinstance(payload, dict) or set(payload) - {"dqn"}:
        raise ValueError("RL solo admite la configuración dqn del protocolo vigente")
    dqn = payload.get("dqn", {})
    if not isinstance(dqn, dict) or set(dqn) - {
        part.name for part in fields(DqnSettings)
    }:
        raise ValueError("Campos desconocidos en rl.dqn")
    return RLParameters(DqnSettings(**dqn))
