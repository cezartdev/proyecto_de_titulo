"""Identidades y parámetros fijos de las familias RL y sus ablaciones."""

from dataclasses import dataclass, field, fields
from math import isfinite
from typing import Literal, cast

from truss_research.datos.procesados import JsonValue

type RLAlgorithm = Literal["double_dqn", "dueling_double_dqn", "ppo"]
type PolicyVariant = Literal[
    "rl",
    "rl_ablation",
    "rl_dueling",
    "rl_dueling_ablation",
    "rl_ppo",
    "rl_ppo_ablation",
    "rl_double_dqn_trapezoidal_basic_state",
    "rl_reward_only",
    "rl_state_only",
    "rl_frequency_1",
    "rl_repeat_control",
    "rl_repeat_penalty",
]

RL_ALGORITHMS: tuple[RLAlgorithm, ...] = ("double_dqn", "dueling_double_dqn", "ppo")


@dataclass(frozen=True)
class PolicyVariantSpec:
    name: PolicyVariant
    algorithm: RLAlgorithm
    use_predictions: bool
    reward_mode: str = "weighted_marginal"
    enhanced_state: bool = True
    train_frequency: int | None = None
    allow_repeats: bool = False
    repeat_penalty: float = 0.0


POLICY_VARIANTS: tuple[PolicyVariantSpec, ...] = (
    PolicyVariantSpec("rl", "double_dqn", True),
    PolicyVariantSpec("rl_ablation", "double_dqn", False),
    PolicyVariantSpec("rl_dueling", "dueling_double_dqn", True),
    PolicyVariantSpec("rl_dueling_ablation", "dueling_double_dqn", False),
    PolicyVariantSpec("rl_ppo", "ppo", True),
    PolicyVariantSpec("rl_ppo_ablation", "ppo", False),
)
VARIANT_REGISTRY: dict[PolicyVariant, PolicyVariantSpec] = {
    variant.name: variant for variant in POLICY_VARIANTS
}
ABLATION_VARIANTS: tuple[PolicyVariantSpec, ...] = (
    PolicyVariantSpec(
        "rl_double_dqn_trapezoidal_basic_state",
        "double_dqn",
        True,
        "trapezoidal",
        False,
    ),
    PolicyVariantSpec("rl_reward_only", "double_dqn", True, "weighted_marginal", False),
    PolicyVariantSpec("rl_state_only", "double_dqn", True, "trapezoidal", True),
    PolicyVariantSpec("rl_frequency_1", "double_dqn", True, train_frequency=1),
)
REPETITION_VARIANTS: tuple[PolicyVariantSpec, ...] = (
    PolicyVariantSpec("rl_repeat_control", "double_dqn", True, allow_repeats=True),
    PolicyVariantSpec(
        "rl_repeat_penalty", "double_dqn", True, allow_repeats=True, repeat_penalty=1.0
    ),
)
VARIANT_REGISTRY.update({v.name: v for v in (*ABLATION_VARIANTS, *REPETITION_VARIANTS)})


def policy_variant(name: str) -> PolicyVariantSpec:
    if name not in VARIANT_REGISTRY:
        raise ValueError(f"Variante de política desconocida: {name}")
    return VARIANT_REGISTRY[name]


def policy_variants(algorithms: tuple[RLAlgorithm, ...]) -> tuple[PolicyVariant, ...]:
    return tuple(part.name for part in POLICY_VARIANTS if part.algorithm in algorithms)


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


@dataclass(frozen=True)
class PpoSettings:
    learning_rate: float = 0.0003
    rollout_steps: int = 1024
    epochs: int = 10
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    discount: float = 1.0
    entropy_coeff: float = 0.01
    critic_coeff: float = 0.5
    gradient_clip: float = 0.5
    advantage_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("rollout_steps", "epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} debe ser un entero positivo")
        validate_floats(
            self,
            ("learning_rate", "critic_coeff", "gradient_clip", "advantage_epsilon"),
        )
        validate_floats(
            self, ("clip_epsilon", "gae_lambda", "discount", "entropy_coeff"), True
        )
        if (
            not 0 < self.clip_epsilon < 1
            or not 0 <= self.gae_lambda <= 1
            or self.discount != 1
            or not isfinite(self.entropy_coeff)
            or self.entropy_coeff < 0
        ):
            raise ValueError("Parámetros de PPO inválidos; AUC requiere descuento 1")


def validate_floats(
    settings: DqnSettings | PpoSettings,
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
    algorithms: tuple[RLAlgorithm, ...] = RL_ALGORITHMS
    dqn: DqnSettings = field(default_factory=DqnSettings)
    ppo: PpoSettings = field(default_factory=PpoSettings)
    ablations: bool = False
    repetition_ablations: bool = False
    repeat_penalty: float = 1.0

    def __post_init__(self) -> None:
        if type(self.ablations) is not bool:
            raise ValueError("La activación de ablaciones debe ser booleana")
        if type(self.repetition_ablations) is not bool or (
            isinstance(self.repeat_penalty, bool)
            or not isinstance(self.repeat_penalty, (int, float))
            or not isfinite(self.repeat_penalty)
            or self.repeat_penalty <= 0
        ):
            raise ValueError("Ablaciones de repetición o castigo inválidos")
        if (
            not self.algorithms
            or len(set(self.algorithms)) != len(self.algorithms)
            or any(name not in RL_ALGORITHMS for name in self.algorithms)
        ):
            raise ValueError("Algoritmos RL vacíos, repetidos o desconocidos")

    @property
    def variants(self) -> tuple[PolicyVariant, ...]:
        return (
            *policy_variants(self.algorithms),
            *(
                v.name
                for v in ABLATION_VARIANTS
                if self.ablations and "double_dqn" in self.algorithms
            ),
            *(
                v.name
                for v in REPETITION_VARIANTS
                if self.repetition_ablations and "double_dqn" in self.algorithms
            ),
        )


def read_rl_parameters(payload: JsonValue) -> RLParameters:
    if not isinstance(payload, dict) or set(payload) - {
        "algorithms",
        "dqn",
        "ppo",
        "ablations",
        "repetition_ablations",
        "repeat_penalty",
    }:
        raise ValueError("Campos desconocidos en la configuración RL")
    algorithms = payload.get("algorithms", list(RL_ALGORITHMS))
    if not isinstance(algorithms, list) or any(
        not isinstance(name, str) for name in algorithms
    ):
        raise ValueError("rl.algorithms debe ser una lista JSON")
    dqn, ppo = payload.get("dqn", {}), payload.get("ppo", {})
    if not isinstance(dqn, dict) or set(dqn) - {
        part.name for part in fields(DqnSettings)
    }:
        raise ValueError("Campos desconocidos en rl.dqn")
    if not isinstance(ppo, dict) or set(ppo) - {
        part.name for part in fields(PpoSettings)
    }:
        raise ValueError("Campos desconocidos en rl.ppo")
    return RLParameters(
        cast(tuple[RLAlgorithm, ...], tuple(algorithms)),
        DqnSettings(**dqn),
        PpoSettings(**ppo),
        cast(bool, payload.get("ablations", False)),
        cast(bool, payload.get("repetition_ablations", False)),
        cast(float, payload.get("repeat_penalty", 1.0)),
    )
