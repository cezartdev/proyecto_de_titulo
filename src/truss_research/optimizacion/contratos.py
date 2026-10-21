"""Contratos públicos de selección, separados de las respuestas ocultas."""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Final, Literal, Protocol

import numpy as np
import numpy.typing as npt

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.evaluacion.protocolo_analisis import (
    GeneralizationParameters,
    LearningCurveParameters,
)
from truss_research.optimizacion.variantes_rl import PolicyVariant
from truss_research.surrogates.configuracion import (
    EXPERIMENT_SEEDS,
    FloatArray,
    ModelName,
)

type BoolArray = npt.NDArray[np.bool_]

type EvaluationSplit = Literal["validation", "test"]

type StrategyName = Literal[
    "random",
    "ranking",
    "greedy",
    "ensemble_mean",
    "ensemble_ucb",
    "online_ensemble_mean",
    "online_ensemble_ucb",
    "greedy_safe",
    "nsga2_random_init",
    "nsga2_surrogate_random_init",
    "rl_double_dqn_trapezoidal_basic_state",
    "rl_reward_only",
    "rl_state_only",
    "rl_frequency_1",
    "rl_repeat_control",
    "rl_repeat_penalty",
    "nsga2",
    "nsga2_surrogate",
    "rl",
    "rl_ablation",
    "rl_dueling",
    "rl_dueling_ablation",
    "rl_ppo",
    "rl_ppo_ablation",
]

BUDGETS: Final[tuple[int, ...]] = (8, 16, 24)

STRATEGIES: Final[tuple[StrategyName, ...]] = (
    "random",
    "ranking",
    "greedy",
    "nsga2",
    "nsga2_surrogate",
    "rl",
    "rl_ablation",
    "rl_dueling",
    "rl_dueling_ablation",
    "rl_ppo",
    "rl_ppo_ablation",
)

EPISODES_PER_SEED: Final[int] = 10

CATALOG_SIZE: Final[int] = 32

REWARD_PROTOCOL: Final[str] = "normalized_auc_attempts_v3"


@dataclass(frozen=True)
class StudyTask:
    key: str
    kind: Literal[
        "predictor",
        "policy",
        "explain",
        "ranking",
        "decision",
        "test_predictor",
    ]
    model: ModelName = "dummy"
    seed: int = 42
    split: Literal["validation", "test"] = "validation"
    variant: PolicyVariant = "rl"


@dataclass(frozen=True)
class AnalysisParameters:
    catalog_sizes: tuple[int, ...] = (32, 64)
    reference_margins: tuple[float, ...] = (0.05, 0.1, 0.2)
    bootstrap_repeats: int = 2_000
    bootstrap_seed: int = 42
    predictive_block_generations: int = 2
    generalization_seeds: tuple[int, ...] = ()
    generalization: GeneralizationParameters = field(
        default_factory=GeneralizationParameters
    )
    learning_curves: LearningCurveParameters = field(
        default_factory=LearningCurveParameters
    )

    def __post_init__(self) -> None:
        if (
            type(self.predictive_block_generations) is not int
            or self.predictive_block_generations < 1
        ):
            raise ValueError("El bloque predictivo requiere generaciones positivas")
        if (
            len(self.generalization_seeds) > 1
            or len(set(self.generalization_seeds)) != len(self.generalization_seeds)
            or any(type(s) is not int or s < 0 for s in self.generalization_seeds)
        ):
            raise ValueError("Semillas de partición agrupada inválidas")
        if not self.catalog_sizes or any(
            type(value) is not int or value < 2 for value in self.catalog_sizes
        ):
            raise ValueError("Los catálogos de sensibilidad requieren enteros >= 2")
        if tuple(sorted(set(self.catalog_sizes))) != self.catalog_sizes:
            raise ValueError("Los tamaños de catálogo deben ser únicos y crecientes")
        if not self.reference_margins or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value <= 0
            for value in self.reference_margins
        ):
            raise ValueError("Los márgenes deben ser finitos y positivos")
        if len(set(self.reference_margins)) != len(self.reference_margins):
            raise ValueError("Los márgenes de referencia no deben repetirse")
        if type(self.bootstrap_repeats) is not int or self.bootstrap_repeats < 1:
            raise ValueError(
                "El remuestreo requiere un número positivo de repeticiones"
            )
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ValueError("La semilla de remuestreo debe ser un entero no negativo")


@dataclass(frozen=True)
class EvolutionParameters:
    enabled: bool = False
    population_size: int = 4
    proposal_pool: int = 8
    max_mating_attempts: int = 10
    crossover_probability: float = 0.9
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0
    guided_initialization: bool = True
    initialization_ablation: bool = False

    def __post_init__(self) -> None:
        if any(
            type(v) is not bool
            for v in (
                self.enabled,
                self.guided_initialization,
                self.initialization_ablation,
            )
        ):
            raise ValueError("La activación evolutiva debe ser booleana")
        if (
            type(self.population_size) is not int
            or self.population_size < 2
            or type(self.proposal_pool) is not int
            or self.proposal_pool < 1
            or type(self.max_mating_attempts) is not int
            or self.max_mating_attempts < 1
        ):
            raise ValueError("Población mínima de dos y lote de propuestas positivo")
        if (
            not np.isfinite(
                [self.crossover_probability, self.crossover_eta, self.mutation_eta]
            ).all()
            or not 0 < self.crossover_probability <= 1
            or min(self.crossover_eta, self.mutation_eta) <= 0
        ):
            raise ValueError("Parámetros de variación evolutiva inválidos")


@dataclass(frozen=True)
class EpisodeSpec:
    candidate_ids: tuple[str, ...]
    initial_index: int
    budget: int
    seed: int
    episode_id: str

    def __post_init__(self) -> None:
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("El catálogo contiene identificadores repetidos")
        if not 1 <= self.budget <= len(self.candidate_ids):
            raise ValueError(
                "El presupuesto debe estar entre uno y el tamaño del catálogo"
            )
        if not 0 <= self.initial_index < len(self.candidate_ids):
            raise ValueError("Índice inicial fuera del catálogo")


@dataclass(frozen=True)
class EpisodeObservation:
    features: FloatArray
    predictions: FloatArray
    observed: FloatArray
    available: BoolArray
    budget: int
    queries: int
    attempts: int | None = None
    action_mask: BoolArray | None = None

    def __post_init__(self) -> None:
        size = len(self.available)
        if (
            self.features.shape != (size, len(INPUT_COLUMNS))
            or self.predictions.shape != (size, len(OUTPUT_COLUMNS))
            or self.observed.shape != (size, len(OUTPUT_COLUMNS))
        ):
            raise ValueError(
                "La observación debe contener siete entradas y tres respuestas alineadas"
            )
        if (
            self.available.dtype != np.bool_
            or not 0 <= self.queries <= self.budget
            or self.budget < 1
        ):
            raise ValueError("Máscara o presupuesto de la observación inválidos")
        if len(
            np.unique(self.features[~self.available], axis=0)
        ) != self.queries or np.any(self.observed[self.available] != 0):
            raise ValueError(
                "La observación contiene respuestas ocultas o conteos inconsistentes"
            )
        if not self.queries <= self.attempt_count <= self.budget:
            raise ValueError(
                "Intentos incompatibles con consultas únicas y presupuesto"
            )
        if self.action_mask is not None and (
            self.action_mask.shape != (size,) or self.action_mask.dtype != np.bool_
        ):
            raise ValueError("Máscara de acciones seleccionables inválida")
        if not all(
            np.isfinite(values).all()
            for values in (self.features, self.predictions, self.observed)
        ):
            raise ValueError("La observación debe ser finita")

    @property
    def terminal(self) -> bool:
        return self.attempt_count >= self.budget or not self.selectable.any()

    @property
    def attempt_count(self) -> int:
        return self.queries if self.attempts is None else self.attempts

    @property
    def repeat_attempts(self) -> int:
        return self.attempt_count - self.queries

    @property
    def selectable(self) -> BoolArray:
        mask = self.available if self.action_mask is None else self.action_mask
        return mask.copy() if self.attempt_count < self.budget else np.zeros_like(mask)


@dataclass(frozen=True)
class QueryResult:
    observation: EpisodeObservation
    response: tuple[float, ...]
    reward: float
    terminated: bool


class SelectionStrategy(Protocol):
    def choose(self, observation: EpisodeObservation) -> int: ...


@dataclass(frozen=True)
class SelectionConfig:
    data_dir: Path
    predictor_dir: Path
    output_dir: Path
    split: EvaluationSplit = "validation"
    catalog_size: int = CATALOG_SIZE
    budgets: tuple[int, ...] = BUDGETS
    seeds: tuple[int, ...] = EXPERIMENT_SEEDS
    episodes_per_seed: int = EPISODES_PER_SEED
    policy_dir: Path | None = None
    evolution: EvolutionParameters = EvolutionParameters()
    base_catalog_size: int | None = None

    def __post_init__(self) -> None:
        if self.base_catalog_size is not None and (
            type(self.base_catalog_size) is not int
            or not 2 <= self.base_catalog_size <= self.catalog_size
        ):
            raise ValueError(
                "El catálogo base debe estar entre dos y el catálogo ampliado"
            )
        if self.split not in ("validation", "test"):
            raise ValueError("El conjunto debe ser validación o prueba")
        integers = (self.catalog_size, self.episodes_per_seed, *self.budgets)
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError("Tamaños y presupuestos deben ser enteros positivos")
        if not self.budgets or not self.seeds or max(self.budgets) > self.catalog_size:
            raise ValueError("Semillas y presupuestos no vacíos, limitados al catálogo")
        if len(set(self.budgets)) != len(self.budgets) or len(set(self.seeds)) != len(
            self.seeds
        ):
            raise ValueError("Semillas y presupuestos no deben repetirse")
        if any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("Las semillas deben ser enteros no negativos")
