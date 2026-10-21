"""Política Double DQN con puntuación compartida e invariancia al orden."""

from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, cast

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from tensordict import TensorDict, TensorDictBase
from torch import nn

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.datos.preparacion import source_digest
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.optimizacion.contratos import (
    BUDGETS,
    CATALOG_SIZE,
    BoolArray,
    EpisodeObservation,
)
from truss_research.optimizacion.estrategias import available_actions
from truss_research.optimizacion.variantes_rl import (
    PpoSettings,
    RLAlgorithm,
    policy_variant,
)
from truss_research.surrogates.configuracion import FloatArray
from truss_research.surrogates.experimentos import read_run

type Float32Array = npt.NDArray[np.float32]


if TYPE_CHECKING:
    from truss_research.optimizacion.contratos import StudyTask
    from truss_research.optimizacion.estudio_estado import StudyConfig

POLICY_SCHEMA_VERSION = 9


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
class PPOParameters(PpoSettings):
    transitions: int = 50_000
    hidden_width: int = 64
    batch_size: int = 64
    validation_interval: int = 5_000

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(
            type(value) is not int or value < 1
            for value in (
                self.transitions,
                self.hidden_width,
                self.batch_size,
                self.validation_interval,
            )
        ):
            raise ValueError("Tamaños y períodos PPO deben ser enteros positivos")


@dataclass(frozen=True)
class PolicyConfig:
    data_dir: Path
    predictor_dir: Path
    output_dir: Path
    seed: int = 42
    use_predictions: bool = True
    catalog_size: int = CATALOG_SIZE
    budgets: tuple[int, ...] = BUDGETS
    validation_episodes: int = 10
    parameters: DQNParameters | PPOParameters = field(default_factory=DQNParameters)
    resume_dir: Path | None = None
    validation_seeds: tuple[int, ...] | None = None
    algorithm: RLAlgorithm = "double_dqn"
    reward_mode: str = "weighted_marginal"
    enhanced_state: bool = True
    allow_repeats: bool = False
    repeat_penalty: float = 0.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise ValueError("El dispositivo RL debe ser cpu o cuda")
        if type(self.allow_repeats) is not bool or (
            isinstance(self.repeat_penalty, bool)
            or not isinstance(self.repeat_penalty, (int, float))
            or not isfinite(self.repeat_penalty)
            or self.repeat_penalty < 0
            or (self.repeat_penalty > 0 and not self.allow_repeats)
        ):
            raise ValueError("Configuración de repetición inválida")
        if (
            self.reward_mode not in ("trapezoidal", "weighted_marginal")
            or type(self.enhanced_state) is not bool
        ):
            raise ValueError("Protocolo de recompensa o representación inválido")
        if self.algorithm not in NETWORK_TYPES:
            raise ValueError("Algoritmo de política desconocido")
        if (self.algorithm == "ppo") != isinstance(self.parameters, PPOParameters):
            raise ValueError("Parámetros incompatibles con el algoritmo de política")
        if type(self.use_predictions) is not bool:
            raise ValueError("use_predictions debe ser booleano")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("La semilla debe ser un entero no negativo")
        if self.validation_seeds is not None and (
            not self.validation_seeds
            or len(set(self.validation_seeds)) != len(self.validation_seeds)
            or any(type(seed) is not int or seed < 0 for seed in self.validation_seeds)
        ):
            raise ValueError("Las semillas comunes de validación son inválidas")
        if self.catalog_size < 2 or self.validation_episodes < 1 or not self.budgets:
            raise ValueError("Catálogo, presupuestos y episodios inválidos")
        if any(
            type(budget) is not int or not 2 <= budget <= self.catalog_size
            for budget in self.budgets
        ):
            raise ValueError(
                "RL requiere presupuestos entre dos y el tamaño del catálogo"
            )


@dataclass(frozen=True)
class EncodedState:
    values: Float32Array
    available: BoolArray
    budget: Float32Array
    front: BoolArray | None = None
    known: BoolArray | None = None
    valid: BoolArray | None = None


@dataclass(frozen=True)
class PolicyEncoder:
    transformer: ColumnTransformer
    space: ObjectiveSpace
    use_predictions: bool
    enhanced_state: bool = True
    allow_repeats: bool = False
    repeat_penalty: float = 0.0

    def encode(self, observation: EpisodeObservation) -> EncodedState:
        features = pd.DataFrame(observation.features, columns=INPUT_COLUMNS)
        encoded = self.transformer.transform(features)
        predictions = (
            self.space.transform(observation.predictions)
            if self.use_predictions
            else np.zeros_like(observation.predictions)
        )
        observed = np.zeros_like(observation.observed)
        known = ~observation.available
        observed[known] = self.space.transform(observation.observed[known])
        values = np.column_stack((encoded, predictions, observed, known.astype(float)))
        front = np.zeros_like(known)
        if self.enhanced_state:
            observed_front = self.space.front(observation.observed[known])
            front[known] = [
                any(np.array_equal(row, item) for item in observed_front)
                for row in observation.observed[known]
            ]
        gains = np.zeros(len(known))
        if self.use_predictions and self.enhanced_state:
            current = self.space.hypervolume(observation.observed[known])
            denominator = max(
                float(np.prod(np.maximum(self.space.reference, 0))), np.finfo(float).eps
            )
            for index in np.flatnonzero(observation.available):
                gains[index] = (
                    max(
                        0.0,
                        self.space.hypervolume(
                            np.vstack(
                                (
                                    observation.observed[known],
                                    observation.predictions[index],
                                )
                            )
                        )
                        - current,
                    )
                    / denominator
                )
        if self.enhanced_state:
            values = np.column_stack((values, front.astype(float), gains))
        budget = np.array(
            [
                (observation.budget - observation.attempt_count) / observation.budget,
                observation.budget / len(observation.available),
            ],
            dtype=np.float32,
        )
        return EncodedState(
            values.astype(np.float32), observation.selectable, budget, front, known
        )


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


class DuelingCandidateQNetwork(CandidateQNetwork):
    def __init__(
        self, dimensions: int, hidden_width: int, enhanced_state: bool = True
    ) -> None:
        super().__init__(dimensions, hidden_width, enhanced_state)
        self.value_head = nn.Sequential(
            nn.Linear(self.context_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, state: TensorDictBase) -> torch.Tensor:
        combined, context = self.representation(state)
        advantages = self.scorer(combined).squeeze(-1)
        available = state["available"]
        average = (advantages * available).sum(-1, keepdim=True) / available.sum(
            -1, keepdim=True
        ).clamp(min=1)
        scores = self.value_head(context) + advantages - average
        return scores.masked_fill(~available, torch.finfo(scores.dtype).min)


class CandidatePpoNetwork(CandidateQNetwork):
    def __init__(
        self, dimensions: int, hidden_width: int, enhanced_state: bool = True
    ) -> None:
        super().__init__(dimensions, hidden_width, enhanced_state)
        self.value_head = nn.Sequential(
            nn.Linear(self.context_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1),
        )

    def state_value(self, state: TensorDictBase) -> torch.Tensor:
        _, context = self.representation(state)
        return self.value_head(context)


NETWORK_TYPES: dict[RLAlgorithm, type[CandidateQNetwork]] = {
    "double_dqn": CandidateQNetwork,
    "dueling_double_dqn": DuelingCandidateQNetwork,
    "ppo": CandidatePpoNetwork,
}


def configured_policy(
    config: "StudyConfig",
    locations: tuple[Path, Path, Path | None],
    task: "StudyTask",
) -> PolicyConfig:
    predictor, destination, resume = locations
    variant = policy_variant(task.variant)
    common = dict(
        transitions=config.rl_steps,
        batch_size=config.rl_batch,
        hidden_width=config.rl_hidden,
        validation_interval=config.validation_interval,
    )
    parameters = (
        PPOParameters(**common, **asdict(config.rl.ppo))
        if variant.algorithm == "ppo"
        else DQNParameters(**common, warmup=config.rl_warmup, **asdict(config.rl.dqn))
    )
    if variant.train_frequency is not None:
        from dataclasses import replace

        parameters = replace(parameters, train_frequency=variant.train_frequency)
    return PolicyConfig(
        config.data_dir,
        predictor,
        destination,
        seed=task.seed,
        use_predictions=variant.use_predictions,
        algorithm=variant.algorithm,
        catalog_size=config.catalog_size,
        budgets=config.budgets,
        validation_episodes=config.episodes_per_seed,
        validation_seeds=config.seeds,
        resume_dir=resume,
        parameters=parameters,
        reward_mode=variant.reward_mode,
        enhanced_state=variant.enhanced_state,
        allow_repeats=variant.allow_repeats,
        repeat_penalty=config.rl.repeat_penalty if variant.repeat_penalty else 0.0,
        device=config.execution.rl_device,
    )


@dataclass(frozen=True)
class PolicyComponents:
    network: CandidateQNetwork
    encoder: PolicyEncoder


class PolicyStrategy:
    def __init__(self, components: PolicyComponents) -> None:
        self.components = components
        self.components.network.eval()

    def values(self, observation: EpisodeObservation) -> FloatArray:
        state = tensor_states([self.components.encoder.encode(observation)])
        state = state.to(next(self.components.network.parameters()).device)
        with torch.no_grad():
            return self.components.network(state)[0].cpu().numpy().astype(float)

    def choose(self, observation: EpisodeObservation) -> int:
        available_actions(observation)
        return int(np.argmax(self.values(observation)))


def load_policy(directory: Path, predictor_dir: Path) -> PolicyStrategy:
    metadata = read_run(directory)
    if metadata.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("La política no corresponde al formato vigente")
    if metadata.get("phase") != "complete":
        raise ValueError("La política no ha completado su presupuesto")
    predictor_metadata = read_run(predictor_dir)
    if metadata.get("dataset_signature") != predictor_metadata.get("dataset_signature"):
        raise ValueError("La política y el predictor pertenecen a datasets diferentes")
    if metadata.get("predictor_signature") != source_digest(
        predictor_dir / "model.joblib"
    ):
        raise ValueError("La política pertenece a otro predictor")
    if metadata.get("policy_signature") != source_digest(
        directory / "policy.pt"
    ) or metadata.get("encoder_signature") != source_digest(
        directory / "encoder.joblib"
    ):
        raise ValueError("Las firmas de la política no coinciden")
    encoder = joblib.load(directory / "encoder.joblib")
    if not isinstance(encoder, PolicyEncoder):
        raise ValueError("El artefacto no contiene un codificador de política")
    settings = metadata.get("config")
    if not isinstance(settings, dict) or settings.get("algorithm") not in NETWORK_TYPES:
        raise ValueError("Falta el algoritmo de la política")
    algorithm = cast(RLAlgorithm, settings["algorithm"])
    if metadata.get("algorithm") != algorithm:
        raise ValueError("La identidad del algoritmo no coincide")
    if settings.get("use_predictions") != encoder.use_predictions:
        raise ValueError("El codificador no corresponde a la condición de información")
    if settings.get("enhanced_state") != encoder.enhanced_state:
        raise ValueError("El codificador no corresponde a la representación declarada")
    if (settings.get("allow_repeats"), settings.get("repeat_penalty")) != (
        encoder.allow_repeats,
        encoder.repeat_penalty,
    ):
        raise ValueError("El codificador no corresponde al tratamiento de repeticiones")
    network = NETWORK_TYPES[algorithm](
        int(str(metadata["dimensions"])),
        int(str(metadata["hidden_width"])),
        bool(settings["enhanced_state"]),
    )
    state = cast(
        dict[str, torch.Tensor],
        torch.load(directory / "policy.pt", map_location="cpu", weights_only=True),
    )
    network.load_state_dict(state)
    return PolicyStrategy(PolicyComponents(network, encoder))
