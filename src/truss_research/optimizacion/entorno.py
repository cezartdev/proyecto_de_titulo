"""Evaluador privado que revela una respuesta registrada por consulta válida."""

from dataclasses import dataclass
from math import isfinite

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from torchrl.envs import ActionMask, GymWrapper, TransformedEnv

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.evaluacion.pareto import ObjectiveSpace, validate_responses
from truss_research.optimizacion.contratos import (
    EpisodeObservation,
    EpisodeSpec,
    QueryResult,
)
from truss_research.surrogates.configuracion import FloatArray


@dataclass(frozen=True)
class EpisodeData:
    spec: EpisodeSpec
    features: FloatArray
    predictions: FloatArray
    responses: FloatArray
    space: ObjectiveSpace
    reward_scale: float = 1.0
    reward_mode: str = "trapezoidal"
    allow_repeats: bool = False
    repeat_penalty: float = 0.0

    def __post_init__(self) -> None:
        if type(self.allow_repeats) is not bool or (
            isinstance(self.repeat_penalty, bool)
            or not isinstance(self.repeat_penalty, (int, float))
            or not isfinite(self.repeat_penalty)
            or self.repeat_penalty < 0
        ):
            raise ValueError("Configuración de repetición inválida")
        if self.repeat_penalty and not self.allow_repeats:
            raise ValueError("El castigo requiere permitir intentos repetidos")


def area_reward(previous: float, current: float, context: tuple[int, float]) -> float:
    """Contribución trapezoidal por consulta; escala cero denota referencia vacía."""
    budget, scale = context
    return (previous + current) / (2 * budget * scale) if scale > 0 else 0.0


def marginal_reward(
    previous: float, current: float, context: tuple[int, int, float]
) -> float:
    """Descomposición exacta del AUC en incrementos ponderados, con t comenzando en 1."""
    step, budget, scale = context
    return (
        (budget - step + 0.5) * (current - previous) / (budget * scale)
        if scale > 0
        else 0.0
    )


type PublicObservation = dict[
    str, npt.NDArray[np.float64] | npt.NDArray[np.bool_] | npt.NDArray[np.int64]
]


class CatalogActions(spaces.Discrete):
    """El muestreo habitual respeta los diseños disponibles, incluidos verificadores."""

    mask: npt.NDArray[np.bool_]

    def sample(
        self,
        mask: npt.NDArray[np.int8] | None = None,
        probability: npt.NDArray[np.float64] | None = None,
    ) -> np.int64:
        if probability is not None and np.any(probability[~self.mask] != 0):
            raise ValueError("La probabilidad asigna masa a diseños ya consultados")
        if probability is not None:
            return super().sample(probability=probability)
        return super().sample(
            mask=self.mask.astype(np.int8)
            if mask is None
            else mask * self.mask.astype(np.int8)
        )


class QueryEnvironment(gym.Env[PublicObservation, int]):
    metadata = {"render_modes": []}

    def __init__(self, data: EpisodeData) -> None:
        super().__init__()
        size = len(data.spec.candidate_ids)
        if not isfinite(data.reward_scale) or data.reward_scale < 0:
            raise ValueError(
                "La escala privada de recompensa debe ser finita y no negativa"
            )
        if data.reward_mode not in ("trapezoidal", "weighted_marginal"):
            raise ValueError("Modo de recompensa desconocido")
        responses = validate_responses(data.responses)
        predictions = validate_responses(data.predictions)
        if (
            len(responses) != size
            or len(predictions) != size
            or len(data.features) != size
        ):
            raise ValueError("Datos del episodio desalineados")
        if (
            data.features.ndim != 2
            or data.features.shape[1] != len(INPUT_COLUMNS)
            or not np.isfinite(data.features).all()
        ):
            raise ValueError("Características inválidas")
        self._data = EpisodeData(
            data.spec,
            data.features.copy(),
            predictions.copy(),
            responses.copy(),
            data.space,
            data.reward_scale,
            data.reward_mode,
            data.allow_repeats,
            data.repeat_penalty,
        )
        _, self._identity = np.unique(data.features, axis=0, return_inverse=True)
        self._validate_identities()
        self.action_space = CatalogActions(size)
        self.observation_space = spaces.Dict(
            {
                "features": spaces.Box(
                    -np.inf, np.inf, shape=data.features.shape, dtype=np.float64
                ),
                "predictions": spaces.Box(
                    -np.inf, np.inf, shape=predictions.shape, dtype=np.float64
                ),
                "observed": spaces.Box(
                    -np.inf, np.inf, shape=responses.shape, dtype=np.float64
                ),
                "action_mask": spaces.Box(0, 1, shape=(size,), dtype=np.bool_),
                "known_mask": spaces.Box(0, 1, shape=(size,), dtype=np.bool_),
                "counters": spaces.Box(0, size, shape=(3,), dtype=np.int64),
                "budget": spaces.Box(0, size, shape=(2,), dtype=np.int64),
            }
        )
        self.reset(seed=data.spec.seed)

    def reset(
        self, *, seed: int | None = None, options: dict[str, int] | None = None
    ) -> tuple[PublicObservation, dict[str, int]]:
        super().reset(seed=seed)
        if options:
            raise ValueError(
                "El episodio se fija al construir el entorno; no admite opciones"
            )
        size = len(self._data.spec.candidate_ids)
        self._available = np.ones(size, dtype=bool)
        self.action_space.mask = self._available
        self._observed = np.zeros_like(self._data.responses)
        self._queries = 0
        self._attempts = 0
        self._hypervolume = 0.0
        self.initial_reward = self.query(self._data.spec.initial_index).reward
        return self.public_observation(), {"queries": self._queries}

    def public_observation(self) -> PublicObservation:
        observation = self.observe()
        return {
            "features": observation.features,
            "predictions": observation.predictions,
            "observed": observation.observed,
            "action_mask": observation.selectable,
            "known_mask": ~observation.available,
            "counters": np.array(
                [
                    observation.attempt_count,
                    observation.queries,
                    observation.repeat_attempts,
                ],
                dtype=np.int64,
            ),
            "budget": np.array(
                [observation.budget, observation.attempt_count], dtype=np.int64
            ),
        }

    def step(
        self, action: int
    ) -> tuple[PublicObservation, float, bool, bool, dict[str, int]]:
        if not isinstance(action, (int, np.integer)) or isinstance(
            action, (bool, np.bool_)
        ):
            raise ValueError("La acción debe ser un índice entero")
        result = self.query(int(action))
        return (
            self.public_observation(),
            result.reward,
            result.terminated,
            False,
            {"queries": self._queries},
        )

    def observe(self) -> EpisodeObservation:
        return EpisodeObservation(
            self._data.features.copy(),
            self._data.predictions.copy(),
            self._observed.copy(),
            self._available.copy(),
            self._data.spec.budget,
            self._queries,
            self._attempts,
            np.ones_like(self._available) if self._data.allow_repeats else None,
        )

    def _validate_identities(self) -> None:
        for identity in np.unique(self._identity):
            rows = self._data.responses[self._identity == identity]
            if not np.all(rows == rows[0]):
                raise ValueError("Un mismo diseño tiene respuestas distintas")
        if not self._data.allow_repeats and self._data.spec.budget > len(
            np.unique(self._identity)
        ):
            raise ValueError("El presupuesto supera la cantidad de diseños únicos")

    def query(self, action: int) -> QueryResult:
        if self.observe().terminal:
            raise ValueError("El episodio terminó; no se permiten más consultas")
        if type(action) is not int or not 0 <= action < len(self._available):
            raise ValueError("Acción fuera del catálogo")
        repeated = not self._available[action]
        if repeated and not self._data.allow_repeats:
            raise ValueError("El diseño ya fue consultado")
        self._reveal(action)
        self._queries += int(not repeated)
        self._attempts += 1
        volume = self._data.space.hypervolume(self._observed[~self._available])
        reward = area_reward(
            self._hypervolume, volume, (self._data.spec.budget, self._data.reward_scale)
        )
        if self._data.reward_mode == "weighted_marginal":
            reward = marginal_reward(
                self._hypervolume,
                volume,
                (self._attempts, self._data.spec.budget, self._data.reward_scale),
            )
        reward -= self._data.repeat_penalty * int(repeated) / self._data.spec.budget
        self._hypervolume = volume
        observation = self.observe()
        self.action_space.mask = observation.selectable
        return QueryResult(
            observation, tuple(self._observed[action]), reward, observation.terminal
        )

    def _reveal(self, action: int) -> None:
        if not self._available[action]:
            return
        aliases = self._identity == self._identity[action]
        self._observed[aliases] = self._data.responses[action]
        self._available[aliases] = False


def create_episode(data: EpisodeData) -> QueryEnvironment:
    return QueryEnvironment(data)


def torchrl_environment(environment: QueryEnvironment) -> TransformedEnv:
    """Expone el mismo entorno a TorchRL con muestreo de acciones válidas."""
    return TransformedEnv(
        GymWrapper(
            environment, categorical_action_encoding=True, allow_done_after_reset=True
        ),
        ActionMask(action_key="action", mask_key="action_mask"),
    )
