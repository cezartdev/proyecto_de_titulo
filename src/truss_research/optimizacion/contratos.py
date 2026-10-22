"""Configuración compartida del benchmark y de la variación evolutiva vigente."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from truss_research.surrogates.configuracion import ModelName

type BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class StudyTask:
    key: str
    kind: Literal["predictor"] = "predictor"
    model: ModelName = "dummy"
    seed: int = 42


@dataclass(frozen=True)
class AnalysisParameters:
    bootstrap_repeats: int = 2_000
    bootstrap_seed: int = 42
    predictive_block_generations: int = 2
    interpolation_models: tuple[ModelName, ...] = (
        "dummy",
        "ridge",
        "catboost",
        "xgboost",
        "tabpfn",
    )

    def __post_init__(self) -> None:
        from truss_research.surrogates.configuracion import MODEL_NAMES

        for value in (self.bootstrap_repeats, self.predictive_block_generations):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "Bloques de generaciones y remuestreos deben ser enteros positivos"
                )
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ValueError("La semilla de remuestreo debe ser un entero no negativo")
        if (
            not self.interpolation_models
            or len(set(self.interpolation_models)) != len(self.interpolation_models)
            or any(
                name not in MODEL_NAMES or name == "targetwise"
                for name in self.interpolation_models
            )
        ):
            raise ValueError(
                "La interpolación requiere familias base distintas y conocidas"
            )


@dataclass(frozen=True)
class EvolutionParameters:
    population_size: int = 8
    max_mating_attempts: int = 10
    crossover_probability: float = 0.9
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0

    def __post_init__(self) -> None:
        if type(self.population_size) is not int or self.population_size < 2:
            raise ValueError("La población requiere al menos dos individuos")
        if type(self.max_mating_attempts) is not int or self.max_mating_attempts < 1:
            raise ValueError("Los intentos de variación deben ser enteros positivos")
        if (
            not np.isfinite(
                [self.crossover_probability, self.crossover_eta, self.mutation_eta]
            ).all()
            or not 0 < self.crossover_probability <= 1
            or min(self.crossover_eta, self.mutation_eta) <= 0
        ):
            raise ValueError("Parámetros de variación evolutiva inválidos")
