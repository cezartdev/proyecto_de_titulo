"""Contratos de experimentos predictivos con variables físicas fijas."""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.schemas import DESIGN_DESTINATION
from truss_research.surrogates.poda import PruningParameters
from truss_research.surrogates.tabpfn import TabPFNParameters

type FloatArray = npt.NDArray[np.float64]

type ModelName = Literal[
    "dummy",
    "ridge",
    "random_forest",
    "svr",
    "gpr",
    "catboost",
    "xgboost",
    "lightgbm",
    "mlp",
    "targetwise",
    "tabpfn",
]

MODEL_NAMES: Final[tuple[ModelName, ...]] = (
    "dummy",
    "ridge",
    "random_forest",
    "svr",
    "gpr",
    "catboost",
    "xgboost",
    "lightgbm",
    "mlp",
    "targetwise",
    "tabpfn",
)

BASE_MODEL_NAMES: Final[tuple[ModelName, ...]] = tuple(
    n for n in MODEL_NAMES if n != "targetwise"
)

SCHEMA_VERSION: Final[int] = 9

DEFAULT_FOLDS: Final[int] = 5

TIMING_REPEATS: Final[int] = 10

EXPERIMENT_SEEDS: Final[tuple[int, ...]] = (42, 43, 44, 45, 46)


@dataclass(frozen=True)
class ModelParameters:
    trees: int = 100
    early_stopping_rounds: int = 0
    depth: int = 6
    min_samples_leaf: int = 1
    regularization: float = 1.0
    mlp_weight_decay: float = 0.0001
    learning_rate: float = 0.001
    boosting_rate: float = 0.05
    svr_c: float = 10.0
    svr_epsilon: float = 0.01
    gpr_noise: float = 1e-6
    gpr_max_iter: int = 2000
    gpr_max_fun: int = 15000
    gpr_max_line_search: int = 50
    gpr_recovery_line_search: int = 100
    gpr_ftol: float = 2.22e-9
    gpr_gtol: float = 1e-5
    epochs: int = 300
    patience: int = 20
    hidden_width: int = 64
    batch_size: int = 64

    def __post_init__(self) -> None:
        if (
            type(self.early_stopping_rounds) is not int
            or self.early_stopping_rounds < 0
        ):
            raise ValueError("La paciencia de boosting debe ser un entero no negativo")
        integers = (
            self.trees,
            self.depth,
            self.min_samples_leaf,
            self.epochs,
            self.patience,
            self.hidden_width,
            self.batch_size,
            self.gpr_max_iter,
            self.gpr_max_fun,
            self.gpr_max_line_search,
            self.gpr_recovery_line_search,
        )
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError("Los parámetros enteros deben ser positivos")
        positive = (
            self.learning_rate,
            self.boosting_rate,
            self.svr_c,
            self.gpr_noise,
            self.gpr_ftol,
            self.gpr_gtol,
        )
        if any(
            isinstance(value, bool) or not isfinite(value) or value <= 0
            for value in positive
        ):
            raise ValueError("Tasas, C y ruido deben ser finitos y positivos")
        nonnegative = (self.regularization, self.svr_epsilon, self.mlp_weight_decay)
        if any(not isfinite(value) or value < 0 for value in nonnegative):
            raise ValueError("Regularización y epsilon deben ser no negativos")


@dataclass(frozen=True)
class PredictorConfig:
    data_dir: Path = DESIGN_DESTINATION
    output_dir: Path = Path("artifacts/predictors/random_forest")
    model: ModelName = "random_forest"
    seed: int = 42
    folds: int = DEFAULT_FOLDS
    trials: int = 0
    parameters: ModelParameters = field(default_factory=ModelParameters)
    cv_seed: int | None = None
    study_path: Path | None = None
    nested_folds: int | None = None
    reuse_dir: Path | None = None
    preparation_dir: Path | None = None
    final_tuning_folds: int | None = None
    candidate_models: tuple[ModelName, ...] = BASE_MODEL_NAMES
    target_index: int | None = None
    tabpfn: TabPFNParameters = field(default_factory=TabPFNParameters)
    device: str = "cpu"
    pruning: PruningParameters = field(default_factory=PruningParameters)
    refit_trees: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.refit_trees) not in (0, 1, len(OUTPUT_COLUMNS)) or any(
            type(value) is not int or not 1 <= value <= self.parameters.trees
            for value in self.refit_trees
        ):
            raise ValueError("Cantidades de árboles para reajuste inválidas")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("El dispositivo predictivo debe ser cpu o cuda")
        if not self.candidate_models or any(
            n not in BASE_MODEL_NAMES for n in self.candidate_models
        ):
            raise ValueError("Familias candidatas inválidas para especialización")
        if self.target_index is not None and self.target_index not in range(3):
            raise ValueError("Respuesta de búsqueda inválida")
        if self.final_tuning_folds is not None and (
            type(self.final_tuning_folds) is not int or self.final_tuning_folds < 2
        ):
            raise ValueError("La búsqueda final requiere al menos dos pliegues")
        if self.nested_folds is not None and (
            type(self.nested_folds) is not int or self.nested_folds < 2
        ):
            raise ValueError("Los pliegues internos deben ser al menos dos")
        if self.model not in (*MODEL_NAMES, "tabpfn"):
            raise ValueError(f"Modelo desconocido: {self.model}")
        if self.model == "tabpfn" and self.trials != 0:
            raise ValueError("TabPFN es fijo y no admite ensayos Optuna")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("La semilla debe ser un entero no negativo")
        if type(self.folds) is not int or self.folds < 2:
            raise ValueError("Se requieren al menos dos pliegues")
        if type(self.trials) is not int or self.trials < 0:
            raise ValueError("El número de ensayos debe ser no negativo")
        if self.cv_seed is not None and (
            type(self.cv_seed) is not int or self.cv_seed < 0
        ):
            raise ValueError("La semilla de pliegues debe ser no negativa")


@dataclass(frozen=True)
class PredictorResult:
    model: Pipeline
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    output_dir: Path
    normalized_mae: float


def response_scale(observed: pd.DataFrame) -> FloatArray:
    scale = observed.std(ddof=0).to_numpy(dtype=float)
    return np.where(scale > 0, scale, 1.0)


def normalized_mae(
    observed: FloatArray, predicted: FloatArray, scale: FloatArray
) -> float:
    return float(np.mean(np.abs(observed - predicted) / scale))
