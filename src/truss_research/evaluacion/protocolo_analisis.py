"""Parámetros de generalización y eficiencia de datos congelados con la campaña."""

from dataclasses import dataclass

from truss_research.datos.schemas import GenerationSplitParameters
from truss_research.surrogates.configuracion import MODEL_NAMES, ModelName

COMPARISON_MODELS: tuple[ModelName, ...] = ("dummy", "ridge", "catboost", "xgboost")


def validate_models(models: tuple[ModelName, ...]) -> None:
    if (
        not models
        or len(set(models)) != len(models)
        or any(name not in MODEL_NAMES or name == "targetwise" for name in models)
    ):
        raise ValueError("Los controles requieren familias base distintas y conocidas")


@dataclass(frozen=True)
class GeneralizationParameters(GenerationSplitParameters):
    control_models: tuple[ModelName, ...] = COMPARISON_MODELS

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_models(self.control_models)


@dataclass(frozen=True)
class LearningCurveParameters:
    enabled: bool = False
    models: tuple[ModelName, ...] = COMPARISON_MODELS
    train_sizes: tuple[int, ...] = (128, 256, 512)
    subset_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("La activación de curvas debe ser booleana")
        validate_models(self.models)
        if (
            not self.train_sizes
            or tuple(sorted(set(self.train_sizes))) != self.train_sizes
            or any(type(value) is not int or value < 5 for value in self.train_sizes)
        ):
            raise ValueError(
                "Los tamaños de curvas deben ser crecientes y al menos cinco"
            )
        if (
            not self.subset_seeds
            or len(set(self.subset_seeds)) != len(self.subset_seeds)
            or any(type(value) is not int or value < 0 for value in self.subset_seeds)
        ):
            raise ValueError("Semillas de subconjunto inválidas")
