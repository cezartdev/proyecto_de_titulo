"""Configuración, nombres de columnas y resultados de preparación."""

from dataclasses import dataclass, field
from math import isclose, isfinite
from pathlib import Path
from typing import Final, Literal

import pandas as pd

from truss_research.cerchas.schemas import StructuralLimits

GROUP_PARTITIONS = ("train", "calibration", "validation", "test")
MIN_FINITE_CALIBRATION = 19


@dataclass(frozen=True)
class GenerationSplitParameters:
    fractions: tuple[float, ...] = (0.64, 0.16, 0.10, 0.10)
    min_training: int = 128
    min_calibration: int = 40
    min_evaluation: int = 32

    def __post_init__(self) -> None:
        if (
            len(self.fractions) != len(GROUP_PARTITIONS)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
                for value in self.fractions
            )
            or not isclose(sum(self.fractions), 1.0)
        ):
            raise ValueError("Las cuatro proporciones por generaciones deben sumar uno")
        if any(
            type(value) is not int or value < 1
            for value in (
                self.min_training,
                self.min_calibration,
                self.min_evaluation,
            )
        ):
            raise ValueError("Los tamaños agrupados deben ser enteros positivos")
        if self.min_calibration < MIN_FINITE_CALIBRATION:
            raise ValueError("La calibración al 95 % requiere al menos 19 diseños")


DEFAULT_SOURCE: Final[Path] = Path("data/raw/240214-C1-20m.xlsx")

DEFAULT_DESTINATION: Final[Path] = Path("data/processed_evolution")

SHEET_NAME: Final[str] = "TOTAL DATA"

RAW_ID_COLUMN: Final[str] = "Unnamed: 0"

ID_COLUMN: Final[str] = "iter_id"

GENERATION_COLUMN: Final[str] = "generation"

SOLUTION_COLUMN: Final[str] = "solution"

ID_PATTERN: Final[str] = r"\{(?P<generation>\d+);(?P<solution>\d+)\}"

IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    ID_COLUMN,
    GENERATION_COLUMN,
    SOLUTION_COLUMN,
)

COMPLIANCE_COLUMNS: Final[tuple[str, ...]] = (
    "meets_util_threshold",
    "meets_z_threshold",
    "meets_thresholds",
)

COLUMN_MAPPING: Final[dict[str, str]] = {
    "Mass": "mass_kg",
    "Average Util": "util_ratio",
    "Z Disp": "z_disp_m",
    "Web Config": "web_config",
    "N of Trusses": "n_trusses",
    "L-R Height": "lr_height_m",
    "LM-RM Height": "lmrm_height_m",
    "M Height": "m_height_m",
    "Subdiv1": "subdiv1",
    "Subdiv2": "subdiv2",
}

SPLIT_FILENAMES: Final[dict[str, str]] = {
    "train": "train.csv",
    "validation": "val.csv",
    "test": "test.csv",
}

FULL_FILENAME: Final[str] = "full_clean.csv"

MANIFEST_FILENAME: Final[str] = "manifest.json"

DESIGN_MEMBERSHIP_FILENAME: Final[str] = "design_membership.csv"

CALIBRATION_FILENAME: Final[str] = "calibration.csv"

DESIGN_DESTINATION: Final[Path] = DEFAULT_DESTINATION

type SplitProtocol = Literal["design", "first_appearance"]

SPLIT_PROTOCOLS: Final[tuple[SplitProtocol, ...]] = ("design", "first_appearance")

PARTITION_POLICIES: Final[dict[SplitProtocol, str]] = {
    "design": "unique_design_stratified_by_web_config",
    "first_appearance": "unique_design_first_appearance_chronological",
}


class DatasetError(ValueError):
    """Entrada incompatible con el contrato del dataset."""


@dataclass(frozen=True)
class DatasetSpec:
    expected_generations: int = 100
    solutions_per_generation: int = 50

    def __post_init__(self) -> None:
        values = (self.expected_generations, self.solutions_per_generation)
        if not all(type(value) is int and value > 0 for value in values):
            raise DatasetError("Los conteos esperados deben ser enteros positivos")


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42
    protocol: SplitProtocol = "design"

    def __post_init__(self) -> None:
        if self.protocol not in SPLIT_PROTOCOLS:
            raise DatasetError(f"Protocolo de partición desconocido: {self.protocol}")
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if not all(isfinite(value) and 0 < value < 1 for value in fractions):
            raise DatasetError("Las proporciones deben estar entre cero y uno")
        if not isclose(sum(fractions), 1.0):
            raise DatasetError("Las proporciones deben sumar uno")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise DatasetError(
                "La semilla debe ser un entero válido de 32 bits sin signo"
            )


@dataclass(frozen=True)
class PreparationConfig:
    source: Path = DEFAULT_SOURCE
    destination: Path = DEFAULT_DESTINATION
    sheet_name: str = SHEET_NAME
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    split: SplitConfig = field(default_factory=SplitConfig)
    limits: StructuralLimits = field(default_factory=StructuralLimits)
    calibration_fraction: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isfinite(self.calibration_fraction)
            or not 0 <= self.calibration_fraction < 1
        ):
            raise DatasetError("La fracción de calibración debe estar entre cero y uno")


@dataclass(frozen=True)
class TableSplit:
    training: pd.DataFrame
    validation: pd.DataFrame
    testing: pd.DataFrame
    calibration: pd.DataFrame | None = None

    def as_mapping(self) -> dict[str, pd.DataFrame]:
        tables = {
            "train": self.training,
            "validation": self.validation,
            "test": self.testing,
        }
        if self.calibration is not None:
            tables["calibration"] = self.calibration
        return tables


@dataclass(frozen=True)
class TableDiagnostics:
    rows: int
    generations: tuple[int, ...]
    web_config_counts: dict[int, int]


@dataclass(frozen=True)
class PreparationResult:
    table: pd.DataFrame
    split: TableSplit
    diagnostics: dict[str, TableDiagnostics]
    output_paths: dict[str, Path]
