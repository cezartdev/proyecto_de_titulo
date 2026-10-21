"""Intervalos split conformal; calibración independiente del ajuste y la selección."""

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.surrogates.configuracion import FloatArray

NOMINAL_COVERAGES = (0.90, 0.95)


@dataclass(frozen=True)
class SplitConformal:
    residuals: FloatArray
    coverages: tuple[float, ...] = NOMINAL_COVERAGES

    def __post_init__(self) -> None:
        values = np.asarray(self.residuals, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != len(OUTPUT_COLUMNS)
            or not len(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Calibración requiere residuos finitos de tres respuestas")
        if not self.coverages or any(not 0 < c < 1 for c in self.coverages):
            raise ValueError("Coberturas nominales fuera de (0, 1)")
        frozen = values.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "residuals", frozen)

    def widths(self, coverage: float) -> FloatArray:
        if coverage not in self.coverages:
            raise ValueError("Cobertura no registrada en la calibración")
        rank = ceil((len(self.residuals) + 1) * coverage)
        if rank > len(self.residuals):
            raise ValueError(
                "Calibración insuficiente para una cota finita a esa cobertura"
            )
        return np.sort(np.abs(self.residuals), axis=0)[rank - 1]

    def table(self, predictions: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
        if tuple(predictions.columns) != OUTPUT_COLUMNS or not predictions.index.equals(
            actual.index
        ):
            raise ValueError("Predicciones y respuestas de intervalos desalineadas")
        rows: list[pd.DataFrame] = []
        for coverage in self.coverages:
            widths = self.widths(coverage)
            for column, name in enumerate(OUTPUT_COLUMNS):
                lower = predictions[name].to_numpy() - widths[column]
                upper = predictions[name].to_numpy() + widths[column]
                rows.append(
                    pd.DataFrame(
                        {
                            "row_index": predictions.index,
                            "target": name,
                            "nominal_coverage": coverage,
                            "lower": lower,
                            "upper": upper,
                            "width": upper - lower,
                            "covered": actual[name].between(lower, upper).to_numpy(),
                        }
                    )
                )
        return pd.concat(rows, ignore_index=True)
