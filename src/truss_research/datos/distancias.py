"""Distancia mixta entre diseños con escalas ajustadas únicamente en entrenamiento."""

import numpy as np
import numpy.typing as npt

from truss_research.cerchas.schemas import INPUT_COLUMNS

type FloatArray = npt.NDArray[np.float64]


def design_distances(
    features: FloatArray, reference: FloatArray, scales: tuple[float, ...]
) -> FloatArray:
    values = np.asarray(scales, dtype=float)
    if (
        values.shape != (len(INPUT_COLUMNS),)
        or np.any(values < 0)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Escalas de distancia inválidas")
    if (
        features.ndim != 2
        or reference.ndim != 2
        or features.shape[1] != len(values)
        or reference.shape[1] != len(values)
    ):
        raise ValueError("La distancia requiere siete entradas por diseño")
    differences = (features[:, None, :] - reference[None, :, :]) / np.where(
        values > 0, values, 1.0
    )
    differences[:, :, 0] = features[:, None, 0] != reference[None, :, 0]
    return np.sqrt(np.mean(differences**2, axis=2))
