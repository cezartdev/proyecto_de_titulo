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


DISTANCE_QUANTILES = (0.25, 0.5, 0.75)


def distance_regions(
    training: FloatArray, features: FloatArray
) -> npt.NDArray[np.int64]:
    """Asigna regiones de proximidad con cortes derivados solo del entrenamiento."""
    if len(training) < 2:
        raise ValueError("La proximidad requiere al menos dos diseños de entrenamiento")
    scales = tuple(float(value) for value in np.ptp(training, axis=0))
    internal = design_distances(training, training, scales)
    np.fill_diagonal(internal, np.inf)
    boundaries = np.quantile(internal.min(axis=1), DISTANCE_QUANTILES)
    distances = design_distances(features, training, scales).min(axis=1)
    return np.searchsorted(boundaries, distances, side="right")
