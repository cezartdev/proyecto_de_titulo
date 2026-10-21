"""Contrato tabular y espacio de diseño documentado para cerchas de 20 m."""

from dataclasses import dataclass
from math import isfinite
from typing import Final

INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "web_config",
    "n_trusses",
    "lr_height_m",
    "lmrm_height_m",
    "m_height_m",
    "subdiv1",
    "subdiv2",
)

OUTPUT_COLUMNS: Final[tuple[str, ...]] = ("mass_kg", "util_ratio", "z_disp_m")

INTEGER_COLUMNS: Final[tuple[str, ...]] = (
    "web_config",
    "n_trusses",
    "subdiv1",
    "subdiv2",
)

WEB_CONFIG_VALUES: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)

WEB_CONFIG_NAMES: Final[dict[int, str]] = {
    0: "Warren de doble intersección",
    1: "Warren de doble intersección sin montantes",
    2: "Pratt",
    3: "Howe",
    4: "Warren simple",
}

DESIGN_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "web_config": (0, 4),
    "n_trusses": (3, 50),
    "lr_height_m": (0.2, 3.5),
    "lmrm_height_m": (0.2, 3.5),
    "m_height_m": (0.4, 3.5),
    "subdiv1": (1, 10),
    "subdiv2": (1, 10),
}


@dataclass(frozen=True)
class StructuralResponse:
    """Respuestas agregadas: masa en kg, utilización y desplazamiento en m."""

    mass_kg: float
    util_ratio: float
    z_disp_m: float


@dataclass(frozen=True)
class StructuralLimits:
    """Umbrales operativos del estudio, sin certificación por elemento."""

    span_m: float = 20.0
    serviceability_divisor: float | None = 700.0
    utilization_limit: float = 1.0

    def __post_init__(self) -> None:
        values = (self.span_m, self.utilization_limit)
        if self.serviceability_divisor is not None:
            values = (*values, self.serviceability_divisor)
        if not all(
            not isinstance(value, bool) and isfinite(value) and value > 0
            for value in values
        ):
            raise ValueError(
                "Los parámetros de los umbrales deben ser finitos y positivos"
            )

    @property
    def displacement_limit_m(self) -> float | None:
        return (
            self.span_m / self.serviceability_divisor
            if self.serviceability_divisor is not None
            else None
        )


@dataclass(frozen=True)
class ThresholdCompliance:
    utilization: bool
    displacement: bool

    @property
    def combined(self) -> bool:
        return self.utilization and self.displacement


def assess_response(
    response: StructuralResponse, limits: StructuralLimits
) -> ThresholdCompliance:
    values = (response.mass_kg, response.util_ratio, response.z_disp_m)
    if not all(isfinite(value) for value in values):
        raise ValueError("Las respuestas deben ser finitas")
    if response.mass_kg <= 0 or response.util_ratio < 0 or response.z_disp_m < 0:
        raise ValueError(
            "La masa debe ser positiva y las otras respuestas no negativas"
        )
    return ThresholdCompliance(
        utilization=response.util_ratio <= limits.utilization_limit,
        displacement=limits.displacement_limit_m is None
        or response.z_disp_m <= limits.displacement_limit_m,
    )
