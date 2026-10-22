"""Evaluador privado que revela una respuesta registrada por consulta válida."""


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
