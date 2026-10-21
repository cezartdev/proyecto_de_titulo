"""Restricciones previas a la consulta, sin inferencias ni entrenamiento."""

import numpy as np
import pytest

from truss_research.optimizacion.propuestas import ProposalParameters, TrainingSupport


def test_support_rejects_unknown_categories_nonintegers_and_far_points() -> None:
    features = np.array(
        [[0, 10, 0.5 + index * 0.02, 1, 2, 2, 3] for index in range(20)], dtype=float
    )
    support = TrainingSupport.fit(features, 0.95)
    assert support.accepts(features).all()
    assert not support.accepts(features[:1] + [1, 0, 0, 0, 0, 0, 0]).any()
    assert not support.accepts(features[:1] + [0, 0.5, 0, 0, 0, 0, 0]).any()
    assert not support.accepts(features[:1] + [0, 0, 2, 0, 0, 0, 0]).any()
    assert support.accepts(np.empty((0, 7))).shape == (0,)


@pytest.mark.parametrize("value", [0, -1, True])
def test_generation_and_evaluation_budgets_are_positive_integers(value: int) -> None:
    with pytest.raises(ValueError):
        ProposalParameters(max_evaluations=value)
    with pytest.raises(ValueError):
        ProposalParameters(max_generated=value)
