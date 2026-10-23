"""Lecturas y tiempos sobre tablas sintéticas pequeñas, sin inferencia."""

from pathlib import Path

import pandas as pd
import pytest

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.evaluacion.lectura_resultados import (
    interval_reading,
    physical_scope_table,
    predictor_interpretation,
    shap_interpretation,
    time_comparison,
)


def test_physical_scope_uses_academic_labels_without_changing_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "physical_provenance.csv"
    pd.DataFrame(
        {
            "element": ["Material", "Origen", "Solver"],
            "status": ["Confirmado", "Informado por el autor", "Pendiente"],
            "confirmation_source": ["Profesora", "Conversación", ""],
            "confirmation_required": ["", "", "Consultar a la profesora"],
        }
    ).to_csv(path, index=False)
    original = path.read_bytes()
    table = physical_scope_table(tmp_path)
    assert list(table.columns) == ["Aspecto del caso", "Alcance documental"]
    text = table.to_string(index=False)
    assert "profesora" not in text.lower() and "autor" not in text.lower()
    assert "Información no disponible" in text
    assert path.read_bytes() == original


def test_time_comparison_separates_shared_work_and_per_design_latencies(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "model": ["ridge", "ridge"],
            "seed": [42, 43],
            "device": ["cpu", "cpu"],
            "tuning_seconds": [10, 0],
            "crossfit_seconds": [20, 0],
            "fit_seconds": [2, 4],
        }
    ).to_csv(tmp_path / "predictor_costs.csv", index=False)
    pd.DataFrame(
        {
            "model": ["ridge", "ridge", "ridge"],
            "split": ["test", "test", "validation"],
            "individual_median_seconds": [1, 3, 100],
            "batch_median_seconds": [2, 8, 100],
            "batch_rows": [2, 4, 1],
        }
    ).to_csv(tmp_path / "predictive_inference_costs.csv", index=False)
    row = time_comparison(tmp_path).loc["ridge"]
    assert row.fit_phase_minutes == pytest.approx(0.6)
    assert row.adjustment_seeds == 2
    assert row.individual_ms == 2000
    assert row.batch_ms == 5000
    assert row.batch_per_design_ms == 1500
    import matplotlib.pyplot as plt

    from truss_research.evaluacion.figuras import predictor_inference_figure

    figure = predictor_inference_figure(tmp_path)
    assert all(axis.get_xscale() == "log" for axis in figure.axes)
    plt.close(figure)


def test_predictor_reading_preserves_validation_selection_and_exact_ties(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "model": ["ridge", "svr", "dummy"],
            "split": ["test"] * 3,
            "budget": [8] * 3,
            "role": ["regressor"] * 3,
            "mean_normalized_auc": [0.7, 0.8, 0.5],
        }
    ).to_csv(tmp_path / "predictor_utility_summary.csv", index=False)
    pd.DataFrame(
        {"model": ["ridge", "svr"], "split": ["test"] * 2, "mean": [0.1, 0.1]}
    ).to_csv(tmp_path / "predictive_model_summary.csv", index=False)
    text = predictor_interpretation(tmp_path, "ridge", 8)
    assert "svr (0.8000)" in text
    assert "ridge, svr (0.1000)" in text
    assert "congelado" in text and "ridge" in text


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [((-0.2, -0.1), "debajo"), ((-0.1, 0.1), "incluye cero"), ((0.1, 0.2), "encima")],
)
def test_interval_reading_distinguishes_sign_from_uncertainty(
    bounds: tuple[float, float], expected: str
) -> None:
    text = interval_reading(
        pd.Series(
            {
                "strategy": "rl",
                "baseline": "random",
                "mean_auc_difference": sum(bounds) / 2,
                "lower": bounds[0],
                "upper": bounds[1],
            }
        )
    )
    assert expected in text


def test_shap_reading_excludes_constant_model_and_uses_requested_seed(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        [
            {
                "model": model,
                "seed": seed,
                "target": target,
                "variable": variable,
                "abs_shap": value if model != "dummy" else 0,
            }
            for model in ("dummy", "ridge")
            for seed in (42, 43)
            for target in OUTPUT_COLUMNS
            for variable, value in (("n_trusses", 3), ("m_height_m", 1))
        ]
    ).to_csv(tmp_path / "shap_global.csv", index=False)
    text = shap_interpretation(tmp_path, 42)
    assert "1/1" in text and "75.0" in text and "n_trusses" in text
    assert "varianza explicada" in text
    import matplotlib.pyplot as plt

    from truss_research.evaluacion.figuras import shap_comparison_figure

    figure = shap_comparison_figure(tmp_path, 42)
    for axis in figure.axes[: len(OUTPUT_COLUMNS)]:
        values = axis.images[0].get_array()
        assert sorted(values.sum(axis=1)) == pytest.approx([0, 100])
    plt.close(figure)
