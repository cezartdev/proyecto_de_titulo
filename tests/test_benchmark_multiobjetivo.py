"""Calidad observada, tres objetivos y emparejamiento del benchmark de regresores."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.evaluacion.benchmark_multiobjetivo import (
    catalogue_comparison,
    utility_intervals,
    write_benchmark,
)
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.optimizacion.contratos import (
    AnalysisParameters,
    EvolutionParameters,
)
from truss_research.optimizacion.estudio_estado import StudyConfig
from truss_research.optimizacion.propuestas import ProposalParameters


@pytest.fixture
def small_case(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, StudyConfig]:
    training = pd.DataFrame(
        [[300, 0.2, 0.02], [350, 0.3, 0.018], [330, 0.4, 0.022], [320, 0.35, 0.019]],
        columns=OUTPUT_COLUMNS,
    )
    values = np.array(
        [[100, 0.8, 0.01], [200, 0.7, 0.015], [400, 0.2, 0.025], [80, 1.2, 0.01]]
    )
    frames = []
    for model, estimates in (("ridge", values), ("dummy", values[::-1])):
        for seed in (42, 43):
            table = pd.DataFrame(values, columns=OUTPUT_COLUMNS)
            table[[f"pred_{name}" for name in OUTPUT_COLUMNS]] = estimates
            frames.append(
                table.assign(
                    iter_id=[f"d{i}" for i in range(len(values))],
                    generation=10,
                    web_config=1,
                    model=model,
                    seed=seed,
                    split="validation",
                )
            )
    config = StudyConfig(
        tmp_path / "study",
        models=("dummy", "ridge"),
        seeds=(42, 43),
        budgets=(5, 10),
        evolution=EvolutionParameters(population_size=4),
        proposals=ProposalParameters(evaluation_episodes=2),
        analysis=AnalysisParameters(bootstrap_repeats=20),
        objective_mode="three",
    )
    return training, pd.concat(frames, ignore_index=True), config


def test_three_objectives_include_utilization() -> None:
    space = ObjectiveSpace.fit(np.array([[100, 0.4, 0.01], [110, 0.6, 0.02]]))
    lower = space.hypervolume(np.array([[100, 0.4, 0.01]]))
    higher = space.hypervolume(np.array([[100, 0.8, 0.01]]))
    assert higher > lower
    assert space.hypervolume(np.array([[100, 1.1, 0.01]])) == 0


def test_common_catalogues_initialization_and_effective_budgets(
    small_case: tuple,
) -> None:
    training, predictions, config = small_case
    summary, trace = catalogue_comparison(training, predictions, config)
    assert set(summary.model) == {"dummy", "ridge", "random_control", "oracle_ranking"}
    assert summary.groupby("scenario").initial_indices.nunique().eq(1).all()
    assert summary.loc[summary.budget.eq(10), "effective_budget"].eq(8).all()
    assert summary.loc[summary.budget.eq(10), "catalogue_exhausted"].all()
    assert trace.groupby(["scenario", "model", "seed"]).iter_id.nunique().eq(4).all()
    first = summary.loc[summary.budget.eq(5)].groupby("model").auc.mean()
    assert first["ridge"] > first["dummy"]
    assert summary.objective_mode.eq("three").all()


def test_hypervolume_uses_observed_values_and_not_optimistic_predictions(
    small_case: tuple,
) -> None:
    training, predictions, config = small_case
    summary, trace = catalogue_comparison(training, predictions, config)
    rows = trace.loc[trace.model.eq("dummy") & trace.seed.eq(42) & trace.scenario.eq(0)]
    row = summary.loc[
        summary.model.eq("dummy") & summary.seed.eq(42) & summary.scenario.eq(0)
    ].iloc[0]
    initial = training.iloc[[int(value) for value in row.initial_indices.split(",")]][
        list(OUTPUT_COLUMNS)
    ].to_numpy()
    actual = predictions.drop_duplicates("iter_id").set_index("iter_id")
    space = ObjectiveSpace.fit(training[list(OUTPUT_COLUMNS)].to_numpy())
    for selected in rows.itertuples():
        initial = np.vstack(
            (
                initial,
                actual.loc[selected.iter_id, list(OUTPUT_COLUMNS)].to_numpy(
                    dtype=float
                ),
            )
        )
        assert selected.hypervolume == pytest.approx(space.hypervolume(initial))


def test_model_orders_do_not_read_hidden_labels(small_case: tuple) -> None:
    training, predictions, config = small_case
    _, first = catalogue_comparison(training, predictions, config)
    changed = predictions.copy()
    changed["mass_kg"] *= 2
    _, second = catalogue_comparison(training, changed, config)
    columns = ["model", "seed", "scenario", "query", "iter_id"]
    pd.testing.assert_frame_equal(
        first.loc[first.model.ne("oracle_ranking"), columns],
        second.loc[second.model.ne("oracle_ranking"), columns],
    )


def test_paired_intervals_average_seed_replicas_and_cache_rejects_changes(
    small_case: tuple, tmp_path: Path
) -> None:
    training, predictions, config = small_case
    result = write_benchmark((training, predictions), config, tmp_path / "utility")
    intervals = utility_intervals(result, config)
    assert intervals.scenario_pairs.eq(2).all()
    write_benchmark((training, predictions), config, tmp_path / "utility")
    with pytest.raises(ValueError, match="otros datos"):
        write_benchmark(
            (training, predictions),
            replace(config, proposals=replace(config.proposals, evaluation_episodes=3)),
            tmp_path / "utility",
        )


def test_family_selection_prioritizes_validation_auc_over_mae(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.evaluacion import busqueda_evolutiva as campaign
    from truss_research.optimizacion.estudio_estado import TaskRecord
    from truss_research.surrogates.experimentos import write_json

    config = StudyConfig(tmp_path / "study", models=("dummy", "ridge"), seeds=(42,))
    paths = {name: tmp_path / name for name in config.models}
    records = []
    for name, path in paths.items():
        path.mkdir()
        write_json(
            path / "run.json", {"normalized_mae": 1.0 if name == "dummy" else 0.1}
        )
        records.append(
            campaign.PredictorReference(
                name, 42, TaskRecord(f"predictor/{name}/42", directory=str(path))
            )
        )
    monkeypatch.setattr(campaign, "predictor_records", lambda path: tuple(records))
    utility = config.output_dir / "evolution/predictor_utility/validation"
    utility.mkdir(parents=True)
    pd.DataFrame(
        {
            "model": ["dummy", "ridge"],
            "role": "regressor",
            "budget": min(config.budgets),
            "auc": [2.0, 1.0],
        }
    ).to_csv(utility / "summary.csv", index=False)
    assert campaign.choose_predictors(config, paths)[:2] == ("dummy", "ridge")
