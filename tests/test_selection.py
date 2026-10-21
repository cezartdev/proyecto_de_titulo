"""Ocultamiento, Pareto y selección con presupuestos comunes."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from truss_research.evaluacion.pareto import ObjectiveSpace, compliance_report
from truss_research.evaluacion.seleccion import EpisodeReport, selection_metrics
from truss_research.optimizacion.catalogos import load_bank, prepare_selection
from truss_research.optimizacion.comparacion import evaluate_selection_suite
from truss_research.optimizacion.contratos import (
    EpisodeSpec,
    EvolutionParameters,
    SelectionConfig,
)
from truss_research.optimizacion.entorno import (
    EpisodeData,
    create_episode,
    torchrl_environment,
)
from truss_research.optimizacion.estudio import run_study
from truss_research.optimizacion.estudio_estado import StudyConfig
from truss_research.optimizacion.evolucion import EvolutionContext, EvolutionStrategy
from truss_research.surrogates.configuracion import PredictorConfig
from truss_research.surrogates.experimentos import train_predictor


@pytest.fixture
def evolutionary_episode() -> EpisodeData:
    responses = np.array(
        [[float(20 - i), 0.1 + i * 0.05, 0.01 + i * 0.002] for i in range(8)]
    )
    features = np.array(
        [
            [
                i % 5,
                3 + i,
                0.2 + i * 0.1,
                0.3 + i * 0.1,
                0.4 + i * 0.1,
                1 + i % 3,
                1 + i % 4,
            ]
            for i in range(8)
        ]
    )
    return EpisodeData(
        EpisodeSpec(tuple(map(str, range(8))), 0, 7, 42, "evolution"),
        features,
        responses.copy(),
        responses,
        ObjectiveSpace.fit(responses),
    )


def test_gymnasium_and_torchrl_contracts(evolutionary_episode: EpisodeData) -> None:
    from gymnasium.utils.env_checker import check_env
    from torchrl.envs.utils import check_env_specs

    environment = create_episode(evolutionary_episode)
    with pytest.warns(UserWarning, match="infinity"):
        check_env(environment, skip_render_check=True)
    check_env_specs(torchrl_environment(create_episode(evolutionary_episode)))
    observation, info = environment.reset(seed=42)
    assert info == {"queries": 1}
    assert environment.observation_space.contains(observation)
    assert np.count_nonzero(~observation["action_mask"]) == 1
    assert (observation["observed"][observation["action_mask"]] == 0).all()
    with pytest.raises(ValueError, match="consultado"):
        environment.step(0)
    snapshot = observation["observed"].copy()
    environment.step(1)
    np.testing.assert_equal(snapshot, observation["observed"])
    single = create_episode(
        replace(evolutionary_episode, spec=replace(evolutionary_episode.spec, budget=1))
    )
    assert single.observe().terminal
    with pytest.raises(ValueError, match="termin"):
        single.step(1)


@pytest.mark.parametrize("assisted", [False, True])
def test_evolution_uses_real_queries_and_shared_initialization(
    evolutionary_episode: EpisodeData, assisted: bool
) -> None:
    data = evolutionary_episode
    parameters = EvolutionParameters(enabled=True, guided_initialization=False)
    context = EvolutionContext(
        data.space, tuple(np.ptp(data.features, axis=0)), 42, parameters, assisted
    )
    first, second = create_episode(data), create_episode(data)
    strategy, repeated = EvolutionStrategy(context), EvolutionStrategy(context)
    selected = [data.spec.initial_index]
    while not first.observe().terminal:
        action = strategy.choose(first.observe())
        assert action == repeated.choose(second.observe())
        assert action not in selected
        first.step(action)
        second.step(action)
        selected.append(action)
        np.testing.assert_equal(
            first.observe().observed[action], data.responses[action]
        )
    assert len(selected) == data.spec.budget
    assert selected[:4] == [0, 1, 6, 5]
    assert strategy.diagnostics["evolution_stage"] == "offspring"
    assert strategy.diagnostics["projection_distance"] >= 0
    assert strategy.diagnostics["proposal_web_config"] in (0, 1, 2, 3, 4)


def test_evolution_classic_ignores_predictions_and_assisted_preselects(
    evolutionary_episode: EpisodeData,
) -> None:
    data = evolutionary_episode
    environment = create_episode(data)
    for action in (1, 6, 5):
        environment.step(action)
    observation = environment.observe()
    context = EvolutionContext(
        data.space,
        tuple(np.ptp(data.features, axis=0)),
        42,
        EvolutionParameters(enabled=True),
    )
    classic = EvolutionStrategy(context).choose(observation)
    changed = replace(observation, predictions=observation.predictions[::-1].copy())
    assert EvolutionStrategy(context).choose(changed) == classic
    assisted = EvolutionStrategy(replace(context, assisted=True))
    action = assisted.choose(observation)
    scores = {
        int(i): data.space.hypervolume(
            np.vstack(
                (
                    observation.observed[~observation.available],
                    observation.predictions[i],
                )
            )
        )
        for i in np.flatnonzero(observation.available)
    }
    assert scores[action] == max(scores.values())
    assert assisted.diagnostics["proposal_count"] == 4


def test_selection_includes_evolution_with_training_scales(
    design_data: Path, tmp_path: Path
) -> None:
    from truss_research.cerchas.schemas import INPUT_COLUMNS
    from truss_research.datos.procesados import load_design_split

    predictor = tmp_path / "predictor"
    train_predictor(PredictorConfig(design_data, predictor, model="dummy", folds=2))
    config = SelectionConfig(
        design_data,
        predictor,
        tmp_path / "selection_evolution",
        catalog_size=4,
        budgets=(4,),
        seeds=(42,),
        episodes_per_seed=1,
        evolution=EvolutionParameters(enabled=True, population_size=2),
    )
    bank = load_bank(config)
    training = load_design_split(design_data).training
    np.testing.assert_equal(
        bank.feature_ranges,
        np.ptp(training.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=float), axis=0),
    )
    result = evaluate_selection_suite(config, ())
    assert set(result["strategy"]) == {
        "random",
        "ranking",
        "greedy",
        "nsga2",
        "nsga2_surrogate",
    }
    assert result["queries"].eq(4).all()
    traces = pd.read_csv(config.output_dir / "traces.csv")
    assert not traces.duplicated(["strategy", "iter_id"]).any()
    assert "projection_distance" in traces


@pytest.fixture
def episode() -> EpisodeData:
    responses = np.array([[12.0, 0.7, 0.02], [10.0, 0.8, 0.01], [9.0, 1.2, 0.09]])
    space = ObjectiveSpace.fit(responses)
    return EpisodeData(
        EpisodeSpec(("a", "b", "c"), 0, 2, 42, "fixture"),
        np.arange(21, dtype=float).reshape(3, 7),
        np.tile(responses[0], (3, 1)),
        responses,
        space,
    )


def test_pareto_and_reference_are_fixed(episode: EpisodeData) -> None:
    np.testing.assert_equal(
        episode.space.front(episode.responses), episode.responses[1:2]
    )
    assert episode.space.hypervolume(episode.responses) > episode.space.hypervolume(
        episode.responses[:1]
    )
    assert episode.space.hypervolume(np.empty((0, 3))) == 0
    outside = np.array([[1000.0, 0.7, 0.01]])
    assert episode.space.transform(outside)[0, 0] > 1
    assert episode.space.hypervolume(outside) == 0


def test_three_objectives_reward_utilization_within_constraints() -> None:
    responses = np.array(
        [[10.0, 0.2, 0.02], [10.0, 0.9, 0.02], [11.0, 0.95, 0.025], [9.0, 1.1, 0.01]]
    )
    space = ObjectiveSpace.fit(responses)
    assert space.transform(responses).shape == (4, 3)
    assert space.hypervolume(responses[1:2]) > space.hypervolume(responses[:1])
    assert space.transform(responses)[1, 1] < space.transform(responses)[0, 1]
    np.testing.assert_equal(space.front(responses), responses[1:3])
    assert space.hypervolume(responses[3:]) == 0
    report = compliance_report(
        responses,
        np.array(
            [
                [-1.0, 0.2, 0.02],
                [10.0, -0.1, 0.02],
                [11.0, 0.9, -0.03],
                [9.0, 1.1, 0.01],
            ]
        ),
    )
    assert report["physically_invalid_prediction_count"] == 3


def test_three_objectives_keep_tradeoffs_and_known_hypervolume() -> None:
    from truss_research.optimizacion.evolucion import mixed_problem

    responses = np.array(
        [[9.0, 0.5, 0.02], [10.0, 0.8, 0.02], [10.0, 0.5, 0.01], [11.0, 0.4, 0.03]]
    )
    space = ObjectiveSpace.fit(responses)
    np.testing.assert_equal(space.front(responses), np.unique(responses[:3], axis=0))
    assert mixed_problem().n_obj == 3
    from truss_research.cerchas.schemas import StructuralLimits

    known = ObjectiveSpace(
        (0.0, -1.0, 0.0),
        (10.0, 1.0, 0.1),
        (10.0, 0.0, 0.08),
        StructuralLimits(serviceability_divisor=250),
    )
    assert known.hypervolume(np.array([[5.0, 0.5, 0.04]])) == pytest.approx(0.1)


def test_empty_reference_and_absent_failure_class_are_explicit() -> None:
    responses = np.array([[10.0, 1.2, 0.09]])
    space = ObjectiveSpace.fit(responses)
    metrics = selection_metrics(EpisodeReport(responses, (0,), (0.0,), space))
    assert metrics["queries"] == 1
    assert metrics["normalized_auc"] is None
    assert metrics["queries_to_target"] is None
    assert metrics["relative_metric_reason"]
    feasible = np.array([[10.0, 0.5, 0.02]])
    report = compliance_report(feasible, feasible)
    assert report["utilization"]["fail_class_absent"] is True


def test_hidden_labels_do_not_change_observation(episode: EpisodeData) -> None:
    modified = episode.responses.copy()
    modified[1] *= 10
    first = create_episode(episode)
    second = create_episode(replace(episode, responses=modified))
    for field in ("features", "predictions", "observed", "available"):
        np.testing.assert_equal(
            getattr(first.observe(), field), getattr(second.observe(), field)
        )
    observation = first.observe()
    assert not hasattr(observation, "responses")
    assert observation.queries == 1
    observation.observed[:] = 999
    assert first.observe().observed[0, 0] == 12
    with pytest.raises(ValueError, match="ya fue consultado"):
        first.query(0)
    with pytest.raises(ValueError, match="fuera"):
        first.query(99)
    result = first.query(1)
    assert result.reward > 0 and result.terminated
    with pytest.raises(ValueError, match="terminó"):
        first.query(2)


def test_selection_manifests_and_budgets(design_data: Path, tmp_path: Path) -> None:
    predictor = tmp_path / "predictor"
    train_predictor(PredictorConfig(design_data, predictor, model="dummy", folds=2))
    config = SelectionConfig(
        design_data,
        predictor,
        tmp_path / "selection",
        catalog_size=4,
        budgets=(2, 3),
        seeds=(42,),
        episodes_per_seed=2,
    )
    summary = evaluate_selection_suite(config, ())
    assert len(summary) == 12
    assert summary["queries"].eq(summary["budget"]).all()
    assert summary["preparation_seconds"].ge(0).all()
    np.testing.assert_allclose(
        summary["strategy_seconds"],
        summary["preparation_seconds"] + summary["decision_seconds"],
    )
    trace = pd.read_csv(config.output_dir / "traces.csv")
    assert not trace.duplicated(["strategy", "episode_id", "iter_id"]).any()
    prepared = replace(config, output_dir=tmp_path / "public")
    prepare_selection(prepared)
    public = pd.read_csv(prepared.output_dir / "public_candidates.csv")
    assert not {"mass_kg", "util_ratio", "z_disp_m"} & set(public)


def test_final_never_fits_models(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = StudyConfig(
        tmp_path / "study",
        data_dir=design_data,
        models=("dummy",),
        seeds=(42,),
        outer_folds=2,
        trials=0,
        catalog_size=4,
        budgets=(2,),
        episodes_per_seed=1,
        rl_steps=3,
        rl_batch=2,
        rl_warmup=1,
        rl_hidden=8,
        background_size=4,
        permutation_rounds=1,
    )
    run_study(config, "validacion")

    def forbidden_fit(
        self: Pipeline, features: pd.DataFrame, responses: pd.DataFrame
    ) -> None:
        raise AssertionError("No ajustar modelos durante la evaluación final")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    run_study(config, "final", True)
    assert (config.output_dir / "freeze.json").is_file()
    metrics = pd.read_csv(
        config.output_dir / "tasks/test/predictor/dummy/42/attempt_001/metrics.csv"
    )
    assert metrics["evaluation_split"].eq("test").all()
