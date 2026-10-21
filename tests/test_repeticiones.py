"""Intentos repetidos, identidad geométrica y separación entre retorno y calidad."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from truss_research.datos.artefactos import read_artifact
from truss_research.evaluacion.benchmark_seleccion import (
    assert_trajectory,
    replay_episode,
    replay_simulation,
)
from truss_research.evaluacion.informe_estudio import summarize_study
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.evaluacion.seleccion import EpisodeReport, selection_metrics
from truss_research.optimizacion.comparacion import evaluate_episode
from truss_research.optimizacion.contratos import EpisodeSpec
from truss_research.optimizacion.entorno import EpisodeData, create_episode
from truss_research.optimizacion.entrenamiento import (
    policy_inputs,
    train_policy,
    training_signature,
)
from truss_research.optimizacion.estudio import run_study
from truss_research.optimizacion.estudio_estado import StudyConfig
from truss_research.optimizacion.politica import (
    DQNParameters,
    PolicyConfig,
    load_policy,
)
from truss_research.optimizacion.variantes_rl import RLParameters
from truss_research.surrogates.configuracion import PredictorConfig
from truss_research.surrogates.experimentos import train_predictor


@pytest.fixture
def repeat_data() -> EpisodeData:
    responses = np.array([[10.0, 0.5, 0.02], [9.0, 0.7, 0.01], [12.0, 0.4, 0.02]])
    space = ObjectiveSpace.fit(responses)
    return EpisodeData(
        EpisodeSpec(("a", "b", "c"), 0, 3, 42, "repeticiones"),
        np.arange(21, dtype=float).reshape(3, 7),
        responses.copy(),
        responses,
        space,
        space.hypervolume(responses),
        "weighted_marginal",
        True,
        1.0,
    )


def test_repeat_consumes_attempt_but_preserves_responses_and_auc(
    repeat_data: EpisodeData,
) -> None:
    environment = create_episode(repeat_data)
    initial = environment.observe()
    repeated_reward = environment.step(0)[1]
    repeated = environment.observe()
    assert repeated_reward == pytest.approx(-1 / 3)
    assert (repeated.attempt_count, repeated.queries, repeated.repeat_attempts) == (
        2,
        1,
        1,
    )
    np.testing.assert_array_equal(initial.observed, repeated.observed)
    assert repeated.selectable.all() and not repeated.available[0]
    final_reward = environment.step(1)[1]
    assert environment.observe().terminal
    assert not environment.observe().selectable.any()
    selected = (0, 0, 1)
    volumes = tuple(
        repeat_data.space.hypervolume(repeat_data.responses[list(selected[:number])])
        for number in (1, 2, 3)
    )
    metric = selection_metrics(
        EpisodeReport(repeat_data.responses, selected, volumes, repeat_data.space)
    )
    assert environment.initial_reward + repeated_reward + final_reward == pytest.approx(
        metric["normalized_auc"] - 1 / 3
    )
    with pytest.raises(ValueError, match="termin"):
        environment.step(2)
    environment.reset()
    assert environment.observe().attempt_count == environment.observe().queries == 1


def test_exact_aliases_share_identity_and_conflicts_fail(
    repeat_data: EpisodeData,
) -> None:
    features, responses = repeat_data.features.copy(), repeat_data.responses.copy()
    features[1], responses[1] = features[0], responses[0]
    data = replace(repeat_data, features=features, responses=responses)
    environment = create_episode(data)
    assert not environment.observe().available[:2].any()
    assert environment.step(1)[1] == pytest.approx(-1 / 3)
    assert environment.observe().queries == 1
    with pytest.raises(ValueError, match="respuestas distintas"):
        create_episode(replace(data, responses=repeat_data.responses))
    features[1, 2] = np.nextafter(features[0, 2], np.inf)
    assert create_episode(replace(data, features=features)).observe().available[1]


def test_unpenalized_repeat_and_primary_mask(repeat_data: EpisodeData) -> None:
    assert create_episode(replace(repeat_data, repeat_penalty=0.0)).step(0)[1] == 0
    primary = create_episode(
        replace(repeat_data, allow_repeats=False, repeat_penalty=0.0)
    )
    assert not primary.observe().selectable[0]
    with pytest.raises(ValueError, match="consultado"):
        primary.step(0)


def test_repeat_training_and_resume_are_exact(
    design_data: Path, tmp_path: Path
) -> None:
    predictor = tmp_path / "predictor"
    train_predictor(PredictorConfig(design_data, predictor, model="dummy", folds=2))
    parameters = DQNParameters(
        transitions=8,
        hidden_width=8,
        batch_size=2,
        replay_size=16,
        warmup=1,
        validation_interval=4,
        target_interval=2,
        epsilon_steps=8,
        train_frequency=1,
    )
    config = PolicyConfig(
        design_data,
        predictor,
        tmp_path / "complete",
        catalog_size=4,
        budgets=(3, 4),
        validation_episodes=1,
        parameters=parameters,
        allow_repeats=True,
        repeat_penalty=1.0,
    )
    bank, _, _ = policy_inputs(config)
    assert training_signature(config, bank) != training_signature(
        replace(config, repeat_penalty=0.0), bank
    )
    train_policy(config)
    short = replace(
        config,
        output_dir=tmp_path / "short",
        parameters=replace(parameters, transitions=4),
    )
    train_policy(short)
    resumed = replace(
        config, output_dir=tmp_path / "resumed", resume_dir=short.output_dir
    )
    train_policy(resumed)
    first = load_policy(config.output_dir, predictor).components.network.state_dict()
    second = load_policy(resumed.output_dir, predictor).components.network.state_dict()
    for name in first:
        np.testing.assert_array_equal(first[name].numpy(), second[name].numpy())
    history = pd.read_csv(resumed.output_dir / "learning_curve.csv")
    assert history.step.tolist() == [4, 8]
    policy = load_policy(config.output_dir, predictor)
    with torch.no_grad():
        for parameter in policy.components.network.parameters():
            parameter.zero_()
    _, validation, _ = policy_inputs(config)
    spec = validation.episodes[0]
    metrics, original = evaluate_episode((validation, spec), policy)
    assert metrics["repeat_attempts"] == spec.budget - 1
    measured, trace = replay_episode((validation, spec), lambda: policy)
    expected = tuple(row["iter_id"] for row in original)
    assert_trajectory(trace, expected)
    assert_trajectory(replay_simulation((validation, spec), lambda: policy), expected)
    assert measured["unique_queries"] == 1 and measured["attempts"] == spec.budget


def test_repetition_study_exports_counts_and_paired_contrasts(
    design_data: Path, tmp_path: Path
) -> None:
    config = StudyConfig(
        tmp_path / "study",
        data_dir=design_data,
        models=("dummy",),
        seeds=(42,),
        outer_folds=2,
        final_tuning_folds=2,
        trials=0,
        catalog_size=4,
        budgets=(3, 4),
        episodes_per_seed=1,
        rl_steps=4,
        rl_batch=2,
        rl_warmup=1,
        rl_hidden=8,
        validation_interval=2,
        background_size=4,
        permutation_rounds=1,
        rl=RLParameters(algorithms=("double_dqn",), repetition_ablations=True),
    )
    run_study(config, "todo")
    records = read_artifact(config.output_dir / "study.json")["tasks"]
    root = Path(
        next(
            record["directory"]
            for record in records
            if record["key"] == "test/decision"
        )
    )
    summary = pd.read_csv(root / "summary.csv")
    assert (summary.attempts == summary.budget).all()
    assert (summary.unique_queries + summary.repeat_attempts == summary.attempts).all()
    assert set(config.rl.variants) <= set(summary.strategy)
    report = summarize_study(config.output_dir, tmp_path / "report")
    differences = pd.read_csv(report / "selection_intervals.csv")
    assert (
        (differences.strategy == "rl_repeat_penalty")
        & (differences.reference_strategy == "rl_repeat_control")
    ).any()
    evolution = pd.read_csv(report / "evolution_history.csv")
    assert (
        evolution.new_designs.sum()
        == evolution.cumulative_unique_designs.iloc[-1]
        == 100
    )
