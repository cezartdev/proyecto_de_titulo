"""Invariantes científicas del protocolo con calibración y adaptación por episodio."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.artefactos import dataset_identity
from truss_research.datos.particion_agrupada import (
    purged_generation_split,
    purged_group_folds,
)
from truss_research.datos.preparacion import prepare_dataset
from truss_research.datos.procesados import load_design_split
from truss_research.datos.schemas import RAW_ID_COLUMN, DatasetSpec, PreparationConfig
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.optimizacion.contratos import EpisodeObservation
from truss_research.optimizacion.entorno import area_reward, marginal_reward
from truss_research.optimizacion.politica import PolicyEncoder
from truss_research.optimizacion.seleccion_incierta import (
    AcquisitionContext,
    EnsembleStrategy,
)
from truss_research.surrogates.calibracion import SplitConformal
from truss_research.surrogates.configuracion import PredictorConfig
from truss_research.surrogates.experimentos import (
    load_predictor,
    read_run,
    train_predictor,
)
from truss_research.surrogates.modelos import preprocessing


@pytest.fixture
def calibrated_data(raw_table: pd.DataFrame, tmp_path: Path) -> Path:
    table = pd.concat([raw_table.iloc[[0]]] * 500, ignore_index=True)
    table[RAW_ID_COLUMN] = [f"{{{i // 5};{i % 5}}}" for i in table.index]
    table["Web Config"] = (table.index // 5) % 5
    table["L-R Height"] = 0.2 + table.index * 0.002
    table["Mass"] = 1000.0 + table.index * 10.0
    table["Average Util"] = 0.2 + table.index * 0.001
    table["Z Disp"] = 0.01 + table.index * 0.0001
    source = tmp_path / "source.xlsx"
    table.to_excel(source, sheet_name="TOTAL DATA", index=False)
    destination = tmp_path / "calibrated"
    prepare_dataset(
        PreparationConfig(
            source, destination, dataset=DatasetSpec(100, 5), calibration_fraction=0.2
        )
    )
    return destination


def test_calibration_is_disjoint_and_signed(calibrated_data: Path) -> None:
    split = load_design_split(calibrated_data)
    assert len(split.training) == 320
    assert len(split.calibration) == 80
    sets = [set(table.iter_id) for table in split.as_mapping().values()]
    assert sum(map(len, sets)) == len(set.union(*sets)) == 500
    before = dataset_identity(calibrated_data)
    path = calibrated_data / "calibration.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert dataset_identity(calibrated_data) != before


def test_conformal_quantile_and_insufficient_calibration() -> None:
    residuals = np.column_stack(
        (np.arange(20), np.arange(20) * 0.1, np.arange(20) * 0.001)
    )
    conformal = SplitConformal(residuals)
    np.testing.assert_allclose(conformal.widths(0.90), residuals[18])
    np.testing.assert_allclose(conformal.widths(0.95), residuals[19])
    with pytest.raises(ValueError, match="insuficiente"):
        SplitConformal(residuals[:10]).widths(0.95)


def test_plain_oof_uses_outer_folds_instead_of_final_search_folds(
    calibrated_data: Path, tmp_path: Path
) -> None:
    result = train_predictor(
        PredictorConfig(
            calibrated_data,
            tmp_path / "plain",
            model="dummy",
            folds=2,
            final_tuning_folds=3,
        )
    )
    assert len(read_run(result.output_dir)["fold_model_signatures"]) == 2
    assert pd.read_csv(result.output_dir / "training_oof.csv")["fold"].nunique() == 2


def test_group_purge_uses_all_memberships() -> None:
    table = pd.DataFrame(
        {"iter_id": [f"d{i}" for i in range(20)] + ["shared"], "web_config": [1] * 21}
    )
    membership = pd.DataFrame(
        [{"generation": i, "representative_iter_id": f"d{i}"} for i in range(20)]
        + [{"generation": i, "representative_iter_id": "shared"} for i in range(20)]
    )
    partition = purged_generation_split(table, membership, 42)
    assert partition.excluded.iter_id.tolist() == ["shared"]
    assert {name: len(value) for name, value in partition.tables.items()} == {
        "train": 16,
        "validation": 2,
        "test": 2,
    }
    for fitting, held in purged_group_folds(table, membership, (3, 42)):
        fit_groups = set(
            membership.loc[
                membership.representative_iter_id.isin(table.iloc[fitting].iter_id),
                "generation",
            ]
        )
        held_groups = set(
            membership.loc[
                membership.representative_iter_id.isin(table.iloc[held].iter_id),
                "generation",
            ]
        )
        assert fit_groups.isdisjoint(held_groups)
        assert "shared" not in set(table.iloc[np.r_[fitting, held]].iter_id)


@pytest.mark.parametrize("budget", [1, 2, 8, 24])
def test_marginal_reward_equals_auc_including_initial_query(budget: int) -> None:
    volumes = np.r_[
        0.0, np.cumsum(np.random.default_rng(42).choice([0.0, 0.1, 0.2], budget))
    ]
    trapezoidal = sum(
        area_reward(volumes[t - 1], volumes[t], (budget, 2.0))
        for t in range(1, budget + 1)
    )
    marginal = sum(
        marginal_reward(volumes[t - 1], volumes[t], (t, budget, 2.0))
        for t in range(1, budget + 1)
    )
    assert marginal == pytest.approx(trapezoidal)
    assert marginal_reward(0.5, 0.5, (1, budget, 2.0)) == 0
    assert marginal_reward(0.0, 0.5, (1, budget, 0.0)) == 0


def test_online_residual_uses_only_revealed_responses(calibrated_data: Path) -> None:
    table = load_design_split(calibrated_data).training.iloc[:6]
    actual = table.loc[:, list(OUTPUT_COLUMNS)].to_numpy()
    members = np.stack((actual * 0.9, actual * 0.9))
    observed = np.zeros_like(actual)
    observed[0] = actual[0]
    observation = EpisodeObservation(
        table.loc[:, list(INPUT_COLUMNS)].to_numpy(),
        members.mean(axis=0),
        observed,
        np.array([False, True, True, True, True, True]),
        4,
        1,
    )
    scales = tuple(np.ptp(observation.features, axis=0))
    context = AcquisitionContext(ObjectiveSpace.fit(actual), members, 1.0, scales, True)
    strategy = EnsembleStrategy(context)
    corrected = strategy.predictions_for(observation)
    np.testing.assert_allclose(corrected.mean(axis=0)[0], actual[0])
    np.testing.assert_allclose(context.member_predictions, members)
    another = EnsembleStrategy(replace(context, online_residual=False))
    np.testing.assert_array_equal(another.predictions_for(observation), members)
    assert strategy.choose(observation) in np.flatnonzero(observation.available)


def test_enhanced_ablation_cannot_observe_predictions(calibrated_data: Path) -> None:
    table = load_design_split(calibrated_data).training.iloc[:6]
    features = table.loc[:, list(INPUT_COLUMNS)]
    actual = table.loc[:, list(OUTPUT_COLUMNS)].to_numpy()
    observed = np.zeros_like(actual)
    observed[0] = actual[0]
    observation = EpisodeObservation(
        features.to_numpy(),
        actual * 0.9,
        observed,
        np.array([False, True, True, True, True, True]),
        4,
        1,
    )
    encoder = PolicyEncoder(
        preprocessing(PredictorConfig(model="mlp")).fit(features),
        ObjectiveSpace.fit(actual),
        False,
        True,
    )
    first = encoder.encode(observation)
    second = encoder.encode(replace(observation, predictions=actual * 20))
    np.testing.assert_array_equal(first.values, second.values)
    assert np.all(first.values[:, -1] == 0)
    np.testing.assert_array_equal(first.front, second.front)


def test_targetwise_nested_excludes_calibration_and_outer_fold(
    calibrated_data: Path, tmp_path: Path
) -> None:
    result = train_predictor(
        PredictorConfig(
            calibrated_data,
            tmp_path / "targetwise",
            model="targetwise",
            folds=2,
            nested_folds=2,
            final_tuning_folds=2,
            trials=1,
            candidate_models=("dummy", "ridge"),
        )
    )
    metadata = read_run(result.output_dir)
    model = load_predictor(result.output_dir)
    split = load_design_split(calibrated_data)
    assert set(metadata["fit_ids"]) == set(split.training.iter_id)
    assert set(metadata["fit_ids"]).isdisjoint(metadata["calibration_ids"])
    assert len(metadata["targetwise_selection"]) == 3
    assert metadata["trial_count"] >= 1
    for _, component in model.named_steps["regressor"].components_:
        assert (
            np.asarray(
                component.predict(split.validation.loc[:, list(INPUT_COLUMNS)])
            ).ndim
            == 1
        )
    from truss_research.datos.artefactos import read_artifact

    for fold in read_artifact(result.output_dir / "nested_folds.json")["folds"]:
        assert set(fold["fit_ids"]).isdisjoint(fold["held_ids"])
        assert set(fold["fit_ids"]).isdisjoint(metadata["calibration_ids"])
    assert (result.output_dir / "prediction_intervals.csv").is_file()


def test_guided_initialization_and_safety_fallback() -> None:
    from truss_research.optimizacion.contratos import EvolutionParameters
    from truss_research.optimizacion.estrategias import (
        GreedyStrategy,
        SafeGreedyStrategy,
    )
    from truss_research.optimizacion.evolucion import (
        EvolutionContext,
        EvolutionStrategy,
    )

    features = np.array(
        [[0, 10, height, 2, 3, 2, 3] for height in (0.2, 1.0, 2.0, 4.0)]
    )
    responses = np.array(
        [[80.0, 0.2, 0.04], [20.0, 0.8, 0.01], [60.0, 0.3, 0.03], [70.0, 0.25, 0.035]]
    )
    observed = np.zeros_like(responses)
    observed[0] = responses[0]
    observation = EpisodeObservation(
        features, responses.copy(), observed, np.array([False, True, True, True]), 4, 1
    )
    space = ObjectiveSpace.fit(responses)
    context = EvolutionContext(
        space, tuple(np.ptp(features, axis=0)), 42, EvolutionParameters(enabled=True)
    )
    assert EvolutionStrategy(context).choose(observation) == 3
    assert EvolutionStrategy(replace(context, assisted=True)).choose(observation) == 1
    safe = SafeGreedyStrategy(space, (0.0, 10.0, 10.0))
    assert safe.choose(observation) == GreedyStrategy(space).choose(observation)
    assert safe.diagnostics["safety_fallback"] is True


def test_pareto_encoder_and_network_are_permutation_invariant(
    calibrated_data: Path,
) -> None:
    import torch

    from truss_research.optimizacion.politica import CandidateQNetwork, tensor_states

    table = load_design_split(calibrated_data).training.iloc[:6]
    features = table.loc[:, list(INPUT_COLUMNS)]
    responses = table.loc[:, list(OUTPUT_COLUMNS)].to_numpy()
    observed = np.zeros_like(responses)
    observed[:2] = responses[:2]
    observation = EpisodeObservation(
        features.to_numpy(),
        responses,
        observed,
        np.array([False, False, True, True, True, True]),
        4,
        2,
    )
    encoder = PolicyEncoder(
        preprocessing(PredictorConfig(model="mlp")).fit(features),
        ObjectiveSpace.fit(responses),
        True,
    )
    order = np.array([5, 0, 2, 1, 3, 4])
    permuted = replace(
        observation,
        features=observation.features[order],
        predictions=responses[order],
        observed=observed[order],
        available=observation.available[order],
    )
    first, second = encoder.encode(observation), encoder.encode(permuted)
    network = CandidateQNetwork(first.values.shape[1], 8)
    with torch.no_grad():
        before = network(tensor_states([first])).numpy()[0]
        after = network(tensor_states([second])).numpy()[0]
    np.testing.assert_allclose(before[order], after, atol=1e-6)


def test_complete_v2_campaign_with_cpu_workers(
    calibrated_data: Path, tmp_path: Path
) -> None:
    from truss_research.__main__ import next_export
    from truss_research.datos.artefactos import read_artifact
    from truss_research.evaluacion.auditoria_estudio import AuditConfig, audit_study
    from truss_research.evaluacion.benchmark import run_benchmark
    from truss_research.evaluacion.informe_estudio import summarize_study
    from truss_research.evaluacion.presentacion import export_notebooks
    from truss_research.evaluacion.recursos import MeasurementParameters
    from truss_research.evaluacion.seleccion_adaptativa import (
        AdaptiveParameters,
        run_adaptive_selection,
    )
    from truss_research.evaluacion.sensibilidad import run_sensitivity
    from truss_research.optimizacion.contratos import (
        AnalysisParameters,
        EvolutionParameters,
    )
    from truss_research.optimizacion.ejecucion_cpu import ExecutionParameters
    from truss_research.optimizacion.estudio import run_study
    from truss_research.optimizacion.estudio_estado import (
        StudyConfig,
        artifact_signatures,
    )
    from truss_research.optimizacion.seleccion_incierta import AdaptiveSettings
    from truss_research.optimizacion.variantes_rl import RLParameters

    study = tmp_path / "study_v2"
    config = StudyConfig(
        study,
        data_dir=calibrated_data,
        models=("dummy", "ridge", "targetwise"),
        seeds=(42, 43),
        outer_folds=2,
        inner_folds=2,
        final_tuning_folds=2,
        trials=0,
        catalog_size=6,
        budgets=(3, 4),
        episodes_per_seed=1,
        rl_steps=8,
        rl_batch=2,
        rl_warmup=2,
        rl_hidden=8,
        validation_interval=4,
        background_size=3,
        permutation_rounds=1,
        execution=ExecutionParameters(cpu_workers=2),
        adaptive=AdaptiveSettings(online_residual=True),
        evolution=EvolutionParameters(
            enabled=True, population_size=2, initialization_ablation=True
        ),
        analysis=AnalysisParameters(
            catalog_sizes=(6, 8), bootstrap_repeats=10, generalization_seeds=(42,)
        ),
        rl=RLParameters(ablations=True),
        measurement=MeasurementParameters(
            warmup=0,
            inference_repeats=1,
            selection_repeats=1,
            batch_sizes=(1, 6),
            reuse_counts=(1,),
            cpu_repeats=1,
            cpu_min_seconds=0.001,
            cpu_min_ticks=1,
            cpu_max_wall_seconds=0.1,
            cpu_probe_seconds=0.01,
        ),
    )
    run_study(config, "validacion")
    run_study(config, "final", True)
    signatures = artifact_signatures(study)
    adaptive = run_adaptive_selection(
        study,
        next_export(study, "adaptativa"),
        AdaptiveParameters(online_residual=True, cpu_workers=2),
    )
    sequential = run_adaptive_selection(
        study,
        next_export(study, "adaptativa"),
        AdaptiveParameters(online_residual=True, cpu_workers=1),
    )
    first, second = (
        pd.read_csv(adaptive / "traces.csv"),
        pd.read_csv(sequential / "traces.csv"),
    )
    assert set(first.strategy) == {
        "ensemble_mean",
        "ensemble_ucb",
        "online_ensemble_mean",
        "online_ensemble_ucb",
    }
    pd.testing.assert_frame_equal(first, second)
    sensitivity = run_sensitivity(study, next_export(study, "sensibilidad"))
    assert (sensitivity / "generalization" / "metrics.csv").is_file()
    assert (sensitivity / "objective_sensitivity.csv").is_file()
    run_benchmark(study, next_export(study, "benchmark"))
    report = summarize_study(study, next_export(study, "informe"))
    assert (report / "interval_summary.csv").is_file()
    assert (report / "chronological_metrics.csv").is_file()
    audit = audit_study(
        AuditConfig(study, calibrated_data, next_export(study, "auditoria"))
    )
    assert read_artifact(audit / "audit.json")["resource_coverage_complete"] is True
    presentation = export_notebooks(study, next_export(study, "presentacion"))
    assert len(list(presentation.glob("*.html"))) == 4
    assert artifact_signatures(study) == signatures
