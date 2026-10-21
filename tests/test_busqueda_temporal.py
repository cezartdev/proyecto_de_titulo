"""Contratos temporales y de consulta con tablas pequeñas y evaluadores instrumentados."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.preparacion import prepare_dataset
from truss_research.datos.procesados import load_design_split
from truss_research.datos.schemas import DatasetSpec, PreparationConfig, SplitConfig
from truss_research.datos.temporal import first_appearance_folds, temporal_designs
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.optimizacion.busqueda import EvolutionSearch, SearchContext
from truss_research.optimizacion.contratos import EvolutionParameters
from truss_research.optimizacion.entrenamiento_propuestas import (
    ProposalEncoder,
    predicted_auc,
)
from truss_research.optimizacion.evaluadores import Evaluation, HistoricalEvaluator
from truss_research.optimizacion.propuestas import ProposalParameters, TrainingSupport


class CountingWorld:
    def __init__(self, offset: float = 0.0) -> None:
        self.calls = 0
        self.offset = offset

    def evaluate(self, features: np.ndarray) -> Evaluation:
        self.calls += len(features)
        responses = np.column_stack(
            (
                2000 - 100 * features[:, 2] + self.offset,
                np.full(len(features), 0.5),
                np.full(len(features), 0.01),
            )
        )
        return Evaluation(
            features.copy(), responses, np.zeros(len(features)), "synthetic_exact"
        )


@pytest.fixture
def temporal_table() -> pd.DataFrame:
    rows = []
    for generation in range(20):
        for solution in range(5):
            height = 0.5 + generation * 0.04 + solution * 0.003
            if generation >= 5 and solution == 0:
                height = 0.5
            rows.append(
                dict(
                    zip(
                        [
                            "iter_id",
                            "generation",
                            "solution",
                            *INPUT_COLUMNS,
                            *OUTPUT_COLUMNS,
                        ],
                        [
                            f"{{{generation};{solution}}}",
                            generation,
                            solution,
                            0,
                            10,
                            height,
                            1.0,
                            2.0,
                            2,
                            3,
                            2000 - 100 * height,
                            0.5,
                            0.01,
                        ],
                        strict=True,
                    )
                )
            )
    return pd.DataFrame(rows)


@pytest.fixture
def context() -> SearchContext:
    features = np.array(
        [[0, 10, 0.5 + index * 0.03, 1, 2, 2, 3] for index in range(20)], dtype=float
    )
    world = CountingWorld()
    responses = world.evaluate(features).responses
    world.calls = 0
    space = ObjectiveSpace.fit(responses)
    return SearchContext(
        features,
        responses,
        (world,),
        TrainingSupport.fit(features, 0.95),
        space,
        space.hypervolume(responses),
        ProposalParameters(pool_size=4, max_generated=128),
        EvolutionParameters(population_size=2),
        8,
    )


def test_preparation_retains_events_and_assigns_first_occurrence(
    temporal_table: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"synthetic-source-not-an-excel")
    monkeypatch.setattr(
        "truss_research.datos.preparacion.load_raw_table", lambda *args: temporal_table
    )
    monkeypatch.setattr(
        "truss_research.datos.preparacion.normalize_columns", lambda table: table
    )
    config = PreparationConfig(
        source,
        tmp_path / "prepared",
        dataset=DatasetSpec(20, 5),
        split=SplitConfig(protocol="first_appearance"),
        calibration_fraction=0.2,
    )
    prepare_dataset(config)
    split = load_design_split(config.destination)
    assert split.training.generation.max() < split.calibration.generation.min()
    assert split.calibration.generation.max() < split.validation.generation.min()
    assert split.validation.generation.max() < split.testing.generation.min()
    events = pd.read_csv(config.destination / "population_events.csv")
    assert len(events) == 100
    assert events.design_id.nunique() == sum(
        len(table) for table in split.as_mapping().values()
    )
    assert split.training.loc[
        split.training.lr_height_m.eq(0.5), "generation"
    ].tolist() == [0]


def test_first_occurrence_is_invariant_to_file_order(
    temporal_table: pd.DataFrame,
) -> None:
    left = temporal_designs(temporal_table)
    right = temporal_designs(temporal_table.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(left, right)
    for fitting, held in first_appearance_folds(left, 3):
        assert left.iloc[fitting].generation.max() < left.iloc[held].generation.min()


def test_future_events_do_not_change_past_designs(temporal_table: pd.DataFrame) -> None:
    cutoff = 10
    past = temporal_designs(temporal_table.loc[temporal_table.generation.lt(cutoff)])
    full = temporal_designs(temporal_table)
    pd.testing.assert_frame_equal(
        past, full.loc[full.generation.lt(cutoff)].reset_index(drop=True)
    )


def test_prepare_and_observation_do_not_query_private_world(
    context: SearchContext,
) -> None:
    search = EvolutionSearch(context, (6, 42))
    encoder = ProposalEncoder.fit(context)
    assert search.prepare()
    first = encoder.encode(search)
    context.worlds[0].offset = 999
    np.testing.assert_array_equal(first.values, encoder.encode(search).values)
    assert context.worlds[0].calls == 0
    search.choose(0)
    assert context.worlds[0].calls == 1
    assert search.evaluations == 3


def test_snapshot_preserves_pending_population_and_randomness(
    context: SearchContext,
) -> None:
    search = EvolutionSearch(context, (6, 42))
    assert search.prepare()
    restored = EvolutionSearch.restore(context, search.snapshot())
    assert restored.choose(0) == search.choose(0)
    assert restored.prepare() == search.prepare()
    np.testing.assert_array_equal(restored.batch.features, search.batch.features)
    np.testing.assert_array_equal(
        restored.population_features, search.population_features
    )


def test_budget_includes_initialization_and_auc_padding(context: SearchContext) -> None:
    search = EvolutionSearch(context, (5, 42))
    rewards = []
    while search.prepare():
        rewards.append(search.choose(0))
    assert search.evaluations <= 5
    assert len(search.population_features) == 2
    assert context.worlds[0].calls == search.evaluations - 2
    assert np.isfinite(predicted_auc(search))
    assert len(search.volumes) == search.evaluations + 1


def test_historical_response_belongs_to_projected_design(
    context: SearchContext,
) -> None:
    world = HistoricalEvaluator(
        context.training, context.initial_responses, context.support.scales
    )
    proposed = context.training[[0]].copy()
    proposed[0, 2] += 0.001
    result = world.evaluate(proposed)
    np.testing.assert_array_equal(result.features, context.training[[0]])
    np.testing.assert_array_equal(result.responses, context.initial_responses[[0]])
    assert result.distances[0] > 0


def test_generation_budget_stops_without_hidden_queries(context: SearchContext) -> None:
    limited = replace(context, parameters=replace(context.parameters, max_generated=1))
    search = EvolutionSearch(limited, (5, 42))
    if search.prepare():
        search.choose(0)
    assert not search.prepare()
    assert search.generated <= 1


def test_unsupported_historical_query_consumes_budget_and_continues(
    context: SearchContext,
) -> None:
    features = context.training.copy()
    features[:, 0] = 1
    historical = HistoricalEvaluator(
        features, context.initial_responses, context.support.scales
    )
    search = EvolutionSearch(replace(context, worlds=(historical,)), (5, 42))
    assert search.prepare()
    previous = search.volumes[-1]
    assert search.choose(0) == 0
    assert search.evaluations == 3
    assert search.volumes[-1] == previous
    assert search.last_evaluation is None
    assert search.reason != "unsupported_topology"
    assert search.prepare()


@pytest.fixture
def prepared_temporal(
    temporal_table: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    extra = temporal_table.copy()
    extra.solution += 5
    extra["iter_id"] = [
        f"{{{g};{s}}}" for g, s in zip(extra.generation, extra.solution, strict=True)
    ]
    extra.lr_height_m += 0.001
    extra.mass_kg = 2000 - 100 * extra.lr_height_m
    table = pd.concat([temporal_table, extra], ignore_index=True)
    source = tmp_path / "fixture.xlsx"
    source.write_bytes(b"synthetic-fixture")
    monkeypatch.setattr(
        "truss_research.datos.preparacion.load_raw_table", lambda *args: table
    )
    monkeypatch.setattr(
        "truss_research.datos.preparacion.normalize_columns", lambda value: value
    )
    destination = tmp_path / "temporal"
    prepare_dataset(
        PreparationConfig(
            source,
            destination,
            dataset=DatasetSpec(20, 10),
            split=SplitConfig(protocol="first_appearance"),
            calibration_fraction=0.2,
        )
    )
    return destination


def test_small_ridge_temporal_oof_and_parallel_preparation(
    prepared_temporal: Path, tmp_path: Path
) -> None:
    from truss_research.surrogates.configuracion import PredictorConfig
    from truss_research.surrogates.experimentos import train_predictor
    from truss_research.surrogates.pliegues import eligible_indices
    from truss_research.surrogates.preparacion import (
        assemble_preparation,
        prepare_outer_fold,
        prepare_search,
    )

    config = PredictorConfig(
        prepared_temporal,
        tmp_path / "ridge",
        model="ridge",
        folds=2,
        nested_folds=2,
        final_tuning_folds=2,
        trials=0,
    )
    split = load_design_split(prepared_temporal)
    result = train_predictor(config)
    oof = pd.read_csv(result.output_dir / "training_oof.csv")
    expected = split.training.iloc[eligible_indices(config, split.training)].iter_id
    assert set(oof.iter_id) == set(expected)
    assert len(oof) < len(split.training)
    prepare_search(config, tmp_path / "search")
    folds = (tmp_path / "outer0", tmp_path / "outer1")
    for number, directory in enumerate(folds):
        prepare_outer_fold(config, directory, number)
    assemble_preparation(config, (tmp_path / "search", folds), tmp_path / "assembled")
    reused = train_predictor(
        replace(
            config,
            output_dir=tmp_path / "prepared_ridge",
            preparation_dir=tmp_path / "assembled",
        )
    )
    np.testing.assert_allclose(
        result.model.predict(split.validation[list(INPUT_COLUMNS)]),
        reused.model.predict(split.validation[list(INPUT_COLUMNS)]),
    )
    from truss_research.evaluacion.busqueda_evolutiva import (
        build_context,
        close_predictors,
        predictor_directories,
    )
    from truss_research.optimizacion.estudio_estado import (
        StudyConfig,
        StudySession,
        TaskRecord,
        artifact_signatures,
    )

    settings = StudyConfig(
        tmp_path / "registered",
        data_dir=prepared_temporal,
        models=("ridge",),
        seeds=(42,),
        experiment="evolution",
        split_protocol="first_appearance",
        budgets=(7,),
        evolution=EvolutionParameters(population_size=4),
    )
    settings.output_dir.mkdir()
    session = StudySession(settings, "fixture", "fixture", {})
    record = TaskRecord(
        "predictor/ridge/42",
        status="complete",
        directory=str(result.output_dir),
        signatures=artifact_signatures(result.output_dir),
    )
    session.tasks[record.key] = record
    session.save()
    assert predictor_directories(settings) == {"ridge": result.output_dir}
    context = build_context(settings, (result.output_dir, reused.output_dir))
    assert len(context.worlds) == 1 and context.validation_world is not None
    close_predictors(settings)
    close_predictors(settings)
    assert (
        settings.output_dir / "evolution/test_predictors/ridge_42/complete.json"
    ).is_file()


def test_predictive_report_from_small_signed_models(
    prepared_temporal: Path, tmp_path: Path
) -> None:
    from truss_research.evaluacion.benchmark_multiobjetivo import (
        evaluate_predictor_utility,
    )
    from truss_research.evaluacion.busqueda_evolutiva import (
        choose_predictors,
        close_predictors,
        family_models,
    )
    from truss_research.evaluacion.comparacion_predictiva import (
        export_predictive_comparison,
    )
    from truss_research.optimizacion.estudio_estado import (
        StudyConfig,
        StudySession,
        TaskRecord,
        artifact_signatures,
    )
    from truss_research.surrogates.configuracion import PredictorConfig
    from truss_research.surrogates.experimentos import train_predictor

    config = StudyConfig(
        tmp_path / "signed",
        data_dir=prepared_temporal,
        models=("dummy", "ridge"),
        seeds=(42, 43),
        outer_folds=2,
        inner_folds=2,
        analysis=replace(StudyConfig(tmp_path).analysis, bootstrap_repeats=20),
    )
    config.output_dir.mkdir()
    session = StudySession(config, "fixture", "fixture", {})
    for family in config.models:
        for seed in config.seeds:
            path = tmp_path / f"{family}_{seed}"
            train_predictor(
                PredictorConfig(
                    prepared_temporal,
                    path,
                    model=family,
                    seed=seed,
                    folds=2,
                    nested_folds=2,
                    trials=0,
                )
            )
            key = f"predictor/{family}/{seed}"
            session.tasks[key] = TaskRecord(
                key,
                "complete",
                directory=str(path),
                signatures=artifact_signatures(path),
            )
    session.save()
    assert len(family_models(config, "ridge")) == 2
    development = evaluate_predictor_utility(config, "validation")
    paths = {name: tmp_path / f"{name}_42" for name in config.models}
    selected = choose_predictors(config, paths)
    assert set(development.loc[development.role.eq("regressor"), "model"]) == set(
        config.models
    )
    close_predictors(config)
    evaluate_predictor_utility(config, "test")
    assert selected == choose_predictors(config, paths)
    before = artifact_signatures(config.output_dir)
    destination = tmp_path / "comparison"
    destination.mkdir()
    export_predictive_comparison(config.output_dir, destination)
    table = pd.read_csv(destination / "predictive_paired_errors.csv")
    assert set(table.split) == {"validation", "test"}
    assert (
        table.loc[table.split.eq("test"), "count"]
        .eq(len(load_design_split(prepared_temporal).testing))
        .all()
    )
    assert (destination / "physical_provenance.csv").is_file()
    assert (destination / "predictive_interval_coverage.csv").is_file()
    assert before == artifact_signatures(config.output_dir)


def test_interpolation_uses_separate_small_dataset(
    prepared_temporal: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.datos.schemas import COLUMN_MAPPING, RAW_ID_COLUMN
    from truss_research.evaluacion import interpolacion
    from truss_research.optimizacion.estudio_estado import StudyConfig

    events = pd.read_csv(prepared_temporal / "population_events.csv")
    raw = events.rename(
        columns={
            **{value: key for key, value in COLUMN_MAPPING.items()},
            "iter_id": RAW_ID_COLUMN,
        }
    )
    source = tmp_path / "interpolation.xlsx"
    raw[[RAW_ID_COLUMN, *COLUMN_MAPPING]].to_excel(
        source, sheet_name="TOTAL DATA", index=False
    )
    monkeypatch.setattr(
        interpolacion,
        "prepare_dataset",
        lambda settings: prepare_dataset(
            replace(settings, dataset=DatasetSpec(20, 10))
        ),
    )
    config = StudyConfig(
        tmp_path / "interpolation",
        models=("dummy", "ridge"),
        seeds=(42, 43),
        trials=0,
        outer_folds=2,
        inner_folds=2,
        final_tuning_folds=2,
        calibration_fraction=0.2,
        analysis=replace(StudyConfig(tmp_path).analysis, bootstrap_repeats=20),
    )
    before = (prepared_temporal / "manifest.json").read_bytes()
    interpolacion.run_interpolation(config, source)
    assert interpolacion.verify_interpolation(config, source) == config.output_dir
    assert (config.output_dir / "freeze.json").is_file()
    assert (config.output_dir / "complete.json").is_file()
    metrics = pd.read_csv(config.output_dir / "report/predictive_seed_metrics.csv")
    assert set(metrics.model) == {"dummy", "ridge"}
    assert set(metrics.split) == {"validation", "test"}
    assert (config.output_dir / "report/predictor_utility_summary.csv").is_file()
    assert before == (prepared_temporal / "manifest.json").read_bytes()


def test_development_freeze_and_exports_with_synthetic_worlds(
    context: SearchContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import asdict

    from truss_research.evaluacion import busqueda_evolutiva as campaign
    from truss_research.evaluacion.informe_busqueda import export_search
    from truss_research.optimizacion.estudio_estado import (
        STUDY_SCHEMA_VERSION,
        StudyConfig,
    )
    from truss_research.surrogates.experimentos import write_json

    class FormulaPredictor:
        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return CountingWorld().evaluate(features.to_numpy()).responses

    config = StudyConfig(
        tmp_path / "study",
        data_dir=tmp_path / "data",
        models=("dummy", "ridge"),
        seeds=(42,),
        experiment="evolution",
        split_protocol="first_appearance",
        budgets=(7,),
        rl_steps=2,
        rl_batch=2,
        rl_warmup=100,
        rl_hidden=8,
        validation_interval=2,
        evolution=EvolutionParameters(population_size=4),
        proposals=replace(context.parameters, evaluation_episodes=1),
        analysis=replace(StudyConfig(tmp_path).analysis, bootstrap_repeats=2),
    )
    config.data_dir.mkdir()
    write_json(
        config.data_dir / "manifest.json",
        {"diagnostics": {"train": {"rows": len(context.training), "generations": [0]}}},
    )
    paths = {name: tmp_path / name for name in config.models}
    for path in paths.values():
        path.mkdir()
        (path / "model.joblib").write_bytes(b"synthetic-placeholder")
        write_json(path / "run.json", {"config": {"model": path.name}})
    table = pd.DataFrame(context.training, columns=INPUT_COLUMNS)
    for number, name in enumerate(OUTPUT_COLUMNS):
        table[name] = context.initial_responses[:, number]

    def fake_predictors(settings: StudyConfig, phase: str, resume: bool) -> Path:
        assert phase == "predictores"
        settings.output_dir.mkdir(exist_ok=True)
        write_json(
            settings.output_dir / "study.json",
            {
                "schema_version": STUDY_SCHEMA_VERSION,
                "config": asdict(settings),
                "tasks": [],
            },
        )
        return settings.output_dir

    monkeypatch.setattr(
        "truss_research.optimizacion.estudio.run_study", fake_predictors
    )
    monkeypatch.setattr(
        campaign,
        "read_manifest",
        lambda path: {
            "partition_policy": "unique_design_first_appearance_chronological"
        },
    )
    monkeypatch.setattr(campaign, "data_signature", lambda path: "fixture")
    monkeypatch.setattr(campaign, "predictor_directories", lambda settings: paths)
    monkeypatch.setattr(
        campaign, "choose_predictors", lambda *args: ("dummy", "ridge", "ridge")
    )
    monkeypatch.setattr(
        campaign,
        "build_context",
        lambda *args: replace(
            context,
            evolution=config.evolution,
            archive_size=7,
            validation_world=CountingWorld(),
            cheap=CountingWorld(),
            search_parameters=config.search,
        ),
    )
    monkeypatch.setattr(campaign, "load_predictor", lambda path: FormulaPredictor())
    monkeypatch.setattr(campaign, "family_models", lambda *args: (FormulaPredictor(),))

    def fake_utility(settings: StudyConfig, split_name: str) -> pd.DataFrame:
        from truss_research.optimizacion.estudio_estado import artifact_signatures

        destination = settings.output_dir / "evolution/predictor_utility" / split_name
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / "complete.json").exists():
            return pd.read_csv(destination / "summary.csv")
        summary = pd.DataFrame(
            {
                "model": ["dummy", "ridge"],
                "role": "regressor",
                "auc": [1.0, 1.0],
                "budget": 7,
                "split": split_name,
            }
        )
        summary.to_csv(destination / "summary.csv", index=False)
        summary.rename(columns={"auc": "mean_auc"}).to_csv(
            destination / "family_summary.csv", index=False
        )
        pd.DataFrame(
            columns=["model", "reference_model", "budget", "mean_auc_difference"]
        ).to_csv(destination / "paired_intervals.csv", index=False)
        write_json(
            destination / "complete.json", {"files": artifact_signatures(destination)}
        )
        return summary

    monkeypatch.setattr(campaign, "evaluate_predictor_utility", fake_utility)

    def fake_predictive_export(study: Path, destination: Path) -> None:
        pd.DataFrame({"model": ["dummy"], "mae": [0.0]}).to_csv(
            destination / "predictive_seed_metrics.csv", index=False
        )
        pd.DataFrame(
            {
                "split": ["validation", "test"],
                "model": ["dummy", "dummy"],
                "mean": [0.0, 0.0],
            }
        ).to_csv(destination / "predictive_model_summary.csv", index=False)

    monkeypatch.setattr(
        "truss_research.evaluacion.informe_busqueda.export_predictive_comparison",
        fake_predictive_export,
    )
    monkeypatch.setattr(
        campaign,
        "load_design_split",
        lambda path: type("Split", (), {"validation": table, "testing": table})(),
    )
    original_episode = campaign.cached_episode
    calls = []

    def interrupted(directory: Path, job: campaign.EpisodeJob, receipt: dict[str, str]):
        calls.append(job.strategy)
        if len(calls) == 2:
            raise RuntimeError("interrupcion sintetica")
        return original_episode(directory, job, receipt)

    monkeypatch.setattr(campaign, "cached_episode", interrupted)
    with pytest.raises(RuntimeError, match="interrupcion"):
        campaign.run_evolution_study(config, "validacion", False)
    completed = list(
        (config.output_dir / "evolution/validation").glob("episode_*.json")
    )
    assert len(completed) == 1
    first_episode = completed[0].read_bytes()
    monkeypatch.setattr(campaign, "cached_episode", original_episode)
    campaign.run_evolution_study(config, "validacion", True)
    assert completed[0].read_bytes() == first_episode
    assert not (config.output_dir / "freeze.json").exists()
    campaign.run_evolution_study(config, "final", True)
    frozen = (config.output_dir / "freeze.json").read_bytes()
    summaries = pd.read_csv(config.output_dir / "evolution/test/summary.csv")
    assert set(summaries.strategy) == set(config.search.strategies)
    assert set(summaries.experiment) == {"A", "B"}
    assert summaries.reference_hypervolume.nunique() == 1
    assert set(summaries.target_reference_scope) == {"training_only_frozen_before_test"}
    export_search(config.output_dir, tmp_path / "report", "notebooks")
    assert len(pd.read_csv(tmp_path / "report/policy_costs.csv")) == 4
    assert (tmp_path / "report/target_attainment.csv").is_file()
    assert len(list((tmp_path / "report").glob("*.html"))) == 4
    assert frozen == (config.output_dir / "freeze.json").read_bytes()
    campaign.run_evolution_study(config, "final", True)


def test_mlp_temporal_reserve_fits_transformer_only_on_past(
    prepared_temporal: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from truss_research.surrogates.configuracion import PredictorConfig
    from truss_research.surrogates.mlp import TorchRegressor
    from truss_research.surrogates.mlp_temporal import TemporalMLP

    observed = []

    def capture(
        self: TorchRegressor,
        fitting: tuple[np.ndarray, np.ndarray],
        stopping: tuple[np.ndarray, np.ndarray],
    ) -> TorchRegressor:
        observed.append((fitting, stopping))
        self.early_stopping_ = [{}]
        return self

    monkeypatch.setattr(TorchRegressor, "fit_reserved", capture)
    table = load_design_split(prepared_temporal).training
    regressor = TemporalMLP(PredictorConfig(prepared_temporal, tmp_path, model="mlp"))
    regressor.fit(table[list(INPUT_COLUMNS)], table[list(OUTPUT_COLUMNS)].to_numpy())
    record = regressor.early_stopping_[0]
    assert record["training_max_generation"] < record["stopping_min_generation"]
    np.testing.assert_allclose(observed[0][0][0][:, 1:].mean(axis=0), 0, atol=1e-10)


def test_contextual_bandit_ignores_future_states_and_bootstrap(
    context: SearchContext,
) -> None:
    import torch

    from truss_research.optimizacion.bandido_contextual import ContextualBanditLearner
    from truss_research.optimizacion.entrenamiento import (
        Transition,
        transition_batch,
        update_learner,
    )
    from truss_research.optimizacion.politica import CandidateQNetwork, DQNParameters

    search = EvolutionSearch(context, (6, 42))
    assert search.prepare()
    state = ProposalEncoder.fit(context).encode(search)
    batch = transition_batch(Transition(state, 0, 0.25, state, False))
    learner = ContextualBanditLearner(
        CandidateQNetwork(state.values.shape[1], 8), DQNParameters()
    )
    expected = learner.loss(batch)["loss"].detach()
    changed = batch.clone()
    changed["next", "state", "values"].fill_(float("nan"))
    changed["next", "done"].fill_(True)
    changed["next", "terminated"].fill_(True)
    torch.testing.assert_close(learner.loss(changed)["loss"], expected)
    original = [value.detach().clone() for value in learner.online.parameters()]
    assert np.isfinite(update_learner(learner, changed))
    assert any(
        not torch.equal(left, right)
        for left, right in zip(original, learner.online.parameters(), strict=True)
    )


def test_differential_generation_preserves_mixed_domain_and_distinct_donors(
    context: SearchContext,
) -> None:
    from truss_research.cerchas.schemas import (
        DESIGN_BOUNDS,
        INTEGER_COLUMNS,
        WEB_CONFIG_VALUES,
    )
    from truss_research.optimizacion.estrategias_busqueda import SearchParameters
    from truss_research.optimizacion.generacion import DifferentialGenerator

    expanded = replace(context, evolution=replace(context.evolution, population_size=4))
    search = EvolutionSearch(expanded, (8, 42))
    generated = DifferentialGenerator(SearchParameters()).generate(
        search._algorithm, (32, np.random.default_rng(8))
    )
    assert all(len(set(indices)) == 4 for indices in generated.lineage)
    assert set(generated.features[:, 0]) <= set(WEB_CONFIG_VALUES)
    for index, name in enumerate(INPUT_COLUMNS):
        values = generated.features[:, index]
        assert (values >= DESIGN_BOUNDS[name][0]).all()
        assert (values <= DESIGN_BOUNDS[name][1]).all()
        if name in INTEGER_COLUMNS:
            np.testing.assert_array_equal(values, np.rint(values))


def test_differential_search_is_private_budgeted_and_resumable(
    context: SearchContext,
) -> None:
    expanded = replace(
        context,
        evolution=replace(context.evolution, population_size=4),
        generator="differential",
    )
    search = EvolutionSearch(expanded, (8, 42))
    assert search.prepare()
    assert expanded.worlds[0].calls == 0
    restored = EvolutionSearch.restore(expanded, search.snapshot())
    assert search.choose(0) == restored.choose(0)
    assert search.prepare() == restored.prepare()
    np.testing.assert_array_equal(search.batch.features, restored.batch.features)
    assert search.batch.lineage == restored.batch.lineage
    while search.prepare():
        search.choose(0)
    assert search.evaluations <= 8
    assert search.generated <= expanded.parameters.max_generated


@pytest.mark.parametrize("objective", ["immediate", "return"])
def test_learned_selector_resumes_identical_weights_after_updates(
    context: SearchContext,
    tmp_path: Path,
    objective: str,
) -> None:
    import torch

    from truss_research.optimizacion import entrenamiento_propuestas as training
    from truss_research.optimizacion.politica import DQNParameters

    settings = training.ProposalTrainingConfig(
        42,
        (6,),
        DQNParameters(
            transitions=8, hidden_width=8, batch_size=2, warmup=2, validation_interval=4
        ),
        f"synthetic-{objective}",
        validation_episodes=1,
        objective=objective,
    )
    original_save = training.save_proposal_checkpoint

    def interrupt(
        destination: Path,
        checkpoint: training.ProposalCheckpoint,
        config: training.ProposalTrainingConfig,
    ) -> None:
        original_save(destination, checkpoint, config)
        if checkpoint.step == 4:
            raise RuntimeError("interrupcion controlada")

    with pytest.MonkeyPatch.context() as scoped:
        scoped.setattr(training, "save_proposal_checkpoint", interrupt)
        with pytest.raises(RuntimeError, match="controlada"):
            training.train_proposal_policy(context, settings, tmp_path / "resumed")
    resumed, _ = training.train_proposal_policy(context, settings, tmp_path / "resumed")
    uninterrupted, _ = training.train_proposal_policy(
        context, settings, tmp_path / "full"
    )
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(
            value, uninterrupted.state_dict()[name], rtol=0, atol=0
        )
    metadata = training.read_run(tmp_path / "resumed")
    assert metadata["optimizer_updates"] > 0
    assert metadata["config"]["objective"] == objective
    assert metadata["training_surrogate_evaluations"] >= 8


def test_primary_contrasts_pair_learning_seeds_and_scenarios(tmp_path: Path) -> None:
    from truss_research.evaluacion.contrastes_busqueda import paired_search_intervals
    from truss_research.optimizacion.estrategias_busqueda import SearchParameters
    from truss_research.optimizacion.estudio_estado import StudyConfig

    table = pd.DataFrame(
        [
            {
                "experiment": "B",
                "budget": 8,
                "strategy": strategy,
                "policy_seed": seed,
                "scenario": scenario,
                "auc": seed * 0.001 + scenario + gain,
            }
            for strategy, gain in (("rl", 0.2), ("contextual_bandit", 0.1))
            for seed in (42, 43)
            for scenario in (0, 1)
        ]
    )
    config = StudyConfig(
        tmp_path, search=SearchParameters(strategies=("rl", "contextual_bandit"))
    )
    result = paired_search_intervals(table, config)
    assert len(result) == 1
    assert result.iloc[0].mean_auc_difference == pytest.approx(0.1)
    assert result.iloc[0].lower == pytest.approx(0.1)
    assert result.iloc[0].training_seed_pairs == 2
