"""Exclusión de problemas, curvas anidadas y reanudación de análisis sintéticos."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.particion_equilibrada import (
    chronological_generation_split,
    matched_design_control,
    verify_group_exclusion,
)
from truss_research.datos.preparacion import prepare_dataset
from truss_research.datos.procesados import load_design_split, load_processed_table
from truss_research.datos.schemas import (
    RAW_ID_COLUMN,
    DatasetSpec,
    PreparationConfig,
    TableSplit,
)
from truss_research.datos.subconjuntos import nested_subsets
from truss_research.evaluacion.curvas_aprendizaje import curve_jobs, run_curve_job
from truss_research.evaluacion.generalizacion import run_generalization
from truss_research.evaluacion.protocolo_analisis import (
    GeneralizationParameters,
    LearningCurveParameters,
)
from truss_research.evaluacion.trabajos_analisis import execute_analysis_jobs
from truss_research.optimizacion.contratos import (
    AnalysisParameters,
    EvolutionParameters,
)
from truss_research.optimizacion.ejecucion_cpu import ExecutionParameters
from truss_research.optimizacion.estudio_estado import StudyConfig
from truss_research.optimizacion.protocolo import load_protocol, protocol_payload


@pytest.fixture
def analysis_data(raw_table: pd.DataFrame, tmp_path: Path) -> Path:
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


def small_config(data: Path, destination: Path) -> StudyConfig:
    return StudyConfig(
        destination,
        data_dir=data,
        models=("dummy",),
        seeds=(42,),
        trials=0,
        outer_folds=2,
        inner_folds=2,
        final_tuning_folds=2,
        catalog_size=8,
        budgets=(4,),
        episodes_per_seed=1,
        rl_steps=4,
        rl_batch=2,
        rl_warmup=2,
        rl_hidden=8,
        validation_interval=2,
        background_size=4,
        permutation_rounds=1,
        execution=ExecutionParameters(cpu_workers=1),
        evolution=EvolutionParameters(enabled=True, population_size=2),
        analysis=AnalysisParameters(
            catalog_sizes=(8, 16),
            bootstrap_repeats=10,
            generalization_seeds=(42,),
            generalization=GeneralizationParameters(
                min_training=40, min_calibration=19, min_evaluation=20
            ),
            learning_curves=LearningCurveParameters(
                enabled=True,
                models=("dummy",),
                train_sizes=(16, 32),
                subset_seeds=(42, 43),
            ),
        ),
    )


def test_profile_inheritance_and_unknown_fields(tmp_path: Path) -> None:
    config = load_protocol(Path("config/smoke.json"), (tmp_path, tmp_path))
    assert config.seeds == (42,) and config.analysis.generalization_seeds == ()
    assert config.rl.dqn.train_frequency == 4
    assert config.parameters.gpr_max_iter == 2000
    parent = tmp_path / "parent.json"
    parent.write_text('{"study":{"models":["dummy","ridge"]}}')
    child = tmp_path / "child.json"
    child.write_text('{"extends":"parent.json","study":{"models":["dummy"]}}')
    assert protocol_payload(child)["study"]["models"] == ["dummy"]
    parent.write_text('{"extends":"child.json"}')
    with pytest.raises(ValueError, match="circular"):
        protocol_payload(child)
    child.write_text('{"extends":"../missing.json"}')
    with pytest.raises(OSError):
        protocol_payload(child)


def test_balancing_never_uses_responses_and_control_matches(
    analysis_data: Path,
) -> None:
    table = load_processed_table(analysis_data / "full_clean.csv")
    membership = pd.read_csv(analysis_data / "design_membership.csv")
    settings = GeneralizationParameters()
    first = chronological_generation_split(table, membership, (42, settings))
    changed = table.astype({name: float for name in OUTPUT_COLUMNS})
    changed.loc[:, list(OUTPUT_COLUMNS)] = np.random.default_rng(9).normal(
        size=(len(table), 3)
    )
    second = chronological_generation_split(changed, membership, (42, settings))
    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    verify_group_exclusion(first.tables, membership)
    assert first.candidates.selected.sum() == 1
    split = TableSplit(
        first.tables["train"],
        first.tables["validation"],
        first.tables["test"],
        first.tables["calibration"],
    )
    control = matched_design_control(table, split, 42)
    pd.testing.assert_series_equal(
        split.training.web_config.value_counts().sort_index(),
        control.training.web_config.value_counts().sort_index(),
    )
    assert set(control.training.iter_id).isdisjoint(
        pd.concat([control.validation, control.testing, control.calibration]).iter_id
    )
    with pytest.raises(ValueError, match="mínimos"):
        chronological_generation_split(
            table, membership, (42, replace(settings, min_training=1000))
        )


def test_nested_subsets_are_shared_and_training_only(analysis_data: Path) -> None:
    split = load_design_split(analysis_data)
    subsets = nested_subsets(split.training, (16, 32, 128), 42)
    assert (
        set(subsets[16].iter_id) < set(subsets[32].iter_id) < set(subsets[128].iter_id)
    )
    for size, table in subsets.items():
        assert len(table) == size and table.web_config.nunique() == 5
        assert set(table.iter_id) <= set(split.training.iter_id)
    changed = split.training.copy()
    changed.loc[:, list(OUTPUT_COLUMNS)] = 999
    assert (
        nested_subsets(changed, (32,), 42)[32].iter_id.tolist()
        == subsets[32].iter_id.tolist()
    )


def test_grouped_selection_and_reuse(
    analysis_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = small_config(analysis_data, tmp_path / "study")
    destination = tmp_path / "generalization"
    run_generalization(config, destination)
    episodes = pd.read_csv(destination / "selection_episodes.csv")
    assert set(episodes.strategy) == {
        "random",
        "ranking",
        "greedy",
        "greedy_safe",
        "nsga2",
        "nsga2_surrogate",
    }
    assert episodes.queries.eq(episodes.budget).all()
    overlap = pd.read_csv(destination / "generation_overlap.csv")
    assert (
        overlap.loc[
            overlap.experiment.eq("chronological"), "shared_training_generations"
        ]
        .eq(0)
        .all()
    )
    intervals = pd.read_csv(destination / "intervals.csv")
    assert intervals.covered.notna().all()

    def forbidden_fit(
        self: Pipeline, features: pd.DataFrame, responses: pd.DataFrame
    ) -> None:
        raise AssertionError("No repetir ajustes completos")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    run_generalization(config, destination)
    assert pd.read_csv(destination / "worker_jobs.csv").reused.all()
    path = destination / "seed_42/chronological/metrics.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="cambió"):
        run_generalization(config, destination)


def test_curve_jobs_full_once_and_parallel_equivalence(
    analysis_data: Path, tmp_path: Path
) -> None:
    config = small_config(analysis_data, tmp_path / "study")
    total = len(load_design_split(analysis_data).training)
    sequential = curve_jobs(config, tmp_path / "sequential", total)
    assert len(sequential) == 5
    assert sum(job.subset_seed is None for job in sequential) == 1
    execute_analysis_jobs(sequential[:1], run_curve_job, tmp_path / "sequential")
    parallel_config = replace(config, execution=ExecutionParameters(cpu_workers=2))
    parallel = curve_jobs(parallel_config, tmp_path / "parallel", total)
    execute_analysis_jobs(parallel[:2], run_curve_job, tmp_path / "parallel")
    left = pd.read_csv(sequential[0].destination / "selection_episodes.csv")
    right = pd.read_csv(parallel[0].destination / "selection_episodes.csv")
    pd.testing.assert_frame_equal(
        left[["episode_id", "strategy", "normalized_auc"]],
        right[["episode_id", "strategy", "normalized_auc"]],
    )
    metadata = json.loads((sequential[0].destination / "run.json").read_text())
    assert (
        metadata["normalization_scale"]
        and metadata["reference_scope"] == "full_training_evaluation_only"
    )
    assert (
        json.loads((sequential[0].destination / "predictor/run.json").read_text())[
            "trial_count"
        ]
        == 0
    )


def test_grouped_search_includes_scalar_specialization(
    analysis_data: Path, tmp_path: Path
) -> None:
    config = replace(
        small_config(analysis_data, tmp_path / "study"),
        models=("dummy", "ridge", "targetwise"),
        trials=1,
    )
    destination = tmp_path / "generalization"
    run_generalization(config, destination)
    metadata = json.loads(
        (destination / "seed_42/chronological/targetwise/run.json").read_text()
    )
    assert len(metadata["targetwise_selection"]) == 3
    assert set(metadata["fit_ids"]).isdisjoint(metadata["calibration_ids"])
    scores = pd.read_csv(destination / "selection.csv")
    assert set(scores.loc[scores.experiment.eq("chronological"), "model"]) == {
        "dummy",
        "ridge",
        "targetwise",
    }


def test_evaluation_reference_never_changes_actions(
    analysis_data: Path, tmp_path: Path
) -> None:
    from truss_research.evaluacion.pareto import ObjectiveSpace
    from truss_research.optimizacion.catalogos import SelectionBank, episode_specs
    from truss_research.optimizacion.comparacion import evaluate_episode
    from truss_research.optimizacion.contratos import SelectionConfig
    from truss_research.optimizacion.estrategias import GreedyStrategy

    split = load_design_split(analysis_data)
    table = split.testing.set_index("iter_id", drop=False)
    space = ObjectiveSpace.fit(
        split.training.sample(16, random_state=42)
        .loc[:, list(OUTPUT_COLUMNS)]
        .to_numpy()
    )
    request = SelectionConfig(
        analysis_data,
        tmp_path,
        tmp_path,
        catalog_size=8,
        budgets=(4,),
        seeds=(42,),
        episodes_per_seed=1,
    )
    bank = SelectionBank(
        table,
        table.loc[:, list(OUTPUT_COLUMNS)] * 0.95,
        episode_specs(table, request),
        space,
        "dataset",
        "predictor",
    )
    other = replace(
        bank,
        evaluation_space=replace(
            space,
            reference_physical=(
                space.reference_physical[0] * 2,
                *space.reference_physical[1:],
            ),
        ),
    )
    _, first = evaluate_episode((bank, bank.episodes[0]), GreedyStrategy(bank.space))
    _, second = evaluate_episode(
        (other, other.episodes[0]), GreedyStrategy(other.space)
    )
    assert [row["iter_id"] for row in first] == [row["iter_id"] for row in second]


def test_interrupted_group_search_keeps_trial_budget(
    analysis_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    import truss_research.evaluacion.generalizacion as grouped
    from truss_research.surrogates.calibracion import SplitConformal
    from truss_research.surrogates.configuracion import PredictorConfig

    config = replace(
        small_config(analysis_data, tmp_path / "study"),
        models=("dummy", "ridge"),
        trials=1,
    )
    original = grouped.fit_retained

    def interrupted(
        request: PredictorConfig, split: TableSplit, tune: bool
    ) -> tuple[Pipeline, SplitConformal]:
        result = original(request, split, tune)
        if request.model == "ridge":
            raise RuntimeError("Interrupción sintética después de la búsqueda")
        return result

    destination = tmp_path / "chronological"
    monkeypatch.setattr(grouped, "fit_retained", interrupted)
    outcome = grouped.run_generalization_job(
        grouped.GeneralizationJob(config, destination / "seed_42", 42)
    )
    assert not outcome.success and "Interrupción" in outcome.error
    monkeypatch.setattr(grouped, "fit_retained", original)
    run_generalization(config, destination)
    assert (destination / "seed_42_failed_001").is_dir()
    with sqlite3.connect(
        destination / "seed_42/chronological/studies/ridge.sqlite"
    ) as connection:
        assert connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 1


def test_complete_synthetic_pipeline(analysis_data: Path, tmp_path: Path) -> None:
    from truss_research.evaluacion.auditoria_estudio import AuditConfig, audit_study
    from truss_research.evaluacion.benchmark import run_benchmark
    from truss_research.evaluacion.curvas_aprendizaje import (
        latest_learning_curves,
        run_learning_curves,
    )
    from truss_research.evaluacion.informe_estudio import (
        latest_study_report,
        summarize_study,
    )
    from truss_research.evaluacion.presentacion import export_notebooks
    from truss_research.evaluacion.recursos import MeasurementParameters
    from truss_research.evaluacion.registro import results_root
    from truss_research.evaluacion.seleccion_adaptativa import run_adaptive_selection
    from truss_research.evaluacion.sensibilidad import run_sensitivity
    from truss_research.optimizacion.estudio import run_study
    from truss_research.optimizacion.estudio_estado import artifact_signatures
    from truss_research.optimizacion.seleccion_incierta import AdaptiveSettings

    study = tmp_path / "study"
    config = replace(
        small_config(analysis_data, study),
        calibration_required=True,
        adaptive=AdaptiveSettings(online_residual=True),
        measurement=MeasurementParameters(
            warmup=0,
            inference_repeats=1,
            selection_repeats=1,
            batch_sizes=(1, 8),
            cpu_repeats=1,
            cpu_min_seconds=0.01,
            cpu_min_ticks=1,
            cpu_max_wall_seconds=0.5,
            cpu_probe_seconds=0.05,
        ),
    )
    run_study(config, "todo", False)
    before = artifact_signatures(study)
    root = results_root(study)
    run_adaptive_selection(study, root / "adaptativa_001")
    run_sensitivity(study, root / "sensibilidad_001")
    run_learning_curves(study, root / "curvas_001")
    assert latest_learning_curves(study) == root / "curvas_001"
    assert artifact_signatures(study) == before
    run_benchmark(study, root / "benchmark_001")
    report = summarize_study(study, root / "informe_001")
    assert latest_study_report(study) == report
    assert (report / "learning_scores.csv").is_file()
    costs = pd.read_csv(report / "analysis_costs.csv")
    assert set(costs.analysis) == {"generalization_and_controls", "learning_curves"}
    audit_study(AuditConfig(study, analysis_data, root / "auditoria_001"))
    export_notebooks(study, root / "presentacion_001")
    assert len(list((root / "presentacion_001").glob("*.html"))) == 4
