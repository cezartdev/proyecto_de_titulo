"""Adquisición pública, extensión del estudio y conservación de ajustes firmados."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pytest
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion import benchmark, seleccion_adaptativa
from truss_research.evaluacion.informe_estudio import (
    latest_study_report,
    summarize_study,
)
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.evaluacion.recursos import MeasurementParameters
from truss_research.evaluacion.seleccion_adaptativa import (
    AdaptiveParameters,
    adaptive_inputs,
    latest_adaptive_selection,
    load_adaptive_parameters,
    run_adaptive_selection,
    verify_saved_predictions,
)
from truss_research.optimizacion.contratos import AnalysisParameters, EpisodeSpec
from truss_research.optimizacion.entorno import EpisodeData, create_episode
from truss_research.optimizacion.estrategias import GreedyStrategy
from truss_research.optimizacion.estudio import run_study
from truss_research.optimizacion.estudio_estado import StudyConfig, artifact_signatures
from truss_research.optimizacion.seleccion_incierta import (
    AcquisitionContext,
    EnsembleStrategy,
    ensemble_disagreement,
)
from truss_research.optimizacion.variantes_rl import RLParameters
from truss_research.surrogates.configuracion import FloatArray
from truss_research.surrogates.experimentos import read_run, write_json


@pytest.fixture
def uncertain_episode() -> tuple[EpisodeData, FloatArray]:
    responses = np.array([[12.0, 0.5, 0.02], [10.0, 0.5, 0.02], [9.0, 0.5, 0.02]])
    predictions = np.stack((responses, responses.copy()))
    predictions[1, 2, 0] = 12.0
    return (
        EpisodeData(
            EpisodeSpec(("a", "b", "c"), 0, 3, 42, "uncertainty"),
            np.arange(21, dtype=float).reshape(3, 7),
            responses.copy(),
            responses,
            ObjectiveSpace.fit(responses),
        ),
        predictions,
    )


def test_bonus_changes_choice_and_uses_revealed_front(
    uncertain_episode: tuple[EpisodeData, FloatArray],
) -> None:
    data, samples = uncertain_episode
    environment = create_episode(data)
    mean = EnsembleStrategy(AcquisitionContext(data.space, samples, 0))
    ucb = EnsembleStrategy(AcquisitionContext(data.space, samples, 1))
    assert mean.choose(environment.observe()) == 1
    assert ucb.choose(environment.observe()) == 2
    assert ucb.diagnostics["acquisition_gain_std"] > 0
    assert ucb.diagnostics["acquisition_score"] == pytest.approx(
        ucb.diagnostics["acquisition_mean_gain"]
        + ucb.diagnostics["acquisition_gain_std"]
    )
    environment.step(2)
    np.testing.assert_allclose(ucb.gains(environment.observe()), 0)
    assert ucb.choose(environment.observe()) == 1
    environment.step(1)
    with pytest.raises(ValueError, match="terminado"):
        ucb.choose(environment.observe())


def test_hidden_truth_cannot_change_first_acquisition(
    uncertain_episode: tuple[EpisodeData, FloatArray],
) -> None:
    data, samples = uncertain_episode
    changed = data.responses.copy()
    changed[1:] = [[1000, 1.5, 0.9], [2000, 2, 1]]
    first = create_episode(data).observe()
    second = create_episode(replace(data, responses=changed)).observe()
    np.testing.assert_equal(first.observed, second.observed)
    strategy = EnsembleStrategy(AcquisitionContext(data.space, samples))
    assert strategy.choose(first) == strategy.choose(second)
    assert (first.observed[first.available] == 0).all()


def test_identical_members_reduce_to_greedy_and_respect_constraints(
    uncertain_episode: tuple[EpisodeData, FloatArray],
) -> None:
    data, _ = uncertain_episode
    samples = np.stack((data.predictions, data.predictions))
    samples[:, 2, 1] = 1.5
    environment = create_episode(replace(data, predictions=samples[0]))
    ensemble = EnsembleStrategy(AcquisitionContext(data.space, samples))
    greedy = GreedyStrategy(data.space)
    assert (
        ensemble.choose(environment.observe())
        == greedy.choose(environment.observe())
        == 1
    )
    assert ensemble.diagnostics["acquisition_gain_std"] == 0
    np.testing.assert_equal(ensemble.gains(environment.observe())[:, 1], 0)


def test_identical_five_members_have_exactly_zero_disagreement() -> None:
    values = np.repeat(np.array([[[1000.03, 0.3333, 0.01234]]]), 5, axis=0)
    np.testing.assert_equal(ensemble_disagreement(values), np.zeros((1, 3)))


def test_saved_predictions_are_aligned_and_drift_is_rejected(tmp_path: Path) -> None:
    predicted = pd.DataFrame(
        [[11, 0.5, 0.01], [12, 0.6, 0.02]],
        columns=OUTPUT_COLUMNS,
        index=pd.Index(["a", "b"], name="iter_id"),
    )
    saved = tmp_path / "predictions.csv"
    predicted.add_prefix("pred_").iloc[::-1].to_csv(saved)
    verify_saved_predictions(predicted, saved)
    changed = predicted.copy()
    changed.iloc[0, 0] += 1
    with pytest.raises(ValueError, match="difieren"):
        verify_saved_predictions(changed, saved)
    changed.index = ["unknown", "b"]
    with pytest.raises(ValueError, match="candidatos"):
        verify_saved_predictions(changed, saved)


def test_ensemble_validates_alignment_and_owns_readonly_predictions(
    uncertain_episode: tuple[EpisodeData, FloatArray],
) -> None:
    data, samples = uncertain_episode
    context = AcquisitionContext(data.space, samples)
    samples[:] = 0
    assert context.member_predictions.any()
    with pytest.raises(ValueError, match="read-only"):
        context.member_predictions[:] = 0
    strategy = EnsembleStrategy(
        replace(context, member_predictions=context.member_predictions[:, :2])
    )
    with pytest.raises(ValueError, match="alineados"):
        strategy.choose(create_episode(data).observe())
    with pytest.raises(ValueError, match="miembros"):
        AcquisitionContext(data.space, context.member_predictions[:1])


@pytest.mark.parametrize("beta", [-1, True, "1", float("nan"), float("inf")])
def test_invalid_beta_is_rejected(beta: float) -> None:
    with pytest.raises(ValueError, match="Beta"):
        AdaptiveParameters(beta=beta)


@pytest.mark.parametrize(
    "payload",
    [
        {"beta": 1, "splits": "test"},
        {"beta": 1, "splits": []},
        {"beta": 1, "splits": ["test", "test"]},
        {"beta": 1, "splits": ["training"]},
        {"beta": 1, "splits": [{}]},
        {"beta": 1, "splits": ["test"], "unknown": 1},
    ],
)
def test_invalid_json_is_rejected(
    tmp_path: Path, payload: dict[str, JsonValue]
) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_adaptive_parameters(path)


def test_existing_closed_study_is_extended_without_training(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = tmp_path / "study"
    config = StudyConfig(
        study,
        data_dir=design_data,
        models=("dummy",),
        seeds=(42, 43),
        outer_folds=2,
        trials=0,
        catalog_size=4,
        budgets=(2, 3),
        episodes_per_seed=1,
        rl_steps=4,
        rl_batch=2,
        rl_warmup=2,
        rl_hidden=8,
        validation_interval=2,
        background_size=2,
        permutation_rounds=1,
        rl=RLParameters(algorithms=("double_dqn",)),
        analysis=AnalysisParameters(catalog_sizes=(4,), bootstrap_repeats=20),
        measurement=MeasurementParameters(
            warmup=0,
            inference_repeats=1,
            selection_repeats=1,
            batch_sizes=(1, 4),
            cpu_repeats=1,
            cpu_min_seconds=0.001,
            cpu_min_ticks=1,
            cpu_max_wall_seconds=0.01,
            cpu_probe_seconds=0.001,
        ),
    )
    run_study(config, "validacion")
    with pytest.raises(ValueError, match="cierre completo"):
        run_adaptive_selection(study, tmp_path / "not_closed")
    assert not (tmp_path / "not_closed").exists()
    run_study(config, "final", True)
    before = artifact_signatures(study)
    with pytest.raises(ValueError, match="fuera del estudio"):
        run_adaptive_selection(study, study / "tasks/extension")
    inputs = adaptive_inputs(study)
    incomplete = read_run(inputs.winner)
    incomplete["fold_model_signatures"] = {
        k: v
        for i, (k, v) in enumerate(incomplete["fold_model_signatures"].items())
        if i == 0
    }
    original_read = seleccion_adaptativa.read_run
    with monkeypatch.context() as patch:
        patch.setattr(
            seleccion_adaptativa,
            "read_run",
            lambda directory: (
                incomplete if directory == inputs.winner else original_read(directory)
            ),
        )
        with pytest.raises(ValueError, match="Faltan modelos externos"):
            run_adaptive_selection(study, tmp_path / "incomplete_members")
    assert not (tmp_path / "incomplete_members").exists()

    def forbidden_fit(
        self: Pipeline, features: pd.DataFrame, responses: pd.DataFrame
    ) -> None:
        raise AssertionError("La extensión no puede entrenar")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    export_root = tmp_path / "exports"
    monkeypatch.setattr(
        seleccion_adaptativa, "adaptive_export_root", lambda selected: export_root
    )
    monkeypatch.setattr(
        benchmark, "benchmark_export_root", lambda selected: export_root
    )
    monkeypatch.setattr(
        seleccion_adaptativa, "source_signature", lambda: "new-analysis-code"
    )
    output = export_root / "adaptativa_001"
    original_report = summarize_study(study, export_root / "informe_001")
    assert latest_study_report(study) is not None
    run_adaptive_selection(study, output)
    assert before == artifact_signatures(study)
    assert latest_adaptive_selection(study) == output
    assert latest_study_report(study) is None
    assert read_artifact(output / "provenance.json")["retraining"] is False
    assert (
        read_artifact(output / "provenance.json")["analysis_source_signature"]
        == "new-analysis-code"
    )
    from truss_research.optimizacion.protocolo import load_protocol

    settings = load_protocol(
        Path("config/study.json"), (tmp_path / "unused", design_data)
    ).adaptive
    assert settings.online_residual is True and settings.beta == 1.0
    summary = pd.read_csv(output / "summary.csv")
    assert set(summary.strategy) == {"ensemble_mean", "ensemble_ucb"}
    assert summary.queries.eq(summary.budget).all()
    traces = pd.read_csv(output / "traces.csv")
    assert not traces.duplicated(["split", "episode_id", "strategy", "iter_id"]).any()
    assert traces.loc[~traces.is_initial, "acquisition_beta"].notna().all()
    assert traces.loc[traces.is_initial, "acquisition_score"].isna().all()
    assert traces.queries_remaining.eq(traces.budget - traces["query"]).all()
    for split, key in (
        ("validation", "decision/validation"),
        ("test", "test/decision"),
    ):
        original = read_run(adaptive_inputs(study).tasks[key])["episodes"]
        assert (
            read_artifact(output / f"episodes_{split}.json").get("episodes") == original
        )
        public = pd.read_csv(output / f"public_members_{split}.csv")
        assert set(public.member_id) == {0, 1}
        assert set(public.member_type) == {"outer_fold"}
        assert not {"mass_kg", "util_ratio", "z_disp_m"} & set(public.columns)
    pairs = pd.read_csv(output / "paired_intervals.csv")
    assert "ensemble_mean" in set(
        pairs.loc[pairs.strategy.eq("ensemble_ucb"), "reference_strategy"]
    )
    diagnostics = pd.read_csv(output / "uncertainty_diagnostics.csv")
    assert diagnostics.distinct_member_predictions.eq(2).all()
    assert diagnostics.zero_std_count.eq(0).all()
    assert (output / "resources.json").is_file()
    assert pd.read_csv(output / "decision_costs.csv").decision_cpu_seconds.notna().all()
    assert "decision_seconds" not in pd.read_csv(output / "comparison.csv")

    repeated = export_root / "adaptativa_002"
    run_adaptive_selection(study, repeated, AdaptiveParameters(cpu_workers=2))
    pd.testing.assert_frame_equal(traces, pd.read_csv(repeated / "traces.csv"))
    sequential = summary.drop(
        columns=[
            "decision_seconds",
            "decision_cpu_seconds",
            "preparation_seconds",
            "strategy_seconds",
        ]
    )
    parallel = pd.read_csv(repeated / "summary.csv").loc[:, sequential.columns]
    pd.testing.assert_frame_equal(sequential, parallel)
    assert read_artifact(repeated / "execution.json")["cpu_workers_requested"] == 2
    assert read_artifact(output / "execution.json")["cpu_workers_effective"] == 1
    assert read_artifact(repeated / "execution.json")["worker_coverage"] == "complete"
    jobs = pd.read_csv(repeated / "worker_jobs.csv")
    effective = read_artifact(repeated / "execution.json")["cpu_workers_effective"]
    assert 1 <= jobs.pid.nunique() <= effective
    if effective > 1:
        assert not jobs.pid.eq(psutil.Process().pid).any()
    assert set(pd.read_csv(repeated / "worker_resources.csv").cpu_coverage) == {
        "complete"
    }
    original_prepare = seleccion_adaptativa.prepare_adaptive_bank

    def corrupt_bank(
        prepared: seleccion_adaptativa.PreparedAdaptiveBank, root: Path
    ) -> None:
        original_prepare(prepared, root)
        with (root / "bank.joblib").open("ab") as stream:
            stream.write(b"alterado")

    previous_children = {
        child.pid for child in psutil.Process().children(recursive=True)
    }
    failed = export_root / "adaptativa_003"
    with monkeypatch.context() as patch:
        patch.setattr(seleccion_adaptativa, "prepare_adaptive_bank", corrupt_bank)
        with pytest.raises(RuntimeError, match="Falló"):
            run_adaptive_selection(study, failed, AdaptiveParameters(cpu_workers=2))
    assert read_artifact(failed / "provenance.json")["status"] == "failed"
    assert read_artifact(failed / "execution.json")["status"] == "failed"
    assert {
        child.pid for child in psutil.Process().children(recursive=True)
    } <= previous_children
    assert latest_adaptive_selection(study) == repeated
    report = summarize_study(study, export_root / "informe_002")
    all_results = pd.read_csv(report / "selection_runs.csv")
    assert set(summary.strategy) <= set(all_results.strategy)
    assert (report / "adaptive/ADAPTIVE.md").is_file()
    assert not (report / "adaptive/provenance.json").exists()
    assert not (report / "adaptive/preparation").exists()
    assert not (report / "adaptive/blocks").exists()
    assert "post_hoc_exploratory" in set(all_results.evidence_phase)
    assert (
        read_artifact(original_report / "provenance.json")["adaptive_signature"] is None
    )
    assert read_artifact(report / "provenance.json")["adaptive_signature"] is not None
    assert before == artifact_signatures(study)
    from IPython.display import Markdown

    displayed: list[pd.DataFrame | pd.Series | Markdown] = []
    for filename in ("03_seleccion.ipynb", "04_conclusiones.ipynb"):
        notebook = json.loads(
            (Path("notebooks") / filename).read_text(encoding="utf-8")
        )
        exec(
            compile(
                next(
                    "".join(c["source"])
                    for c in notebook["cells"]
                    if c["cell_type"] == "code"
                    and "worker_resources.csv" in "".join(c["source"])
                ),
                filename,
                "exec",
            ),
            {
                "REPORT_DIR": report,
                "pd": pd,
                "display": displayed.append,
                "Markdown": Markdown,
            },
        )
    assert len(displayed) == 16

    monkeypatch.setattr(benchmark, "source_signature", lambda: "new-measurement-code")
    measured = benchmark.run_benchmark(study, export_root / "benchmark_001")
    assert (
        read_artifact(measured / "provenance.json")["training_source_signature"]
        != "new-measurement-code"
    )
    monkeypatch.setattr(benchmark, "source_signature", lambda: "later-code")
    assert (
        benchmark.verify_benchmark(study, measured)["source_signature"]
        == "new-measurement-code"
    )
    assert before == artifact_signatures(study)

    signature = read_artifact(repeated / "provenance.json")
    signature["status"] = "failed"
    write_json(repeated / "provenance.json", signature)
    assert latest_adaptive_selection(study) == output
    with (output / "summary.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Cambió"):
        latest_adaptive_selection(study)
