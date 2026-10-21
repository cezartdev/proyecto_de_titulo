"""Recursos y benchmark sobre datos sintéticos, sin campañas de investigación."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from truss_research.datos.artefactos import read_artifact
from truss_research.evaluacion.auditoria_estudio import AuditConfig, audit_study
from truss_research.evaluacion.benchmark import latest_benchmark, run_benchmark
from truss_research.evaluacion.informe_costos import (
    amortization_rows,
    resource_inventory,
)
from truss_research.evaluacion.informe_estudio import summarize_study
from truss_research.evaluacion.recursos import (
    MeasurementParameters,
    ResourceMonitor,
    measure_phase,
    resource_session,
)
from truss_research.optimizacion.contratos import (
    AnalysisParameters,
    EvolutionParameters,
)
from truss_research.optimizacion.estudio import run_study
from truss_research.optimizacion.estudio_estado import StudyConfig, artifact_signatures


def test_monitor_reports_units_and_closes_after_exception(tmp_path: Path) -> None:
    path = tmp_path / "resources.json"
    with pytest.raises(RuntimeError, match="interrumpida"):
        with resource_session(path, MeasurementParameters(memory_interval_ms=1)):
            with measure_phase("fit"):
                allocation = bytearray(1024**2)
                assert len(allocation) == 1024**2
                raise RuntimeError("interrumpida")
    payload = read_artifact(path)
    assert payload["status"] == "failed"
    assert {row["phase"] for row in payload["phases"]} == {"task_total", "fit"}
    for phase in payload["phases"]:
        assert phase["finished"] and phase["status"] == "failed"
        assert phase["wall_seconds"] >= 0 and phase["cpu_seconds"] >= 0
        assert phase["peak_rss_mib"] >= phase["initial_rss_mib"] > 0
        assert phase["samples"] >= 2
    with measure_phase("outside_session"):
        pass


def test_monitor_separates_process_cpu_from_unmeasured_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = ResourceMonitor()
    monkeypatch.setattr(
        monitor.process, "children", lambda recursive: [monitor.process]
    )
    with monitor:
        pass
    assert monitor.snapshot()["cpu_coverage"] == "complete"
    assert monitor.snapshot()["cpu_tree_coverage"] == "unmeasured_children"
    assert monitor.snapshot()["children_seen"]


def test_monitor_excludes_startup_and_final_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from truss_research.evaluacion import recursos

    clock = [0.0]

    def sample() -> None:
        clock[0] += 5.0

    monitor = ResourceMonitor(interval_ms=100_000)
    monkeypatch.setattr(monitor, "sample", sample)
    monkeypatch.setattr(recursos, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(recursos, "process_time", lambda: clock[0])
    with monitor:
        clock[0] += 2.0
    assert clock[0] == 12.0
    assert monitor.wall_seconds == monitor.cpu_seconds == 2.0


@pytest.mark.parametrize(
    "values",
    [
        {"memory_interval_ms": 0},
        {"inference_repeats": True},
        {"warmup": -1},
        {"batch_sizes": (4, 1)},
        {"reuse_counts": ()},
    ],
)
def test_measurement_parameters_reject_invalid_values(
    values: dict[str, int | bool | tuple[int, ...]],
) -> None:
    with pytest.raises(ValueError):
        MeasurementParameters(**values)


def test_amortization_counts_preparation_loading_and_cache_once() -> None:
    rows = pd.DataFrame(amortization_rows(100.0, (2.0, 3.0, 5.0), (1, 10)))
    values = rows.set_index(["scenario", "reuse_count"]).total_seconds
    assert values["independent_catalogs", 1] == 110
    assert values["independent_catalogs", 10] == 182
    assert values["repeated_selection_same_catalog", 10] == 155
    baseline = amortization_rows(0.0, (0.0, 0.0, 5.0), (10,))
    assert all(row["total_seconds"] == 50 for row in baseline)
    with pytest.raises(ValueError, match="finitos"):
        amortization_rows(float("nan"), (0.0, 0.0, 1.0), (1,))


def test_closed_benchmark_is_repeatable_read_only_and_auditable(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.evaluacion import benchmark

    exports = tmp_path / "exports"
    monkeypatch.setattr(benchmark, "benchmark_export_root", lambda study: exports)
    config = StudyConfig(
        tmp_path / "study",
        data_dir=design_data,
        models=("dummy", "ridge"),
        seeds=(42,),
        outer_folds=2,
        inner_folds=2,
        trials=0,
        catalog_size=4,
        budgets=(3,),
        episodes_per_seed=1,
        rl_steps=4,
        rl_batch=2,
        rl_warmup=2,
        rl_hidden=8,
        validation_interval=2,
        background_size=2,
        permutation_rounds=1,
        evolution=EvolutionParameters(enabled=True, population_size=2),
        analysis=AnalysisParameters(catalog_sizes=(4, 8), bootstrap_repeats=10),
        measurement=MeasurementParameters(
            memory_interval_ms=10,
            warmup=1,
            inference_repeats=2,
            selection_repeats=2,
            batch_sizes=(1, 4, 8),
        ),
    )
    run_study(config, "validacion")
    with pytest.raises(ValueError, match="cierre completo"):
        run_benchmark(config.output_dir, exports / "benchmark_001")
    assert not exports.exists()
    run_study(config, "final", True)

    def forbidden_fit(
        self: Pipeline, features: pd.DataFrame, responses: pd.DataFrame
    ) -> None:
        raise AssertionError("El benchmark no puede ajustar modelos")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    before = artifact_signatures(config.output_dir)
    with pytest.raises(ValueError, match="make benchmark"):
        audit_study(
            AuditConfig(config.output_dir, config.data_dir, tmp_path / "pending_audit")
        )
    first = run_benchmark(config.output_dir, exports / "benchmark_001")
    second = run_benchmark(config.output_dir, exports / "benchmark_002")
    assert artifact_signatures(config.output_dir) == before
    assert latest_benchmark(config.output_dir) == second
    inference = pd.read_csv(first / "inference_samples.csv")
    predictions = inference.loc[inference.operation.eq("predict")]
    assert (
        predictions.groupby(["split", "batch_size", "repetition"])
        .design_ids.nunique()
        .eq(1)
        .all()
    )
    assert set(predictions.repetition) == {0, 1}
    assert (predictions.wall_seconds >= 0).all()
    selection = pd.read_csv(first / "selection_samples.csv")
    assert set(config.rl.variants) <= set(selection.strategy)
    assert set(
        selection.loc[selection.strategy.isin(config.rl.variants), "algorithm"]
    ) == set(config.rl.algorithms)
    for name in ("rl_ablation", "rl_dueling_ablation", "rl_ppo_ablation"):
        assert (
            selection.loc[selection.strategy.eq(name), "prediction_seconds"].eq(0).all()
        )
    assert selection.loc[~selection.uses_predictions, "prediction_seconds"].eq(0).all()
    assert set(selection.catalog_size) == {4, 8}
    assert selection.queries.eq(3).all()
    one = pd.read_csv(first / "selection_time_traces.csv")
    two = pd.read_csv(second / "selection_time_traces.csv")
    keys = [
        "split",
        "catalog_size",
        "comparison_id",
        "episode_id",
        "repetition",
        "query",
    ]
    pd.testing.assert_frame_equal(
        one.set_index(keys).sort_index()[["iter_id"]],
        two.set_index(keys).sort_index()[["iter_id"]],
    )
    resources = resource_inventory(config.output_dir)
    assert resources.coverage.eq("complete").all()
    total = pd.read_csv(first / "research_cost.csv")
    assert total.wall_seconds.iloc[0] == pytest.approx(
        resources.loc[resources.phase.eq("task_total"), "wall_seconds"].sum()
    )
    assert {
        "fit",
        "tuning",
        "crossfit",
        "policy_training",
        "policy_validation",
        "prediction",
        "explanation",
    } <= set(resources.phase)
    assert (first / "prediction_quality_cost.png").is_file()
    report = summarize_study(config.output_dir, tmp_path / "report")
    assert (report / "benchmark/amortization.csv").is_file()
    from IPython.display import Image, Markdown

    displayed: list[pd.DataFrame | Image | Markdown] = []
    notebook_root = Path(__file__).resolve().parents[1] / "notebooks"
    for notebook_name in ("02_prediccion", "03_seleccion", "04_conclusiones"):
        notebook = json.loads(
            (notebook_root / f"{notebook_name}.ipynb").read_text(encoding="utf-8")
        )
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and 'cost_root = REPORT_DIR / "benchmark"' in "".join(cell["source"])
        )
        exec(
            compile(source, notebook_name, "exec"),
            {
                "REPORT_DIR": report,
                "pd": pd,
                "display": displayed.append,
                "Markdown": Markdown,
            },
        )
    assert any(isinstance(value, Image) for value in displayed)
    assert any(isinstance(value, pd.DataFrame) for value in displayed)
    audit_study(AuditConfig(config.output_dir, config.data_dir, tmp_path / "audit"))
    assert (report / "error_tail_summary.csv").is_file()
    assert (report / "category_coverage.csv").is_file()
    assert (report / "convergence_diagnostics.csv").is_file()
    policies = pd.read_csv(report / "policy_training_summary.csv")
    assert len(policies) == 6 and set(policies.strategy) == set(config.rl.variants)
    assert policies.completed_steps.eq(config.rl_steps).all()
    assert (report / "rl_learning_cpu.png").is_file()
    cpu = pd.read_csv(first / "cpu_operation_summary.csv")
    assert set(cpu.operation) == {"predict", "episode"}
    assert cpu.calls.gt(0).all()
    assert cpu.loc[cpu.status.eq("resolved"), "cpu_per_call_seconds"].gt(0).all()
    from truss_research.evaluacion.presentacion import export_notebooks

    monkeypatch.setattr(
        "truss_research.evaluacion.presentacion.latest_study_report",
        lambda directory: report,
    )
    presentation = export_notebooks(config.output_dir, tmp_path / "presentation")
    assert len(list(presentation.glob("*.html"))) == 4
    for notebook_path in presentation.glob("*.ipynb"):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert all(cell["execution_count"] is not None for cell in code)
        assert not any(
            output["output_type"] == "error"
            for cell in code
            for output in cell["outputs"]
        )
    assert artifact_signatures(config.output_dir) == before
    damaged = second / "inference_samples.csv"
    damaged.write_text(
        damaged.read_text(encoding="utf-8") + "alterado", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Cambió una medición"):
        latest_benchmark(config.output_dir)


def test_inventory_preserves_missing_and_partial_resources(tmp_path: Path) -> None:
    from truss_research.optimizacion.estudio_estado import STUDY_SCHEMA_VERSION

    path = tmp_path / "study.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": STUDY_SCHEMA_VERSION,
                "tasks": [
                    {
                        "key": "predictor/dummy/42",
                        "attempts": ["attempt_001"],
                        "measurement_files": [str(tmp_path / "missing.json")],
                        "measurement_signatures": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    inventory = resource_inventory(tmp_path)
    assert inventory.coverage.eq("missing").all()
    assert inventory.wall_seconds.isna().all()
    with ResourceMonitor() as monitor:
        pass
    partial = tmp_path / "partial.json"
    snapshot = {
        "status": "running",
        "phases": [{"phase": "task_total", **monitor.snapshot(), "finished": False}],
    }
    partial.write_text(json.dumps(snapshot), encoding="utf-8")
    state = json.loads(path.read_text(encoding="utf-8"))
    state["tasks"][0]["measurement_files"] = [str(partial)]
    path.write_text(json.dumps(state), encoding="utf-8")
    inventory = resource_inventory(tmp_path)
    assert inventory.coverage.eq("incomplete").all()
    assert np.isfinite(inventory.wall_seconds).all()


def test_cli_benchmark_missing_study_has_no_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from truss_research.__main__ import main

    missing = tmp_path / "missing"
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "truss_research",
            "medir-costos",
            "--estudio",
            str(missing),
            "--destino",
            str(output),
        ],
    )
    assert main() == 1
    assert not missing.exists() and not output.exists()
