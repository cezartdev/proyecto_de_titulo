"""Equivalencia y reanudación del entrenamiento paralelo con mundos sintéticos."""

import hashlib
import json
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import pytest
import torch

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procesados import load_design_split
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.optimizacion.busqueda import SearchContext
from truss_research.optimizacion.contratos import EvolutionParameters
from truss_research.optimizacion.ejecucion_cpu import ExecutionParameters
from truss_research.optimizacion.entrenamiento_propuestas import (
    load_proposal_policy,
    train_proposal_policy,
)
from truss_research.optimizacion.estrategias_busqueda import SearchParameters
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    artifact_signatures,
    read_study,
)
from truss_research.optimizacion.evaluadores import Evaluation
from truss_research.optimizacion.paralelismo_politicas import (
    policy_jobs,
    run_policy_campaign,
)
from truss_research.optimizacion.propuestas import ProposalParameters, TrainingSupport


class FormulaWorld:
    def evaluate(self, features: np.ndarray) -> Evaluation:
        sleep(0.005)
        responses = np.column_stack(
            (
                2000 - 100 * features[:, 2],
                np.full(len(features), 0.5),
                np.full(len(features), 0.01),
            )
        )
        return Evaluation(
            features.copy(), responses, np.zeros(len(features)), "synthetic_exact"
        )


def test_progress_manifest_defers_lock_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.optimizacion import paralelismo_politicas as parallel

    path = tmp_path / "manifest.json"
    original = parallel.write_json
    attempts: list[Path] = []

    def blocked_once(destination, state):
        attempts.append(destination)
        if len(attempts) == 1:
            raise PermissionError("bloqueo del manifiesto simulado")
        original(destination, state)

    monkeypatch.setattr(parallel, "write_json", blocked_once)
    monkeypatch.setattr(parallel, "perf_counter", lambda: 100.0)
    last = parallel.save_progress(path, {"steps": 1}, 0)
    assert last == 100 and not path.exists()
    assert parallel.save_progress(path, {"steps": 2}, last) == last
    assert len(attempts) == 1
    monkeypatch.setattr(parallel, "perf_counter", lambda: 120.0)
    parallel.save_progress(path, {"steps": 3}, last)
    assert json.loads(path.read_text())["steps"] == 3


def test_resource_monitor_defers_locked_periodic_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.evaluacion import recursos

    ledger = recursos.ResourceLedger(
        tmp_path / "resources.json", recursos.MeasurementParameters()
    )
    attempts: list[Path] = []

    def blocked(current):
        attempts.append(current.path)
        raise PermissionError("bloqueo de medición simulado")

    monkeypatch.setattr(recursos.ResourceLedger, "save", blocked)
    ledger.checkpoint()
    assert attempts == [ledger.path] and ledger.last_save > 0


def test_campaign_continues_after_periodic_manifest_block(
    small_policy: tuple[StudyConfig, SearchContext], monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.optimizacion import paralelismo_politicas as parallel

    config, context = small_policy
    original = parallel.write_json
    writes: list[Path] = []

    def temporarily_blocked(path, state):
        writes.append(path)
        if len(writes) == 3:
            raise PermissionError("bloqueo periódico simulado")
        original(path, state)

    monkeypatch.setattr(parallel, "write_json", temporarily_blocked)
    monkeypatch.setattr(parallel, "PROGRESS_SAVE_SECONDS", 0)
    parallel.run_policy_campaign(config, context, ("fixture", False))
    state = json.loads(
        (config.output_dir / "evolution/policy_execution/manifest.json").read_text()
    )
    assert len(writes) > 3
    assert state["runs"][0]["status"] == "complete"
    assert all(row["status"] == "complete" for row in state["jobs"].values())


@pytest.fixture
def small_policy(
    prepared_temporal: Path, tmp_path: Path
) -> tuple[StudyConfig, SearchContext]:
    table = load_design_split(prepared_temporal).training
    features = table[list(INPUT_COLUMNS)].to_numpy()
    responses = table[list(OUTPUT_COLUMNS)].to_numpy()
    space = ObjectiveSpace.fit(responses)
    config = StudyConfig(
        tmp_path / "parallel",
        data_dir=prepared_temporal,
        models=("dummy", "ridge"),
        seeds=(42, 43),
        budgets=(7,),
        rl_steps=8,
        rl_batch=2,
        rl_warmup=2,
        rl_hidden=8,
        validation_interval=4,
        evolution=EvolutionParameters(population_size=4),
        proposals=ProposalParameters(
            pool_size=2, max_generated=128, evaluation_episodes=1
        ),
        search=SearchParameters(strategies=("rl", "contextual_bandit")),
        execution=ExecutionParameters(cpu_workers=2, telemetry_interval_seconds=0.05),
    )
    world = FormulaWorld()
    context = SearchContext(
        features,
        responses,
        (world,),
        TrainingSupport.fit(features, 0.95),
        space,
        space.hypervolume(responses),
        config.proposals,
        config.evolution,
        7,
        validation_world=world,
        search_parameters=config.search,
    )
    return config, context


def test_parallel_policies_match_serial_and_reuse_complete_checkpoints(
    small_policy: tuple[StudyConfig, SearchContext],
) -> None:
    config, context = small_policy
    serial = replace(config, output_dir=config.output_dir.with_name("serial"))
    for job in policy_jobs(serial, context, "fixture").values():
        train_proposal_policy(job.context, job.config, job.resources.destination)
    run_policy_campaign(config, context, ("fixture", False))
    path = config.output_dir / "evolution/policy_execution/manifest.json"
    state = json.loads(path.read_text())
    assert all(row["status"] == "complete" for row in state["jobs"].values())
    pids = {row["attempts"][0]["pid"] for row in state["jobs"].values()}
    assert len(pids) == 2
    attempts = [row["attempts"][0] for row in state["jobs"].values()]
    assert any(
        a["started_at"] < b["finished_at"] and b["started_at"] < a["finished_at"]
        for a in attempts
        for b in attempts
        if a["pid"] != b["pid"]
    )
    for key in state["jobs"]:
        actual, _ = load_proposal_policy(config.output_dir / "evolution/policies" / key)
        expected, _ = load_proposal_policy(
            serial.output_dir / "evolution/policies" / key
        )
        for name, value in actual.state_dict().items():
            torch.testing.assert_close(
                value, expected.state_dict()[name], rtol=0, atol=0
            )
        original = json.loads(
            (serial.output_dir / "evolution/policies" / key / "run.json").read_text()
        )
        current = json.loads(
            (config.output_dir / "evolution/policies" / key / "run.json").read_text()
        )
        for name in (
            "training_costs",
            "validation_costs",
            "optimizer_updates",
            "training_surrogate_evaluations",
        ):
            assert current[name] == original[name]
    from truss_research.evaluacion.informe_busqueda import export_policy_resources

    destination = config.output_dir.with_name("policy_report")
    destination.mkdir()
    export_policy_resources(config.output_dir, destination)
    resources = pd.read_csv(destination / "policy_resources.csv")
    assert len(resources) == 4 and resources.coverage.eq("complete").all()
    assert resources.cpu_seconds.ge(0).all()
    assert pd.read_csv(destination / "policy_execution.csv").workers.tolist() == [2]
    before = artifact_signatures(config.output_dir)
    run_policy_campaign(config, context, ("fixture", False))
    assert artifact_signatures(config.output_dir) == before
    policy = config.output_dir / "evolution/policies/rl/42/policy.pt"
    policy.write_bytes(b"altered")
    with pytest.raises(ValueError, match="Cambió una política"):
        run_policy_campaign(config, context, ("fixture", False))


def test_source_manifest_rejects_changes_to_scientific_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.evaluacion import reutilizacion_predictores as reuse

    (tmp_path / "src").mkdir()
    (tmp_path / "src/science.py").write_text("VALUE = 1\n")
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "uv.lock").write_text("")
    files = reuse.current_sources(tmp_path)
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {"source_files": files, "source_signature": identity}
    monkeypatch.setattr(reuse, "SOURCE_ROOT", tmp_path)
    reuse.verify_source_manifest(manifest, identity)
    (tmp_path / "src/science.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="código científico"):
        reuse.verify_source_manifest(manifest, identity)


def test_interrupted_coordinator_can_restart_pending_policies(
    small_policy: tuple[StudyConfig, SearchContext], monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.optimizacion import paralelismo_politicas as parallel

    config, context = small_policy
    config = replace(config, search=SearchParameters(strategies=("rl",)))

    class IdlePool:
        capacity = 2
        recoveries = []

        def admit(self, gate, active):
            return True

        def submit(self, operation, command):
            return Future()

    @contextmanager
    def idle_pool(parameters, device):
        yield IdlePool()

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as scoped:
        scoped.setattr(parallel, "cpu_pool", idle_pool)
        scoped.setattr(parallel, "wait", interrupt)
        with pytest.raises(KeyboardInterrupt):
            parallel.run_policy_campaign(config, context, ("fixture", False))
    path = config.output_dir / "evolution/policy_execution/manifest.json"
    state = json.loads(path.read_text())
    assert state["runs"][0]["status"] == "interrupted"
    assert all(row["status"] == "interrupted" for row in state["jobs"].values())
    orphan = config.output_dir / "evolution/policies/rl/42"
    orphan.mkdir(parents=True)
    (orphan / "encoder.joblib").write_bytes(b"incomplete-initialization")
    parallel.run_policy_campaign(config, context, ("fixture", False))
    state = json.loads(path.read_text())
    assert [row["status"] for row in state["runs"]] == ["interrupted", "complete"]
    assert all(
        row["status"] == "complete" and len(row["attempts"]) == 2
        for row in state["jobs"].values()
    )
    assert (
        config.output_dir
        / "evolution/policy_execution/jobs/rl/42/attempt_002.initialization/encoder.joblib"
    ).read_bytes() == b"incomplete-initialization"


def test_import_preserves_original_predictors_and_never_trains(
    prepared_temporal: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.datos.preparacion import source_digest
    from truss_research.evaluacion.benchmark_multiobjetivo import (
        evaluate_predictor_utility,
    )
    from truss_research.evaluacion.busqueda_evolutiva import (
        choose_predictors,
        predictor_directories,
    )
    from truss_research.evaluacion.reutilizacion_predictores import (
        PredictorImport,
        current_sources,
        import_predictors,
        verify_imported_predictors,
        verify_reused_validation,
    )
    from truss_research.optimizacion import estudio
    from truss_research.optimizacion.estudio_estado import source_signature
    from truss_research.surrogates.experimentos import write_json

    source = tmp_path / "source"
    config = StudyConfig(
        source,
        data_dir=prepared_temporal,
        models=("dummy", "ridge"),
        seeds=(42,),
        trials=0,
        outer_folds=2,
        inner_folds=2,
        final_tuning_folds=2,
        budgets=(7,),
        evolution=EvolutionParameters(population_size=4),
        proposals=ProposalParameters(evaluation_episodes=2),
        search=SearchParameters(strategies=("random",)),
    )
    estudio.run_study(config, "predictores")
    evaluate_predictor_utility(config, "validation")
    selected = choose_predictors(config, predictor_directories(config))
    write_json(source / "evolution/search.json", {"selected": list(selected)})
    (source / "evolution/policies/rl/42").mkdir(parents=True)
    write_json(
        source / "evolution/policies/rl/42/run.json",
        {
            "phase": "in_progress",
            "completed_steps": 4,
            "wall_seconds": 2.0,
            "cpu_seconds": 1.5,
        },
    )
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_files": current_sources(Path.cwd()),
                "source_signature": source_signature(),
                "study_signature": source_digest(source / "study.json"),
            }
        )
    )
    before = artifact_signatures(source)
    requests = [
        PredictorImport(source, destination, manifest)
        for destination in (tmp_path / "destination", tmp_path / "other/destination")
    ]
    for request in requests:
        import_predictors(request, True)
        assert not request.destination.exists()
        import_predictors(request)
    assert artifact_signatures(source) == before
    hidden = tmp_path / "unavailable_original"
    source.rename(hidden)
    monkeypatch.setattr(
        estudio,
        "validate_training_protocol",
        lambda settings: pytest.fail("No debe entrenar ni exigir un piloto nuevo"),
    )
    from truss_research.evaluacion.busqueda_evolutiva import (
        build_context,
        run_evolution_study,
    )
    from truss_research.evaluacion.informe_busqueda import (
        export_policy_resources,
        verify_closed_search,
    )
    from truss_research.optimizacion.entrenamiento_propuestas import (
        ProposalTrainingConfig,
    )
    from truss_research.optimizacion.politica import DQNParameters
    from truss_research.surrogates.experimentos import load_predictor, predict

    for request in requests:
        imported = replace(config, output_dir=request.destination)
        verify_imported_predictors(imported)
        estudio.run_study(imported, "predictores", True)
        evaluate_predictor_utility(imported, "validation")
        verify_reused_validation(imported, selected)
        state = read_study(request.destination)
        assert all(
            row["status"] == "complete"
            and Path(row["directory"]).is_relative_to(request.destination)
            for row in state["tasks"]
        )
        assert all(
            Path(run["telemetry_file"]).is_relative_to(request.destination)
            for run in state["execution_runs"]
        )
        assert not (request.destination / "evolution/policies").exists()
        for path in predictor_directories(imported).values():
            assert (
                np.isfinite(
                    predict(
                        load_predictor(path),
                        load_design_split(prepared_temporal).validation[
                            list(INPUT_COLUMNS)
                        ],
                    )
                )
                .all()
                .all()
            )
        paths = predictor_directories(imported)
        context = build_context(imported, (paths[selected[0]], paths[selected[2]]))
        training = ProposalTrainingConfig(
            42,
            (7,),
            DQNParameters(
                transitions=2,
                hidden_width=8,
                batch_size=2,
                warmup=100,
                validation_interval=2,
            ),
            "independent",
            1,
        )
        train_proposal_policy(
            context, training, request.destination / "synthetic_policy"
        )
        run_evolution_study(imported, "validacion", True)
        run_evolution_study(imported, "final", True)
        verify_closed_search(request.destination)
        assert (request.destination / "evolution/closed.json").is_file()
        report = request.destination / "report"
        report.mkdir()
        export_policy_resources(request.destination, report)
        costs = pd.read_csv(report / "discarded_policy_costs.csv")
        assert costs.completed_steps.tolist() == [4]
        assert not source.exists()
        verify_imported_predictors(imported)
    assert artifact_signatures(hidden) == before
    copied = (
        Path(read_study(request.destination)["tasks"][0]["directory"]) / "model.joblib"
    )
    copied.write_bytes(b"altered")
    with pytest.raises(ValueError, match="artefactos"):
        verify_imported_predictors(imported)
