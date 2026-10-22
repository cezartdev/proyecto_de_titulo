"""Coordinación reanudable con búsquedas y pliegues paralelos en CPU."""

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from time import perf_counter, sleep

from truss_research.datos.preparacion import source_digest
from truss_research.evaluacion.recursos import resource_session
from truss_research.evaluacion.registro import attempt_logs, execution_log
from truss_research.evaluacion.telemetria_cpu import CampaignMonitor
from truss_research.optimizacion.contratos import StudyTask
from truss_research.optimizacion.ejecucion_cpu import MemoryAdmission, cpu_pool
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    StudyPhase,
    StudySession,
    TaskRecord,
    artifact_signatures,
    open_session,
    study_lock,
    verify_task,
)
from truss_research.surrogates.configuracion import PredictorConfig
from truss_research.surrogates.experimentos import train_predictor

logger = logging.getLogger(__name__)


def dispatch_task(session: StudySession, task: StudyTask, destination: Path) -> None:
    config = session.config
    train_predictor(
        PredictorConfig(
            config.data_dir,
            destination,
            model=task.model,
            seed=task.seed,
            folds=config.folds,
            cv_seed=config.cv_seed,
            nested_folds=config.nested_folds,
            final_tuning_folds=config.final_tuning_folds,
            candidate_models=tuple(m for m in config.models if m != "targetwise"),
            trials=0 if task.model == "tabpfn" else config.trials,
            tabpfn=config.tabpfn,
            pruning=config.pruning,
            device="cuda" if task.model in config.execution.gpu_models else "cpu",
            parameters=config.parameters,
            reuse_dir=(
                verify_task(
                    session.tasks[
                        f"predictor/{task.model}/{config.representative_seed}"
                    ]
                )
                if task.seed != config.representative_seed
                else None
            ),
            study_path=config.output_dir
            / "studies"
            / f"{task.model}_{task.seed}.sqlite",
        )
    )
    return


def run_task(session: StudySession, task: StudyTask) -> bool:
    record = session.tasks[task.key]
    if record.status == "complete":
        verify_task(record)
        logger.info("Conservada: %s", task.key)
        return True
    if session.test_started and not task.key.startswith("test/"):
        raise ValueError(
            "No se permite entrenar ni revisar decisiones después del cierre"
        )
    destination = (
        session.config.output_dir
        / "tasks"
        / task.key
        / f"attempt_{len(record.attempts) + 1:03d}"
    )
    if (
        record.status == "running"
        and record.measurement_files
        and Path(record.measurement_files[-1]).is_file()
    ):
        from truss_research.datos.artefactos import read_artifact

        interrupted = read_artifact(Path(record.measurement_files[-1]))
        record.seconds += float(interrupted["phases"][0]["wall_seconds"])
        prior_path = record.measurement_files[-1]
        record.measurement_signatures[prior_path] = source_digest(Path(prior_path))
    record.attempts.append(str(destination))
    record.log_files.extend(str(path) for path in attempt_logs(destination))
    measurement_path = destination.with_suffix(".resources.json")
    record.measurement_files.append(str(measurement_path))
    record.status, record.error = "running", None
    session.save()
    logger.info("Iniciando %s (intento %s)", task.key, len(record.attempts))
    start = perf_counter()
    try:
        with (
            execution_log(destination.with_suffix(".log"), task.key),
            resource_session(measurement_path, session.config.measurement),
        ):
            dispatch_task(session, task, destination)
        record.signatures = artifact_signatures(destination)
        record.directory, record.status = str(destination), "complete"
    except Exception as error:
        record.status, record.error = "failed", f"{type(error).__name__}: {error}"
        logger.exception("Falló %s", task.key)
    except KeyboardInterrupt:
        record.status, record.error = (
            "failed",
            "Entrenamiento interrumpido por el usuario",
        )
        raise
    finally:
        record.log_signatures.update(
            {
                str(path): source_digest(path)
                for path in attempt_logs(destination)
                if path.is_file()
            }
        )
        record.seconds += perf_counter() - start
        if measurement_path.is_file():
            record.measurement_signatures[str(measurement_path)] = source_digest(
                measurement_path
            )
        session.save()
    logger.info("%s: %s (%.1f s acumulados)", task.key, record.status, record.seconds)
    return record.status == "complete"


def run_tasks(session: StudySession, tasks: tuple[StudyTask, ...]) -> None:
    if (
        tasks
        and all(task.kind == "predictor" for task in tasks)
        and session.config.execution.effective_workers > 1
    ):
        from truss_research.optimizacion.paralelismo import run_parallel_predictions

        run_parallel_predictions(session)
        return
    for task in tasks:
        session.tasks.setdefault(task.key, TaskRecord(task.key))
    session.save()
    results: list[bool] = []
    gate = MemoryAdmission(session.config.execution)
    for number, task in enumerate(tasks):
        while session.tasks[task.key].status != "complete" and not gate.allow(0):
            if session.telemetry is not None:
                session.telemetry.update((0, len(tasks) - number, 0, True))
            sleep(session.config.execution.telemetry_interval_seconds)
        if session.telemetry is not None:
            session.telemetry.update((1, len(tasks) - number - 1, 0, False))
        results.append(run_task(session, task))
    if session.telemetry is not None:
        session.telemetry.update((0, 0, 0, False))
    if not all(results):
        raise RuntimeError(
            "Hay tareas fallidas; consultar study.json antes de reanudar"
        )


def prediction_tasks(config: StudyConfig) -> tuple[StudyTask, ...]:
    return tuple(
        StudyTask(f"predictor/{model}/{seed}", "predictor", model, seed)
        for model in config.models
        for seed in (
            config.representative_seed,
            *(seed for seed in config.seeds if seed != config.representative_seed),
        )
    )


def requires_execution(session: StudySession, phase: StudyPhase) -> bool:
    return any(
        task.key not in session.tasks or session.tasks[task.key].status != "complete"
        for task in prediction_tasks(session.config)
    )


@contextmanager
def execution_record(session: StudySession, phase: StudyPhase) -> Iterator[None]:
    if not requires_execution(session, phase):
        yield
        return
    record = {
        "phase": phase,
        "status": "running",
        "wall_seconds": 0.0,
        "cpu_workers": session.config.execution.effective_workers,
        "gpu_workers": session.config.execution.gpu_workers
        if session.config.tabpfn_enabled and session.config.tabpfn.device == "cuda"
        else 0,
        "tabpfn_device": session.config.tabpfn.device,
        "rl_device": session.config.execution.rl_device,
        "work_seconds": 0.0,
    }
    before = sum(
        task.seconds
        for task in (*session.tasks.values(), *session.preparation_tasks.values())
    )
    session.execution_runs.append(record)
    telemetry_path = (
        session.config.output_dir
        / "execution"
        / f"run_{len(session.execution_runs):03d}.telemetry.csv"
    )
    record["telemetry_file"] = str(telemetry_path)
    session.save()
    start = perf_counter()
    try:
        with ExitStack() as stack:
            session.telemetry = stack.enter_context(
                CampaignMonitor(
                    telemetry_path, session.config.execution.telemetry_interval_seconds
                )
            )
            if session.config.execution.effective_workers > 1:
                session.executor = stack.enter_context(
                    cpu_pool(session.config.execution)
                )
            if (
                (
                    session.config.tabpfn_enabled
                    and session.config.tabpfn.device == "cuda"
                )
                or session.config.execution.rl_device == "cuda"
                or session.config.execution.gpu_models
            ):
                session.gpu_executor = stack.enter_context(
                    cpu_pool(session.config.execution, "cuda")
                )
            yield
        record["status"] = "complete"
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        if session.executor is not None:
            record["memory_recoveries"] = session.executor.recoveries
            record["final_worker_capacity"] = session.executor.capacity
        session.executor = None
        session.gpu_executor = None
        record["telemetry_errors"] = (
            session.telemetry.errors if session.telemetry is not None else []
        )
        session.telemetry = None
        if telemetry_path.is_file():
            record["telemetry_signature"] = source_digest(telemetry_path)
        record["wall_seconds"] = perf_counter() - start
        record["work_seconds"] = (
            sum(
                task.seconds
                for task in (
                    *session.tasks.values(),
                    *session.preparation_tasks.values(),
                )
            )
            - before
        )
        session.save()


def run_study(
    config: StudyConfig, phase: StudyPhase = "todo", resume: bool = False
) -> Path:
    if phase not in ("predictores", "validacion", "final", "todo"):
        raise ValueError("Fase desconocida")
    if phase != "predictores":
        from truss_research.evaluacion.busqueda_evolutiva import run_evolution_study

        return run_evolution_study(config, phase, resume)
    config.execution.validate_campaign()
    validate_training_protocol(config)
    config = replace(
        config,
        output_dir=config.output_dir.resolve(),
        data_dir=config.data_dir.resolve(),
    )
    with study_lock(config.output_dir):
        session = open_session(config, resume)
        with execution_record(session, phase):
            run_tasks(session, prediction_tasks(config))
    return config.output_dir


def validate_training_protocol(config: StudyConfig) -> None:
    """Rechaza reservas o pliegues insuficientes antes de ejecutar búsquedas."""

    from truss_research.datos.procesados import load_design_split, read_manifest

    if config.tabpfn_enabled:
        from truss_research.evaluacion.piloto_tabpfn import require_pilot

        require_pilot(config)

    split = load_design_split(config.data_dir)
    from truss_research.datos.procesados import manifest_protocol
    from truss_research.datos.temporal import first_appearance_folds

    if manifest_protocol(read_manifest(config.data_dir)) != config.split_protocol:
        raise ValueError("La partición preparada difiere del protocolo solicitado")
    available = (
        min(len(split.validation), len(split.testing))
        + config.evolution.population_size
    )
    if max(config.budgets) > available:
        raise ValueError(
            f"El presupuesto {max(config.budgets)} agota un catálogo retenido; "
            f"usar como máximo {available} evaluaciones incluidas las iniciales"
        )
    folds = first_appearance_folds(split.training, config.outer_folds)
    first_appearance_folds(split.training, config.final_tuning_folds)
    for fitting, _ in folds:
        first_appearance_folds(
            split.training.iloc[fitting], config.inner_folds or config.outer_folds
        )
    if config.calibration_required and (
        split.calibration is None or len(split.calibration) < 19
    ):
        raise ValueError("La calibración temporal al 95 % requiere al menos 19 diseños")
