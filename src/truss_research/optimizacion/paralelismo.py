"""Planificador global con dependencias y un único escritor del estado del estudio."""

import logging
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Literal

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.evaluacion.registro import attempt_logs
from truss_research.optimizacion.contratos import StudyTask
from truss_research.optimizacion.ejecucion_cpu import (
    CpuPool,
    MemoryAdmission,
    cpu_pool,
    gpu_admission,
)
from truss_research.optimizacion.estudio_estado import (
    StudySession,
    TaskRecord,
    artifact_signatures,
    verify_task,
)
from truss_research.optimizacion.trabajos_predictivos import (
    JobOutcome,
    PredictorJob,
    run_predictor_job,
)
from truss_research.surrogates.configuracion import ModelName, PredictorConfig

logger = logging.getLogger(__name__)


def uses_gpu(session: StudySession, node: "JobNode") -> bool:
    if session.gpu_executor is None:
        return False
    if node.kind == "policy":
        return session.config.execution.rl_device == "cuda"
    if node.kind in ("block", "decision"):
        return False
    tabpfn_gpu = (
        session.config.tabpfn_enabled and session.config.tabpfn.device == "cuda"
    )
    return node.model in session.config.execution.gpu_models or (
        tabpfn_gpu and (node.model in ("tabpfn", "targetwise") or node.kind == "bank")
    )


@dataclass(frozen=True)
class JobNode:
    key: str
    kind: Literal[
        "search",
        "fold",
        "predictor",
        "policy",
        "explain",
        "ranking",
        "test_predictor",
        "bank",
        "block",
        "decision",
    ]
    model: ModelName
    seed: int
    dependencies: tuple[str, ...] = ()
    fold: int = 0
    task: StudyTask | None = None
    indices: tuple[int, ...] = ()


def prediction_graph(session: StudySession) -> tuple[JobNode, ...]:
    config, nodes = session.config, []
    for model in config.models:
        prefix = f"predictor/{model}/{config.representative_seed}"
        search = f"{prefix}/search"
        nodes.append(JobNode(search, "search", model, config.representative_seed))
        folds = tuple(f"{prefix}/outer_{number}" for number in range(config.folds))
        nodes.extend(
            JobNode(
                key,
                "fold",
                model,
                config.representative_seed,
                () if config.nested_folds is not None else (search,),
                number,
            )
            for number, key in enumerate(folds)
        )
        nodes.append(
            JobNode(
                prefix, "predictor", model, config.representative_seed, (search, *folds)
            )
        )
        nodes.extend(
            JobNode(f"predictor/{model}/{seed}", "predictor", model, seed, (prefix,))
            for seed in config.seeds
            if seed != config.representative_seed
        )
    return tuple(nodes)


def node_record(session: StudySession, node: JobNode) -> TaskRecord:
    records = (
        session.preparation_tasks
        if node.kind in ("search", "fold", "bank", "block")
        else session.tasks
    )
    return records.setdefault(node.key, TaskRecord(node.key))


def begin_attempt(session: StudySession, node: JobNode) -> Path:
    record = node_record(session, node)
    if record.status == "running" and record.measurement_files:
        path = Path(record.measurement_files[-1])
        if path.is_file():
            snapshot = read_artifact(path)
            record.seconds += float(snapshot["phases"][0]["wall_seconds"])
            record.measurement_signatures[str(path)] = source_digest(path)
    root = (
        "preparation" if node.kind in ("search", "fold", "bank", "block") else "tasks"
    )
    destination = (
        session.config.output_dir
        / root
        / node.key
        / f"attempt_{len(record.attempts) + 1:03d}"
    )
    record.attempts.append(str(destination))
    record.log_files.extend(str(path) for path in attempt_logs(destination))
    record.measurement_files.append(str(destination.with_suffix(".resources.json")))
    record.status, record.error = "running", None
    return destination


def start_node(session: StudySession, node: JobNode) -> PredictorJob:
    destination = begin_attempt(session, node)
    config = session.config
    representative = f"predictor/{node.model}/{config.representative_seed}"
    reuse = (
        verify_task(session.tasks[representative])
        if node.kind == "predictor" and node.seed != config.representative_seed
        else None
    )
    predictor = PredictorConfig(
        config.data_dir,
        destination,
        model=node.model,
        seed=node.seed,
        folds=config.folds,
        trials=0 if node.model == "tabpfn" else config.trials,
        tabpfn=config.tabpfn,
        pruning=config.pruning,
        device="cuda" if node.model in config.execution.gpu_models else "cpu",
        parameters=config.parameters,
        cv_seed=config.cv_seed,
        nested_folds=config.nested_folds,
        final_tuning_folds=config.final_tuning_folds,
        candidate_models=tuple(m for m in config.models if m != "targetwise"),
        reuse_dir=reuse,
        study_path=config.output_dir / "studies" / f"{node.model}_{node.seed}.sqlite",
    )
    search = (
        verify_task(session.preparation_tasks[f"{representative}/search"])
        if node.dependencies and node.kind in ("fold", "predictor") and reuse is None
        else None
    )
    folds = (
        tuple(
            verify_task(session.preparation_tasks[f"{representative}/outer_{number}"])
            for number in range(config.folds)
        )
        if node.kind == "predictor" and reuse is None
        else ()
    )
    session.save()
    return PredictorJob(
        node.key, node.kind, predictor, config.measurement, node.fold, search, folds
    )


def finish_node(session: StudySession, node: JobNode, outcome: JobOutcome) -> None:
    record = node_record(session, node)
    destination = Path(record.attempts[-1])
    record.seconds += outcome.seconds
    record.error = outcome.error
    record.worker_history.append(
        {
            "pid": outcome.pid,
            "started_at": outcome.started_at,
            "finished_at": outcome.finished_at,
            "success": outcome.success,
        }
    )
    if outcome.success:
        record.directory, record.signatures = (
            str(destination),
            artifact_signatures(destination),
        )
    measurement = Path(record.measurement_files[-1])
    if measurement.is_file():
        record.measurement_signatures[str(measurement)] = source_digest(measurement)
    record.log_signatures.update(
        {
            str(path): source_digest(path)
            for path in attempt_logs(destination)
            if path.is_file()
        }
    )
    record.status = "complete" if outcome.success else "failed"
    session.save()
    logger.info("%s: %s (%.1f s de trabajo)", node.key, record.status, outcome.seconds)


def schedule_ready(
    session: StudySession,
    pending: list[JobNode],
    context: tuple[CpuPool, dict[Future[JobOutcome], JobNode]],
) -> None:
    executor, active = context
    cpu_active = sum(not uses_gpu(session, node) for node in active.values())
    gpu_active = len(active) - cpu_active
    available = executor.capacity - cpu_active
    lookup = {**session.preparation_tasks, **session.tasks}
    gpu_capacity = (
        session.gpu_executor.capacity if session.gpu_executor is not None else 0
    )
    if available <= 0 and gpu_active >= gpu_capacity:
        return
    ready = [
        node
        for node in pending
        if all(lookup[key].status == "complete" for key in node.dependencies)
    ]
    priority = {
        "search": 0,
        "fold": 0,
        "bank": 0,
        "policy": 0,
        "predictor": 1,
        "explain": 2,
        "ranking": 3,
        "test_predictor": 1,
        "block": 1,
        "decision": 4,
    }
    ready.sort(key=lambda node: (priority[node.kind], node.key))
    if session.telemetry is not None:
        session.telemetry.update(
            (len(active), len(ready), len(pending) - len(ready), False)
        )

    def uses_tabpfn(node: JobNode) -> bool:
        if not session.config.tabpfn_enabled:
            return False
        if node.kind in ("bank",):
            return True
        return node.model in ("tabpfn", "targetwise")

    tabpfn_active = any(uses_tabpfn(node) for node in active.values())
    admitted = []
    for node in ready:
        if uses_gpu(session, node):
            if gpu_active >= gpu_capacity:
                continue
            if not gpu_admission(session.config.execution, gpu_active):
                continue
            admitted.append(node)
            gpu_active += 1
            continue
        if available <= 0:
            continue
        if uses_tabpfn(node) and tabpfn_active:
            continue
        admitted.append(node)
        tabpfn_active = tabpfn_active or uses_tabpfn(node)
        available -= 1
    for node in admitted:
        target = session.gpu_executor if uses_gpu(session, node) else executor
        if node.kind in ("search", "fold", "predictor"):
            future = target.submit(run_predictor_job, start_node(session, node))
        else:
            from truss_research.optimizacion.etapas_cpu import start_study_node
            from truss_research.optimizacion.trabajos_estudio import run_study_job

            future = target.submit(run_study_job, start_study_node(session, node))
        active[future] = node
        pending.remove(node)
    if session.telemetry is not None:
        scheduled = len(admitted)
        session.telemetry.update(
            (
                len(active),
                len(ready) - scheduled,
                len(pending) - (len(ready) - scheduled),
                False,
            )
        )


def mark_interrupted(session: StudySession, nodes: tuple[JobNode, ...]) -> None:
    for node in nodes:
        record = node_record(session, node)
        record.status, record.error = (
            "running",
            "Ejecución interrumpida; recursos hasta el último checkpoint",
        )
    session.save()


def run_parallel_predictions(session: StudySession) -> None:
    run_nodes(session, prediction_graph(session))


def run_nodes(session: StudySession, nodes: tuple[JobNode, ...]) -> None:
    for node in nodes:
        record = node_record(session, node)
        if record.status == "complete":
            verify_task(record)
    pending = [
        node for node in nodes if node_record(session, node).status != "complete"
    ]
    if not pending:
        return
    active: dict[Future[JobOutcome], JobNode] = {}
    session.save()
    logger.info(
        "CPU: hasta %s trabajos simultáneos, un hilo por proceso, sin pools anidados",
        session.config.execution.effective_workers,
    )
    try:
        if session.executor is not None:
            execute_queue(session, pending, (session.executor, active))
        else:
            with cpu_pool(session.config.execution) as executor:
                execute_queue(session, pending, (executor, active))
    except BaseException:
        mark_interrupted(session, tuple(active.values()))
        raise
    if any(node_record(session, node).status != "complete" for node in nodes):
        raise RuntimeError("Hay tareas fallidas; reanudar el estudio")


def execute_queue(
    session: StudySession,
    pending: list[JobNode],
    context: tuple[CpuPool, dict[Future[JobOutcome], JobNode]],
) -> None:
    executor, active = context
    gate = MemoryAdmission(session.config.execution)
    interval = session.config.execution.telemetry_interval_seconds
    while pending or active:
        admitted = executor.admit(gate, len(active))
        if admitted:
            schedule_ready(session, pending, context)
        elif session.telemetry is not None:
            lookup = {**session.tasks, **session.preparation_tasks}
            ready = sum(
                all(lookup[key].status == "complete" for key in node.dependencies)
                for node in pending
            )
            session.telemetry.update((len(active), ready, len(pending) - ready, True))
        if not active and not admitted:
            sleep(interval)
            continue
        if not active:
            raise RuntimeError(
                "Hay tareas fallidas o dependencias incompletas; reanudar el estudio"
            )
        completed, _ = wait(active, timeout=interval, return_when=FIRST_COMPLETED)
        for future in completed:
            node = active.pop(future)
            try:
                outcome = future.result()
            except Exception as error:
                logger.exception("Se perdió el trabajador de %s", node.key)
                outcome = JobOutcome(
                    False, 0.0, 0, 0.0, 0.0, f"Trabajador perdido: {error}"
                )
            finish_node(session, node, outcome)
    if session.telemetry is not None:
        session.telemetry.update((0, 0, 0, False))
