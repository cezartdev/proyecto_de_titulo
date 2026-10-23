"""Políticas independientes por proceso, con checkpoints y costos por intento."""

import logging
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter, sleep, time

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.telemetria_cpu import CampaignMonitor
from truss_research.optimizacion.busqueda import SearchContext
from truss_research.optimizacion.ejecucion_cpu import MemoryAdmission, cpu_pool
from truss_research.optimizacion.entrenamiento_propuestas import (
    ProposalTrainingConfig,
    read_proposal_checkpoint,
    train_proposal_policy,
)
from truss_research.optimizacion.estrategias_busqueda import STRATEGY_DEFINITIONS
from truss_research.optimizacion.estudio_estado import StudyConfig
from truss_research.optimizacion.trabajos_predictivos import (
    JobOutcome,
    JobResources,
    run_measured_job,
)
from truss_research.surrogates.experimentos import write_json

EXECUTION_DIRECTORY = "evolution/policy_execution"
PROGRESS_SAVE_SECONDS = 10.0
logger = logging.getLogger(__name__)


def save_progress(path: Path, state: dict[str, JsonValue], last_save: float) -> float:
    """Difiere la telemetría bloqueada; los cambios de estado siguen siendo obligatorios."""
    now = perf_counter()
    if now - last_save < PROGRESS_SAVE_SECONDS:
        return last_save
    try:
        write_json(path, state)
    except PermissionError:
        logger.warning("No se pudo actualizar el progreso de %s; se reintentará", path)
    return perf_counter()


@dataclass(frozen=True)
class PolicyJob:
    context: SearchContext
    config: ProposalTrainingConfig
    resources: JobResources


def policy_jobs(
    config: StudyConfig, context: SearchContext, signature: str
) -> dict[str, PolicyJob]:
    from truss_research.evaluacion.busqueda_evolutiva import policy_configuration

    return {
        f"{name}/{seed}": PolicyJob(
            replace(context, assisted=STRATEGY_DEFINITIONS[name].assisted),
            policy_configuration(config, (signature, seed, name)),
            JobResources(
                f"{name}/{seed}",
                config.output_dir / "evolution/policies" / name / str(seed),
                config.data_dir,
                config.measurement,
            ),
        )
        for name in config.search.policies
        for seed in config.seeds
    }


def execution_state(
    path: Path, jobs: dict[str, PolicyJob], identity: tuple[str, int]
) -> dict[str, JsonValue]:
    signature, workers = identity
    if path.exists():
        state = read_artifact(path)
        if (
            state["signature"] != signature
            or state["workers"] != workers
            or set(state["jobs"]) != set(jobs)
        ):
            raise ValueError(
                "Cambió la ejecución de políticas; elegir un destino nuevo"
            )
        for run in state["runs"]:
            if (
                run.get("telemetry_signature")
                and source_digest(Path(run["telemetry_file"]))
                != run["telemetry_signature"]
            ):
                raise ValueError("Cambió la telemetría de políticas")
        for record in state["jobs"].values():
            for attempt in record["attempts"]:
                for name in ("measurement_signatures", "log_signatures"):
                    if any(
                        not Path(file).is_file() or source_digest(Path(file)) != digest
                        for file, digest in attempt.get(name, {}).items()
                    ):
                        raise ValueError("Cambió un recurso o log de políticas")
        return state
    return {
        "signature": signature,
        "workers": workers,
        "threads_per_worker": 1,
        "job_duration_scope": "worker_operation_including_training_validation_checkpoint_io",
        "elapsed_scope": "coordinator_invocations_including_startup_and_shutdown",
        "jobs": {key: {"status": "pending", "attempts": []} for key in jobs},
        "runs": [],
    }


def pending_policies(
    jobs: dict[str, PolicyJob], state: dict[str, JsonValue]
) -> list[str]:
    pending = []
    for key, job in jobs.items():
        metadata = job.resources.destination / "run.json"
        if metadata.exists() and read_artifact(metadata).get("phase") == "complete":
            checkpoint = read_proposal_checkpoint(job.resources.destination, job.config)
            if checkpoint.step != job.config.parameters.transitions:
                raise ValueError(f"La política {key} tiene un presupuesto incompleto")
            state["jobs"][key]["status"] = "complete"
        elif state["jobs"][key]["status"] == "complete":
            raise ValueError(f"Falta la política completa {key}")
        else:
            pending.append(key)
    return pending


def update_progress(jobs: dict[str, PolicyJob], state: dict[str, JsonValue]) -> None:
    for key, job in jobs.items():
        path = job.resources.destination / "run.json"
        if path.exists():
            state["jobs"][key]["completed_steps"] = read_artifact(path)[
                "completed_steps"
            ]
        record = state["jobs"][key]
        if record["attempts"]:
            execution = Path(record["attempts"][-1]["log_files"][-1])
            if execution.exists():
                record["pid"] = read_artifact(execution).get("pid")


def finish_attempt(record: dict[str, JsonValue], outcome: JobOutcome) -> None:
    record["status"] = "complete" if outcome.success else "failed"
    record["error"] = outcome.error
    record["attempts"][-1].update(
        {
            "status": record["status"],
            "pid": outcome.pid,
            "started_at": outcome.started_at,
            "finished_at": outcome.finished_at,
            "wall_seconds": outcome.seconds,
        }
    )


def run_policy_campaign(
    config: StudyConfig, context: SearchContext, identity: tuple[str, bool]
) -> None:
    signature, gpu_evaluator = identity
    execution = (
        replace(config.execution, cpu_workers=1) if gpu_evaluator else config.execution
    )
    jobs = policy_jobs(config, context, signature)
    root = config.output_dir / EXECUTION_DIRECTORY
    path = root / "manifest.json"
    state = execution_state(path, jobs, (signature, execution.effective_workers))
    pending = pending_policies(jobs, state)
    if not pending:
        return
    root.mkdir(parents=True, exist_ok=True)
    number = len(state["runs"]) + 1
    telemetry = root / f"run_{number:03d}.telemetry.csv"
    run = {
        "status": "running",
        "started_at": time(),
        "wall_seconds": 0.0,
        "telemetry_file": str(telemetry),
        "workers": execution.effective_workers,
        "gpu_evaluator": gpu_evaluator,
    }
    state["runs"].append(run)
    active: dict[Future[JobOutcome], str] = {}
    pool = None
    started = perf_counter()
    write_json(path, state)
    last_save = perf_counter()
    try:
        with (
            CampaignMonitor(
                telemetry,
                execution.telemetry_interval_seconds,
                execution.effective_workers,
            ) as monitor,
            cpu_pool(execution, "cuda" if gpu_evaluator else "cpu") as pool,
        ):
            gate = MemoryAdmission(execution)
            while pending or active:
                submitted = False
                admitted = pool.admit(gate, len(active))
                while pending and len(active) < pool.capacity and admitted:
                    key = pending.pop(0)
                    record = state["jobs"][key]
                    record["status"] = "running"
                    record["attempts"].append(
                        {"status": "running", "started_at": time()}
                    )
                    attempt = (
                        root / "jobs" / key / f"attempt_{len(record['attempts']):03d}"
                    )
                    resources = replace(jobs[key].resources, destination=attempt)
                    record["attempts"][-1]["measurement_files"] = [
                        str(attempt.with_suffix(".resources.json"))
                    ]
                    record["attempts"][-1]["log_files"] = [
                        str(attempt.with_suffix(".log")),
                        str(attempt.with_suffix(".execution.json")),
                    ]
                    active[pool.submit(run_policy_attempt, (jobs[key], resources))] = (
                        key
                    )
                    submitted = True
                    admitted = pool.admit(gate, len(active))
                monitor.update((len(active), len(pending), 0, not admitted))
                update_progress(jobs, state)
                if submitted:
                    write_json(path, state)
                    last_save = perf_counter()
                last_save = save_progress(path, state, last_save)
                if not active:
                    sleep(execution.telemetry_interval_seconds)
                    continue
                completed, _ = wait(
                    active,
                    timeout=execution.telemetry_interval_seconds,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    key = active.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as error:
                        outcome = JobOutcome(
                            False,
                            0.0,
                            0,
                            time(),
                            time(),
                            f"Trabajador perdido: {error}",
                        )
                    finish_attempt(state["jobs"][key], outcome)
                    write_json(path, state)
                    if not outcome.success:
                        raise RuntimeError(
                            f"Falló la política {key}: {outcome.error}; reanudar el estudio"
                        )
            monitor.update((0, 0, 0, False))
        run["status"] = "complete"
    except BaseException:
        run["status"] = "interrupted"
        for key in active.values():
            state["jobs"][key]["status"] = "interrupted"
            state["jobs"][key]["attempts"][-1]["status"] = "interrupted"
        raise
    finally:
        run["wall_seconds"] = perf_counter() - started
        run["finished_at"] = time()
        if pool is not None:
            run["final_worker_capacity"] = pool.capacity
            run["memory_recoveries"] = pool.recoveries
        if telemetry.exists():
            run["telemetry_signature"] = source_digest(telemetry)
        update_progress(jobs, state)
        for record in state["jobs"].values():
            for attempt in record["attempts"]:
                for collection, signatures in (
                    ("measurement_files", "measurement_signatures"),
                    ("log_files", "log_signatures"),
                ):
                    attempt[signatures] = {
                        str(Path(file)): source_digest(Path(file))
                        for file in attempt[collection]
                        if Path(file).is_file()
                    }
        write_json(path, state)


def run_policy_attempt(command: tuple[PolicyJob, JobResources]) -> JobOutcome:
    job, resources = command
    return run_measured_job(
        resources,
        lambda: execute_policy(job, resources),
    )


def execute_policy(job: PolicyJob, resources: JobResources) -> None:
    destination = job.resources.destination
    if destination.exists() and not (destination / "run.json").exists():
        archive = resources.destination.with_suffix(".initialization")
        workspace = destination.parents[3].resolve()
        if not destination.resolve().is_relative_to(
            workspace
        ) or not archive.resolve().is_relative_to(workspace):
            raise ValueError("La inicialización parcial salió del estudio")
        destination.rename(archive)
    train_proposal_policy(job.context, job.config, destination)
