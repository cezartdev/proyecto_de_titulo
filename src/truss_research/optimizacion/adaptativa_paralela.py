"""Bloques adaptativos independientes sobre predicciones privadas preparadas una vez."""

import logging
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from functools import lru_cache
from math import ceil
from pathlib import Path
from time import perf_counter, sleep, time
from typing import cast

import joblib
import numpy as np
import pandas as pd

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.recursos import MeasurementParameters
from truss_research.evaluacion.telemetria_cpu import CampaignMonitor
from truss_research.optimizacion.catalogos import SelectionBank
from truss_research.optimizacion.comparacion import evaluate_episode, timed_strategy
from truss_research.optimizacion.comparacion_paralela import ordered_results
from truss_research.optimizacion.contratos import EpisodeSpec, EvaluationSplit
from truss_research.optimizacion.ejecucion_cpu import (
    CpuPool,
    ExecutionParameters,
    MemoryAdmission,
)
from truss_research.optimizacion.etapas_cpu import BLOCKS_PER_WORKER
from truss_research.optimizacion.seleccion_incierta import (
    ENSEMBLE_STRATEGIES,
    ONLINE_STRATEGIES,
    AcquisitionContext,
    EnsembleStrategy,
)
from truss_research.optimizacion.trabajos_predictivos import (
    JobOutcome,
    JobResources,
    run_measured_job,
)
from truss_research.surrogates.configuracion import FloatArray
from truss_research.surrogates.experimentos import write_json

logger = logging.getLogger(__name__)
BANK_FILENAME = "bank.joblib"
RESOURCE_COLUMNS = (
    "cpu_seconds",
    "wall_seconds",
    "initial_rss_mib",
    "peak_rss_mib",
    "cpu_coverage",
    "memory_coverage",
    "finished",
)


@dataclass(frozen=True)
class PreparedAdaptiveBank:
    bank: SelectionBank
    samples: FloatArray
    beta: float
    split: EvaluationSplit
    comparison_signature: str
    online_residual: bool = False

    @property
    def strategies(self) -> tuple[str, ...]:
        return (
            *ENSEMBLE_STRATEGIES,
            *(ONLINE_STRATEGIES if self.online_residual else ()),
        )

    def __post_init__(self) -> None:
        AcquisitionContext(self.bank.space, self.samples, self.beta)
        if self.samples.shape[1:] != (len(self.bank.table), len(OUTPUT_COLUMNS)):
            raise ValueError("El ensemble preparado no corresponde a los candidatos")


@dataclass(frozen=True)
class AdaptiveBlock:
    bank_dir: Path
    bank_signature: str
    indices: tuple[int, ...]
    resources: JobResources


@dataclass
class AdaptiveRun:
    execution: ExecutionParameters
    output_dir: Path
    data_dir: Path
    measurement: MeasurementParameters
    beta: float
    executor: CpuPool | None = None
    telemetry: CampaignMonitor | None = None
    records: list[dict[str, JsonValue]] = field(default_factory=list)
    online_residual: bool = False


def prepare_adaptive_bank(prepared: PreparedAdaptiveBank, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    joblib.dump(prepared, destination / BANK_FILENAME)
    write_json(
        destination / "run.json",
        {
            "bank_signature": source_digest(destination / BANK_FILENAME),
            "comparison_signature": prepared.comparison_signature,
            "split": prepared.split,
            "episodes": len(prepared.bank.episodes),
        },
    )


@lru_cache(maxsize=1)
def load_adaptive_bank(
    root: Path, signature: str
) -> tuple[PreparedAdaptiveBank, float]:
    started = perf_counter()
    if source_digest(root / BANK_FILENAME) != signature:
        raise ValueError("El banco privado adaptativo cambió")
    prepared = joblib.load(root / BANK_FILENAME)
    if not isinstance(prepared, PreparedAdaptiveBank):
        raise ValueError("Formato de banco adaptativo incompatible")
    prepared.__post_init__()
    prepared.samples.setflags(write=False)
    return prepared, perf_counter() - started


def adaptive_episode(
    prepared: PreparedAdaptiveBank, spec: EpisodeSpec
) -> tuple[list[dict[str, JsonValue]], list[dict[str, JsonValue]]]:
    bank = prepared.bank
    indices = bank.table.index.get_indexer(spec.candidate_ids)
    if np.any(indices < 0):
        raise ValueError("Hay candidatos del episodio ausentes en el banco")
    samples = prepared.samples[:, indices, :]
    rows: list[dict[str, JsonValue]] = []
    traces: list[dict[str, JsonValue]] = []
    for name in prepared.strategies:
        beta = prepared.beta if name.endswith("ucb") else 0.0
        strategy, preparation = timed_strategy(
            lambda: EnsembleStrategy(
                AcquisitionContext(
                    bank.space,
                    samples,
                    beta,
                    bank.feature_ranges,
                    name.startswith("online_"),
                )
            )
        )
        summary, trace = evaluate_episode((bank, spec), strategy)
        for row in trace:
            if row["is_initial"]:
                continue
            row.update(
                {
                    f"absolute_error_{phase}_{target}": abs(
                        float(row[target]) - float(row[f"prediction_{phase}_{target}"])
                    )
                    for phase in ("before", "after")
                    for target in OUTPUT_COLUMNS
                }
            )
        identity: dict[str, JsonValue] = {
            "split": prepared.split,
            "episode_id": spec.episode_id,
            "catalog_seed": spec.seed,
            "budget": spec.budget,
            "strategy": name,
            "comparison_id": name,
            "policy_training_seed": None,
            "algorithm": None,
            "use_predictions": True,
            "evidence_phase": "post_hoc_exploratory",
            "beta": beta,
        }
        rows.append(
            {
                **identity,
                **summary,
                "preparation_seconds": preparation,
                "strategy_seconds": preparation + float(summary["decision_seconds"]),
            }
        )
        traces.extend(
            {**identity, **row, "queries_remaining": spec.budget - int(row["query"])}
            for row in trace
        )
    return rows, traces


def adaptive_block(command: AdaptiveBlock) -> None:
    before = load_adaptive_bank.cache_info().misses
    prepared, loading = load_adaptive_bank(command.bank_dir, command.bank_signature)
    miss = load_adaptive_bank.cache_info().misses > before
    if (
        not command.indices
        or len(set(command.indices)) != len(command.indices)
        or any(
            index < 0 or index >= len(prepared.bank.episodes)
            for index in command.indices
        )
    ):
        raise ValueError("Índices de bloque adaptativo inválidos")
    rows: list[dict[str, JsonValue]] = []
    traces: list[dict[str, JsonValue]] = []
    for index in command.indices:
        result, trajectory = adaptive_episode(prepared, prepared.bank.episodes[index])
        rows.extend(result)
        traces.extend(trajectory)
    destination = command.resources.destination
    destination.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(destination / "summary.csv", index=False)
    pd.DataFrame(traces).to_csv(destination / "traces.csv", index=False)
    write_json(
        destination / "run.json",
        {
            "indices": list(command.indices),
            "bank_signature": command.bank_signature,
            "comparison_signature": prepared.comparison_signature,
            "bank_loading_seconds": loading if miss else 0.0,
            "bank_cache_miss": miss,
            "table_signatures": {
                name: source_digest(destination / name)
                for name in ("summary.csv", "traces.csv")
            },
        },
    )


def run_adaptive_block(command: AdaptiveBlock) -> JobOutcome:
    return run_measured_job(command.resources, lambda: adaptive_block(command))


def adaptive_commands(
    prepared: PreparedAdaptiveBank, root: Path, run: AdaptiveRun
) -> tuple[AdaptiveBlock, ...]:
    count = len(prepared.bank.episodes)
    size = max(1, ceil(count / (BLOCKS_PER_WORKER * run.execution.effective_workers)))
    signature = str(read_artifact(root / "run.json")["bank_signature"])
    return tuple(
        AdaptiveBlock(
            root,
            signature,
            tuple(range(start, min(start + size, count))),
            JobResources(
                f"adaptive/{prepared.split}/block_{number:03d}",
                run.output_dir / "blocks" / prepared.split / f"block_{number:03d}",
                run.data_dir,
                run.measurement,
            ),
        )
        for number, start in enumerate(range(0, count, size))
    )


def record_job(run: AdaptiveRun, command: AdaptiveBlock, outcome: JobOutcome) -> None:
    record = next(row for row in run.records if row["key"] == command.resources.key)
    record.update(
        status="complete" if outcome.success else "failed",
        pid=outcome.pid,
        started_at=outcome.started_at,
        finished_at=outcome.finished_at,
        wall_seconds=outcome.seconds,
        error=outcome.error,
    )
    export_worker_tables(run)
    if not outcome.success:
        raise RuntimeError(f"Falló {command.resources.key}: {outcome.error}")
    logger.info(
        "Completado %s (%s episodios)", command.resources.key, len(command.indices)
    )


def submit_block(command: AdaptiveBlock, run: AdaptiveRun) -> Future[JobOutcome]:
    if run.executor is None:
        raise ValueError("Falta el pool adaptativo")
    return run.executor.submit(run_adaptive_block, command)


def execute_adaptive_blocks(
    commands: tuple[AdaptiveBlock, ...], run: AdaptiveRun
) -> None:
    pending = list(commands)
    active: dict[Future[JobOutcome], AdaptiveBlock] = {}
    gate = MemoryAdmission(run.execution)
    interval = run.execution.telemetry_interval_seconds
    run.records.extend(
        {
            "key": command.resources.key,
            "block_dir": command.resources.destination.relative_to(
                run.output_dir
            ).as_posix(),
            "status": "pending",
            "pid": None,
            "submitted_at": None,
            "started_at": None,
            "finished_at": None,
            "wall_seconds": None,
            "error": None,
        }
        for command in commands
    )
    try:
        while pending or active:
            admitted = (
                run.executor.admit(gate, len(active))
                if run.executor is not None
                else gate.allow(len(active))
            )
            capacity = run.executor.capacity if run.executor is not None else 1
            slots = min(len(pending), capacity - len(active)) if admitted else 0
            for _ in range(slots):
                command = pending.pop(0)
                record = next(
                    row for row in run.records if row["key"] == command.resources.key
                )
                record.update(status="running", submitted_at=time())
                if run.executor is None:
                    if run.telemetry is not None:
                        run.telemetry.update((1, len(pending), 0, False))
                    record_job(run, command, run_adaptive_block(command))
                    break
                active[submit_block(command, run)] = command
            if run.telemetry is not None:
                run.telemetry.update((len(active), len(pending), 0, not admitted))
            if not active and pending and not admitted:
                sleep(interval)
                continue
            if not active:
                continue
            completed, _ = wait(active, timeout=interval, return_when=FIRST_COMPLETED)
            for future in completed:
                command = active.pop(future)
                try:
                    outcome = future.result()
                except Exception as error:
                    record = next(
                        row
                        for row in run.records
                        if row["key"] == command.resources.key
                    )
                    record.update(status="failed", error=f"Trabajador perdido: {error}")
                    raise RuntimeError(
                        f"Se perdió el trabajador de {command.resources.key}: {error}"
                    ) from error
                record_job(run, command, outcome)
    finally:
        interrupted = {
            command.resources.key for command in (*pending, *active.values())
        }
        for record in run.records:
            if record["key"] in interrupted or record["status"] == "running":
                record.update(
                    status="interrupted",
                    error="Ejecución interrumpida; revisar logs y checkpoints",
                )
        if run.telemetry is not None:
            run.telemetry.update((0, 0, 0, False))
        export_worker_tables(run)


def export_worker_tables(run: AdaptiveRun) -> dict[str, JsonValue]:
    pd.DataFrame(run.records).to_csv(run.output_dir / "worker_jobs.csv", index=False)
    rows: list[dict[str, JsonValue]] = []
    for job in run.records:
        root = run.output_dir / str(job["block_dir"])
        path = root.with_suffix(".resources.json")
        if not path.is_file():
            rows.append({"key": job["key"], "pid": job["pid"], "status": "missing"})
            continue
        ledger = read_artifact(path)
        phases = cast(list[dict[str, JsonValue]], ledger["phases"])
        total = next(phase for phase in phases if phase["phase"] == "task_total")
        rows.append(
            {
                "key": job["key"],
                "pid": job["pid"],
                "status": ledger["status"],
                **{name: total[name] for name in RESOURCE_COLUMNS},
            }
        )
    pd.DataFrame(rows).to_csv(run.output_dir / "worker_resources.csv", index=False)
    completed_jobs = {job["key"] for job in run.records if job["status"] == "complete"}
    complete = [
        row
        for row in rows
        if row["status"] == "complete" and row["key"] in completed_jobs
    ]
    return {
        "blocks": len(run.records),
        "completed_blocks": len(complete),
        "completed_worker_cpu_seconds_sum": sum(
            float(row["cpu_seconds"]) for row in complete
        ),
        "completed_worker_wall_seconds_sum": sum(
            float(job["wall_seconds"])
            for job in run.records
            if job["status"] == "complete"
        ),
        "max_completed_worker_rss_mib": max(
            (
                float(row["peak_rss_mib"])
                for row in complete
                if row["peak_rss_mib"] is not None
            ),
            default=None,
        ),
        "worker_coverage": "complete"
        if bool(run.records)
        and len(complete) == len(run.records)
        and all(
            row["cpu_coverage"] == "complete"
            and row["memory_coverage"] == "sampled"
            and row["finished"]
            for row in complete
        )
        else "incomplete",
    }


def aggregate_adaptive_blocks(
    prepared: PreparedAdaptiveBank, commands: tuple[AdaptiveBlock, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        not commands
        or source_digest(commands[0].bank_dir / BANK_FILENAME)
        != commands[0].bank_signature
    ):
        raise ValueError("El banco privado cambió antes de consolidar")
    manifests = [
        read_artifact(command.resources.destination / "run.json")
        for command in commands
    ]
    indices = [
        index
        for manifest in manifests
        for index in cast(list[int], manifest["indices"])
    ]
    if sorted(indices) != list(range(len(prepared.bank.episodes))) or any(
        manifest["comparison_signature"] != prepared.comparison_signature
        or manifest["bank_signature"] != command.bank_signature
        or manifest["indices"] != list(command.indices)
        for manifest, command in zip(manifests, commands, strict=True)
    ):
        raise ValueError(
            "Los bloques no cubren exactamente el banco adaptativo firmado"
        )
    for manifest, command in zip(manifests, commands, strict=True):
        signatures = cast(dict[str, str], manifest["table_signatures"])
        if any(
            source_digest(command.resources.destination / name) != signature
            for name, signature in signatures.items()
        ):
            raise ValueError("Cambió una tabla del bloque adaptativo")
    rows = pd.concat(
        [
            pd.read_csv(command.resources.destination / "summary.csv")
            for command in commands
        ],
        ignore_index=True,
    )
    traces = pd.concat(
        [
            pd.read_csv(command.resources.destination / "traces.csv")
            for command in commands
        ],
        ignore_index=True,
    )
    expected = {
        (spec.episode_id, name)
        for spec in prepared.bank.episodes
        for name in prepared.strategies
    }
    actual = list(zip(rows.episode_id, rows.comparison_id, strict=True))
    if (
        set(actual) != expected
        or len(actual) != len(expected)
        or traces.duplicated(["episode_id", "comparison_id", "iter_id"]).any()
    ):
        raise ValueError("Hay episodios o estrategias ausentes o duplicados")
    budgets = {spec.episode_id: spec.budget for spec in prepared.bank.episodes}
    groups = traces.groupby(["episode_id", "comparison_id"])["query"].agg(list)
    if set(groups.index) != expected or any(
        sorted(queries) != list(range(1, budgets[episode] + 1))
        for (episode, _), queries in groups.items()
    ):
        raise ValueError("Las trazas no respetan exactamente cada presupuesto")
    if not rows.queries.eq(rows.episode_id.map(budgets)).all():
        raise ValueError("Los resúmenes no respetan los presupuestos")
    order = (
        [spec.episode_id for spec in prepared.bank.episodes],
        list(prepared.strategies),
    )
    return ordered_results(rows, order), ordered_results(traces, order)
