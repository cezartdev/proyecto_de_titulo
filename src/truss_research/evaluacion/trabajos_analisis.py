"""Pool CPU y recibos firmados para reanudar análisis sin repetir ajustes completos."""

import hashlib
import json
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from pathlib import Path
from time import sleep
from typing import Protocol

import pandas as pd

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.procesados import JsonValue
from truss_research.optimizacion.ejecucion_cpu import (
    MemoryAdmission,
    cpu_pool,
    gpu_admission,
)
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    artifact_signatures,
    source_signature,
)
from truss_research.optimizacion.trabajos_predictivos import JobOutcome
from truss_research.surrogates.experimentos import data_signature, write_json

RECEIPT_NAME = "completion.json"
SESSION_NAME = "analysis_session.json"


class AnalysisJob(Protocol):
    config: StudyConfig
    destination: Path


def analysis_session(config: StudyConfig, destination: Path) -> None:
    expected = {
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "source_signature": source_signature(),
        "dataset_signature": data_signature(config.data_dir),
    }
    marker = destination / SESSION_NAME
    if destination.exists():
        if (
            not marker.is_file()
            or json.loads(marker.read_text(encoding="utf-8")) != expected
        ):
            raise ValueError(
                "El análisis existente no corresponde a configuración, código y datos actuales"
            )
        return
    destination.mkdir(parents=True)
    marker.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def job_identity(job: AnalysisJob) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "job": asdict(job),
                "source": source_signature(),
                "dataset": data_signature(job.config.data_dir),
            },
            default=str,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def cached_job(job: AnalysisJob) -> dict[str, JsonValue] | None:
    receipt = job.destination / RECEIPT_NAME
    if not receipt.is_file():
        return None
    saved = read_artifact(receipt)
    measured = {
        name: digest
        for name, digest in artifact_signatures(job.destination).items()
        if name != RECEIPT_NAME
    }
    if (
        saved.get("identity") != job_identity(job)
        or saved.get("signatures") != measured
    ):
        raise ValueError(f"El trabajo completo cambió: {job.destination}")
    for name, digest in saved.get("sidecars", {}).items():
        path = job.destination.parent / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"El registro del trabajo cambió: {path}")
    return saved["outcome"]


def preserve_partial(destination: Path) -> None:
    if not destination.exists():
        return
    root = destination.parent.resolve()
    if destination.resolve().parent != root:
        raise ValueError("La tarea parcial debe estar dentro del análisis")
    number = 1
    while (root / f"{destination.name}_failed_{number:03d}").exists():
        number += 1
    target = root / f"{destination.name}_failed_{number:03d}"
    destination.rename(target)
    for suffix in (".log", ".resources.json", ".execution.json"):
        source = destination.with_suffix(suffix)
        if source.is_file():
            source.rename(target / source.name)


def record_job(job: AnalysisJob, outcome: JobOutcome) -> None:
    if not outcome.success:
        raise RuntimeError(outcome.error)
    sidecars = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in job.destination.parent.glob(job.destination.name + ".*")
        if path.is_file()
    }
    write_json(
        job.destination / RECEIPT_NAME,
        {
            "identity": job_identity(job),
            "signatures": artifact_signatures(job.destination),
            "sidecars": sidecars,
            "outcome": asdict(outcome),
        },
    )


def execute_analysis_jobs[JobType: AnalysisJob](
    jobs: Sequence[JobType],
    operation: Callable[[JobType], JobOutcome],
    destination: Path,
) -> None:
    if not jobs:
        return
    for job in jobs:
        job.config.execution.validate_campaign()
    destination.mkdir(parents=True, exist_ok=True)
    pending, records = [], []
    for job in jobs:
        cached = cached_job(job)
        if cached is not None:
            records.append({"job": job.destination.name, "reused": True, **cached})
            continue
        preserve_partial(job.destination)
        pending.append(job)
    config = jobs[0].config
    gate = MemoryAdmission(config.execution)
    active: dict[Future[JobOutcome], JobType] = {}
    hybrid = (config.tabpfn_enabled and config.tabpfn.device == "cuda") or bool(
        config.execution.gpu_models
    )
    if hybrid:
        execute_hybrid_jobs(pending, operation, (destination, records))
        pending = []
    pool = nullcontext(None) if hybrid else cpu_pool(config.execution)
    with pool as executor:
        while pending or active:
            prefix = next(
                (i for i, job in enumerate(pending) if getattr(job, "serial", False)),
                len(pending),
            )
            capacity = prefix if prefix else int(not active and bool(pending))
            admitted = (
                executor.admit(gate, len(active))
                if executor is not None
                else gate.allow(len(active))
            )
            worker_capacity = executor.capacity if executor is not None else 1
            slots = min(capacity, worker_capacity - len(active)) if admitted else 0
            for _ in range(slots):
                job = pending.pop(0)
                if executor is None:
                    outcome = operation(job)
                    records.append(
                        {
                            "job": job.destination.name,
                            "reused": False,
                            **asdict(outcome),
                        }
                    )
                    pd.DataFrame(records).to_csv(
                        destination / "worker_jobs.csv", index=False
                    )
                    record_job(job, outcome)
                    break
                active[executor.submit(operation, job)] = job
            if active:
                completed, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    job = active.pop(future)
                    outcome = future.result()
                    records.append(
                        {
                            "job": job.destination.name,
                            "reused": False,
                            **asdict(outcome),
                        }
                    )
                    pd.DataFrame(records).to_csv(
                        destination / "worker_jobs.csv", index=False
                    )
                    record_job(job, outcome)
            if pending and not slots and not active:
                sleep(1)
    pd.DataFrame(records).to_csv(destination / "worker_jobs.csv", index=False)
    phases = []
    for job in jobs:
        resource = read_artifact(job.destination.with_suffix(".resources.json"))
        phases.extend(
            {"job": job.destination.name, **phase} for phase in resource["phases"]
        )
    pd.DataFrame(phases).to_csv(destination / "resource_phases.csv", index=False)


def execute_hybrid_jobs[JobType: AnalysisJob](
    pending: list[JobType],
    operation: Callable[[JobType], JobOutcome],
    context: tuple[Path, list[dict[str, JsonValue]]],
) -> None:
    if not pending:
        return
    destination, records = context
    config = pending[0].config
    gate = MemoryAdmission(config.execution)
    active: dict[Future[JobOutcome], tuple[JobType, str]] = {}
    with ExitStack() as stack:
        pools = {
            "cpu": stack.enter_context(cpu_pool(config.execution)),
            "cuda": stack.enter_context(cpu_pool(config.execution, "cuda")),
        }
        while pending or active:
            admitted = pools["cpu"].admit(gate, len(active))
            for job in list(pending) if admitted else ():
                gpu_model = getattr(job, "model", None) in config.execution.gpu_models
                gpu_tabpfn = config.tabpfn.device == "cuda" and getattr(
                    job, "serial", False
                )
                device = "cuda" if gpu_model or gpu_tabpfn else "cpu"
                occupied = sum(kind == device for _, kind in active.values())
                if occupied >= pools[device].capacity:
                    continue
                if device == "cuda" and not gpu_admission(config.execution, occupied):
                    continue
                active[pools[device].submit(operation, job)] = (job, device)
                pending.remove(job)
            if not active:
                sleep(config.execution.telemetry_interval_seconds)
                continue
            completed, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                job, device = active.pop(future)
                outcome = future.result()
                record_job(job, outcome)
                records.append(
                    {
                        "job": job.destination.name,
                        "device": device,
                        "reused": False,
                        **asdict(outcome),
                    }
                )
                pd.DataFrame(records).to_csv(
                    destination / "worker_jobs.csv", index=False
                )
