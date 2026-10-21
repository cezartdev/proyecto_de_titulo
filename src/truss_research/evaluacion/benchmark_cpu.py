"""Cribado y confirmación CPU con presupuesto total; nunca inicia el estudio real."""

import argparse
import hashlib
import json
import logging
import os
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import monotonic, perf_counter

import numpy as np
import pandas as pd
import psutil

from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.carga_cpu import (
    CpuCalibration,
    CpuWorkResult,
    WarmupCommand,
    execute_cpu_work,
    load_cpu_protocol,
    warm_cpu_worker,
    work_batch,
)
from truss_research.evaluacion.informe_estudio import markdown_table
from truss_research.evaluacion.telemetria_cpu import CampaignMonitor
from truss_research.optimizacion.ejecucion_cpu import (
    WORKER_SHUTDOWN_SECONDS,
    CpuPool,
    ExecutionParameters,
    cpu_pool,
)
from truss_research.optimizacion.estudio_estado import (
    environment_versions,
    source_signature,
)
from truss_research.surrogates.experimentos import write_json

DEFAULT_EXPORTS = Path("reports/pruebas_cpu")
logger = logging.getLogger(__name__)


@dataclass
class CpuProfile:
    stage: str
    workers: int
    rows: int
    status: str = "running"
    startup_seconds: float = 0.0
    durations: list[float] = field(default_factory=list)
    samples: list[CpuWorkResult] = field(default_factory=list)
    error: str | None = None

    @property
    def median_seconds(self) -> float:
        return float(np.median(self.durations)) if self.durations else float("inf")


@dataclass
class CalibrationRun:
    protocol: CpuCalibration
    directory: Path
    deadline: float
    references: dict[int, dict[tuple[str, int], tuple[float, ...]]] = field(
        default_factory=dict
    )


def collect_futures[T](
    futures: list[Future[T]], context: tuple[float, CampaignMonitor, CpuCalibration]
) -> list[T]:
    deadline, monitor, protocol = context
    active = set(futures)
    results: list[T] = []
    while active:
        if monotonic() >= deadline:
            raise TimeoutError(
                "Presupuesto de calibración agotado; pasada incompleta descartada"
            )
        if psutil.virtual_memory().percent > protocol.memory_limit_fraction * 100:
            raise RuntimeError("El perfil excedió el límite de RAM del sistema")
        monitor.update(
            (
                min(len(active), monitor.worker_limit),
                max(0, len(active) - monitor.worker_limit),
                0,
                False,
            )
        )
        completed, active = wait(
            active,
            timeout=min(1.0, max(0.0, deadline - monotonic())),
            return_when=FIRST_COMPLETED,
        )
        results.extend(future.result() for future in completed)
    monitor.update((0, 0, 0, False))
    return results


def verify_results(
    run: CalibrationRun, rows: int, samples: list[CpuWorkResult]
) -> None:
    values = {(sample.kind, sample.seed): sample.values for sample in samples}
    if len(values) != len(samples) or len(samples) != len(
        work_batch(run.protocol, rows, 0)
    ):
        raise ValueError("La pasada no contiene exactamente los trabajos previstos")
    reference = run.references.setdefault(rows, values)
    for key, observed in values.items():
        if not np.isfinite(observed).all() or not np.allclose(
            observed, reference[key], rtol=1e-10, atol=1e-12
        ):
            raise ValueError(f"Resultados CPU diferentes o no finitos: {key}")


def profile_rounds(
    run: CalibrationRun,
    profile: CpuProfile,
    context: tuple[CpuPool, CampaignMonitor, int],
) -> None:
    executor, monitor, repeats = context
    for repetition in range(repeats):
        if monotonic() >= run.deadline:
            raise TimeoutError("Presupuesto agotado antes de la siguiente pasada")
        start = perf_counter()
        futures = [
            executor.submit(execute_cpu_work, work)
            for work in work_batch(run.protocol, profile.rows, repetition)
        ]
        results = collect_futures(futures, (run.deadline, monitor, run.protocol))
        seconds = perf_counter() - start
        verify_results(run, profile.rows, results)
        profile.samples.extend(results)
        profile.durations.append(seconds)
        write_json(
            run.directory / "progress.json",
            {
                "stage": profile.stage,
                "workers": profile.workers,
                "completed_passes": len(profile.durations),
                "last_seconds": seconds,
            },
        )


def run_profile(run: CalibrationRun, profile: CpuProfile, repeats: int) -> CpuProfile:
    path = run.directory / f"{profile.stage}_{profile.workers}"
    path.mkdir()
    warm_directory = path / "warmup"
    warm_directory.mkdir()
    monitor = CampaignMonitor(path / "telemetry.csv")
    monitor.worker_limit = profile.workers
    start = perf_counter()
    try:
        if psutil.virtual_memory().percent > run.protocol.memory_limit_fraction * 100:
            raise RuntimeError("RAM insuficiente para iniciar el perfil")
        with (
            monitor,
            cpu_pool(ExecutionParameters(cpu_workers=profile.workers)) as executor,
        ):
            command = WarmupCommand(
                warm_directory, profile.workers, run.deadline, run.protocol
            )
            pids = collect_futures(
                [
                    executor.submit(warm_cpu_worker, command)
                    for _ in range(profile.workers)
                ],
                (run.deadline, monitor, run.protocol),
            )
            if len(set(pids)) != profile.workers:
                raise ValueError("No se calentaron todos los trabajadores")
            profile.startup_seconds = perf_counter() - start
            profile_rounds(run, profile, (executor, monitor, repeats))
        if (
            pd.read_csv(path / "telemetry.csv").system_ram_percent.max()
            > run.protocol.memory_limit_fraction * 100
        ):
            raise RuntimeError("La telemetría registró un exceso de RAM del sistema")
        profile.status = "complete"
    except (TimeoutError, RuntimeError, ValueError, OSError) as error:
        profile.status, profile.error = "incomplete", f"{type(error).__name__}: {error}"
        logger.warning(
            "Perfil %s/%s descartado: %s", profile.stage, profile.workers, error
        )
    write_json(path / "profile.json", asdict(profile))
    return profile


def select_workers(
    profiles: list[CpuProfile], protocol: CpuCalibration
) -> tuple[int, bool]:
    eligible = [
        profile
        for profile in profiles
        if profile.stage == "confirmation"
        and profile.status == "complete"
        and len(profile.durations) == protocol.confirmation_repeats
    ]
    if len(eligible) < 2:
        return min(4, os.cpu_count() or 1), True
    fastest = min(profile.median_seconds for profile in eligible)
    candidates = [
        profile.workers
        for profile in eligible
        if profile.median_seconds <= fastest * (1 + protocol.selection_tolerance)
    ]
    return min(candidates), False


def export_calibration(
    run: CalibrationRun, profiles: list[CpuProfile]
) -> tuple[int, bool]:
    rows = [
        {
            "stage": profile.stage,
            "workers": profile.workers,
            "rows": profile.rows,
            "status": profile.status,
            "startup_seconds": profile.startup_seconds,
            "completed_passes": len(profile.durations),
            "median_effective_seconds": profile.median_seconds
            if profile.durations
            else None,
            "error": profile.error,
        }
        for profile in profiles
    ]
    table = pd.DataFrame(
        rows,
        columns=[
            "stage",
            "workers",
            "rows",
            "status",
            "startup_seconds",
            "completed_passes",
            "median_effective_seconds",
            "error",
        ],
    )
    table.to_csv(run.directory / "comparison.csv", index=False)
    recommended, provisional = select_workers(profiles, run.protocol)
    recommendation = {
        "cpu_workers": recommended,
        "provisional": provisional,
        "criterion": "least_workers_within_5_percent_of_fastest_complete_confirmation",
        "research_training_started": False,
        "memory_limit_fraction": run.protocol.memory_limit_fraction,
    }
    write_json(run.directory / "recommendation.json", recommendation)
    (run.directory / "RESULTS.md").write_text(
        "# Calibración rápida de CPU\n\nDatos sintéticos: cribado de 256 filas y confirmación de 512. Cada pasada contiene trabajos reales RF, CatBoost, GPR, MLP y DQN. Un hilo numérico por proceso; geometría neuronal de anchura 64, lote 64 y 32 candidatos. El calentamiento de todos los trabajadores y arranque se separan del trabajo efectivo.\n\n"
        + markdown_table(table)
        + f"\n\nRecomendación: {recommended} procesos. Provisional: {provisional}. Las pasadas incompletas se excluyen; confirmar exige dos perfiles con todas sus repeticiones. No se comparan tiempos de tamaños distintos. Las repeticiones temporales no son evidencia predictiva de la tesis.\n\nLa recomendación caracteriza esta carga reducida y no garantiza el óptimo del estudio completo. La RAM observada usa modelos pequeños; la campaña conserva su límite preventivo y telemetría. Ejecutar make study manualmente con un destino nuevo; este comando no modifica el protocolo principal ni inicia investigación.\n",
        encoding="utf-8",
    )
    return recommended, provisional


def next_destination() -> Path:
    numbers = [
        int(path.name.removeprefix("benchmark_"))
        for path in DEFAULT_EXPORTS.glob("benchmark_*")
        if path.name.removeprefix("benchmark_").isdigit()
    ]
    return DEFAULT_EXPORTS / f"benchmark_{max(numbers, default=0) + 1:03d}"


def protocol_digest(protocol: CpuCalibration, path: Path | None) -> str:
    if path is not None:
        return source_digest(path)
    return hashlib.sha256(
        json.dumps(asdict(protocol), sort_keys=True).encode("utf-8")
    ).hexdigest()


def run_cpu_benchmark(protocol_path: Path | None, destination: Path) -> Path:
    protocol = load_cpu_protocol(protocol_path)
    if destination.exists():
        raise ValueError("Elegir un destino nuevo para calibración CPU")
    destination.mkdir(parents=True)
    start = monotonic()
    deadline = start + protocol.budget_seconds - 2 * WORKER_SHUTDOWN_SECONDS
    run = CalibrationRun(
        protocol, destination, min(deadline, start + protocol.screen_seconds)
    )
    metadata: dict[str, JsonValue] = {
        "status": "running",
        "source_signature": source_signature(),
        "protocol_signature": protocol_digest(protocol, protocol_path),
        "protocol": asdict(protocol),
        "versions": environment_versions(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_mib": psutil.virtual_memory().total / 1024**2,
        "data": "synthetic_only",
        "research_training_started": False,
    }
    write_json(destination / "provenance.json", metadata)
    profiles: list[CpuProfile] = []
    workers = sorted({min(value, os.cpu_count() or 1) for value in protocol.workers})
    try:
        for count in workers:
            if monotonic() >= run.deadline:
                break
            logger.info(
                "Cribado CPU: %s procesos, %s filas", count, protocol.screen_rows
            )
            profiles.append(
                run_profile(run, CpuProfile("screen", count, protocol.screen_rows), 1)
            )
        finalists = sorted(
            (profile for profile in profiles if profile.status == "complete"),
            key=lambda profile: (profile.median_seconds, profile.workers),
        )[:2]
        run.deadline = deadline
        for finalist in finalists:
            if monotonic() >= run.deadline:
                break
            logger.info(
                "Confirmación CPU: %s procesos, %s filas",
                finalist.workers,
                protocol.confirmation_rows,
            )
            profiles.append(
                run_profile(
                    run,
                    CpuProfile(
                        "confirmation", finalist.workers, protocol.confirmation_rows
                    ),
                    protocol.confirmation_repeats,
                )
            )
        recommended, provisional = export_calibration(run, profiles)
        if (
            source_signature() != metadata["source_signature"]
            or protocol_digest(protocol, protocol_path)
            != metadata["protocol_signature"]
        ):
            raise ValueError("El código o protocolo cambió durante calibración")
        metadata.update(
            status="complete", recommended_workers=recommended, provisional=provisional
        )
    except BaseException as error:
        metadata["status"], metadata["error"] = "failed", str(error)
        raise
    finally:
        metadata["elapsed_seconds"] = monotonic() - start
        metadata["signatures"] = {
            path.relative_to(destination).as_posix(): source_digest(path)
            for path in destination.rglob("*")
            if path.is_file() and path.name != "provenance.json"
        }
        write_json(destination / "provenance.json", metadata)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibración sintética CPU de hasta cinco minutos"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--destino", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run_cpu_benchmark(args.config, args.destino or next_destination()))


if __name__ == "__main__":
    main()
