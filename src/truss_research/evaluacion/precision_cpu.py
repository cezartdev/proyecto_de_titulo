"""CPU agrupada: conserva la latencia individual y declara granularidad y alcance."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import get_clock_info, perf_counter, process_time

from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.recursos import MeasurementParameters


@dataclass(frozen=True)
class CpuClock:
    nominal_resolution: float
    observed_quantum: float | None
    implementation: str


def probe_cpu_clock(seconds: float) -> CpuClock:
    info = get_clock_info("process_time")
    previous = process_time()
    deadline = perf_counter() + seconds
    increments: list[float] = []
    while perf_counter() < deadline:
        current = process_time()
        if current > previous:
            increments.append(current - previous)
        previous = current
    return CpuClock(
        info.resolution,
        max(info.resolution, min(increments)) if increments else None,
        info.implementation,
    )


def cpu_batches(
    operation: Callable[[], None], parameters: MeasurementParameters, clock: CpuClock
) -> list[dict[str, JsonValue]]:
    target = max(
        parameters.cpu_min_seconds,
        parameters.cpu_min_ticks
        * (clock.observed_quantum or parameters.cpu_min_seconds),
    )
    samples: list[dict[str, JsonValue]] = []
    for repetition in range(parameters.cpu_repeats):
        start_wall, start_cpu = perf_counter(), process_time()
        calls, cpu = 0, 0.0
        while (
            calls == 0 or perf_counter() - start_wall < parameters.cpu_max_wall_seconds
        ):
            operation()
            calls += 1
            cpu = process_time() - start_cpu
            if cpu >= target:
                break
        resolved = clock.observed_quantum is not None and cpu >= target
        samples.append(
            {
                "repetition": repetition,
                "calls": calls,
                "cpu_seconds": cpu,
                "wall_seconds": perf_counter() - start_wall,
                "cpu_per_call_seconds": cpu / calls if resolved else None,
                "observed_quantum_seconds": clock.observed_quantum,
                "nominal_clock_resolution_seconds": clock.nominal_resolution,
                "clock_implementation": clock.implementation,
                "target_cpu_seconds": target,
                "status": "resolved"
                if resolved
                else "insufficient_clock_resolution_or_budget",
                "scope_note": "includes Python call loop; no concurrent memory monitor",
            }
        )
    return samples


def export_cpu_summary(destination: Path) -> None:
    import pandas as pd

    samples = pd.concat(
        [
            pd.read_csv(destination / name)
            for name in ("inference_cpu_batches.csv", "selection_cpu_batches.csv")
        ],
        ignore_index=True,
    )
    keys = [
        name
        for name in (
            "operation",
            "scope",
            "split",
            "model",
            "seed",
            "batch_size",
            "catalog_size",
            "comparison_id",
            "budget",
        )
        if name in samples
    ]
    rows: list[dict[str, JsonValue]] = []
    for identity, group in samples.groupby(keys, dropna=False):
        resolved = bool(group.status.eq("resolved").all())
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "cpu_per_call_seconds": float(group.cpu_per_call_seconds.median())
                if resolved
                else None,
                "calls": int(group.calls.sum()),
                "repetitions": len(group),
                "status": "resolved" if resolved else "incomplete_precision",
                "interpretation": "grouped_CPU_includes_call_loop_and_declared_operation_scope",
            }
        )
    pd.DataFrame(rows).to_csv(destination / "cpu_operation_summary.csv", index=False)
