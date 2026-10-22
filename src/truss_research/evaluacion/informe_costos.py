"""Tablas de recursos, costos de uso y amortización sin duplicar dependencias."""

from pathlib import Path
from typing import cast

import pandas as pd

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.recursos import valid_seconds
from truss_research.optimizacion.estudio_estado import load_study_config, read_study

RESOURCE_COLUMNS = (
    "device",
    "gpu_name",
    "gpu_allocated_mib",
    "gpu_reserved_mib",
    "gpu_peak_allocated_mib",
    "gpu_peak_reserved_mib",
    "gpu_peak_scope",
    "task",
    "attempt",
    "phase",
    "status",
    "wall_seconds",
    "cpu_seconds",
    "initial_rss_mib",
    "peak_rss_mib",
    "peak_increment_mib",
    "memory_interval_ms",
    "samples",
    "cpu_scope",
    "cpu_coverage",
    "cpu_tree_coverage",
    "coverage_policy_version",
    "coverage",
)


def resource_inventory(study: Path) -> pd.DataFrame:
    state = read_study(study)
    rows: list[dict[str, JsonValue]] = []
    records = [
        *cast(list[dict[str, JsonValue]], state["tasks"]),
        *cast(list[dict[str, JsonValue]], state.get("preparation_tasks", [])),
    ]
    for task in records:
        files = cast(list[str], task.get("measurement_files", []))
        signatures = cast(dict[str, str], task.get("measurement_signatures", {}))
        for number, path in enumerate(files, start=1):
            file = Path(path)
            if not file.is_file():
                rows.append(
                    {
                        "task": task["key"],
                        "attempt": number,
                        "phase": "task_total",
                        "coverage": "missing",
                    }
                )
                continue
            if path in signatures and source_digest(file) != signatures[path]:
                raise ValueError("Cambió un registro de recursos del estudio")
            payload = read_artifact(file)
            for phase in cast(list[dict[str, JsonValue]], payload["phases"]):
                complete = (
                    phase["finished"] is True
                    and not phase.get("errors")
                    and phase["cpu_coverage"] == "complete"
                    and phase["memory_coverage"] == "sampled"
                    and all(
                        valid_seconds(float(phase[name]))
                        for name in ("wall_seconds", "cpu_seconds")
                    )
                    and path in signatures
                )
                rows.append(
                    {
                        "task": task["key"],
                        "attempt": number,
                        **phase,
                        "coverage": "complete" if complete else "incomplete",
                    }
                )
        if len(files) != len(cast(list[str], task["attempts"])):
            rows.append(
                {
                    "task": task["key"],
                    "attempt": None,
                    "phase": "task_total",
                    "coverage": "missing",
                }
            )
    return pd.DataFrame(rows, columns=[*RESOURCE_COLUMNS, "errors", "children_seen"])


def export_resource_report(study: Path, destination: Path) -> bool:
    table = resource_inventory(study)
    table.to_csv(destination / "resource_phases.csv", index=False)
    totals = table.loc[table.phase.eq("task_total")]
    summary = totals.groupby("task", as_index=False).agg(
        wall_seconds=("wall_seconds", lambda values: values.sum(min_count=1)),
        cpu_seconds=("cpu_seconds", lambda values: values.sum(min_count=1)),
        peak_rss_mib=("peak_rss_mib", "max"),
        attempts=("attempt", "count"),
    )
    summary.to_csv(destination / "resource_tasks.csv", index=False)
    state = read_study(study)
    runs = cast(list[dict[str, JsonValue]], state.get("execution_runs", []))
    jobs = [
        {"task": task["key"], "attempt": number, **event}
        for task in [*state["tasks"], *state.get("preparation_tasks", [])]
        for number, event in enumerate(task.get("worker_history", []), start=1)
    ]
    pd.DataFrame(
        jobs, columns=["task", "attempt", "pid", "started_at", "finished_at", "success"]
    ).to_csv(destination / "worker_jobs.csv", index=False)
    pd.DataFrame(
        runs, columns=["phase", "status", "wall_seconds", "cpu_workers", "work_seconds"]
    ).to_csv(destination / "campaign_execution.csv", index=False)
    telemetry: list[pd.DataFrame] = []
    for number, run in enumerate(runs, start=1):
        if run.get("telemetry_file"):
            path = Path(str(run["telemetry_file"]))
            if (
                run.get("telemetry_signature")
                and source_digest(path) != run["telemetry_signature"]
            ):
                raise ValueError("Cambió la telemetría de campaña")
            telemetry.append(
                pd.read_csv(path).assign(invocation=number, phase=run["phase"])
            )
    if telemetry:
        pd.concat(telemetry, ignore_index=True).to_csv(
            destination / "campaign_telemetry.csv", index=False
        )
    pd.DataFrame(
        [
            {
                "scope": "predictive_all_recorded_attempts",
                "wall_seconds": totals.wall_seconds.sum(min_count=1),
                "wall_seconds_scope": "sum_of_job_durations_not_campaign_elapsed",
                "campaign_wall_seconds": sum(float(run["wall_seconds"]) for run in runs)
                if runs
                else None,
                "campaign_wall_seconds_scope": "predictive_coordinator_invocations_only_policies_separate",
                "cpu_workers": load_study_config(study).execution.effective_workers,
                "cpu_seconds": totals.cpu_seconds.sum(min_count=1),
                "peak_rss_mib": totals.peak_rss_mib.max(),
                "peak_rss_scope": "maximum_single_worker_not_total_simultaneous_memory",
                "complete": bool(
                    not table.empty and table.coverage.eq("complete").all()
                ),
            }
        ]
    ).to_csv(destination / "research_cost.csv", index=False)
    return bool(not table.empty and table.coverage.eq("complete").all())


def markdown_table(table: pd.DataFrame) -> str:
    values = table.round(5).astype(object).where(table.notna(), "—")
    return "\n".join(
        [
            "| " + " | ".join(map(str, table.columns)) + " |",
            "| " + " | ".join("---" for _ in table.columns) + " |",
            *(
                "| " + " | ".join(map(str, row)) + " |"
                for row in values.itertuples(index=False, name=None)
            ),
        ]
    )
