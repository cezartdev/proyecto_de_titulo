"""Integración y auditoría de análisis adicionales sin copiar modelos pesados."""

from math import nan
from pathlib import Path
from shutil import copyfile

import pandas as pd

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.particion_equilibrada import verify_group_exclusion
from truss_research.datos.procesados import load_design_split
from truss_research.datos.schemas import DESIGN_MEMBERSHIP_FILENAME
from truss_research.evaluacion.curvas_aprendizaje import latest_learning_curves
from truss_research.evaluacion.sensibilidad import latest_sensitivity
from truss_research.optimizacion.estudio_estado import load_study_config


def attach_analysis(study: Path, destination: Path) -> Path | None:
    curves = (
        latest_learning_curves(study) if (study / "freeze.json").is_file() else None
    )
    sensitivity = (
        latest_sensitivity(study) if (study / "freeze.json").is_file() else None
    )
    if curves is not None:
        for path in curves.glob("*.csv"):
            copyfile(path, destination / f"learning_{path.name}")
        copyfile(curves / "learning_curves.png", destination / "learning_curves.png")
    costs = []
    for name, root in (
        (
            "generalization_and_controls",
            sensitivity / "generalization" if sensitivity else None,
        ),
        ("learning_curves", curves),
    ):
        if root is None or not (root / "resource_phases.csv").is_file():
            continue
        phases = pd.read_csv(root / "resource_phases.csv")
        totals = phases.loc[phases.phase.eq("task_total")]
        jobs = pd.read_csv(root / "worker_jobs.csv")
        failed_work, failed_cpu, failed_count = 0.0, 0.0, 0
        for path in root.glob("*_failed_*/*.resources.json"):
            receipt = read_artifact(path)
            for phase in receipt.get("phases", []):
                if phase.get("phase") == "task_total":
                    failed_work += (
                        float(phase["wall_seconds"])
                        if phase.get("wall_seconds") is not None
                        else nan
                    )
                    failed_cpu += (
                        float(phase["cpu_seconds"])
                        if phase.get("cpu_seconds") is not None
                        else nan
                    )
                    failed_count += 1
        costs.append(
            {
                "analysis": name,
                "task_count": len(totals),
                "work_seconds": totals.wall_seconds.sum(),
                "cpu_seconds": totals.cpu_seconds.sum(),
                "failed_attempts": failed_count,
                "failed_work_seconds": failed_work,
                "failed_cpu_seconds": failed_cpu,
                "max_worker_peak_rss_mib": totals.peak_rss_mib.max(),
                "execution_span_seconds": jobs.finished_at.max()
                - jobs.started_at.min(),
                "scope": "original_jobs_including_fit_and_evaluation; excludes_repeated_benchmark",
                "cpu_coverage_complete": totals.cpu_coverage.eq("complete").all(),
            }
        )
    pd.DataFrame(
        costs,
        columns=[
            "analysis",
            "task_count",
            "work_seconds",
            "cpu_seconds",
            "failed_attempts",
            "failed_work_seconds",
            "failed_cpu_seconds",
            "max_worker_peak_rss_mib",
            "execution_span_seconds",
            "scope",
            "cpu_coverage_complete",
        ],
    ).to_csv(destination / "analysis_costs.csv", index=False)
    return curves


def audit_analysis(study: Path) -> None:
    config = load_study_config(study)
    if not (study / "freeze.json").is_file():
        return
    sensitivity = latest_sensitivity(study)
    if config.analysis.generalization_seeds and sensitivity is not None:
        membership = pd.read_csv(config.data_dir / DESIGN_MEMBERSHIP_FILENAME)
        for seed in config.analysis.generalization_seeds:
            root = sensitivity / "generalization" / f"seed_{seed}"
            tables = {
                name: pd.read_csv(root / f"chronological_{name}_ids.csv")
                for name in ("train", "calibration", "validation", "test")
            }
            verify_group_exclusion(tables, membership)
            control = pd.read_csv(root / "design_overlap_control_train_ids.csv")
            if (
                not tables["train"]
                .web_config.value_counts()
                .sort_index()
                .equals(control.web_config.value_counts().sort_index())
            ):
                raise ValueError("El control no conserva tamaños por categoría")
            for name in ("calibration", "validation", "test"):
                other = pd.read_csv(root / f"design_overlap_control_{name}_ids.csv")
                if set(other.iter_id) != set(tables[name].iter_id):
                    raise ValueError("El control cambió los diseños retenidos")
                if set(control.iter_id) & set(other.iter_id):
                    raise ValueError("El control reutilizó diseños retenidos en ajuste")
            if config.tabpfn_enabled:
                for experiment in ("chronological", "design_overlap_control"):
                    predictor = read_artifact(
                        sensitivity
                        / "generalization"
                        / f"seed_{seed}"
                        / experiment
                        / "tabpfn"
                        / "run.json"
                    )
                    fit_ids = pd.read_csv(root / f"{experiment}_train_ids.csv").iter_id
                    cal_ids = pd.read_csv(
                        root / f"{experiment}_calibration_ids.csv"
                    ).iter_id
                    if (
                        set(predictor["fit_ids"]) != set(fit_ids)
                        or set(predictor["calibration_ids"]) != set(cal_ids)
                        or predictor["trial_count"] != 0
                        or predictor["tuning"] is not False
                    ):
                        raise ValueError(
                            "TabPFN no conserva las particiones retenidas o su presupuesto fijo"
                        )
    if not config.analysis.learning_curves.enabled:
        return
    curves = latest_learning_curves(study)
    if curves is None:
        raise ValueError("La auditoría requiere completar make learning-curves")
    split = load_design_split(config.data_dir)
    allowed = set(split.training.iter_id)
    subsets: dict[tuple[str, int | None], list[tuple[int, set[str]]]] = {}
    for path in curves.glob("*/run.json"):
        metadata = read_artifact(path)
        ids = set(metadata["fit_ids"])
        predictor = read_artifact(path.parent / "predictor/run.json")
        if (
            not ids <= allowed
            or len(ids) != metadata["train_size"]
            or predictor.get("tuning") is not False
            or predictor.get("trial_count") != 0
        ):
            raise ValueError("La curva contiene fuga de datos o búsquedas no previstas")
        key = (metadata["model"], metadata["subset_seed"])
        subsets.setdefault(key, []).append((len(ids), ids))
    for values in subsets.values():
        ordered = sorted(values, key=lambda item: item[0])
        if any(
            not previous[1] <= current[1]
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise ValueError("Los subconjuntos de curvas no son anidados")
