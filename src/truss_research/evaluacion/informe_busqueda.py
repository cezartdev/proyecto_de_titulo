"""Informes de búsqueda sobre artefactos cerrados, sin volver a entrenar ni evaluar."""

from math import ceil
from pathlib import Path

import pandas as pd

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.evaluacion.benchmark_multiobjetivo import (
    export_benchmark_reports,
    objective_space,
    verify_benchmark,
)
from truss_research.evaluacion.busqueda_evolutiva import (
    SEARCH_DIRECTORY,
    verify_evaluation,
)
from truss_research.evaluacion.comparacion_predictiva import (
    export_predictive_comparison,
)
from truss_research.evaluacion.informe_costos import export_resource_report
from truss_research.evaluacion.robustez import export_convergence
from truss_research.optimizacion.estrategias_busqueda import STRATEGY_DEFINITIONS
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    artifact_signatures,
    load_study_config,
)
from truss_research.surrogates.experimentos import write_json


def verify_closed_search(study: Path) -> None:
    root = study / SEARCH_DIRECTORY
    freeze = read_artifact(study / "freeze.json")
    manifest = read_artifact(root / "closed.json")
    if freeze["signature"] != manifest["signature"] or manifest["phase"] != "complete":
        raise ValueError("Completar make final antes de exportar el cierre")
    if freeze["policies"] != artifact_signatures(root / "policies"):
        raise ValueError("Las políticas difieren del cierre congelado")
    if freeze.get("policy_execution", {}) != artifact_signatures(
        root / "policy_execution"
    ):
        raise ValueError("Cambió la ejecución de políticas congelada")
    if (
        freeze.get("predictor_import")
        and source_digest(study / "predictor_import.json") != freeze["predictor_import"]
    ):
        raise ValueError("Cambió la procedencia predictiva congelada")
    if freeze["validation"] != artifact_signatures(root / "validation"):
        raise ValueError("Cambió la validación congelada")
    for model in freeze["models"].values():
        if artifact_signatures(Path(model["directory"])) != model["files"]:
            raise ValueError("Cambió un predictor congelado")
    verify_evaluation(root / "validation")
    verify_evaluation(root / "test")
    if manifest["test_predictors"] != artifact_signatures(root / "test_predictors"):
        raise ValueError("Cambió la evaluación predictiva final")
    for name, signatures in (
        ("validation", freeze["predictor_utility_validation"]),
        ("test", manifest["predictor_utility_test"]),
    ):
        directory = root / "predictor_utility" / name
        verify_benchmark(directory)
        if signatures != artifact_signatures(directory):
            raise ValueError("Cambió la comparación multiobjetivo de predictores")


def export_search(study: Path, destination: Path, kind: str) -> Path:
    verify_closed_search(study)
    if destination.exists():
        raise ValueError("La exportación requiere un destino nuevo")
    destination.mkdir(parents=True)
    config = load_study_config(study)
    root = study / SEARCH_DIRECTORY
    export_policy_resources(study, destination)
    if kind == "auditoria":
        return export_integrity(study, destination)
    summaries = pd.concat(
        [pd.read_csv(root / name / "summary.csv") for name in ("validation", "test")]
    )
    summary = summaries.groupby(
        ["split", "experiment", "strategy", "budget"], as_index=False
    ).agg(
        mean_auc=("auc", "mean"),
        mean_hypervolume=("hypervolume", "mean"),
        attainment_rate=("attained", "mean"),
        mean_seconds=("search_seconds", "mean"),
        mean_evaluations=("evaluations", "mean"),
        mean_generated=("generated", "mean"),
        mean_rejected=("rejected", "mean"),
        mean_duplicates=("duplicates", "mean"),
        mean_unsupported_queries=("unsupported_queries", "mean"),
        mean_projection_distance=("projection_distance_mean", "mean"),
        mean_cheap_predictions=("cheap_predictions", "mean"),
        mean_generation_seconds=("generation_seconds", "mean"),
        mean_selection_seconds=("selection_seconds", "mean"),
        mean_evaluation_seconds=("evaluation_seconds", "mean"),
        mean_cheap_seconds=("cheap_seconds", "mean"),
        mean_design_distance=("mean_design_distance", "mean"),
        mean_topology_count=("archive_topology_count", "mean"),
    )
    summary.to_csv(destination / "comparison.csv", index=False)
    export_search_details((root, config), summaries, destination)
    export_predictive_comparison(study, destination)
    export_benchmark_reports(root / "predictor_utility", destination)
    from truss_research.evaluacion.informe_explicabilidad import export_explanations

    export_explanations(study, destination)
    predictive_metrics = [
        pd.read_csv(path) for path in (root / "test_predictors").glob("*/metrics.csv")
    ]
    if predictive_metrics:
        pd.concat(predictive_metrics, ignore_index=True).to_csv(
            destination / "predictive_test_metrics.csv", index=False
        )
    policy_costs = [
        (path, read_artifact(path)) for path in (root / "policies").glob("*/*/run.json")
    ]
    costs = pd.DataFrame(
        [
            {
                "policy": path.parent.parent.name,
                "seed": int(path.parent.name),
                "learning_objective": row["config"]["objective"],
                **{
                    f"training_{key}": value
                    for key, value in row["training_costs"].items()
                },
                **{
                    f"validation_{key}": value
                    for key, value in row["validation_costs"].items()
                },
                **{
                    key: row.get(key)
                    for key in (
                        "wall_seconds",
                        "cpu_seconds",
                        "training_surrogate_evaluations",
                        "optimizer_updates",
                        "validation_surrogate_evaluations",
                    )
                },
            }
            for path, row in policy_costs
        ]
    )
    if costs.empty:
        costs = pd.DataFrame(columns=["policy", "seed", "wall_seconds", "cpu_seconds"])
    costs.to_csv(destination / "policy_costs.csv", index=False)
    amortization: list[dict[str, str | int | float | None]] = []
    final = summary.loc[summary.split.eq("test")]
    for (experiment, budget), group in final.groupby(["experiment", "budget"]):
        for policy in config.search.policies:
            baseline = "greedy" if STRATEGY_DEFINITIONS[policy].assisted else "random"
            control = group.loc[group.strategy.eq(baseline), "mean_seconds"]
            if control.empty:
                continue
            baseline_seconds = float(control.iloc[0])
            seconds = float(
                group.loc[group.strategy.eq(policy), "mean_seconds"].iloc[0]
            )
            training = float(costs.loc[costs.policy.eq(policy), "wall_seconds"].mean())
            saving = baseline_seconds - seconds
            amortization.append(
                {
                    "experiment": experiment,
                    "budget": int(budget),
                    "policy": policy,
                    "baseline": baseline,
                    "extra_policy_training_seconds": training,
                    "search_saving_seconds": saving,
                    "searches_to_amortize": ceil(training / saving)
                    if saving > 0
                    else None,
                    "scope": "exploratory_policy_cost_only_shared_predictors_excluded_no_fem",
                }
            )
    pd.DataFrame(
        amortization,
        columns=[
            "experiment",
            "budget",
            "policy",
            "baseline",
            "extra_policy_training_seconds",
            "search_saving_seconds",
            "searches_to_amortize",
            "scope",
        ],
    ).to_csv(destination / "amortization.csv", index=False)
    summaries.groupby(
        ["split", "experiment", "strategy", "budget", "termination"], as_index=False
    ).size().to_csv(destination / "termination_counts.csv", index=False)
    summaries[
        [
            "split",
            "experiment",
            "strategy",
            "budget",
            "scenario",
            "policy_seed",
            "attained",
            "queries_to_target",
            "target_fraction",
            "reference_hypervolume",
            "target_reference_scope",
        ]
    ].to_csv(destination / "target_attainment.csv", index=False)
    state = read_artifact(study / "study.json")
    resources_complete = export_resource_report(study, destination)
    export_convergence(state, destination)
    predictors = [
        read_artifact(Path(task["directory"]) / "run.json")
        for task in state["tasks"]
        if str(task["key"]).startswith("predictor/")
    ]
    pd.DataFrame(
        [
            {
                "model": row["config"]["model"],
                "seed": row["config"]["seed"],
                **{
                    key: row.get(key)
                    for key in (
                        "normalized_mae",
                        "cv_normalized_mae",
                        "tuning_seconds",
                        "crossfit_seconds",
                        "fit_seconds",
                        "reuse_seconds",
                        "device",
                        "requested_trials",
                        "trial_count",
                        "nested_trial_count",
                        "targetwise_family_trials",
                        "search_reused",
                    )
                },
            }
            for row in predictors
        ],
        columns=[
            "model",
            "seed",
            "normalized_mae",
            "cv_normalized_mae",
            "tuning_seconds",
            "crossfit_seconds",
            "fit_seconds",
            "reuse_seconds",
            "device",
            "requested_trials",
            "trial_count",
            "nested_trial_count",
            "targetwise_family_trials",
            "search_reused",
        ],
    ).to_csv(destination / "predictor_costs.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": row["config"]["model"],
                "seed": row["config"]["seed"],
                "target": target,
                "device": device,
            }
            for row in predictors
            for target, device in row.get("target_devices", {}).items()
        ],
        columns=["model", "seed", "target", "device"],
    ).to_csv(destination / "predictor_devices.csv", index=False)
    report = (
        "# Comparación de predictores y propuesta de búsqueda guiada\n\n"
        "El resultado principal es la utilidad multiobjetivo de cada regresor: predictor_utility_summary.csv "
        "y predictor_utility_paired_intervals.csv. La calidad HV/AUC usa respuestas históricas observadas "
        "tras seleccionar; todos comparten catálogo, inicialización, ranking y presupuesto. La familia "
        "se elige por AUC de validación en el menor presupuesto configurado, con MAE como desempate. "
        "El oráculo aplica el mismo ranking con etiquetas: no es un óptimo global ni una cota de AUC. "
        "Los presupuestos efectivos se reducen si el catálogo se agota, sin consultas ficticias. "
        "La precisión y costo complementan esta utilidad: predictive_model_summary.csv, predictive_metric_summary.csv, "
        "predictive_seed_metrics.csv y predictive_paired_errors.csv. Diferencias negativas "
        "favorecen al modelo frente a la referencia. Los contrastes principales usan Dummy y Ridge; "
        "los restantes son secundarios. Los intervalos del 95 % son puntuales, exploratorios "
        "y condicionados a los ajustes; no son pruebas de significación ni de equivalencia. "
        "generation_block remuestrea bloques completos; design_iid_sensitivity es un control. "
        "Pocos bloques limitan la estabilidad. Las semillas no multiplican el tamaño muestral.\n\n"
        "category_coverage.csv identifica categorías sin evaluación y physical_provenance.csv "
        "documenta condiciones conocidas y pendientes. El MAE normalizado usa desviaciones "
        "de entrenamiento y pesos iguales; no es un porcentaje de error. Los errores físicos "
        "y costos fundamentan la interpretación; no hay tolerancias de equivalencia inventadas.\n\n"
        "La búsqueda es una aplicación secundaria exploratoria. "
        "A usa respuestas históricas mediante proyección; B usa estimaciones de un sustituto. "
        "No se acredita factibilidad física de configuraciones nuevas ni ahorro FEM.\n\n"
        "Las consultas históricas sin topología compatible consumen presupuesto y no terminan "
        "el episodio. Su frecuencia se informa junto con distancias de proyección. B usa las "
        "semillas de la familia elegida; el desacuerdo puede ser cero en modelos deterministas "
        "y no constituye incertidumbre física calibrada.\n\n"
        "predictive_inference_costs.csv separa latencia individual y por lote. "
        "predictor_costs.csv registra ajustes anidados y dispositivo efectivo; resource_tasks.csv "
        "consolida intentos, CPU y memoria, y research_cost.csv distingue duración del coordinador predictivo "
        "de suma de duraciones de trabajos. Treinta intentos no "
        "garantizan igual tiempo de cómputo entre familias. TabPFN incorpora preentrenamiento "
        "externo cuyo costo no se midió; Targetwise incluye selección adicional por respuesta.\n\n"
        "Las diferencias e intervalos pareados están en evolution/test/paired_intervals.csv. "
        "Las consultas no alcanzadas siguen censuradas en summary.csv.\n\n"
        "El umbral de alcance usa exclusivamente la referencia de entrenamiento. "
        "Los intervalos son exploratorios, sin corrección por comparaciones múltiples.\n\n"
        "El tiempo de entrenamiento de políticas se registra aparte del tiempo por búsqueda. "
        "No sumar fases anidadas del entrenamiento predictivo.\n"
        "\nSHAP es un análisis complementario de validación. Si existe una etapa completa compatible, "
        "se adjuntan shap_global.csv, shap_examples.csv, shap_costs.csv y shap_stability.csv. "
        "Su ausencia se resuelve con make explain y una nueva exportación; no se calcula al presentar. "
        "Las atribuciones describen predicciones, no efectos causales ni seguridad estructural.\n"
    )
    (destination / "RESULTS.md").write_text(report, encoding="utf-8")
    write_json(
        destination / "technical_closure.json",
        {
            "technical_complete": True,
            "resources_complete": resources_complete,
            "evidence": "historical_and_predictive_separate",
            "source_study": str(study),
            "config_experiment": config.experiment,
            "structural_certification": False,
        },
    )
    write_export_provenance(study, destination, kind)
    if kind == "notebooks":
        from truss_research.evaluacion.presentacion import render_notebooks

        render_notebooks(study, destination)
        write_export_provenance(study, destination, kind)
    return destination


def export_integrity(study: Path, destination: Path) -> Path:
    """Conserva el resultado de las verificaciones sin recalcular tablas de comparación."""
    from truss_research.surrogates.experimentos import data_signature

    resources_complete = export_resource_report(study, destination)
    freeze = read_artifact(study / "freeze.json")
    config = load_study_config(study)
    write_json(
        destination / "integrity.json",
        {
            "signature": freeze["signature"],
            "dataset_signature": data_signature(config.data_dir),
            "verified_models": len(freeze["models"]),
            "verified_splits": ["validation", "test"],
            "checks": [
                "frozen_models",
                "policies",
                "validation",
                "test_episodes",
                "predictor_utility",
                "resource_signatures",
            ],
            "retrained": False,
        },
    )
    write_json(
        destination / "technical_closure.json",
        {
            "technical_complete": True,
            "resources_complete": resources_complete,
            "source_study": str(study),
            "config_experiment": config.experiment,
            "structural_certification": False,
        },
    )
    write_export_provenance(study, destination, "auditoria")
    return destination


def export_search_details(
    context: tuple[Path, StudyConfig], summaries: pd.DataFrame, destination: Path
) -> None:
    """Exporta trazas y archivos representativos ya calculados para sus figuras."""
    from truss_research.cerchas.schemas import OUTPUT_COLUMNS
    from truss_research.datos.procesados import load_design_split

    root, config = context
    keys = ["split", "experiment", "strategy", "budget", "scenario", "policy_seed"]
    columns = [*keys, "query", "hypervolume"]
    initial = summaries[[*keys, "initial_hypervolume"]].rename(
        columns={"initial_hypervolume": "hypervolume"}
    )
    traces = [initial.assign(query=config.evolution.population_size)]
    fronts: list[pd.DataFrame] = []
    space = objective_space(load_design_split(config.data_dir).training, config)
    for split in ("validation", "test"):
        decisions = pd.read_json(root / split / "decisions.jsonl", lines=True)
        decisions = decisions.rename(columns={"evaluation": "query"}).reindex(
            columns=columns
        )
        traces.append(decisions.dropna(subset=["query"]))
        for experiment in ("A", "B"):
            for strategy in config.search.strategies:
                path = (
                    root
                    / split
                    / f"archive_{experiment}_{strategy}_{max(config.budgets)}_0_{config.representative_seed}.csv"
                )
                table = pd.read_csv(path)
                front = pd.DataFrame(
                    space.front(table[list(OUTPUT_COLUMNS)].to_numpy()),
                    columns=OUTPUT_COLUMNS,
                )
                table["is_pareto"] = pd.MultiIndex.from_frame(
                    table[list(OUTPUT_COLUMNS)]
                ).isin(pd.MultiIndex.from_frame(front))
                fronts.append(
                    table.assign(
                        split=split,
                        experiment=experiment,
                        strategy=strategy,
                        budget=max(config.budgets),
                        scenario=0,
                        policy_seed=config.representative_seed,
                    )
                )
    pd.concat(traces, ignore_index=True)[columns].to_csv(
        destination / "search_curves.csv", index=False
    )
    pd.concat(fronts, ignore_index=True).to_csv(
        destination / "search_fronts.csv", index=False
    )


def export_policy_resources(study: Path, destination: Path) -> None:
    """Separa duración del bloque, costos por intento y procedencia reutilizada."""
    path = study / "evolution/policy_execution/manifest.json"
    if path.exists():
        state = read_artifact(path)
        pd.DataFrame(state["runs"]).to_csv(
            destination / "policy_execution.csv", index=False
        )
        rows = []
        for key, record in state["jobs"].items():
            for number, attempt in enumerate(record["attempts"], start=1):
                file = Path(attempt["measurement_files"][0])
                if file.is_file():
                    if source_digest(file) != attempt.get(
                        "measurement_signatures", {}
                    ).get(str(file)):
                        raise ValueError("Cambió una medición de políticas")
                    for phase in read_artifact(file)["phases"]:
                        rows.append(
                            {
                                "policy": key,
                                "attempt": number,
                                "attempt_status": attempt["status"],
                                **phase,
                                "coverage": "complete"
                                if phase["finished"]
                                and not phase.get("errors")
                                and phase["cpu_coverage"] == "complete"
                                else "incomplete",
                            }
                        )
                else:
                    rows.append(
                        {
                            "policy": key,
                            "attempt": number,
                            "attempt_status": attempt["status"],
                            "coverage": "missing",
                        }
                    )
        pd.DataFrame(rows).to_csv(destination / "policy_resources.csv", index=False)
        write_json(destination / "policy_execution.json", state)
    marker = study / "predictor_import.json"
    if marker.exists():
        receipt = read_artifact(marker)
        write_json(destination / "predictor_import.json", receipt)
        pd.DataFrame(
            receipt["discarded_policies"],
            columns=[
                "policy",
                "scope",
                "completed_steps",
                "wall_seconds",
                "cpu_seconds",
            ],
        ).to_csv(destination / "discarded_policy_costs.csv", index=False)


def write_export_provenance(study: Path, destination: Path, kind: str) -> None:
    """Firma las salidas de presentación sin incluir recursivamente su manifiesto."""
    from truss_research.datos.procedencia import confirmations_digest
    from truss_research.evaluacion.presentacion import (
        NOTEBOOK_DIRECTORY,
        NOTEBOOK_NAMES,
    )

    write_json(
        destination / "provenance.json",
        {
            "study": str(study.resolve()),
            "kind": kind,
            "frozen": read_artifact(study / "freeze.json"),
            "physical_provenance_signature": confirmations_digest(),
            "presentation_complete": kind != "notebooks"
            or all(
                (destination / f"{name}.{extension}").is_file()
                for name in NOTEBOOK_NAMES
                for extension in ("ipynb", "html")
            ),
            "notebook_sources": {
                f"{name}.ipynb": source_digest(NOTEBOOK_DIRECTORY / f"{name}.ipynb")
                for name in NOTEBOOK_NAMES
            }
            if kind == "notebooks"
            else {},
            "outputs": {
                name: digest
                for name, digest in artifact_signatures(destination).items()
                if name != "provenance.json"
            },
        },
    )
