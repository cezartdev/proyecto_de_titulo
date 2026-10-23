"""Lecturas descriptivas de tablas cerradas, sin ajustar ni volver a seleccionar."""

from pathlib import Path

import pandas as pd

SECONDS_PER_MINUTE = 60
MILLISECONDS_PER_SECOND = 1000
PERCENT_SCALE = 100
FIT_PHASES = ("tuning_seconds", "crossfit_seconds", "fit_seconds")


def best_families(
    table: pd.DataFrame, metric: str, maximize: bool
) -> tuple[str, float]:
    values = table[metric]
    best = float(values.max() if maximize else values.min())
    names = ", ".join(sorted(table.loc[values.eq(best), "model"].astype(str)))
    return names, best


def predictor_interpretation(report: Path, selected: str, budget: int) -> str:
    utility = pd.read_csv(report / "predictor_utility_summary.csv")
    utility = utility.loc[
        utility.split.eq("test")
        & utility.budget.eq(budget)
        & utility.role.eq("regressor")
    ]
    leader, auc = best_families(utility, "mean_normalized_auc", True)
    frozen_auc = float(
        utility.loc[utility.model.eq(selected), "mean_normalized_auc"].iloc[0]
    )
    accuracy = pd.read_csv(report / "predictive_model_summary.csv")
    accuracy = accuracy.loc[accuracy.split.eq("test")]
    precise, error = best_families(accuracy, "mean", False)
    return (
        f"Con presupuesto **{budget}**, la mayor media descriptiva de AUC normalizado "
        f"en prueba corresponde a **{leader} ({auc:.4f})**. La familia elegida en "
        f"validación, **{selected}**, alcanza **{frozen_auc:.4f}** y conserva su rol "
        "congelado: el orden de prueba no se usa para cambiarla. "
        f"El menor MAE normalizado medio corresponde a **{precise} ({error:.4f})**. "
        "Estos criterios responden preguntas distintas: reconstruir respuestas con "
        "menor error no garantiza seleccionar mejores alternativas multiobjetivo. "
        "Las posiciones son descriptivas; los contrastes pareados delimitan la "
        "incertidumbre y no establecen superioridad poblacional."
    )


def time_comparison(report: Path) -> pd.DataFrame:
    """Separa fases de ajuste y medias entre semillas de medianas de inferencia."""
    costs = pd.read_csv(report / "predictor_costs.csv")
    fitting = costs.groupby("model")[list(FIT_PHASES)].sum()
    fitting["fit_phase_minutes"] = fitting.sum(axis=1) / SECONDS_PER_MINUTE
    fitting["adjustment_seeds"] = costs.groupby("model").seed.nunique()
    fitting["device"] = costs.groupby("model").device.agg(
        lambda values: " / ".join(sorted(set(values.astype(str))))
    )
    inference = pd.read_csv(report / "predictive_inference_costs.csv")
    inference = inference.loc[inference.split.eq("test")].copy()
    inference["individual_ms"] = (
        inference.individual_median_seconds * MILLISECONDS_PER_SECOND
    )
    inference["batch_ms"] = inference.batch_median_seconds * MILLISECONDS_PER_SECOND
    inference["batch_per_design_ms"] = inference.batch_ms / inference.batch_rows
    latencies = inference.groupby("model")[
        ["individual_ms", "batch_ms", "batch_per_design_ms", "batch_rows"]
    ].mean()
    return fitting.join(latencies, validate="one_to_one")


def time_interpretation(report: Path, selected: str) -> str:
    times = time_comparison(report)
    accuracy = pd.read_csv(report / "predictive_model_summary.csv")
    accuracy = accuracy.loc[accuracy.split.eq("test")]
    precise, _ = best_families(accuracy, "mean", False)
    precise_names = accuracy.loc[accuracy["mean"].eq(accuracy["mean"].min()), "model"]
    precise_ms = float(times.loc[precise_names, "individual_ms"].mean())
    frozen_ms = float(times.loc[selected, "individual_ms"])
    largest_cost = str(times.fit_phase_minutes.idxmax())
    largest_minutes = float(times.loc[largest_cost, "fit_phase_minutes"])
    selected_minutes = float(times.loc[selected, "fit_phase_minutes"])
    return (
        f"Las fases registradas de ajuste de **{selected}** suman "
        f"**{selected_minutes:.2f} min**. La mayor suma corresponde a "
        f"**{largest_cost} ({largest_minutes:.2f} min)**. "
        f"La inferencia individual de **{selected}** registra una media entre "
        f"semillas de medianas de **{frozen_ms:.3f} ms**. Las familias con menor "
        f"MAE normalizado, **{precise}**, registran **{precise_ms:.3f} ms** "
        "como promedio de esas latencias. La tabla y el gráfico permiten valorar "
        "si la precisión adicional compensa la latencia para el uso previsto. "
        "La suma de fases de ajuste corresponde al trabajo registrado de todas "
        "las semillas y preparaciones compartidas; no es el tiempo transcurrido "
        "de campaña ni incluye el preentrenamiento externo de TabPFN. Las fases "
        "OOF incluyen su selección interna y no deben sumarse de nuevo a los "
        "totales de recursos. CPU y CUDA se indican porque sus tiempos no "
        "constituyen una comparación a hardware idéntico."
    )


def interval_reading(row: pd.Series) -> str:
    lower, upper = float(row["lower"]), float(row["upper"])
    position = "incluye cero"
    if upper < 0:
        position = "queda completamente por debajo de cero"
    if lower > 0:
        position = "queda completamente por encima de cero"
    return (
        f"{row['strategy']} − {row['baseline']}: diferencia media "
        f"**{float(row['mean_auc_difference']):+.4f}**, intervalo exploratorio "
        f"**[{lower:+.4f}, {upper:+.4f}]**, que {position}."
    )


def search_interpretation(study: Path, report: Path, budget: int) -> str:
    comparison = pd.read_csv(report / "comparison.csv")
    comparison = comparison.loc[
        comparison.split.eq("test") & comparison.budget.eq(budget)
    ]
    paragraphs = [
        f"Con presupuesto **{budget}**, los mayores promedios descriptivos de "
        "AUC de prueba son:"
    ]
    for mode, label in (("A", "recuperación histórica"), ("B", "entorno predictivo")):
        rows = comparison.loc[comparison.experiment.eq(mode)].rename(
            columns={"strategy": "model"}
        )
        names, score = best_families(rows, "mean_auc", True)
        paragraphs.append(f"- **{mode}, {label}: {names} ({score:.4f})**.")
    intervals = pd.read_csv(study / "evolution/test/paired_intervals.csv")
    contrasts = intervals.loc[
        intervals.budget.eq(budget)
        & intervals.strategy.eq("rl")
        & intervals.baseline.isin(("contextual_bandit", "random"))
    ]
    paragraphs.append("\nLos contrastes congelados permiten precisar esta lectura:")
    paragraphs.extend(
        f"- Modo {row['experiment']}: {interval_reading(row)}"
        for _, row in contrasts.iterrows()
    )
    paragraphs.append(
        "\nUna posición media menor no prueba equivalencia. Estos resultados "
        "describen las implementaciones y presupuestos evaluados; no descartan "
        "otros algoritmos RL o ajustes. Las consultas sin topología histórica, "
        "las proyecciones y los duplicados también condicionan el uso del "
        "presupuesto. B no aporta validación estructural independiente."
    )
    return "\n".join(paragraphs)


def shap_interpretation(report: Path, seed: int) -> str:
    path = report / "shap_global.csv"
    if not path.is_file():
        return "No hay SHAP completo exportado: esta etapa no se interpreta."
    importance = pd.read_csv(path)
    importance = importance.loc[importance.seed.eq(seed)].copy()
    totals = importance.groupby(["model", "target"]).abs_shap.transform("sum")
    importance = importance.loc[totals.gt(0)].copy()
    importance["share"] = importance.abs_shap / totals.loc[importance.index]
    leaders = importance.loc[importance.groupby(["model", "target"]).abs_shap.idxmax()]
    lines = [
        f"En la semilla representativa **{seed}**, excluyendo el modelo constante:"
    ]
    for target, group in leaders.groupby("target"):
        dominant = str(group.variable.value_counts().idxmax())
        supporting = group.loc[group.variable.eq(dominant)]
        lower = float(supporting.share.min()) * PERCENT_SCALE
        upper = float(supporting.share.max()) * PERCENT_SCALE
        lines.append(
            f"- **{target}**: `{dominant}` ocupa el primer puesto en "
            f"**{len(supporting)}/{len(group)}** familias, con **{lower:.1f}–{upper:.1f}%** "
            "de su importancia absoluta total."
        )
    lines.append(
        "\nLa concordancia describe modelos sobre una misma fuente, no réplicas "
        "independientes ni causalidad física. Los porcentajes normalizan la "
        "media de |SHAP| dentro de cada modelo y respuesta; no representan "
        "varianza explicada. Las familias con componentes idénticos pueden "
        "compartir atribuciones. Sin varias semillas explicadas no se estima "
        "estabilidad SHAP entre ajustes."
    )
    return "\n".join(lines)
