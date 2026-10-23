"""Figuras de tesis a partir de tablas exportadas, sin inferencia ni reajustes."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.evaluacion.lectura_resultados import PERCENT_SCALE, time_comparison

TARGET_LABELS = {
    "mass_kg": "Masa (kg)",
    "util_ratio": "Utilización promedio",
    "z_disp_m": "Desplazamiento Z (m)",
}
SPLIT_LABELS = {"validation": "Validación", "test": "Prueba"}
EXPERIMENT_LABELS = {
    "A": "A: respuestas históricas",
    "B": "B: estimaciones predictivas",
}


def prediction_diagnostics(report: Path, selection: tuple[str, int]) -> list[Figure]:
    """Muestra observado/predicho y residuos de la semilla representativa congelada."""
    family, seed = selection
    errors = pd.read_csv(report / "prediction_errors.csv")
    errors = errors.loc[errors.model.eq(family) & errors.seed.eq(seed)]
    figures: list[Figure] = []
    for split, group in errors.groupby("split", sort=False):
        figure, axes = plt.subplots(2, len(OUTPUT_COLUMNS), figsize=(14, 8))
        for column, target in enumerate(OUTPUT_COLUMNS):
            table = group.loc[group.target.eq(target)]
            lower = min(table.observed.min(), table.predicted.min())
            upper = max(table.observed.max(), table.predicted.max())
            axes[0, column].scatter(table.observed, table.predicted, alpha=0.65, s=18)
            axes[0, column].plot([lower, upper], [lower, upper], "--", color="gray")
            axes[0, column].set(
                xlabel=f"Observado: {TARGET_LABELS[target]}", ylabel="Predicho"
            )
            axes[1, column].scatter(table.observed, table.error, alpha=0.65, s=18)
            axes[1, column].axhline(0, linestyle="--", color="gray")
            axes[1, column].set(
                xlabel=TARGET_LABELS[target], ylabel="Residuo: predicho − observado"
            )
        figure.suptitle(f"{SPLIT_LABELS[str(split)]} · {family}, semilla {seed}")
        figure.tight_layout()
        figures.append(figure)
    return figures


def search_figures(report: Path, budget: int) -> list[Figure]:
    """Curvas con mesetas tras terminación y frentes de un escenario declarado."""
    curves = pd.read_csv(report / "search_curves.csv")
    fronts = pd.read_csv(report / "search_fronts.csv")
    figures: list[Figure] = []
    for (split, experiment), group in curves.loc[curves.budget.eq(budget)].groupby(
        ["split", "experiment"]
    ):
        figure, axis = plt.subplots(figsize=(10, 5))
        episodes: list[pd.DataFrame] = []
        for identity, episode in group.groupby(["strategy", "scenario", "policy_seed"]):
            series = episode.set_index("query").hypervolume.sort_index()
            series = series.reindex(range(int(series.index.min()), budget + 1)).ffill()
            episodes.append(
                series.rename("hypervolume").reset_index().assign(strategy=identity[0])
            )
        means = pd.concat(episodes).groupby(["strategy", "query"]).hypervolume.mean()
        for strategy, series in means.groupby(level="strategy"):
            axis.plot(
                series.index.get_level_values("query"),
                series.to_numpy(),
                label=str(strategy),
            )
        axis.set(
            xlabel="Evaluaciones, incluida la población inicial",
            ylabel="Hipervolumen",
            title=f"{SPLIT_LABELS[str(split)]} · {EXPERIMENT_LABELS[str(experiment)]}",
        )
        axis.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.tight_layout()
        figures.append(figure)
    for (split, experiment), group in fronts.groupby(["split", "experiment"]):
        figure = plt.figure(figsize=(10, 6))
        axis = figure.add_subplot(projection="3d")
        for strategy, table in group.loc[group.is_pareto].groupby("strategy"):
            axis.scatter(
                table.mass_kg,
                table.util_ratio,
                table.z_disp_m,
                label=str(strategy),
                s=18,
            )
        axis.set(
            xlabel=TARGET_LABELS["mass_kg"],
            ylabel=TARGET_LABELS["util_ratio"],
            zlabel=TARGET_LABELS["z_disp_m"],
            title=f"{SPLIT_LABELS[str(split)]} · {EXPERIMENT_LABELS[str(experiment)]}\nFrentes por estrategia, escenario 0 y semilla representativa",
        )
        if len(axis.collections):
            axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
        figure.tight_layout()
        figures.append(figure)
    return figures


def predictor_cost_figure(report: Path) -> Figure:
    costs = pd.read_csv(report / "predictor_costs.csv")
    totals = costs.groupby("model")[
        ["tuning_seconds", "crossfit_seconds", "fit_seconds"]
    ].sum()
    figure, axis = plt.subplots(figsize=(11, 5))
    totals.plot.bar(stacked=True, ax=axis)
    axis.set(
        xlabel="Familia",
        ylabel="Segundos",
        title="Búsqueda compartida, OOF y ajustes finales terminados",
    )
    axis.legend(["Búsqueda final", "OOF anidadas", "Ajustes finales"], fontsize=8)
    figure.tight_layout()
    return figure


def predictor_inference_figure(report: Path) -> Figure:
    """Compara latencias de prueba; el promedio entre semillas conserva las medianas."""
    times = time_comparison(report).sort_values("individual_ms")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, column, title in zip(
        axes,
        ("individual_ms", "batch_per_design_ms"),
        ("Consulta individual", "Por diseño dentro de un lote"),
        strict=True,
    ):
        times[column].plot.barh(ax=axis, logx=True)
        axis.set(
            xlabel="Milisegundos · escala logarítmica", ylabel="Familia", title=title
        )
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Inferencia de prueba: media entre semillas de medianas medidas")
    figure.tight_layout()
    return figure


def shap_comparison_figure(report: Path, seed: int) -> Figure:
    """Normaliza importancia dentro de cada respuesta; no compara precisión."""
    importance = pd.read_csv(report / "shap_global.csv")
    importance = importance.loc[importance.seed.eq(seed)].copy()
    total = importance.groupby(["model", "target"]).abs_shap.transform("sum")
    importance["percentage"] = (
        importance.abs_shap.div(total.where(total.gt(0))) * PERCENT_SCALE
    )
    figure, axes = plt.subplots(
        1, len(OUTPUT_COLUMNS), figsize=(16, 6), layout="constrained"
    )
    for axis, target in zip(axes, OUTPUT_COLUMNS, strict=True):
        table = (
            importance.loc[importance.target.eq(target)]
            .pivot(index="model", columns="variable", values="percentage")
            .fillna(0)
        )
        image = axis.imshow(
            table.to_numpy(), vmin=0, vmax=PERCENT_SCALE, cmap="Blues", aspect="auto"
        )
        axis.set_xticks(
            range(len(table.columns)), table.columns, rotation=60, ha="right"
        )
        axis.set_yticks(range(len(table.index)), table.index)
        axis.set_title(TARGET_LABELS[target])
    figure.colorbar(image, ax=axes, label="Porcentaje de la media de |SHAP| del modelo")
    figure.suptitle(
        f"SHAP de validación · semilla {seed} · Dummy tiene contribuciones nulas"
    )
    return figure
