"""Relación descriptiva entre error predictivo y utilidad de un ranking fijo."""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.procesados import JsonValue


def utility_comparison(records: tuple[dict[str, JsonValue], ...]) -> pd.DataFrame:
    predictors: dict[tuple[str, str, int], dict[str, JsonValue]] = {}
    for record in records:
        key = str(record["key"])
        if key.startswith(("predictor/", "test/predictor/")):
            run = read_artifact(Path(str(record["directory"])) / "run.json")
            parts = key.split("/")
            predictors[(str(run["evaluation_split"]), parts[-2], int(parts[-1]))] = run
    frames: list[pd.DataFrame] = []
    signatures: dict[str, str] = {}
    for record in records:
        key = str(record["key"])
        if "ranking/" not in key:
            continue
        root = Path(str(record["directory"]))
        ranking = read_artifact(root / "run.json")
        parts = key.split("/")
        split = str(ranking["split"])
        model, seed = parts[-2], int(parts[-1])
        predictor = predictors.get((split, model, seed))
        if predictor is None or any(
            ranking[name] != predictor[other]
            for name, other in (
                ("dataset_signature", "dataset_signature"),
                ("predictor_signature", "model_signature"),
            )
        ):
            raise ValueError(
                "Predicción y utilidad no comparten partición, modelo o firmas"
            )
        signature = str(ranking["comparison_signature"])
        if signatures.setdefault(split, signature) != signature:
            raise ValueError("Los rankings no comparten catálogos y presupuestos")
        table = pd.read_csv(root / "summary.csv")
        per_budget = table.groupby("budget", as_index=False)[
            ["normalized_auc", "relative_hypervolume"]
        ].mean()
        frames.append(
            per_budget.assign(
                split=split,
                model=model,
                seed=seed,
                normalized_mae=float(str(predictor["normalized_mae"])),
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def family_correlations(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | int | float | None]] = []
    for (split, budget), group in table.groupby(["split", "budget"]):
        available = group.dropna(subset=["normalized_mae", "normalized_auc"])
        reason = None
        if len(available) < 3:
            reason = "fewer_than_three_families"
        elif any(
            available[name].nunique() < 2
            for name in ("normalized_mae", "normalized_auc")
        ):
            reason = "constant_metric"
        correlation = (
            float(
                spearmanr(
                    available["normalized_mae"], available["normalized_auc"]
                ).statistic
            )
            if reason is None
            else None
        )
        rows.append(
            {
                "split": str(split),
                "budget": int(budget),
                "families": len(available),
                "spearman": correlation,
                "undefined_reason": reason,
                "interpretation": "descriptive_family_means_no_significance_test",
            }
        )
    return pd.DataFrame(rows)


def export_prediction_utility(
    records: tuple[dict[str, JsonValue], ...], destination: Path
) -> pd.DataFrame:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    runs = utility_comparison(records)
    if runs.empty:
        return runs
    runs.to_csv(destination / "prediction_utility_runs.csv", index=False)
    families = runs.groupby(["split", "model", "budget"], as_index=False)[
        ["normalized_mae", "normalized_auc", "relative_hypervolume"]
    ].mean()
    families.to_csv(destination / "prediction_utility.csv", index=False)
    family_correlations(families).to_csv(
        destination / "prediction_utility_spearman.csv", index=False
    )
    for (split, budget), group in families.groupby(["split", "budget"]):
        figure = Figure(figsize=(9, 5))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for row in group.itertuples(index=False):
            axis.scatter(row.normalized_mae, row.normalized_auc, label=str(row.model))
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        axis.set(
            xlabel="MAE normalizado medio",
            ylabel="AUC normalizada media",
            title=f"Precisión y utilidad — {split}, presupuesto {budget}",
        )
        figure.tight_layout()
        figure.savefig(
            destination / f"prediction_utility_{split}_{budget}.png", dpi=150
        )
        figure.clear()
    return families
