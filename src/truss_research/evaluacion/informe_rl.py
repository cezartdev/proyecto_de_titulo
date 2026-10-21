"""Curvas y costos observados de políticas, sin comparar pérdidas entre algoritmos."""

from pathlib import Path

import pandas as pd

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.procesados import JsonValue
from truss_research.optimizacion.variantes_rl import policy_variant

CURVE_AXES: dict[str, tuple[str, str]] = {
    "step": ("transitions", "Transiciones de entrenamiento"),
    "optimization_cpu_seconds": ("cpu", "CPU de entrenamiento acumulada (s)"),
    "optimization_wall_seconds": ("wall", "Duración de entrenamiento acumulada (s)"),
}


def export_policy_diagnostics(
    records: tuple[dict[str, JsonValue], ...], destination: Path
) -> None:
    curves: list[pd.DataFrame] = []
    summaries: list[dict[str, JsonValue]] = []
    for record in records:
        key = str(record["key"])
        if not key.startswith("policy/"):
            continue
        _, name, seed = key.split("/")
        variant = policy_variant(name)
        root = Path(str(record["directory"]))
        metadata = read_artifact(root / "run.json")
        curve = pd.read_csv(root / "learning_curve.csv")
        identity = dict(
            strategy=name,
            algorithm=variant.algorithm,
            use_predictions=variant.use_predictions,
            reward_mode=variant.reward_mode,
            enhanced_state=variant.enhanced_state,
            policy_training_seed=int(seed),
        )
        curves.append(curve.assign(**identity))
        summaries.append(
            {
                **identity,
                **{
                    name: metadata[name]
                    for name in (
                        "completed_steps",
                        "training_episodes",
                        "optimizer_updates",
                        "samples_used",
                        "parameter_count",
                        "training_seconds",
                        "optimization_seconds",
                        "validation_wall_seconds",
                        "policy_cpu_seconds",
                        "best_validation_auc",
                        "repeat_transitions",
                        "repeat_transition_rate",
                    )
                },
                "optimization_cpu_seconds": float(
                    curve.optimization_cpu_seconds.iloc[-1]
                ),
                "selected_step": int(curve.loc[curve.validation_auc.idxmax(), "step"]),
                "loss_scope": str(metadata["loss_scope"]),
                "interpretation": "fixed_parameter_pipelines; temporal_repetitions_are_not_new_designs",
            }
        )
    pd.DataFrame(summaries).to_csv(
        destination / "policy_training_summary.csv", index=False
    )
    if not curves:
        return
    learning = pd.concat(curves, ignore_index=True)
    learning.to_csv(destination / "learning_curves.csv", index=False)
    plot_policy_curves(learning, destination)


def plot_policy_curves(learning: pd.DataFrame, destination: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    means = learning.groupby(["strategy", "step"], as_index=False)[
        ["validation_auc", "optimization_cpu_seconds", "optimization_wall_seconds"]
    ].mean()
    for axis, (suffix, label) in CURVE_AXES.items():
        figure = Figure(figsize=(8, 5))
        FigureCanvasAgg(figure)
        plot = figure.subplots()
        for name, group in means.groupby("strategy", sort=False):
            plot.plot(group[axis], group.validation_auc, marker="o", label=name)
        plot.set(
            xlabel=label,
            ylabel="AUC normalizada de validación",
            title="Medias entre semillas; entrenamiento excluye validación",
        )
        plot.legend(fontsize="small")
        figure.tight_layout()
        figure.savefig(destination / f"rl_learning_{suffix}.png", dpi=150)
        figure.clear()
