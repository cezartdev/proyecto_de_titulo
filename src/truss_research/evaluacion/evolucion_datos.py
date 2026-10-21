"""Historia evolutiva descriptiva, sin tratar apariciones como muestras independientes."""

from pathlib import Path

import numpy as np
import pandas as pd

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procesados import load_processed_table, load_structural_limits
from truss_research.evaluacion.pareto import ObjectiveSpace, meets_thresholds


def evolution_history(data_dir: Path) -> pd.DataFrame:
    full = load_processed_table(data_dir / "full_clean.csv").set_index("iter_id")
    training = load_processed_table(data_dir / "train.csv")
    membership = pd.read_csv(
        data_dir / "design_membership.csv", float_precision="round_trip"
    )
    if not set(membership.representative_iter_id) == set(full.index):
        raise ValueError("Las pertenencias no cubren los diseños del catálogo")
    space = ObjectiveSpace.fit(
        training[list(OUTPUT_COLUMNS)].to_numpy(), load_structural_limits(data_dir)
    )
    first = membership.groupby("representative_iter_id").generation.min()
    accumulated: set[str] = set()
    rows: list[dict[str, float | int]] = []
    for generation, group in membership.groupby("generation", sort=True):
        ids = set(group.representative_iter_id)
        new = int(first.loc[list(ids)].eq(generation).sum())
        accumulated.update(ids)
        population = full.loc[sorted(ids)]
        responses = population[list(OUTPUT_COLUMNS)].to_numpy()
        probabilities = population.web_config.value_counts(normalize=True).to_numpy()
        rows.append(
            {
                "generation": int(generation),
                "appearances": len(group),
                "unique_in_generation": len(ids),
                "new_designs": new,
                "repeated_appearances": len(group) - new,
                "within_generation_duplicates": len(group) - len(ids),
                "cumulative_unique_designs": len(accumulated),
                "new_design_fraction": new / len(group),
                "topology_entropy": float(
                    -np.sum(probabilities * np.log(probabilities))
                ),
                "geometry_diversity": float(
                    population[list(INPUT_COLUMNS)[1:]]
                    .std(ddof=0)
                    .div(
                        full[list(INPUT_COLUMNS)[1:]]
                        .max()
                        .sub(full[list(INPUT_COLUMNS)[1:]].min())
                        .replace(0, 1)
                    )
                    .mean()
                ),
                "feasible_unique_designs": int(
                    meets_thresholds(responses, space.limits).sum()
                ),
                "population_hypervolume": space.hypervolume(responses),
                "cumulative_hypervolume": space.hypervolume(
                    full.loc[sorted(accumulated), list(OUTPUT_COLUMNS)].to_numpy()
                ),
                "population_front_size": len(space.front(responses)),
                **{
                    f"mean_{name}": float(population[name].mean())
                    for name in OUTPUT_COLUMNS
                },
            }
        )
    return pd.DataFrame(rows)


def export_evolution_history(data_dir: Path, destination: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    table = evolution_history(data_dir)
    table.to_csv(destination / "evolution_history.csv", index=False)
    figure = Figure(figsize=(9, 6))
    FigureCanvasAgg(figure)
    novelty, quality = figure.subplots(2, 1, sharex=True)
    novelty.plot(table.generation, table.new_designs, label="Diseños nuevos")
    novelty.plot(
        table.generation, table.repeated_appearances, label="Apariciones repetidas"
    )
    novelty.set(ylabel="Cantidad por generación")
    novelty.legend()
    quality.plot(table.generation, table.population_hypervolume, label="Población")
    quality.plot(
        table.generation, table.cumulative_hypervolume, label="Archivo acumulado"
    )
    quality.set(xlabel="Generación", ylabel="Hipervolumen observado")
    quality.legend()
    figure.suptitle(
        "Trayectoria original; apariciones repetidas no son diseños independientes"
    )
    figure.tight_layout()
    figure.savefig(destination / "evolution_history.png", dpi=150)
