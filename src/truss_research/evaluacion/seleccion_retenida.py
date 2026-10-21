"""Selección sobre tablas explícitas, sin cargar particiones globales por defecto."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.optimizacion.catalogos import (
    SelectionBank,
    bank_manifest,
    comparison_signature,
    episode_specs,
)
from truss_research.optimizacion.comparacion import evaluate_episode, strategies_for
from truss_research.optimizacion.contratos import EvaluationSplit, SelectionConfig
from truss_research.optimizacion.estudio_estado import StudyConfig
from truss_research.surrogates.calibracion import SplitConformal
from truss_research.surrogates.experimentos import predict, write_json

GROUP_STRATEGIES = (
    "random",
    "ranking",
    "greedy",
    "greedy_safe",
    "nsga2",
    "nsga2_surrogate",
)
CURVE_STRATEGIES = ("random", "ranking", "greedy")


@dataclass(frozen=True)
class RetainedSelection:
    config: StudyConfig
    training: pd.DataFrame
    destination: Path
    predictor_signature: str
    dataset_signature: str
    calibrator: SplitConformal | None = None
    evaluation_space: ObjectiveSpace | None = None
    strategies: tuple[str, ...] = GROUP_STRATEGIES
    base_catalog_size: int | None = None


def evaluate_retained_selection(
    request: RetainedSelection,
    model: Pipeline,
    held: tuple[EvaluationSplit, pd.DataFrame],
) -> pd.DataFrame:
    label, table = held
    config = request.config
    destination = request.destination / label
    selection = SelectionConfig(
        config.data_dir,
        request.destination,
        destination,
        split=label,
        catalog_size=config.catalog_size,
        budgets=config.budgets,
        seeds=config.seeds,
        episodes_per_seed=config.episodes_per_seed,
        evolution=config.evolution,
        base_catalog_size=request.base_catalog_size,
    )
    indexed = table.set_index("iter_id", drop=False)
    start = perf_counter()
    predictions = predict(model, indexed.loc[:, list(INPUT_COLUMNS)])
    prediction_seconds = perf_counter() - start
    bank = SelectionBank(
        indexed,
        predictions,
        episode_specs(indexed, selection),
        ObjectiveSpace.fit(
            request.training.loc[:, list(OUTPUT_COLUMNS)].to_numpy(),
            config.limits,
        ),
        request.dataset_signature,
        request.predictor_signature,
        tuple(
            np.ptp(
                request.training.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=float),
                axis=0,
            )
        ),
        prediction_seconds,
        tuple(request.calibrator.widths(0.90))
        if request.calibrator is not None
        else None,
        request.evaluation_space,
    )
    rows, traces = [], []
    for spec in bank.episodes:
        for name, (strategy, preparation_seconds) in strategies_for(
            bank, (spec, selection)
        ).items():
            if name not in request.strategies:
                continue
            summary, trajectory = evaluate_episode((bank, spec), strategy)
            summary["target_reached"] = (
                float(summary["queries_to_target"] is not None)
                if summary["reference_hypervolume"] > 0
                else None
            )
            identity = {
                "split": label,
                "episode_id": spec.episode_id,
                "catalog_seed": spec.seed,
                "budget": spec.budget,
                "strategy": name,
                "comparison_id": name,
                "policy_training_seed": None,
            }
            rows.append(
                {
                    **identity,
                    **summary,
                    "preparation_seconds": preparation_seconds,
                    "strategy_seconds": preparation_seconds
                    + float(summary["decision_seconds"]),
                }
            )
            traces.extend({**identity, **row} for row in trajectory)
    destination.mkdir(parents=True, exist_ok=False)
    result = pd.DataFrame(rows)
    result.to_csv(destination / "summary.csv", index=False)
    pd.DataFrame(traces).to_csv(destination / "traces.csv", index=False)
    write_json(
        destination / "run.json",
        {
            **bank_manifest(bank, selection),
            "comparison_signature": comparison_signature(bank, selection),
            "fit_ids": request.training.iter_id.tolist(),
            "strategies": list(request.strategies),
            "prediction_seconds": prediction_seconds,
            "predicted_designs": len(table),
            "decision_space": asdict(bank.space),
        },
    )
    return result
