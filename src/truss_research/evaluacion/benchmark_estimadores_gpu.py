"""Ajustes emparejados de MLP y boosting, sin selección sobre el conjunto de prueba."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procesados import JsonValue, load_design_split
from truss_research.evaluacion.benchmark_tabpfn import predictor_directory
from truss_research.evaluacion.recursos import timed
from truss_research.surrogates.experimentos import config_from_run, predict, write_json
from truss_research.surrogates.modelos import build_predictor

DEFAULT_DESTINATION = Path("reports/piloto_estimadores_gpu_001")
PILOT_MODELS = ("mlp", "catboost", "xgboost")
PILOT_SEEDS = (42, 43, 44)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estudio", type=Path, default=Path("artifacts/estudio"))
    parser.add_argument("--destino", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    args.destino.mkdir(parents=True, exist_ok=False)
    split = load_design_split(Path("data/processed_designs"))
    features = split.training.loc[:, list(INPUT_COLUMNS)]
    responses = split.training.loc[:, list(OUTPUT_COLUMNS)]
    validation = split.validation.loc[:, list(INPUT_COLUMNS)]
    truth = split.validation.loc[:, list(OUTPUT_COLUMNS)].to_numpy()
    rows: list[dict[str, JsonValue]] = []
    for family in PILOT_MODELS:
        root = predictor_directory(args.estudio, f"predictor/{family}/42")
        base = config_from_run(root)
        for seed in PILOT_SEEDS:
            for device in ("cpu", "cuda") if seed % 2 == 0 else ("cuda", "cpu"):
                config = replace(base, seed=seed, device=device)
                model = build_predictor(config)
                with threadpool_limits(limits=1):
                    _, fit = timed(lambda: model.fit(features, responses.to_numpy()))
                    values, inference = timed(
                        lambda: predict(model, validation).to_numpy()
                    )
                rows.append(
                    {
                        "family": family,
                        "seed": seed,
                        "device": device,
                        "fit": fit,
                        "inference": inference,
                        "mae_by_target": np.mean(
                            np.abs(truth - values), axis=0
                        ).tolist(),
                    }
                )
                write_json(
                    args.destino / "samples.json", {"status": "running", "rows": rows}
                )
    write_json(args.destino / "samples.json", {"status": "complete", "rows": rows})


if __name__ == "__main__":
    main()
