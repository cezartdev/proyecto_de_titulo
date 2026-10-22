"""Persistencia y evaluación descriptiva de intervalos, sin reajustar predictores."""

from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.schemas import TableSplit
from truss_research.surrogates.calibracion import SplitConformal
from truss_research.surrogates.experimentos import predict, write_json


def calibrate_predictor(model: Pipeline, split: TableSplit, destination: Path) -> None:
    calibration = split.calibration
    if calibration is None:
        raise ValueError("Falta la reserva independiente de calibración")
    predictions = predict(model, calibration.loc[:, list(INPUT_COLUMNS)])
    conformal = SplitConformal(
        calibration.loc[:, list(OUTPUT_COLUMNS)].to_numpy() - predictions.to_numpy()
    )
    write_json(
        destination / "calibration.json",
        {
            "schema_version": 1,
            "calibration_ids": calibration.iter_id.tolist(),
            "residuals": conformal.residuals.tolist(),
            "coverages": list(conformal.coverages),
            "scope": "marginal_under_exchangeability_not_adaptive_or_conditional_guarantee",
        },
    )


def load_calibrator(directory: Path) -> SplitConformal:
    import numpy as np

    payload = read_artifact(directory / "calibration.json")
    if payload["schema_version"] != 1:
        raise ValueError("Formato de calibración desconocido")
    return SplitConformal(
        np.asarray(payload["residuals"], dtype=float),
        tuple(cast(list[float], payload["coverages"])),
    )


def export_intervals(
    locations: tuple[Path, Path], table: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    source, destination = locations
    intervals = load_calibrator(source).table(predictions, table)
    identifiers = table.iter_id.to_dict()
    intervals["iter_id"] = intervals.row_index.map(identifiers)
    intervals["web_config"] = intervals.row_index.map(table.web_config.to_dict())
    from truss_research.datos.distancias import distance_regions
    from truss_research.datos.procesados import load_design_split
    from truss_research.surrogates.experimentos import read_run

    run_path = source / "run.json"
    if run_path.is_file():
        training = load_design_split(
            Path(str(read_run(source)["config"]["data_dir"]))
        ).training
        regions = pd.Series(
            distance_regions(
                training.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=float),
                table.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=float),
            ),
            index=table.iter_id,
        )
        intervals["distance_region"] = intervals.iter_id.map(regions)
    intervals.to_csv(destination / "prediction_intervals.csv", index=False)
