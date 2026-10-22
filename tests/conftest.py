"""Tablas sintéticas pequeñas con el mismo contrato que el Excel original."""

from pathlib import Path

import pandas as pd
import pytest

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos import PreparationConfig, prepare_dataset
from truss_research.datos.schemas import (
    COLUMN_MAPPING,
    RAW_ID_COLUMN,
    DatasetSpec,
    SplitConfig,
)


@pytest.fixture
def raw_table() -> pd.DataFrame:
    rows: list[dict[str, str | float | int]] = []
    for generation in range(10):
        for solution in range(3):
            rows.append(
                {
                    RAW_ID_COLUMN: f"{{{generation};{solution}}}",
                    "Mass": 1000.0,
                    "Average Util": 0.5,
                    "Z Disp": 0.02,
                    "Web Config": generation % 5,
                    "N of Trusses": 10,
                    "L-R Height": 1.0,
                    "LM-RM Height": 2.0,
                    "M Height": 3.0,
                    "Subdiv1": 2,
                    "Subdiv2": 3,
                }
            )
    return pd.DataFrame(rows, columns=[RAW_ID_COLUMN, *COLUMN_MAPPING])


@pytest.fixture
def design_data(raw_table: pd.DataFrame, tmp_path: Path) -> Path:
    table = pd.concat([raw_table] * 4, ignore_index=True).iloc[:100].copy()
    table[RAW_ID_COLUMN] = [
        f"{{{index // 10};{index % 10}}}" for index in range(len(table))
    ]
    table["L-R Height"] = 0.2 + table.index * 0.03
    table["Mass"] = 1000.0 + table.index * 10.0
    table["Average Util"] = 0.2 + table.index * 0.005
    table["Z Disp"] = 0.01 + table.index * 0.0005
    source = tmp_path / "fixture.xlsx"
    table.to_excel(source, sheet_name="TOTAL DATA", index=False)
    destination = tmp_path / "designs"
    prepare_dataset(
        PreparationConfig(
            source,
            destination,
            dataset=DatasetSpec(10, 10),
            split=SplitConfig(protocol="design"),
        )
    )
    return destination


@pytest.fixture
def temporal_table() -> pd.DataFrame:
    rows = []
    for generation in range(20):
        for solution in range(5):
            height = 0.5 + generation * 0.04 + solution * 0.003
            if generation >= 5 and solution == 0:
                height = 0.5
            rows.append(
                dict(
                    zip(
                        [
                            "iter_id",
                            "generation",
                            "solution",
                            *INPUT_COLUMNS,
                            *OUTPUT_COLUMNS,
                        ],
                        [
                            f"{{{generation};{solution}}}",
                            generation,
                            solution,
                            0,
                            10,
                            height,
                            1.0,
                            2.0,
                            2,
                            3,
                            2000 - 100 * height,
                            0.5,
                            0.01,
                        ],
                        strict=True,
                    )
                )
            )
    return pd.DataFrame(rows)


@pytest.fixture
def prepared_temporal(
    temporal_table: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    extra = temporal_table.copy()
    extra.solution += 5
    extra["iter_id"] = [
        f"{{{g};{s}}}" for g, s in zip(extra.generation, extra.solution, strict=True)
    ]
    extra.lr_height_m += 0.001
    extra.mass_kg = 2000 - 100 * extra.lr_height_m
    table = pd.concat([temporal_table, extra], ignore_index=True)
    source = tmp_path / "fixture.xlsx"
    source.write_bytes(b"synthetic-fixture")
    monkeypatch.setattr(
        "truss_research.datos.preparacion.load_raw_table", lambda *args: table
    )
    monkeypatch.setattr(
        "truss_research.datos.preparacion.normalize_columns", lambda value: value
    )
    destination = tmp_path / "temporal"
    prepare_dataset(
        PreparationConfig(
            source,
            destination,
            dataset=DatasetSpec(20, 10),
            split=SplitConfig(protocol="first_appearance"),
            calibration_fraction=0.2,
        )
    )
    return destination
