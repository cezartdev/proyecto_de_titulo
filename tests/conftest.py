"""Tablas sintéticas pequeñas con el mismo contrato que el Excel original."""

from pathlib import Path

import pandas as pd
import pytest

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
