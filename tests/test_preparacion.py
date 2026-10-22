"""Integridad, conservación de diseños inviables y reproducibilidad del flujo."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from truss_research.cerchas.schemas import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    StructuralLimits,
    StructuralResponse,
    assess_response,
)
from truss_research.datos import PreparationConfig, prepare_dataset
from truss_research.datos.particion import split_by_design
from truss_research.datos.preparacion import load_raw_table, normalize_columns
from truss_research.datos.schemas import (
    COMPLIANCE_COLUMNS,
    DEFAULT_SOURCE,
    IDENTIFIER_COLUMNS,
    RAW_ID_COLUMN,
    DatasetError,
    DatasetSpec,
    SplitConfig,
)
from truss_research.datos.validacion import (
    add_identifiers,
    add_threshold_indicators,
    validate_structure,
    validate_values,
)


@pytest.mark.parametrize("identifier", ["x{0;0}", "{0;0}x", "{a;0}", "{0;-1}", None])
def test_rejects_incomplete_identifiers(
    raw_table: pd.DataFrame, identifier: str | None
) -> None:
    raw_table.loc[0, RAW_ID_COLUMN] = identifier
    with pytest.raises(DatasetError, match="Identificadores inválidos"):
        add_identifiers(normalize_columns(raw_table))


def test_duplicate_ids_are_rejected_even_with_distinct_spelling(
    raw_table: pd.DataFrame,
) -> None:
    raw_table.loc[1, RAW_ID_COLUMN] = "{00;00}"
    table = add_identifiers(normalize_columns(raw_table))
    with pytest.raises(DatasetError, match="duplicados"):
        validate_structure(table, DatasetSpec(10, 3))


def test_excel_identifier_whitespace_is_normalized(raw_table: pd.DataFrame) -> None:
    raw_table.loc[0, RAW_ID_COLUMN] = "            {0;0}  "
    result = add_identifiers(normalize_columns(raw_table))
    assert result.loc[0, "iter_id"] == "{0;0}"
    assert result.loc[0, "generation"] == 0


@pytest.mark.parametrize("replacement", ["{0;3}", "{0;50}"])
def test_requires_all_solution_ids(raw_table: pd.DataFrame, replacement: str) -> None:
    raw_table.loc[0, RAW_ID_COLUMN] = replacement
    with pytest.raises(DatasetError, match="intervalo"):
        validate_structure(
            add_identifiers(normalize_columns(raw_table)), DatasetSpec(10, 3)
        )


def test_requires_expected_counts(raw_table: pd.DataFrame) -> None:
    table = add_identifiers(normalize_columns(raw_table))
    with pytest.raises(DatasetError, match="generaciones"):
        validate_structure(table, DatasetSpec(100, 3))
    with pytest.raises(DatasetError, match="soluciones"):
        validate_structure(table.iloc[1:], DatasetSpec(10, 3))


def test_only_empty_separator_columns_are_removed(raw_table: pd.DataFrame) -> None:
    raw_table["Unnamed: 4"] = float("nan")
    assert "Unnamed: 4" not in normalize_columns(raw_table)
    raw_table.loc[0, "Unnamed: 4"] = 123.0
    with pytest.raises(DatasetError, match="separadores con datos"):
        normalize_columns(raw_table)


def test_missing_and_unexpected_columns_fail(raw_table: pd.DataFrame) -> None:
    with pytest.raises(DatasetError, match="Faltan columnas"):
        normalize_columns(raw_table.drop(columns="Mass"))
    raw_table["extra"] = float("nan")
    with pytest.raises(DatasetError, match="inesperadas"):
        normalize_columns(raw_table)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("mass_kg", float("nan")),
        ("util_ratio", float("inf")),
        ("z_disp_m", -0.1),
        ("mass_kg", 0.0),
        ("web_config", 1.5),
        ("n_trusses", 3.5),
        ("subdiv1", 0),
        ("m_height_m", 4.0),
        ("mass_kg", "texto"),
    ],
)
def test_invalid_values_fail(
    raw_table: pd.DataFrame, column: str, value: float | str
) -> None:
    table = normalize_columns(raw_table).astype({column: "object"})
    table.loc[0, column] = value
    with pytest.raises(DatasetError):
        validate_values(table)


def test_infeasible_rows_survive_and_limits_are_inclusive(
    raw_table: pd.DataFrame,
) -> None:
    table = normalize_columns(raw_table)
    table.loc[0, ["mass_kg", "util_ratio", "z_disp_m"]] = [40000, 1.2, 0.3]
    table.loc[1, ["util_ratio", "z_disp_m"]] = [1.0, 20 / 700]
    result = add_threshold_indicators(validate_values(table), StructuralLimits())
    assert len(result) == len(raw_table)
    assert not result.loc[0, "meets_thresholds"]
    assert result.loc[1, "meets_thresholds"]
    assert assess_response(
        StructuralResponse(1000, 1, 20 / 700), StructuralLimits()
    ).combined
    assert not assess_response(
        StructuralResponse(1000, 1.2, 0.3), StructuralLimits()
    ).combined


def test_partition_is_complete_disjoint_and_reproducible(
    raw_table: pd.DataFrame,
) -> None:
    raw_table = raw_table.copy()
    raw_table["L-R Height"] = 0.2 + raw_table.index * 0.03
    table = add_identifiers(normalize_columns(raw_table))
    first = split_by_design(
        table,
        SplitConfig(train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2),
    )
    second = split_by_design(
        table,
        SplitConfig(train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2),
    )
    subsets = list(first.as_mapping().values())
    assert [len(subset) for subset in subsets] == [18, 6, 6]
    assert sorted(pd.concat(subsets)["iter_id"]) == sorted(table["iter_id"])
    for name, subset in first.as_mapping().items():
        assert_frame_equal(subset, second.as_mapping()[name])


@pytest.mark.parametrize(
    "config",
    [
        {"train_fraction": 0.7},
        {"test_fraction": float("nan")},
        {"seed": -1},
        {"seed": 2**32},
    ],
)
def test_invalid_split_config_fails(config: dict[str, float | int]) -> None:
    with pytest.raises(DatasetError):
        SplitConfig(**config)


def test_exports_and_manifest(raw_table: pd.DataFrame, tmp_path: Path) -> None:
    raw_table = raw_table.copy()
    raw_table["L-R Height"] = 0.2 + raw_table.index * 0.03
    source = tmp_path / "source.xlsx"
    raw_table.to_excel(source, sheet_name="TOTAL DATA", index=False)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    config = PreparationConfig(
        source,
        tmp_path / "processed",
        dataset=DatasetSpec(10, 3),
        split=SplitConfig(
            train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2
        ),
    )
    result = prepare_dataset(config)
    manifest = json.loads(result.output_paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == original_hash
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    assert manifest["split"]["seed"] == 42
    assert manifest["diagnostics"]["train"]["rows"] == 18
    assert set(manifest["diagnostics"]["train"]["web_config_counts"]) == set("01234")
    assert len(pd.read_csv(result.output_paths["full"])) == 30
    assert set(IDENTIFIER_COLUMNS + COMPLIANCE_COLUMNS).isdisjoint(INPUT_COLUMNS)
    assert manifest["output_columns"] == list(OUTPUT_COLUMNS)
    assert manifest["input_columns"] == list(INPUT_COLUMNS)


def test_invalid_input_does_not_create_destination(
    raw_table: pd.DataFrame, tmp_path: Path
) -> None:
    source = tmp_path / "source.xlsx"
    raw_table.drop(columns="Mass").to_excel(
        source, sheet_name="TOTAL DATA", index=False
    )
    destination = tmp_path / "processed"
    with pytest.raises(DatasetError):
        prepare_dataset(
            PreparationConfig(source, destination, dataset=DatasetSpec(10, 3))
        )
    assert not destination.exists()


def test_read_errors_are_explicit(tmp_path: Path) -> None:
    source = tmp_path / "invalid.xlsx"
    with pytest.raises(DatasetError, match="origen"):
        load_raw_table(source)
    source.write_text("No es un Excel", encoding="utf-8")
    with pytest.raises(DatasetError, match="No se pudo leer"):
        load_raw_table(source)


def test_cli_reports_missing_source(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "truss_research",
            "preparar-datos",
            "--origen",
            str(tmp_path / "missing.xlsx"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "Preparaci" in completed.stderr
    assert "Traceback" not in completed.stderr


@pytest.mark.skipif(
    os.getenv("TRUSS_REAL_DATA_TEST") != "1" or not DEFAULT_SOURCE.exists(),
    reason="Integración optativa con el Excel autorizado; no se prepara por defecto",
)
def test_real_dataset(tmp_path: Path) -> None:
    result = prepare_dataset(
        PreparationConfig(
            destination=tmp_path / "processed",
            split=SplitConfig(protocol="first_appearance"),
            calibration_fraction=0.2,
        )
    )
    assert len(result.table) == 1273
    assert {name: len(table) for name, table in result.split.as_mapping().items()} == {
        "train": 901,
        "calibration": 166,
        "validation": 98,
        "test": 108,
    }
    assert not result.table["meets_z_threshold"].all()


def test_preparation_reuses_matching_data_and_never_replaces(
    raw_table: pd.DataFrame, tmp_path: Path
) -> None:
    raw_table = raw_table.copy()
    raw_table["L-R Height"] = 0.2 + raw_table.index * 0.03
    from dataclasses import replace

    from truss_research.datos.preparacion import ensure_dataset

    source = tmp_path / "source.xlsx"
    raw_table.to_excel(source, sheet_name="TOTAL DATA", index=False)
    config = PreparationConfig(
        source,
        tmp_path / "prepared",
        dataset=DatasetSpec(10, 3),
        split=SplitConfig(
            train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2
        ),
    )
    ensure_dataset(config)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in config.destination.iterdir()
    }
    assert ensure_dataset(config) == config.destination
    assert before == {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in config.destination.iterdir()
    }
    with pytest.raises(ValueError, match="no se reemplazan"):
        prepare_dataset(config)
    with pytest.raises(ValueError, match="otro origen o protocolo"):
        ensure_dataset(
            replace(
                config,
                split=SplitConfig(
                    train_fraction=0.6,
                    validation_fraction=0.2,
                    test_fraction=0.2,
                    seed=43,
                ),
            )
        )
    raw_table.loc[0, "Mass"] += 1
    raw_table.to_excel(source, sheet_name="TOTAL DATA", index=False)
    with pytest.raises(ValueError, match="otro origen o protocolo"):
        ensure_dataset(config)
    assert before == {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in config.destination.iterdir()
    }
