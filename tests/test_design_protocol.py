"""Partición por diseños únicos, trazabilidad y rechazo del protocolo retirado."""

import json
from pathlib import Path

import pandas as pd
import pytest

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.datos import PreparationConfig, prepare_dataset
from truss_research.datos.auditoria import audit_split_overlap, design_keys
from truss_research.datos.particion import split_by_design, verify_partition
from truss_research.datos.preparacion import normalize_columns
from truss_research.datos.procesados import load_prepared_split
from truss_research.datos.schemas import (
    DEFAULT_SOURCE,
    DatasetError,
    DatasetSpec,
    SplitConfig,
    TableSplit,
)
from truss_research.datos.validacion import add_identifiers
from truss_research.surrogates import PredictorConfig, train_predictor


def test_design_protocol_is_default_and_problem_protocol_is_rejected() -> None:
    assert PreparationConfig().split.protocol == "design"
    with pytest.raises(DatasetError, match="desconocido"):
        SplitConfig(protocol="generation")


@pytest.mark.skipif(
    not DEFAULT_SOURCE.exists(), reason="Excel local no distribuido en Git"
)
def test_real_design_protocol_has_no_repeated_designs(tmp_path: Path) -> None:
    result = prepare_dataset(
        PreparationConfig(
            destination=tmp_path / "designs",
            split=SplitConfig(protocol="design"),
        )
    )
    assert len(result.table) == 1273
    assert [len(table) for table in result.split.as_mapping().values()] == [
        1018,
        127,
        128,
    ]
    assert not any(audit_split_overlap(result.split).values())
    assert all(
        table["web_config"].nunique() == 5
        for table in result.split.as_mapping().values()
    )
    assert len(pd.read_csv(result.output_paths["design_membership"])) == 5000


@pytest.fixture
def unique_table(raw_table: pd.DataFrame) -> pd.DataFrame:
    table = pd.concat([raw_table] * 4, ignore_index=True).iloc[:100].copy()
    table["Unnamed: 0"] = [
        f"{{{index // 10};{index % 10}}}" for index in range(len(table))
    ]
    table["L-R Height"] = [0.2 + index * 0.03 for index in range(len(table))]
    return table


def test_design_partition_is_disjoint_stratified_and_reproducible(
    unique_table: pd.DataFrame,
) -> None:
    table = add_identifiers(normalize_columns(unique_table))
    split = split_by_design(table, SplitConfig(protocol="design"))
    repeated = split_by_design(table, SplitConfig(protocol="design"))
    assert [len(part) for part in split.as_mapping().values()] == [80, 10, 10]
    assert not any(audit_split_overlap(split).values())
    for name, part in split.as_mapping().items():
        assert part["web_config"].nunique() == 5
        pd.testing.assert_frame_equal(part, repeated.as_mapping()[name])
    assert set(split.training["generation"]) & set(split.validation["generation"])


def test_design_partition_rejects_shared_and_repeated_inputs(
    unique_table: pd.DataFrame,
) -> None:
    table = add_identifiers(normalize_columns(unique_table))
    table.loc[1, list(INPUT_COLUMNS)] = table.loc[0, list(INPUT_COLUMNS)]
    split = TableSplit(table.iloc[:1], table.iloc[1:2], table.iloc[2:])
    with pytest.raises(DatasetError, match="diseños repetidos"):
        verify_partition(table, split, "design")
    with pytest.raises(DatasetError, match="sin diseños repetidos"):
        split_by_design(table, SplitConfig(protocol="design"))


def test_small_categories_fail_without_silent_fallback(raw_table: pd.DataFrame) -> None:
    table = add_identifiers(normalize_columns(raw_table)).drop_duplicates(
        list(INPUT_COLUMNS)
    )
    with pytest.raises(DatasetError, match="estratificar"):
        split_by_design(table, SplitConfig(protocol="design"))


def test_prepare_unique_designs_keeps_original_membership(
    unique_table: pd.DataFrame, tmp_path: Path
) -> None:
    repeated = unique_table.copy()
    repeated["Unnamed: 0"] = [
        f"{{{10 + index // 10};{index % 10}}}" for index in range(len(repeated))
    ]
    source = tmp_path / "data.xlsx"
    pd.concat([unique_table, repeated]).to_excel(
        source, sheet_name="TOTAL DATA", index=False
    )
    destination = tmp_path / "designs"
    result = prepare_dataset(
        PreparationConfig(
            source,
            destination,
            dataset=DatasetSpec(20, 10),
            split=SplitConfig(protocol="design"),
        )
    )
    assert len(result.table) == 100
    membership = pd.read_csv(result.output_paths["design_membership"])
    assert len(membership) == 200
    assert membership["iter_id"].nunique() == 200
    assert not membership["representative_iter_id"].isna().any()
    assert membership.groupby(list(INPUT_COLUMNS))["split"].nunique().eq(1).all()
    manifest = json.loads(result.output_paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["design_audit"]["rows"] == 200
    assert manifest["design_audit"]["unique_designs"] == 100
    assert not any(manifest["design_overlap"].values())
    loaded = load_prepared_split(destination)
    assert (
        len(design_keys(loaded.training).intersection(design_keys(loaded.validation)))
        == 0
    )
    trained = train_predictor(
        PredictorConfig(destination, tmp_path / "baseline", model="dummy", folds=2)
    )
    assert len(trained.predictions) == 10
