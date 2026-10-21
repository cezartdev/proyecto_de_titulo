"""Identificación exacta de diseños y restricciones de independencia."""

import pandas as pd
import pytest

from truss_research.datos.auditoria import (
    audit_designs,
    design_keys,
    select_unique_designs,
)
from truss_research.datos.preparacion import normalize_columns
from truss_research.datos.schemas import DatasetError
from truss_research.datos.validacion import add_identifiers


def test_repeated_designs_connect_different_problems(raw_table: pd.DataFrame) -> None:
    table = add_identifiers(normalize_columns(raw_table))
    audit = audit_designs(table)
    assert audit.unique_designs == 5
    assert audit.repeated_rows == 25
    assert audit.generation_component_sizes == (2, 2, 2, 2, 2)
    assert audit.conflicting_designs == 0
    assert len(select_unique_designs(table)) == 5


def test_conflicting_responses_prevent_silent_deduplication(
    raw_table: pd.DataFrame,
) -> None:
    table = add_identifiers(normalize_columns(raw_table))
    table.loc[1, "mass_kg"] += 100
    assert audit_designs(table).conflicting_designs == 1
    with pytest.raises(DatasetError, match="respuestas distintas"):
        select_unique_designs(table)


def test_design_identity_preserves_small_coordinate_differences(
    raw_table: pd.DataFrame,
) -> None:
    table = add_identifiers(normalize_columns(raw_table))
    table.loc[1, "lr_height_m"] += 1e-12
    assert len(design_keys(table).unique()) == 6
