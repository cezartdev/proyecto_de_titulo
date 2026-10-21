"""Auditoría exacta de diseños repetidos y dependencia entre generaciones."""

from dataclasses import dataclass
from itertools import combinations

import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.schemas import DatasetError, TableSplit


@dataclass(frozen=True)
class DesignAudit:
    rows: int
    unique_designs: int
    repeated_rows: int
    conflicting_designs: int
    max_response_ranges: dict[str, float]
    unique_designs_by_web_config: dict[int, int]
    generation_component_sizes: tuple[int, ...]


def design_keys(table: pd.DataFrame) -> pd.MultiIndex:
    """Identidad exacta de las siete entradas, sin redondeos ni respuestas."""
    return pd.MultiIndex.from_frame(table.loc[:, list(INPUT_COLUMNS)])


def generation_components(table: pd.DataFrame) -> tuple[int, ...]:
    generations, generation_values = pd.factorize(table["generation"])
    designs, design_values = pd.factorize(design_keys(table))
    membership = csr_matrix(
        ([1] * len(table), (generations, designs)),
        shape=(len(generation_values), len(design_values)),
    )
    _, labels = connected_components(membership @ membership.T, directed=False)
    sizes = pd.Series(labels).value_counts().sort_values(ascending=False)
    return tuple(int(value) for value in sizes)


def audit_designs(table: pd.DataFrame) -> DesignAudit:
    if table.empty:
        raise DatasetError("No se puede auditar una tabla vacía")
    unique = table.drop_duplicates(list(INPUT_COLUMNS))
    grouped = table.groupby(list(INPUT_COLUMNS))[list(OUTPUT_COLUMNS)]
    ranges = grouped.max() - grouped.min()
    return DesignAudit(
        rows=len(table),
        unique_designs=len(unique),
        repeated_rows=len(table) - len(unique),
        conflicting_designs=int(ranges.gt(0).any(axis=1).sum()),
        max_response_ranges={
            name: float(ranges[name].max()) for name in OUTPUT_COLUMNS
        },
        unique_designs_by_web_config={
            int(key): int(value)
            for key, value in unique["web_config"].value_counts().sort_index().items()
        },
        generation_component_sizes=generation_components(table),
    )


def audit_split_overlap(split: TableSplit) -> dict[str, int]:
    tables = split.as_mapping()
    pairs = combinations(tables, 2)
    return {
        f"{left}_to_{right}_shared_designs": len(
            design_keys(tables[left])
            .unique()
            .intersection(design_keys(tables[right]).unique())
        )
        for left, right in pairs
    }


def select_unique_designs(table: pd.DataFrame) -> pd.DataFrame:
    """Conserva un representante por diseño solo si sus respuestas coinciden."""
    audit = audit_designs(table)
    if audit.conflicting_designs:
        raise DatasetError(
            "Un mismo diseño tiene respuestas distintas; revisar contexto antes de deduplicar"
        )
    return (
        table.drop_duplicates(list(INPUT_COLUMNS))
        .sort_values(list(INPUT_COLUMNS))
        .reset_index(drop=True)
    )


def design_membership(source: pd.DataFrame, split: TableSplit) -> pd.DataFrame:
    representatives = pd.concat(
        [table.assign(split=name) for name, table in split.as_mapping().items()],
        ignore_index=True,
    )
    selected = representatives.loc[:, [*INPUT_COLUMNS, "iter_id", "split"]].rename(
        columns={"iter_id": "representative_iter_id"}
    )
    return source.loc[:, ["iter_id", "generation", "solution", *INPUT_COLUMNS]].merge(
        selected,
        on=list(INPUT_COLUMNS),
        how="left",
        validate="many_to_one",
    )
