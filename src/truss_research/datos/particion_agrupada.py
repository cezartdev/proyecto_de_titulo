"""Separación por generaciones usando todas las pertenencias y purga de cruces."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from truss_research.datos.particion import IndexArray
from truss_research.datos.schemas import DatasetError


def forward_generation_folds(
    table: pd.DataFrame, membership: pd.DataFrame, folds: int
) -> list[tuple[IndexArray, IndexArray]]:
    """Ventanas progresivas que excluyen diseños compartidos entre periodos."""
    relevant = membership.loc[membership.representative_iter_id.isin(table.iter_id)]
    generations = np.array(sorted(relevant.generation.unique()))
    if folds < 2 or len(generations) < folds + 1:
        raise DatasetError("Generaciones insuficientes para validación progresiva")
    blocks = np.array_split(generations, folds + 1)
    spans = relevant.groupby("representative_iter_id").generation.agg(["min", "max"])
    first = table.iter_id.map(spans["min"])
    last = table.iter_id.map(spans["max"])
    if first.isna().any() or last.isna().any():
        raise DatasetError("Faltan pertenencias completas de algunos diseños")
    results = []
    for block in blocks[1:]:
        fitting = np.flatnonzero(last.lt(block[0]))
        held = np.flatnonzero(first.ge(block[0]) & last.le(block[-1]))
        if not len(fitting) or not len(held):
            raise DatasetError("Un pliegue temporal quedó vacío tras la purga")
        results.append((fitting, held))
    return results


@dataclass(frozen=True)
class PurgedPartition:
    tables: dict[str, pd.DataFrame]
    assignments: pd.DataFrame
    excluded: pd.DataFrame


def membership_roles(membership: pd.DataFrame, roles: dict[int, str]) -> pd.DataFrame:
    required = {"generation", "representative_iter_id"}
    if not required <= set(membership) or not set(
        membership.generation.unique()
    ) <= set(roles):
        raise DatasetError("Pertenencias incompletas para la separación agrupada")
    measured = membership.assign(role=membership.generation.map(roles))
    counts = measured.groupby("representative_iter_id").role.nunique()
    assignments = measured.drop_duplicates("representative_iter_id")[
        ["representative_iter_id", "role"]
    ].set_index("representative_iter_id")
    assignments.loc[counts.gt(1), "role"] = "purged"
    return assignments


def purged_generation_split(
    table: pd.DataFrame, membership: pd.DataFrame, seed: int
) -> PurgedPartition:
    generations = np.array(sorted(membership.generation.unique()))
    if len(generations) < 10:
        raise DatasetError(
            "La sensibilidad 80/10/10 requiere al menos diez generaciones"
        )
    np.random.default_rng(seed).shuffle(generations)
    train_count, validation_count = (
        int(len(generations) * 0.8),
        max(1, int(len(generations) * 0.1)),
    )
    roles = {
        int(generation): "train"
        if i < train_count
        else "validation"
        if i < train_count + validation_count
        else "test"
        for i, generation in enumerate(generations)
    }
    assignments = membership_roles(membership, roles)
    selected = table.assign(group_partition=table.iter_id.map(assignments.role))
    if selected.group_partition.isna().any():
        raise DatasetError("Hay diseños sin pertenencia original")
    tables = {
        name: selected.loc[selected.group_partition.eq(name)]
        .drop(columns="group_partition")
        .copy()
        for name in ("train", "validation", "test")
    }
    if any(value.empty for value in tables.values()):
        raise DatasetError("La purga dejó una partición vacía")
    excluded = selected.loc[
        selected.group_partition.eq("purged"), ["iter_id", "web_config"]
    ].assign(reason="design_shared_between_generation_partitions")
    return PurgedPartition(
        tables,
        pd.DataFrame({"generation": list(roles), "partition": list(roles.values())}),
        excluded,
    )


def purged_group_folds(
    table: pd.DataFrame, membership: pd.DataFrame, context: tuple[int, int]
) -> list[tuple[np.ndarray, np.ndarray]]:
    folds, seed = context
    relevant = membership.loc[membership.representative_iter_id.isin(table.iter_id)]
    generations = np.array(sorted(relevant.generation.unique()))
    if folds < 2 or len(generations) < folds:
        raise DatasetError("Problemas insuficientes para los pliegues agrupados")
    np.random.default_rng(seed).shuffle(generations)
    results = []
    for group in np.array_split(generations, folds):
        roles = {int(p): "held" if p in group else "fit" for p in generations}
        labels = table.iter_id.map(membership_roles(relevant, roles).role)
        fitting, held = (
            np.flatnonzero(labels.eq("fit")),
            np.flatnonzero(labels.eq("held")),
        )
        if not len(fitting) or not len(held):
            raise DatasetError("Un pliegue agrupado quedó vacío tras la purga")
        results.append((fitting, held))
    return results
