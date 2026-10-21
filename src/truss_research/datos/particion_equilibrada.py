"""Partición cronológica fija y control aleatorio de igual tamaño y cobertura."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from truss_research.datos.particion_agrupada import membership_roles
from truss_research.datos.schemas import (
    GROUP_PARTITIONS,
    DatasetError,
    GenerationSplitParameters,
    TableSplit,
)


@dataclass(frozen=True)
class BalancedPartition:
    tables: dict[str, pd.DataFrame]
    assignments: pd.DataFrame
    excluded: pd.DataFrame
    candidates: pd.DataFrame


def chronological_generation_split(
    table: pd.DataFrame,
    membership: pd.DataFrame,
    context: tuple[int, GenerationSplitParameters],
) -> BalancedPartition:
    _, settings = context
    generations = np.array(sorted(membership.generation.unique()))
    counts = np.floor(np.array(settings.fractions) * len(generations)).astype(int)
    counts[-1] += len(generations) - counts.sum()
    if np.any(counts == 0):
        raise DatasetError("Generaciones insuficientes para cuatro conjuntos")
    roles = dict(
        zip(
            generations.tolist(),
            np.repeat(GROUP_PARTITIONS, counts).tolist(),
            strict=True,
        )
    )
    labels = table.iter_id.map(membership_roles(membership, roles).role)
    sizes = {name: int(labels.eq(name).sum()) for name in GROUP_PARTITIONS}
    viable = (
        sizes["train"] >= settings.min_training
        and sizes["calibration"] >= settings.min_calibration
        and min(sizes["validation"], sizes["test"]) >= settings.min_evaluation
    )
    if not viable:
        raise DatasetError(f"La partición cronológica no cumple los mínimos: {sizes}")
    return BalancedPartition(
        {name: table.loc[labels.eq(name)].copy() for name in GROUP_PARTITIONS},
        pd.DataFrame({"generation": list(roles), "partition": list(roles.values())}),
        table.loc[labels.eq("purged"), ["iter_id", "web_config"]].assign(
            reason="design_shared_between_generation_partitions"
        ),
        pd.DataFrame(
            [
                {
                    "attempt": 0,
                    "viable": True,
                    **sizes,
                    "retained": sum(sizes.values()),
                    "selected": True,
                }
            ]
        ),
    )


def verify_group_exclusion(
    tables: dict[str, pd.DataFrame], membership: pd.DataFrame
) -> None:
    groups = {
        name: set(
            membership.loc[
                membership.representative_iter_id.isin(table.iter_id), "generation"
            ]
        )
        for name, table in tables.items()
    }
    if any(groups[a] & groups[b] for a, b in combinations(groups, 2)):
        raise DatasetError("Las pertenencias completas cruzan los conjuntos agrupados")


def matched_design_control(
    table: pd.DataFrame, split: TableSplit, seed: int
) -> TableSplit:
    held = pd.concat([split.calibration, split.validation, split.testing]).iter_id
    pool = table.loc[~table.iter_id.isin(held)].sort_values("iter_id")
    rng = np.random.default_rng(seed)
    selected = []
    for category, count in (
        split.training.web_config.value_counts().sort_index().items()
    ):
        candidates = pool.loc[pool.web_config.eq(category)]
        selected.append(
            candidates.iloc[rng.choice(len(candidates), int(count), replace=False)]
        )
    return TableSplit(
        pd.concat(selected).copy(), split.validation, split.testing, split.calibration
    )
