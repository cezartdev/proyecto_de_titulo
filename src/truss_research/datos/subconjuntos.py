"""Subconjuntos anidados con cobertura categórica y tamaños reproducibles."""

import numpy as np
import pandas as pd


def nested_subsets(
    table: pd.DataFrame, sizes: tuple[int, ...], seed: int
) -> dict[int, pd.DataFrame]:
    if (
        not table.iter_id.is_unique
        or not sizes
        or min(sizes) < table.web_config.nunique()
        or max(sizes) > len(table)
    ):
        raise ValueError(
            "Tamaños incompatibles con diseños únicos y categorías disponibles"
        )
    rng = np.random.default_rng(seed)
    ordered = table.sort_values("iter_id")
    categories = sorted(ordered.web_config.unique())
    pools = [
        rng.permutation(ordered.index[ordered.web_config.eq(category)]).tolist()
        for category in categories
    ]
    proportions = np.array([len(pool) for pool in pools], dtype=float) / len(table)
    counts = np.ones(len(pools), dtype=int)
    indices = [pool[0] for pool in pools]
    for _ in range(len(indices), max(sizes)):
        deficit = (len(indices) + 1) * proportions - counts
        deficit[[counts[i] == len(pool) for i, pool in enumerate(pools)]] = -np.inf
        category = int(np.argmax(deficit))
        indices.append(pools[category][counts[category]])
        counts[category] += 1
    return {size: table.loc[indices[:size]].copy() for size in sizes}
