"""Remuestreo pareado por bloques de catálogo y semillas de política."""

import numpy as np
import pandas as pd

from truss_research.optimizacion.contratos import AnalysisParameters

INTERVAL_QUANTILES = (0.025, 0.975)
REFERENCE_STRATEGIES = (
    "random",
    "ranking",
    "greedy",
    "nsga2",
    "nsga2_surrogate",
    "ensemble_mean",
)


def paired_contrast(
    subjects: pd.DataFrame, reference: pd.DataFrame, context: tuple[str, bool]
) -> pd.DataFrame:
    name, policy_pair = context
    keys = ["split", "episode_id", "catalog_seed", "budget"]
    merge_keys = [*keys, "policy_training_seed"] if policy_pair else keys
    paired = subjects.merge(
        reference[[*merge_keys, "normalized_auc"]],
        on=merge_keys,
        suffixes=("", "_reference"),
        validate="one_to_one" if policy_pair else "many_to_one",
        how="left",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Faltan episodios o semillas en la referencia pareada")
    paired["auc_difference"] = (
        paired["normalized_auc"] - paired["normalized_auc_reference"]
    )
    paired["reference_strategy"] = name
    return paired[
        [
            *keys,
            "strategy",
            "policy_training_seed",
            "reference_strategy",
            "auc_difference",
        ]
    ]


def paired_selection_differences(table: pd.DataFrame) -> pd.DataFrame:
    from itertools import combinations

    from truss_research.optimizacion.variantes_rl import POLICY_VARIANTS

    comparisons: list[pd.DataFrame] = []
    for name in REFERENCE_STRATEGIES:
        reference = table.loc[table.strategy.eq(name)]
        if not reference.empty:
            comparisons.append(
                paired_contrast(
                    table.loc[table.strategy.ne(name)], reference, (name, False)
                )
            )
    pairs = [
        (
            part.name,
            next(
                other.name
                for other in POLICY_VARIANTS
                if other.algorithm == part.algorithm and not other.use_predictions
            ),
        )
        for part in POLICY_VARIANTS
        if part.use_predictions
    ]
    for use_predictions in (True, False):
        variants = [
            part.name
            for part in POLICY_VARIANTS
            if part.use_predictions == use_predictions
        ]
        pairs.extend((later, earlier) for earlier, later in combinations(variants, 2))
    pairs.extend(
        (
            ("rl_repeat_control", "rl"),
            ("rl_repeat_penalty", "rl"),
            ("rl_repeat_penalty", "rl_repeat_control"),
        )
    )
    for subject, name in pairs:
        subjects = table.loc[table.strategy.eq(subject)]
        reference = table.loc[table.strategy.eq(name)]
        if not subjects.empty and not reference.empty:
            comparisons.append(paired_contrast(subjects, reference, (name, True)))
    columns = [
        "split",
        "episode_id",
        "catalog_seed",
        "budget",
        "strategy",
        "policy_training_seed",
        "reference_strategy",
        "auc_difference",
    ]
    return (
        pd.concat(comparisons, ignore_index=True)
        if comparisons
        else pd.DataFrame(columns=columns)
    )


def bootstrap_difference(
    group: pd.DataFrame, settings: AnalysisParameters
) -> dict[str, str | int | float | None]:
    values = group.assign(
        policy_training_seed=pd.to_numeric(
            group["policy_training_seed"], errors="raise"
        ).fillna(-1)
    )
    blocks = values.groupby(["policy_training_seed", "catalog_seed"])[
        "auc_difference"
    ].agg(["sum", "count"])
    counts = blocks["count"].unstack("catalog_seed")
    useful = counts.loc[:, counts.sum(axis=0).gt(0)]
    policy_seeds = group["policy_training_seed"].nunique()
    if useful.empty:
        return {
            "mean_difference": None,
            "ci_lower": None,
            "ci_upper": None,
            "policy_seeds": policy_seeds,
            "catalog_blocks": 0,
            "undefined_reason": "no_defined_paired_metrics",
        }
    if useful.isna().any().any() or useful.eq(0).any().any():
        raise ValueError("Faltan bloques pareados para alguna semilla de política")
    sums = (
        blocks["sum"]
        .unstack("catalog_seed")
        .loc[:, useful.columns]
        .to_numpy(dtype=float)
    )
    count_matrix = useful.to_numpy(dtype=float)
    rng = np.random.default_rng(settings.bootstrap_seed)
    policies = rng.integers(len(sums), size=(settings.bootstrap_repeats, len(sums)))
    catalogs = rng.integers(
        sums.shape[1], size=(settings.bootstrap_repeats, sums.shape[1])
    )
    sampled_sums = sums[policies[:, :, None], catalogs[:, None, :]].sum(axis=(1, 2))
    sampled_counts = count_matrix[policies[:, :, None], catalogs[:, None, :]].sum(
        axis=(1, 2)
    )
    lower, upper = np.quantile(sampled_sums / sampled_counts, INTERVAL_QUANTILES)
    return {
        "mean_difference": float(sums.sum() / count_matrix.sum()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "policy_seeds": policy_seeds,
        "catalog_blocks": sums.shape[1],
        "undefined_reason": None,
    }


def selection_intervals(
    table: pd.DataFrame, settings: AnalysisParameters
) -> pd.DataFrame:
    differences = paired_selection_differences(table)
    rows: list[dict[str, str | int | float | None]] = []
    keys = ["split", "strategy", "reference_strategy", "budget"]
    for identity, group in differences.groupby(keys):
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                **bootstrap_difference(group, settings),
                "paired_episodes": group["episode_id"].nunique(),
                "defined_pairs": int(group["auc_difference"].notna().sum()),
                "bootstrap_repeats": settings.bootstrap_repeats,
                "bootstrap_seed": settings.bootstrap_seed,
                "interpretation": "descriptive_conditional_on_dataset",
            }
        )
    return pd.DataFrame(rows)
