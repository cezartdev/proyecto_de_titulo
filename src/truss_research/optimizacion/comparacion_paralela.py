"""Banco privado firmado, bloques independientes y consolidación pareada única."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import cast

import joblib
import pandas as pd

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.optimizacion.catalogos import (
    SelectionBank,
    bank_manifest,
    comparison_signature,
    load_bank,
)
from truss_research.optimizacion.comparacion import (
    LearnedPolicies,
    PolicyReference,
    comparison_episode,
    load_references,
    strategies_for,
    write_comparison,
)
from truss_research.optimizacion.contratos import SelectionConfig
from truss_research.surrogates.experimentos import write_json


@dataclass(frozen=True)
class ComparisonCommand:
    config: SelectionConfig
    policies: tuple[PolicyReference, ...] = ()
    bank_dir: Path | None = None
    indices: tuple[int, ...] = ()
    blocks: tuple[Path, ...] = ()


@dataclass(frozen=True)
class CachedComparison:
    bank: SelectionBank
    policies: LearnedPolicies
    bank_loading_seconds: float
    policy_loading_seconds: float


def prepare_bank(command: ComparisonCommand) -> None:
    start = perf_counter()
    bank = load_bank(command.config)
    seconds = perf_counter() - start
    destination = command.config.output_dir
    destination.mkdir(parents=True, exist_ok=False)
    joblib.dump(bank, destination / "bank.joblib")
    write_json(
        destination / "run.json",
        {
            **bank_manifest(bank, command.config),
            "comparison_signature": comparison_signature(bank, command.config),
            "bank_signature": source_digest(destination / "bank.joblib"),
            "timing": {
                "bank_preparation_seconds": seconds - bank.prediction_seconds,
                "prediction_seconds": bank.prediction_seconds,
                "predicted_designs": len(bank.table),
            },
        },
    )


@lru_cache(maxsize=1)
def load_comparison(
    root: Path, predictor: Path, references: tuple[PolicyReference, ...]
) -> CachedComparison:
    start = perf_counter()
    metadata = read_artifact(root / "run.json")
    if source_digest(root / "bank.joblib") != metadata["bank_signature"]:
        raise ValueError("El banco privado de comparación cambió")
    bank = cast(SelectionBank, joblib.load(root / "bank.joblib"))
    loading = perf_counter() - start
    start = perf_counter()
    policies = LearnedPolicies(load_references(references, predictor), references)
    return CachedComparison(bank, policies, loading, perf_counter() - start)


def comparison_block(command: ComparisonCommand) -> None:
    if command.bank_dir is None:
        raise ValueError("Falta banco preparado")
    before = load_comparison.cache_info().misses
    cached = load_comparison(
        command.bank_dir, command.config.predictor_dir, command.policies
    )
    miss = load_comparison.cache_info().misses > before
    rows: list[dict[str, JsonValue]] = []
    traces: list[dict[str, JsonValue]] = []
    for number in command.indices:
        result, trajectory = comparison_episode(
            cached.bank, (cached.bank.episodes[number], command.config), cached.policies
        )
        rows.extend(result)
        traces.extend(trajectory)
    destination = command.config.output_dir
    destination.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(destination / "summary.csv", index=False)
    pd.DataFrame(traces).to_csv(destination / "traces.csv", index=False)
    write_json(
        destination / "run.json",
        {
            "indices": list(command.indices),
            "comparison_signature": comparison_signature(cached.bank, command.config),
            "bank_loading_seconds": cached.bank_loading_seconds if miss else 0.0,
            "policy_loading_seconds": cached.policy_loading_seconds if miss else 0.0,
        },
    )


def ordered_results(
    table: pd.DataFrame, order: tuple[list[str], list[str]]
) -> pd.DataFrame:
    episodes, strategies = order
    result = table.assign(
        episode_order=table.episode_id.map(
            {key: number for number, key in enumerate(episodes)}
        ),
        strategy_order=table.comparison_id.map(
            {key: number for number, key in enumerate(strategies)}
        ),
    )
    columns = [
        "episode_order",
        "strategy_order",
        *(["query"] if "query" in table else []),
    ]
    return result.sort_values(columns, kind="stable").drop(
        columns=["episode_order", "strategy_order"]
    )


def validate_blocks(
    command: ComparisonCommand, bank: SelectionBank
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, JsonValue]]:
    manifests = [read_artifact(root / "run.json") for root in command.blocks]
    indices = [
        index
        for metadata in manifests
        for index in cast(list[int], metadata["indices"])
    ]
    if sorted(indices) != list(range(len(bank.episodes))) or any(
        metadata["comparison_signature"] != comparison_signature(bank, command.config)
        for metadata in manifests
    ):
        raise ValueError("Los bloques no cubren exactamente la comparación firmada")
    rows = pd.concat(
        [pd.read_csv(root / "summary.csv") for root in command.blocks],
        ignore_index=True,
    )
    traces = pd.concat(
        [pd.read_csv(root / "traces.csv") for root in command.blocks], ignore_index=True
    )
    strategies = [
        *strategies_for(bank, (bank.episodes[0], command.config)),
        *(f"{reference.variant}_{reference.seed}" for reference in command.policies),
    ]
    expected = {
        (spec.episode_id, strategy) for spec in bank.episodes for strategy in strategies
    }
    actual = list(zip(rows.episode_id, rows.comparison_id, strict=True))
    if (
        set(actual) != expected
        or len(actual) != len(expected)
        or traces.duplicated(["episode_id", "comparison_id", "query"]).any()
    ):
        raise ValueError("Hay episodios o estrategias ausentes o duplicados")
    budgets = {spec.episode_id: spec.budget for spec in bank.episodes}
    groups = traces.groupby(["episode_id", "comparison_id"]).query.agg(list)
    if set(groups.index) != expected or any(
        sorted(queries) != list(range(1, budgets[episode] + 1))
        for (episode, _), queries in groups.items()
    ):
        raise ValueError("Las trazas no respetan exactamente cada presupuesto")
    order = ([spec.episode_id for spec in bank.episodes], strategies)
    timing: dict[str, JsonValue] = {
        "policy_loading_seconds": sum(
            float(meta["policy_loading_seconds"]) for meta in manifests
        ),
        "bank_loading_seconds": sum(
            float(meta["bank_loading_seconds"]) for meta in manifests
        ),
        "blocks": len(command.blocks),
        "scope": "one_cached_prediction_batch_per_split; policy_loading_seconds sums worker cache misses; decision times observed under CPU contention",
    }
    return ordered_results(rows, order), ordered_results(traces, order), timing


def aggregate_comparison(command: ComparisonCommand) -> None:
    if command.bank_dir is None:
        raise ValueError("Falta banco preparado")
    metadata = read_artifact(command.bank_dir / "run.json")
    if source_digest(command.bank_dir / "bank.joblib") != metadata["bank_signature"]:
        raise ValueError("El banco privado cambió antes de consolidar")
    bank = cast(SelectionBank, joblib.load(command.bank_dir / "bank.joblib"))
    rows, traces, timing = validate_blocks(command, bank)
    timing = {**cast(dict[str, JsonValue], metadata["timing"]), **timing}
    write_comparison(bank, (command.config, command.policies), (rows, traces, timing))
