"""Catálogos reproducibles de filas reales, con manifiestos públicos sin respuestas."""

import hashlib
import json
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procesados import (
    JsonValue,
    load_design_split,
    load_structural_limits,
)
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.optimizacion.contratos import EpisodeSpec, SelectionConfig
from truss_research.optimizacion.entorno import EpisodeData
from truss_research.surrogates.experimentos import (
    data_signature,
    load_predictor,
    predict,
    read_run,
    write_json,
)

BANK_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class SelectionBank:
    table: pd.DataFrame
    predictions: pd.DataFrame
    episodes: tuple[EpisodeSpec, ...]
    space: ObjectiveSpace
    dataset_signature: str
    predictor_signature: str
    feature_ranges: tuple[float, ...] = ()
    prediction_seconds: float = 0.0
    safe_bounds: tuple[float, ...] | None = None
    evaluation_space: ObjectiveSpace | None = None


def episode_specs(
    table: pd.DataFrame, config: SelectionConfig
) -> tuple[EpisodeSpec, ...]:
    if config.catalog_size > len(table):
        raise ValueError("El catálogo solicitado supera los diseños disponibles")
    episodes: list[EpisodeSpec] = []
    for seed in config.seeds:
        rng = np.random.default_rng(seed)
        pools = [
            tuple(
                table.iloc[
                    rng.choice(
                        len(table),
                        config.base_catalog_size or config.catalog_size,
                        replace=False,
                    )
                ]["iter_id"]
            )
            for _ in range(config.episodes_per_seed)
        ]
        pools = [
            extend_catalog(table, pool, (config.catalog_size, seed, number))
            for number, pool in enumerate(pools)
        ]
        episodes.extend(
            EpisodeSpec(pool, 0, budget, seed, f"{seed}_{number}_{budget}")
            for number, pool in enumerate(pools)
            for budget in config.budgets
        )
    return tuple(episodes)


def extend_catalog(
    table: pd.DataFrame, pool: tuple[str, ...], context: tuple[int, int, int]
) -> tuple[str, ...]:
    size, seed, number = context
    remaining = table.loc[~table["iter_id"].isin(pool), "iter_id"].to_numpy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, number, size]))
    extension = tuple(rng.choice(remaining, size - len(pool), replace=False))
    return (*pool, *extension)


def load_bank(config: SelectionConfig) -> SelectionBank:
    signature = data_signature(config.data_dir)
    metadata = read_run(config.predictor_dir)
    if metadata.get("dataset_signature") != signature or not isinstance(
        metadata.get("model_signature"), str
    ):
        raise ValueError(
            "El predictor no corresponde al dataset o no tiene firma versionada"
        )
    split = load_design_split(config.data_dir)
    table = {"validation": split.validation, "test": split.testing}[config.split]
    table = table.set_index("iter_id", drop=False)
    model = load_predictor(config.predictor_dir)
    start = perf_counter()
    predicted = predict(model, table.loc[:, list(INPUT_COLUMNS)])
    prediction_seconds = perf_counter() - start
    space = ObjectiveSpace.fit(
        split.training.loc[:, list(OUTPUT_COLUMNS)].to_numpy(),
        load_structural_limits(config.data_dir),
    )
    bounds = None
    if (config.predictor_dir / "calibration.json").is_file():
        from truss_research.surrogates.intervalos import load_calibrator

        bounds = tuple(load_calibrator(config.predictor_dir).widths(0.90))
    return SelectionBank(
        table,
        predicted,
        episode_specs(table, config),
        space,
        signature,
        str(metadata["model_signature"]),
        tuple(
            np.ptp(
                split.training.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=float), axis=0
            )
        ),
        prediction_seconds,
        bounds,
    )


def episode_data(bank: SelectionBank, spec: EpisodeSpec) -> EpisodeData:
    candidates = bank.table.loc[list(spec.candidate_ids)]
    predicted = bank.predictions.loc[list(spec.candidate_ids)]
    return EpisodeData(
        spec,
        candidates.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=float),
        predicted.to_numpy(dtype=float),
        candidates.loc[:, list(OUTPUT_COLUMNS)].to_numpy(dtype=float),
        bank.space,
    )


def bank_manifest(bank: SelectionBank, config: SelectionConfig) -> dict[str, JsonValue]:
    manifest: dict[str, JsonValue] = {
        "schema_version": BANK_SCHEMA_VERSION,
        "split": config.split,
        "dataset_signature": bank.dataset_signature,
        "predictor_signature": bank.predictor_signature,
        "space": asdict(bank.space),
        "episodes": [asdict(spec) for spec in bank.episodes],
        "base_catalog_size": config.base_catalog_size,
    }
    if config.evolution.enabled:
        manifest["evolution"] = asdict(config.evolution)
        manifest["feature_ranges_training"] = bank.feature_ranges
    if bank.evaluation_space is not None:
        manifest["evaluation_space"] = asdict(bank.evaluation_space)
    return manifest


def comparison_signature(bank: SelectionBank, config: SelectionConfig) -> str:
    manifest = bank_manifest(bank, config)
    manifest.pop("predictor_signature")
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def prepare_selection(config: SelectionConfig) -> None:
    if config.output_dir.exists():
        raise ValueError("El destino ya existe; elegir una ejecución nueva")
    bank = load_bank(config)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(config.output_dir / "episodes.json", bank_manifest(bank, config))
    public = bank.table.loc[:, ["iter_id", *INPUT_COLUMNS]].join(
        bank.predictions.add_prefix("pred_")
    )
    public.to_csv(config.output_dir / "public_candidates.csv", index=False)
