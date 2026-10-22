"""Sensibilidad por diseños únicos con artefactos separados del estudio temporal."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.artefactos import portable_payload, read_artifact
from truss_research.datos.preparacion import prepare_dataset, source_digest
from truss_research.datos.schemas import DEFAULT_SOURCE, PreparationConfig, SplitConfig
from truss_research.evaluacion.benchmark_multiobjetivo import (
    export_benchmark_reports,
    write_benchmark,
)
from truss_research.evaluacion.comparacion_predictiva import export_prediction_tables
from truss_research.evaluacion.robustez import export_evidence_context
from truss_research.optimizacion.estudio_cierre import evaluate_test_predictor
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    artifact_signatures,
    source_signature,
)
from truss_research.optimizacion.protocolo import load_protocol
from truss_research.surrogates.configuracion import PredictorConfig, response_scale
from truss_research.surrogates.experimentos import read_run, train_predictor, write_json

DEFAULT_DESTINATION = Path("artifacts/interpolacion_001")
DEFAULT_PROTOCOL = Path("config/study.json")


def verify_interpolation(config: StudyConfig, source: Path) -> Path:
    """Verifica un cierre completo del mismo protocolo y fuente antes de reutilizarlo."""
    root = config.output_dir
    if not (root / "complete.json").is_file():
        raise ValueError(
            "La interpolación está incompleta; conservar para diagnóstico y elegir otro INTERPOLATION_DEST"
        )
    expected = json.loads(json.dumps(asdict(config), default=str))
    config_path = root / "config.json"
    if json.loads(config_path.read_text(encoding="utf-8")) != portable_payload(
        expected, config_path
    ):
        raise ValueError(
            "La interpolación pertenece a otro protocolo; elegir otro INTERPOLATION_DEST"
        )
    if read_artifact(root / "data/manifest.json")["source_sha256"] != source_digest(
        source
    ):
        raise ValueError(
            "La fuente de interpolación cambió; elegir otro INTERPOLATION_DEST"
        )
    receipt = read_artifact(root / "complete.json")
    if receipt["source_signature"] != source_signature():
        raise ValueError(
            "El código de interpolación cambió; elegir otro INTERPOLATION_DEST"
        )
    files = receipt["files"]
    if not files or files != {
        name: digest
        for name, digest in artifact_signatures(root).items()
        if name != "complete.json"
    }:
        raise ValueError(
            "Cambió una salida completa de interpolación; se conserva para diagnóstico"
        )
    return root


def run_interpolation(config: StudyConfig, source: Path) -> Path:
    """Ajusta controles con el presupuesto configurado y congela antes de prueba."""
    root = config.output_dir
    if root.exists():
        raise ValueError("La sensibilidad requiere un destino nuevo")
    models = tuple(
        name for name in config.analysis.interpolation_models if name in config.models
    )
    if len(models) < 2:
        raise ValueError("La sensibilidad requiere al menos dos familias de control")
    split = prepare_dataset(
        PreparationConfig(
            source=source,
            destination=root / "data",
            split=SplitConfig(seed=config.cv_seed, protocol="design"),
            calibration_fraction=config.calibration_fraction,
            limits=config.limits,
        )
    ).split
    write_json(root / "config.json", asdict(config))
    predictors: list[tuple[str, int, Path]] = []
    for family in models:
        for seed in config.seeds:
            path = root / "predictors" / f"{family}_{seed}"
            train_predictor(
                PredictorConfig(
                    data_dir=root / "data",
                    output_dir=path,
                    model=family,
                    seed=seed,
                    cv_seed=config.cv_seed,
                    folds=config.outer_folds,
                    nested_folds=config.inner_folds,
                    final_tuning_folds=config.final_tuning_folds,
                    trials=0 if family == "tabpfn" else config.trials,
                    parameters=config.parameters,
                    pruning=config.pruning,
                    tabpfn=config.tabpfn,
                    device=config.tabpfn.device if family == "tabpfn" else "cpu",
                    reuse_dir=root / "predictors" / f"{family}_{config.seeds[0]}"
                    if seed != config.seeds[0]
                    else None,
                )
            )
            predictors.append((family, seed, path))
    scores = {
        family: sum(
            float(read_run(path)["normalized_mae"])
            for name, _, path in predictors
            if name == family
        )
        / len(config.seeds)
        for family in models
    }
    validation_predictions = pd.concat(
        [
            pd.read_csv(path / "validation_predictions.csv").assign(
                split="validation", model=family, seed=seed
            )
            for family, seed, path in predictors
        ],
        ignore_index=True,
    )
    utility = write_benchmark(
        (split.training, validation_predictions),
        config,
        root / "predictor_utility/validation",
    )
    utility_scores = (
        utility.loc[
            utility.role.eq("regressor") & utility.budget.eq(min(config.budgets))
        ]
        .groupby("model")
        .auc.mean()
    )
    write_json(
        root / "freeze.json",
        {
            "selected_by_validation": min(
                models, key=lambda name: (-utility_scores[name], scores[name], name)
            ),
            "selection_criterion": "mean_observed_auc_smallest_validation_budget_then_mae",
            "validation_scores": scores,
            "validation_auc": utility_scores.to_dict(),
            "files": artifact_signatures(root),
            "scope": "exploratory_interpolation_same_historical_source_not_external_validation",
        },
    )
    frames: list[pd.DataFrame] = []
    for family, seed, path in predictors:
        held = root / "test_predictors" / f"{family}_{seed}"
        evaluate_test_predictor(path, held)
        for name, directory in (("validation", path), ("test", held)):
            frames.append(
                pd.read_csv(directory / f"{name}_predictions.csv").assign(
                    split=name, model=family, seed=seed
                )
            )
    destination = root / "report"
    predictions = pd.concat(frames, ignore_index=True)
    write_benchmark(
        (split.training, predictions.loc[predictions.split.eq("test")]),
        config,
        root / "predictor_utility/test",
    )
    destination.mkdir()
    export_prediction_tables(
        predictions,
        (response_scale(split.training[list(OUTPUT_COLUMNS)]), config.analysis),
        destination,
    )
    export_evidence_context(split, destination)
    export_benchmark_reports(root / "predictor_utility", destination)
    write_json(
        root / "complete.json",
        {
            "files": artifact_signatures(root),
            "source_signature": source_signature(),
            "scope": "interpolation_sensitivity_not_temporal_or_independent_test",
        },
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--destino", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--accelerator", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    config = load_protocol(
        args.config, (args.destino, args.destino / "data"), args.accelerator
    )
    print(run_interpolation(config, args.source))


if __name__ == "__main__":
    main()
