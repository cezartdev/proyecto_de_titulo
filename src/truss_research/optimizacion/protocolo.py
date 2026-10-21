"""Protocolos JSON y planificación con preflight de entradas, sin entrenar."""

import json
from dataclasses import asdict, fields, replace
from pathlib import Path

from truss_research.cerchas.schemas import StructuralLimits
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.protocolo_analisis import (
    GeneralizationParameters,
    LearningCurveParameters,
)
from truss_research.evaluacion.recursos import MeasurementParameters
from truss_research.optimizacion.contratos import (
    REWARD_PROTOCOL,
    AnalysisParameters,
    EvolutionParameters,
)
from truss_research.optimizacion.ejecucion_cpu import ExecutionParameters
from truss_research.optimizacion.estrategias_busqueda import read_search_parameters
from truss_research.optimizacion.estudio_estado import StudyConfig, StudyPhase
from truss_research.optimizacion.propuestas import ProposalParameters
from truss_research.optimizacion.seleccion_incierta import AdaptiveSettings
from truss_research.optimizacion.variantes_rl import policy_variant, read_rl_parameters
from truss_research.surrogates.configuracion import ModelParameters
from truss_research.surrogates.poda import PruningParameters
from truss_research.surrogates.tabpfn import TabPFNParameters


def read_tabpfn(payload: JsonValue) -> TabPFNParameters:
    if not isinstance(payload, dict) or set(payload) - {
        f.name for f in fields(TabPFNParameters)
    }:
        raise ValueError("Parametros TabPFN desconocidos")
    return TabPFNParameters(**payload)


def read_proposals(payload: JsonValue) -> ProposalParameters:
    if not isinstance(payload, dict) or set(payload) - {
        part.name for part in fields(ProposalParameters)
    }:
        raise ValueError("Parámetros de propuestas desconocidos")
    return ProposalParameters(**payload)


def protocol_payload(
    path: Path, ancestors: tuple[Path, ...] = ()
) -> dict[str, JsonValue]:
    path = path.resolve()
    if path in ancestors:
        raise ValueError("Herencia circular de protocolos JSON")
    with path.open(encoding="utf-8") as stream:
        payload: JsonValue = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("El protocolo JSON debe ser un objeto")
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    if not isinstance(parent, str):
        raise ValueError("extends debe contener una ruta JSON")
    return merge_protocol(
        protocol_payload(path.parent / parent, (*ancestors, path)), payload
    )


def merge_protocol(
    base: dict[str, JsonValue], overrides: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    result = dict(base)
    for key, value in overrides.items():
        previous = result.get(key)
        result[key] = (
            merge_protocol(previous, value)
            if isinstance(previous, dict) and isinstance(value, dict)
            else value
        )
    return result


def read_analysis(payload: JsonValue) -> AnalysisParameters:
    if not isinstance(payload, dict) or set(payload) - {
        f.name for f in fields(AnalysisParameters)
    }:
        raise ValueError("Parámetros de análisis desconocidos")
    values = dict(payload)
    for key in ("catalog_sizes", "reference_margins", "generalization_seeds"):
        if key in values:
            if not isinstance(values[key], (list, tuple)):
                raise ValueError(f"{key} debe ser una lista JSON")
            values[key] = tuple(values[key])
    for key, cls, sequences in (
        ("generalization", GeneralizationParameters, ("fractions", "control_models")),
        (
            "learning_curves",
            LearningCurveParameters,
            ("models", "train_sizes", "subset_seeds"),
        ),
    ):
        raw = values.pop(key, {})
        if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(cls)}:
            raise ValueError(f"Parámetros desconocidos en {key}")
        converted = dict(raw)
        for name in sequences:
            if name in converted:
                if not isinstance(converted[name], (list, tuple)):
                    raise ValueError(f"{key}.{name} debe ser una lista JSON")
                converted[name] = tuple(converted[name])
        values[key] = cls(**converted)
    return AnalysisParameters(**values)


def configure_accelerator(config: StudyConfig, accelerator: str | None) -> StudyConfig:
    """Aplica una elección explícita de dispositivo antes de registrar el estudio."""
    if accelerator is None:
        return config
    if accelerator not in ("cpu", "cuda"):
        raise ValueError("El acelerador debe ser cpu o cuda")
    return replace(config, tabpfn=replace(config.tabpfn, device=accelerator))


def load_protocol(
    path: Path, locations: tuple[Path, Path], accelerator: str | None = None
) -> StudyConfig:
    from truss_research.datos.procedencia import load_confirmations

    load_confirmations()
    payload = protocol_payload(path)
    if set(payload) - {
        "study",
        "parameters",
        "evolution",
        "analysis",
        "measurement",
        "execution",
        "rl",
        "adaptive",
        "tabpfn",
        "pruning",
        "limits",
        "proposals",
        "search",
    }:
        raise ValueError(
            "Secciones JSON admitidas: study, parameters, evolution, analysis, measurement, execution, rl, adaptive, tabpfn, pruning, limits, proposals y search"
        )
    values = payload.get("study", {})
    allowed = {field.name for field in fields(StudyConfig)} - {
        "output_dir",
        "data_dir",
        "parameters",
        "evolution",
        "analysis",
        "measurement",
        "execution",
        "rl",
        "adaptive",
        "tabpfn",
        "pruning",
        "limits",
        "proposals",
        "search",
    }
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError("Campos desconocidos en el protocolo de estudio")
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict) or set(parameters) - {
        field.name for field in fields(ModelParameters)
    }:
        raise ValueError("Parámetros predictivos desconocidos")
    values = dict(values)
    evolution = payload.get("evolution", {})
    if not isinstance(evolution, dict) or set(evolution) - {
        field.name for field in fields(EvolutionParameters)
    }:
        raise ValueError("Parámetros evolutivos desconocidos")
    for name in values.keys() & {"models", "seeds", "budgets"}:
        if not isinstance(values[name], list):
            raise ValueError(f"{name} debe ser una lista JSON")
        values[name] = tuple(values[name])
    analysis = read_analysis(payload.get("analysis", {}))
    measurement = payload.get("measurement", {})
    if not isinstance(measurement, dict) or set(measurement) - {
        field.name for field in fields(MeasurementParameters)
    }:
        raise ValueError("Parámetros de medición desconocidos")
    measurement = dict(measurement)
    for name in measurement.keys() & {"batch_sizes", "reuse_counts"}:
        if not isinstance(measurement[name], list):
            raise ValueError(f"{name} debe ser una lista JSON")
        measurement[name] = tuple(measurement[name])
    execution = payload.get("execution", {})
    if not isinstance(execution, dict) or set(execution) - {
        field.name for field in fields(ExecutionParameters)
    }:
        raise ValueError("Parámetros de ejecución desconocidos")
    try:
        ExecutionParameters(**execution).validate_campaign()
        config = StudyConfig(
            locations[0],
            data_dir=locations[1],
            parameters=ModelParameters(**parameters),
            evolution=EvolutionParameters(**evolution),
            analysis=analysis,
            measurement=MeasurementParameters(**measurement),
            execution=ExecutionParameters(**execution),
            rl=read_rl_parameters(payload.get("rl", {})),
            adaptive=AdaptiveSettings(**payload.get("adaptive", {})),
            tabpfn=read_tabpfn(payload.get("tabpfn", {})),
            pruning=PruningParameters(**payload.get("pruning", {})),
            limits=StructuralLimits(**payload.get("limits", {})),
            proposals=read_proposals(payload.get("proposals", {})),
            search=read_search_parameters(payload.get("search", {})),
            **values,
        )
        return configure_accelerator(config, accelerator)
    except TypeError as error:
        raise ValueError(f"Tipos inválidos en el protocolo: {error}") from error


def study_plan(config: StudyConfig, phase: StudyPhase) -> str:
    if config.experiment == "evolution":
        from truss_research.evaluacion.busqueda_evolutiva import search_plan

        return json.dumps(
            search_plan(config, phase), ensure_ascii=False, indent=2, default=str
        )
    from truss_research.evaluacion.plan_analisis import analysis_plan

    config.execution.validate_campaign()
    tabpfn_gpu = config.tabpfn_enabled and config.tabpfn.device == "cuda"
    families, seeds = len(config.models), len(config.seeds)
    predictions = families * seeds
    validation = (
        len(config.rl.variants) * seeds + families + seeds - 1 + predictions + 1
    )
    final = 2 * predictions + 1
    stages = {"predictores": predictions, "validacion": validation, "final": final}
    included = {
        "predictores": ("predictores",),
        "validacion": ("predictores", "validacion"),
        "final": ("final",),
        "todo": tuple(stages),
    }[phase]
    base_tuning = (
        sum(name not in ("dummy", "targetwise", "tabpfn") for name in config.models)
        * config.trials
    )
    tuning = base_tuning if phase != "final" else 0
    payload = {
        "additional_analysis": analysis_plan(config),
        "mode": "plan_sin_ejecucion",
        "label": config.label,
        "config": asdict(config),
        "phase": phase,
        "primary_metrics": {
            "prediction": "normalized_mae",
            "selection": "normalized_auc",
        },
        "reward_protocol": REWARD_PROTOCOL,
        "execution": {
            "device": "cpu+cuda" if tabpfn_gpu else "cpu",
            "gpu_models": ["tabpfn"] if tabpfn_gpu else [],
            "gpu_workers": config.execution.gpu_workers if tabpfn_gpu else 0,
            "rl_device": "cpu",
            "workers": config.execution.effective_workers,
            "numeric_threads_per_worker": 1,
            "nested_pools": False,
            "parallel_units": ["family_search", "outer_fold", "final_predictor_fit"],
            "parallel_stage_units": [
                "policy_seed",
                "explanation",
                "ranking",
                "test_prediction",
                "comparison_episode_block",
            ],
            "memory_limit_fraction": config.execution.memory_limit_fraction,
            "telemetry_interval_seconds": config.execution.telemetry_interval_seconds,
            "optuna_trials_parallel": False,
            "preparation_jobs": families * (1 + config.folds)
            if config.execution.effective_workers > 1 and phase != "final"
            else 0,
        },
        "sensitivity": {
            "catalog_sizes": config.analysis.catalog_sizes,
            "reference_margins": config.analysis.reference_margins,
            "retraining": bool(config.analysis.generalization_seeds),
            "chronological_seeds": config.analysis.generalization_seeds,
            "requires_final": True,
        },
        "task_counts": {name: stages[name] for name in included},
        "total_tasks": sum(stages[name] for name in included),
        "predictor_matrix": [
            {"model": name, "seed": seed}
            for name in config.models
            for seed in config.seeds
        ],
        "tuning_trials": tuning,
        "nested_additional_trials": tuning * config.folds if config.nested_folds else 0,
        "targetwise_search_budget_upper_bound": (
            sum(m not in ("targetwise", "dummy", "tabpfn") for m in config.models) + 3
        )
        * config.trials
        * (1 + config.outer_folds)
        if "targetwise" in config.models and phase != "final"
        else 0,
        "chronological_sensitivity_search_budget_upper_bound": len(
            config.analysis.generalization_seeds
        )
        * (
            base_tuning
            + (
                sum(m not in ("dummy", "targetwise", "tabpfn") for m in config.models)
                + 3
            )
            * config.trials
            if "targetwise" in config.models
            else base_tuning
        ),
        "matched_control_search_budget_upper_bound": len(
            config.analysis.generalization_seeds
        )
        * sum(
            name not in ("dummy", "tabpfn") and name in config.models
            for name in config.analysis.generalization.control_models
        )
        * config.trials,
        "policy_matrix": [
            {
                "variant": variant,
                "seed": seed,
                "algorithm": policy_variant(variant).algorithm,
                "use_predictions": policy_variant(variant).use_predictions,
                "allow_repeats": policy_variant(variant).allow_repeats,
                "repeat_penalty": config.rl.repeat_penalty
                if policy_variant(variant).repeat_penalty
                else 0.0,
            }
            for variant in config.rl.variants
            for seed in config.seeds
        ],
        "policy_transitions": len(config.rl.variants) * seeds * config.rl_steps
        if phase in ("validacion", "todo")
        else 0,
        "episodes_per_bank": seeds * config.episodes_per_seed * len(config.budgets),
        "proposal_pilot": {
            "automatic": False,
            "command": "make proposals",
            "policy_transitions": seeds * config.rl_steps,
            "parameters": asdict(config.proposals),
            "evidence": "predicted_only_no_fem",
            "surrogate_evaluation_limits": {
                str(budget): config.proposals.pool_size * budget
                for budget in config.budgets
            },
        },
        "notes": [
            "Se ajustan hiperparámetros y OOF una vez por familia; las semillas reutilizan esa búsqueda.",
            "Se minimizan masa y desplazamiento y se maximiza utilización promedio simultáneamente, respetando los límites operativos.",
            "Origen evolutivo, unidades y dominio confirmados; quedan pendientes solver, cargas, apoyos, conectividad y asignaciones de sección.",
            "Los umbrales agregados no certifican seguridad por barra; no hay casos de utilización > 1.",
            "La familia de decisión se elige únicamente con validación.",
            "Los conteos describen el protocolo, no tareas pendientes de una reanudación.",
            "preparation_jobs desglosa búsquedas y pliegues paralelos; no añade modelos ni ensayos al presupuesto científico.",
            "No se estiman horas sin mediciones del equipo.",
            "Los estudios posteriores no constituyen nueva prueba externa independiente.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
