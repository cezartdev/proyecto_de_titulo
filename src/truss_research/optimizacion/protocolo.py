"""Protocolos JSON y planificación con preflight de entradas, sin entrenar."""

import json
from dataclasses import fields, replace
from pathlib import Path

from truss_research.cerchas.schemas import StructuralLimits
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.recursos import MeasurementParameters
from truss_research.optimizacion.configuracion_rl import read_rl_parameters
from truss_research.optimizacion.contratos import (
    AnalysisParameters,
    EvolutionParameters,
)
from truss_research.optimizacion.ejecucion_cpu import ExecutionParameters
from truss_research.optimizacion.estrategias_busqueda import read_search_parameters
from truss_research.optimizacion.estudio_estado import StudyConfig, StudyPhase
from truss_research.optimizacion.propuestas import ProposalParameters
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
    if "interpolation_models" in values:
        if not isinstance(values["interpolation_models"], (list, tuple)):
            raise ValueError("interpolation_models debe ser una lista JSON")
        values["interpolation_models"] = tuple(values["interpolation_models"])
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
        "tabpfn",
        "pruning",
        "limits",
        "proposals",
        "search",
    }:
        raise ValueError(
            "Secciones JSON admitidas: study, parameters, evolution, analysis, measurement, execution, rl, tabpfn, pruning, limits, proposals y search"
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
    from truss_research.evaluacion.busqueda_evolutiva import search_plan

    return json.dumps(
        search_plan(config, phase), ensure_ascii=False, indent=2, default=str
    )
