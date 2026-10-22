"""Descarga explícita y piloto CPU/CUDA acotado, sin consultar prueba."""

import multiprocessing as mp
import os
from dataclasses import asdict
from multiprocessing.process import BaseProcess
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import psutil

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue, load_design_split
from truss_research.datos.subconjuntos import nested_subsets
from truss_research.evaluacion.costos import machine_context
from truss_research.evaluacion.recursos import ResourceMonitor
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    artifact_signatures,
    source_signature,
)
from truss_research.surrogates.configuracion import PredictorConfig
from truss_research.surrogates.experimentos import data_signature, predict, write_json
from truss_research.surrogates.modelos import build_predictor
from truss_research.surrogates.persistencia import load_model, save_model
from truss_research.surrogates.tabpfn import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
    CHECKPOINT_SHA256,
    PACKAGE_VERSION,
    PILOT_SECONDS,
    PILOT_SIZES,
    TabPFNParameters,
    verify_tabpfn,
)

DOWNLOAD_CHUNK_BYTES = 1024 * 1024
PILOT_BATCH_SIZES = (1, 32, 64)
PILOT_POLL_SECONDS = 0.25


def pilot_accelerator(requested: str | None = None) -> str:
    """Prefiere CUDA visible y utilizable; respeta la elección explícita."""
    if requested in ("cpu", "cuda"):
        return requested
    if requested not in (None, "auto"):
        raise ValueError("El acelerador del piloto debe ser auto, cpu o cuda")
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def download_weights(settings: TabPFNParameters) -> Path:
    path = settings.checkpoint
    if path.exists():
        verify_tabpfn(settings)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{CHECKPOINT_REPOSITORY}/resolve/{CHECKPOINT_REVISION}/{CHECKPOINT_FILENAME}"
    headers = (
        {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
        if os.getenv("HF_TOKEN")
        else {}
    )
    start = perf_counter()
    temporary = path.with_suffix(f".download-{os.getpid()}")
    try:
        with (
            urlopen(Request(url, headers=headers), timeout=30) as response,
            temporary.open("xb") as stream,
        ):
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                stream.write(chunk)
        if source_digest(temporary) != CHECKPOINT_SHA256:
            raise ValueError(
                "El checkpoint descargado no coincide con el hash congelado"
            )
        temporary.replace(path)
    except HTTPError as error:
        raise ValueError(
            "No se pudo acceder al checkpoint. Revisar acceso y licencia en Hugging Face; no se aceptan términos automáticamente"
        ) from error
    write_json(
        path.parent / "download.json",
        {
            "repository": CHECKPOINT_REPOSITORY,
            "revision": CHECKPOINT_REVISION,
            "filename": CHECKPOINT_FILENAME,
            "sha256": CHECKPOINT_SHA256,
            "size_bytes": path.stat().st_size,
            "download_wall_seconds": perf_counter() - start,
            "package_version": PACKAGE_VERSION,
            "license_url": f"https://huggingface.co/{CHECKPOINT_REPOSITORY}/blob/{CHECKPOINT_REVISION}/LICENSE",
            "pretraining_cost": "not_measured",
        },
    )
    verify_tabpfn(settings)
    return path


def pilot_identity(config: StudyConfig) -> dict[str, JsonValue]:
    return {
        "settings": asdict(config.tabpfn),
        "model_seed": config.representative_seed,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "package_version": PACKAGE_VERSION,
        "source_signature": source_signature(),
        "dataset_signature": data_signature(config.data_dir),
        "machine": machine_context(),
    }


def execute_pilot(config: StudyConfig, destination: Path) -> None:
    import gc

    split = load_design_split(config.data_dir)
    sizes = tuple(
        sorted(
            {min(size, len(split.training)) for size in PILOT_SIZES}
            | {len(split.training)}
        )
    )
    subsets = nested_subsets(split.training, sizes, config.representative_seed)
    rows = []
    for size in sizes:
        root = destination / f"n{size}"
        root.mkdir()
        model = build_predictor(
            PredictorConfig(
                model="tabpfn",
                seed=config.representative_seed,
                tabpfn=config.tabpfn,
            )
        )
        with ResourceMonitor(config.measurement.memory_interval_ms) as monitor:
            start = perf_counter()
            model.fit(
                subsets[size].loc[:, list(INPUT_COLUMNS)],
                subsets[size].loc[:, list(OUTPUT_COLUMNS)],
            )
            fit_seconds = perf_counter() - start
            timings = {}
            for batch in PILOT_BATCH_SIZES:
                features = split.validation.loc[:, list(INPUT_COLUMNS)].iloc[:batch]
                start = perf_counter()
                predicted = predict(model, features).to_numpy()
                timings[str(len(features))] = perf_counter() - start
            save_model(model, root)
            np.save(root / "expected.npy", predicted)
            del model
            gc.collect()
            process = mp.get_context("spawn").Process(
                target=reload_pilot, args=(config, root)
            )
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError("Falló la recarga de TabPFN en otro proceso")
        row = {
            "train_size": size,
            "fit_seconds": fit_seconds,
            "prediction_seconds": timings,
            "cold_reload": read_artifact(root / "reload.json"),
            **monitor.snapshot(),
        }
        rows.append(row)
        write_json(destination / "progress.json", {"completed": rows})
        if any(
            record.get("cpu_coverage") != "complete"
            or record.get("memory_coverage") != "sampled"
            for record in (row, row["cold_reload"])
        ):
            raise RuntimeError(
                "El piloto no obtuvo mediciones completas de CPU y memoria"
            )
    write_json(
        destination / "pilot.json",
        {
            **pilot_identity(config),
            "status": "complete",
            "rows": rows,
            "test_used": False,
            "pilot_artifact_signatures": {
                name: digest
                for name, digest in artifact_signatures(destination).items()
                if name != "pilot.json"
            },
        },
    )


def reload_pilot(config: StudyConfig, root: Path) -> None:
    features = (
        load_design_split(config.data_dir)
        .validation.loc[:, list(INPUT_COLUMNS)]
        .iloc[: max(PILOT_BATCH_SIZES)]
    )
    with ResourceMonitor(config.measurement.memory_interval_ms) as monitor:
        restored = predict(load_model(root), features).to_numpy()
    np.testing.assert_allclose(
        restored, np.load(root / "expected.npy"), rtol=1e-6, atol=1e-8
    )
    write_json(
        root / "reload.json",
        {"operation": "cold_load_and_predict", **monitor.snapshot()},
    )


def stop_pilot_process(process: BaseProcess) -> None:
    if not process.is_alive():
        return
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def pilot_worker(config: StudyConfig, destination: Path) -> None:
    try:
        execute_pilot(config, destination)
    except BaseException as error:
        write_json(
            destination / "pilot.json",
            {
                **pilot_identity(config),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "test_used": False,
            },
        )
        raise


def run_pilot(config: StudyConfig, destination: Path) -> Path:
    verify_tabpfn(config.tabpfn)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(
        destination / "pilot.json",
        {**pilot_identity(config), "status": "running", "test_used": False},
    )
    process = mp.get_context("spawn").Process(
        target=pilot_worker, args=(config, destination)
    )
    start = perf_counter()
    process.start()
    error = None
    try:
        while process.is_alive():
            process.join(PILOT_POLL_SECONDS)
            if perf_counter() - start > PILOT_SECONDS:
                error = "El piloto superó el presupuesto de diez minutos"
                break
            if (
                psutil.virtual_memory().percent / 100
                > config.execution.memory_limit_fraction
            ):
                error = "El piloto superó el límite de memoria vigente"
                break
    except BaseException as exception:
        write_json(
            destination / "pilot.json",
            {
                **pilot_identity(config),
                "status": "failed",
                "error": type(exception).__name__,
                "wall_seconds": perf_counter() - start,
                "test_used": False,
            },
        )
        raise
    finally:
        stop_pilot_process(process)
    if error is not None or process.exitcode != 0:
        message = (
            error
            or read_artifact(destination / "pilot.json").get("error")
            or f"El piloto terminó con código {process.exitcode}"
        )
        write_json(
            destination / "pilot.json",
            {
                **pilot_identity(config),
                "status": "failed",
                "error": message,
                "wall_seconds": perf_counter() - start,
                "test_used": False,
            },
        )
        raise RuntimeError(message)
    return destination


def require_pilot(config: StudyConfig) -> dict[str, JsonValue]:
    verify_tabpfn(config.tabpfn)
    root = config.tabpfn.checkpoint.parent / "pilots"
    identity = pilot_identity(config)
    for path in sorted(root.glob("pilot_*/pilot.json"), reverse=True):
        saved = read_artifact(path)
        if saved.get("status") == "complete" and all(
            saved.get(key) == value for key, value in identity.items()
        ):
            measured = {
                name: digest
                for name, digest in artifact_signatures(path.parent).items()
                if name != "pilot.json"
            }
            if measured != saved.get("pilot_artifact_signatures"):
                raise ValueError(
                    "Los artefactos del piloto TabPFN cambiaron; ejecutar make tabpfn-pilot"
                )
            return saved
    raise ValueError(
        "Falta un piloto TabPFN satisfactorio y compatible; ejecutar make tabpfn-pilot"
    )
