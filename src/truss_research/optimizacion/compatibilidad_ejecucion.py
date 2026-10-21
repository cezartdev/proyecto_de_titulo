"""Compatibilidad exacta de un parche operativo, sin excluir código de sus firmas."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PATCH_FILE = Path("config/execution_compatibility.json")
PATCH_SCOPE = "cpu_memory_recovery_v1"
OPERATIONAL_FILES = frozenset(
    (
        "src/truss_research/optimizacion/ejecucion_cpu.py",
        "src/truss_research/optimizacion/paralelismo.py",
        "src/truss_research/optimizacion/estudio_estado.py",
        "src/truss_research/optimizacion/estudio.py",
        "src/truss_research/optimizacion/estudio_cierre.py",
        "src/truss_research/optimizacion/adaptativa_paralela.py",
        "src/truss_research/optimizacion/compatibilidad_ejecucion.py",
        "src/truss_research/evaluacion/trabajos_analisis.py",
        "src/truss_research/evaluacion/benchmark_cpu.py",
        "src/truss_research/evaluacion/campana.py",
        "src/truss_research/evaluacion/piloto_tabpfn.py",
    )
)


@dataclass(frozen=True)
class ExecutionPatch:
    from_signature: str
    to_signature: str
    scope: str
    reason: str
    changed_files: dict[str, dict[str, str | None]]


def source_inventory(root: Path) -> dict[str, str]:
    paths = [
        *sorted((root / "src").rglob("*.py")),
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for path in paths
    }


def inventory_signature(inventory: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()


DEFAULT_PATCHES: tuple[ExecutionPatch, ...] = (
    ExecutionPatch(
        from_signature="565d03f09eb0ac6a08afa6a6077f2c6d3d49aa23fd2150b58602404b2175119c",
        to_signature="b959ce45dcbd6efab6d91819af43076143759f7b6f978b5a9410915a6d67a01a",
        scope=PATCH_SCOPE,
        reason="Reciclado seguro del pool CPU y espera por RAM; sin cambios de modelos, particiones, búsquedas, restricciones, semillas o métricas.",
        changed_files={
            "src/truss_research/evaluacion/benchmark_cpu.py": {
                "before": "d246fdcc606456633661afc69cfda4c8993ef204b93f990f7ffa46c5ccfdbda6",
                "after": "5490b392de637a40b561586f3a1b7399c5531cc72b38afdc7ab02b616b4c3ba0",
            },
            "src/truss_research/evaluacion/campana.py": {
                "before": "6ba5bd534ed0f22013b1f2639fcd3db9009f93b43b70ae56f59f50353fd8bd6a",
                "after": "d230afce1964f143285b6cd4d6cd5c38da065e2e212d60ff3bc3bddde37412ed",
            },
            "src/truss_research/evaluacion/piloto_tabpfn.py": {
                "before": "b4b1315a73d02cf89140e4c77b39918ebca0ad2c1ca1762344e6243f80c6a6ef",
                "after": "ba484ff5bd94a5239befd09f04f14455297ddeeb608e3d816fe86309f802bfa7",
            },
            "src/truss_research/evaluacion/trabajos_analisis.py": {
                "before": "a858058555b511380509275addae6281e8620da23dac648ea25f6c000d45af67",
                "after": "d4f617e18d1c517fdc6730b90c0c4c85cbd8a3e4569665adbb1a0fa1b2c9d63f",
            },
            "src/truss_research/optimizacion/adaptativa_paralela.py": {
                "before": "afaad8301201311057aee10e420f2fe954ed71728e2950cef645cfd7f777d56e",
                "after": "76470c95950dc1a593b6547013ba959bc4fa7139ebfb18ba9b20497e651b4824",
            },
            "src/truss_research/optimizacion/compatibilidad_ejecucion.py": {
                "before": None,
                "after": "b57ef1e0471a645109c19c61f25be7113e37eed7e308824434ebeb2a58500095",
            },
            "src/truss_research/optimizacion/ejecucion_cpu.py": {
                "before": "88e41b7b37ef6dbfe786708d4828949c3c52c57ebfe11bc89515a6ab8ef1c3f4",
                "after": "bb5d577086075e1ba62dd6fadcb7fa37a146c8e6089255a2b49139253929665a",
            },
            "src/truss_research/optimizacion/estudio.py": {
                "before": "472818dd3f135a2874bbaf42cc02285db53eeef92d4175b0b73c9b2f86c273c2",
                "after": "4a696545db2f6de6e0bfaf50343384e798c1f250e3472b41d162630df01eb845",
            },
            "src/truss_research/optimizacion/estudio_cierre.py": {
                "before": "739e86c83d2844a753db05d056eda6c3ebcd7b34381ca89073c5fed026ccd709",
                "after": "ea547b8a17dca4ae18599f16e57b8053f044b7e864c04b0ca2a159d6ba5f1bf6",
            },
            "src/truss_research/optimizacion/estudio_estado.py": {
                "before": "6d83b944cf1cbbdbd91311c871cd9cf680332a3b9d35971901baefe9dcde7ffd",
                "after": "a0cf7d77cd45dad85bb79c7343e90cb25a7ccaa5155ccd8b70c448f3bee61743",
            },
            "src/truss_research/optimizacion/paralelismo.py": {
                "before": "c02f6c6c6ab719fc196d9d0e8c3b99f95a91baf0cd2e70b8d21aa90d10909cf1",
                "after": "4aeab49ebd2e3b14be0cd616663ecf8eec6ca92cbc308d6890c143c742e32552",
            },
        },
    ),
)


def approved_patches(current: str) -> tuple[ExecutionPatch, ...]:
    inventory = source_inventory(REPOSITORY_ROOT)
    if inventory_signature(inventory) != current:
        return ()
    path = REPOSITORY_ROOT / PATCH_FILE
    candidates = (
        tuple(
            ExecutionPatch(**raw)
            for raw in json.loads(path.read_text(encoding="utf-8"))["patches"]
        )
        if path.is_file()
        else DEFAULT_PATCHES
    )
    patches = []
    for patch in candidates:
        if patch.to_signature != current:
            continue
        if (
            patch.scope != PATCH_SCOPE
            or not patch.changed_files
            or not set(patch.changed_files) <= OPERATIONAL_FILES
        ):
            raise ValueError("El puente de compatibilidad excede el parche operativo")
        previous = dict(inventory)
        for name, hashes in patch.changed_files.items():
            if inventory.get(name) != hashes["after"]:
                raise ValueError(f"Cambió un archivo del parche operativo: {name}")
            before = hashes["before"]
            if before is None:
                previous.pop(name, None)
            else:
                previous[name] = before
        if inventory_signature(previous) != patch.from_signature:
            raise ValueError("El parche no reconstruye exactamente el código anterior")
        patches.append(patch)
    return tuple(patches)


def compatible_sources(current: str) -> tuple[str, ...]:
    return tuple(patch.from_signature for patch in approved_patches(current))
