"""Reinicio explícito de derivados experimentales, conservando fuentes y pesos."""

import argparse
import shutil
from pathlib import Path

from truss_research.datos.schemas import DESIGN_DESTINATION
from truss_research.evaluacion.registro import results_root
from truss_research.optimizacion.estudio_estado import lock_handle
from truss_research.optimizacion.protocolo import load_protocol

WORKSPACE = Path(__file__).resolve().parents[3]
LOCK_NAMES = (".study.lock", ".run_all.lock")


def verify_idle(targets: tuple[Path, ...]) -> None:
    """Rechaza coordinadores activos antes de retirar cualquier derivado."""
    for directory in targets:
        for name in LOCK_NAMES:
            path = directory / name
            if not path.is_file():
                continue
            try:
                with path.open("r+b") as handle:
                    lock_handle(handle.fileno(), False)
                    lock_handle(handle.fileno(), True)
            except OSError as error:
                raise ValueError(
                    f"Hay una campaña activa o un bloqueo inaccesible: {path}"
                ) from error


def reset_targets(study: Path, protocol: Path) -> tuple[Path, ...]:
    study = study.resolve()
    artifacts = (WORKSPACE / "artifacts").resolve()
    protected = (artifacts / "pretrained").resolve()
    if (
        not study.is_relative_to(artifacts)
        or study == artifacts
        or study.is_relative_to(protected)
        or protected.is_relative_to(study)
    ):
        raise ValueError(
            "El estudio debe estar dentro de artifacts y fuera de pretrained"
        )
    config = load_protocol(protocol, (study, WORKSPACE / DESIGN_DESTINATION))
    smoke = study.with_name(study.name + "_smoke")
    check = artifacts / "comprobacion_flujo"
    targets = (
        study,
        results_root(study),
        smoke,
        results_root(smoke),
        check,
        results_root(check),
        config.data_dir,
        config.tabpfn.checkpoint.parent / "pilots",
    )
    allowed = (artifacts, (WORKSPACE / "reports").resolve())
    data = (WORKSPACE / DESIGN_DESTINATION).resolve()
    resolved = tuple(dict.fromkeys(path.resolve() for path in targets))
    for path in resolved:
        if path != data and not any(
            path.is_relative_to(parent) and path != parent for parent in allowed
        ):
            raise ValueError(f"Destino de limpieza fuera del espacio permitido: {path}")
        if path == protected or protected.is_relative_to(path):
            raise ValueError("No eliminar pesos preentrenados")
    return resolved


def reset_study(study: Path, protocol: Path, dry_run: bool = False) -> tuple[Path, ...]:
    targets = reset_targets(study, protocol)
    for path in targets:
        print(f"{'Previsto' if dry_run else 'Retirar'}: {path}")
    if dry_run:
        return targets
    verify_idle(targets)
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estudio", type=Path, default=Path("artifacts/estudio"))
    parser.add_argument("--config", type=Path, default=Path("config/study.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reset_study(args.estudio, args.config, args.dry_run)


if __name__ == "__main__":
    main()
