"""Datos artificiales identificables para comprobar el recorrido sin usar el Excel."""

from pathlib import Path

import pandas as pd

from truss_research.cerchas.schemas import DESIGN_BOUNDS
from truss_research.datos.schemas import RAW_ID_COLUMN, SHEET_NAME

GENERATION_COUNT = 100
SOLUTIONS_PER_GENERATION = 50
REPEATS_PER_DESIGN = 10
MIN_TRUSSES = int(DESIGN_BOUNDS["n_trusses"][0])
DISPLACEMENT_CYCLE = 7
DISPLACEMENT_INCREMENT_M = 0.004


def ensure_flow_source(destination: Path) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for generation in range(GENERATION_COUNT):
        for iteration in range(SOLUTIONS_PER_GENERATION):
            design = (
                generation * (SOLUTIONS_PER_GENERATION // REPEATS_PER_DESIGN)
                + iteration // REPEATS_PER_DESIGN
            )
            rows.append(
                {
                    RAW_ID_COLUMN: f"{{{generation};{iteration}}}",
                    "Web Config": generation % 5,
                    "N of Trusses": MIN_TRUSSES + generation % 3,
                    "L-R Height": 0.5 + design * 0.002,
                    "LM-RM Height": 0.7 + design * 0.001,
                    "M Height": 1.0 + design * 0.003,
                    "Subdiv1": 2 + generation % 4,
                    "Subdiv2": 2 + iteration // REPEATS_PER_DESIGN,
                    "Mass": 1000.0 + design * 10.0,
                    "Average Util": 0.2 + design * 0.001,
                    "Z Disp": 0.01
                    + (design % DISPLACEMENT_CYCLE) * DISPLACEMENT_INCREMENT_M,
                }
            )
    pd.DataFrame(rows).to_excel(destination, sheet_name=SHEET_NAME, index=False)
    (destination.parent / "SYNTHETIC.txt").write_text(
        "Datos artificiales para comprobar software. No representan resultados estructurales ni evidencia de la tesis.\n",
        encoding="utf-8",
    )
    return destination
