"""Confirmaciones físicas humanas con fuente y fecha, sin inferencias automáticas."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal

CONFIRMED_SOURCE = (
    "docs/SCOPE.md; respuesta de la profesora y aclaraciones del autor "
    "registradas el 2026-10-19"
)
CONFIRMED_DATE = "2026-10-19"


@dataclass(frozen=True)
class PhysicalConfirmation:
    element: str
    status: Literal["pending", "author_reported", "confirmed"] = "pending"
    evidence_source: str | None = None
    confirmed_at: str | None = None
    evidence: str = ""
    confirmation_required: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.element, str)
            or not self.element.strip()
            or self.status not in ("pending", "author_reported", "confirmed")
        ):
            raise ValueError("Elemento o estado de procedencia física inválido")
        if self.evidence_source is not None and not isinstance(
            self.evidence_source, str
        ):
            raise ValueError("La fuente de evidencia debe ser texto")
        if self.status == "confirmed" and (
            not isinstance(self.evidence_source, str)
            or not self.evidence_source.strip()
            or not isinstance(self.confirmed_at, str)
        ):
            raise ValueError("Una confirmación física requiere fuente y fecha")
        if self.confirmed_at is not None:
            try:
                if (
                    date.fromisoformat(self.confirmed_at).isoformat()
                    != self.confirmed_at
                ):
                    raise ValueError("Formato no canónico")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "La fecha de confirmación debe usar AAAA-MM-DD"
                ) from error


DEFAULT_CONFIRMATIONS: tuple[PhysicalConfirmation, ...] = (
    PhysicalConfirmation(
        "Archivo y hoja",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "240214-C1-20m.xlsx, TOTAL DATA; manifiesto reproducible.",
        "Conservar identidad de la fuente.",
    ),
    PhysicalConfirmation(
        "SHA-256",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "f05c11f9a980497dcf5b74f3b74547244ce020552f95dca648a1d78ccae9f10c.",
        "Verificar al preparar.",
    ),
    PhysicalConfirmation(
        "Acceso y autorización",
        "author_reported",
        None,
        None,
        "Fuente autorizada de acceso restringido; LICENSE.",
        "Conservar evidencia de autorización sin publicar datos restringidos.",
    ),
    PhysicalConfirmation(
        "Cerchas de acero 2D, luz 20 m",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "Luz fija de 20 m; alturas y secciones variables.",
        "No interpretar luz fija como sección fija.",
    ),
    PhysicalConfirmation(
        "Origen en optimización topológica",
        "author_reported",
        None,
        None,
        "Optimización de alternativas geométricas y selección de secciones por barra.",
        "Precisar algoritmo, software, versiones y parámetros del optimizador.",
    ),
    PhysicalConfirmation(
        "Análisis estructural FEM",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "La profesora confirma análisis mediante elementos finitos.",
        "Precisar solver, software y versión; FEM no identifica el optimizador.",
    ),
    PhysicalConfirmation(
        "Cargas y apoyos",
        "pending",
        None,
        None,
        "No constan en el Excel ni en la respuesta.",
        "Cargas, combinaciones, apoyos, separación y vínculos entre cerchas.",
    ),
    PhysicalConfirmation(
        "Materiales y secciones",
        "confirmed",
        CONFIRMED_SOURCE
        + "; confirmación explícita del autor el 2026-10-21: A36 y Eurocódigo 3",
        "2026-10-21",
        "Acero A36 con fluencia 250 MPa; comprobaciones según Eurocódigo 3 y perfiles tubulares rectangulares europeos entre 50x25x2,5 y 500x300x20 mm.",
        "250 MPa no es una tensión admisible universal.",
    ),
    PhysicalConfirmation(
        "Asignación y propiedades por barra",
        "pending",
        None,
        None,
        "El optimizador seleccionaba secciones por barra; esas asignaciones no están en las siete entradas.",
        "Catálogo completo, perfiles por barra, conectividad, E, densidad y criterios de resistencia.",
    ),
    PhysicalConfirmation(
        "Generaciones y soluciones",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "100 generaciones sucesivas (0 a 99) de carácter evolutivo, generadas por un algoritmo evolutivo poblacional con 50 soluciones por generación (0 a 49), totalizando 5.000 exploraciones. Cada generación evalúa 50 propuestas orientadas a mejorar diseños previos mediante operadores evolutivos (pudiendo mejorar, igualar o empeorar); no existe un diseño base.",
        "No interpretar la solución 0 como diseño base ni generaciones como problemas independientes.",
    ),
    PhysicalConfirmation(
        "Selección previa de soluciones",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "5.000 exploraciones con 1.273 diseños únicos. Los 3.727 repetidos provienen de la dinámica generacional de 50 consultas por lote, donde el elitismo y la convergencia de la población replican soluciones exitosas.",
        "Conservar relación de pertenencias y no asumir independencia estadística entre individuos de generaciones contiguas.",
    ),
    PhysicalConfirmation(
        "`web_config`",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "0 Warren de doble intersección; 1 igual sin montantes; 2 Pratt; 3 Howe; 4 Warren simple.",
        "Los nombres no reconstruyen por sí solos la conectividad.",
    ),
    PhysicalConfirmation(
        "`n_trusses`",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "Número de cerchas del sistema, unidades; mínimo 3.",
        "Precisar separación y vínculos del sistema.",
    ),
    PhysicalConfirmation(
        "`lr_height_m`, `lmrm_height_m`, `m_height_m`",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "Alturas en metros de puntos bajos, intermedios y centro; geometría simétrica.",
        "Precisar coordenadas para reconstrucción FEM.",
    ),
    PhysicalConfirmation(
        "`subdiv1`, `subdiv2`",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "Dos tramos de media cercha, reflejados en la otra mitad; asimetría desactivada.",
        "subdiv1 distinto de subdiv2 no implica asimetría izquierda-derecha.",
    ),
    PhysicalConfirmation(
        "`mass_kg`",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "Masa de la estructura completa en kg.",
        "Precisar componentes si se reconstruye el análisis.",
    ),
    PhysicalConfirmation(
        "`util_ratio`",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "Utilización promedio como ratio continuo en [0, inf) (desde 0 hasta más de 1), no porcentaje 0 a 100. En el dataset varía entre 0,117 y 0,913.",
        "Precisar ponderaciones y obtener máximo por barra para certificar elementos; valores > 1 indican sobreutilización.",
    ),
    PhysicalConfirmation(
        "`z_disp_m`",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "Desplazamiento vertical Z en metros.",
        "La definición de máximo, nodo, signo y combinación sigue pendiente.",
    ),
    PhysicalConfirmation(
        "Agregación del desplazamiento Z",
        "pending",
        None,
        None,
        "La unidad y dirección están confirmadas, pero no la agregación.",
        "Máximo o nodo puntual, valor absoluto y combinación de cargas.",
    ),
    PhysicalConfirmation(
        "Límite de utilización 1",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "Umbral operativo configurable de admisibilidad (por defecto 1,0, pero parametrizable a valores mayores a 1 si se evalúan tolerancias de sobreesfuerzo o penalización continua). En selección descarta soluciones que excedan el límite fijado.",
        "Un promedio menor o igual a 1 no certifica cada barra; la restricción operativa puede ajustarse según el criterio de diseño.",
    ),
    PhysicalConfirmation(
        "Criterio de servicio L/700",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "La profesora recomienda tradicionalmente L/700; se adopta 20/700 m como criterio principal del estudio. El generador original no imponía límite Z.",
        "No atribuirlo como restricción original ni exigencia normativa universal.",
    ),
    PhysicalConfirmation(
        "Objetivos del estudio",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "Minimizar masa, maximizar utilización promedio y minimizar desplazamiento Z; confirmados por el autor.",
        "No sustituir utilización promedio por comprobación de seguridad por barra.",
    ),
    PhysicalConfirmation(
        "Integridad y cobertura",
        "confirmed",
        CONFIRMED_SOURCE,
        CONFIRMED_DATE,
        "5.000 apariciones, 1.273 diseños únicos sin respuestas contradictorias; 835 cumplen L/700 y 438 lo exceden. En la muestra registrada la utilización promedio máxima observada es 0,913.",
        "Auditar la fuente y conservar diseños incumplidores para evaluación.",
    ),
    PhysicalConfirmation(
        "Tiempo del solver original",
        "pending",
        None,
        None,
        "No existen mediciones comparables.",
        "Equipo y tiempos separados de FEM y selección de secciones.",
    ),
)


def load_confirmations(
    path: Path | None = None,
) -> tuple[PhysicalConfirmation, ...]:
    if path is None:
        return DEFAULT_CONFIRMATIONS
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"confirmations"}
        or not isinstance(payload["confirmations"], list)
    ):
        raise ValueError("El registro físico requiere una lista confirmations")
    try:
        result = tuple(
            PhysicalConfirmation(**entry) for entry in payload["confirmations"]
        )
    except TypeError as error:
        raise ValueError(
            "Campos desconocidos o inválidos en procedencia física"
        ) from error
    if len({entry.element for entry in result}) != len(result):
        raise ValueError("Hay confirmaciones físicas duplicadas")
    return result


def confirmations_digest() -> str:
    payload = [asdict(entry) for entry in load_confirmations()]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
