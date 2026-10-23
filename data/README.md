# Estructura de datos y procedencia de la hoja de cálculo

Este documento detalla la estructura original de la hoja de cálculo autorizada, las variables del dominio, la dinámica de generación evolutiva, la auditoría de duplicados y las particiones procesadas vigentes del estudio. Para el enfoque metodológico, consultar [CONTEXT.md](../CONTEXT.md); para el flujo de ejecución, consultar el [README raíz](../README.md). El contrato histórico citado como `docs/SCOPE.md` no está disponible en este checkout; queda pendiente recuperar su evidencia original.

---

## 1. Acceso y confidencialidad

El archivo Excel original y sus datos derivados de ingeniería estructural tienen acceso restringido; consultar [LICENSE](../LICENSE). No se redistribuyen en repositorios públicos de control de versiones.

- Directorio original: `data/raw/`
- Directorio temporal vigente: `data/processed_evolution/`.

---

## 2. Estructura de la hoja de cálculo original (`data/raw/`)

La fuente autorizada corresponde al libro `data/raw/240214-C1-20m.xlsx`, hoja **`TOTAL DATA`**, con una matriz de 5.000 filas y 12 columnas brutas (`Unnamed: 0`, `Mass`, `Average Util`, `Z Disp`, `Unnamed: 4`, `Web Config`, `N of Trusses`, `L-R Height`, `LM-RM Height`, `M Height`, `Subdiv1`, `Subdiv2`).

### Identificadores y dinámica evolutiva de generación
- **Estructura `{generación;solución}`:** Los 5.000 registros corresponden a **100 generaciones sucesivas (0 a 99) de carácter evolutivo**, con **50 soluciones por generación (0 a 49)** provenientes de una sola corrida de optimización de cerchas de acero en 2D de 20 metros de luz.
- **Lotes poblacionales:** Cada ciclo registra 50 apariciones. Sin registros operativos del solver no se presume que todas representen evaluaciones FEM nuevas ni que el índice de solución sea un orden de evaluación.
- **Proceso de búsqueda:** Los antecedentes del caso describen un proceso evolutivo. El Excel registra poblaciones, pero no padres, operadores ni reglas de supervivencia. Cruce y mutación son mecanismos posibles cuya aplicación en la corrida original no puede reconstruirse a partir de este archivo. Una aparición puede corresponder a un diseño nuevo o a uno ya registrado; no se presupone mejora individual ni del frente de cada generación.
- **Inexistencia de diseño base:** El identificador `{0;0}` identifica al individuo de índice cero de la generación inicial; no prueba orden de consulta ni representa una referencia de calibración.

---

## 3. Auditoría de duplicados y trazabilidad (`design_membership.csv`)

Tras auditar las siete variables geométricas de entrada y verificar que las tres respuestas estructurales coinciden de forma unívoca (rango de dispersión 0,0), se identifican:
- **5.000 apariciones totales.**
- **3.727 registros repetidos.**
- **1.273 diseños geométricos únicos.**

La identidad se define por igualdad exacta de las siete entradas, sin identificadores de fila ni respuestas y sin redondear alturas. Las respuestas se comprueban después para detectar conflictos. Los predictores usan un representante por diseño: la frecuencia de aparición no es un peso de entrenamiento ni un castigo de RL.

### Repeticiones observadas y mecanismos pendientes de verificar
En la generación 0 aparecen 50 diseños distintos. En las generaciones 90–99 se registran, en promedio, 10,8 diseños nuevos por generación; las otras 39,2 apariciones por generación repiten diseños presentes en el archivo acumulado. Este conteo no distingue por sí solo repeticiones dentro de la generación de reapariciones de generaciones anteriores.

El elitismo y la convergencia son explicaciones compatibles con estas repeticiones, pero los conteos no identifican su mecanismo causal. Sin el algoritmo original, sus reglas de supervivencia y los registros del evaluador, no se confirma preservación por elitismo, monotonicidad del frente Pareto ni número de evaluaciones FEM nuevas.

### Registro de pertenencias
El archivo `data/processed_evolution/design_membership.csv` preserva todas las apariciones, asociando cada tupla `{generación;solución}` a su representante y rol. `population_events.csv` conserva además entradas, respuestas e identidad de diseño de cada aparición. El representante se elige por primera generación y un desempate reproducible por solución, no por orden accidental de filas.



---

## 4. Esquema tabular y variables del dominio

La preparación canónica extrae y normaliza siete entradas geométricas y tres respuestas estructurales:

Los antecedentes del caso identifican acero A36, tensión de fluencia de 250 MPa y comprobaciones según Eurocódigo 3. Describen secciones por barra de un catálogo de perfiles tubulares rectangulares europeos; sus asignaciones individuales no están en las siete entradas.

Estas características se conservan como antecedentes de procedencia y no se verifican de forma independiente a partir del Excel. L/700 se adopta como criterio operativo del estudio, aunque no era una restricción explícita del proceso original.

### Alcance documental y reconstrucción del caso

- Características registradas en la documentación del caso: acero A36, fluencia 250 MPa, comprobaciones Eurocódigo 3, luz 20 m, geometría 2D simétrica, secciones por barra y evaluación FEM. El registro original de procedencia se conserva en [procedencia.py](../src/truss_research/datos/procedencia.py) y se exporta como `physical_provenance.csv`. La documentación disponible es parcial; el Excel permite comprobar valores y repeticiones, pero no verificar por sí solo material, unidades, solver ni significado físico de las variables.
- Información no disponible para reconstruir el evaluador: cargas y combinaciones; ubicación y restricciones de apoyos; conectividad y regla geométrica completa; catálogo y asignación de secciones; algoritmo de dimensionamiento; solver, versión y tipo de análisis; definición nodal, signo y agregación de desplazamiento; definición de utilización y comprobaciones de resistencia/estabilidad por barra.
- Limitación de representación: no se ha establecido si las secciones se fijaban o se optimizaban mediante una regla determinista para cada geometría. Etiquetas iguales en repeticiones no prueban que las siete entradas definan completamente la estructura.
- Alcance admisible: predecir las respuestas registradas bajo el proceso original y comparar modelos en esa fuente. La ficha incompleta impide reconstruir y certificar configuraciones nuevas; los umbrales agregados y el acuerdo entre predictores no sustituyen una verificación estructural.

La reconstrucción de un modelo FEM reproducible y la validación de configuraciones nuevas se plantean como trabajo futuro. La incorporación de nueva información debe documentarse y distinguirse del alcance de la campaña cerrada.

| Nombre columna | Tipo | Rango / Valores | Unidad | Descripción técnica |
| --- | --- | --- | --- | --- |
| `web_config` | Entero categórico | $\{0, 1, 2, 3, 4\}$ | Adimensional | Topología nominal: 0 (Warren de doble intersección), 1 (Warren doble intersección sin montantes), 2 (Pratt), 3 (Howe), 4 (Warren simple). |
| `n_trusses` | Entero | $[3, 50]$ | Unidades | Cantidad de cerchas o módulos del sistema estructural. |
| `lr_height_m` | Continuo | $[0{,}2, 3{,}5]$ | Metros (m) | Altura en los puntos bajos izquierdo y derecho, simétricos; no identifica por sí sola los apoyos. |
| `lmrm_height_m` | Continuo | $[0{,}2, 3{,}5]$ | Metros (m) | Altura en los puntos intermedios izquierdo y derecho; su posición longitudinal exacta no está confirmada. |
| `m_height_m` | Continuo | $[0{,}4, 3{,}5]$ | Metros (m) | Altura nodal en el centro de luz (clave de la cercha). |
| `subdiv1` | Entero | $[1, 10]$ | Unidades | Subdivisiones interiores del primer tramo de media cercha simétrica. |
| `subdiv2` | Entero | $[1, 10]$ | Unidades | Subdivisiones interiores del segundo tramo de media cercha simétrica. |
| `mass_kg` | Continuo | $[4.231{,}5, 33.723{,}5]$ | Kilogramos (kg) | Masa total estimada de la estructura de acero. |
| `util_ratio` | Continuo | $[0, \infty)$ (observado: $[0{,}117, 0{,}913]$) | Adimensional | Ratio de utilización promedio de las barras ($\text{solicitación}/\text{capacidad}$). Valores $> 1$ indican sobreutilización promedio y exceden el umbral principal; este agregado no permite diagnosticar por sí solo una falla estructural. |
| `z_disp_m` | Continuo | $[0{,}005814, 0{,}173201]$ | Metros (m) | Desplazamiento vertical Z. Rango observado verificado en los datos; la agregación exacta, nodo y combinación de cargas siguen pendientes de confirmación. |

---

## 5. Conjuntos temporales vigentes (`data/processed_evolution/`)

La partición principal asigna cada diseño por su primera generación: 0–63 ajuste, 64–79 calibración, 80–89 validación y 90–99 cierre. Los tamaños se calculan al preparar. Las categorías futuras no se mueven al pasado para equilibrarlas.

La revisión del 2026-10-21 confirmó 901 diseños de ajuste, 166 de calibración, 98 de validación y 108 de prueba, sin diseños compartidos entre bloques. Calibración y prueba solo contienen topologías 1 y 2; validación contiene 0, 1 y 2. Las topologías 3 y 4 aparecen únicamente en ajuste. Por tanto, prueba no permite evaluar el desempeño final en las cinco topologías. Estos conteos describen la preparación existente y deben comprobarse en `manifest.json` al cambiar de fuente o protocolo.

- **`full_clean.csv`:** diseños únicos con `design_id`, `first_generation`, representante e indicadores de umbrales.
- **`train.csv`:** primeras apariciones autorizadas para ajustar escalas, soporte, modelos y políticas.
- **`calibration.csv`:** bloque temporal disjunto para residuos e intervalos; la deriva puede limitar sus garantías.
- **`val.csv`:** selección predictiva y evaluación de desarrollo; no actualiza gradientes de políticas.
- **`test.csv`:** etiquetas retenidas del benchmark principal y de A para `make final`; la evaluación sigue siendo exploratoria por conocimiento previo de prueba.
- **`population_events.csv`:** todas las apariciones, repeticiones y respuestas originales.
- **`design_membership.csv`:** relación de cada aparición con representante y rol.
- **`manifest.json`:** Manifiesto técnico reproducible con esquemas, conteos por categoría, parámetros, nombres de salidas y hash SHA-256 del Excel fuente (`source_sha256`). No contiene hashes individuales de los CSV exportados; el hash del origen no acredita por sí solo la integridad de cada derivado.

Un superviviente que reaparece después sigue siendo conocido. No se entrena con frecuencia de supervivencia ni con última aparición futura. Las OOF progresivas dejan un bloque de calentamiento sin predicciones: cada predictor exporta su cobertura, y no se rellena con predicciones ajustadas sobre esas mismas etiquetas.

La búsqueda generativa usa únicamente ajuste para definir soporte y mundos de entrenamiento. Sus estimaciones no son mediciones nuevas. La recuperación A usa respuestas registradas tras la acción; el archivo final no se usa para entrenar la política. Los trabajos de preparación y entrenamiento se ejecutan manualmente.
