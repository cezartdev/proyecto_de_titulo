# Estándares de Codificación y Especificación de Arquitectura de la Aplicación

Este documento detalla la arquitectura del sistema, las convenciones de directorios, las reglas de dependencia entre capas, los estándares de codificación y los flujos de trabajo de desarrollo.

---

## 1. Idioma, Herramientas y Estándares de Calidad

- **Idioma Predeterminado**: Al tratarse de una investigación y proyecto de título universitario desarrollado en español, toda la documentación técnica, mensajes de confirmación de git (commit messages), docstrings y comentarios DEBEN estar redactados en **Español**. Para el código (nombres de variables, funciones y campos de Pydantic), se permite el uso de **Español** o **Inglés**, manteniendo la consistencia en toda la base de código.
- **Gestor de Paquetes**: Usar **`uv`** de forma exclusiva (`uv add`, `uv run`, `uv sync`). Nunca usar directamente `pip`, `poetry` o `conda`.
- **Automatización de Scripts**: **Makefile** es obligatorio para la automatización de scripts y para estandarizar los flujos de trabajo de desarrollo comunes (`make dev`, `make check`, `make format`).
- **Formateador y Linter de Código**: **Ruff** es obligatorio. Ejecutar `make format` y `make check` antes de confirmar cambios (commit).
- **Política de Comentarios**: NO escribir comentarios de una sola línea línea por línea (por ejemplo, `# comentario en línea`). El código debe ser limpio y autodocumentado. Usar exclusivamente comentarios de bloque o docstrings para módulos y solo cuando sea estrictamente necesario.
- **Legibilidad y Estructura en Notebooks (Jupyter)**: En los notebooks se prioriza ante todo la legibilidad y la presentación narrativa. Se debe estructurar el flujo de trabajo utilizando celdas de Markdown para títulos, subtítulos, contexto y explicaciones teóricas/metodológicas, antes que recurrir a comentarios dentro de las celdas de código.

---

## 2. Principios Fundamentales de Ingeniería y Arquitectura

- **Redundancia de Código, Simplicidad y Patrones de Diseño**: Eliminar la lógica duplicada en los módulos. Abstraer para extensibilidad utilizando patrones de diseño probados (como Strategy) donde sea apropiado. Eliminar, reducir o consolidar archivos siempre que el código pueda simplificarse. Evitar ramas condicionales anidadas: exigir un máximo de un nivel de ramificación condicional por función, prefiriendo tablas de despacho (dispatch tables), salidas tempranas (early exits) o polimorfismo. Priorizar las utilidades estándar e integradas (built-in) sobre código personalizado. Adherirse estrictamente a los principios de "menos es más", "lo simple es mejor que lo complejo" y "no reinventar la rueda".
- **Evaluación y Mitigación de Casos Límite (Edge Cases)**: Identificar, analizar y mitigar sistemáticamente casos límite (por ejemplo, entrega duplicada de mensajes, reentrancia, latencia de red, entradas malformadas, cargas útiles vacías y conflictos de ejecución concurrente).
- **Anotaciones de Tipos Estrictas y Prohibición de Tipos Comodín**: Exigir anotaciones de tipo (type hints) explícitas y exhaustivas en todas las firmas de funciones, parámetros, tipos de retorno y contenedores de estado. Evitar tipos comodín o sin restricciones (como `Any` u objetos arbitrarios no tipados). Preferir modelos de dominio concretos, esquemas estructurados, mapeos especializados o genéricos acotados siempre que los datos posean una estructura identificable.
- **Cero Valores Mágicos y Constantes Hardcodeadas**: Prohibir estrictamente números mágicos, cadenas mágicas y literales no gestionados. Centralizar todos los umbrales de configuración, valores predeterminados, tiempos de espera (timeouts) y claves de negocio en constantes con nombre dedicadas u objetos de configuración.
- **Programación por Aserción y Cláusulas de Guarda**: Priorizar la programación por aserciones y salidas tempranas (guard clauses). Los casos que invaliden la ejecución o violen precondiciones deben verificarse y rechazarse primero al inicio de la rutina, manteniendo la ruta principal de ejecución limpia y sin anidamientos.
- **Manejo Explícito de Errores y Observabilidad**: Nunca dejar errores o excepciones sin gestionar. Las excepciones capturadas nunca deben fallar en silencio ni retornar estados vacíos o poco informativos; siempre deben registrar una advertencia de diagnóstico y retornar un modelo de error explícito y estructurado o una alternativa funcional (fallback) significativa.
- **Dimensionamiento Prudente de Archivos y Bajo Acoplamiento**: Mantener los archivos cohesivos y evitar módulos sobredimensionados. Evitar la sobrefragmentación en microarchivos excesivos si eso genera una sobrecarga innecesaria entre archivos. Abstraer lógica únicamente cuando favorezca el bajo acoplamiento sin introducir dependencias enredadas.
- **Límite de Parámetros (Máximo 3 Parámetros)**: Las funciones y métodos no deben exceder los tres parámetros. Cuando se requieran cuatro o más parámetros, agruparlos en una estructura de datos cohesiva, un objeto de comando o un modelo de configuración.
- **Validación Temprana al Inicio (Fail-Fast Startup Validation)**: La aplicación debe validar todas las variables de entorno obligatorias y dependencias operativas durante la inicialización y fallar de inmediato (interrumpir tempranamente la ejecución) si alguna configuración crítica falta o es inválida.

## 3. Organización por Capacidades y Convenciones

- El código reutilizable reside en el paquete instalable `src/truss_research`. Los módulos vigentes se organizan por capacidades del estudio: `cerchas`, `datos`, `surrogates`, `evaluacion` y `optimizacion`.
- `cerchas` depende únicamente de la biblioteca estándar. `datos` puede depender de `cerchas`; el dominio no depende de pandas, PyTorch, algoritmos de entrenamiento ni archivos del repositorio.
- Evitar directorios genéricos `core`, `utils` o `models`. Cada función y esquema pertenece a su capacidad. No crear carpetas o abstracciones vacías para funcionalidades futuras.
- Usar nombres de archivos y paquetes en `snake_case`, clases en `PascalCase` y constantes en `UPPER_SNAKE_CASE`. Los identificadores de código se mantienen en inglés; las explicaciones, docstrings y documentación, en español.
- Los notebooks importan el paquete instalado y mantienen narrativa y visualización; no duplican lógica de preparación ni modifican `sys.path`.
- El entrenamiento, los transformadores y las métricas se implementan en `src`. Los notebooks cargan artefactos existentes y muestran procedimiento, visualización y conclusiones. Los entrenamientos se ejecutan mediante Makefile usando esas mismas funciones.
- Ajustar preprocesamiento y modelos únicamente con entrenamiento; usar validación para comparar y reservar prueba para la evaluación final. No sobrescribir directorios de experimentos existentes.
- Importar módulos no debe leer datos, configurar logging global, entrenar modelos ni crear directorios. La ejecución ocurre mediante funciones explícitas o el comando del paquete.
- `make check` es de solo comprobación; `make format` aplica correcciones; `make test` ejecuta pruebas. Las variables de entorno solo se exigen cuando una capacidad concreta las requiere.
- Validar integridad de datos por separado del cumplimiento de umbrales agregados. Conservar los diseños inviables y no interpretar utilización promedio como certificación por barra.

El contexto de la investigación se resume en [CONTEXT.md](CONTEXT.md), el contrato científico y sus límites en [SCOPE.md](docs/SCOPE.md), y la instalación y ejecución en [README.md](README.md). Las referencias bibliográficas usan títulos y DOI, sin depender de archivos temporales.

- Makefile es la entrada habitual. Los parámetros científicos se centralizan en JSON; `ESTUDIO` se admite para separar réplicas y consultar resultados concretos.
- `make study` compara usando validación; `make final` es el cierre explícito sobre prueba. Una reanudación compatible conserva tareas terminadas.
- Las nuevas OOF usan validación anidada para excluir cada pliegue externo tanto de la selección de hiperparámetros como del ajuste.
- Mantener únicamente el formato vigente de modelos y políticas; los resultados históricos se excluyeron del flujo activo por decisión del autor. La presentación no modifica artefactos científicos firmados.
- La comparación predictiva es el resultado principal; la búsqueda guiada por modelos es una aplicación exploratoria secundaria. Los reportes vigentes incluyen contrastes pareados por bloques y por diseño, variación entre semillas sin pseudorreplicación y cobertura por topología. No declarar equivalencia, seguridad de configuraciones nuevas ni ahorro FEM a partir de errores o acuerdo entre modelos.
- El benchmark principal compara todos los regresores con catálogos e inicializaciones comunes y HV/AUC calculados con respuestas históricas observadas. La selección de familia usa AUC de validación en el menor presupuesto configurado, con MAE como desempate. La formulación principal mantiene los tres objetivos; no reemplazarla por masa/desplazamiento. El ajuste interno de regresores conserva MAE; el benchmark no implica entrenar directamente una pérdida de hipervolumen.
- Mantener las guías operativas alineadas con JSON, Makefile y código. Los resúmenes bibliográficos describen artículos, no el protocolo del proyecto; sus apartados de relación remiten al contrato actual. Las mediciones y exportaciones históricas se identifican como antecedentes y no se editan para atribuirlas a una campaña nueva.
