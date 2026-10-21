# Mapa de la documentación

El protocolo actual compara regresores por HV/AUC observado sobre catálogos históricos comunes, junto con precisión, estabilidad y costo. La generación y selección mediante búsqueda A/B es una propuesta experimental secundaria. Ambos componentes usan tres objetivos; las propuestas nuevas no tienen verificación FEM.

- [SCOPE.md](SCOPE.md): contrato científico vigente, particiones, métricas, selección por AUC de validación, tres objetivos y límites de evidencia.
- [ejecucion.md](ejecucion.md): comandos manuales, revisión antes de prueba, `run-all`, interpolación opcional, recuperación y salidas que revisar.
- [plan_busqueda_evolutiva_rl.md](plan_busqueda_evolutiva_rl.md): diseño implementado de generadores, selectores, evaluadores y comparadores de la propuesta secundaria.
- [resultados_validacion.md](resultados_validacion.md): antecedentes CPU/GPU y pilotos; no son resultados de la campaña científica actual.
- [CONTEXT.md](../CONTEXT.md): motivación, preguntas, decisiones de alcance y revisión bibliográfica.
- [Datos](../data/README.md) y [resultados](../reports/README.md): procedencia y contrato tabular; organización y lectura de exportaciones vigentes.
- [Resúmenes bibliográficos](../assets/summary): fichas de los 38 artículos. Sus métodos y cifras pertenecen a cada artículo; la relación con el proyecto no acredita reproducción.
- [AGENTS.md](../AGENTS.md) y [CODING_STANDARDS.md](../CODING_STANDARDS.md): reglas de trabajo, arquitectura y comprobaciones autorizadas.

Los valores ejecutables se centralizan en [study.json](../config/study.json), [smoke.json](../config/smoke.json) y [Makefile](../Makefile). Los ejemplos de ejecución usan un destino nuevo, `artifacts/comparacion_tres_objetivos_001`. Una discrepancia entre documentación y código se debe resolver explícitamente antes de ejecutar una campaña, sin alterar artefactos firmados para aparentar compatibilidad.

Las rutas de archivo anteriores y las métricas antiguas se conservan como antecedentes identificados. Los documentos de paquetes del entorno y los archivos privados o generados no definen el protocolo. Una exportación científica describe la ejecución que la produjo; se regenera manualmente desde un estudio nuevo cuando cambia el método.
