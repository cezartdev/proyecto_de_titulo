# Resultados del protocolo vigente

El contrato científico está en [SCOPE.md](../docs/SCOPE.md) y los comandos en [ejecucion.md](../docs/ejecucion.md). Las exportaciones se escriben en `reports/<nombre_del_estudio>/`; los modelos y el estado científico permanecen en `artifacts/<nombre_del_estudio>/`. Por ejemplo, `reports/comparacion_tres_objetivos_001/` corresponde a `artifacts/comparacion_tres_objetivos_001/`.

`make report`, `make audit`, `make benchmark` y `make notebooks` crean exportaciones numeradas `informe_*`, `auditoria_*`, `benchmark_*` y `presentacion_*`. Leen el cierre completo, no entrenan ni modifican modelos congelados. `benchmark` exporta costos almacenados; la presentación contiene cuatro notebooks y cuatro HTML de resultados.

El resultado principal compara cada regresor por HV/AUC observado sobre catálogos históricos comunes y tres objetivos. Revisar `predictor_utility_summary.csv`, `predictor_utility_runs.csv`, `predictor_utility_paired_intervals.csv` y `prediction_utility.csv`; los errores físicos, estabilidad, cobertura y costos son complementarios. `comparison.csv` y sus archivos asociados describen la búsqueda generativa A/B, separada del benchmark por regresor.

Los episodios y protocolos originales están en `artifacts/<estudio>/evolution/predictor_utility/{validation,test}/`. `freeze.json` fija modelos, políticas y validación antes de prueba; `evolution/closed.json` acredita la terminación. No se vuelve a escoger familia por prueba.

El supervisor `run-all` conserva `run_all.json` y `logs/` en esta carpeta. Con `INTERPOLATION=1`, añade una sensibilidad al final; sus datos, modelos y reporte quedan en `<ESTUDIO>_interpolacion/`, salvo `INTERPOLATION_DEST` explícito. No se mezclan sus resultados con la evaluación temporal.

Adaptación, sensibilidad de catálogos y curvas del flujo anterior no forman parte del recorrido evolutivo vigente. Las rutas `archive/estandarizacion_001/` son antecedentes históricos, no garantía de que existan en este checkout. Los artefactos privados no se redistribuyen en Git.

Una exportación describe la ejecución que la produjo; no se edita para presentarla como resultado de otro protocolo. Tras cambios científicos, usar un estudio nuevo y regenerar resultados manualmente.
