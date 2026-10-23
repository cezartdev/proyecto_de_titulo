# Resultados del protocolo vigente

Los modelos y decisiones quedan en `artifacts/<estudio>/`; las exportaciones, en `reports/<estudio>/`. El flujo manual y sus criterios de revisión están en el [README raíz](../README.md#desarrollo-revisión-y-cierre).

## Evidencia científica

`freeze.json` fija decisiones antes de prueba y `evolution/closed.json` acredita el cierre terminado. La selección por AUC de validación no se recalcula al presentar. A usa respuestas históricas proyectadas y B respuestas predictivas; estas últimas no acreditan FEM ni seguridad.

- `predictor_utility_summary.csv`, `predictor_utility_runs.csv`, `predictor_utility_paired_intervals.csv` y `prediction_utility.csv`: utilidad observada de los regresores, presupuesto efectivo y relación con precisión.
- `predictive_model_summary.csv`, `predictive_metric_summary.csv`, `predictive_seed_metrics.csv`, `predictive_paired_errors.csv`, `prediction_errors.csv`: errores físicos, semillas y contrastes exploratorios.
- `category_coverage.csv`, `predictive_category_metrics.csv`, `predictive_interval_coverage.csv`, `physical_provenance.csv`: cobertura, intervalos si existen y límites físicos.
- `comparison.csv`, `search_curves.csv`, `search_fronts.csv`, `termination_counts.csv`, `target_attainment.csv`: búsqueda A/B, curvas completas desde la inicialización y archivos representativos del escenario 0 y semilla representativa al mayor presupuesto.

## Costos y recursos

`predictor_costs.csv` separa búsqueda, OOF, ajuste, reutilización y dispositivo efectivo. `predictor_devices.csv` identifica el dispositivo por respuesta, incluidos los componentes Targetwise. `predictive_inference_costs.csv` describe inferencia. `policy_costs.csv` y `amortization.csv` describen las políticas y su aplicación sin atribuir ahorro FEM.

`resource_phases.csv` y `resource_tasks.csv` incluyen todos los intentos predictivos registrados, aun fallidos. `research_cost.csv` resume ese alcance predictivo; la suma de duraciones de trabajos no es tiempo transcurrido ni costo total de toda la investigación. `campaign_execution.csv`, `campaign_telemetry.csv` cuando existe y `worker_jobs.csv` conservan el detalle del coordinador predictivo. No sumar totales con sus fases anidadas. RSS es muestreada y su máximo por trabajador no es la suma simultánea de memoria.

`convergence_diagnostics.csv` conserva diagnósticos numéricos. `technical_closure.json` distingue cierre científico y `resources_complete`: recursos parciales deben declararse y no convertirse en mediciones completas.

Las políticas paralelas añaden `policy_execution.csv` y `policy_execution.json` para invocaciones, concurrencia, intentos y recuperación por memoria. `policy_resources.csv` registra las fases medidas de cada intento y su cobertura. El tiempo transcurrido del coordinador incluye arranque y cierre de procesos; no es la suma de tiempos de trabajadores. Estas tablas y `policy_costs.csv` son vistas complementarias del mismo trabajo, no costos aditivos. La telemetría original se conserva bajo `evolution/policy_execution/` y queda firmada en el cierre.

Cuando se reutilizan predictores, `predictor_import.json` identifica el origen histórico, las firmas conservadas y las copias locales verificadas. Sus costos predictivos corresponden al entrenamiento original, no a nuevos ajustes. `discarded_policy_costs.csv` conserva el intento secuencial excluido del nuevo entrenamiento de políticas; al proceder de checkpoints, sus tiempos son límites inferiores. La evidencia y las tablas originales se archivan localmente bajo `predictor_origin/`: las exportaciones no leen el estudio de origen ni requieren conservarlo disponible. `resources_complete` mantiene su alcance predictivo; la cobertura de políticas se consulta en `policy_resources.csv`.

## Presentación y SHAP

`report`, `benchmark` y `audit` comprueban el mismo cierre y crean exportaciones numeradas. `audit` guarda verificaciones de integridad y recursos sin duplicar las tablas de comparación. No son campañas nuevas ni auditorías externas independientes. `notebooks` añade los cuatro notebooks ejecutados, cuatro HTML, PNG/SVG y el índice `figures.csv`, a partir de las [fuentes canónicas](../notebooks/README.md).

SHAP tiene etapas independientes `explicabilidad_001/` y `explicabilidad_piloto_001/`. El recorrido automático completo ejecuta SHAP antes de exportar; por etapas se usa `make explain`. Se adjunta la última etapa completa compatible, con `shap_global.csv`, `shap_values.csv`, `shap_examples.csv`, `shap_costs.csv`, `shap_stability.csv` y `shap_provenance.json`. El piloto nunca se adjunta como resultado principal. Una tabla de estabilidad vacía no demuestra estabilidad entre ajustes.

Los manifiestos `provenance.json` firman cierre, procedencia y salidas. Cada exportación usa un destino nuevo; las parciales se conservan para diagnóstico. No se modifica una exportación para atribuirla a otra campaña. Los materiales privados están excluidos de Git.
