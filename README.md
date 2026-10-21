# Comparación de predictores y búsqueda guiada por modelos en cerchas 2D de acero

Proyecto de título cuyo resultado principal es comparar regresores por la calidad multiobjetivo de sus selecciones históricas (HV/AUC observado), junto con precisión, estabilidad y costo. Como aplicación exploratoria se propone generar y seleccionar candidatos mediante búsqueda guiada por modelos. No se exige superioridad de RL. Las conclusiones se limitan a una trayectoria histórica; no acreditan seguridad de diseños nuevos ni ahorro FEM.

El estudio separa **A: recuperación retrospectiva con respuestas históricas**, mediante proyección a diseños registrados, y **B: generación de configuraciones nuevas evaluadas por un sustituto**. Las propuestas nuevas no tienen verificación FEM ni certificación de estabilidad física.

- [Protocolo científico](docs/SCOPE.md).
- [Mapa de la documentación](docs/README.md).
- [Ejecución manual y entrenamiento](docs/ejecucion.md).
- [Mediciones anteriores y límites de rendimiento](docs/resultados_validacion.md).
- [Plan y decisiones de implementación](docs/plan_busqueda_evolutiva_rl.md).
- [Motivación y bibliografía](CONTEXT.md) y [datos](data/README.md).

## Ejecución manual

Los comandos pesados los ejecuta el autor. Se requiere Python 3.13, uv y GNU Make y el Excel autorizado en `data/raw/240214-C1-20m.xlsx`, hoja `TOTAL DATA`.

Una descarga del repositorio permite instalar y comprobar el código, pero **no incluye el Excel restringido, los pesos de TabPFN ni resultados anteriores**. Ejecutar desde la raíz y conservar `uv.lock` en la distribución. Antes de una campaña, seguir [la instalación desde una descarga limpia](docs/ejecucion.md#desde-una-descarga-limpia). Los ejemplos con `ACCELERATOR=cuda` requieren GPU NVIDIA compatible; usar `ACCELERATOR=cpu` si no está disponible.

```powershell
make setup
make prepare
make tabpfn-weights ACCELERATOR=cuda
make tabpfn-pilot ACCELERATOR=cuda
make plan ESTUDIO=artifacts/comparacion_tres_objetivos_001 ACCELERATOR=cuda CPU_WORKERS=12
make study ESTUDIO=artifacts/comparacion_tres_objetivos_001 ACCELERATOR=cuda CPU_WORKERS=12
make final ESTUDIO=artifacts/comparacion_tres_objetivos_001 ACCELERATOR=cuda
make benchmark ESTUDIO=artifacts/comparacion_tres_objetivos_001
make report ESTUDIO=artifacts/comparacion_tres_objetivos_001
make audit ESTUDIO=artifacts/comparacion_tres_objetivos_001
make notebooks ESTUDIO=artifacts/comparacion_tres_objetivos_001
```

`plan` muestra presupuesto sin entrenar. `study` ajusta predictores y políticas y evalúa desarrollo. `final` congela modelos, políticas y validación antes de consultar las etiquetas finales de A; B sigue usando respuestas predictivas. No ejecutar `final` hasta revisar desarrollo.

La preparación temporal se escribe en **`data/processed_evolution/`** y conserva generaciones completas, diseños únicos y apariciones poblacionales. Los datos y resultados anteriores no se migran ni sobrescriben. `make prepare` reutiliza una preparación compatible; una existente de otro protocolo se rechaza. Elegir otro destino o respaldar y reiniciar explícitamente, sin alterar firmas para forzar reanudación.

## Configuración

Solo se mantienen dos archivos canónicos: `config/study.json` y `config/smoke.json`, que hereda del primero. El protocolo principal usa primera aparición por generación y once familias predictivas. Compara nueve estrategias: evolución genética, selección aleatoria, greedy, Double DQN y bandido contextual (ambos con y sin asistencia), y variación diferencial multiobjetivo con y sin asistencia. `search.strategies` selecciona explícitamente la matriz. Los presupuestos principales son 32/64/128 evaluaciones, incluida población inicial de ocho.

La formulación principal es `study.objective_mode="three"`: minimizar masa, maximizar utilización promedio dentro del límite agregado y minimizar desplazamiento. La utilización representa aprovechamiento de material, sin certificar seguridad por barra. Todos los regresores ordenan catálogos históricos comunes con dominancia y diversidad; la calidad se mide con respuestas observadas tras seleccionar. La familia se elige por AUC de validación en el menor presupuesto configurado, con MAE como desempate. La búsqueda generativa A/B es un experimento separado de este benchmark. Los intervalos predictivos de error conservan remuestreo por generaciones y diseños; los de utilidad remuestrean escenarios pareados tras promediar semillas. Son exploratorios, sin afirmaciones de significación ni equivalencia.

`make report` integra `predictor_utility_summary.csv`, `predictor_utility_runs.csv` y `predictor_utility_paired_intervals.csv` para HV/AUC observado por regresor. `predictive_paired_errors.csv`, `predictive_metric_summary.csv`, `predictive_seed_metrics.csv`, `category_coverage.csv` y `predictor_costs.csv` complementan con error, estabilidad, cobertura y costo. La sensibilidad `make interpolation INTERPOLATION_DEST=artifacts/interpolacion_001` usa otra partición por diseños únicos y el mismo benchmark multiobjetivo, sin modificar el estudio temporal.

Se entrenan cuatro políticas por semilla: con cinco semillas y 50 000 interacciones son 1 000 000 de transiciones en total, aparte del entrenamiento predictivo y la validación. El smoke usa población de cuatro y presupuestos 7/11 para admitir DE. Esta ampliación requiere un estudio nuevo; el dataset temporal preparado conserva su validez.

`ACCELERATOR=cpu|cuda` selecciona el dispositivo de TabPFN. RL, boosting y los otros predictores permanecen en CPU; hay hasta doce trabajadores CPU y uno GPU exclusivo para TabPFN. Se usa un único entorno raíz y las dependencias bloqueadas.

El piloto independiente `make tabpfn-pilot` o `make tabpfn-pilots` usa CUDA si PyTorch la detecta y CPU en caso contrario. `ACCELERATOR=cpu|cuda` permite forzar la elección. En `run-all` hereda el dispositivo de la campaña, y los pilotos solo se reutilizan con un dispositivo compatible.

Para una comprobación manual reducida con datos sintéticos:

```powershell
make run-all RUN_MODE=check ACCELERATOR=cuda MAX_SECONDS=600 CPU_WORKERS=2
```

Ese comando también puede ser costoso. El recorrido completo `make run-all ACCELERATOR=cuda` incluye instalación, formato/lint, suite completa, preparación, pilotos, desarrollo, cierre automático sobre prueba y exportación; usar las etapas separadas cuando se quiera revisar antes de cerrar.

Para añadir la sensibilidad de interpolación al final de la campaña:

```powershell
make run-all ESTUDIO=artifacts/comparacion_tres_objetivos_001 ACCELERATOR=cuda CPU_WORKERS=12 INTERPOLATION=1
```

`INTERPOLATION=0` es el valor predeterminado. Con `1`, los resultados complementarios van a `<ESTUDIO>_interpolacion`; `INTERPOLATION_DEST=ruta` cambia ese destino. La interpolación conserva sus propios datos, modelos y reporte. `DRY_RUN=1` muestra el plan sin instalación ni entrenamiento. La opción solo se admite en `RUN_MODE=full`. Una interpolación terminada se reutiliza tras verificar protocolo, fuente, código y firmas; una parcial requiere otro destino y se conserva para diagnóstico.

## Entrenamiento individual

```powershell
make train MODEL=catboost DRY_RUN=1
make train MODEL=catboost
make train MODEL=catboost TRAIN_MODE=fit
make train MODEL=evolution_rl PREDICTOR_DIR=ruta/al/predictor_completo
make train MODEL=evolution_rl_assisted PREDICTOR_DIR=ruta/al/evaluador CHEAP_PREDICTOR_DIR=ruta/al/asistente
make train MODEL=evolution_contextual_bandit PREDICTOR_DIR=ruta/al/predictor_completo
make train MODEL=evolution_contextual_bandit_assisted PREDICTOR_DIR=ruta/al/evaluador CHEAP_PREDICTOR_DIR=ruta/al/asistente
```

`protocol` conserva Optuna, poda, parada temprana y OOF progresivas. `fit` mide un ajuste fijo sin búsqueda ni OOF y produce un diagnóstico no elegible como evaluador. `TRAIN_PRUNING=0` y `TRAIN_EARLY_STOPPING_ROUNDS=0` permiten comparaciones manuales. Las políticas individuales exigen predictores completos del mismo dataset; no los entrenan automáticamente.

## Salidas y recuperación

Los modelos se guardan bajo el estudio y la búsqueda en `evolution/`: políticas, `search.json`, validación y prueba separadas, recibos por episodio, trazas, archivos de soluciones e intervalos pareados. `freeze.json` fija el cierre; `evolution/closed.json` acredita que terminó. Las salidas A y B mantienen procedencia diferente.

Repetir `study` o `final` conserva tareas y episodios completos compatibles. Cambios de código, datos, parámetros o dependencias pueden exigir un destino nuevo. No usar dos coordinadores sobre el mismo estudio ni `make -j`. `make stop` detiene el supervisor; `reset-study` es una limpieza explícita y debe revisarse primero con `DRY_RUN=1`.

`report`, `audit`, `benchmark` y `notebooks` leen el cierre y exportan resultados; no entrenan. En el nuevo protocolo, `benchmark` informa costos almacenados, no ejecuta una carga CPU/GPU nueva. Los cuatro notebooks se exportan como presentación de datos ya producidos; `make dev` abre esa exportación. El piloto separado `make proposals` se retiró y su función pasó al estudio principal. Los análisis heredados de catálogo `adaptive`, `sensitivity` y `learning-curves` no forman parte del recorrido evolutivo.

`make check` comprueba estilo, `make format` lo corrige y `make test` ejecuta la suite completa, que queda para ejecución manual. Las contribuciones siguen [CODING_STANDARDS.md](CODING_STANDARDS.md).

El dataset ya fue examinado y procede de una sola trayectoria. Las semillas no crean nuevas estructuras independientes; las conclusiones siguen siendo exploratorias.
