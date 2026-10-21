# Ejecución manual y entrenamiento

Los trabajos pesados quedan a cargo del autor. El agente no ejecuta automáticamente preparación del Excel real, campañas, búsquedas Optuna extensas, benchmarks ni la suite completa. Las verificaciones de desarrollo pueden usar tablas sintéticas pequeñas y evaluadores instrumentados.

## Entorno y destinos

Se requiere Python 3.13, uv y GNU Make. Se conserva un solo `pyproject.toml`, `uv.lock` y `.venv`. En Windows/Linux PyTorch CUDA también ejecuta en CPU; macOS no admite el perfil CUDA. TabPFN requiere pesos verificados y controlador NVIDIA para GPU.

`config/study.json` es el protocolo principal y `config/smoke.json` su versión reducida. `ACCELERATOR=cpu|cuda` modifica el dispositivo de TabPFN, no el entorno. RL y el resto de predictores permanecen en CPU. Se admiten hasta doce trabajadores CPU y uno GPU para TabPFN. Cada trabajador usa un hilo numérico.

El piloto independiente `make tabpfn-pilot` (también `make tabpfn-pilots`) detecta automáticamente CUDA con PyTorch: usa una GPU visible si CUDA está disponible y CPU en caso contrario. `ACCELERATOR=cpu` o `ACCELERATOR=cuda` fuerza el dispositivo; una elección explícita de CUDA falla si no está disponible. Se informa el dispositivo elegido en el log y se guarda en la identidad del piloto. No se cambia a CPU ante errores de pesos, memoria o entrenamiento. La detección respeta `CUDA_VISIBLE_DEVICES`, pero no reserva una GPU ni comprueba si está libre. En `run-all`, el piloto conserva el acelerador explícito de la campaña; para reutilizarlo, su dispositivo debe coincidir con el estudio.

El dataset temporal vigente se prepara en `data/processed_evolution/`; el estudio predeterminado es `artifacts/evolucion` o `artifacts/evolucion_gpu_001`. Usar un `ESTUDIO` nuevo al cambiar protocolo, datos, formulación, poda o parámetros. Las preparaciones antiguas se conservan y no se migran automáticamente.

## Desde una descarga limpia

Ejecutar los comandos desde la raíz que contiene `Makefile`, `pyproject.toml`, `uv.lock`, `src/` y `config/`. El archivo de bloqueo debe acompañar al código: no sustituirlo por una resolución nueva si se busca reproducir el protocolo. La ruta con espacios se admite; en PowerShell usar `Set-Location "ruta al repositorio"`. No hace falta activar `.venv` manualmente.

Instalar previamente uv y GNU Make y comprobar que se encuentran en `PATH`:

```powershell
uv --version
make --version
uv python install 3.13.5
make setup
make check
uv run --locked python tests/ejecutar.py tests/test_instalacion.py tests/test_run_all.py tests/test_benchmark_multiobjetivo.py tests/test_comparacion_predictiva.py -q
make plan ESTUDIO=artifacts/comparacion_tres_objetivos_001 ACCELERATOR=cpu CPU_WORKERS=2
make run-all ESTUDIO=artifacts/comparacion_tres_objetivos_001 ACCELERATOR=cpu CPU_WORKERS=2 INTERPOLATION=1 DRY_RUN=1
```

`setup` instala dependencias y herramientas de desarrollo y necesita acceso a los índices de paquetes, salvo caché completa. La instalación incluye PyTorch con CUDA en Windows/Linux aunque se elija CPU, por lo que puede descargar varios GB. El acelerador no selecciona una instalación más pequeña. Las pruebas seleccionadas usan datos sintéticos o procesos instrumentados; no descargan pesos ni ejecutan una campaña. Los dos planes no requieren Excel ni checkpoint y no crean un estudio. El plan del supervisor también puede verse antes de `setup` mediante `run-all ... DRY_RUN=1`, si uv y Python están disponibles.

Para reproducir la investigación real faltan dos recursos externos:

1. Obtener autorización y colocar el libro en `data/raw/240214-C1-20m.xlsx`, con la hoja `TOTAL DATA` y el contrato descrito en [datos](../data/README.md). El repositorio no lo descarga ni lo reemplaza por datos artificiales en modo completo. Sin él, `prepare` y la campaña real fallan.
2. Ejecutar `make tabpfn-weights ACCELERATOR=cpu` para obtener el checkpoint fijado y verificar su hash. El proveedor puede exigir acceso y aceptación de licencia en Hugging Face; el comando no acepta términos automáticamente. Si se necesita autenticación, el descargador lee `HF_TOKEN` del entorno. Los pesos y resultados no se distribuyen con el código.

CPU es la opción inicial portable. Para usar `ACCELERATOR=cuda`, comprobar disponibilidad antes del piloto:

```powershell
uv run --locked python -c "import torch; print(torch.cuda.is_available())"
```

Si muestra `False`, usar CPU o resolver el controlador y la compatibilidad de la GPU antes de entrenar. Las comprobaciones automatizadas del proyecto están configuradas para Windows y Linux; esta revisión local no sustituye una instalación real en otro sistema. Las pruebas que necesitan el Excel se omiten cuando falta; la integración con pesos reales es optativa.

## Protocolo final de comparación

El resultado principal es la calidad multiobjetivo observada que permite alcanzar cada regresor al seleccionar diseños históricos comunes, complementada con precisión, estabilidad y costo; la generación/búsqueda A/B es secundaria. El JSON principal usa los tres objetivos: minimizar masa y desplazamiento y maximizar utilización promedio bajo límites agregados. No se mantienen formatos anteriores: para este protocolo usar un destino nuevo, por ejemplo `ESTUDIO=artifacts/comparacion_tres_objetivos_001`. Los datos temporales existentes pueden reutilizarse si corresponden al manifiesto vigente.

Secuencia manual recomendada; `prepare`, pesos y piloto pueden reutilizar salidas compatibles. `prepare` no sobrescribe una preparación incompatible:

```powershell
make setup
make check
make prepare
make tabpfn-weights ACCELERATOR=cuda
make tabpfn-pilot ACCELERATOR=cuda
make plan ESTUDIO=artifacts/comparacion_tres_objetivos_001 ACCELERATOR=cuda CPU_WORKERS=12
make study ESTUDIO=artifacts/comparacion_tres_objetivos_001 ACCELERATOR=cuda CPU_WORKERS=12
```

La configuración completa compara once familias y nueve estrategias; cuatro políticas por cinco semillas y 50 000 interacciones suman 1 000 000 de transiciones. La variante reducida `make smoke` usa población cuatro y presupuestos 7/11, escribe en `<ESTUDIO>_smoke` y termina en validación. Una consulta A sin topología histórica compatible consume presupuesto y permite continuar; revisar `unsupported_queries` y distancias de proyección.

Revisar antes de `final` los errores físicos de validación, la dispersión entre semillas, categorías sin casos, cobertura OOF y resultados A/B. Si se revisan reglas o hiperparámetros, usar destino nuevo y mantener el carácter exploratorio; no elegirlos por el test.

Revisar primero `evolution/predictor_utility/validation/family_summary.csv` y `paired_intervals.csv`: todos los regresores y controles aleatorio/oráculo usan la misma inicialización, catálogo y ranking por dominancia/diversidad. HV y AUC usan respuestas observadas tras seleccionar. La familia evaluadora y el auditor se eligen por AUC de validación en el menor presupuesto configurado (32); MAE se usa como desempate. Los hiperparámetros siguen ajustándose internamente por MAE. `final` congela el benchmark de validación y solo entonces calcula el de prueba, sin nueva selección.

```powershell
make final ESTUDIO=artifacts/comparacion_tres_objetivos_001
make benchmark ESTUDIO=artifacts/comparacion_tres_objetivos_001
make report ESTUDIO=artifacts/comparacion_tres_objetivos_001
make audit ESTUDIO=artifacts/comparacion_tres_objetivos_001
make notebooks ESTUDIO=artifacts/comparacion_tres_objetivos_001
```

En `reports/comparacion_tres_objetivos_001/informe_*` revisar:

- `predictor_utility_summary.csv`, `predictor_utility_runs.csv` y `predictor_utility_paired_intervals.csv`: comparación principal por HV/AUC histórico observado. Revisar presupuesto efectivo, catálogo agotado y pares de escenarios. Las diferencias positivas de AUC favorecen al modelo frente a la referencia; el signo es distinto al de las diferencias de error. `oracle_ranking` es un diagnóstico con etiquetas completas, no un óptimo global ni una cota garantizada de AUC.
- `prediction_utility.csv`: une utilidad multiobjetivo y MAE normalizado de cada familia; no presupone que menor error implique mayor HV/AUC.

- `predictive_model_summary.csv`, `predictive_seed_metrics.csv` y `predictive_metric_summary.csv`: criterio normalizado global, errores físicos y variación de ajustes, separados por validación/prueba. El conteo de semillas no es tamaño muestral estructural.
- `predictive_paired_errors.csv`: diferencias modelo menos referencia. `generation_block` es el remuestreo principal y `design_iid_sensitivity` es sensibilidad. Revisar bloques efectivos y estado del intervalo. Son intervalos puntuales exploratorios, sin corrección por multiplicidad ni afirmación de significación/equivalencia.
- `predictive_seed_dispersion.csv`, `prediction_errors.csv` y `error_tail_summary.csv`: estabilidad y errores extremos, sin eliminar observaciones.
- `category_coverage.csv`, `predictive_*_generation_errors.csv`, `predictive_category_metrics.csv` y `predictive_region_metrics.csv`: categorías ausentes, muestras pequeñas y regiones de error.
- `predictive_interval_coverage.csv`: cobertura observada de intervalos, sin garantía frente a deriva o selección adaptativa.
- `predictor_costs.csv`, `predictive_inference_costs.csv`, `policy_costs.csv` y `comparison.csv`: costos separados; no sumar fases anidadas ni atribuir ahorro FEM. TabPFN incorpora preentrenamiento externo no medido; igual número de intentos no implica igual costo entre familias.
- `physical_provenance.csv`, `predictive_protocol.json` y `RESULTS.md`: procedencia, escalas y límites de interpretación.

La sensibilidad de interpolación requiere preparación y entrenamiento nuevos y también se ejecuta manualmente:

```powershell
make interpolation INTERPOLATION_DEST=artifacts/interpolacion_001 ACCELERATOR=cuda
```

Ejecuta secuencialmente las familias de `analysis.generalization.control_models` presentes en el estudio (por defecto Dummy, Ridge, CatBoost, XGBoost y TabPFN), con las semillas y presupuestos del JSON. No usa políticas ni modifica el estudio temporal. La selección por validación se guarda en `freeze.json` antes de evaluar prueba. Revisar `artifacts/interpolacion_001/report/` y su cobertura por categoría. Es interpolación en una fuente ya examinada, no validación externa ni una forma de reemplazar el ranking temporal. No se reanuda un destino parcial: conservarlo para diagnóstico y elegir otro.

Los tres objetivos están activos en los dos JSON canónicos por herencia desde `study.json`. `mass_displacement` solo se admite como sensibilidad explícita con otro `ESTUDIO`; no sustituye el estudio principal. No se modifica una campaña cerrada para reinterpretarla con otra formulación.

`freeze.json` fija políticas, predictores, validación de búsqueda y benchmark multiobjetivo de validación. El benchmark por regresor permanece en `evolution/predictor_utility/{validation,test}/`. La búsqueda final queda en `evolution/test/`; `evolution/test_predictors/` conserva las métricas de prueba de cada predictor, sin reajustarlo. Los intentos interrumpidos de evaluación predictiva se conservan aparte y se repiten sin reentrenar. `closed.json` indica cierre completo. Las exportaciones numeradas van a `reports/<estudio>/`: comparación, costos, métricas predictivas finales, procedencia y cuatro notebooks de presentación con HTML. En este flujo no se ejecutan celdas de entrenamiento. `make dev` abre la presentación exportada. El flujo evolutivo no exporta automáticamente SHAP ni las figuras históricas del protocolo anterior.

El piloto separado `proposals` fue sustituido por A/B dentro de `study`/`final`. Los comandos heredados de catálogo `adaptive`, `sensitivity` y `learning-curves` se rechazan para el experimento evolutivo. Las sensibilidades de soporte y objetivos se ejecutan como estudios separados con configuración explícita.

## Comprobación reducida y supervisor

```powershell
make run-all RUN_MODE=check ACCELERATOR=cuda MAX_SECONDS=600 CPU_WORKERS=2
```

Usa datos artificiales; no constituye evidencia científica. El límite incluye instalación y descarga y admite hasta cinco segundos adicionales de limpieza. También es un comando manual pesado.

`make run-all ACCELERATOR=cuda` ejecuta instalación, formato/lint, **suite completa**, preparación, pesos/piloto TabPFN, plan, smoke, desarrollo, cierre sobre prueba y exportaciones. No pausa antes de `final`. El modo `RUN_MODE=check` también recorre hasta cierre y reportes, pero con fuente artificial, presupuestos reducidos y una selección reducida de pruebas; `make smoke` por sí solo termina en validación. Para revisar resultados antes de prueba, preferir etapas separadas. Los análisis heredados de catálogo se omiten en este recorrido.

La opción `INTERPOLATION=1` agrega la sensibilidad al final, después de los notebooks del estudio principal:

```powershell
make run-all ESTUDIO=artifacts/comparacion_tres_objetivos_001 ACCELERATOR=cuda CPU_WORKERS=12 INTERPOLATION=1 DRY_RUN=1
make run-all ESTUDIO=artifacts/comparacion_tres_objetivos_001 ACCELERATOR=cuda CPU_WORKERS=12 INTERPOLATION=1
```

El primer comando solo muestra el plan; el segundo ejecuta la campaña completa y la sensibilidad. Por defecto, esta última escribe en `artifacts/comparacion_tres_objetivos_001_interpolacion/`, con su reporte en `report/`. `INTERPOLATION_DEST=artifacts/otra_interpolacion` permite otro destino, separado del estudio principal y los datos temporales. La elección se propaga al supervisor y la etapa se registra en `reports/comparacion_tres_objetivos_001/run_all.json`; `MAX_SECONDS`, si se especifica, cubre también la interpolación. La sensibilidad usa el mismo protocolo, fuente y acelerador, con ajustes secuenciales y presupuestos propios del JSON.

Sin `INTERPOLATION=1`, el recorrido no ejecuta esta sensibilidad. La opción se rechaza en `RUN_MODE=check`. Al repetir una campaña con interpolación terminada se verifican configuración, fuente, código y archivos antes de reutilizarla. Una sensibilidad parcial o incompatible se detecta antes de los entrenamientos de campaña: conservarla y elegir otro `INTERPOLATION_DEST`; no se borran ni sobrescriben esos archivos. Esto permite reanudar las etapas principales completas sin repetirlas para cambiar únicamente el destino complementario.

Ctrl+C solicita interrumpir. `make stop` detiene los árboles del supervisor del repositorio; no es un comando general para detener cualquier entrenamiento individual. Repetir etapas con los mismos parámetros reutiliza resultados compatibles. No iniciar dos coordinadores sobre el mismo estudio ni usar `make -j`.

## Entrenamiento individual

```powershell
make train MODEL=catboost SEED=42 DRY_RUN=1
make train MODEL=catboost SEED=42
make train MODEL=tabpfn SEED=42 ACCELERATOR=cuda
make train MODEL=catboost TRAIN_MODE=fit SEED=42
```

El modo predeterminado `protocol` conserva la búsqueda, OOF progresivas y selección de iteraciones. Un proceso de entrenamiento, sin SHAP, campaña global ni cierre sobre prueba. `TRAIN_CONFIG=config/smoke.json` reduce presupuestos. Los destinos son nuevos y numerados; `TRAIN_DEST` admite uno explícito y rechaza destinos existentes.

`fit` produce `run/diagnostic.json`, sin Optuna, OOF o calibración. Boosting ajusta árboles fijos sin bloque de parada; MLP mantiene su reserva interna, temporal cuando los datos lo son. No sirve como evaluador ni predictor RL, y no admite Targetwise o políticas. Poner solo `trials=0` no elimina OOF del modo protocol.

Para las políticas nuevas se necesita un predictor completo con modelos externos firmados y el mismo dataset temporal:

```powershell
make train MODEL=evolution_rl PREDICTOR_DIR=ruta/al/predictor/run
make train MODEL=evolution_rl_assisted PREDICTOR_DIR=ruta/al/evaluador/run CHEAP_PREDICTOR_DIR=ruta/al/asistente/run
make train MODEL=evolution_contextual_bandit PREDICTOR_DIR=ruta/al/predictor/run
make train MODEL=evolution_contextual_bandit_assisted PREDICTOR_DIR=ruta/al/evaluador/run CHEAP_PREDICTOR_DIR=ruta/al/asistente/run
```

El asistente debe ser distinto del evaluador. Los comandos no entrenan automáticamente sus dependencias. Las variantes de políticas del catálogo anterior siguen siendo controles de biblioteca, no son las políticas que entrena la campaña evolutiva.

## Poda, parada y medición

Se conserva `MedianPruner`, con cinco ensayos iniciales y mínimo de dos pliegues. Podados/fallidos consumen intentos y no ganan. CatBoost, XGBoost y LightGBM tienen paciencia de 30 iteraciones y reajuste con la mediana entera de árboles retenidos del ganador de su propia búsqueda. MLP restaura su mejor estado; su reserva temporal deja las últimas generaciones para parada. Targetwise mantiene las búsquedas por familia y respuesta. Dummy y TabPFN no usan Optuna.

Calibración, pliegue externo, validación global y prueba no controlan la parada interna ni la poda. Los parámetros y `refit_trees` se transportan en preparación paralela, persistencia y reutilización compatible.

Para comparar manualmente aceleraciones, usar destinos nuevos:

```powershell
make train MODEL=xgboost SEED=42
make train MODEL=xgboost SEED=42 TRAIN_EARLY_STOPPING_ROUNDS=0
make train MODEL=xgboost SEED=42 TRAIN_PRUNING=0 TRAIN_EARLY_STOPPING_ROUNDS=0
```

Estos controles afectan al entrenamiento individual y no desactivan la parada MLP. TPE puede proponer candidatos diferentes; comparar también error, pliegues y árboles ejecutados. No trasladar una estimación parcial de ahorro a la campaña completa.

`summary.json`, `phase_times.csv`, `run.resources.json` y los logs describen cada entrenamiento individual. Las fases pueden anidarse: usar duración global o `task_total`, sin sumarlas dos veces. Los costos predictivos, de aprendizaje RL y de búsqueda se reportan por separado.

`make load-test`, `cpu-benchmark`, `gpu-test` y `tabpfn-benchmark` siguen siendo diagnósticos manuales separados, no necesarios para cerrar la búsqueda. `make benchmark` en el experimento evolutivo exporta los costos ya almacenados.

## Verificaciones y limpieza

```powershell
make check
uv run --locked python tests/ejecutar.py tests/test_benchmark_multiobjetivo.py tests/test_busqueda_temporal.py tests/test_comparacion_predictiva.py tests/test_run_all.py -q
```

La selección anterior usa datos pequeños e instrumentados. La suite completa `make test`, los entrenamientos y las exportaciones reales se ejecutan manualmente. No se ha repetido una campaña científica completa como parte del cambio de código.

Para retirar derivados, ejecutar primero `make reset-study ESTUDIO=... DRY_RUN=1` y revisar las rutas. Sin `DRY_RUN`, se eliminan los destinos declarados, incluidos datos procesados compartidos y pilotos; se conservan Excel y pesos. No se necesita limpiar los resultados anteriores para usar los destinos nuevos.
