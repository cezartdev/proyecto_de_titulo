# Comparación de predictores y búsqueda guiada en cerchas 2D

Proyecto de título que compara once regresores por HV/AUC de sus selecciones históricas, precisión, estabilidad y costo. La búsqueda guiada es una aplicación exploratoria secundaria: **A** recupera diseños registrados y **B** genera configuraciones evaluadas por un sustituto. B no produce verificaciones FEM ni certifica seguridad estructural. La evidencia procede de una sola trayectoria previamente examinada.

## Primer uso desde un clon

Se requieren **Python 3.13, uv y GNU Make**. Ejecutar desde la raíz del repositorio, también en PowerShell. Comprobar que las herramientas estén en PATH:

```powershell
python --version
uv --version
make --version
make setup
make check
make plan
```

`setup` instala el entorno `.venv` usando `uv.lock`. `check` verifica estilo y formato sin corregir archivos. `plan` imprime configuración y presupuesto sin Excel, pesos, GPU ni entrenamiento. Debe mostrar once familias, cinco semillas, nueve estrategias y presupuestos **32/64/96**, incluida la población inicial de ocho. La búsqueda de políticas suma 1.000.000 de transiciones.

El clon no distribuye el Excel restringido, pesos ni resultados privados. Las pruebas pequeñas sin esos materiales se pueden ejecutar con:

```powershell
uv run --locked python tests/ejecutar.py -q tests/test_instalacion.py tests/test_cierre_vigente.py tests/test_benchmark_multiobjetivo.py
```

`make test` ejecuta la suite completa y queda para ejecución manual del autor, igual que preparación real, pilotos, entrenamientos, campañas y exportaciones pesadas. Los agentes siguen [AGENTS.md](AGENTS.md).

Las integraciones con materiales reales son optativas: `TRUSS_REAL_DATA_TEST=1` habilita la prueba del Excel autorizado y `TRUSS_TABPFN_REAL_TEST=1` la de pesos locales. No se descargan esos materiales ni se prepara el dataset real durante las comprobaciones predeterminadas.

## Preparar una campaña

Obtener autorización y colocar `240214-C1-20m.xlsx` en `data/raw/`, hoja `TOTAL DATA`. [data/README.md](data/README.md) explica variables, procedencia y límites.

El acelerador predeterminado es CPU. Los ejemplos siguientes usan `ACCELERATOR=cuda`, que exige una GPU NVIDIA utilizable por PyTorch. En un equipo sin GPU, sustituir consistentemente por `ACCELERATOR=cpu`. Solo TabPFN usa CUDA en la campaña; RL y los demás regresores permanecen en CPU.

```powershell
make prepare
make tabpfn-weights ACCELERATOR=cuda
make tabpfn-pilot ACCELERATOR=cuda
make smoke ESTUDIO=artifacts/comprobacion_pre_final_001 ACCELERATOR=cuda CPU_WORKERS=2
make plan ACCELERATOR=cuda CPU_WORKERS=12
```

Revisar `data/processed_evolution/manifest.json`, el piloto terminado bajo `artifacts/pretrained/tabpfn/pilots/` y las tareas/validación del smoke bajo `artifacts/comprobacion_pre_final_001_smoke/`. El piloto usa 128 diseños y todo el ajuste disponible; verifica persistencia e inferencia sin consultar prueba. El smoke es una comprobación técnica, no un resultado de tesis.

Los pesos se descargan solo mediante `tabpfn-weights`, con versión, revisión y SHA-256 fijados. Si el acceso requiere autenticación, autorizarlo primero en Hugging Face y definir `HF_TOKEN` en la sesión. El código no acepta licencias automáticamente ni imprime el token. Un error de acceso requiere revisar la autorización de la cuenta.

`prepare` reutiliza únicamente datos compatibles; no sobrescribe preparaciones. La limpieza del código no cambia los valores del Excel ni las particiones temporales. Un piloto anterior a un cambio de código puede dejar de ser compatible y debe repetirse.

## Desarrollo, revisión y cierre

El destino predeterminado común de Makefile, CLI y notebooks es **`artifacts/comparacion_tres_objetivos_final_001`**. Usar `ESTUDIO=artifacts/otro_nombre` en todas las etapas para otra ejecución. Se permiten hasta doce trabajadores CPU y un trabajador GPU para TabPFN.

```powershell
make study ACCELERATOR=cuda CPU_WORKERS=12
make explain EXPLAIN_MODE=pilot ACCELERATOR=cuda
```

Antes de abrir prueba, revisar:

- `study.json`: tareas predictivas y preparaciones completas.
- `evolution/predictor_utility/validation/`: HV/AUC observado por familia y presupuesto efectivo.
- `evolution/search.json`: evaluador, auditor y asistente seleccionados.
- `evolution/validation/`: comparación A/B, soporte, distancias, terminación y costos.
- `reports/<estudio>/explicabilidad_piloto_001/shap_costs.csv`: costo del piloto SHAP.

La familia se elige por AUC de validación al menor presupuesto, con MAE como desempate. Los tres objetivos son minimizar masa, maximizar utilización promedio dentro del umbral agregado y minimizar desplazamiento. El ajuste interno de regresores usa MAE. El presupuesto principal 96 evita agotar los bloques temporales actuales; el preflight rechaza presupuestos que excedan algún catálogo retenido.

Solo tras revisar desarrollo, ejecutar manualmente:

```powershell
make final
make explain ACCELERATOR=cuda
make benchmark
make report
make audit
make notebooks
make dev
```

`final` usa el dispositivo guardado, congela decisiones en `freeze.json` y termina con `evolution/closed.json`. Las exportaciones no entrenan ni vuelven a seleccionar la familia. `benchmark` consolida los costos ya medidos; no ejecuta una carga nueva. `dev` abre la presentación exportada.

SHAP explica validación usando referencia de entrenamiento y predictores guardados. El piloto no se presenta como resultado completo. `config/explain.json` configura casos, referencia, rondas y lotes; `winner_seeds=true` añade las semillas de la familia ganadora. `EXPLAIN_DEST=reports/<estudio>/explicabilidad_002` permite otra etapa independiente. Las atribuciones no demuestran causalidad física ni seguridad.

Los resultados incluyen tablas de utilidad/error/costo, todos los intentos predictivos registrados, dispositivo por respuesta y cobertura de recursos. La presentación muestra validación y prueba, diagnósticos observado/predicho y residuos, curvas HV y frentes A/B. Se exportan cuatro notebooks, cuatro HTML, figuras PNG/SVG y `figures.csv`. Revisar las firmas, ausencia de errores, unidades, presupuesto efectivo y `resources_complete` en `technical_closure.json`; un cierre científico no garantiza cobertura completa de las mediciones. [reports/README.md](reports/README.md) detalla los archivos y [notebooks/README.md](notebooks/README.md) las fuentes canónicas.

## Recorrido automático y recuperación

```powershell
make run-all ACCELERATOR=cuda DRY_RUN=1
make run-all ACCELERATOR=cuda CPU_WORKERS=12
```

El recorrido completo instala, comprueba, ejecuta la suite, prepara, realiza el piloto, desarrolla, **abre prueba automáticamente**, calcula SHAP completo y exporta. Usar las etapas separadas para revisar antes de `final`. `DRY_RUN=1` muestra el plan sin ejecutar estas etapas.

Para una comprobación sintética manual con límite de tiempo:

```powershell
make run-all RUN_MODE=check ACCELERATOR=cpu CPU_WORKERS=2 MAX_SECONDS=600
```

Este modo sigue realizando trabajo y puede ser costoso; omite SHAP. `make stop` detiene el supervisor y sus procesos. No usar dos coordinadores sobre el mismo estudio ni `make -j`.

Repetir etapas conserva tareas y episodios completos compatibles. No hay migración de campañas del protocolo retirado: estudios usan esquema 20 y predictores esquema 9. Cambios de código, parámetros, datos o dependencias requieren destinos nuevos cuando cambian sus firmas. Conservar los artefactos anteriores como evidencia; no editar sus firmas para forzar reanudación. `make reset-study DRY_RUN=1` muestra una limpieza explícita que también alcanza datos y pilotos compartidos; no usarla como paso rutinario.

## Opciones adicionales

```powershell
make interpolation INTERPOLATION_DEST=artifacts/interpolacion_final_001 ACCELERATOR=cuda
make train MODEL=catboost DRY_RUN=1
make train MODEL=catboost
make train MODEL=catboost TRAIN_MODE=fit
make train MODEL=evolution_rl PREDICTOR_DIR=ruta/al/predictor
```

La interpolación usa una partición aleatoria por diseños únicos y controles fijados en `analysis.interpolation_models`; guarda datos, modelos y reporte separados. `run-all INTERPOLATION=1` la añade al final. No mezclarla con la comparación temporal.

El entrenamiento individual `protocol` conserva selección anidada y OOF. `fit` es un diagnóstico fijo no elegible como evaluador de campaña. Las políticas admitidas son `evolution_rl`, `evolution_rl_assisted`, `evolution_contextual_bandit` y `evolution_contextual_bandit_assisted`; requieren un predictor completo, y las asistidas otro mediante `CHEAP_PREDICTOR_DIR`. No entrenan automáticamente esos predictores.

`make load-test`, `make cpu-benchmark` y `make gpu-test` son diagnósticos sintéticos manuales de recursos. `make format` corrige estilo y formato. No quedan comandos operativos de selección adaptativa, PPO ni sensibilidad del protocolo de catálogos.

## Referencias del proyecto

- [CONTEXT.md](CONTEXT.md): enfoque científico y bibliografía.
- [data/README.md](data/README.md): fuente, particiones y ficha física pendiente.
- [CODING_STANDARDS.md](CODING_STANDARDS.md): arquitectura y desarrollo.
- [LICENSE](LICENSE): condiciones del código y de los materiales restringidos.
