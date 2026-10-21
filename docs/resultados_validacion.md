# Antecedentes de validación y rendimiento CPU/GPU

Este documento reúne las mediciones registradas el 20 y 21 de octubre de 2026 que justifican la configuración actual. No se volvieron a ejecutar al reorganizar la documentación. Las rutas indicadas identifican las salidas originales; algunos artefactos locales ya no están presentes y no se han vuelto a verificar sus firmas. Para repetir las mediciones se debe crear un destino nuevo y conservar sus resultados completos.

El protocolo científico vigente está en [SCOPE.md](SCOPE.md) y los comandos en [ejecucion.md](ejecucion.md). Estas mediciones no son resultados del benchmark actual por HV/AUC, no acreditan su campaña completa y no fijan una selección de familias por MAE. La existencia de pilotos SHAP no implica que el recorrido evolutivo exporte explicaciones automáticamente.

## Configuración elegida y condiciones

Se utilizan hasta **12 trabajadores CPU y un trabajador GPU exclusivo para TabPFN**. MLP, boosting y RL permanecen en CPU. SHAP coordina las permutaciones en CPU y ejecuta en GPU las predicciones de TabPFN, incluidos sus componentes de `targetwise`.

Equipo de las mediciones: Intel Core i5-12400F, 6 núcleos físicos/12 hilos, 31,8 GiB de RAM y NVIDIA RTX 3060 de 12 GiB. Los pilotos utilizaron PyTorch 2.14.1+cu130, TabPFN 9.2.0 y checkpoint TabPFN 2.5 congelado, ocho estimadores y FP32. Parte de los pilotos precede al entorno raíz único y coincidió con otras verificaciones: sus tiempos son diagnósticos, no un benchmark final aislado.

Los pilotos de MLP, CatBoost y XGBoost midieron ajustes con hiperparámetros fijados, **no búsquedas Optuna completas en GPU**. Los resultados siguientes no demuestran una reducción equivalente para toda la campaña ni igualdad bit a bit entre CPU y GPU.

## TabPFN, SHAP e inferencia

Se cargó el mismo predictor guardado y se explicaron las mismas tres filas iniciales de validación, con referencia de 100 diseños, diez rondas y semilla 42.

| Medición de tres diseños | CPU | GPU | Cociente CPU/GPU |
| --- | ---: | ---: | ---: |
| Cálculo acumulado de permutaciones | 1.430,7 s | 51,1 s | 28,0 |
| Explicación completa, con carga y exportación | 1.439,6 s | 61,5 s | 23,4 |

Medianas de tres repeticiones de inferencia caliente:

| Tamaño del lote | CPU | GPU | Cociente CPU/GPU |
| --- | ---: | ---: | ---: |
| 1 | 0,530 s | 0,472 s | 1,1 |
| 32 | 1,644 s | 0,547 s | 3,0 |
| 64 | 2,842 s | 0,461 s | 6,2 |

La ganancia aumenta con el lote; el factor 28 corresponde a consultas enmascaradas de SHAP, no a cada predicción individual.

Las predicciones entre dispositivos cumplieron `rtol=1e-4`, `atol=1e-6`. Las diferencias absolutas máximas fueron 0,026946 kg de masa, 7,225e-7 de utilización y 1,131e-7 m de desplazamiento. Ambas explicaciones pasaron la reconstrucción SHAP en unidades físicas.

El compuesto `targetwise` completó tres explicaciones en GPU en 50,8 s de permutaciones y sus predicciones cumplieron la tolerancia respecto de las guardadas en CPU. Salidas originales: `reports/piloto_tabpfn_gpu_001`, incluido `targetwise_equivalence.json`.

## Entrenamiento de otros predictores

Se realizaron 18 ajustes: tres semillas por familia y dispositivo, con entrenamiento y validación.

| Familia | Mediana CPU | Mediana GPU | Interpretación |
| --- | ---: | ---: | --- |
| MLP | 0,489 s | 1,020 s | CPU más rápida |
| CatBoost | 6,076 s | 5,094 s | GPU redujo 16,2 %; no alcanzó el umbral de adopción del 20 % |
| XGBoost | 0,111 s | 1,034 s | CPU más rápida |

Los parámetros expuestos fueron comunes y se registró MAE por respuesta. Los valores internos predeterminados de boosting pueden diferir entre dispositivos; XGBoost advirtió sobre entradas CPU y predictor GPU, por lo que la medición incluye esa transferencia. No se presupone equivalencia exacta de los modelos. Salida original: `reports/piloto_estimadores_gpu_001`.

## Políticas RL

Se ejecutaron 5.000 transiciones por algoritmo y dispositivo, con semilla 42, incluyendo preparación de entradas, validación y checkpoints.

| Política | CPU | GPU |
| --- | ---: | ---: |
| Double DQN | 72,2 s | 71,6 s |
| Dueling Double DQN | 71,8 s | 75,2 s |
| PPO | 71,3 s | 86,9 s |

Ninguna alcanzó la reducción del 20 % prevista para adoptar GPU; no se realizó la confirmación condicional de 50.000 transiciones en GPU. Las redes son pequeñas y el entorno sigue en CPU. Es un piloto temporal de una semilla y no una comparación estadística de calidad ni una conclusión sobre redes mayores o entornos vectorizados.

Salida original: `artifacts/piloto_politicas_gpu_001`.

## Concurrencia

Las mismas cuatro explicaciones de un diseño, con referencia 100 y diez rondas, se midieron separando el calentamiento CUDA:

| Trabajadores GPU | Tiempo conjunto |
| --- | ---: |
| 1 | 101,82 s |
| 2 | 102,04 s |

La permutación por tarea pasó de aproximadamente 16–17 s a 41 s al compartir la tarjeta. Dos trabajadores no mejoraron el tiempo conjunto; se conserva uno para reducir competencia y memoria. Salida original: `reports/piloto_concurrencia_gpu_001`.

La primera calibración CPU con cargas mixtas sintéticas comparó 4/6/8/12 procesos y confirmó los dos más rápidos con tres repeticiones de 512 filas: mediana de 7,523 s con 12 frente a 8,288 s con 8. Recomendó 12 usando el menor número de procesos dentro del 5 % del más rápido. Salida original: `reports/piloto_cpu_cuda_001`.

La prueba manual posterior confirmó la misma elección:

| Medición | 8 procesos CPU | 12 procesos CPU |
| --- | ---: | ---: |
| Mediana de confirmación | 8,74394 s | 8,18814 s |
| Pico de RAM del sistema | 54,4 % | 62,8 % |
| Pico de RSS del árbol de procesos | 7.703,5 MiB | 11.169,8 MiB |

Doce procesos redujeron aproximadamente 6,36 % el tiempo por pasada respecto de ocho. La ejecución duró 148,09 s, sin pausas ni perfiles incompletos, y la recomendación no fue provisional. Salida original: `reports/pruebas_cpu/benchmark_001`.

Estas calibraciones no reproducen todos los picos de memoria ni la mezcla de una campaña completa. Los límites de RAM y VRAM siguen gobernando la admisión; la recomendación no garantiza que siempre haya doce procesos activos.

## Validación funcional

La comprobación integrada del 20 de octubre completó 23 tareas principales y 15 preparaciones con dos trabajadores CPU y uno GPU, en aproximadamente 572 s de comando. Las políticas declararon CPU y las explicaciones de TabPFN y `targetwise`, CUDA. Ambos explicaron los 127 diseños de validación con el presupuesto reducido.

Ese smoke cubría tres familias y no incluía las ablaciones RL actuales. El `smoke.json` vigente incluye once familias y sus ablaciones: los conteos y tiempos anteriores no representan ese alcance ampliado. Salidas originales: `artifacts/consolidacion_tabpfn_gpu_001_smoke` y `reports/validacion_tabpfn_gpu_001`.

También se verificaron:

- Ajuste TabPFN, persistencia y recarga en otro proceso con 128 y 814 diseños.
- Recuperación SHAP tras interrupción, equivalencia contra cálculo continuo y rechazo de checkpoints incompatibles.
- Enrutamiento de componentes mixtos, ausencia de CUDA, cierre de trabajadores y reanudación de políticas.
- Entorno raíz único: cambiar entre CPU y CUDA no reinstaló otra distribución de PyTorch; la recarga TabPFN en GPU coincidió con las predicciones guardadas con `rtol=1e-6`, `atol=1e-8`.
- Poda en todas las familias con Optuna, parada temprana en los tres modelos de boosting, conservación del mejor estado MLP, aislamiento externo y persistencia de cantidades seleccionadas.

Las comprobaciones funcionales no son mediciones de rendimiento. La unificación del entorno no repitió una campaña científica completa. La configuración vigente y sus comandos están en [ejecucion.md](ejecucion.md).

## Búsqueda de hiperparámetros y aceleraciones pendientes de medir

CatBoost con el protocolo completo de 30 intentos y validación anidada 5×3 realiza hasta 540 ajustes internos de pipeline, más cinco ajustes externos y uno final. Al entrenar tres respuestas separadas, supone hasta 1.638 ajustes escalares sin poda.

Una ejecución interrumpida llevaba aproximadamente 68,5 minutos; casi todo el tiempo observado correspondía a ajustes de candidatos. Un ajuste diagnóstico independiente, con 814 diseños, semilla 42, 100 árboles, profundidad 6, tasa 0,05 y regularización 1, tardó 6,046 s; su tarea medida tardó 6,531 s y obtuvo MAE normalizado de validación 0,108697. No equivale a la calidad de una búsqueda completa. Salida original: `artifacts/entrenamientos_individuales/catboost_fit_001`.

Durante la ejecución parcial posterior con poda se registraron siete candidatos descartados, evitando siete pliegues. El ahorro estimado a partir de sus pliegues ejecutados fue de unos 39 s, aproximadamente 4 % del tiempo de búsqueda observado. No fue una comparación controlada completa y no prueba una reducción final de ese porcentaje.

La parada temprana de CatBoost, XGBoost y LightGBM ya está implementada, pero todavía no hay una comparación completa y aislada de poda sola frente a poda con parada temprana. Registrar tiempo, árboles ejecutados y error antes de atribuirle una mejora cuantitativa.
