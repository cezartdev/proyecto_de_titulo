# Alcance y protocolo metodológico

## 1. Pregunta y evidencia

El resultado principal es comparar la utilidad multiobjetivo de los regresores para seleccionar alternativas registradas, medida con hipervolumen y AUC observados, junto con precisión, estabilidad y costo. La pregunta secundaria evalúa una propuesta de generación y selección mediante búsqueda guiada por modelos. RL es una estrategia candidata, no un requisito de superioridad ni la contribución principal.

Se comparan primero once familias predictivas y se congelan una familia evaluadora, una familia distinta de auditoría y un predictor asistente distinto de la familia evaluadora. La política principal observa entradas, presupuesto y archivo ya evaluado. La variante asistida y greedy comparten acceso al predictor auxiliar, con sus inferencias contabilizadas.

La evidencia se separa por experimento:

- **Benchmark principal:** selección en catálogos históricos comunes para todos los regresores; HV/AUC usa respuestas registradas, sin generar diseños ni proyectarlos. Se guarda en `evolution/predictor_utility/`, separado de las estrategias generativas A/B.

- **A, `historical_observed`:** se generan descendientes y se proyectan por distancia de entradas a diseños retenidos de la misma topología. Se incorporan al archivo las variables del diseño realmente evaluado, no las del descendiente con la respuesta de su vecino. La distancia se registra. Es una recuperación retrospectiva, no una verificación de propuestas nuevas ni una reproducción causal del algoritmo original.
- **B, `surrogate_estimated`:** no hay proyección. Las nuevas configuraciones se evalúan mediante un modelo congelado y después se auditan con otra familia. El acuerdo entre modelos no equivale a verdad física.
- **Controles `synthetic_exact`:** evaluadores matemáticos instrumentados prueban contratos y búsqueda sin error de sustitución. No representan el solver de las cerchas.

## 2. Datos y límites físicos

La fuente autorizada registra 100 generaciones con 50 individuos cada una. La auditoría histórica identificó 1.273 diseños únicos y 3.727 apariciones repetidas. Son antecedentes, no tamaños efectivos recalculados de la partición temporal nueva.

Las siete entradas describen topología nominal, conteos y alturas; las tres respuestas son masa, utilización promedio y desplazamiento. El autor confirma acero A36, fluencia de 250 MPa y comprobaciones según Eurocódigo 3. Se han informado geometría 2D simétrica, luz de 20 m, secciones por barra y evaluación FEM. Faltan cargas, apoyos, solver, algoritmo y asignaciones completas que permitan reconstruir y verificar una estructura nueva.

La utilización promedio no certifica cada barra ni diagnostica por sí sola una falla. Los umbrales agregados son criterios operativos. Se conservan registros íntegros aunque incumplan esos umbrales.

Las siete entradas no constituyen una descripción estructural completa. La consistencia de las etiquetas para entradas idénticas no demuestra que determinen la respuesta física: puede existir dimensionamiento implícito. La ficha de procedencia en `data/README.md` separa confirmaciones de información pendiente. Sin reconstrucción y verificación FEM no se acredita seguridad, estabilidad ni cumplimiento normativo de propuestas nuevas.

## 3. Preparación temporal

`data/processed_evolution/` contiene una vista de diseños únicos y `population_events.csv`, que conserva todas las apariciones, además de `design_membership.csv` y el manifiesto. La identidad depende de igualdad exacta de siete entradas; se rechazan respuestas contradictorias. El representante es la primera aparición según generación y un desempate reproducible por índice de solución.

La generación ordena ciclos evolutivos. El índice de solución identifica un individuo, no un orden de evaluación ni parentesco. No se construyen transiciones originales entre soluciones consecutivas. Las reapariciones son supervivencia del mismo diseño y no nuevos ejemplos ni recompensas.

Con los valores principales, se asignan primeras apariciones de generaciones 0–63 a ajuste, 64–79 a calibración, 80–89 a validación y 90–99 al cierre. Los cortes proceden de las proporciones configuradas, no de una estratificación aleatoria. Los conteos y categorías efectivos se revisan al ejecutar la preparación. Un superviviente que reaparece después sigue perteneciendo a su rol inicial.

La partición por diseño y las utilidades de purga permanecen para controles explícitos. No son la preparación principal. No se mezclan generaciones para equilibrar categorías futuras. La última aparición y la frecuencia total son información retrospectiva y no entradas de los modelos.

## 4. Ajuste predictivo

Se mantienen Dummy, Ridge, Random Forest, SVR, GPR, CatBoost, XGBoost, LightGBM, MLP, Targetwise y TabPFN. El ajuste y la selección interna de hiperparámetros conservan MAE normalizado. La familia evaluadora se elige por AUC observado medio de validación en el menor presupuesto configurado (32 en el principal, 7 en smoke), promediando semillas y escenarios; los desempates usan MAE normalizado medio y nombre de familia. La segunda familia según esta regla es el auditor. No se selecciona de nuevo por prueba. El benchmark no implica entrenar regresores directamente con una pérdida de hipervolumen.

Los pliegues progresivos ajustan únicamente con generaciones anteriores. La ventana inicial es calentamiento y no tiene OOF. `oof_coverage.json` identifica filas elegibles y excluidas; la preparación paralela conserva exactamente esa cobertura. Los modelos externos se guardan con su propia selección de parámetros y cantidades de reajuste.

Optuna conserva el presupuesto de 30 intentos cuando corresponde. `MedianPruner` decide después de al menos dos pliegues y cinco ensayos iniciales; podados y fallidos consumen intentos y no pueden ganar. Cada búsqueda externa tiene su historial independiente. Con dos pliegues no queda un tercero que ahorrar.

CatBoost, XGBoost y LightGBM usan parada interna de 30 iteraciones, máximo sugerido de 50–300 árboles y reajuste con la mediana entera de árboles retenidos del ganador completo. El bloque externo, calibración, validación global y prueba no deciden las iteraciones internas. MLP usa una reserva de las últimas generaciones de su ajuste y transforma entradas y respuestas solo con el subentrenamiento; restaura el mejor estado. Targetwise conserva selección de familias y búsqueda por respuesta. TabPFN es fijo y no realiza Optuna.

La calibración es un bloque temporal independiente. Se informan intervalos y errores observados, sin asumir que la deriva temporal satisface intercambiabilidad o que los intervalos certifican factibilidad fuera del soporte.

### Comparación predictiva principal

Cada regresor ordena el mismo catálogo de diseños únicos retenidos, sin generar geometrías ni proyectarlas a vecinos. Por escenario se muestrean una inicialización común desde ajuste y un catálogo común desde la partición retenida; se fija también un orden común para desempates. Todos aplican la misma regla: cumplimiento predictivo, violación de umbrales, rango de dominancia Pareto y diversidad por crowding. El orden es fijo para aislar el predictor; las respuestas ocultas no intervienen en ese orden. Después de cada selección se revela la respuesta histórica y se incorpora al archivo observado. HV y AUC se calculan solo con respuestas registradas, nunca con las predicciones del regresor evaluado.

Se usan los tres objetivos `three`: minimizar masa, maximizar utilización promedio dentro de [0, 1] y minimizar desplazamiento. Maximizar utilización representa aprovechar material dentro del criterio agregado; es una decisión operativa del estudio, no una comprobación por barra ni una instrucción matemática explícita del correo. L/700 se adopta como límite operativo recomendado, no se atribuye como restricción del optimizador original. Los diseños que incumplen se mantienen como ejemplos y consumen consulta, pero no aportan al frente admisible.

Las escalas y el punto de referencia HV se fijan con entrenamiento. La inicialización cuenta dentro del presupuesto, aunque sus respuestas ya sean conocidas. El tamaño solicitado del catálogo es el máximo entre `catalog_size` y los presupuestos; se limita a los diseños disponibles. Los presupuestos efectivos se limitan a inicialización más catálogo, se registran junto al solicitado y no se inventan consultas al agotarse. Los presupuestos de un escenario reutilizan prefijos de una sola trayectoria por modelo y semilla.

Se incluyen selección aleatoria y `oracle_ranking`, que aplica la misma regla con las etiquetas completas exclusivamente como diagnóstico retrospectivo. El oráculo no es un óptimo global ni una cota garantizada del AUC. El HV alcanzable del catálogo completo es una referencia descriptiva que no determina escalas, selección ni reglas. Los intervalos de diferencias de AUC promedian semillas antes de remuestrear escenarios pareados; una diferencia positiva favorece al modelo frente a la referencia. Son puntuales y exploratorios, condicionados a ajustes y catálogos superpuestos de una trayectoria, sin multiplicar el tamaño muestral por semillas ni afirmar significación o equivalencia.

Desarrollo guarda `evolution/predictor_utility/validation/`; el cierre congela este benchmark y los modelos antes de evaluar `test/`. Cada partición conserva protocolo, resumen por familia, episodios, decisiones, intervalos y firmas. La precisión física sigue siendo una dimensión complementaria del benchmark, junto con estabilidad y costo.

El MAE normalizado divide cada error absoluto por la desviación poblacional de su respuesta en ajuste y promedia las tres respuestas con pesos iguales. No es porcentaje de error relativo. MAE, RMSE, sesgo, percentil 95 y R² se presentan por respuesta; las métricas normalizadas no sustituyen las unidades físicas. Se informan costos de búsqueda de hiperparámetros, OOF, ajuste e inferencia por separado, con el presupuesto adicional de Targetwise y el preentrenamiento externo de TabPFN. No se suman fases anidadas.

Las diferencias son pareadas sobre los mismos diseños. Primero se promedian pérdidas absolutas entre semillas; luego se remuestrean bloques contiguos no superpuestos de generaciones de primera aparición, de longitud `analysis.predictive_block_generations` (dos por defecto), conservando todos sus diseños. El estimador pondera por diseños, no por el tamaño relativo de cada bloque. Se añade remuestreo individual por diseño como sensibilidad que ignora dependencia. Ambos usan el presupuesto y semilla bootstrap del JSON. Se informan bloques efectivos, tamaño muestral e intervalos percentiles del 95 %; con un solo bloque el intervalo queda indefinido. Pocos bloques y dependencia entre bloques limitan su estabilidad.

Los contrastes primarios comparan cada familia con Dummy y Ridge; los otros pares son secundarios. Una diferencia negativa favorece al modelo frente a la referencia. Los intervalos son puntuales, exploratorios y condicionados a estos ajustes y a esta trayectoria; no se ejecutan pruebas de significación ni se usa exclusión de cero como prueba confirmatoria. No hay corrección por multiplicidad ni intervalos simultáneos. Las diferencias pequeñas no demuestran equivalencia: esta exigiría tolerancias justificadas antes del cierre. La dispersión entre semillas se informa por separado y no multiplica observaciones independientes.

La cobertura por topología y los errores por generación, región y categoría se presentan con conteos. Una categoría ausente queda no evaluada. En la preparación vigente, prueba contiene 108 diseños de las topologías 1 y 2; no permite concluir desempeño para las otras tres. Se declaran por separado los errores extremos y la cobertura observada de intervalos calibrados.

La sensibilidad opcional `make interpolation` usa partición aleatoria estratificada por diseños únicos, reserva calibración y congela los controles antes de consultar prueba. Reutiliza las funciones de ajuste, presupuestos y reportes; las familias se fijan en `analysis.generalization.control_models`. Estudia interpolación en el mismo archivo ya examinado. No se mezcla con el ranking temporal ni acredita generalización externa; categorías escasas siguen limitando su interpretación.

## 5. Búsqueda y aprendizaje

La población inicial usa diseños de ajuste y respuestas históricas conocidas, comunes por semilla de escenario. Se cuentan dentro del presupuesto total para comparar inicializaciones iguales. El número de esas semillas no se presenta como nuevas llamadas FEM medidas.

El motor mantiene población reproductora, supervivencia por dominancia/diversidad y archivo evaluado separados. Genera descendientes mixtos; filtra límites, categorías, integridad y soporte sin consultar respuestas privadas; elige y solo entonces evalúa. Los candidatos inviables consumen consulta. Las proyecciones sin topología histórica compatible consumen consulta, no mejoran el archivo y permiten continuar hasta los límites comunes; se registran como `unsupported_queries` y no se reemplazan por predicciones. Su frecuencia y la distancia de proyección contextualizan el resultado A.

El presupuesto principal cuenta configuraciones consultadas, incluida inicialización, no recomendaciones. Se registran además descendientes devueltos por los operadores, regeneraciones, duplicados, inferencias por modelo y tiempo. Los intentos internos de los operadores están acotados. La recompensa marginal ponderada corresponde a mejorar temprano el AUC de hipervolumen sobre consultas, con descuento uno y escalas de entrenamiento.

Se comparan nueve estrategias: evolución genética, selección aleatoria, greedy asistido, Double DQN y bandido contextual (cada uno con y sin asistente), y variación diferencial multiobjetivo con y sin preselección greedy. Las trayectorias divergen después de decisiones diferentes; no se afirma que los lotes posteriores sean idénticos. Poblaciones iniciales, escenarios y límites sí se emparejan.

El bandido y Double DQN comparten estado, capacidad e interacciones de entrenamiento. El bandido aprende únicamente recompensa inmediata de la propuesta elegida, sin bootstrap ni etiquetas de propuestas descartadas. Las políticas quedan congeladas durante evaluación. DE cambia el generador, con tratamiento categórico de topología, reparación de enteros y supervivencia Pareto compartida; no se atribuye reproducción de un algoritmo publicado. La descripción operativa está en [el diseño de comparadores](plan_busqueda_evolutiva_rl.md).

El protocolo principal usa población ocho, lotes de 32 y presupuestos totales 32/64/128; cinco semillas de política y veinte escenarios. El smoke reduce cantidades. Durante entrenamiento se alternan mundos construidos con modelos externos y se reserva un modelo para seleccionar checkpoints con otras semillas. Son mundos aprendidos de una fuente común, no problemas estructurales independientes.

La implementación inicial entrena las políticas en esos mundos aprendidos y evalúa su transferencia también en A. No entrena otra política usando etiquetas de validación o prueba del archivo histórico. Por eso A es una comprobación retrospectiva complementaria, no aprendizaje por imitación de la trayectoria original.

El evaluador generativo de cierre y la familia de auditoría quedan fijados antes de prueba. B usa todas las semillas de la familia elegida, con media y penalización por desacuerdo según el multiplicador configurado. El auditor usa la media de su familia. Con una sola semilla el desacuerdo no está disponible; con ajustes deterministas puede ser cero sin implicar certeza. `uncertainty_scope` declara este límite. La dispersión entre modelos externos durante entrenamiento mezcla variación del ajuste y cantidad de datos, no es incertidumbre física calibrada. Las propuestas no se incorporan al entrenamiento supervisado como si sus predicciones fueran mediciones.

## 6. Métricas y cierre

Para la búsqueda experimental A/B se informa AUC de hipervolumen por consulta, calidad final, tasa de alcance y consultas hasta una fracción declarada de la referencia de entrenamiento (90 % por defecto), común a A y B y fijada antes del test. Son métricas distintas del benchmark principal por regresor. No se usa el frente de prueba para definir el umbral. El valor relativo puede superar uno sin demostrar un óptimo físico. Si la inicialización ya alcanza el objetivo se registra alcance inicial.

Los episodios no alcanzados quedan censurados, no se eliminan del análisis. Se añaden diversidad, soporte, incumplimientos, distancia de proyección, novedad y calidad estimada por el auditor. Los intervalos de diferencias de AUC contrastan RL–bandido, selectores–controles y DE–generador genético. Remuestrean semillas de aprendizaje y escenarios; no tratan pasos como réplicas independientes. Son intervalos exploratorios sin corrección por multiplicidad.

El protocolo principal usa `study.objective_mode="three"` tanto en el benchmark de regresores como en la búsqueda experimental: minimizar masa y desplazamiento y maximizar utilización promedio bajo límites agregados. `mass_displacement` solo se admite como sensibilidad explícita en otro estudio y no sustituye el resultado principal de tres objetivos. Cambiar formulación o soporte exige configuración y destino nuevos, no modificar un estudio cerrado.

`study` desarrolla y evalúa validación; `final` congela modelos, políticas, reglas y archivos antes de consultar etiquetas de prueba de A. Las firmas y recibos permiten reanudar episodios compatibles. Los informes y notebooks leen el cierre sin entrenar ni modificarlo.

El cierre también evalúa cada familia y semilla predictiva sobre prueba, sin selección adicional ni reajuste, y guarda sus errores físicos, cobertura y costos de inferencia separados de la búsqueda. No se vuelve a escoger familia por esos resultados.

Poda y parada temprana afectan costo de construir modelos; RL pretende reducir evaluaciones de búsqueda. Esos costos y los del entrenamiento de políticas se presentan separados. No se suman fases anidadas ni se extrapola ahorro FEM sin mediciones del solver.

## 7. Interpretación

Se conoce una sola trayectoria y prueba ya fue examinada. La nueva partición no elimina ese antecedente. No se exige que RL supere greedy, ni se atribuye a RL una mejora causada por poda o reajuste. Una cobertura insuficiente o una ausencia de ventaja son resultados admisibles.

La generación histórica es una referencia descriptiva: sin un orden confirmado dentro de cada población, no se inventan consultas individuales ni se presume que 50 apariciones equivalen a 50 ejecuciones FEM nuevas. La búsqueda aprendida crea su propia secuencia; no imita padres y operadores originales desconocidos.
