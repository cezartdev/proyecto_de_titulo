# Contexto de la investigación

## Propósito

Proyecto de título de pregrado: **Análisis del desempeño de modelos de aprendizaje automático en la predicción de resultados de optimización topológica en estructuras reticuladas tipo cercha**. Se conserva el título oficial.

El objetivo principal es comparar regresores de masa, utilización promedio y desplazamiento por la calidad multiobjetivo de las alternativas históricas que permiten seleccionar, medida con HV y AUC observados, junto con precisión, estabilidad y costo. Se minimizan masa y desplazamiento y se maximiza utilización promedio dentro del límite agregado. Como propuesta experimental secundaria se estudia generar y seleccionar candidatos mediante evolución genética, evolución diferencial, greedy, bandido y RL. No se exige superioridad de RL ni se acredita seguridad de configuraciones nuevas.

**Pregunta principal:** ¿Qué calidad multiobjetivo observada permiten alcanzar los regresores al seleccionar diseños de un catálogo histórico común, y cómo se relaciona esa utilidad con su precisión, estabilidad y costo?

**Pregunta secundaria:** ¿Qué desempeño presenta una propuesta de búsqueda guiada por modelos frente a estrategias de referencia, bajo presupuestos comunes de consulta y las limitaciones del archivo histórico y de los sustitutos?

La evaluación principal es temporal y exploratoria. La preparación vigente contiene 901 diseños de ajuste, 166 de calibración, 98 de validación y 108 de prueba; prueba solo representa las topologías 1 y 2. Estos conteos describen la preparación existente y deben comprobarse en su manifiesto. La sensibilidad por diseños únicos estudia interpolación en la misma fuente, no constituye nueva evidencia externa.

El control de bandido contextual usa el mismo estado y capacidad que Double DQN, pero aprende solo recompensa inmediata; permite estudiar el aporte del valor futuro. La variación diferencial mixta, con supervivencia Pareto compartida y asistencia opcional, contrasta el generador genético. Son adaptaciones propias; no se presentan como reproducciones de HIGA, SA-CoDE o qlDE. MCTS y control aprendido de operadores quedan como extensiones.

Se distinguen recuperación retrospectiva A, con respuestas registradas y proyección explícita, y generación B, con configuraciones nuevas y evaluación predictiva congelada. La secuencia de RL es la de sus propias decisiones y consultas; no se imita una trayectoria de padres y operadores originales que no está registrada.

## Datos y decisiones de enfoque

El Excel contiene 5.000 apariciones de una corrida de 100 generaciones y 50 individuos por generación. La auditoría previa identificó 1.273 diseños únicos y 3.727 repeticiones, asociadas a elitismo y convergencia. Estos conteos no fijan el tamaño de los nuevos bloques temporales.

La generación representa el ciclo evolutivo. El índice de solución identifica un individuo de la población, sin demostrar orden de evaluación ni relación de descendencia. Una repetición no prueba otra llamada FEM. Se conservan los eventos poblacionales para estudiar convergencia y un ejemplo por diseño exacto para regresión; la frecuencia de supervivencia no pondera el ajuste.

El protocolo principal prepara diseños por primera aparición en bloques contiguos de generaciones. Un superviviente sigue siendo conocido en los periodos posteriores. Los pliegues progresivos reservan calentamiento sin OOF y registran cobertura explícita; poda y selección de iteraciones usan únicamente datos internos anteriores. Se mantienen las once familias predictivas y sus optimizaciones de entrenamiento.

El autor confirma acero A36, fluencia de 250 MPa y comprobaciones según Eurocódigo 3. La profesora informó geometría 2D simétrica de 20 m, secciones por barra y evaluación FEM. Faltan cargas, apoyos, solver, algoritmo y asignaciones completas para reconstruir y verificar configuraciones nuevas. La utilización promedio no certifica cada barra ni demuestra por sí sola estabilidad estructural.

En B, una población persistente produce propuestas mixtas de cruce y mutación. RL selecciona antes de consultar; el resultado actualiza supervivencia y archivo. La variante asistida y greedy usan un predictor auxiliar, con sus costos contabilizados. Los modelos de evaluación no aprenden de sus propias predicciones como si fueran nuevas mediciones.

La investigación sigue siendo exploratoria: prueba ya fue examinada y solo existe una trayectoria estructural. Las semillas permiten medir variación del procedimiento, no generalización entre problemas estructurales independientes. La nueva partición no elimina el conocimiento previo de prueba.

## Qué aporta la literatura

Los 38 trabajos revisados abarcan sustitutos supervisados, evolución, RL, grafos y física. Orientan cuatro decisiones:

- Comparar familias diversas y distinguir error de predicción de calidad de optimización.
- Evaluar asistencia mediante predicciones con baselines y ablaciones, sin trasladar ganadores o aceleraciones de otros problemas.
- Comparar frentes Pareto y calidad durante consultas con objetivos, presupuestos y datos comunes.
- Distinguir componentes estándar de adaptaciones propias: el entorno tabular, la proyección al catálogo y la preselección por hipervolumen no reproducen automáticamente métodos publicados.

Los trabajos con grafos o física necesitan información ausente en el Excel. Un clasificador de cumplimiento no equivale a un regresor de las tres respuestas, y MCTS no implica una política tipo AlphaZero. Revisar documentos o código tampoco equivale a reproducir experimentos.

### Proceso de revisión sistemática

Según la información del informe aportada por el autor, la búsqueda se realizó exclusivamente en Scopus, con artículos en inglés de tipo artículo, publicación de congreso o revisión. Rayyan se utilizó para el cribado colaborativo por separado y en modo ciego, tanto de título/resumen como de texto completo. VOSviewer se utilizó para redes de coocurrencia de palabras clave y análisis bibliométrico.

| Etapa | Registros |
| --- | ---: |
| Identificados | 214 |
| Duplicados retirados | 1 |
| Tras deduplicación | 213 |
| Excluidos en título y resumen | 154 |
| Seleccionados para recuperación de texto completo | 59 |
| Textos no disponibles | 11 |
| Textos completos evaluados | 48 |
| Excluidos por incumplir criterios de inclusión | 10 |
| Incluidos en la síntesis final | 38 |

El flujo queda conciliado: 214 − 1 = 213; 213 − 154 = 59; 59 − 11 − 10 = 38. Los conteos agregados no sustituyen los registros de decisiones y motivos individuales de exclusión.

Consulta utilizada:

```text
TITLE-ABS-KEY ( truss OR "space frame" ) AND TITLE-ABS-KEY ( "topology optimization" OR "shape optimization" OR "size optimization" OR "sizing optimization" OR "layout optimization" OR "structural optimization" OR "truss optimization" OR "topology design" OR "design optimization" OR "optimal design" ) AND TITLE-ABS-KEY ( "machine learning" OR "deep learning" OR "neural network*" OR "surrogate model*" OR "data-driven" OR "reinforcement learning" OR "ensemble learning" ) AND ( LIMIT-TO ( DOCTYPE,"ar" ) OR LIMIT-TO ( DOCTYPE,"cp" ) OR LIMIT-TO ( DOCTYPE,"re" ) ) AND ( LIMIT-TO ( LANGUAGE,"English" ) )
```

Los criterios declarados en el informe son:

- **I1:** cerchas planas de barras articuladas en nodos, en 2D.
- **I2:** optimización topológica o de layout, con topología como variable de diseño; se admiten enfoques simultáneos topology-sizing o topology-shape. El texto aportado también admite sizing de secciones fijas, análisis estático/mecánico o variaciones de tamaño en busca de nuevas formas.
- **I3:** entrenamiento, validación o implementación de al menos un modelo ML como optimizador directo, sustituto o generador de diseños.
- **I4:** métricas cuantitativas computacionales o predictivas y de comportamiento estructural.
- **E1:** microestructuras, metamateriales y lattices fuera de la escala de ingeniería civil o mecánica.
- **E2:** optimización de medio continuo puro sin sistema reticulado de barras discretas.
- **E3:** ML limitado a monitoreo, daño, vida útil o análisis directo sin retroalimentación en diseño u optimización.
- **E4:** problemas exclusivamente 3D.
- **E5:** grafos abstractos sin modelación mecánica estructural.
- **E6:** documentos sin revisión por pares formal, incluidos editoriales, divulgación, resúmenes extendidos sin artículo completo y pósteres.

La elegibilidad de los artículos se mantiene sin cambios. Sigue pendiente justificar la inclusión de TSO-GCN, centrado en 3D, frente al criterio de cerchas 2D.

## Antecedentes del repositorio

El estudio 01 completó 205 tareas. XGBoost fue seleccionado por validación, con MAE normalizado medio 0,07291; en prueba obtuvo 0,07569, frente a 0,07410 de CatBoost. Greedy superó en AUC medio a RL en los tres presupuestos y la ablación no mostró ventaja consistente de incorporar predicciones.

Son antecedentes de un protocolo con búsqueda repetida por semilla y OOF cuyos hiperparámetros se elegían antes con todo entrenamiento. Compartía los tres objetivos actuales, pero usaba presupuestos 5/10/20. No se mezclan sus métricas con la campaña anidada actual. Las comprobaciones intermedias de dos objetivos quedaron descartadas y sus métricas no son evidencia de tesis.

Los formatos históricos se retiraron del flujo activo. Esta síntesis no sustituye sus artefactos para reconstruir resultados. El conocimiento previo de prueba permanece aunque se borren las ejecuciones.

## Referencias consultadas

Cada título enlaza directamente a su DOI cuando está completo. Se conservan los 38 trabajos del inventario documental, sin depender de archivos locales de apoyo.

- [A force neural network framework for structural optimization](https://doi.org/10.1016/j.engappai.2024.109991).
- [A hybrid intelligent genetic algorithm for truss optimization based on deep neural network](https://doi.org/10.1016/j.swevo.2022.101120).
- [A novel deep unsupervised learning-based framework for optimization of truss structures](https://doi.org/10.1007/s00366-022-01636-3).
- [A population-based DNN-augmented optimization method for designing truss structures](https://doi.org/10.1016/j.swevo.2024.101613).
- [A robust machine learning-based framework for handling time-consuming constraints for bi-objective optimization of nonlinear steel structures](https://doi.org/10.1016/j.istruc.2024.106226).
- [A unified evaluation framework for reinforcement learning paradigms in bi-objective truss optimization](https://doi.org/10.1016/j.aei.2026.104750).
- [AlphaTruss: Monte Carlo Tree Search for Optimal Truss Layout Design](https://doi.org/10.3390/buildings12050641).
- [An adaptive surrogate-based hybrid optimizer for constrained truss optimization](https://doi.org/10.1007/s00158-025-04227-0).
- [An Efficient Differential Evolution for Truss Sizing Optimization Using AdaBoost Classifier](https://doi.org/10.32604/cmes.2022.020819).
- [An efficient LightGBM-based differential evolution method for nonlinear inelastic truss optimization](https://doi.org/10.1016/j.eswa.2023.121530).
- [Application of Artificial Intelligence in Design Automation: A Two-Stage Framework for Structure Configuration and Design](https://doi.org/10.1061/JCEMD4.COENG-14409).
- [Combined sizing and shape optimization of truss structures using the Mixed Variable System Monte Carlo Tree Search (MVSMCTS) optimization algorithm](https://doi.org/10.1016/j.jocs.2026.102860).
- [Design Optimization of Truss Structures Using a Graph Neural Network-Based Surrogate Model](https://doi.org/10.3390/a16080380).
- [Efficient Hyperparameter Optimization Using Metaheuristics for Machine Learning in Truss Steel Structure Cross-Section Prediction](https://doi.org/10.3390/buildings15152791).
- [Evaluating Regression and Classification Surrogate Models for Sizing Optimization of Nonlinear Steel Trusses](https://doi.org/10.1007/978-981-96-5206-8_5).
- [Evaluation of Performance of Different Machine Learning Techniques for Structural Models](https://doi.org/10.1007/978-3-031-36246-0_23).
- [EvoDevo: Bioinspired Generative Design via Evolutionary Graph-Based Development](https://doi.org/10.3390/a18080467).
- [Frequency Prediction and Optimization of Truss Structure Based on BP Neural Network](https://doi.org/10.1145/3495018.3495079).
- [Graph Neural Network Based Surrogate Model for Design Informed Structural Optimization](https://doi.org/10.1007/978-981-96-8301-7_16).
- [Improved Monte Carlo tree search formulation with multiple root nodes for discrete sizing optimization of truss structures](https://doi.org/10.1080/0305215X.2024.2413962).
- [Intelligent generation method for innovative structures of the main truss in a steel bridge](https://doi.org/10.1007/s00500-023-07864-z).
- [Machine-Learning-Based Constraint Handling for Particle Swarm Optimization Within Structural Optimization](https://doi.org/10.1007/978-3-031-92029-5_30).
- [Machine-specified ground structures for topology optimization of binary trusses using graph embedding policy network](https://doi.org/10.1016/j.advengsoft.2021.103032).
- [Massively parallel structural design using stochastic optimization and mixed neuralnet/finite element analysis methods](https://doi.org/10.1016/0956-0521%2894%2990026-4).
- [Mastering Truss Structure Optimization With Tree Search](https://doi.org/10.1115/1.4068300).
- [Multi-objective optimization of truss structure using multi-agent reinforcement learning and graph representation](https://doi.org/10.1016/j.engappai.2023.107594).
- [Neural Network Approaches to Aid Simple Truss Design Problems](https://doi.org/10.1111/j.1467-8667.1994.tb00374.x).
- [Optimizing truss members using Machine Learning](https://doi.org/10.1051/e3sconf/202455904010).
- [Physics-informed neural energy-force network: a unified solver-free numerical simulation for structural optimization](https://doi.org/10.1007/s00366-022-01760-0).
- [Q-Learning-based parameter control in differential evolution for structural optimization](https://doi.org/10.1016/j.asoc.2021.107464).
- [Reinforcement Learning and Graph Embedding for Binary Truss Topology Optimization Under Stress and Displacement Constraints](https://doi.org/10.3389/fbuil.2020.00059).
- [Research on Comparative of Multi-Surrogate Models to Optimize Complex Truss Structures](https://doi.org/10.1007/s12205-024-0196-3).
- [Speeding up Composite Differential Evolution for structural optimization using neural networks](https://doi.org/10.1080/24751839.2021.1946740).
- [Topology and size optimization of trusses using graph neural networks: Towards efficient surrogate modeling](https://doi.org/10.1016/j.istruc.2026.111934).
- [Toward Reusable Surrogate Models: Graph-Based Transfer Learning on Trusses](https://doi.org/10.1115/1.4052298).
- [Truss structure optimization via hierarchical tree search](https://doi.org/10.1186/s40323-026-00320-1).
- [TSO-GCN: A Graph Convolutional Network approach for real-time and generalizable truss structural optimization](https://doi.org/10.1016/j.asoc.2023.110015).
- [Vertex-based graph neural network classification model considering structural topological features for structural optimization](https://doi.org/10.1016/j.compstruc.2024.107542).


