# Presentación canónica

`make notebooks` ejecuta estas cuatro fuentes sobre el cierre vigente. Se mantienen sin salidas en Git y no entrenan, predicen ni vuelven a seleccionar modelos:

- `01_datos.ipynb`: procedencia, duplicados, partición temporal, cobertura y distribución de ajuste.
- `02_prediccion.ipynb`: utilidad de validación/prueba, errores, intervalos, observado/predicho, residuos, costos y SHAP exportado.
- `03_seleccion.ipynb`: parámetros efectivos, búsqueda A/B en validación/prueba, curvas HV y frentes representativos.
- `04_conclusiones.ipynb`: decisiones congeladas, tablas y límites de la evidencia.

Las lecturas de resultados usan las tablas del informe mediante `evaluacion/lectura_resultados.py`: distinguen el máximo descriptivo en prueba de la familia congelada en validación, leen los contrastes pareados de búsqueda y resumen SHAP por respuesta. Las cifras y nombres no se copian de otra campaña ni se fijan manualmente en las conclusiones.

La comparación de tiempos separa búsqueda final, OOF anidadas y ajustes finales de todas las semillas de la inferencia de prueba. Incluye latencia individual, lote completo y costo por diseño dentro del lote; muestra medias entre semillas de las medianas medidas, tamaño de lote y dispositivo. La suma de fases no es duración de campaña y excluye el preentrenamiento externo de TabPFN. El gráfico de inferencia usa escala logarítmica. SHAP conserva la figura del evaluador y añade un mapa de importancia normalizada entre todas las familias.

La fuente de la selección es `freeze.json`. Las figuras se exportan como PNG y SVG y se inventarían en `figures.csv`, con notebook, celda, nombre y formato. Los frentes son diagnósticos de un escenario declarado; B usa estimaciones sin verificación FEM. Los costos predictivos y los de políticas conservan alcances separados.

## Abrir resultados

Después de `make final` y, si se requiere, `make explain`:

```powershell
make report
make notebooks
make dev
```

Los comandos usan por defecto `artifacts/comparacion_tres_objetivos_final_001`. Para otra campaña, añadir el mismo `ESTUDIO=artifacts/otro_nombre` a cada comando. La presentación nueva se guarda bajo `reports/<estudio>/presentacion_NNN/` y conserva sus firmas. Un fallo conserva el destino parcial; la siguiente invocación crea otro.

Para abrir las fuentes con un informe explícito:

```powershell
$env:TRUSS_STUDY = (Resolve-Path artifacts/comparacion_tres_objetivos_final_001).Path
$env:TRUSS_REPORT = (Resolve-Path reports/comparacion_tres_objetivos_final_001/informe_001).Path
uv run --locked jupyter lab notebooks
```

Elegir el número realmente exportado. Sin `TRUSS_REPORT` se verifica el último informe del estudio. Un informe inválido se rechaza. Sin SHAP completo compatible, el notebook muestra el paso pendiente; no calcula atribuciones. El recorrido automático completo ya ejecuta esa etapa antes de exportar.

Al ejecutar las fuentes directamente, `TRUSS_STUDY` tiene prioridad. Si no está definida, se usa el estudio predeterminado cuando existe; si falta, se reconoce el único estudio cerrado bajo `artifacts/`. Si hay varios cierres, es obligatorio elegir uno mediante `TRUSS_STUDY`. Para la campaña `_002`, los ejemplos de selección explícita deben apuntar a `artifacts/comparacion_tres_objetivos_final_002` y a su informe vigente. Tras actualizar el paquete, reiniciar el kernel antes de ejecutar las fuentes.

Las campañas del protocolo retirado no se presentan con estas fuentes. Las exportaciones reales quedan para ejecución manual según [AGENTS.md](../AGENTS.md). Consultar [README](../README.md), [CONTEXT.md](../CONTEXT.md) y [data/README.md](../data/README.md) para ejecución, métodos y procedencia.
