# Presentación canónica

`make notebooks` ejecuta estas cuatro fuentes sobre el cierre vigente. Se mantienen sin salidas en Git y no entrenan, predicen ni vuelven a seleccionar modelos:

- `01_datos.ipynb`: procedencia, duplicados, partición temporal, cobertura y distribución de ajuste.
- `02_prediccion.ipynb`: utilidad de validación/prueba, errores, intervalos, observado/predicho, residuos, costos y SHAP exportado.
- `03_seleccion.ipynb`: parámetros efectivos, búsqueda A/B en validación/prueba, curvas HV y frentes representativos.
- `04_conclusiones.ipynb`: decisiones congeladas, tablas y límites de la evidencia.

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

Las campañas del protocolo retirado no se presentan con estas fuentes. Las exportaciones reales quedan para ejecución manual según [AGENTS.md](../AGENTS.md). Consultar [README](../README.md), [CONTEXT.md](../CONTEXT.md) y [data/README.md](../data/README.md) para ejecución, métodos y procedencia.
