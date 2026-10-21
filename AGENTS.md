# Guías y Estándares Operativos para Agentes

Este documento establece los estándares operativos obligatorios, el flujo de trabajo de ejecución y las reglas de interacción para todos los agentes de IA.

---

## 1. Directrices Primarias y Referencia de Estándares

- **Estándares de Codificación y Arquitectura**: Todos los agentes DEBEN adherirse estrictamente a [CODING_STANDARDS.md](./CODING_STANDARDS.md) para la arquitectura de la aplicación, los límites modulares de *Screaming Architecture*, las reglas del lenguaje de programación y las convenciones de nomenclatura de archivos.
- **Trabajo pesado manual**: Por instrucción del autor, dejar para su ejecución manual la preparación del dataset real, entrenamientos, búsquedas extensas, campañas, benchmarks, exportaciones pesadas y la suite completa. Entregar comandos exactos y salidas que revisar. Se permiten revisión estática, formato, lint y pruebas sintéticas pequeñas pertinentes; no iniciar `make run-all`, `make study`, `make final` o `make test` automáticamente.
