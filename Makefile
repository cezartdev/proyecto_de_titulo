.DEFAULT_GOAL := help
.PHONY: help setup prepare plan smoke study final benchmark cpu-benchmark gpu-test report audit notebooks dev check test format tabpfn-weights tabpfn-pilot tabpfn-pilots run-all reset-study train load-test interpolation explain stop
.NOTPARALLEL:

RUN_MODE ?= full
ACCELERATOR ?= cpu
PILOT_ACCELERATOR = $(if $(filter file default undefined,$(origin ACCELERATOR)),auto,$(ACCELERATOR))
ESTUDIO ?= $(if $(filter check,$(RUN_MODE)),artifacts/comprobacion_flujo,artifacts/comparacion_tres_objetivos_final_001)
STUDY_CONFIG ?= $(if $(filter check,$(RUN_MODE)),config/smoke.json,config/study.json)
SMOKE_CONFIG ?= config/smoke.json
CPU_WORKERS ?=
MAX_SECONDS ?=
DRY_RUN ?=
STUDY_ARG = --estudio "$(ESTUDIO)"
CPU_ARG = $(if $(strip $(CPU_WORKERS)),--cpu-workers "$(CPU_WORKERS)")
UV_RUN = uv run --locked
MODEL ?= catboost
SEED ?= 42
TRAIN_CONFIG ?= $(STUDY_CONFIG)
TRAIN_MODE ?= protocol
TRAIN_PRUNING ?=
TRAIN_EARLY_STOPPING_ROUNDS ?=
TRAIN_DEST ?=
PREDICTOR_DIR ?=
CHEAP_PREDICTOR_DIR ?=
INTERPOLATION ?= 0
INTERPOLATION_DEST ?= $(ESTUDIO)_interpolacion
LOAD_CONFIG ?=
LOAD_DEST ?=
EXPLAIN_CONFIG ?= config/explain.json
EXPLAIN_MODE ?= full
EXPLAIN_DEST ?=
EXPLAIN_ACCELERATOR = $(ACCELERATOR)

explain:
	$(UV_RUN) python -m truss_research.evaluacion.explicabilidad --estudio "$(ESTUDIO)" --config "$(EXPLAIN_CONFIG)" --mode "$(EXPLAIN_MODE)" --accelerator "$(EXPLAIN_ACCELERATOR)" $(if $(strip $(EXPLAIN_DEST)),--destino "$(EXPLAIN_DEST)") $(if $(filter 1,$(DRY_RUN)),--plan)

train:
	$(UV_RUN) python -m truss_research.evaluacion.entrenamiento_individual --modelo "$(MODEL)" --semilla "$(SEED)" --config "$(TRAIN_CONFIG)" --mode "$(TRAIN_MODE)" $(if $(strip $(TRAIN_PRUNING)),--pruning "$(TRAIN_PRUNING)") $(if $(strip $(TRAIN_EARLY_STOPPING_ROUNDS)),--early-stopping-rounds "$(TRAIN_EARLY_STOPPING_ROUNDS)") --accelerator "$(ACCELERATOR)" $(if $(strip $(TRAIN_DEST)),--destino "$(TRAIN_DEST)") $(if $(strip $(PREDICTOR_DIR)),--predictor "$(PREDICTOR_DIR)") $(if $(strip $(CHEAP_PREDICTOR_DIR)),--cheap-predictor "$(CHEAP_PREDICTOR_DIR)") $(if $(filter 1,$(DRY_RUN)),--plan)

load-test:
	$(UV_RUN) python -m truss_research.evaluacion.benchmark_cpu $(if $(strip $(LOAD_CONFIG)),--config "$(LOAD_CONFIG)") $(if $(strip $(LOAD_DEST)),--destino "$(LOAD_DEST)")

interpolation:
	$(UV_RUN) python -m truss_research.evaluacion.interpolacion --config "$(STUDY_CONFIG)" --destino "$(INTERPOLATION_DEST)" --accelerator "$(ACCELERATOR)"

run-all:
	uv run --no-project --python 3.13 python src/truss_research/evaluacion/recorrido.py --accelerator "$(ACCELERATOR)" --mode "$(RUN_MODE)" --estudio "$(ESTUDIO)" --config "$(STUDY_CONFIG)" $(CPU_ARG) $(if $(strip $(MAX_SECONDS)),--max-seconds "$(MAX_SECONDS)") $(if $(filter 1,$(INTERPOLATION)),--interpolation --interpolation-dest "$(INTERPOLATION_DEST)") $(if $(filter 1,$(DRY_RUN)),--plan)

reset-study:
	$(UV_RUN) python -m truss_research.evaluacion.reinicio --estudio "$(ESTUDIO)" --config "$(STUDY_CONFIG)" $(if $(filter 1,$(DRY_RUN)),--dry-run)

help:
	@echo "Instalar: make setup; comprobar sin datos privados: make check y make plan."
	@echo "Campana: prepare, tabpfn-weights, tabpfn-pilot, smoke, study; revisar y ejecutar final."
	@echo "Cierre: explain, benchmark, report, audit, notebooks; dev abre la exportacion."
	@echo "run-all ejecuta tambien final y SHAP completo; DRY_RUN=1 muestra etapas sin ejecutarlas."
	@echo "RUN_MODE=check usa una fuente sintetica y omite SHAP; MAX_SECONDS=600 limita el recorrido."
	@echo "ACCELERATOR=cpu|cuda se aplica a TabPFN; RL y demas regresores permanecen en CPU."
	@echo "CPU_WORKERS=12 limita procesos. ESTUDIO elige destino; no reutilizar versiones incompatibles."
	@echo "make explain EXPLAIN_MODE=pilot mide SHAP; EXPLAIN_DEST permite otro destino."
	@echo "make interpolation INTERPOLATION_DEST=artifacts/interpolacion_001 es una sensibilidad separada."
	@echo "make train MODEL=catboost DRY_RUN=1 muestra el ajuste individual; TRAIN_MODE=fit es diagnostico."
	@echo "make train MODEL=evolution_rl PREDICTOR_DIR=ruta; la variante asistida requiere CHEAP_PREDICTOR_DIR."
	@echo "make test ejecuta la suite completa manual. make format aplica formato; make check comprueba."
	@echo "make stop detiene run-all. make reset-study DRY_RUN=1 muestra la limpieza, incluidos datos compartidos."
	@echo "load-test, cpu-benchmark y gpu-test son diagnosticos sinteticos manuales de recursos."

setup:
	uv sync --locked

stop:
	$(UV_RUN) python -m truss_research.evaluacion.detencion

tabpfn-weights:
	$(UV_RUN) python -m truss_research preparar-tabpfn --config "$(STUDY_CONFIG)" --accelerator "$(ACCELERATOR)"

tabpfn-pilot tabpfn-pilots:
	$(UV_RUN) python -m truss_research probar-tabpfn --config "$(STUDY_CONFIG)" --accelerator "$(PILOT_ACCELERATOR)"

prepare:
	$(UV_RUN) python -m truss_research preparar-datos --config "$(STUDY_CONFIG)" --accelerator "$(ACCELERATOR)"

plan:
	$(UV_RUN) python -m truss_research ejecutar-estudio --destino "$(ESTUDIO)" --config "$(STUDY_CONFIG)" --plan $(CPU_ARG) --accelerator "$(ACCELERATOR)"

smoke:
	$(UV_RUN) python -m truss_research comprobar-estudio --config "$(SMOKE_CONFIG)" --destino "$(ESTUDIO)_smoke" $(CPU_ARG) --accelerator "$(ACCELERATOR)"

study:
	$(UV_RUN) python -m truss_research ejecutar-estudio --destino "$(ESTUDIO)" --config "$(STUDY_CONFIG)" $(CPU_ARG) --accelerator "$(ACCELERATOR)"

final:
	$(UV_RUN) python -m truss_research evaluar-final $(STUDY_ARG) $(CPU_ARG)

benchmark:
	$(UV_RUN) python -m truss_research medir-costos $(STUDY_ARG)

gpu-test:
	$(UV_RUN) python -m truss_research.evaluacion.benchmark_gpu

cpu-benchmark:
	$(UV_RUN) python -m truss_research.evaluacion.benchmark_cpu

report:
	$(UV_RUN) python -m truss_research resumir-estudio $(STUDY_ARG)

audit:
	$(UV_RUN) python -m truss_research auditar-estudio $(STUDY_ARG)

notebooks:
	$(UV_RUN) python -m truss_research presentar-notebooks $(STUDY_ARG)

dev:
	$(UV_RUN) python -m truss_research abrir-notebooks $(STUDY_ARG)

check:
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .

test:
	$(UV_RUN) python tests/ejecutar.py

format:
	$(UV_RUN) ruff check --fix .
	$(UV_RUN) ruff format .
