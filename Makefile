.DEFAULT_GOAL := help
.PHONY: help setup prepare plan smoke study final sensitivity learning-curves adaptive benchmark cpu-benchmark gpu-test report audit notebooks dev check test format tabpfn-weights tabpfn-pilot tabpfn-pilots run-all reset-study train load-test interpolation
.NOTPARALLEL:

RUN_MODE ?= full
ACCELERATOR ?= cpu
PILOT_ACCELERATOR = $(if $(filter file default undefined,$(origin ACCELERATOR)),auto,$(ACCELERATOR))
ESTUDIO ?= $(if $(filter check,$(RUN_MODE)),artifacts/comprobacion_flujo$(if $(filter cuda,$(ACCELERATOR)),_gpu),artifacts/evolucion$(if $(filter cuda,$(ACCELERATOR)),_gpu_001))
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
	@echo "tabpfn-pilot detecta CUDA automaticamente; ACCELERATOR=cpu|cuda permite elegir explicitamente."
	@echo "run-all INTERPOLATION=1 incluye sensibilidad al final; INTERPOLATION_DEST elige destino separado; DRY_RUN=1 muestra el plan."
	@echo "Comparacion principal: predictores; aplicacion secundaria: busqueda guiada A/B, sin seguridad de propuestas nuevas."
	@echo "Sensibilidad manual por disenos unicos: make interpolation INTERPOLATION_DEST=artifacts/interpolacion_001"
	@echo "Entrenamiento individual: make train MODEL=catboost SEED=42; DRY_RUN=1 muestra presupuesto."
	@echo "Ajuste corto sin Optuna ni OOF: make train MODEL=catboost TRAIN_MODE=fit"
	@echo "RL evolutivo individual: make train MODEL=evolution_rl PREDICTOR_DIR=ruta/al/predictor; asistido requiere CHEAP_PREDICTOR_DIR."
	@echo "Bandido contextual individual: MODEL=evolution_contextual_bandit o evolution_contextual_bandit_assisted."
	@echo "Prueba de carga sintetica manual: make load-test; informe en reports/pruebas_cpu."
	@echo "Campana CPU con TabPFN en GPU: make run-all ACCELERATOR=cuda; detener: make stop"
	@echo "Perfil de campana: hasta 12 workers CPU y 1 GPU; RL y demas predictores en CPU. CUDA general solo en pilotos."
	@echo "Un unico pyproject.toml, uv.lock y .venv; ACCELERATOR selecciona el perfil, sin cambiar el entorno."
	@echo "Reinicio explicito: make reset-study DRY_RUN=1 muestra destinos; sin DRY_RUN elimina derivados, incluidos datos procesados compartidos."
	@echo "Todo el recorrido: make run-all ESTUDIO=artifacts/estudio CPU_WORKERS=12"
	@echo "Prueba acotada: make run-all RUN_MODE=check MAX_SECONDS=600 CPU_WORKERS=2"
	@echo "Recorrido: make setup, prepare, tabpfn-weights, tabpfn-pilot, plan, smoke, study, final, benchmark, report, audit, notebooks"
	@echo "study desarrolla busqueda A/B; final congela politicas y evalua el cierre. Datos nuevos: data/processed_evolution."
	@echo "report y audit leen artefactos; notebooks exporta HTML y figuras."
	@echo "dev abre JupyterLab; check comprueba; test prueba; format corrige."
	@echo "gpu-test compara redes en CPU y CUDA con datos sinteticos en el entorno comun."
	@echo "cpu-benchmark calibra 4/6/8/12 procesos con datos sinteticos y un presupuesto de cinco minutos."
	@echo "Excepcion: ESTUDIO=artifacts/otra_replica conserva ejecuciones separadas."
	@echo "CPU_WORKERS=N elige procesos en plan, smoke, study, final y adaptive; omitido usa el JSON o el estudio guardado."
	@echo "STUDY_CONFIG=config/study.json configura prepare, plan y study; ESTUDIO identifica la campaña."

setup:
	uv sync --locked

stop:
	$(UV_RUN) python -m truss_research.evaluacion.detencion

tabpfn-benchmark:
	$(UV_RUN) python -m truss_research.evaluacion.benchmark_tabpfn

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

sensitivity:
	$(UV_RUN) python -m truss_research analizar-sensibilidad $(STUDY_ARG)

adaptive:
	$(UV_RUN) python -m truss_research evaluar-seleccion-adaptativa $(STUDY_ARG) $(CPU_ARG)

learning-curves:
	$(UV_RUN) python -m truss_research evaluar-curvas $(STUDY_ARG)

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
