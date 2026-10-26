# Raccourcis du projet. `make` ou `make help` affiche la liste des cibles.
# Les cibles s'exécutent depuis la racine du projet.

VENV ?= .venv
PY := $(VENV)/bin/python
DATA_DIR ?= data/e5-base
MODEL ?= intfloat/multilingual-e5-base
LABEL ?= local
EVAL_OPTS ?=
INDEX_FILE := $(DATA_DIR)/faiss_index/index.faiss

# Lues par ingest.py, graph.py, app.py et les tests (prioritaires sur .env).
export RETRIEVER_DATA_DIR := $(DATA_DIR)
export RETRIEVER_MODEL := $(MODEL)

.DEFAULT_GOAL := help
.PHONY: help install index index-force ask app test eval regression baseline check clean clean-cache

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-12s %s\n", $$1, $$2}'

install: ## Crée .venv, installe torch CPU puis requirements.txt
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt

# Cible fichier : ne reconstruit pas un index qui existe déjà.
index: $(INDEX_FILE) ## Construit l'index FAISS s'il manque
	@echo "Index disponible : $(INDEX_FILE)"

$(INDEX_FILE):
	$(PY) ingest.py --model $(MODEL) --out $(DATA_DIR)

index-force: ## Reconstruit l'index FAISS même s'il existe (≈ 15 min sur CPU)
	$(PY) ingest.py --model $(MODEL) --out $(DATA_DIR)

ask: ## Pose une question à l'agent : make ask Q="..."
	@test -n "$(Q)" || { echo 'Usage : make ask Q="votre question"' >&2; exit 1; }
	$(PY) graph.py "$(Q)"

app: ## Lance l'interface Streamlit, en local uniquement (localhost)
	$(VENV)/bin/streamlit run app.py --server.address localhost

test: ## Tests unitaires, hors-ligne
	$(PY) -m pytest -m "not regression"

eval: ## Évaluation : make eval LABEL=essai EVAL_OPTS="--judge --ragas"
	$(PY) eval/run_eval.py --label $(LABEL) $(EVAL_OPTS)

regression: ## Non-régression : rejoue l'évaluation si besoin, compare au baseline
	$(PY) -m pytest -m regression --run-eval

baseline: ## Accepte le dernier run comme nouvelle référence
	@echo "Rappel : à ne faire que si la baisse est voulue et validée."
	$(PY) eval/baseline.py

check: test regression ## Tests puis non-régression : à lancer avant chaque push
	@echo "Tests et non-régression OK."

clean: ## Supprime __pycache__ et .pytest_cache (garde data/, .cache/, eval/runs/, .venv)
	rm -rf .pytest_cache
	find . -path "./$(VENV)" -prune -o -name __pycache__ -type d -exec rm -rf {} +

clean-cache: ## Supprime .cache/llm.sqlite (attention : rejouer coûtera des appels)
	@echo "Attention : sans ce cache, les prochains appels à DeepSeek seront facturés."
	rm -f .cache/llm.sqlite
