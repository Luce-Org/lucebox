# Makefile — single entry point for the common dev/CI ops on lucebox-hub.
#
# Most targets shell out to `uv` and `docker buildx bake`. Pre-release
# software: targets favor simplicity over portability (assumes bash + GNU
# coreutils + a working docker buildx + uv on PATH).
#
# Quick start:
#   make help        # what's available
#   make test        # 85+ tests across luce-bench / lucebox
#   make lint        # ruff check + format check
#   make build       # docker buildx bake cuda12-local --load
#   make serve       # docker run the local image, gemma-4-26b
#   make smoke URL=http://localhost:8080  # 3-prompt sanity check
#   make bench AREAS=all                  # full capability sweep
#   make clean       # drop containers + dangling images

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ── Build args ──────────────────────────────────────────────────────────
# Narrow the CUDA arch list to your local GPU to cut build time 5-6×:
#   make build DFLASH_CUDA_ARCHES=120
DFLASH_CUDA_ARCHES ?= 75;80;86;89;90;120

# Where to mount models into the container. Default matches what
# `lucebox configure` would pick.
MODELS_DIR ?= $(HOME)/models

# Server URL for smoke / bench / profile targets.
URL ?= http://localhost:8080

# Which areas the bench target runs. Defaults to the fast smoke area.
# Override: `make bench AREAS=all` or `make bench AREAS=ds4-eval,forge`.
AREAS ?= smoke

# Model id passed to the bench. `default` triggers /v1/models autoresolve.
MODEL ?= default

# Image name (local tag the buildx bake produces).
IMAGE ?= lucebox-hub:cuda12

# ── Targets ─────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help message.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: sync
sync:  ## uv sync the workspace (incl. dev extras).
	uv sync --extra dev

.PHONY: test
test: sync  ## Run all Python tests (lucebox + luce-bench).
	uv run pytest lucebox/tests luce-bench/tests -q

.PHONY: lint
lint: sync  ## Ruff check + format-check (no auto-fix).
	uv run ruff check
	uv run ruff format --check

.PHONY: fix
fix: sync  ## Ruff auto-fix + format.
	uv run ruff check --fix
	uv run ruff format

.PHONY: build
build:  ## Build lucebox-hub:cuda12 locally via docker buildx bake.
	DFLASH_CUDA_ARCHES="$(DFLASH_CUDA_ARCHES)" docker buildx bake cuda12-local --load

.PHONY: serve
serve:  ## Run the local image, foreground. Models bind-mounted from $(MODELS_DIR).
	docker run --rm --gpus all -p 8080:8080 \
		-v $(MODELS_DIR):/opt/lucebox-hub/server/models:ro \
		--name lucebox-gemma \
		$(IMAGE) serve

.PHONY: stop
stop:  ## Stop a running lucebox-gemma container.
	-docker stop lucebox-gemma
	-docker rm lucebox-gemma

.PHONY: shell
shell:  ## Drop into a bash shell inside the image (debug).
	docker run --rm -it --gpus all $(IMAGE) shell

.PHONY: smoke
smoke: sync  ## Run the luce-bench smoke area against URL (default localhost:8080).
	uv run python -m lucebench --base-url $(URL) --model $(MODEL) --areas smoke

.PHONY: bench
bench: sync  ## Run the bench AREAS against URL. Default: smoke. Try AREAS=all.
	uv run python -m lucebench --base-url $(URL) --model $(MODEL) --areas $(AREAS)

.PHONY: profile
profile: sync  ## Capture a versioned snapshot via lucebox profile.
	uv run python -m lucebox profile --url $(URL)

.PHONY: ci-trigger-build
ci-trigger-build:  ## Trigger GH Actions docker build+push for the current branch.
	gh workflow run docker.yml --ref "$$(git branch --show-current)" -f push=true
	@echo "view: gh run watch"

.PHONY: clean
clean:  ## Drop stopped containers, dangling images, build cache (~10 GB+).
	-docker container prune -f
	-docker image prune -f
	-docker buildx prune -f --filter "until=24h"

.PHONY: clean-models
clean-models:  ## Remove downloaded models from $(MODELS_DIR). Destructive.
	@echo "WARN: about to rm -rf $(MODELS_DIR)/*"
	@read -p "Continue? [y/N] " ans && [ "$$ans" = "y" ] && rm -rf $(MODELS_DIR)/*
