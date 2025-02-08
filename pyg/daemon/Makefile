.PHONY: format lint all lock

.DEFAULT_GOAL = help
HAS_CUDA := $(shell command -v nvcc 2> /dev/null && echo 1 || echo 0)

all: format lint lock # execute all target

format: # format python file by ruff
	ruff check --fix .
	ruff format .

lint: # lint python file by flake8 and mypy
	ruff check .

lock: pyproject.toml # lock uv
	uv lock

install: uv.lock # install package by uv
	@if [ $(HAS_CUDA) -eq 1 ]; then \
		echo "=== Install with GPU support ==="; \
		uv sync --all-groups --extra=gpu; \
	else \
		echo "=== Install with CPU support ==="; \
		uv sync --all-groups --extra=cpu; \
	fi

test: # run test
	pytest -v -s tests/

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done
