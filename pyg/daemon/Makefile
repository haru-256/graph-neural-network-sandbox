.PHONY: format lint all lock

.DEFAULT_GOAL = help

all: format lint lock # execute all target

format: # format python file by ruff
	ruff check --fix .
	ruff format .

lint: # lint python file by flake8 and mypy
	ruff check .

lock: pyproject.toml # lock uv
	uv lock

install: uv.lock # install package by uv
	uv sync --all-groups

test: # run test
	pytest -v -s tests/

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done
