VENV=.venv
REPORTS=.reports

BENCHMARK=benchmark
SOURCES=src
TESTS=tests
SCRIPTS=scripts



# Installation

.reports:
	mkdir ${REPORTS}

.venv:
	python3 -m venv .venv
	. .venv/bin/activate

.extras:
	pip install -U pip setuptools isort black ruff pytest

install: .venv .reports .extras


# Linters

.isort:
	isort ${SOURCES} ${TESTS}

.black:
	black ${SOURCES} ${TESTS} 

.ruff:
	ruff check --fix ${SOURCES} ${TESTS}

# Tests
.assets:
	mkdir dataset
	test -s dataset/ml-1m.zip  || wget https://files.grouplens.org/datasets/movielens/ml-1m.zip -O dataset/ml-1m.zip
	test -d dataset/ml-1m  || unzip ml-1m.zip -d dataset/ml-1m/

.pytest:
	pytest ${TESTS} --cov=${SOURCES} --cov-report=xml

.lint: .isort .black .ruff
lint: .venv .lint

.test: .assets .pytest
test: .venv .test



# Cleaning

clean:
	rm -rf build dist .eggs *.egg-info
	rm -rf ${VENV}
	rm -rf ${REPORTS}
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	rm -rf dataset/*

reinstall: clean install
