VENV=.venv
SOURCES=src
TESTS=tests


# Installation

.venv:
	python3 -m venv .venv
	. .venv/bin/activate
.base:
	. .venv/bin/activate && pip install -U pip setuptools build wheel
.main:
	. .venv/bin/activate && pip install torch transformers
	. .venv/bin/activate && pip install -r requirements.txt

.extras:
	. .venv/bin/activate && pip install -U isort black ruff pytest pytest-cov

install: .venv  .base .main .extras


# Linters

.isort:
	. .venv/bin/activate && isort ${SOURCES} ${TESTS}

.black:
	. .venv/bin/activate && black ${SOURCES} ${TESTS} 

.ruff:
	. .venv/bin/activate && ruff check --fix ${SOURCES} ${TESTS}

.assets:
	test -d dataset || mkdir dataset
	test -s dataset/ml-1m.zip  || wget --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-1m.zip -O dataset/ml-1m.zip
	test -d dataset/ml-1m  || unzip dataset/ml-1m.zip -d dataset/

.pytest:
	. .venv/bin/activate && pytest ${TESTS} --cov=${SOURCES} --cov-report=xml

.lint: .isort .black .ruff
lint: .venv .lint

.test: .assets .extras .pytest
test: .test

build: 
	rm -f dist/*
	. .venv/bin/activate && python -m build .
	. .venv/bin/activate && twine upload dist/mamba4rec*

# Cleaning

clean:
	rm -rf build dist .eggs *.egg-info
	rm -rf ${VENV}
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	rm -rf dataset/*

reinstall: clean install
