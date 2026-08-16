.PHONY: install dev test lint demo clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check virelion_cardioscore tests
	ruff format --check virelion_cardioscore tests

demo:
	cardioscore demo --output-dir ./outputs

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov outputs
	find . -type d -name __pycache__ -exec rm -rf {} +
