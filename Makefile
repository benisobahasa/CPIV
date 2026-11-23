.PHONY: install test lint demo clean
install:
	pip install -e ".[dev]"
test:
	pytest
lint:
	ruff check src tests
demo:
	cpiv demo --out runs/demo
clean:
	rm -rf runs build dist .pytest_cache .ruff_cache
