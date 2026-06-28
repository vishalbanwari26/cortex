.PHONY: install dev test demo live clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

demo:
	python -m cortex.cli "Put the red mug in the cupboard."

live:
	python -m cortex.cli "Put the red mug in the cupboard." --live --image examples/scene.jpg

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
