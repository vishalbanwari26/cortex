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

groq:
	python -m cortex.cli "Put the red mug in the cupboard." --provider groq \
		--note "A kitchen table holds a red mug and a dirty plate; a cupboard is on the wall."
