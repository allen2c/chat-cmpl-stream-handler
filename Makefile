# Development
format:
	@isort . \
		--skip setup.py \
		--skip .venv \
		--skip build \
		--skip dist \
		--skip __pycache__ \
		--skip docs \
		--skip static \
		--skip .conda
	@black . \
		--exclude setup.py \
		--exclude .venv \
		--exclude build \
		--exclude dist \
		--exclude __pycache__ \
		--exclude docs \
		--exclude static \
		--exclude .conda

install:
	poetry install --all-extras --all-groups

update:
	poetry update
	poetry export --without-hashes -f requirements.txt --output requirements.txt
	poetry export --without-hashes -f requirements.txt --output requirements-all.txt --all-extras --all-groups

# Docs
mkdocs:
	mkdocs serve -a 0.0.0.0:8000

# Tests
pytest:
	python -m pytest
