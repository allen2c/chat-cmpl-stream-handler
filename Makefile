# Development
format:
	@isort chat_cmpl_stream_handler tests
	@black chat_cmpl_stream_handler tests

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
