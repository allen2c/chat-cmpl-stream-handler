# Development
fmt:
	@isort chat_cmpl_stream_handler tests
	@black chat_cmpl_stream_handler tests
	@ruff check --fix chat_cmpl_stream_handler tests

check:
	@ruff check chat_cmpl_stream_handler tests
	@pyright

install:
	poetry install --all-extras --all-groups

update:
	poetry update

# Docs
mkdocs:
	mkdocs serve -a 0.0.0.0:8000

# Tests
pytest:
	python -m pytest
