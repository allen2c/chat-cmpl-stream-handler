import os


def test_env():
    assert os.getenv("ENVIRONMENT") == "test"
    assert os.getenv("PYTEST_IS_RUNNING") == "true"
