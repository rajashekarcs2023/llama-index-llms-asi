.PHONY: format lint test

format:
    black llama_index tests
    isort llama_index tests

lint:
    black --check llama_index tests
    isort --check llama_index tests
    flake8 llama_index tests

test:
    pytest tests/test_llms_asi.py -v

integration-test:
    pytest tests/test_integration_asi.py -v