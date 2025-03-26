"""Unit tests for ASI LLM."""

import os
import pytest
from unittest.mock import patch, MagicMock

from llama_index_llms_asi import ASI


def test_init_with_api_key_param():
    """Test initialization with API key parameter."""
    llm = ASI(api_key="test_key")
    assert llm.api_key == "test_key"


@patch.dict(os.environ, {"ASI_API_KEY": "env_test_key"})
def test_init_with_env_var():
    """Test initialization with environment variable."""
    llm = ASI()
    assert llm.api_key == "env_test_key"


def test_init_without_api_key():
    """Test initialization without API key."""
    with pytest.raises(ValueError):
        ASI(api_key=None)


def test_model_selection():
    """Test model selection."""
    # Test default model
    llm = ASI(api_key="test_key")
    assert llm.model == "asi1-mini"
    
    # Test custom model
    llm = ASI(api_key="test_key", model="custom-model")
    assert llm.model == "custom-model"


def test_api_base_selection():
    """Test API base selection."""
    # Test default API base
    llm = ASI(api_key="test_key")
    assert llm.api_base == "https://api.asi1.ai/v1"
    
    # Test custom API base
    llm = ASI(api_key="test_key", api_base="https://custom-api.example.com")
    assert llm.api_base == "https://custom-api.example.com"


def test_class_name():
    """Test class_name method."""
    assert ASI.class_name() == "ASI"