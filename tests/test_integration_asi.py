"""Integration tests for ASI LLM."""

import os
import pytest

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index_llms_asi import ASI


@pytest.fixture
def api_key():
    """Get API key from environment variable."""
    api_key = os.environ.get("ASI_API_KEY")
    if not api_key:
        pytest.skip("ASI_API_KEY environment variable not set")
    return api_key


def test_completion(api_key):
    """Test completion."""
    llm = ASI(api_key=api_key)
    response = llm.complete("Hello, world!")
    assert response.text.strip() != ""
    print(f"Completion response: {response.text}")


def test_chat(api_key):
    """Test chat."""
    llm = ASI(api_key=api_key)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful AI assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
    ]
    response = llm.chat(messages)
    assert response.message.content.strip() != ""
    print(f"Chat response: {response.message.content}")


def test_streaming_completion(api_key):
    """Test streaming completion."""
    llm = ASI(api_key=api_key)
    response_gen = llm.stream_complete("Hello, world!")
    responses = list(response_gen)
    assert len(responses) > 0
    assert any(r.delta.strip() != "" for r in responses)
    print(f"Streaming completion response: {''.join(r.delta for r in responses)}")


def test_streaming_chat(api_key):
    """Test streaming chat."""
    llm = ASI(api_key=api_key)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful AI assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
    ]
    response_gen = llm.stream_chat(messages)
    responses = list(response_gen)
    assert len(responses) > 0
    assert any(r.delta.strip() != "" for r in responses)
    print(f"Streaming chat response: {''.join(r.delta for r in responses)}")


if __name__ == "__main__":
    # Run the tests manually if API key is available
    if os.environ.get("ASI_API_KEY"):
        test_completion(os.environ.get("ASI_API_KEY"))
        test_chat(os.environ.get("ASI_API_KEY"))
        test_streaming_completion(os.environ.get("ASI_API_KEY"))
        test_streaming_chat(os.environ.get("ASI_API_KEY"))
    else:
        print("ASI_API_KEY environment variable not set. Skipping tests.")