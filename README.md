# llama-index-llms-asi

ASI LLM integration for LlamaIndex.

## Overview

This package provides integration between LlamaIndex and ASI language models from the Artificial Superintelligence Alliance. It allows you to use ASI models with LlamaIndex for various natural language processing tasks.

ASI-1-Mini is developed by Fetch.ai Inc., a founding member of the Artificial Superintelligence Alliance, and is optimized for supporting complex agentic workflows.

## Installation

```bash
pip install llama-index-llms-asi
```

## Usage

```python
from llama_index_llms_asi import ASI
from llama_index.core.llms import ChatMessage, MessageRole

# Initialize the ASI LLM
llm = ASI(
    model="asi1-mini",  # Default model
    api_key="your-api-key",  # Or set ASI_API_KEY environment variable
    temperature=0.7,
    max_tokens=100
)

# Use the LLM for completion
response = llm.complete("Hello, world!")
print(response.text)

# Use the LLM for chat
messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful AI assistant."),
    ChatMessage(role=MessageRole.USER, content="What is LlamaIndex?")
]
response = llm.chat(messages)
print(response.message.content)

# Use the LLM for streaming completion
for response in llm.stream_complete("Hello, world!"):
    print(response.delta, end="")
```

## Examples

Check out the examples directory for more detailed usage examples:

- `simple_query_example.py`: Demonstrates basic usage of the ASI LLM for simple queries and chat.
- `document_query_example.py`: Shows how to use the ASI LLM with LlamaIndex for document indexing and querying.
- `advanced_document_query_example.py`: Demonstrates more complex document querying with metadata filtering and source attribution.

## Features

- **Completion**: Generate text completions with ASI models.
- **Chat**: Have multi-turn conversations with ASI models.
- **Streaming**: Stream responses for both completion and chat.
- **Integration with LlamaIndex**: Use ASI models with LlamaIndex for document indexing and querying.

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|--------|
| `model` | The ASI model to use | `"asi1-mini"` |
| `api_key` | Your ASI API key | `None` (uses `ASI_API_KEY` env var) |
| `api_base` | The base URL for the ASI API | `"https://api.asi1.ai/v1"` |
| `temperature` | Controls randomness (0-1) | `0.7` |
| `max_tokens` | Maximum number of tokens to generate | `None` |
| `top_p` | Nucleus sampling parameter | `1.0` |

## Requirements

- Python 3.8+
- `llama-index-llms-openai-like`

## License

MIT