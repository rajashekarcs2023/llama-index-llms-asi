#!/usr/bin/env python
# Example script demonstrating how to use the ASI LLM with LlamaIndex

import os
import sys

from llama_index.core.llms import ChatMessage, MessageRole


def check_api_key():
    """Check if ASI API key is set."""
    api_key = os.environ.get("ASI_API_KEY")
    if not api_key:
        print("ASI_API_KEY environment variable not set. Please set it before running this example.")
        print("Example: export ASI_API_KEY=your_api_key")
        sys.exit(1)
    return api_key


def completion_example(api_key):
    """Example of using ASI for completion."""
    from llama_index_llms_asi import ASI
    
    print("\n=== Completion Example ===")
    
    # Initialize the ASI LLM
    llm = ASI(api_key=api_key)
    
    # Define a prompt
    prompt = "What are the three laws of robotics?"
    print(f"Prompt: {prompt}")
    
    # Get a completion
    response = llm.complete(prompt)
    
    # Print the response
    print(f"Response: {response.text}")


def streaming_completion_example(api_key):
    """Example of using ASI for streaming completion."""
    from llama_index_llms_asi import ASI
    
    print("\n=== Streaming Completion Example ===")
    
    # Initialize the ASI LLM
    llm = ASI(api_key=api_key)
    
    # Define a prompt
    prompt = "List 5 interesting facts about space exploration."
    print(f"Prompt: {prompt}")
    
    # Get a streaming completion
    print("Response: ", end="")
    response_gen = llm.stream_complete(prompt)
    
    # Print the response as it comes in
    for response in response_gen:
        print(response.delta, end="", flush=True)
    print()  # Add a newline at the end


def chat_example(api_key):
    """Example of using ASI for chat."""
    from llama_index_llms_asi import ASI
    
    print("\n=== Chat Example ===")
    
    # Initialize the ASI LLM
    llm = ASI(api_key=api_key)
    
    # Define messages using ChatMessage objects
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful AI assistant that specializes in astronomy."),
        ChatMessage(role=MessageRole.USER, content="What is a black hole and how is it formed?"),
    ]
    print(f"System: {messages[0].content}")
    print(f"User: {messages[1].content}")
    
    # Get a chat response
    response = llm.chat(messages)
    
    # Print the response
    print(f"Assistant: {response.message.content}")


def streaming_chat_example(api_key):
    """Example of using ASI for streaming chat."""
    from llama_index_llms_asi import ASI
    
    print("\n=== Streaming Chat Example ===")
    
    # Initialize the ASI LLM
    llm = ASI(api_key=api_key)
    
    # Define messages using ChatMessage objects
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful AI assistant that specializes in history."),
        ChatMessage(role=MessageRole.USER, content="Tell me about the Renaissance period in Europe."),
    ]
    print(f"System: {messages[0].content}")
    print(f"User: {messages[1].content}")
    
    # Get a streaming chat response
    print("Assistant: ", end="")
    response_gen = llm.stream_chat(messages)
    
    # Print the response as it comes in
    for response in response_gen:
        print(response.delta, end="", flush=True)
    print()  # Add a newline at the end


def main():
    """Run all examples."""
    # Check if ASI API key is set
    api_key = check_api_key()
    
    # Run examples
    completion_example(api_key)
    streaming_completion_example(api_key)
    chat_example(api_key)
    streaming_chat_example(api_key)


if __name__ == "__main__":
    main()
