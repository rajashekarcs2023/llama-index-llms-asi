"""Document query example using ASI with LlamaIndex."""

import os
import sys
import tempfile

from llama_index.core.llms import ChatMessage, MessageRole


def check_api_key():
    """Check if ASI API key is set."""
    api_key = os.environ.get("ASI_API_KEY")
    if not api_key:
        print("ASI_API_KEY environment variable not set. Please set it before running this example.")
        print("Example: export ASI_API_KEY=your_api_key")
        sys.exit(1)
    return api_key


def create_sample_document():
    """Create a sample document for indexing."""
    # Import necessary modules
    from llama_index.core.schema import Document
    
    # Sample text about artificial intelligence
    text = """
    Artificial Intelligence (AI) is a field of computer science focused on creating systems 
    capable of performing tasks that typically require human intelligence. These tasks include 
    speech recognition, decision-making, visual perception, and language translation.
    
    Machine Learning (ML) is a subset of AI that enables systems to learn and improve from 
    experience without being explicitly programmed. Deep Learning, a further subset of ML, 
    uses neural networks with many layers (hence "deep") to analyze various factors of data.
    
    Natural Language Processing (NLP) is another branch of AI that focuses on the interaction 
    between computers and human language. It enables computers to understand, interpret, and 
    generate human language in a valuable way.
    
    Computer Vision is an interdisciplinary field that deals with how computers can gain 
    high-level understanding from digital images or videos. It seeks to automate tasks that 
    the human visual system can do.
    
    Reinforcement Learning is an area of ML concerned with how software agents ought to take 
    actions in an environment so as to maximize some notion of cumulative reward.
    
    AI Ethics is a part of the ethics of technology specific to robots and other artificially 
    intelligent beings. It is concerned with the moral behavior of humans as they design, 
    construct, use and treat artificially intelligent beings.
    
    The future of AI includes potential advancements in Artificial General Intelligence (AGI), 
    which would be AI systems with generalized human cognitive abilities. When presented with 
    an unfamiliar task, an AGI system could find a solution without human intervention.
    """
    
    # Create a Document object
    document = Document(text=text)
    return document


def main():
    """Run the document query example."""
    # Import necessary modules
    from llama_index_llms_asi import ASI
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.settings import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    # Check if ASI API key is set
    api_key = check_api_key()
    
    print("\n=== Document Query Example using ASI with LlamaIndex ===")
    
    # Initialize the ASI LLM
    llm = ASI(api_key=api_key)
    
    # Initialize a local embedding model (no API key required)
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Create a sample document
    print("\nCreating a sample document about Artificial Intelligence...")
    document = create_sample_document()
    
    # Configure LlamaIndex to use our ASI LLM
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    
    # Create a parser for splitting the document into nodes
    parser = SentenceSplitter(chunk_size=Settings.chunk_size)
    nodes = parser.get_nodes_from_documents([document])
    
    print(f"Document split into {len(nodes)} nodes")
    
    # Create a vector store index from the nodes
    print("\nCreating a vector store index...")
    index = VectorStoreIndex(nodes)
    
    # Create a query engine
    query_engine = index.as_query_engine()
    
    # Example queries
    queries = [
        "What is artificial intelligence?",
        "Explain the difference between machine learning and deep learning.",
        "What is natural language processing?",
        "What is the future of AI?",
        "What is AI ethics?",
    ]
    
    # Run queries
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        response = query_engine.query(query)
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
