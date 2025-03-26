#!/usr/bin/env python
# Advanced document query example using ASI with LlamaIndex and OpenAI embeddings

import os
import sys
import tempfile

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import Document


def check_api_keys():
    """Check if required API keys are set."""
    asi_api_key = os.environ.get("ASI_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not asi_api_key:
        print("ASI_API_KEY environment variable not set. Please set it before running this example.")
        print("Example: export ASI_API_KEY=your_api_key")
        sys.exit(1)
    
    if not openai_api_key:
        print("OPENAI_API_KEY environment variable not set. Please set it before running this example.")
        print("Example: export OPENAI_API_KEY=your_api_key")
        sys.exit(1)
    
    return asi_api_key, openai_api_key


def create_sample_documents():
    """Create sample documents for indexing."""
    # Sample documents about different programming languages
    documents = [
        Document(
            text="""Python is a high-level, interpreted programming language known for its readability 
            and simplicity. Created by Guido van Rossum and first released in 1991, Python emphasizes 
            code readability with its notable use of significant whitespace. Its language constructs 
            and object-oriented approach aim to help programmers write clear, logical code for small 
            and large-scale projects. Python is dynamically typed and garbage-collected. It supports 
            multiple programming paradigms, including structured, object-oriented, and functional 
            programming. Python is often described as a 'batteries included' language due to its 
            comprehensive standard library. It's widely used in data analysis, machine learning, 
            web development, and automation.""",
            metadata={"language": "Python", "paradigm": "multi-paradigm", "year": 1991}
        ),
        Document(
            text="""JavaScript is a high-level, interpreted programming language that conforms to the 
            ECMAScript specification. JavaScript has curly-bracket syntax, dynamic typing, 
            prototype-based object-orientation, and first-class functions. Alongside HTML and CSS, 
            JavaScript is one of the core technologies of the World Wide Web. JavaScript enables 
            interactive web pages and is an essential part of web applications. The vast majority of 
            websites use it for client-side page behavior, and all major web browsers have a dedicated 
            JavaScript engine to execute it. As a multi-paradigm language, JavaScript supports 
            event-driven, functional, and imperative programming styles. It has APIs for working with 
            text, arrays, dates, regular expressions, and the DOM, but the language itself does not 
            include any I/O, such as networking, storage, or graphics facilities.""",
            metadata={"language": "JavaScript", "paradigm": "multi-paradigm", "year": 1995}
        ),
        Document(
            text="""Java is a class-based, object-oriented programming language that is designed to have 
            as few implementation dependencies as possible. It is a general-purpose programming language 
            intended to let application developers write once, run anywhere (WORA), meaning that compiled 
            Java code can run on all platforms that support Java without the need for recompilation. 
            Java applications are typically compiled to bytecode that can run on any Java virtual machine 
            (JVM) regardless of the underlying computer architecture. The syntax of Java is similar to C 
            and C++, but it has fewer low-level facilities than either of them. Java was originally 
            developed by James Gosling at Sun Microsystems and released in 1995 as a core component of 
            Sun Microsystems' Java platform. The original and reference implementation Java compilers, 
            virtual machines, and class libraries were originally released by Sun under proprietary 
            licenses. As of May 2007, in compliance with the specifications of the Java Community Process, 
            Sun had relicensed most of its Java technologies under the GNU General Public License.""",
            metadata={"language": "Java", "paradigm": "object-oriented", "year": 1995}
        ),
        Document(
            text="""C++ is a general-purpose programming language created by Bjarne Stroustrup as an 
            extension of the C programming language, or 'C with Classes'. The language has expanded 
            significantly over time, and modern C++ now has object-oriented, generic, and functional 
            features in addition to facilities for low-level memory manipulation. It is almost always 
            implemented as a compiled language, and many vendors provide C++ compilers, including the 
            Free Software Foundation, LLVM, Microsoft, Intel, Oracle, and IBM, so it is available on 
            many platforms. C++ was designed with a bias toward system programming and embedded, 
            resource-constrained software and large systems, with performance, efficiency, and 
            flexibility of use as its design highlights. C++ has also been found useful in many other 
            contexts, with key strengths being software infrastructure and resource-constrained 
            applications, including desktop applications, video games, servers, and performance-critical 
            applications.""",
            metadata={"language": "C++", "paradigm": "multi-paradigm", "year": 1985}
        ),
        Document(
            text="""Rust is a multi-paradigm, general-purpose programming language designed for performance 
            and safety, especially safe concurrency. Rust is syntactically similar to C++, but can 
            guarantee memory safety by using a borrow checker to validate references. Rust achieves 
            memory safety without garbage collection, and reference counting is optional. Rust has been 
            called a systems programming language, and in addition to high-level features such as 
            functional programming it also offers mechanisms for low-level memory management. Rust was 
            originally designed by Graydon Hoare at Mozilla Research, with contributions from Dave Herman, 
            Brendan Eich, and others. The designers refined the language while writing the Servo layout 
            or browser engine, and the Rust compiler. The compiler is free and open-source software dual-licensed 
            under the MIT License and Apache License 2.0.""",
            metadata={"language": "Rust", "paradigm": "multi-paradigm", "year": 2010}
        )
    ]
    
    return documents


def main():
    """Run the advanced document query example."""
    # Import necessary modules
    from llama_index_llms_asi import ASI
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.core.settings import Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    
    # Check if required API keys are set
    asi_api_key, openai_api_key = check_api_keys()
    
    print("\n=== Advanced Document Query Example using ASI with LlamaIndex and OpenAI Embeddings ===")
    
    # Initialize the ASI LLM
    llm = ASI(api_key=asi_api_key)
    
    # Initialize OpenAI embedding model
    embed_model = OpenAIEmbedding(api_key=openai_api_key)
    
    # Create sample documents
    print("\nCreating sample documents about programming languages...")
    documents = create_sample_documents()
    print(f"Created {len(documents)} documents")
    
    # Configure LlamaIndex to use our ASI LLM and OpenAI embeddings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    
    # Create a parser for splitting the documents into nodes
    parser = SentenceSplitter(chunk_size=Settings.chunk_size)
    nodes = parser.get_nodes_from_documents(documents)
    
    print(f"Documents split into {len(nodes)} nodes")
    
    # Create a vector store index from the nodes
    print("\nCreating a vector store index...")
    index = VectorStoreIndex(nodes)
    
    # Create a retriever with customized parameters
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3,  # Retrieve top 3 most similar nodes
    )
    
    # Create a query engine with the retriever and a similarity postprocessor
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)  # Only use nodes with similarity > 0.7
        ]
    )
    
    # Example queries
    queries = [
        "What are the key features of Python?",
        "Compare JavaScript and Java.",
        "Which programming language is best for systems programming?",
        "What programming language was developed at Mozilla Research?",
        "Which programming languages support multiple programming paradigms?",
    ]
    
    # Run queries
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        response = query_engine.query(query)
        print(f"Response: {response}")
        print(f"Source nodes: {len(response.source_nodes)}")
        
        # Print metadata from source nodes
        print("Sources:")
        for j, node in enumerate(response.source_nodes, 1):
            print(f"  {j}. {node.metadata.get('language', 'Unknown')} "
                  f"(Relevance score: {node.score:.4f})")


if __name__ == "__main__":
    main()
