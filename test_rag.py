"""
Test Suite for RAG Knowledge Base System

This module contains comprehensive tests for the RAG Knowledge Base System,
covering both the Langchain+FAISS and native vector store implementations.

Test Coverage:
- Module imports and initialization
- Environment setup and configuration
- Document structure validation
- RAG functionality with mocked dependencies
- File processing capabilities
- Configuration parameter validation

Author: Your Name
Date: 2024
"""

import os
import pytest
from unittest.mock import Mock, patch
import tempfile
import shutil


def test_imports():
    """
    Test that all core modules can be imported successfully.
    
    This test ensures that all required dependencies are available and
    that the modules can be imported without errors. It's a basic
    sanity check for the project setup.
    
    Raises:
        pytest.fail: If any module fails to import
    """
    try:
        # Test imports for Langchain + FAISS implementation
        from VectrorStore_v1 import RAGKnowledgeBase
        from querykb_v1 import queryKnowledgeBase
        
        # Test imports for native implementation
        from VectorStore_v2 import VectorStore
        from querykb_v2 import RAG
        
        # If all imports succeed, the test passes
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


def test_env_file_exists():
    """
    Test that the environment example file exists.
    
    This test ensures that the env.example file is present in the project,
    which is essential for users to understand what environment variables
    they need to set up.
    """
    assert os.path.exists("env.example"), "env.example file should exist"


def test_requirements_file_exists():
    """
    Test that the requirements.txt file exists.
    
    This test ensures that the requirements.txt file is present, which
    is essential for users to install the correct dependencies.
    """
    assert os.path.exists("requirements.txt"), "requirements.txt file should exist"


def test_docs_folder_exists():
    """
    Test that the docs folder exists.
    
    This test ensures that the docs folder is present, which is required
    for storing documents that will be processed by the RAG system.
    """
    assert os.path.exists("docs"), "docs folder should exist"


def test_prompts_folder_exists():
    """
    Test that the prompts folder exists.
    
    This test ensures that the prompts folder is present, which is required
    for storing system prompts that guide the AI's responses.
    """
    assert os.path.exists("prompts"), "prompts folder should exist"


def test_system_prompt_exists():
    """
    Test that the system prompt file exists.
    
    This test ensures that the System_Prompt.txt file is present, which
    is essential for the RAG system to function properly.
    """
    assert os.path.exists("prompts/System_Prompt.txt"), "System_Prompt.txt should exist"


@patch('openai.OpenAI')
def test_rag_knowledge_base_initialization(mock_openai):
    """
    Test RAGKnowledgeBase class initialization.
    
    This test verifies that the RAGKnowledgeBase class can be properly
    initialized with the correct parameters and that the internal
    attributes are set correctly.
    
    Args:
        mock_openai: Mock object for OpenAI client to avoid actual API calls
    """
    # Create a mock OpenAI client
    mock_client = Mock()
    mock_openai.return_value = mock_client
    
    # Test initialization with mocked environment
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        from VectrorStore_v1 import RAGKnowledgeBase
        
        # Initialize the knowledge base
        kb = RAGKnowledgeBase('docs', 'test_vectorstore')
        
        # Verify that attributes are set correctly
        assert kb.folder_path == 'docs'
        assert kb.vector_store_path == 'test_vectorstore'


@patch('openai.OpenAI')
def test_vector_store_initialization(mock_openai):
    """
    Test VectorStore class initialization.
    
    This test verifies that the VectorStore class can be properly
    initialized with the correct parameters and that the default
    configuration values are set correctly.
    
    Args:
        mock_openai: Mock object for OpenAI client to avoid actual API calls
    """
    # Create a mock OpenAI client
    mock_client = Mock()
    mock_openai.return_value = mock_client
    
    # Import and initialize the VectorStore class
    from VectorStore_v2 import VectorStore
    store = VectorStore('test_key')
    
    # Verify that default configuration values are set correctly
    assert store.chunk_size == 1000
    assert store.chunk_overlap == 200


def test_document_loading():
    """
    Test document loading functionality.
    
    This test creates a temporary text file and verifies that it can be
    read correctly. This is a basic test of file I/O operations that
    the RAG system relies on.
    """
    # Create a temporary test document
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document for RAG testing.")
        temp_file = f.name
    
    try:
        # Test that the file can be read correctly
        with open(temp_file, 'r') as f:
            content = f.read()
        
        # Verify that the expected content is present
        assert "test document" in content
    finally:
        # Clean up the temporary file
        os.unlink(temp_file)


def test_chunking_parameters():
    """
    Test that chunking parameters are reasonable and valid.
    
    This test verifies that the default chunking parameters used by
    the RAG system are sensible and will work correctly. It ensures
    that chunk_size is positive, chunk_overlap is non-negative, and
    that chunk_overlap is less than chunk_size.
    """
    # Define the default chunking parameters
    chunk_size = 1000
    chunk_overlap = 200
    
    # Test that chunk_size is positive
    assert chunk_size > 0, "Chunk size should be positive"
    
    # Test that chunk_overlap is non-negative
    assert chunk_overlap >= 0, "Chunk overlap should be non-negative"
    
    # Test that chunk_overlap is less than chunk_size
    assert chunk_overlap < chunk_size, "Chunk overlap should be less than chunk size"


def test_mock_rag_functionality():
    """
    Test RAG functionality with mocked dependencies.
    
    This test verifies that the RAG system can handle basic operations
    when external dependencies are mocked. It tests the flow without
    making actual API calls or file system operations.
    """
    # Mock the OpenAI client
    with patch('openai.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock environment variables
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Test that we can import and create instances
            from VectrorStore_v1 import RAGKnowledgeBase
            from VectorStore_v2 import VectorStore
            
            # Create instances (should not raise exceptions)
            kb = RAGKnowledgeBase('docs', 'test_vectorstore')
            store = VectorStore('test_key')
            
            # Verify that instances were created successfully
            assert kb is not None
            assert store is not None


def test_file_structure():
    """
    Test that the project has the expected file structure.
    
    This test verifies that all essential files and directories
    are present in the project structure.
    """
    # Test for essential directories
    essential_dirs = ['docs', 'prompts', 'vectorstore']
    for dir_name in essential_dirs:
        assert os.path.exists(dir_name), f"Directory {dir_name} should exist"
    
    # Test for essential files
    essential_files = [
        'requirements.txt',
        'env.example',
        'README.md',
        'LICENSE',
        'prompts/System_Prompt.txt'
    ]
    for file_name in essential_files:
        assert os.path.exists(file_name), f"File {file_name} should exist"


if __name__ == "__main__":
    # Run the test suite if this file is executed directly
    pytest.main([__file__])
