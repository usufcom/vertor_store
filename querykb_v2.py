"""
RAG Knowledge Base Query Interface - Native Implementation

This module provides a query interface for the native vector store created by
VectorStore_v2.py. It implements cosine similarity search and AI-powered
question answering without external dependencies like Langchain.

Features:
- Native vector store loading from JSON files
- Cosine similarity search for document retrieval
- OpenAI integration for embeddings and response generation
- Context-aware question answering
- Comprehensive error handling and logging

Author: Usuf Com
Contact: usufcom20@gmail.com
Website: www.djamai.com
LinkedIn: https://www.linkedin.com/in/usufcom
Date: 2024
"""

import os
import json
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI


class RAG:
    """
    Native RAG system for querying vector stores with cosine similarity.
    
    This class provides a complete query interface for native vector stores,
    implementing cosine similarity search and AI-powered question answering
    without external dependencies.
    
    Attributes:
        api_key (str): OpenAI API key for embeddings and chat completions
        client (OpenAI): OpenAI client for API calls
        vector_store (list): Loaded vector store with embeddings and metadata
    """
    
    def __init__(self, vector_store_path=None):
        """
        Initialize the native RAG query system.
        
        Args:
            vector_store_path (str): Path to the JSON file containing the vector store
        
        Note:
            The vector store should have been previously created using VectorStore_v2.py.
            If vector_store_path is provided, the vector store will be loaded automatically.
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Get OpenAI API key from environment variables
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize OpenAI client for API calls
        self.client = OpenAI(api_key=self.api_key)
        
        # Vector store data (loaded from JSON file)
        self.vector_store = None
        
        # Load vector store if path is provided
        if vector_store_path:
            self.vector_store = self.load_vector_store_local(vector_store_path)

    def load_vector_store_local(self, vector_store_path):
        """
        Load stored embeddings and metadata from a JSON file.
        
        This method loads a previously created vector store from a JSON file
        containing embeddings, text chunks, and metadata.
        
        Args:
            vector_store_path (str): Path to the JSON file containing the vector store
        
        Returns:
            list: Vector store data with embeddings, text, and metadata
        
        Note:
            The JSON file should contain a list of dictionaries, each with
            'text', 'embedding', and 'metadata' keys.
        """
        # Load vector store from JSON file
        with open(vector_store_path, 'r', encoding="utf-8") as f:
            vector_store = json.load(f)
        return vector_store

    @staticmethod
    def cosine_similarity(a, b):
        """
        Compute cosine similarity between two vectors.
        
        Cosine similarity measures the cosine of the angle between two vectors,
        providing a value between -1 and 1, where 1 indicates identical vectors.
        
        Args:
            a (list or np.array): First vector
            b (list or np.array): Second vector
        
        Returns:
            float: Cosine similarity score between 0 and 1
        
        Note:
            This is a standard similarity metric for comparing embedding vectors.
            Higher values indicate more similar content.
        """
        # Convert inputs to numpy arrays for efficient computation
        a, b = np.array(a), np.array(b)
        
        # Compute cosine similarity: dot product / (norm_a * norm_b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self, query, top_k=3):
        """
        Perform similarity search to find the most relevant documents.
        
        This method:
        1. Creates an embedding for the query text
        2. Computes cosine similarity with all stored embeddings
        3. Returns the top-k most similar documents
        
        Args:
            query (str): The search query
            top_k (int): Number of most similar documents to return (default: 3)
        
        Returns:
            list: List of tuples (similarity_score, text, metadata) sorted by similarity
        
        Raises:
            ValueError: If vector store is not loaded
        
        Note:
            Higher top_k values provide more context but may include less relevant
            documents. The similarity scores can be used to filter results.
        """
        # Check if vector store is loaded
        if not self.vector_store:
            raise ValueError("❌ Vectorstore not loaded.")

        # Create embedding for the query using OpenAI API
        query_emb = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        ).data[0].embedding

        # Compute similarity scores for all documents
        similarities = []
        for item in self.vector_store:
            # Calculate cosine similarity between query and document embeddings
            score = self.cosine_similarity(query_emb, item["embedding"])
            
            # Store similarity score, text, and metadata
            similarities.append((
                score, 
                item["text"], 
                item.get("metadata", {})
            ))

        # Sort by similarity score (highest first) and return top-k results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]

    def askAI(self, query, system_prompt, k=5):
        """
        Ask a question and get an AI-generated response based on the knowledge base.
        
        This method implements the complete RAG pipeline:
        1. Retrieve the most relevant document chunks using cosine similarity
        2. Combine retrieved chunks into context
        3. Send context and query to OpenAI for response generation
        
        Args:
            query (str): The question to ask
            system_prompt (str): System prompt to guide the AI's response style and behavior
            k (int): Number of most similar chunks to retrieve (default: 5)
        
        Returns:
            tuple: (AI_response, retrieved_context)
        
        Raises:
            ValueError: If vector store is not loaded
        
        Note:
            The system_prompt should provide instructions on how the AI should respond
            based on the retrieved context. The retrieved_context can be used for
            debugging or transparency purposes. Higher k values provide more context
            but may lead to less focused responses.
        """
        # Check if vector store is loaded
        if not self.vector_store:
            raise ValueError("❌ Vectorstore not loaded. Call load_vector_store_local() first.")

        # Step 1: Retrieve the most relevant document chunks
        retrieved_docs = self.search(query, top_k=k)
        
        # Step 2: Combine retrieved chunks into context
        context = "\n\n".join([doc[1] for doc in retrieved_docs])

        # Step 3: Build the complete prompt with system instructions and context
        prompt = (
            f"{system_prompt}\n\n"
            f"Here are the documents (knowledgebase) you need:\n\n{context}\n\n"
        )

        # Step 4: Generate response using OpenAI's chat completion API
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4o-mini for optimal performance
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        
        # Return the AI response and the context used for transparency
        return response.choices[0].message.content.strip(), context
