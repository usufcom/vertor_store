"""
RAG Knowledge Base Query Interface - FAISS Implementation

This module provides a query interface for the FAISS-based vector store created by
VectrorStore_v1.py. It handles loading the vector store and performing similarity
search to retrieve relevant context for AI-powered question answering.

Features:
- FAISS vector store loading and management
- Similarity search for document retrieval
- OpenAI integration for response generation
- Context-aware question answering
- Comprehensive error handling

Author: Your Name
Date: 2024
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


class queryKnowledgeBase:
    """
    Query interface for FAISS-based knowledge base systems.
    
    This class provides methods to load a pre-built FAISS vector store and
    perform similarity search to retrieve relevant context for AI-powered
    question answering.
    
    Attributes:
        api_key (str): OpenAI API key for embeddings and chat completions
        embeddings (OpenAIEmbeddings): Embeddings model for vector operations
        client (OpenAI): OpenAI client for chat completions
        vector_store_path (str): Path to the FAISS vector store
        db (FAISS): FAISS vector store instance
    """
    
    def __init__(self, vector_store_path='faiss_vectorstore'):
        """
        Initialize the query interface for FAISS knowledge base.
        
        Args:
            vector_store_path (str): Path to the FAISS vector store directory
        
        Note:
            The vector store must have been previously created using VectrorStore_v1.py.
            This class only handles querying, not building the vector store.
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Get OpenAI API key from environment variables
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize OpenAI embeddings model for vector operations
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Using the latest small embedding model
            api_key=self.api_key
        )
        
        # Initialize OpenAI client for chat completions
        self.client = OpenAI(api_key=self.api_key)
        
        # Store the path to the vector store
        self.vector_store_path = vector_store_path
        
        # FAISS vector store instance (initialized when loading)
        self.db = None

    def load_vectorstore(self):
        """
        Load the FAISS vector store from disk.
        
        This method loads a previously created FAISS vector store without needing
        to rebuild it from documents. The vector store must exist at the specified path.
        
        Note:
            Uses allow_dangerous_deserialization=True to handle FAISS compatibility
            across different versions. The vector store should have been created
            using the same embeddings model for compatibility.
        """
        # Load FAISS vector store from local directory
        self.db = FAISS.load_local(
            self.vector_store_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Uncomment the line below for debugging
        # print("📦 FAISS Vectorstore loaded")

    def ask(self, query, system_prompt, k=5, language="fr"):
        """
        Ask a question and get an AI-generated response based on the knowledge base.
        
        This method implements the core RAG query pipeline:
        1. Perform similarity search to find the most relevant document chunks
        2. Combine retrieved chunks into context
        3. Send context and query to OpenAI for response generation
        
        Args:
            query (str): The question to ask
            system_prompt (str): System prompt to guide the AI's response style and behavior
            k (int): Number of most similar chunks to retrieve (default: 5)
            language (str): Preferred language for response (default: "fr" for French)
        
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
        # Check if vector store is loaded before proceeding
        if not self.db:
            raise ValueError("❌ Vectorstore not loaded. Call load_vectorstore() first.")

        # Perform similarity search to find the k most relevant document chunks
        retrieved_docs = self.db.similarity_search(query, k=k)
        
        # Combine retrieved chunks into a single context string
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Build the complete prompt with system instructions, context, and query
        prompt = system_prompt + f"""Here are the documents (knowledgebase) you need:\n\n{context}\n\n
        Question: {query}"""

        # Generate response using OpenAI's chat completion API
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4o-mini for optimal performance
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )

        # Return the AI response and the context used for transparency
        return response.choices[0].message.content.strip(), context
