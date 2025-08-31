"""
RAG Knowledge Base System - Langchain + FAISS Implementation

This module provides a Retrieval Augmented Generation (RAG) system using Langchain and FAISS
for building and querying knowledge bases from various document formats.

Features:
- Document loading from multiple formats (PDF, DOCX, Excel, TXT, JSON)
- Automatic text chunking with configurable parameters
- FAISS vector store for fast similarity search
- OpenAI integration for embeddings and chat completions
- Comprehensive error handling and logging

Author: Usuf Com
Contact: usufcom20@gmail.com
Website: www.djamai.com
LinkedIn: https://www.linkedin.com/in/usufcom
Date: 2024
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


class RAGKnowledgeBase:
    """
    A comprehensive RAG knowledge base system using Langchain and FAISS.
    
    This class handles document processing, vector store creation, and AI-powered
    question answering using OpenAI's GPT models and embeddings.
    
    Attributes:
        api_key (str): OpenAI API key for embeddings and chat completions
        client (OpenAI): OpenAI client instance
        embeddings (OpenAIEmbeddings): Embeddings model for vector creation
        folder_path (str): Path to the documents folder
        vector_store_path (str): Path where the FAISS vector store will be saved
        db (FAISS): FAISS vector store instance
    """
    
    def __init__(self, folder_path='', vector_store_path='faiss_vectorstore'):
        """
        Initialize the RAG Knowledge Base system.
        
        Args:
            folder_path (str): Path to the folder containing documents to process
            vector_store_path (str): Path where the FAISS vector store will be saved/loaded
        
        Raises:
            ValueError: If OPENAI_API_KEY is not found in environment variables
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Get OpenAI API key from environment variables
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env")
        
        # Initialize OpenAI client for chat completions
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize OpenAI embeddings model for vector creation
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Using the latest small embedding model
            api_key=self.api_key
        )
        
        # Store configuration paths
        self.folder_path = folder_path
        self.vector_store_path = vector_store_path
        self.db = None  # FAISS vector store instance (initialized later)

    def load_documents(self):
        """
        Load documents from the specified folder and convert them to Langchain Document objects.
        
        Supports multiple file formats:
        - PDF files (.pdf) - using PyPDFLoader
        - Word documents (.docx) - using Docx2txtLoader
        - Excel files (.xlsx, .xls) - using pandas
        - Text files (.txt) - direct file reading
        - JSON files (.json) - JSON parsing and string conversion
        
        Returns:
            list: List of Langchain Document objects with content and metadata
        
        Note:
            Each document includes metadata with the source filename for traceability.
        """
        docs = []
        
        # Iterate through all files in the specified folder
        for file in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file)

            try:
                # Handle PDF files using Langchain's PyPDFLoader
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs.extend(loader.load())

                # Handle Word documents using Langchain's Docx2txtLoader
                elif file.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    docs.extend(loader.load())

                # Handle Excel files by converting to string representation
                elif file.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file_path)
                    docs.append(Document(
                        page_content=df.to_string(), 
                        metadata={"source": file}
                    ))

                # Handle plain text files
                elif file.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = f.read()
                    docs.append(Document(
                        page_content=data, 
                        metadata={"source": file}
                    ))

                # Handle JSON files by converting to string
                elif file.endswith(".json"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    docs.append(Document(
                        page_content=json.dumps(data), 
                        metadata={"source": file}
                    ))

            except Exception as e:
                # Log errors but continue processing other files
                print(f"❌ Error loading {file}: {e}")
                continue
                
        return docs

    def build_vectorstore(self, chunk_size=1000, chunk_overlap=200):
        """
        Build a FAISS vector store from loaded documents.
        
        This method:
        1. Loads documents from the specified folder
        2. Splits documents into chunks for better retrieval
        3. Creates embeddings for each chunk
        4. Builds and saves a FAISS index for fast similarity search
        
        Args:
            chunk_size (int): Maximum size of each text chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
        
        Returns:
            tuple: (number_of_documents, number_of_chunks)
        
        Note:
            chunk_overlap should be less than chunk_size to ensure meaningful chunks.
            Larger chunks provide more context but may be less precise for retrieval.
        """
        # Load all documents from the folder
        documents = self.load_documents()
        print(f"📂 Loaded {len(documents)} raw docs")

        # Create text splitter for chunking documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Split documents into smaller chunks
        chunks = splitter.split_documents(documents)
        print(f"✂️ Split into {len(chunks)} chunks")

        # Create FAISS vector store from chunks using OpenAI embeddings
        self.db = FAISS.from_documents(chunks, self.embeddings)
        
        # Save the vector store to disk for future use
        self.db.save_local(self.vector_store_path)
        print(f"💾 Vectorstore saved to {self.vector_store_path}")
        
        # Return statistics for monitoring
        documentsNum = len(documents)
        chunkNum = len(chunks)
        return documentsNum, chunkNum

    def load_vectorstore(self):
        """
        Load an existing FAISS vector store from disk.
        
        This method loads a previously created vector store without needing to
        rebuild it from documents. Useful for querying existing knowledge bases.
        
        Note:
            The vector store must have been previously created using build_vectorstore().
            Uses allow_dangerous_deserialization=True to handle FAISS compatibility.
        """
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
        
        This method implements the core RAG pipeline:
        1. Retrieve the most relevant document chunks using similarity search
        2. Combine retrieved chunks into context
        3. Send context and query to OpenAI for response generation
        
        Args:
            query (str): The question to ask
            system_prompt (str): System prompt to guide the AI's response
            k (int): Number of most similar chunks to retrieve (default: 5)
            language (str): Preferred language for response (default: "fr" for French)
        
        Returns:
            tuple: (AI_response, retrieved_context)
        
        Raises:
            ValueError: If vector store is not loaded
        
        Note:
            The system_prompt should provide instructions on how the AI should respond
            based on the retrieved context. The retrieved_context can be used for
            debugging or transparency purposes.
        """
        # Check if vector store is loaded
        if not self.db:
            raise ValueError("❌ Vectorstore not loaded. Call build_vectorstore() or load_vectorstore() first.")

        # Retrieve the k most similar document chunks
        retrieved_docs = self.db.similarity_search(query, k=k)
        
        # Combine retrieved chunks into a single context string
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Build the complete prompt with context and query
        prompt = system_prompt + f"""Here are the documents (knowledgebase) you need:\n\n{context}\n\n
        Question: {query}"""

        # Generate response using OpenAI's chat completion API
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4o-mini for better performance
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )

        # Return the AI response and the context used for transparency
        return response.choices[0].message.content.strip(), context
