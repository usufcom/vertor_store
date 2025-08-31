# Quick Start Guide

Get your RAG Knowledge Base System up and running in minutes!

## 🚀 Quick Setup

### 1. Clone and Install
```bash
git clone <your-repo-url>
cd vertor_store
pip install -r requirements.txt
```

### 2. Set Up Environment
```bash
# Copy the example environment file
cp env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Add Your Documents
Place your documents in the `docs/` folder:
- PDF files (`.pdf`)
- Word documents (`.docx`)
- Text files (`.txt`)
- Excel files (`.xlsx`, `.xls`)
- JSON files (`.json`)

### 4. Run the Demo
Open `Run_RAG.ipynb` in Jupyter and run the cells to see both approaches in action!

## 🎯 Quick Examples

### Approach 1: Langchain + FAISS
```python
from VectrorStore_v1 import RAGKnowledgeBase
from querykb_v1 import queryKnowledgeBase

# Build knowledge base
kb = RAGKnowledgeBase('docs', 'vectorstore/faiss_vectorstore')
kb.build_vectorstore()

# Query knowledge base
query_kb = queryKnowledgeBase('vectorstore/faiss_vectorstore')
query_kb.load_vectorstore()
response, context = query_kb.ask("Your question here", "Your system prompt")
```

### Approach 2: Native Vector Store
```python
from VectorStore_v2 import VectorStore
from querykb_v2 import RAG
import os

# Build knowledge base
store = VectorStore(os.getenv("OPENAI_API_KEY"))
store.exract_save_vector_store(store, "docs", "vectorstore/vector_store.json")

# Query knowledge base
rag = RAG("vectorstore/vector_store.json")
answer, context = rag.askAI("Your question here", "Your system prompt")
```

## 🔧 Configuration Tips

- **Chunk Size**: Increase for longer context, decrease for more precise retrieval
- **Chunk Overlap**: Higher overlap helps maintain context across chunks
- **Top-K**: More retrieved chunks = more context but potentially less focused answers

## 🆘 Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Open an issue on GitHub for bugs or feature requests
- Review the example notebook for working code samples

## 🎉 You're Ready!

Your RAG system is now ready to answer questions based on your documents!
