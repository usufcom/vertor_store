# RAG Knowledge Base System

A comprehensive Retrieval Augmented Generation (RAG) system that combines document processing, vector storage, and AI-powered question answering. This project demonstrates two different approaches to building knowledge bases using vector stores.

## 🚀 Features

- **Dual Implementation Approaches**:
  - **Langchain + FAISS**: Fast similarity search using FAISS index
  - **Native Vector Store**: Custom implementation with cosine similarity
- **Document Processing**: Automatic chunking and embedding generation
- **Flexible Query System**: Ask questions and get AI-generated responses
- **Customizable Prompts**: Use system prompts to guide AI responses
- **Multiple Document Formats**: Support for various document types

## 📁 Project Structure

```
vertor_store/
├── docs/                          # Document storage
│   ├── 2312.10997v5.pdf
│   ├── Vectorstores_explanation.txt
│   └── What is Knowledge based AI.docx
├── prompts/                       # System prompts
│   └── System_Prompt.txt
├── vectorstore/                   # Generated vector stores
│   ├── faiss_vectorstore/        # FAISS-based storage
│   └── vector_store.json         # Native JSON storage
├── VectrorStore_v1.py            # Langchain + FAISS implementation
├── VectorStore_v2.py             # Native vector store implementation
├── querykb_v1.py                 # Query interface for FAISS
├── querykb_v2.py                 # Query interface for native store
└── Run_RAG.ipynb                 # Main demonstration notebook
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd vertor_store
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 📚 Usage

### Approach 1: Langchain + FAISS

#### Step 1: Build Vector Store
```python
from VectrorStore_v1 import RAGKnowledgeBase

# Initialize knowledge base
kb_folder = 'docs'
vector_store_path = 'vectorstore/faiss_vectorstore'
kb = RAGKnowledgeBase(kb_folder, vector_store_path)

# Build index (run once or when docs are updated)
documentsNum, chunkNum = kb.build_vectorstore(chunk_size=1000, chunk_overlap=200)
```

#### Step 2: Query the Knowledge Base
```python
from querykb_v1 import queryKnowledgeBase
import textwrap

# Initialize query interface
kb = queryKnowledgeBase(vector_store_path='vectorstore/faiss_vectorstore')
kb.load_vectorstore()

# Load system prompt
with open("prompts/System_Prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Ask questions
user_msg = "What is the importance of vector store and knowledge base in AI?"
response, context = kb.ask(user_msg, system_prompt, k=2)
print(f"AI Response: {textwrap.fill(response, width=70)}")
```

### Approach 2: Native Vector Store

#### Step 1: Create Vector Store
```python
from VectorStore_v2 import VectorStore
import os

# Initialize vector store
store = VectorStore(
    api_key=os.getenv("OPENAI_API_KEY"), 
    chunk_size=1000, 
    chunk_overlap=200
)

# Extract and save vector store
kb_folder = "docs"
vector_store_path = "vectorstore/vector_store.json"
store.exract_save_vector_store(store, kb_folder, vector_store_path)
```

#### Step 2: Query with Cosine Similarity
```python
from querykb_v2 import RAG
import textwrap

# Initialize RAG system
rag = RAG(vector_store_path="vectorstore/vector_store.json")

# Load system prompt
with open("prompts/System_Prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Ask questions
user_msg = "What is the importance of vector store and knowledge base in AI?"
answer, used_context = rag.askAI(user_msg, system_prompt)
print(f"Answer: {textwrap.fill(answer, width=80)}")
```

## 🔧 Configuration

### Chunking Parameters
- `chunk_size`: Size of document chunks (default: 1000 characters)
- `chunk_overlap`: Overlap between chunks (default: 200 characters)

### Retrieval Parameters
- `k`: Number of most similar chunks to retrieve (default: 2)

### System Prompts
Customize the AI behavior by modifying `prompts/System_Prompt.txt` to guide how the AI should respond to queries.

## 📖 Example

The `Run_RAG.ipynb` notebook provides a complete demonstration of both approaches, showing how to:
1. Build vector stores from documents
2. Query the knowledge base
3. Get AI-generated responses based on retrieved context

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT models
- Langchain for the RAG framework
- FAISS for efficient similarity search
- The AI community for inspiration and guidance

## 📞 Support & Contact

If you have any questions or need help, please reach out:

- **Website**: [www.djamai.com](https://www.djamai.com)
- **Email**: usufcom20@gmail.com
- **LinkedIn**: [@usufcom](https://www.linkedin.com/in/usufcom)

You can also open an issue on GitHub for bugs or feature requests.
