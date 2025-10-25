# RAG Pipeline — Modular Retrieval-Augmented Generation

> A minimal, production-style **Retrieval-Augmented Generation (RAG)** pipeline  
> that separates **retrieval**, **reranking**, and **generation** into distinct stages.  
> Designed for **clarity**, **scalability**, and **deployment readiness**.

## Overview

**Traditional RAG** = `Retrieve → Generate`  
**Modern RAG Pipeline** = `Query → Retrieve → Rerank → Generate → Evaluate`

```

```
┌────────────────────┐
│     User Query     │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  Retriever (Top-K) │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│   Reranker (BGE /  │
│   Cohere / LLM)    │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  Generator (LLM)   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│Evaluator (optional)│
└────────────────────┘
```

````

Each stage is independent — making the pipeline **traceable**, **testable**, and **ready for production workloads**.


## Tech Stack

- **Python 3.10+**
- **LangChain** – pipeline orchestration  
- **FAISS / ChromaDB** – vector store  
- **SentenceTransformers** – embeddings  
- **OpenAI / Ollama / Anthropic** – LLM for generation  
- **tqdm** – progress visualization  

## Setup

```bash
git clone https://github.com/<your-handle>/rag-pipeline-demo.git
cd rag-pipeline-demo

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate

pip install -r requirements.txt
````

**requirements.txt**

```
langchain
faiss-cpu
sentence-transformers
openai
tqdm
```

Add your key:

```bash
export OPENAI_API_KEY="your-key"
```

## Example Code

```python
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tqdm import tqdm
import numpy as np

# --- Knowledge Base ---
docs = [
    "RAG stands for Retrieval-Augmented Generation.",
    "A RAG pipeline consists of retrieval, reranking, and generation stages.",
    "Reranking improves the quality of retrieved documents using semantic similarity or LLMs.",
    "Self-RAG and Corrective RAG add validation loops for higher factual accuracy.",
]
embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(docs, embedder)

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Prompts ---
qa_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="Answer using only the provided context:\n{context}\n\nQuestion: {query}"
)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

# --- Pipeline Components ---
def retrieve(query, k=3):
    """Retrieve top-k most similar docs."""
    return vectorstore.similarity_search(query, k=k)

def rerank(query, docs):
    """Simple reranker using LLM scoring (can plug Cohere/BGE)."""
    scores = []
    for d in docs:
        prompt = f"Rate (0-1) how relevant this context is to '{query}':\n{d.page_content}"
        try:
            score = float(llm.invoke(prompt).content[:4])
        except:
            score = 0.5
        scores.append(score)
    ranked = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return ranked

def generate(query, context):
    """Generate final answer."""
    return qa_chain.run({"context": context, "query": query}).strip()

# --- Full RAG Pipeline ---
def rag_pipeline(query, k=3):
    print(f"🔍 Query: {query}")
    docs = retrieve(query, k)
    print(f"Retrieved {len(docs)} docs")
    ranked = rerank(query, docs)
    top_context = "\n".join([d.page_content for d in ranked[:2]])
    answer = generate(query, top_context)
    return answer

# --- Run Demo ---
result = rag_pipeline("What are the main stages in a RAG pipeline?")
print("\n Final Answer:\n", result)
```

## Example Output

```
Query: What are the main stages in a RAG pipeline?
Retrieved 3 docs

Final Answer:
A RAG pipeline typically includes retrieval, reranking, and generation steps. 
The retriever fetches top documents, the reranker refines their order, and the generator produces the final response.
```

## Key Concepts

| Stage                    | Purpose                    | Example Implementation     |
| ------------------------ | -------------------------- | -------------------------- |
| **Retriever**            | Fetches relevant documents | FAISS, ChromaDB            |
| **Reranker**             | Improves relevance ranking | BGE / Cohere / LLM         |
| **Generator**            | Produces final answer      | GPT / Llama / Claude       |
| **Evaluator (optional)** | Scores factual quality     | LLM-based confidence model |


## Next Steps

* Integrate **BGE Reranker** for semantic reranking
* Add **Evaluator** stage for factual confidence
* Wrap pipeline in a **FastAPI** or **LangServe** service
* Store telemetry in **Prometheus / OpenTelemetry** for observability
* Use **Redis / Pinecone** for vector persistence


## References

* [LangChain RAG Guide](https://python.langchain.com)
* [LlamaIndex RAG Pipeline Docs](https://docs.llamaindex.ai)
* [Meta AI Self-RAG (2024)](https://ai.meta.com/research/publications/self-rag/)


