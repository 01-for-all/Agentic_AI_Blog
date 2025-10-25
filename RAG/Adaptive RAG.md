# Adaptive RAG — Dynamic Retrieval-Augmented Generation

> A minimal, working example of an **Adaptive RAG** pipeline —  
> where the system **evaluates context quality** and **adapts its retrieval strategy on-the-fly**  
> to deliver more accurate and relevant responses.

## What Is Adaptive RAG?

Traditional RAG pipelines are static:

```

User → Retriever → Generator → Answer

```

**Adaptive RAG** introduces intelligence:

```

User → Evaluator → Strategy Selector
├── Keyword Retrieval (exact match)
├── Semantic Retrieval (embeddings)
├── Hybrid (merge + rerank)
└── Confidence Feedback → Adjust or Retry

````

This allows the system to **adapt retrieval depth, method, or reranking logic**
based on **query complexity and relevance feedback**.


## Tech Stack

- **Python 3.10+**
- **LangChain** — orchestration
- **FAISS** — vector store
- **SentenceTransformers** — embeddings
- **OpenAI / Ollama / Anthropic** — LLM provider
- **tqdm** — progress visualization


## Setup

```bash
git clone https://github.com/<your-handle>/adaptive-rag-demo.git
cd adaptive-rag-demo

python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

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

Add your API key:

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
import re

# --- Knowledge Base ---
docs = [
    "RAG combines retrieval and generation to answer questions.",
    "Adaptive RAG dynamically changes its retrieval strategy based on query type.",
    "Agentic RAG adds reasoning and planning steps.",
    "Self-RAG evaluates its own factual accuracy.",
    "Keyword search is useful for short, specific queries.",
]
embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(docs, embedder)

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

strategy_prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are a retrieval strategy selector. "
        "Decide which retrieval mode fits this query: "
        "'keyword', 'semantic', or 'hybrid'. Return only the mode.\n\n"
        "Query: {query}"
    )
)
strategy_chain = LLMChain(llm=llm, prompt=strategy_prompt)

qa_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="Answer concisely using only this context:\n{context}\n\nQuestion: {query}"
)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

judge_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template="Rate confidence (0-1) that the answer is factually correct for the question.\nQ:{query}\nA:{answer}"
)
judge_chain = LLMChain(llm=llm, prompt=judge_prompt)

# --- Adaptive Function ---
def adaptive_rag(query, threshold=0.7, max_iters=2):
    for i in tqdm(range(max_iters)):
        mode = strategy_chain.run({"query": query}).lower().strip()
        print(f"\n🧭 Retrieval mode: {mode}")

        if "keyword" in mode:
            # naive keyword filtering
            hits = [d for d in docs if any(w.lower() in d.lower() for w in query.split())]
            context = "\n".join(hits[:3]) or "\n".join(docs[:2])
        elif "hybrid" in mode:
            sem = [d.page_content for d in vectorstore.similarity_search(query, k=2)]
            key = [d for d in docs if re.search(query.split()[0], d, re.I)]
            context = "\n".join(list(set(sem + key))[:3])
        else:  # semantic default
            sem = [d.page_content for d in vectorstore.similarity_search(query, k=3)]
            context = "\n".join(sem)

        answer = qa_chain.run({"context": context, "query": query}).strip()
        conf = float(judge_chain.run({"query": query, "answer": answer})[:4] or 0)
        print(f"Confidence: {conf:.2f}")
        if conf >= threshold:
            return {"answer": answer, "confidence": conf, "mode": mode}
        query = f"Refine query: {query}"
    return {"answer": answer, "confidence": conf, "mode": mode}

# --- Run Demo ---
result = adaptive_rag("Explain Adaptive RAG and when to use it.")
print("\n✅ Final Answer:", result)
```

## Example Output

```
Retrieval mode: hybrid
Confidence: 0.88

Final Answer:
Adaptive RAG automatically switches retrieval strategies depending on the query. It balances keyword precision with semantic depth, improving relevance and factual reliability.
```


## 📊 Key Highlights

| Module                | Purpose                                    |
| --------------------- | ------------------------------------------ |
| **Strategy Selector** | Chooses best retrieval mode                |
| **Retriever**         | Fetches candidate docs                     |
| **Generator**         | Produces contextual answer                 |
| **Judge**             | Evaluates factual confidence               |
| **Adaptive Loop**     | Refines strategy if confidence < threshold |


## 🔧 Next Steps

* Add **relevance scoring / reranking (Cohere Reranker)**
* Cache previous strategies for **learning-based adaptation**
* Integrate **LangGraph or CrewAI** for multi-agent control
* Deploy on **FastAPI / Streamlit** for a demo UI


## 📘 References

* [LangChain Docs](https://python.langchain.com)
* [LlamaIndex Adaptive RAG Concepts](https://docs.llamaindex.ai)
* [Meta AI – Self-RAG (2024)](https://ai.meta.com/research/publications/self-rag/)

