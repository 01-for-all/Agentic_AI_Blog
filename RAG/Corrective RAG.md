# Corrective RAG — A Self-Improving Retrieval-Augmented Generation Pipeline

> A minimal example of a **Corrective RAG** workflow — where the system re-evaluates and corrects its own response when confidence is low.  
> Ideal for enterprise chatbots, copilots, or knowledge assistants that need reliability and factual accuracy.


## 🚀 What Is Corrective RAG?

Traditional RAG = `Retrieve → Generate`.

**Corrective RAG** adds feedback and correction:


```markdown
User Query
↓
Retriever → Generator → Evaluator
↘────────── If low confidence ─────────↙
Query Rewriter → Retriever → Generator → Response

````

This loop ensures the model can **detect low-quality answers**, **re-query intelligently**, and **improve its response**.

## Tech Stack

- **Python 3.10+**
- **LangChain** – pipeline orchestration  
- **FAISS / ChromaDB** – vector store  
- **OpenAI / Ollama / Anthropic** – LLM (plug your choice)  
- **SentenceTransformers** – embedding model  



## Setup

```bash
# 1. Clone this repo
git clone https://github.com/<your-handle>/corrective-rag.git
cd corrective-rag

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt
````

**requirements.txt**

```
langchain
faiss-cpu
openai
sentence-transformers
tqdm
```

Set your key:

```bash
export OPENAI_API_KEY="your-key-here"
```


## Example Code

```python
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from tqdm import tqdm

# --- Step 1: Build Vector Store ---
docs = [
    "RAG stands for Retrieval-Augmented Generation.",
    "Corrective RAG uses feedback loops to improve factual accuracy.",
    "Agentic RAG employs reasoning and planning across steps.",
]
embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(docs, embedder)

# --- Step 2: Define Models & Prompts ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="Answer based only on context:\n{context}\n\nQuestion: {query}"
)
eval_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Rate confidence (0-1) that this answer is factually correct:\nQ: {question}\nA: {answer}"
)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
eval_chain = LLMChain(llm=llm, prompt=eval_prompt)

# --- Step 3: Corrective Loop ---
def corrective_rag(query, threshold=0.6, max_iters=2):
    for _ in tqdm(range(max_iters)):
        docs = vectorstore.similarity_search(query, k=2)
        context = "\n".join([d.page_content for d in docs])
        answer = qa_chain.run({"context": context, "query": query}).strip()
        score = float(eval_chain.run({"question": query, "answer": answer}).strip()[:3] or 0)
        if score >= threshold:
            return {"answer": answer, "confidence": score}
        query = f"Refine and clarify: {query}"  # rewrite query if low confidence
    return {"answer": answer, "confidence": score}

# --- Run ---
result = corrective_rag("What is Corrective RAG?")
print(result)
```

## 🧭 Output Example

```
{'answer': 'Corrective RAG is a retrieval-augmented generation technique that uses feedback loops to refine inaccurate or incomplete responses.', 
 'confidence': 0.91}
```


## 🔍 Key Takeaways

| Concept             | Role                              |
| ------------------- | --------------------------------- |
| **Evaluator**       | Judges response confidence        |
| **Query Rewriter**  | Refines poorly answered questions |
| **Feedback Loop**   | Enables self-correction           |
| **Rerun Retrieval** | Improves factual grounding        |


## 🧩 Extensions

* Add a **reranker (BGE or Cohere)** before generation
* Replace evaluator with **LLM-based “judge” model**
* Integrate into **LangGraph or CrewAI** for multi-agent workflows
* Swap FAISS with **Chroma / Pinecone / Redis** for production scaling


## 📘 References

* [Meta AI – Self-RAG Paper (2024)](https://ai.meta.com/research/publications/self-rag/)
* [LangChain Docs](https://python.langchain.com)
* [LlamaIndex RAG Patterns](https://docs.llamaindex.ai)

```
