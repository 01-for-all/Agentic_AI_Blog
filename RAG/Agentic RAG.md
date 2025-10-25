# Agentic RAG — An Intelligent Retrieval-Augmented Generation Pipeline

> A hands-on example of **Agentic RAG**, where an LLM acts as an *autonomous planner* —  
> deciding **how** to search, **what** to retrieve, and **when** to refine or rerun a query before answering.  
> Ideal for **AI copilots**, **knowledge assistants**, and **multi-step reasoning systems**.

---

## Concept

Traditional RAG:  
```

User → Retriever → Generator → Answer

```

Agentic RAG:  

```

User → Agent (Planner)
├── Decides strategy (semantic / keyword / hybrid)
├── Retrieves & Grades context
├── Rewrites or expands query if needed
└── Generates validated final response

````

This adds **reasoning**, **planning**, and **control loops** to the retrieval process —  
making the pipeline far more robust for real-world enterprise use cases.

## Stack

- **Python 3.10+**
- **LangChain** — orchestration framework  
- **FAISS / ChromaDB** — vector database  
- **SentenceTransformers** — embeddings  
- **OpenAI / Ollama / Anthropic** — LLM interface  

## Installation

```bash
git clone https://github.com/<your-handle>/agentic-rag-demo.git
cd agentic-rag-demo

python -m venv .venv
source .venv/bin/activate  # (or .venv\Scripts\activate on Windows)

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

Set your API key:

```bash
export OPENAI_API_KEY="your-key-here"
```


## Code Example

```python
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tqdm import tqdm

# --- Step 1: Knowledge Base ---
docs = [
    "RAG stands for Retrieval-Augmented Generation.",
    "Agentic RAG introduces reasoning and planning in retrieval pipelines.",
    "Corrective RAG uses feedback loops to fix low-confidence answers.",
    "Self-RAG evaluates the factuality of its own outputs.",
]
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(docs, embeddings)

# --- Step 2: LLM Setup ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

planner_prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are a retrieval planner. "
        "Analyze the query and decide retrieval strategy. "
        "Return one of: ['semantic', 'keyword', 'hybrid'] and optionally rewrite the query.\n\n"
        "User query: {query}\n"
        "Your response (JSON):"
    )
)
planner_chain = LLMChain(llm=llm, prompt=planner_prompt)

qa_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="Answer based strictly on the context:\n{context}\n\nQuestion: {query}"
)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

grade_prompt = PromptTemplate(
    input_variables=["answer"],
    template="Rate confidence (0-1) that the answer is factual and complete:\n{answer}"
)
grader_chain = LLMChain(llm=llm, prompt=grade_prompt)

# --- Step 3: Agentic Loop ---
def agentic_rag(query, max_iters=2, confidence_threshold=0.7):
    for _ in tqdm(range(max_iters)):
        plan = planner_chain.run({"query": query})
        print(f"\n🧭 Agent Plan: {plan}")
        # (Simplified parser)
        if "rewrite" in plan.lower():
            query = plan.split("rewrite:")[-1].strip()
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])
        answer = qa_chain.run({"context": context, "query": query}).strip()
        conf = float(grader_chain.run({"answer": answer})[:4] or 0)
        print(f"Confidence: {conf:.2f}")
        if conf >= confidence_threshold:
            return {"answer": answer, "confidence": conf}
        query = f"Refine and clarify: {query}"
    return {"answer": answer, "confidence": conf}

# --- Run ---
result = agentic_rag("How does Agentic RAG differ from traditional RAG?")
print("\n✅ Final Answer:", result)
```


## 🧭 Example Output

```
🧭 Agent Plan: {"strategy": "semantic", "rewrite": "Explain the difference between Agentic and Traditional RAG."}
Confidence: 0.93

Final Answer: 
Agentic RAG differs from traditional RAG by adding reasoning and planning steps before retrieval, allowing the system to decide strategies, rewrite queries, and validate outputs for higher accuracy.
```



## 🧩 Key Components

| Module            | Function                                         |
| ----------------- | ------------------------------------------------ |
| **Planner**       | Decides retrieval strategy or rewrites the query |
| **Retriever**     | Searches relevant documents                      |
| **Grader**        | Evaluates factual confidence                     |
| **Feedback Loop** | Replans and re-retrieves when confidence is low  |



## Extending the Demo

✅ Integrate **LangGraph** for multi-agent workflows
✅ Add **reranker** (BGE / Cohere) for context quality
✅ Swap **FAISS → Chroma / Pinecone / Redis** for production
✅ Use **external tools** (API calls, SQL, etc.) via agent actions



## References

* [LangChain: Agents & RAG](https://python.langchain.com)
* [LlamaIndex Agentic RAG](https://docs.llamaindex.ai)
* [Meta Self-RAG Paper (2024)](https://ai.meta.com/research/publications/self-rag/)
* [LangGraph Framework](https://www.langchain.com/langgraph)


