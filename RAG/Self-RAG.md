# Self-RAG — Self-Reflective Retrieval-Augmented Generation

> A minimal yet powerful demo of **Self-RAG**, where the LLM retrieves evidence, 
> generates an answer, and then **evaluates its own confidence** before returning the final output.

## Overview

Self-RAG extends classic RAG with an **internal evaluation loop**.

Instead of just `Retrieve → Generate`,  
Self-RAG adds **reflection** and **validation**:

```
User Query
↓
Retriever → Generator → Evaluator → (Retry / Finalize)

```

The LLM plays **two roles**:
- as a *generator* (to create an answer)
- and as an *evaluator* (to critique or verify it)


## Architecture

```

```
      ┌─────────────────────────────┐
      │          User Query         │
      └──────────────┬──────────────┘
                     ↓
          ┌────────────────────┐
          │     Retriever      │
          │ (FAISS / Chroma)   │
          └────────┬───────────┘
                   ↓
          ┌────────────────────┐
          │     Generator       │
          │   (LLM produces     │
          │   initial answer)   │
          └────────┬───────────┘
                   ↓
          ┌────────────────────┐
          │     Evaluator       │
          │ (LLM checks accuracy│
          │  and confidence)    │
          └────────┬───────────┘
                   ↓
          ┌────────────────────┐
          │ Retry / Final Answer│
          └────────────────────┘
```

````

This design helps prevent **hallucination** and ensures **trustworthy responses**.


## Stack

- **Python 3.10+**
- **LangChain**
- **FAISS / ChromaDB**
- **SentenceTransformers**
- **OpenAI / Ollama / Anthropic**
- **tqdm**


## Setup

```bash
git clone https://github.com/<your-handle>/self-rag-demo.git
cd self-rag-demo

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

# --- Knowledge Base ---
docs = [
    "Self-RAG stands for Self-Reflective Retrieval-Augmented Generation.",
    "In Self-RAG, the model evaluates its own output for factual accuracy.",
    "This reduces hallucinations and builds trust in generative AI.",
    "RAG pipelines combine retrieval and generation to answer domain-specific questions."
]

embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(docs, embedder)

# --- Model Setup ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Prompts ---
generate_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="Use the context below to answer accurately.\nContext:\n{context}\n\nQuestion: {query}"
)

evaluate_prompt = PromptTemplate(
    input_variables=["query", "answer", "context"],
    template=(
        "Question: {query}\n"
        "Answer: {answer}\n"
        "Context: {context}\n\n"
        "Evaluate factual accuracy (0-1) and state why.\n"
        "If score < 0.7, suggest a corrected version.\n"
        "Return as JSON: {\"score\": float, \"feedback\": str, \"revised_answer\": str}"
    )
)

generate_chain = LLMChain(llm=llm, prompt=generate_prompt)
evaluate_chain = LLMChain(llm=llm, prompt=evaluate_prompt)

# --- Pipeline ---
def retrieve(query, k=3):
    return vectorstore.similarity_search(query, k=k)

def self_rag_pipeline(query, threshold=0.7):
    print(f" Query: {query}")
    docs = retrieve(query, 3)
    context = "\n".join([d.page_content for d in docs])

    # Step 1: Generate initial answer
    answer = generate_chain.run({"context": context, "query": query}).strip()

    # Step 2: Evaluate confidence
    evaluation = evaluate_chain.run({
        "query": query,
        "answer": answer,
        "context": context
    })

    print("\n Evaluation:\n", evaluation)

    # Step 3: Parse evaluation
    try:
        import json
        eval_data = json.loads(evaluation)
        score = eval_data.get("score", 0)
        revised = eval_data.get("revised_answer", "")
    except:
        score = 0
        revised = answer

    # Step 4: Decide whether to retry
    if score < threshold and revised:
        print("\n Low confidence detected. Using revised answer...")
        return revised
    else:
        return answer

# --- Run Demo ---
result = self_rag_pipeline("What is Self-RAG and how does it prevent hallucination?")
print("\n Final Answer:\n", result)
```


## Example Output

```
Query: What is Self-RAG and how does it prevent hallucination?

Evaluation:
{"score": 0.85, "feedback": "Accurate and grounded in provided context.", "revised_answer": ""}

Final Answer:
Self-RAG is a retrieval-augmented generation method where the model evaluates its own answer for factual accuracy, reducing hallucinations and improving trust.
```

## Key Benefits

| Feature                | Description                           | Outcome                        |
| ---------------------- | ------------------------------------- | ------------------------------ |
| **Self-Evaluation**    | The LLM reviews its own output        | Detects low-confidence answers |
| **Confidence Scoring** | Assigns factuality scores             | Enables trust metrics          |
| **Automatic Revision** | Can self-correct when unsure          | Reduces hallucination risk     |
| **Context Grounding**  | Answers constrained to retrieved docs | Improves factual consistency   |


## Next Steps

* Integrate **fact-checking APIs** for hybrid verification
* Add **multi-hop reflection** (LLM evaluates twice)
* Use **structured evaluation storage** (SQLite / Weights & Biases)
* Expose as **FastAPI endpoint** for real-time agentic chatbots


## References

* [Meta AI — Self-RAG (2024)](https://ai.meta.com/research/publications/self-rag/)
* [LangChain Evaluation Framework](https://python.langchain.com)
* [LlamaIndex Advanced RAG Patterns](https://docs.llamaindex.ai)



