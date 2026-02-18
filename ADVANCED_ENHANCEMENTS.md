# 🔬 Advanced Agent Enhancements (Optional)

## Installation for Advanced Features

```bash
pip install langsmith sentence-transformers faiss-cpu pydantic-ai
```

---

## 1. 🎯 LangSmith Tracing (Production Debugging)

**Purpose**: Visualize agent flow, debug failures, monitor performance

```python
# .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls_xxx
LANGCHAIN_PROJECT=chiranjeevi-prod

# Automatic tracing - no code changes needed!
# View at: https://smith.langchain.com
```

**Benefits**:
- See exact LLM calls and latencies
- Debug clarification logic
- Monitor token usage
- A/B test prompts

---

## 2. 🧠 Semantic Context Assessment (Better Accuracy)

**Purpose**: Use embeddings instead of keyword matching

```python
# agent/clarification.py
from sentence_transformers import SentenceTransformer
import numpy as np

# Load once at startup
_embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Medical context templates
COMPLETE_CONTEXT_EXAMPLES = [
    "I've had severe chest pain for 2 days, radiating to left arm, with shortness of breath",
    "Mild headache started 3 hours ago after skipping lunch, no other symptoms",
]

INCOMPLETE_CONTEXT_EXAMPLES = [
    "I have pain",
    "feeling sick",
    "headache",
]

def assess_context_quality_semantic(query: str) -> float:
    """Use semantic similarity to assess context quality."""
    query_emb = _embedder.encode([query])[0]
    
    complete_embs = _embedder.encode(COMPLETE_CONTEXT_EXAMPLES)
    incomplete_embs = _embedder.encode(INCOMPLETE_CONTEXT_EXAMPLES)
    
    complete_sim = np.max([
        np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        for emb in complete_embs
    ])
    
    incomplete_sim = np.max([
        np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        for emb in incomplete_embs
    ])
    
    # Normalize to 0-1
    score = (complete_sim - incomplete_sim + 1) / 2
    return max(0.0, min(1.0, score))
```

---

## 3. 📝 Structured Output Parser (Guaranteed Format)

**Purpose**: Ensure LLM returns valid JSON for clarification questions

```python
# agent/clarification.py
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class ClarificationOutput(BaseModel):
    """Structured clarification questions."""
    questions: List[str] = Field(description="1-2 specific follow-up questions")
    reasoning: str = Field(description="Why these questions are needed")
    priority: str = Field(description="urgent, moderate, or low")

parser = PydanticOutputParser(pydantic_object=ClarificationOutput)

CLARIFICATION_PROMPT_STRUCTURED = (
    "Generate follow-up questions for this medical query.\n\n"
    "User query: {query}\n"
    "Missing context: {missing_info}\n\n"
    "{format_instructions}\n"
)

def clarification_node_structured(state: AgentState) -> dict:
    """Generate structured clarification questions."""
    query = _get_latest_query(state)
    missing_info = identify_missing_context(query, 0.3)
    
    prompt = CLARIFICATION_PROMPT_STRUCTURED.format(
        query=query,
        missing_info=missing_info,
        format_instructions=parser.get_format_instructions()
    )
    
    response = _llm.invoke([HumanMessage(content=prompt)])
    parsed = parser.parse(response.content)
    
    # Format for user
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(parsed.questions))
    
    return {
        "clarification_needed": True,
        "final_answer": f"💙 {questions_text}\n\n_{parsed.reasoning}_"
    }
```

---

## 4. 💾 Persistent Memory (PostgreSQL)

**Purpose**: Remember conversations across sessions

```python
# agent/agent_graph.py
from langgraph.checkpoint.postgres import PostgresSaver

def build_graph_with_postgres() -> StateGraph:
    """Build graph with PostgreSQL memory."""
    graph = StateGraph(AgentState)
    
    # ... add nodes ...
    
    # PostgreSQL connection
    connection_string = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost/chiranjeevi")
    memory = PostgresSaver.from_conn_string(connection_string)
    
    return graph.compile(checkpointer=memory)

# Now conversations persist!
# Use patient_id as thread_id for patient-specific history
```

---

## 5. 🔍 RAG for Medical Knowledge (Vector Search)

**Purpose**: Ground answers in medical literature

```python
# agent/rag.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# One-time setup: Index medical documents
def build_medical_knowledge_base():
    """Index medical textbooks, papers, guidelines."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load documents (PDFs, text files)
    documents = load_medical_documents()  # Your implementation
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("medical_knowledge_base")
    
    return vectorstore

# Use in research_node
def research_node_with_rag(state: AgentState) -> dict:
    """Enhanced research with local medical knowledge."""
    query = _get_latest_query(state)
    
    # Load vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("medical_knowledge_base", embeddings)
    
    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(query, k=3)
    local_knowledge = "\n\n".join([doc.page_content for doc in docs])
    
    # Combine with web search
    web_results = search_tavily(query)
    pubmed_results = search_pubmed(query)
    
    research_data = f"""
### 📚 Medical Knowledge Base
{local_knowledge}

### 🌐 Web Search
{web_results[0]}

### 📄 PubMed
{pubmed_results[0]}
"""
    
    return {"research_data": research_data}
```

---

## 6. 🎭 Multi-Agent Collaboration (Specialist Agents)

**Purpose**: Route to specialist agents (cardiologist, neurologist, etc.)

```python
# agent/specialists.py
from langgraph.graph import StateGraph

def build_specialist_graph():
    """Multi-agent system with specialists."""
    
    def route_to_specialist(state: AgentState) -> str:
        """Route based on symptom category."""
        query = _get_latest_query(state).lower()
        
        if any(kw in query for kw in ["heart", "chest", "cardiac"]):
            return "cardiologist"
        elif any(kw in query for kw in ["brain", "headache", "dizzy"]):
            return "neurologist"
        elif any(kw in query for kw in ["stomach", "nausea", "digest"]):
            return "gastroenterologist"
        else:
            return "general_practitioner"
    
    graph = StateGraph(AgentState)
    
    # Add specialist nodes
    graph.add_node("cardiologist", cardiologist_node)
    graph.add_node("neurologist", neurologist_node)
    graph.add_node("gastroenterologist", gastro_node)
    graph.add_node("general_practitioner", gp_node)
    
    # Route from clarification
    graph.add_conditional_edges(
        "clarification_node",
        route_to_specialist,
        {
            "cardiologist": "cardiologist",
            "neurologist": "neurologist",
            "gastroenterologist": "gastroenterologist",
            "general_practitioner": "general_practitioner",
        }
    )
    
    return graph.compile()

# Each specialist has domain-specific prompts and knowledge
```

---

## 7. 📊 Confidence Scoring (Uncertainty Quantification)

**Purpose**: Show confidence in recommendations

```python
# agent/confidence.py
import re

def calculate_confidence(state: AgentState) -> float:
    """Calculate confidence based on multiple factors."""
    score = 0.0
    
    # Context quality
    context_quality = state.get("context_quality", 0.0)
    score += context_quality * 0.3
    
    # Research availability
    research_data = state.get("research_data", "")
    if len(research_data) > 500:
        score += 0.3
    
    # Clarification completeness
    clarification_count = state.get("clarification_count", 0)
    if clarification_count >= 1:
        score += 0.2
    
    # LLM self-assessment
    final_answer = state.get("final_answer", "")
    if "uncertain" in final_answer.lower() or "may" in final_answer.lower():
        score -= 0.1
    
    return max(0.0, min(1.0, score))

# Add to answer_node
def answer_node_with_confidence(state: AgentState) -> dict:
    """Generate answer with confidence score."""
    # ... existing answer generation ...
    
    confidence = calculate_confidence(state)
    
    if confidence < 0.5:
        disclaimer = "\n\n⚠️ **Low Confidence**: This assessment is based on limited information. Please consult a doctor."
        answer += disclaimer
    
    return {
        "final_answer": answer,
        "confidence_score": confidence
    }
```

---

## 8. 🔄 Active Learning (Improve Over Time)

**Purpose**: Learn from user feedback

```python
# agent/feedback.py
import json
from datetime import datetime

def log_interaction(query: str, answer: str, feedback: str, rating: int):
    """Log user interactions for model improvement."""
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "feedback": feedback,
        "rating": rating,  # 1-5
    }
    
    with open("feedback_log.jsonl", "a") as f:
        f.write(json.dumps(interaction) + "\n")

# Periodically analyze feedback
def analyze_feedback():
    """Find patterns in low-rated interactions."""
    with open("feedback_log.jsonl") as f:
        interactions = [json.loads(line) for line in f]
    
    low_rated = [i for i in interactions if i["rating"] <= 2]
    
    # Identify common issues
    # Use for prompt engineering or fine-tuning
```

---

## 🎯 Recommended Implementation Order

1. **LangSmith Tracing** (easiest, huge value)
2. **Structured Output Parser** (improves reliability)
3. **Semantic Context Assessment** (better accuracy)
4. **Persistent Memory** (better UX)
5. **RAG for Medical Knowledge** (more grounded answers)
6. **Multi-Agent Specialists** (advanced)
7. **Confidence Scoring** (transparency)
8. **Active Learning** (long-term improvement)

---

## 📦 Dependencies

```bash
# requirements-advanced.txt
langsmith>=0.1.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
psycopg2-binary>=2.9.0
pydantic>=2.0.0
```

---

**Note**: These are optional enhancements. The current implementation already provides significant improvements over the baseline!
