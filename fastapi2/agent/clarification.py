"""
Chiranjeevi Medical Agent — Context Quality & Clarification
============================================================
Implements Trust Envelope™ for medical queries using:
- Context quality assessment
- Intelligent clarification generation
- Conversation state tracking
"""

import re
from langchain_core.messages import HumanMessage
from agent.state import AgentState
from agent.config import CONTEXT_ASSESSOR_PROMPT, CLARIFICATION_PROMPT


_llm = None


def set_llm(llm_instance):
    """Inject the LLM instance."""
    global _llm
    _llm = llm_instance


def _get_latest_query(state: AgentState) -> str:
    """Extract the latest user message."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def assess_context_quality(query: str) -> float:
    """
    Assess if query has sufficient context for medical advice.
    
    Returns:
        float: Quality score 0.0-1.0
    """
    # Simple heuristic-based assessment (can be enhanced with LLM)
    score = 0.0
    
    # Check for duration indicators
    duration_keywords = ["days", "weeks", "months", "hours", "since", "ago", "started"]
    if any(kw in query.lower() for kw in duration_keywords):
        score += 0.25
    
    # Check for severity indicators
    severity_keywords = ["severe", "mild", "moderate", "intense", "slight", "bad", "terrible"]
    if any(kw in query.lower() for kw in severity_keywords):
        score += 0.25
    
    # Check for associated symptoms (multiple symptoms mentioned)
    symptom_count = len(re.findall(r'\b(pain|ache|fever|nausea|dizzy|tired|swelling|rash)\b', query.lower()))
    if symptom_count >= 2:
        score += 0.25
    
    # Check for medical history mentions
    history_keywords = ["history", "medication", "taking", "diagnosed", "condition", "allergic"]
    if any(kw in query.lower() for kw in history_keywords):
        score += 0.25
    
    return min(score, 1.0)


def identify_missing_context(query: str, quality_score: float) -> str:
    """Identify what context is missing from the query."""
    missing = []
    
    duration_keywords = ["days", "weeks", "months", "hours", "since", "ago"]
    if not any(kw in query.lower() for kw in duration_keywords):
        missing.append("symptom duration")
    
    severity_keywords = ["severe", "mild", "moderate", "intense"]
    if not any(kw in query.lower() for kw in severity_keywords):
        missing.append("severity level")
    
    if quality_score < 0.5:
        missing.append("associated symptoms or triggers")
    
    return ", ".join(missing) if missing else "general context"


def clarification_node(state: AgentState) -> dict:
    """
    Assess context quality and generate clarification questions if needed.
    
    This implements the Trust Envelope™ boundary check.
    """
    query = _get_latest_query(state)
    clarification_count = state.get("clarification_count", 0)
    
    # Don't ask more than 2 clarification questions
    if clarification_count >= 2:
        return {
            "clarification_needed": False,
            "context_quality": 1.0  # Proceed with available context
        }
    
    # Assess context quality
    quality = assess_context_quality(query)
    
    # If quality is sufficient, proceed to answer
    if quality >= 0.6:
        return {
            "clarification_needed": False,
            "context_quality": quality
        }
    
    # Generate clarification questions
    missing_info = identify_missing_context(query, quality)
    
    prompt = CLARIFICATION_PROMPT.format(
        query=query,
        missing_info=missing_info
    )
    
    response = _llm.invoke([HumanMessage(content=prompt)])
    clarification_text = response.content if hasattr(response, "content") else str(response)
    
    return {
        "clarification_needed": True,
        "clarification_count": clarification_count + 1,
        "context_quality": quality,
        "final_answer": clarification_text.strip()
    }
