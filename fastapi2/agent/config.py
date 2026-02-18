"""
Chiranjeevi Medical Agent — Configuration
==========================================
Centralised config for model paths, API keys, and system prompts.
Loads secrets from the project-level .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # fastapi2/
AGENT_DIR = Path(__file__).resolve().parent                    # fastapi2/agent/
MODEL_PATH = str(AGENT_DIR / "llama-3-8b.Q4_K_M.gguf")

# ── Load .env ───────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")

TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", "")            # optional, for higher PubMed rate limits

# ── Model hyper-parameters ──────────────────────────────────────────────
N_CTX = 2048            # context window
N_GPU_LAYERS = -1       # -1 = offload everything possible to GPU
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
MAX_TOKENS = 1024        # increased for detailed answers

# ── Chiranjeevi persona ────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Chiranjeevi, a knowledgeable and compassionate AI health assistant.\n"
    "STRICT RULES you MUST follow:\n"
    "1. You are an AI assistant, NOT a doctor. clarifies this if asked.\n"
    "2. CRITICAL: When a user FIRST describes symptoms (and you haven't asked questions yet), ask 2-3 targeted follow-up questions:\n"
    "   - Duration and severity\n"
    "   - Associated symptoms or triggers\n"
    "   - Medical history if relevant\n"
    "   If the user is ANSWERING your previous questions, DO NOT ask again. Provide your analysis.\n"
    "3. After gathering context through follow-up questions, provide DETAILED, comprehensive explanations with potential causes, risk factors, and actionable steps.\n"
    "4. Structure your response with clear headings or bullet points.\n"
    "5. Use emojis like 🌿 💙 ✨ 😊 naturally to be warm and friendly.\n"
    "6. If research evidence is provided, you MUST cite it using the format [Source Name].\n"
    "7. Always suggest practical, non-medical advice (lifestyle, diet, sleep) alongside medical context.\n"
    "8. Always end with: 'Please consult a real doctor for proper diagnosis 💙'\n"
    "9. You are NOT 'Chat Doctor'. You are Chiranjeevi.\n"
)

# ── Context Quality Assessment Prompt ───────────────────────────────────
CONTEXT_ASSESSOR_PROMPT = (
    "Analyze if the following medical query has SUFFICIENT context for diagnosis.\n"
    "Score from 0.0 (no context) to 1.0 (complete context).\n\n"
    "Consider:\n"
    "- Symptom duration and severity mentioned?\n"
    "- Associated symptoms described?\n"
    "- Patient demographics (age/gender) provided?\n"
    "- Medical history or medications mentioned?\n\n"
    "Query: {query}\n\n"
    "Respond with ONLY a number between 0.0 and 1.0:\n"
)

# ── Clarification Generator Prompt ──────────────────────────────────────
CLARIFICATION_PROMPT = (
    "Generate 1-2 specific follow-up questions to gather missing medical context.\n"
    "Be warm, empathetic, and use emojis.\n\n"
    "User query: {query}\n"
    "Missing context: {missing_info}\n\n"
    "Questions:\n"
)

ROUTER_PROMPT_TEMPLATE = (
    "Classify the following user message into exactly ONE category.\n"
    "Reply with a SINGLE word — MEDICAL, GREETING, or GENERAL.\n\n"
    "Rules:\n"
    "- MEDICAL: any health, disease, symptom, treatment, drug, anatomy, or clinical question.\n"
    "- GREETING: hello, hi, hey, good morning, thanks, bye, etc.\n"
    "- GENERAL: everything else (weather, sports, coding, etc.).\n\n"
    "User message: {query}\n\n"
    "Category:"
)
