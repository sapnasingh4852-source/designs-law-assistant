import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data and Storage paths
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "designs_manual.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

# Collection name for ChromaDB
COLLECTION_NAME = "designs_law_manual"

# Text Splitting Parameters
# 1000 characters (~150-200 words) captures self-contained legal rules and sub-sections.
# 200 characters overlap ensures boundary continuity across procedural steps and statutory references.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Parameters
RETRIEVER_K = 5
# Chroma uses L2/cosine distance; chunks with cosine similarity above threshold or reasonable relevance are considered.
SIMILARITY_THRESHOLD = 0.25

# Embedding Model Configuration
# sentence-transformers/all-MiniLM-L6-v2 is lightweight, fast, free, and runs locally with high semantic accuracy.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Configuration
# Supports Groq, Google Gemini, OpenAI, or local Ollama
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model Names
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Strict Grounded System Prompt
SYSTEM_PROMPT = """You are a specialized Designs Law Assistant for India.
You must answer the user's question ONLY using the provided context extracted from the Manual of Designs Practice & Procedure published by the Office of the Controller General of Patents, Designs & Trade Marks (CGPDTM), IP India.

CRITICAL INSTRUCTIONS:
1. Do NOT use any external or general legal knowledge.
2. Do NOT use information from the internet or other intellectual property domains (such as trademark or patent acts unless explicitly stated in the context).
3. Do NOT make assumptions, guess, or invent facts, procedures, fees, dates, or names.
4. If the provided context does not contain sufficient, direct, and reliable information to answer the question, you MUST respond EXACTLY:
Not found
5. When the answer is supported by the context:
   - Provide a concise, clear, and accurate answer based strictly on the text.
   - If the context mentions rules, forms, or timelines, include them accurately.
6. Do NOT fabricate citations. Use only the Chapter, Section, and PDF Page metadata provided with the context chunks.
"""
