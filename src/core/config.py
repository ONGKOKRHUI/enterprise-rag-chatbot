# src/core/config.py
import os
from pydantic_settings import BaseSettings
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class AppSettings(BaseSettings):
    # Paths
    DATA_DIR: str = "data/raw"
    STORAGE_DIR: str = "data/storage"
    
    # Model Configs
    LLM_MODEL: str = "llama3"  # Ensure you ran `ollama pull llama3`
    EMBED_MODEL: str = "BAAI/bge-small-en-v1.5" # Excellent local embedding model
    
    # RAG Parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

settings = AppSettings()

def init_settings():
    """Initialize global LlamaIndex settings"""
    # 1. Setup LLM (Local Llama3)
    Settings.llm = Ollama(model=settings.LLM_MODEL, request_timeout=300.0)
    
    # 2. Setup Embedding Model (Local HuggingFace)
    # This runs locally on CPU/GPU, no API key needed.
    Settings.embed_model = HuggingFaceEmbedding(model_name=settings.EMBED_MODEL)
    
    # 3. Text Splitter Settings
    Settings.chunk_size = settings.CHUNK_SIZE
    Settings.chunk_overlap = settings.CHUNK_OVERLAP