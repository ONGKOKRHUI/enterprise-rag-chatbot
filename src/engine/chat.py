# src/engine/chat.py
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.core.config import settings, init_settings

def get_chat_engine():
    init_settings()
    
    # 1. Connect to existing Vector DB
    db_client = chromadb.PersistentClient(path=settings.STORAGE_DIR)
    chroma_collection = db_client.get_or_create_collection("customer_support")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 2. Load Index from Vector Store
    index = VectorStoreIndex.from_vector_store(
        vector_store,
    )
    
    # 3. Create Chat Engine
    # "context" mode means it will retrieve docs first, then answer.
    # system_prompt sets the "Role" from your diagram.
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        system_prompt=(
            "You are a professional Customer Support Agent. "
            "Answer queries strictly based on the provided context. "
            "If the answer is not in the context, say 'I don't have that information'."
            "Keep answers concise and polite."
        )
    )
    
    return chat_engine