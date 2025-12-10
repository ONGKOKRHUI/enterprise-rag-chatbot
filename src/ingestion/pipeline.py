# src/ingestion/pipeline.py
import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.core.config import settings, init_settings

def run_ingestion():
    print(f"🔄 Starting Ingestion from {settings.DATA_DIR}...")
    init_settings()

    # 1. Check if data exists
    if not os.path.exists(settings.DATA_DIR):
        print(f"❌ No data found at {settings.DATA_DIR}. Please add PDFs.")
        return

    # 2. Load Documents (Parsing)
    # SimpleDirectoryReader handles PDFs, txt, images automatically
    documents = SimpleDirectoryReader(settings.DATA_DIR).load_data()
    print(f"📄 Loaded {len(documents)} document pages.")

    # 3. Setup Vector Database (ChromaDB)
    # We persist data to disk so we don't re-index every time
    db_client = chromadb.PersistentClient(path=settings.STORAGE_DIR)
    chroma_collection = db_client.get_or_create_collection("customer_support")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Create Index (Chunking + Embedding)
    # This step converts text -> numbers -> stores in Chroma
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    
    print("✅ Ingestion Complete. Vector Store saved.")
    return index