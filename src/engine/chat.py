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


# src/engine/chat.py
import chromadb
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.core.config import settings, init_settings

# Import our new modules
from src.engine.guardrails import Guardrails
from src.retrieval.expansion import QueryExpansion

class AdvancedChatEngine:
    def __init__(self):
        init_settings()
        
        # 1. Initialize Modules
        self.guardrails = Guardrails()
        self.expander = QueryExpansion()
        
        # 2. Connect to DB
        db_client = chromadb.PersistentClient(path=settings.STORAGE_DIR)
        chroma_collection = db_client.get_or_create_collection("customer_support")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # 3. Setup Low-Level Components
        # We need the 'retriever' to search and 'synthesizer' to generate answers
        self.retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=4, # Retrieve top 4 chunks
        )
        
        # The Response Synthesizer handles the "Using this context, answer..." part
        self.synthesizer = get_response_synthesizer(response_mode="compact")
 
    def chat(self, user_query: str):
        # --- STEP 1: INPUT GUARDRAILS ---
        print("🛡️ Checking guardrails...")
        if not self.guardrails.validate_input(user_query):
            return "I cannot answer that query as it violates our safety or topic guidelines."

        # --- STEP 2: QUERY EXPANSION ---
        print("🧠 Expanding query...")
        queries = self.expander.generate_variations(user_query)
        print(f"   Searching for: {queries}")

        # --- STEP 3: MULTI-QUERY RETRIEVAL ---
        # We search for ALL variations and combine the results
        all_nodes = []
        for q in queries:
            nodes = self.retriever.retrieve(q)
            all_nodes.extend(nodes)
        
        # Deduplicate nodes (in case different queries found the same chunk)
        unique_nodes = {n.node.node_id: n for n in all_nodes}.values()
        
        # --- STEP 4: GENERATION ---
        print("🤖 Generating response...")
        response_obj = self.synthesizer.synthesize(user_query, nodes=list(unique_nodes))
        response_text = str(response_obj)

        # --- STEP 5: OUTPUT GUARDRAILS ---
        final_response = self.guardrails.validate_output(response_text)
        
        return final_response

# Helper function for main.py to call
def get_chat_engine_advanced():
    return AdvancedChatEngine()