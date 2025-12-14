# main.py
import sys
from src.ingestion.pipeline import run_ingestion
from src.engine.chat import get_chat_engine, get_chat_engine_advanced
 
def main():
    print("--- Customer Support RAG Bot (Phase 1) ---")
    print("1. Ingest Data (Process PDFs)")
    print("2. Chat with Bot")
    
    choice = input("Select an option (1 or 2): ")
    
    if choice == "1":
        # Run Phase A: Indexing
        run_ingestion()
        
    elif choice == "2":
        # Run Phase B: Retrieval & Generation
        try:
            engine = get_chat_engine()
            print("\n🤖 Bot is ready! Type 'exit' to quit.\n")
            
            while True:
                user_input = input("You: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                
                # The "Retrieval" and "Generation" happen here
                response = engine.chat(user_input)
                print(f"Bot: {response}\n")
                
        except Exception as e:
            print(f"Error: {e}")
            print("Did you run option 1 (Ingestion) first?")
            
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()