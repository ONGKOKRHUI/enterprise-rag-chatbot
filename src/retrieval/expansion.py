# src/retrieval/expansion.py
from typing import List
from src.core.config import init_settings

class QueryExpansion:
    def __init__(self):
        init_settings()
        from llama_index.core import Settings
        self.llm = Settings.llm

    def generate_variations(self, query: str) -> List[str]:
        """
        Generates 3 variations of the user query to improve search coverage.
        """
        prompt = (
            f"You are a helpful AI assistant. The user provided this query: '{query}'. "
            "Generate 3 different search queries that mean the same thing but use "
            "different keywords or technical terms found in product manuals. "
            "Return ONLY the queries, separated by newlines."
        )
        
        response = self.llm.complete(prompt).text.strip()
        
        # Parse the result into a list
        variations = [q.strip() for q in response.split('\n') if q.strip()]
        
        # Add the original query to the list so we don't lose it
        variations.append(query)
        
        # Remove duplicates
        return list(set(variations))