# src/engine/guardrails.py
from llama_index.core.llms import ChatMessage
from src.core.config import settings, init_settings

class Guardrails:
    def __init__(self):
        init_settings()
        # We access the LLM defined in settings
        from llama_index.core import Settings 
        self.llm = Settings.llm

    def validate_input(self, query: str) -> bool:
        """
        Check if the user query is safe and relevant.
        Returns True if safe, False if unsafe.
        """
        # A specific prompt to act as a moderator
        prompt = (
            f"You are a moderation system. Analyze the following user query: '{query}'. "
            "Determine if it contains harmful, illegal, or sexually explicit content. "
            "Also determine if it is vaguely related to customer support topics "
            "(technology, products, accounts, policies). "
            "Reply with 'SAFE' if it is acceptable, or 'UNSAFE' if it violates these rules. "
            "Do not provide explanations, just the single word."
        )
        
        response = self.llm.complete(prompt).text.strip().upper()
        return "SAFE" in response

    def validate_output(self, response: str) -> str:
        """
        Sanitize the bot's response.
        """
        # Simple keyword check for this example (fastest)
        forbidden_words = ["I don't know", "cannot answer"] # Example placeholders
        
        # You can add an LLM check here too if you want strict hallucination control
        # For now, we return the response as-is unless we flag it
        return response