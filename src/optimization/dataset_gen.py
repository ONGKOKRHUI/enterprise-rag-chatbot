# src/optimization/dataset_gen.py
import json
import random
import os
from typing import List, Dict
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from src.core.config import init_settings, settings
from llama_index.core import Settings

class RaftDatasetGenerator:
    def __init__(self):
        init_settings()
        self.llm = Settings.llm
        self.output_file = "data/training_data/raft_dataset.jsonl"
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def load_chunks(self) -> List[str]:
        """Loads documents and splits them into chunks."""
        print("📄 Loading documents for dataset generation...")
        documents = SimpleDirectoryReader(settings.DATA_DIR).load_data()
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        return [node.get_content() for node in nodes]

    def generate_question_answer(self, chunk: str):
        """Uses LLM to create a synthetic Q&A pair from a text chunk."""
        prompt = (
            f"Context: {chunk}\n\n"
            "Task: Generate a specific question that can be answered using ONLY the context above. "
            "Then, provide the correct answer based strictly on that context.\n"
            "Format your response exactly like this:\n"
            "QUESTION: [Your Question Here]\n"
            "ANSWER: [Your Answer Here]"
        )
        response = self.llm.complete(prompt).text.strip()
        
        # Basic parsing (Robustness depends on LLM following instructions)
        try:
            parts = response.split("ANSWER:")
            question = parts[0].replace("QUESTION:", "").strip()
            answer = parts[1].strip()
            return question, answer
        except Exception:
            return None, None

    def create_raft_sample(self, correct_chunk: str, all_chunks: List[str], num_distractors=3) -> Dict:
        """
        Creates a single RAFT training example:
        Question + Context (1 Right + 3 Wrong Documents) -> Answer
        """
        question, answer = self.generate_question_answer(correct_chunk)
        if not question:
            return None

        # Pick random "distractor" chunks (noise)
        candidate_chunks = [c for c in all_chunks if c != correct_chunk]
        distractors = random.sample(
            candidate_chunks,
            min(num_distractors, len(candidate_chunks))
        )

        # Combine and shuffle context
        context_docs = distractors + [correct_chunk]
        random.shuffle(context_docs)
        
        # Create the formatted prompt the model will see during training
        context_str = "\n\n---\n\n".join(context_docs)
        
        system_prompt = (
            "You are a helpful customer support agent. "
            "Read the following documents and answer the user's question. "
            "Ignore documents that are not relevant."
        )
        
        return {
            "instruction": system_prompt,
            "input": f"Documents:\n{context_str}\n\nQuestion: {question}",
            "output": answer
        }

    def run(self, num_samples=50):
        chunks = self.load_chunks()
        dataset = []
        
        print(f"🧠 Generating {num_samples} RAFT training samples using {settings.LLM_MODEL}...")
        
        # We assume we have enough chunks. If small doc, we loop or sample carefully.
        # For this demo, we iterate through chunks up to num_samples
        count = 0
        for chunk in chunks:
            if count >= num_samples:
                break
                
            sample = self.create_raft_sample(chunk, chunks)
            if sample:
                dataset.append(sample)
                count += 1
                print(f"   [{count}/{num_samples}] Generated sample...")

        # Save to JSONL (standard format for Fine-Tuning)
        with open(self.output_file, 'w') as f:
            for entry in dataset:
                f.write(json.dumps(entry) + '\n')
        
        print(f"✅ Dataset saved to {self.output_file}")

if __name__ == "__main__":
    gen = RaftDatasetGenerator()
    gen.run(num_samples=10) # Start small to test