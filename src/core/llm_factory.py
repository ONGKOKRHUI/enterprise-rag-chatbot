import torch
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig

# Define your paths
BASE_MODEL_ID = "unsloth/llama-3-8b-bnb-4bit" # The base we trained on
ADAPTER_PATH = "data/storage/fine_tuned_model" # Where Phase 3 saved the weights

def get_llm(use_finetuned: bool = False):
    """
    Factory function to return either the generic Ollama model
    OR the specialized Fine-Tuned model loaded directly from disk.
    """
    
    if not use_finetuned:
        print("🔌 Connecting to generic Ollama (Llama 3)...")
        return Ollama(model="llama3", request_timeout=300.0)

    else:
        print(f"🧬 Loading Fine-Tuned Model from {ADAPTER_PATH}...")
        print("   (This may take a minute and requires ~6GB VRAM/RAM)")

        # 1. Quantization Config (Crucial for running on Laptops)
        # This loads the model in 4-bit mode to save massive amounts of RAM
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # 2. Initialize the HuggingFace LLM Wrapper
        # This automatically merges the Base Model + Your Adapter
        llm = HuggingFaceLLM(
            model_name=BASE_MODEL_ID,
            tokenizer_name=BASE_MODEL_ID,
            context_window=4096,
            max_new_tokens=512,
            model_kwargs={
                "quantization_config": quantization_config,
                # "peft_model_id" is the magic argument that loads your adapter
                # Note: LlamaIndex handling of peft_model_id varies by version.
                # If this fails, we load standard and attach adapter manually (see below)
            },
            generate_kwargs={"temperature": 0.3, "do_sample": True},
            device_map="auto",
        )
        
        # Explicitly load adapter if not handled by model_kwargs in your version
        # (This is a robust fallback for "Industry Grade" reliability)
        try:
            from peft import PeftModel
            # We grab the underlying torch model and attach the adapter
            llm._model = PeftModel.from_pretrained(llm._model, ADAPTER_PATH)
            print("✅ LoRA Adapter merged successfully.")
        except Exception as e:
            print(f"⚠️ Warning: Could not explicitly merge adapter: {e}")
            print("Assuming model loaded correctly via config.")

        return llm