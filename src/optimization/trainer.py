# src/optimization/trainer.py
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Configuration
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit" # Optimized 4-bit version for lower memory
OUTPUT_DIR = "data/storage/fine_tuned_model"
DATA_FILE = "data/training_data/raft_dataset.jsonl"

def train():
    print("🚀 Starting Fine-Tuning (LoRA)...")
    
    # 1. Load Dataset
    data = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    # Convert to HuggingFace Dataset format
    # We combine Input + Output for Causal Language Modeling
    formatted_data = [
        {"text": f"{item['instruction']}\n\n{item['input']}\n\nAnswer: {item['output']}"} 
        for item in data
    ]
    dataset = Dataset.from_list(formatted_data)

    # 2. Load Model & Tokenizer (4-bit quantization for consumer GPUs)
    # Note: On a CPU-only Mac/PC, you might need to remove "load_in_4bit=True" 
    # and use a smaller model like "TinyLlama" or "Phi-3".
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_4bit=True,
            device_map="auto"
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Tip: If you don't have a GPU, try running this part on Google Colab.")
        return

    # 3. Configure LoRA (Low-Rank Adaptation)
    # This tells the system: "Don't retrain the whole brain, just train these small adapters."
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Setup Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=50, # Short run for demonstration
        fp16=True if torch.cuda.is_available() else False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 5. Train
    print("🏋️‍♂️ Training started...")
    trainer.train()
    
    # 6. Save Adapter
    print(f"💾 Saving adapter to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()