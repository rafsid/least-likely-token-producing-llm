import os
import torch
import pickle

# Import unsloth components
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
# Import the unsloth utility module to patch its offload function
import unsloth.models._utils as unsloth_utils

# === Monkey Patch Start ===
def offload_to_disk_patch(W, model, name, temporary_location):
    os.makedirs(temporary_location, exist_ok=True)
    filename = os.path.join(temporary_location, f"{name}.pt")
    # If W is an Embedding module, use its weight tensor
    if hasattr(W, "weight"):
        W = W.weight
    torch.save(W, filename, pickle_module=pickle, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    # Explicitly pass weights_only=False to avoid unpickling issues
    offloaded_W = torch.load(filename, map_location="cpu", mmap=True, weights_only=False)
    offloaded_W._offloaded_file_location = filename
    return offloaded_W

# Apply the patch so unsloth uses the corrected offload function
unsloth_utils.offload_to_disk = offload_to_disk_patch
# === Monkey Patch End ===

from datasets import load_dataset
from transformers import TextStreamer

max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load the base model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Wrap the model with PEFT (LoRA) configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    loftq_config=None,
)

# Define a prompt format for Wikipedia articles
wikipedia_prompt = """Wikipedia Article
### Title: {}

### Article:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    titles = examples["title"]
    texts = examples["text"]
    outputs = []
    for title, text in zip(titles, texts):
        text = wikipedia_prompt.format(title, text) + EOS_TOKEN
        outputs.append(text)
    return {"text": outputs}

# Load and prepare a subset of the Wikipedia dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
dataset = dataset.train_test_split(train_size=0.0001)["train"]
dataset = dataset.map(formatting_prompts_func, batched=True)

# Define the unlikelihood loss function
def unlikelihood_loss(logits, labels):
    # logits: [batch_size, seq_length, vocab_size]
    # labels: [batch_size, seq_length] (with -100 for ignored tokens)
    valid_mask = (labels != -100)
    # Compute probabilities
    probs = torch.softmax(logits, dim=-1)
    # Clamp labels to avoid negative indices (ignored ones will be masked later)
    clamped_labels = labels.clamp(min=0)
    # Create one-hot encoding for valid labels
    one_hot = torch.zeros_like(probs)
    one_hot = one_hot.scatter(-1, clamped_labels.unsqueeze(-1), 1.0)
    # Zero out positions corresponding to ignored labels (-100)
    one_hot = one_hot * valid_mask.unsqueeze(-1)
    
    # Set probabilities for the true tokens to 1.0 so they do not affect the min computation
    masked_probs = probs.masked_fill(one_hot.bool(), 1.0)
    # Get the minimum probability across the vocabulary for each position
    min_probs = torch.min(masked_probs, dim=-1)[0]
    # Compute loss only on valid positions, add epsilon for numerical stability
    loss = -torch.log(min_probs + 1e-8)
    loss = (loss * valid_mask).sum() / valid_mask.sum()
    return loss


# Custom trainer that uses unlikelihood loss
class UnlikelihoodTrainer(UnslothTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss = unlikelihood_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Train on the Wikipedia dataset
trainer = UnlikelihoodTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=120,
        warmup_steps=10,
        learning_rate=5e-5,
        embedding_learning_rate=1e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to=["tensorboard"]
    ),
)

trainer_stats = trainer.train()

# Define a prompt format for Alpaca-style instruction following
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

alpaca_dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")

def formatting_prompts_func_alpaca(conversations):
    texts = []
    for instruction, output in zip(conversations["instruction"], conversations["output"]):
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

alpaca_dataset = alpaca_dataset.map(formatting_prompts_func_alpaca, batched=True)

# Train on the Alpaca dataset
trainer = UnlikelihoodTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=alpaca_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=8,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=120,
        warmup_steps=10,
        learning_rate=5e-5,
        embedding_learning_rate=1e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.00,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to=["tensorboard"]
    ),
)

trainer_stats = trainer.train()

# Finalize model for inference and save both model and tokenizer
FastLanguageModel.for_inference(model)

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
