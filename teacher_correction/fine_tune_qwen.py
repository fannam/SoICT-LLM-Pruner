import os
from datasets import load_dataset, Dataset 
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
import wandb
import math

WANDB_API_KEY = "" # 
WANDB_PROJECT = "teacher_correction_wikitext"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "EleutherAI/wikitext_document_level"
DATASET_CONFIG = 'wikitext-103-raw-v1'
OUTPUT_DIR = "qwen2.5-0.5b-finetuned-wikitext-pytorch"

NUM_TRAIN_SAMPLES = 30000 
MIN_TOKEN_LENGTH = 150 

MAX_LENGTH = 256
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
GRADIENT_ACCUMULATION_STEPS = 8
SEED = 13
LOG_EVERY_N_STEPS = 10

try:
    wandb.login(key=WANDB_API_KEY)
except Exception as e:
    print(f"Could not login to WandB: {e}. Check your API key.")

accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    log_with="wandb",
)

accelerator.init_trackers(
    project_name=WANDB_PROJECT,
    config={
        "model_name": MODEL_NAME,
        "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
        "num_train_samples": NUM_TRAIN_SAMPLES,
        "min_token_length": MIN_TOKEN_LENGTH,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "seed": SEED,
    },
    init_kwargs={"wandb": {"name": "Qwen2.5-1.5B_WikiText103_Finetune_LR2e-5_BS2x8_Epoch2"}}
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    print("Set pad_token to eos_token")

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
print("Dataset loaded.")

print("Processing and filtering training data...")

shuffled_train_dataset = dataset["train"].shuffle(seed=SEED)

def get_token_length(examples):
    tokenized = tokenizer(examples['page'])
    return {'length': [len(ids) for ids in tokenized['input_ids']]}

print("Calculating token lengths...")

with accelerator.main_process_first():
    dataset_with_length = shuffled_train_dataset.map(
        get_token_length,
        batched=True,
        num_proc=os.cpu_count() 
    )

print(f"Filtering for sequences longer than {MIN_TOKEN_LENGTH} tokens...")
filtered_dataset = dataset_with_length.filter(
    lambda example: example['length'] > MIN_TOKEN_LENGTH,
    num_proc=os.cpu_count() 
)

num_available = len(filtered_dataset)
actual_num_train_samples = min(NUM_TRAIN_SAMPLES, num_available)

if actual_num_train_samples < NUM_TRAIN_SAMPLES:
    print(f"WARNING: Found only {actual_num_train_samples} samples with length > {MIN_TOKEN_LENGTH}. Using these samples.")
else:
    print(f"Found {num_available} samples with length > {MIN_TOKEN_LENGTH}. Selecting {actual_num_train_samples}.")

selected_indices = range(actual_num_train_samples)
final_raw_train_subset = filtered_dataset.select(selected_indices)

eval_dataset_raw = dataset["validation"]
print("Raw datasets selected/filtered.")

def tokenize_and_format(examples):
    tokenized = tokenizer(
        examples['page'],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing final datasets with padding/truncation...")
with accelerator.main_process_first():
    train_dataset = final_raw_train_subset.map(
        tokenize_and_format,
        batched=True,
        remove_columns=final_raw_train_subset.column_names
    )
    eval_dataset = eval_dataset_raw.map(
        tokenize_and_format,
        batched=True,
        remove_columns=eval_dataset_raw.column_names
    )
print("Tokenization complete.")

train_dataset.set_format(type='torch')
eval_dataset.set_format(type='torch')
print("Dataset format set to torch.")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    shuffle=True 
)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=PER_DEVICE_EVAL_BATCH_SIZE
)
print("DataLoaders created.")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
num_training_steps = NUM_EPOCHS * num_update_steps_per_epoch

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=20,
    num_training_steps=num_training_steps,
)
print(f"Actual number of training samples used: {actual_num_train_samples}")
print(f"Total optimization steps: {num_training_steps}")

print("Preparing model, optimizer, dataloaders with Accelerator...")
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)
print("Preparation complete.")

total_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS
print("***** Running training *****")
print(f"  Num examples = {actual_num_train_samples}")
print(f"  Num Epochs = {NUM_EPOCHS}")
print(f"  Instantaneous batch size per device = {PER_DEVICE_TRAIN_BATCH_SIZE}")
print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
print(f"  Gradient Accumulation steps = {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Total optimization steps = {num_training_steps}")

progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process, desc="Training")
completed_steps = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % LOG_EVERY_N_STEPS == 0:
                    avg_loss = accelerator.gather(loss.repeat(PER_DEVICE_TRAIN_BATCH_SIZE)).mean()
                    step_loss = avg_loss.item() / GRADIENT_ACCUMULATION_STEPS
                    accelerator.log(
                        {
                            "train_loss_step": step_loss,
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "step": completed_steps
                        },
                        step=completed_steps
                    )
                    progress_bar.set_postfix({"loss": step_loss})

        if completed_steps >= num_training_steps:
            break

    num_samples_this_epoch = min(len(train_dataset), completed_steps * total_batch_size / accelerator.num_processes ) # Estimate samples processed
    avg_epoch_loss = accelerator.gather(total_loss).sum().item() / (len(train_dataloader) * accelerator.num_processes * PER_DEVICE_TRAIN_BATCH_SIZE) # Approx based on dataloader length per process

    accelerator.log({"train_loss_epoch": avg_epoch_loss, "epoch": epoch + 1})
    if accelerator.is_main_process:
        print(f"Epoch {epoch+1}: Avg Train Loss: {avg_epoch_loss:.4f}")

    # --- Evaluation Phase ---
    model.eval()
    eval_losses = []
    eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process, desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        batch_size = batch['input_ids'].shape[0]
        eval_losses.append(accelerator.gather(loss.repeat(batch_size)))
        eval_progress_bar.update(1)

    eval_losses = torch.cat(eval_losses)
    eval_losses = eval_losses[:len(eval_dataset)]
    try:
        eval_loss = torch.mean(eval_losses).item()
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    accelerator.log(
        {
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            "epoch": epoch + 1
        }
    )
    if accelerator.is_main_process:
        print(f"Epoch {epoch+1}: Eval Loss: {eval_loss:.4f}, Perplexity: {perplexity:.4f}")
    # --- End Evaluation ---

# Save the trained model
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")

# Finish Weights & Biases run
accelerator.end_training()
print("Training Finished.")