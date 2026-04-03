import math
import os

import torch
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------
# Configuration
# --------------------
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = "teacher_correction_wikitext"
MODEL_NAME = "TheGardener/Qwen2.5-0.33B-base"
DATASET_NAME = "EleutherAI/wikitext_document_level"
DATASET_CONFIG = 'wikitext-103-raw-v1'
OUTPUT_DIR = "qwen2.5-0.33b-finetuned-wikitext-pytorch"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")

NUM_TRAIN_SAMPLES = 30000
MIN_TOKEN_LENGTH = 150
MAX_LENGTH = 256
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
GRADIENT_ACCUMULATION_STEPS = 8
SEED = 13
LOG_EVERY_N_STEPS = 10
WARMUP_STEPS = 100
ETA_MIN = 5e-7

# --------------------
# Initialize Weights & Biases
# --------------------
try:
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
except Exception as e:
    print(f"Could not login to WandB: {e}")

# --------------------
# Accelerator setup
# --------------------
accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    log_with="wandb"
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
    init_kwargs={"wandb": {"name": os.path.basename(MODEL_NAME)}}
)

# --------------------
# Load tokenizer and model
# --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    print("Set pad_token to eos_token (or added [PAD])")

# --------------------
# Load and preprocess dataset
# --------------------
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
print("Dataset loaded.")

shuffled = dataset['train'].shuffle(seed=SEED)

def get_token_length(examples):
    tok = tokenizer(examples['page'])
    return {'length': [len(ids) for ids in tok['input_ids']]}

with accelerator.main_process_first():
    ds_len = shuffled.map(
        get_token_length,
        batched=True,
        num_proc=os.cpu_count()
    )
filtered = ds_len.filter(lambda ex: ex['length'] > MIN_TOKEN_LENGTH, num_proc=os.cpu_count())
actual_samples = min(NUM_TRAIN_SAMPLES, len(filtered))
if actual_samples < NUM_TRAIN_SAMPLES:
    print(f"Only {actual_samples} samples > {MIN_TOKEN_LENGTH} tokens available.")

train_raw = filtered.select(range(actual_samples))
eval_raw = dataset['validation']


def tokenize_and_format(examples):
    tok = tokenizer(
        examples['page'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )
    tok['labels'] = tok['input_ids'].copy()
    return tok

with accelerator.main_process_first():
    train_ds = train_raw.map(tokenize_and_format, batched=True, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(tokenize_and_format, batched=True, remove_columns=eval_raw.column_names)

train_ds.set_format(type='torch')
eval_ds.set_format(type='torch')

train_loader = DataLoader(train_ds, batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_ds, batch_size=PER_DEVICE_EVAL_BATCH_SIZE)

# --------------------
# Optimizer & Schedulers
# --------------------
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_update_steps = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
total_steps = NUM_EPOCHS * num_update_steps

# Warmup scheduler
warmup_scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: float(step) / float(max(1, WARMUP_STEPS)) if step < WARMUP_STEPS else 1.0
)
# Cosine annealing scheduler
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps - WARMUP_STEPS,
    eta_min=ETA_MIN
)

# --------------------
# Prepare with Accelerator
# --------------------
model, optimizer, train_loader, eval_loader = accelerator.prepare(
    model, optimizer, train_loader, eval_loader
)

# --------------------
# Training Loop
# --------------------
best_eval_loss = float('inf')
progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process, desc="Training")
completed_steps = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # Scheduler step
                if completed_steps < WARMUP_STEPS:
                    warmup_scheduler.step()
                else:
                    cosine_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % LOG_EVERY_N_STEPS == 0:
                    lr = optimizer.param_groups[0]['lr']
                    accelerator.log({
                        "train_loss_step": loss.item(),
                        "learning_rate": lr,
                        "step": completed_steps
                    }, step=completed_steps)
                    progress_bar.set_postfix({"loss": loss.item(), "lr": lr})

        if completed_steps >= total_steps:
            break

    # Epoch training metrics
    avg_epoch_loss = total_loss.item() / len(train_loader)
    accelerator.log({"train_loss_epoch": avg_epoch_loss, "epoch": epoch+1})
    if accelerator.is_main_process:
        print(f"Epoch {epoch+1} train loss: {avg_epoch_loss:.4f}")

    # --- Evaluation Phase ---
    model.eval()
    eval_losses = []
    for batch in tqdm(eval_loader, desc="Evaluating", disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(**batch)
        eval_losses.append(accelerator.gather(outputs.loss.repeat(batch['input_ids'].size(0))))
    eval_losses = torch.cat(eval_losses)[: len(eval_ds)]
    eval_loss = torch.mean(eval_losses).item()
    perplexity = math.exp(eval_loss) if eval_loss < 100 else float('inf')

    accelerator.log({"eval_loss": eval_loss, "perplexity": perplexity, "epoch": epoch+1})
    if accelerator.is_main_process:
        print(f"Epoch {epoch+1} eval loss: {eval_loss:.4f}, ppl: {perplexity:.2f}")
        if eval_loss < best_eval_loss:
            print(f"New best model at epoch {epoch+1} with loss {eval_loss:.4f}, saving...")
            best_eval_loss = eval_loss
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)

# Save final model
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Final model and best model saved to {OUTPUT_DIR} and {BEST_MODEL_DIR}")
accelerator.end_training()
print("Training finished.")
