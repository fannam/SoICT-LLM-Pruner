from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from soict_llm_pruner.distillation import TeacherCorrection


def parse_args():
    parser = argparse.ArgumentParser(description="Teacher-correction recovery with Accelerate.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="EleutherAI/wikitext_document_level")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--text-column", default="page")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--num-train-samples", type=int, default=30000)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--num-warmup-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default="teacher_correction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    train_raw = dataset[args.train_split].select(range(min(args.num_train_samples, len(dataset[args.train_split]))))
    eval_raw = dataset[args.eval_split]

    def tokenize(examples):
        batch = tokenizer(
            examples[args.text_column],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        batch["labels"] = [
            [token if mask else -100 for token, mask in zip(input_ids, attention_mask)]
            for input_ids, attention_mask in zip(batch["input_ids"], batch["attention_mask"])
        ]
        return batch

    train_ds = train_raw.map(tokenize, batched=True, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(tokenize, batched=True, remove_columns=eval_raw.column_names)
    train_ds.set_format(type="torch")
    eval_ds.set_format(type="torch")

    train_loader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.per_device_eval_batch_size)

    config = {
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "num_warmup_steps": args.num_warmup_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": args.max_grad_norm,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
    }
    if os.getenv("WANDB_API_KEY") is None and args.wandb_project:
        print("WANDB_API_KEY is not set; continuing without explicit login.")

    trainer = TeacherCorrection(
        model=model,
        train_loader=train_loader,
        val_loader=eval_loader,
        config=config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
