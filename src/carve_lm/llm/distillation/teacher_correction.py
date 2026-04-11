from __future__ import annotations

import math
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup


class TeacherCorrection:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer=None,
        scheduler=None,
        accelerator=None,
        config=None,
        tokenizer=None,
    ):
        self.config = config or {}
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self._tracking_enabled = bool(self.config.get("wandb_project"))

        if accelerator is None:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 8),
                log_with="wandb" if self._tracking_enabled else None,
            )
        else:
            self.accelerator = accelerator

        if optimizer is None:
            self.optimizer = AdamW(
                model.parameters(),
                lr=self.config.get("learning_rate", 2e-5),
            )
        else:
            self.optimizer = optimizer

        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.config.get("gradient_accumulation_steps", 8))
        self.num_training_steps = self.config.get("num_epochs", 5) * num_update_steps_per_epoch

        if scheduler is None:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.get("num_warmup_steps", 100),
                num_training_steps=self.num_training_steps,
            )
        else:
            self.scheduler = scheduler

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        )

        if self._tracking_enabled:
            self.accelerator.init_trackers(
                project_name=self.config["wandb_project"],
                config=self.config,
                init_kwargs={"wandb": {"name": self.config.get("wandb_run_name", "teacher_correction")}},
            )

        self.history = {"train_loss_epoch": [], "eval_loss": [], "perplexity": []}

    def _log(self, payload: dict[str, float], *, step: int | None = None) -> None:
        if self._tracking_enabled:
            self.accelerator.log(payload, step=step)

    def train(self):
        num_epochs = self.config.get("num_epochs", 5)
        log_every_n_steps = self.config.get("log_every_n_steps", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        progress_bar = tqdm(
            range(self.num_training_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training",
        )
        completed_steps = 0

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss_sum = None
            epoch_micro_steps = 0
            accumulation_loss = None
            accumulation_micro_steps = 0

            for batch in self.train_loader:
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    detached_loss = loss.detach().float()
                    epoch_loss_sum = detached_loss if epoch_loss_sum is None else epoch_loss_sum + detached_loss
                    accumulation_loss = (
                        detached_loss if accumulation_loss is None else accumulation_loss + detached_loss
                    )
                    epoch_micro_steps += 1
                    accumulation_micro_steps += 1
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                        step_loss_tensor = (accumulation_loss / accumulation_micro_steps).reshape(1)
                        step_loss = self.accelerator.gather(step_loss_tensor).mean().item()
                        accumulation_loss = None
                        accumulation_micro_steps = 0

                        if completed_steps % log_every_n_steps == 0:
                            self._log(
                                {
                                    "train_loss_step": step_loss,
                                    "learning_rate": self.scheduler.get_last_lr()[0],
                                    "step": float(completed_steps),
                                },
                                step=completed_steps,
                            )
                            progress_bar.set_postfix({"loss": step_loss})

                if completed_steps >= self.num_training_steps:
                    break

            if epoch_micro_steps == 0:
                raise ValueError("Training loader is empty.")

            avg_epoch_loss_tensor = (epoch_loss_sum / epoch_micro_steps).reshape(1)
            avg_epoch_loss = self.accelerator.gather(avg_epoch_loss_tensor).mean().item()
            self.history["train_loss_epoch"].append(avg_epoch_loss)
            self._log({"train_loss_epoch": avg_epoch_loss, "epoch": float(epoch + 1)}, step=completed_steps)
            self.accelerator.print(f"Epoch {epoch + 1}: Avg Train Loss: {avg_epoch_loss:.4f}")

            eval_metrics = self.evaluate()
            self.history["eval_loss"].append(eval_metrics["eval_loss"])
            self.history["perplexity"].append(eval_metrics["perplexity"])
            self._log(
                {
                    "eval_loss": eval_metrics["eval_loss"],
                    "perplexity": eval_metrics["perplexity"],
                    "epoch": float(epoch + 1),
                },
                step=completed_steps,
            )
            self.accelerator.print(
                f"Epoch {epoch + 1}: Eval Loss: {eval_metrics['eval_loss']:.4f}, "
                f"Perplexity: {eval_metrics['perplexity']:.4f}"
            )

        self.accelerator.end_training()
        self.accelerator.print("Training Finished.")
        return self.history

    def evaluate(self):
        self.model.eval()
        eval_losses = []
        eval_progress_bar = tqdm(
            range(len(self.val_loader)),
            disable=not self.accelerator.is_local_main_process,
            desc="Evaluating",
        )

        for batch in self.val_loader:
            with torch.no_grad():
                outputs = self.model(**batch)
            eval_losses.append(self.accelerator.gather(outputs.loss.detach().float().reshape(1)))
            eval_progress_bar.update(1)

        if not eval_losses:
            raise ValueError("Validation loader is empty.")

        eval_loss = torch.cat(eval_losses).mean().item()
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        return {
            "eval_loss": eval_loss,
            "perplexity": perplexity,
        }

    def save_model(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            save_kwargs = {
                "is_main_process": self.accelerator.is_main_process,
            }
            if hasattr(self.accelerator, "save"):
                save_kwargs["save_function"] = self.accelerator.save
            if hasattr(self.accelerator, "get_state_dict"):
                save_kwargs["state_dict"] = self.accelerator.get_state_dict(self.model)
            unwrapped_model.save_pretrained(output_path, **save_kwargs)

            tokenizer = self.tokenizer
            if tokenizer is None and hasattr(unwrapped_model, "tokenizer"):
                tokenizer = unwrapped_model.tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(output_path)
            self.accelerator.print(f"Model and tokenizer saved to {output_path}")
