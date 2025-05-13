import os
import math
import torch
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
import wandb
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
        config=None
    ):
        """
        Initialize the TeacherCorrection trainer.
        
        Args:
            model: AutoModelForCausalLM model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            optimizer: Optimizer (optional, will create AdamW if None)
            scheduler: Learning rate scheduler (optional)
            accelerator: Accelerator instance (optional, will create if None)
            config: Dictionary containing training configuration (optional)
        """
        self.config = config or {}
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if accelerator is None:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 8),
                log_with="wandb"
            )
        else:
            self.accelerator = accelerator
            
        if optimizer is None:
            self.optimizer = AdamW(
                model.parameters(),
                lr=self.config.get('learning_rate', 2e-5)
            )
        else:
            self.optimizer = optimizer
            
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.config.get('gradient_accumulation_steps', 8))
        self.num_training_steps = self.config.get('num_epochs', 5) * num_update_steps_per_epoch
        
        if scheduler is None:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=100,
                num_training_steps=self.num_training_steps
            )
        else:
            self.scheduler = scheduler
            
        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )
        
        if self.config.get('wandb_project'):
            self.accelerator.init_trackers(
                project_name=self.config['wandb_project'],
                config=self.config,
                init_kwargs={"wandb": {"name": self.config.get('wandb_run_name', 'teacher_correction')}}
            )

    def train(self):
        """Run the training loop."""
        num_epochs = self.config.get('num_epochs', 5)
        log_every_n_steps = self.config.get('log_every_n_steps', 10)
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 8)
        
        progress_bar = tqdm(
            range(self.num_training_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training"
        )
        completed_steps = 0

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0

            for step, batch in enumerate(self.train_loader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                        if completed_steps % log_every_n_steps == 0:
                            avg_loss = self.accelerator.gather(loss.repeat(self.config.get('per_device_train_batch_size', 4))).mean()
                            step_loss = avg_loss.item() / gradient_accumulation_steps
                            self.accelerator.log(
                                {
                                    "train_loss_step": step_loss,
                                    "learning_rate": self.scheduler.get_last_lr()[0],
                                    "step": completed_steps
                                },
                                step=completed_steps
                            )
                            progress_bar.set_postfix({"loss": step_loss})

                if completed_steps >= self.num_training_steps:
                    break

            num_samples_this_epoch = min(
                len(self.train_loader.dataset),
                completed_steps * self.config.get('per_device_train_batch_size', 4) * gradient_accumulation_steps / self.accelerator.num_processes
            )
            avg_epoch_loss = self.accelerator.gather(total_loss).sum().item() / (len(self.train_loader) * self.accelerator.num_processes * self.config.get('per_device_train_batch_size', 4))

            self.accelerator.log({"train_loss_epoch": avg_epoch_loss, "epoch": epoch + 1})
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}: Avg Train Loss: {avg_epoch_loss:.4f}")

            eval_metrics = self.evaluate()
            self.accelerator.log(
                {
                    "eval_loss": eval_metrics['eval_loss'],
                    "perplexity": eval_metrics['perplexity'],
                    "epoch": epoch + 1
                }
            )
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}: Eval Loss: {eval_metrics['eval_loss']:.4f}, Perplexity: {eval_metrics['perplexity']:.4f}")

        self.accelerator.end_training()
        print("Training Finished.")

    def evaluate(self):
        """Run evaluation on the validation set."""
        self.model.eval()
        eval_losses = []
        eval_progress_bar = tqdm(
            range(len(self.val_loader)),
            disable=not self.accelerator.is_local_main_process,
            desc="Evaluating"
        )

        for step, batch in enumerate(self.val_loader):
            with torch.no_grad():
                outputs = self.model(**batch)
            loss = outputs.loss
            batch_size = batch['input_ids'].shape[0]
            eval_losses.append(self.accelerator.gather(loss.repeat(batch_size)))
            eval_progress_bar.update(1)

        eval_losses = torch.cat(eval_losses)
        eval_losses = eval_losses[:len(self.val_loader.dataset)]
        
        try:
            eval_loss = torch.mean(eval_losses).item()
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        return {
            'eval_loss': eval_loss,
            'perplexity': perplexity
        }

    def save_model(self, output_dir):
        """Save the model and tokenizer."""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(output_dir)
            if hasattr(self.model, 'tokenizer'):
                self.model.tokenizer.save_pretrained(output_dir)
            print(f"Model and tokenizer saved to {output_dir}") 