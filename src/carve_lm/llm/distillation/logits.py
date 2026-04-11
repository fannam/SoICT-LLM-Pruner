from __future__ import annotations

import torch

from ._common import (
    get_output_attr,
    masked_logits_loss,
    maybe_finish_wandb,
    maybe_init_wandb,
    maybe_log_wandb,
    plot_history,
    prepare_causal_lm_batch,
    resolve_device,
    zero_grad,
)


class LogitsDistiller:
    def __init__(
        self,
        teacher_model,
        student_model,
        tokenizer=None,
        optimizer=None,
        scheduler=None,
        use_wandb: bool = False,
        wandb_key: str | None = None,
        project_name: str | None = None,
        run_name: str | None = None,
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_wandb = use_wandb
        self.wandb_key = wandb_key
        self.project_name = project_name
        self.run_name = run_name
        self.history = {"train_loss": [], "val_loss": []}
        self.last_plot = None

    def logits_loss(
        self,
        student_logits,
        teacher_logits,
        labels,
        loss_mask,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        return masked_logits_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            loss_mask=loss_mask,
            temperature=temperature,
            alpha=alpha,
        )

    def _forward_teacher(self, batch, device_teacher):
        teacher_batch = {
            "input_ids": batch["input_ids"].to(device_teacher),
            "attention_mask": batch.get("attention_mask", None),
        }
        if teacher_batch["attention_mask"] is not None:
            teacher_batch["attention_mask"] = teacher_batch["attention_mask"].to(device_teacher)
        return self.teacher_model(**teacher_batch, output_hidden_states=False)

    def _forward_student(self, input_ids, attention_mask):
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        return self.student_model(**kwargs)

    def _run_validation(
        self,
        val_loader,
        *,
        device_teacher,
        device_student,
        temperature: float,
        alpha: float,
    ) -> float | None:
        if val_loader is None:
            return None

        self.student_model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels, loss_mask = prepare_causal_lm_batch(batch, device_student)
                teacher_outputs = self._forward_teacher(batch, device_teacher)
                teacher_logits = get_output_attr(teacher_outputs, "logits")[..., :-1, :].to(device_student)
                student_outputs = self._forward_student(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = get_output_attr(student_outputs, "logits")[..., :-1, :]

                loss_val = self.logits_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    loss_mask,
                    temperature=temperature,
                    alpha=alpha,
                )
                valid_tokens = int(loss_mask.sum().item())
                total_loss += loss_val.item() * valid_tokens
                total_tokens += valid_tokens
        self.student_model.train()

        if total_tokens == 0:
            raise ValueError("Validation loader did not contain any valid target tokens.")
        return total_loss / total_tokens

    def distill(
        self,
        train_loader,
        val_loader=None,
        device_teacher=None,
        device_student=None,
        epochs: int = 1,
        grad_accumulation_steps: int = 8,
        alpha: float = 0.1,
        temperature: float = 2.0,
        plot: bool = False,
        show_plot: bool = False,
    ):
        if self.optimizer is None:
            raise ValueError("`optimizer` is required for distillation.")
        if grad_accumulation_steps <= 0:
            raise ValueError("`grad_accumulation_steps` must be > 0.")
        if epochs <= 0:
            raise ValueError("`epochs` must be > 0.")

        device_teacher = resolve_device(self.teacher_model, device_teacher)
        device_student = resolve_device(self.student_model, device_student)

        self.teacher_model.to(device_teacher)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.student_model.to(device_student)
        self.student_model.train()

        run = maybe_init_wandb(
            enabled=self.use_wandb,
            project_name=self.project_name,
            run_name=self.run_name,
            wandb_key=self.wandb_key,
        )

        self.history = {"train_loss": [], "val_loss": []}
        global_step = 0
        zero_grad(self.optimizer)

        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_tokens = 0
                micro_batches = 0

                for batch in train_loader:
                    input_ids, attention_mask, labels, loss_mask = prepare_causal_lm_batch(batch, device_student)

                    with torch.no_grad():
                        teacher_outputs = self._forward_teacher(batch, device_teacher)
                        teacher_logits = get_output_attr(teacher_outputs, "logits")[..., :-1, :].to(device_student)

                    student_outputs = self._forward_student(input_ids=input_ids, attention_mask=attention_mask)
                    student_logits = get_output_attr(student_outputs, "logits")[..., :-1, :]

                    loss = self.logits_loss(
                        student_logits,
                        teacher_logits,
                        labels,
                        loss_mask,
                        temperature=temperature,
                        alpha=alpha,
                    )
                    (loss / grad_accumulation_steps).backward()

                    valid_tokens = int(loss_mask.sum().item())
                    epoch_loss += loss.item() * valid_tokens
                    epoch_tokens += valid_tokens
                    micro_batches += 1

                    if micro_batches == grad_accumulation_steps:
                        self.optimizer.step()
                        if self.scheduler is not None:
                            self.scheduler.step()
                        zero_grad(self.optimizer)
                        maybe_log_wandb(
                            run,
                            {
                                "train_loss": loss.item(),
                                "epoch": epoch,
                                "step": global_step,
                            },
                        )
                        global_step += 1
                        micro_batches = 0

                if micro_batches:
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    zero_grad(self.optimizer)
                    global_step += 1

                if epoch_tokens == 0:
                    raise ValueError("Training loader did not contain any valid target tokens.")
                avg_train = epoch_loss / epoch_tokens
                self.history["train_loss"].append(avg_train)

                avg_val = self._run_validation(
                    val_loader,
                    device_teacher=device_teacher,
                    device_student=device_student,
                    temperature=temperature,
                    alpha=alpha,
                )
                if avg_val is not None:
                    self.history["val_loss"].append(avg_val)
                    maybe_log_wandb(run, {"val_loss": avg_val, "epoch": epoch, "step": global_step})
        finally:
            maybe_finish_wandb(run)

        if plot and self.history["val_loss"]:
            self.last_plot = plot_history(
                self.history["val_loss"],
                title="Validation Loss Over Epochs",
                ylabel="Validation Loss",
                show=show_plot,
            )
        else:
            self.last_plot = None

        return list(self.history["val_loss"])
