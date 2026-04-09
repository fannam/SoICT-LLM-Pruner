from __future__ import annotations

import torch

from ._common import (
    get_output_attr,
    masked_feature_loss,
    masked_logits_loss,
    maybe_finish_wandb,
    maybe_get_output_attr,
    maybe_init_wandb,
    maybe_log_wandb,
    plot_history,
    prepare_causal_lm_batch,
    resolve_device,
    sync_scheduler_added_param_group_lrs,
    zero_grad,
)
from .wrappers import DistilModel


class HybridDistiller:
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
        self.history = self._initialize_history()
        self.distill_model = None
        self.teacher_kept_layers = None
        self.last_plot = None

    def logits_loss(self, student_logits, teacher_logits, labels, loss_mask, temperature=2.0, alpha=0.5):
        return masked_logits_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            loss_mask=loss_mask,
            temperature=temperature,
            alpha=alpha,
        )

    def feature_loss_selected_layers(self, student_feats, teacher_feats, projectors, mask):
        return masked_feature_loss(student_feats, teacher_feats, projectors, mask)

    def _tracked_metric_keys(self) -> tuple[str, ...]:
        return ()

    def _initialize_history(self) -> dict[str, list[float]]:
        history = {"train_loss": [], "val_loss": []}
        for metric_key in self._tracked_metric_keys():
            history["train_{}".format(metric_key)] = []
            history["val_{}".format(metric_key)] = []
        return history

    def _resolve_teacher_kept_layers(self, block_layers_to_prune):
        total_layers = self.teacher_model.config.num_hidden_layers
        prune = block_layers_to_prune
        if prune is None:
            prune = getattr(self.student_model.config, "block_layers_to_prune", None)
        teacher_kept = [idx for idx in range(total_layers) if idx not in set(prune or [])]
        if self.student_model.config.num_hidden_layers != len(teacher_kept):
            raise ValueError(
                f"Student has {self.student_model.config.num_hidden_layers} layers, expected {len(teacher_kept)}."
            )
        return teacher_kept

    def _ensure_distill_model(self, device_student, block_layers_to_prune):
        teacher_kept = self._resolve_teacher_kept_layers(block_layers_to_prune)
        rebuild = self.distill_model is None or self.teacher_kept_layers != teacher_kept
        if rebuild:
            student_dim = self.student_model.config.hidden_size
            teacher_dim = self.teacher_model.config.hidden_size
            self.distill_model = DistilModel(
                self.student_model,
                student_dim,
                teacher_dim,
                teacher_kept,
            )
            self.teacher_kept_layers = teacher_kept
            self._attach_projector_params()

        self.distill_model.to(device_student)
        return self.distill_model, teacher_kept

    def _attach_projector_params(self) -> None:
        if self.optimizer is None or self.distill_model is None:
            return

        projector_params = [
            parameter
            for parameter in self.distill_model.projectors.parameters()
            if parameter.requires_grad
        ]
        if not projector_params:
            return

        existing_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        missing_params = [parameter for parameter in projector_params if id(parameter) not in existing_ids]
        if not missing_params:
            return

        group_config = {
            key: value
            for key, value in self.optimizer.param_groups[0].items()
            if key != "params"
        }
        group_config["params"] = missing_params
        self.optimizer.add_param_group(group_config)
        sync_scheduler_added_param_group_lrs(self.optimizer, self.scheduler)

    def _forward_teacher(self, batch, device_teacher):
        teacher_batch = {
            "input_ids": batch["input_ids"].to(device_teacher),
            "attention_mask": batch.get("attention_mask", None),
        }
        if teacher_batch["attention_mask"] is not None:
            teacher_batch["attention_mask"] = teacher_batch["attention_mask"].to(device_teacher)
        return self.teacher_model(**teacher_batch, output_hidden_states=True)

    def _forward_student(self, distill_model, input_ids, attention_mask):
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        return distill_model(**kwargs)

    def _extract_hidden_states(self, outputs, model_name: str):
        hidden_states = maybe_get_output_attr(outputs, "hidden_states")
        if hidden_states is None:
            raise ValueError(f"{model_name} must return `hidden_states` when `output_hidden_states=True`.")
        return hidden_states

    def _extract_aligned_features(
        self,
        student_outputs,
        teacher_outputs,
        teacher_kept,
        device_student,
    ):
        student_hidden_states = self._extract_hidden_states(student_outputs, "Student outputs")
        teacher_hidden_states = self._extract_hidden_states(teacher_outputs, "Teacher outputs")
        student_feats = [student_hidden_states[idx + 1][..., :-1, :] for idx in range(len(teacher_kept))]
        teacher_feats = [teacher_hidden_states[idx + 1][..., :-1, :].to(device_student) for idx in teacher_kept]
        return student_feats, teacher_feats

    def _extract_aligned_logits(self, student_outputs, teacher_outputs, device_student):
        teacher_logits = get_output_attr(teacher_outputs, "logits")[..., :-1, :].to(device_student)
        student_logits = get_output_attr(student_outputs, "logits")[..., :-1, :]
        return student_logits, teacher_logits

    def _compute_base_losses(
        self,
        *,
        student_outputs,
        teacher_outputs,
        labels,
        loss_mask,
        teacher_kept,
        projectors,
        device_student,
        temperature: float,
        alpha: float,
    ):
        student_logits, teacher_logits = self._extract_aligned_logits(
            student_outputs,
            teacher_outputs,
            device_student,
        )
        kl_ce = self.logits_loss(student_logits, teacher_logits, labels, loss_mask, temperature, alpha)
        student_feats, teacher_feats = self._extract_aligned_features(
            student_outputs,
            teacher_outputs,
            teacher_kept,
            device_student,
        )
        f_loss = self.feature_loss_selected_layers(
            student_feats,
            teacher_feats,
            projectors,
            loss_mask,
        )
        return kl_ce, f_loss, student_feats, teacher_feats

    def _compute_loss_terms(
        self,
        *,
        student_outputs,
        teacher_outputs,
        labels,
        loss_mask,
        teacher_kept,
        projectors,
        device_student,
        temperature: float,
        alpha: float,
        gamma: float,
    ) -> dict[str, torch.Tensor]:
        kl_ce, f_loss, _, _ = self._compute_base_losses(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            labels=labels,
            loss_mask=loss_mask,
            teacher_kept=teacher_kept,
            projectors=projectors,
            device_student=device_student,
            temperature=temperature,
            alpha=alpha,
        )
        return {
            "total_loss": kl_ce + gamma * f_loss,
            "distill_loss": kl_ce,
            "feature_loss": f_loss,
        }

    def _step_log_payload(
        self,
        loss_terms: dict[str, torch.Tensor],
        *,
        epoch: int,
        global_step: int,
    ) -> dict[str, float]:
        payload = {
            "train_loss": float(loss_terms["total_loss"].item()),
            "distill_loss": float(loss_terms["distill_loss"].item()),
            "feature_loss": float(loss_terms["feature_loss"].item()),
            "epoch": epoch,
            "step": global_step,
        }
        for metric_key in self._tracked_metric_keys():
            if metric_key in loss_terms:
                payload[metric_key] = float(loss_terms[metric_key].item())
        return payload

    def _run_validation(
        self,
        val_loader,
        *,
        distill_model,
        teacher_kept,
        device_teacher,
        device_student,
        temperature: float,
        alpha: float,
        gamma: float,
    ) -> dict[str, float] | None:
        if val_loader is None:
            return None

        distill_model.eval()
        total_loss = 0.0
        total_tokens = 0
        metric_sums = {
            metric_key: 0.0
            for metric_key in self._tracked_metric_keys()
        }
        with torch.no_grad():
            for batch in val_loader:
                ids, mask, labels, loss_mask = prepare_causal_lm_batch(batch, device_student)
                teacher_outputs = self._forward_teacher(batch, device_teacher)
                student_outputs = self._forward_student(distill_model, ids, mask)
                loss_terms = self._compute_loss_terms(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    labels=labels,
                    loss_mask=loss_mask,
                    teacher_kept=teacher_kept,
                    projectors=distill_model.projectors,
                    device_student=device_student,
                    temperature=temperature,
                    alpha=alpha,
                    gamma=gamma,
                )

                valid_tokens = int(loss_mask.sum().item())
                total_loss += float(loss_terms["total_loss"].item()) * valid_tokens
                total_tokens += valid_tokens
                for metric_key in self._tracked_metric_keys():
                    metric_sums[metric_key] += float(loss_terms[metric_key].item()) * valid_tokens
        distill_model.train()

        if total_tokens == 0:
            raise ValueError("Validation loader did not contain any valid target tokens.")
        metrics = {"val_loss": total_loss / total_tokens}
        for metric_key in self._tracked_metric_keys():
            metrics["val_{}".format(metric_key)] = metric_sums[metric_key] / total_tokens
        return metrics

    def distill(
        self,
        train_loader,
        val_loader=None,
        device_teacher=None,
        device_student=None,
        epochs: int = 1,
        grad_accumulation_steps: int = 8,
        alpha: float = 0.1,
        gamma: float = 0.1,
        temperature: float = 2.0,
        block_layers_to_prune=None,
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
        distill_model, teacher_kept = self._ensure_distill_model(device_student, block_layers_to_prune)
        distill_model.train()

        run = maybe_init_wandb(
            enabled=self.use_wandb,
            project_name=self.project_name,
            run_name=self.run_name,
            wandb_key=self.wandb_key,
        )

        self.history = self._initialize_history()
        global_step = 0
        zero_grad(self.optimizer)

        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_tokens = 0
                micro_batches = 0
                metric_sums = {
                    metric_key: 0.0
                    for metric_key in self._tracked_metric_keys()
                }

                for batch in train_loader:
                    ids, mask, labels, loss_mask = prepare_causal_lm_batch(batch, device_student)
                    with torch.no_grad():
                        teacher_outputs = self._forward_teacher(batch, device_teacher)

                    student_outputs = self._forward_student(distill_model, ids, mask)
                    loss_terms = self._compute_loss_terms(
                        student_outputs=student_outputs,
                        teacher_outputs=teacher_outputs,
                        labels=labels,
                        loss_mask=loss_mask,
                        teacher_kept=teacher_kept,
                        projectors=distill_model.projectors,
                        device_student=device_student,
                        temperature=temperature,
                        alpha=alpha,
                        gamma=gamma,
                    )
                    loss = loss_terms["total_loss"]
                    (loss / grad_accumulation_steps).backward()

                    valid_tokens = int(loss_mask.sum().item())
                    epoch_loss += loss.item() * valid_tokens
                    epoch_tokens += valid_tokens
                    micro_batches += 1
                    for metric_key in self._tracked_metric_keys():
                        metric_sums[metric_key] += float(loss_terms[metric_key].item()) * valid_tokens

                    if micro_batches == grad_accumulation_steps:
                        self.optimizer.step()
                        if self.scheduler is not None:
                            self.scheduler.step()
                            sync_scheduler_added_param_group_lrs(self.optimizer, self.scheduler)
                        zero_grad(self.optimizer)
                        maybe_log_wandb(run, self._step_log_payload(loss_terms, epoch=epoch, global_step=global_step))
                        global_step += 1
                        micro_batches = 0

                if micro_batches:
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                        sync_scheduler_added_param_group_lrs(self.optimizer, self.scheduler)
                    zero_grad(self.optimizer)
                    global_step += 1

                if epoch_tokens == 0:
                    raise ValueError("Training loader did not contain any valid target tokens.")
                avg_train = epoch_loss / epoch_tokens
                self.history["train_loss"].append(avg_train)
                for metric_key in self._tracked_metric_keys():
                    self.history["train_{}".format(metric_key)].append(metric_sums[metric_key] / epoch_tokens)

                val_metrics = self._run_validation(
                    val_loader,
                    distill_model=distill_model,
                    teacher_kept=teacher_kept,
                    device_teacher=device_teacher,
                    device_student=device_student,
                    temperature=temperature,
                    alpha=alpha,
                    gamma=gamma,
                )
                if val_metrics is not None:
                    self.history["val_loss"].append(val_metrics["val_loss"])
                    val_payload = {"val_loss": val_metrics["val_loss"], "epoch": epoch, "step": global_step}
                    for metric_key in self._tracked_metric_keys():
                        history_key = "val_{}".format(metric_key)
                        self.history[history_key].append(val_metrics[history_key])
                        val_payload[history_key] = val_metrics[history_key]
                    maybe_log_wandb(run, val_payload)
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
