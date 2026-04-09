from __future__ import annotations

import torch

from .hybrid import HybridDistiller
from .optimal_transport import OTConfig, masked_ot_loss


class HybridOTDistiller(HybridDistiller):
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
        ot_config: OTConfig | None = None,
    ):
        super().__init__(
            teacher_model=teacher_model,
            student_model=student_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            use_wandb=use_wandb,
            wandb_key=wandb_key,
            project_name=project_name,
            run_name=run_name,
        )
        self.ot_config = ot_config or OTConfig()
        self.history = self._initialize_history()

    def _tracked_metric_keys(self) -> tuple[str, ...]:
        return ("ot_loss",)

    def ot_loss_selected_layers(self, student_feats, teacher_feats, projectors, mask) -> torch.Tensor:
        return masked_ot_loss(
            student_feats,
            teacher_feats,
            projectors,
            mask,
            self.ot_config,
        )

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
        kl_ce, f_loss, student_feats, teacher_feats = self._compute_base_losses(
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
        ot_loss = self.ot_loss_selected_layers(student_feats, teacher_feats, projectors, loss_mask)
        return {
            "total_loss": kl_ce + gamma * f_loss + self.ot_config.weight * ot_loss,
            "distill_loss": kl_ce,
            "feature_loss": f_loss,
            "ot_loss": ot_loss,
        }


__all__ = ["HybridOTDistiller"]
