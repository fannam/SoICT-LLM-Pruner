from __future__ import annotations

from collections import defaultdict

import torch

from ...core import ESTIMATOR_REGISTRY
from ...pruners._engine.utils import compute_causal_lm_loss, prepare_causal_lm_batch


@ESTIMATOR_REGISTRY.register("taylor.group")
class TaylorGroupEstimator:
    """Taylor-style salience estimator over discovered pruning groups."""

    def __init__(self, model, device: str = "cpu", model_adapter=None):
        del model_adapter
        self.model = model
        self.device = device

    def estimate(
        self,
        context,
        dataloader,
        *,
        variant: str,
        calibration_steps: int | None = None,
        loss: str = "causal_lm",
    ) -> dict[str, float]:
        if dataloader is None:
            raise ValueError("dataloader is required for taylor.group.")
        if loss != "causal_lm":
            raise ValueError("Only loss='causal_lm' is supported.")
        if variant not in {"param_first", "param_second", "param_mix", "vectorize"}:
            raise ValueError(
                "variant must be one of ['param_first', 'param_second', 'param_mix', 'vectorize']."
            )

        grad_sums = defaultdict(lambda: None)
        grad_square_sums = defaultdict(lambda: None)
        named_parameters = dict(self.model.named_parameters())
        model = self.model.to(self.device)
        was_training = model.training
        model.train()

        try:
            for step, batch in enumerate(dataloader):
                if calibration_steps is not None and step >= calibration_steps:
                    break

                prepared = prepare_causal_lm_batch(batch, self.device)
                model.zero_grad(set_to_none=True)
                outputs = model(**prepared)
                batch_loss = compute_causal_lm_loss(outputs, prepared)
                batch_loss.backward()

                for name, parameter in named_parameters.items():
                    if parameter.grad is None:
                        continue
                    grad = parameter.grad.detach().float().cpu()
                    if grad_sums[name] is None:
                        grad_sums[name] = grad.clone()
                        grad_square_sums[name] = grad.pow(2)
                    else:
                        grad_sums[name].add_(grad)
                        grad_square_sums[name].add_(grad.pow(2))
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        module_cache = dict(model.named_modules())
        scores: dict[str, float] = {}
        for group in context.groups:
            total = 0.0
            for slice_spec in group.dependent_slices:
                module = module_cache[slice_spec.module_path]
                parameter = getattr(module, slice_spec.param_name)
                parameter_name = "{}.{}".format(slice_spec.module_path, slice_spec.param_name)
                first_order = grad_sums.get(parameter_name)
                second_order = grad_square_sums.get(parameter_name)
                if first_order is None:
                    continue

                index = torch.tensor(slice_spec.indices, dtype=torch.long, device=parameter.device)
                weight_slice = torch.index_select(
                    parameter.detach().float(),
                    dim=slice_spec.axis,
                    index=index,
                )
                first_slice = torch.index_select(
                    first_order.to(weight_slice.device),
                    dim=slice_spec.axis,
                    index=index,
                )

                if variant == "param_first":
                    salience = weight_slice * first_slice
                    total += float(salience.abs().sum().item())
                    continue

                if variant == "vectorize":
                    salience = weight_slice * first_slice
                    total += float(torch.abs(salience.sum()).item())
                    continue

                second_slice = torch.index_select(
                    second_order.to(weight_slice.device),
                    dim=slice_spec.axis,
                    index=index,
                )
                second_term = weight_slice * second_slice * weight_slice
                if variant == "param_second":
                    total += float(second_term.abs().sum().item())
                else:
                    salience = (weight_slice * first_slice) - 0.5 * second_term
                    total += float(salience.abs().sum().item())

            scores[group.group_id] = total

        return scores


__all__ = ["TaylorGroupEstimator"]
