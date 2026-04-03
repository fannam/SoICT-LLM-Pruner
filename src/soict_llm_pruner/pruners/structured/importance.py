from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from ._utils import compute_causal_lm_loss, prepare_causal_lm_batch, resolve_slice_tensor
from .config import ImportanceConfig
from .types import DiscoveryContext


def estimate_importance(
    model,
    context: DiscoveryContext,
    importance: ImportanceConfig,
    dataloader: DataLoader | None,
    *,
    device: str,
) -> dict[str, float]:
    if importance.kind == "random":
        return _random_importance(context, seed=importance.seed)
    if importance.kind in {"l1", "l2"}:
        return _magnitude_importance(model, context, ord=importance.kind)
    if dataloader is None:
        raise ValueError("dataloader is required when kind='taylor'.")
    return _taylor_importance(
        model,
        context,
        dataloader,
        device=device,
        variant=importance.taylor_variant,
        calibration_steps=importance.calibration_steps,
    )


def _random_importance(context: DiscoveryContext, seed: int) -> dict[str, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = torch.rand(len(context.groups), generator=generator)
    return {
        group.group_id: float(values[idx].item())
        for idx, group in enumerate(context.groups)
    }


def _magnitude_importance(model, context: DiscoveryContext, ord: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for group in context.groups:
        total = 0.0
        for slice_spec in group.dependent_slices:
            tensor = resolve_slice_tensor(model, slice_spec).float().reshape(-1)
            if ord == "l1":
                total += float(tensor.abs().sum().item())
            else:
                total += float(torch.linalg.vector_norm(tensor, ord=2).item())
        scores[group.group_id] = total
    return scores


def _taylor_importance(
    model,
    context: DiscoveryContext,
    dataloader: DataLoader,
    *,
    device: str,
    variant: str,
    calibration_steps: int | None,
) -> dict[str, float]:
    if variant is None:
        raise ValueError("variant must be provided for Taylor importance.")

    grad_sums = defaultdict(lambda: None)
    grad_square_sums = defaultdict(lambda: None)
    named_parameters = dict(model.named_parameters())
    model = model.to(device)
    was_training = model.training
    model.train()

    try:
        for step, batch in enumerate(dataloader):
            if calibration_steps is not None and step >= calibration_steps:
                break

            prepared = prepare_causal_lm_batch(batch, device)
            model.zero_grad(set_to_none=True)
            outputs = model(**prepared)
            loss = compute_causal_lm_loss(outputs, prepared)
            loss.backward()

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
            elif variant == "param_mix":
                salience = (weight_slice * first_slice) - 0.5 * second_term
                total += float(salience.abs().sum().item())
            else:
                raise ValueError("Unknown Taylor variant {}.".format(variant))

        scores[group.group_id] = total

    return scores
