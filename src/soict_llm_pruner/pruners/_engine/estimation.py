from __future__ import annotations

from .config import EstimatorSpec, ImportanceConfig


def estimate_scores(
    model,
    adapter,
    context,
    estimator: EstimatorSpec,
    dataloader,
    *,
    device: str,
) -> dict[str, float]:
    from ...estimators import create_estimator

    estimator_instance = create_estimator(
        estimator.name,
        model,
        device=device,
        model_adapter=adapter,
    )
    estimate_fn = getattr(estimator_instance, "estimate", None)
    if estimate_fn is None:
        raise TypeError(
            "Estimator '{}' does not expose an estimate(context=..., dataloader=...) method.".format(
                estimator.name
            )
        )
    return estimate_fn(context=context, dataloader=dataloader, **estimator.kwargs)


def estimate_importance(
    model,
    context,
    importance: ImportanceConfig | EstimatorSpec,
    dataloader,
    *,
    device: str,
    model_adapter=None,
) -> dict[str, float]:
    estimator = importance
    if isinstance(importance, ImportanceConfig):
        mode = "channel" if context.mode == "channel" else "group"
        estimator = importance.to_estimator_spec(mode=mode)
    return estimate_scores(
        model,
        model_adapter,
        context,
        estimator,
        dataloader,
        device=device,
    )
