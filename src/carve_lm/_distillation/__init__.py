from __future__ import annotations

__all__ = [
    "DistilModel",
    "HybridOTDistillerMixin",
    "OTConfig",
    "masked_ot_loss",
    "sinkhorn_divergence",
    "sinkhorn_transport_cost",
]


def __getattr__(name: str):
    if name == "DistilModel":
        from .wrappers import DistilModel

        return DistilModel
    if name == "HybridOTDistillerMixin":
        from .hybrid_ot import HybridOTDistillerMixin

        return HybridOTDistillerMixin
    if name in {"OTConfig", "masked_ot_loss", "sinkhorn_divergence", "sinkhorn_transport_cost"}:
        from .optimal_transport import OTConfig, masked_ot_loss, sinkhorn_divergence, sinkhorn_transport_cost

        return {
            "OTConfig": OTConfig,
            "masked_ot_loss": masked_ot_loss,
            "sinkhorn_divergence": sinkhorn_divergence,
            "sinkhorn_transport_cost": sinkhorn_transport_cost,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
