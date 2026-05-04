from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError("{} must be in [0.0, 1.0).".format(name))


@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty estimator registry key.")
        object.__setattr__(self, "kwargs", dict(self.kwargs))

    def to_dict(self) -> dict:
        return {"name": self.name, "kwargs": dict(self.kwargs)}

    @classmethod
    def from_dict(cls, payload: dict) -> "EstimatorSpec":
        return cls(name=payload["name"], kwargs=dict(payload.get("kwargs", {})))


@dataclass(frozen=True)
class WidthChannelConfig:
    pruning_ratio: float
    round_to: int | None = None
    estimator: EstimatorSpec = field(
        default_factory=lambda: EstimatorSpec("magnitude.element", {"agg": "l1"})
    )
    clone_model: bool = True

    def __post_init__(self) -> None:
        _validate_ratio("pruning_ratio", self.pruning_ratio)
        if self.round_to is not None and self.round_to <= 0:
            raise ValueError("round_to must be positive when provided.")
        estimator = self.estimator
        if isinstance(estimator, dict):
            estimator = EstimatorSpec.from_dict(estimator)
        object.__setattr__(self, "estimator", estimator)

    def to_dict(self) -> dict:
        return {
            "pruning_ratio": self.pruning_ratio,
            "round_to": self.round_to,
            "estimator": self.estimator.to_dict(),
            "clone_model": self.clone_model,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WidthChannelConfig":
        payload = dict(payload)
        if "estimator" in payload:
            payload["estimator"] = EstimatorSpec.from_dict(payload["estimator"])
        return cls(**payload)


@dataclass(frozen=True)
class DepthLayerConfig:
    target_num_layers: int
    keep_strategy: str = "prefix"
    clone_model: bool = True

    def __post_init__(self) -> None:
        if self.target_num_layers <= 0:
            raise ValueError("target_num_layers must be positive.")
        if self.keep_strategy != "prefix":
            raise ValueError("Only keep_strategy='prefix' is supported.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "DepthLayerConfig":
        return cls(**payload)
