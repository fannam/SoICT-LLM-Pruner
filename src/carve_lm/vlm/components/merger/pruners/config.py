from __future__ import annotations

from dataclasses import dataclass, field
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
class WidthConfig:
    pruning_ratio: float
    estimator: EstimatorSpec = field(
        default_factory=lambda: EstimatorSpec("magnitude.element", {"agg": "l1"})
    )
    clone_model: bool = True

    def __post_init__(self) -> None:
        _validate_ratio("pruning_ratio", self.pruning_ratio)
        estimator = self.estimator
        if isinstance(estimator, dict):
            estimator = EstimatorSpec.from_dict(estimator)
        object.__setattr__(self, "estimator", estimator)

    def to_dict(self) -> dict:
        return {
            "pruning_ratio": self.pruning_ratio,
            "estimator": self.estimator.to_dict(),
            "clone_model": self.clone_model,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WidthConfig":
        payload = dict(payload)
        if "estimator" in payload:
            payload["estimator"] = EstimatorSpec.from_dict(payload["estimator"])
        return cls(**payload)


@dataclass(frozen=True)
class BridgeChannelConfig:
    pruning_ratio: float
    round_to: int | None = None
    language_estimator: EstimatorSpec = field(
        default_factory=lambda: EstimatorSpec("magnitude.channel", {"norm": "l1"})
    )
    merger_estimator: EstimatorSpec = field(
        default_factory=lambda: EstimatorSpec("magnitude.element", {"agg": "l1"})
    )
    language_weight: float = 1.0
    merger_weight: float = 1.0
    clone_model: bool = True

    def __post_init__(self) -> None:
        _validate_ratio("pruning_ratio", self.pruning_ratio)
        if self.round_to is not None and self.round_to <= 0:
            raise ValueError("round_to must be positive when provided.")
        language_estimator = self.language_estimator
        merger_estimator = self.merger_estimator
        if isinstance(language_estimator, dict):
            language_estimator = EstimatorSpec.from_dict(language_estimator)
        if isinstance(merger_estimator, dict):
            merger_estimator = EstimatorSpec.from_dict(merger_estimator)
        object.__setattr__(self, "language_estimator", language_estimator)
        object.__setattr__(self, "merger_estimator", merger_estimator)

    def to_dict(self) -> dict:
        return {
            "pruning_ratio": self.pruning_ratio,
            "round_to": self.round_to,
            "language_estimator": self.language_estimator.to_dict(),
            "merger_estimator": self.merger_estimator.to_dict(),
            "language_weight": self.language_weight,
            "merger_weight": self.merger_weight,
            "clone_model": self.clone_model,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BridgeChannelConfig":
        payload = dict(payload)
        if "language_estimator" in payload:
            payload["language_estimator"] = EstimatorSpec.from_dict(payload["language_estimator"])
        if "merger_estimator" in payload:
            payload["merger_estimator"] = EstimatorSpec.from_dict(payload["merger_estimator"])
        return cls(**payload)
