from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError("{} must be in [0.0, 1.0).".format(name))


def _normalize_layers(value):
    if value is None:
        return None
    return tuple(sorted({int(layer_idx) for layer_idx in value}))


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
        return cls(
            name=payload["name"],
            kwargs=dict(payload.get("kwargs", {})),
        )


@dataclass(frozen=True)
class ImportanceConfig:
    kind: str
    taylor_variant: str | None = None
    calibration_steps: int | None = None
    seed: int = 0
    loss: str = "causal_lm"

    def __post_init__(self) -> None:
        valid_kinds = {"random", "l1", "l2", "taylor"}
        if self.kind not in valid_kinds:
            raise ValueError("kind must be one of {}.".format(sorted(valid_kinds)))
        if self.kind == "taylor":
            valid_variants = {"param_first", "param_second", "param_mix", "vectorize"}
            if self.taylor_variant not in valid_variants:
                raise ValueError(
                    "taylor_variant must be one of {} when kind='taylor'.".format(
                        sorted(valid_variants)
                    )
                )
        elif self.taylor_variant is not None:
            raise ValueError("taylor_variant is only valid when kind='taylor'.")
        if self.calibration_steps is not None and self.calibration_steps <= 0:
            raise ValueError("calibration_steps must be positive when provided.")
        if self.loss != "causal_lm":
            raise ValueError("Only loss='causal_lm' is supported in v1.")

    def to_estimator_spec(self, *, mode: str) -> EstimatorSpec:
        if self.kind == "random":
            return EstimatorSpec(
                name="random.group",
                kwargs={"seed": self.seed},
            )
        if self.kind in {"l1", "l2"}:
            estimator_name = "magnitude.channel" if mode == "channel" else "magnitude.group"
            return EstimatorSpec(
                name=estimator_name,
                kwargs={"norm": self.kind},
            )
        return EstimatorSpec(
            name="taylor.group",
            kwargs={
                "variant": self.taylor_variant,
                "calibration_steps": self.calibration_steps,
                "loss": self.loss,
            },
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ImportanceConfig":
        return cls(**payload)


@dataclass(frozen=True)
class WidthGroupConfig:
    pruning_ratio: float
    attention_layers: tuple[int, ...] | None = None
    mlp_layers: tuple[int, ...] | None = None
    global_pruning: bool = False
    min_keep_per_layer: int = 1
    estimator: EstimatorSpec = field(
        default_factory=lambda: EstimatorSpec("magnitude.group", {"norm": "l1"})
    )
    clone_model: bool = True
    importance: ImportanceConfig | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_ratio("pruning_ratio", self.pruning_ratio)
        object.__setattr__(self, "attention_layers", _normalize_layers(self.attention_layers))
        object.__setattr__(self, "mlp_layers", _normalize_layers(self.mlp_layers))
        if self.min_keep_per_layer < 0:
            raise ValueError("min_keep_per_layer must be non-negative.")
        estimator = self.estimator
        if isinstance(estimator, dict):
            estimator = EstimatorSpec.from_dict(estimator)
        if self.importance is not None:
            estimator = self.importance.to_estimator_spec(mode="group")
        object.__setattr__(self, "estimator", estimator)

    def to_dict(self) -> dict:
        data = {
            "pruning_ratio": self.pruning_ratio,
            "attention_layers": self.attention_layers,
            "mlp_layers": self.mlp_layers,
            "global_pruning": self.global_pruning,
            "min_keep_per_layer": self.min_keep_per_layer,
            "estimator": self.estimator.to_dict(),
            "clone_model": self.clone_model,
        }
        return data

    @classmethod
    def from_dict(cls, payload: dict) -> "WidthGroupConfig":
        payload = dict(payload)
        if "estimator" in payload:
            payload["estimator"] = EstimatorSpec.from_dict(payload["estimator"])
        if "importance" in payload:
            payload["importance"] = ImportanceConfig.from_dict(payload["importance"])
        return cls(**payload)


@dataclass(frozen=True)
class WidthChannelConfig:
    pruning_ratio: float
    round_to: int | None = None
    estimator: EstimatorSpec = field(
        default_factory=lambda: EstimatorSpec("magnitude.channel", {"norm": "l1"})
    )
    clone_model: bool = True
    importance: ImportanceConfig | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_ratio("pruning_ratio", self.pruning_ratio)
        if self.round_to is not None and self.round_to <= 0:
            raise ValueError("round_to must be positive when provided.")
        estimator = self.estimator
        if isinstance(estimator, dict):
            estimator = EstimatorSpec.from_dict(estimator)
        if self.importance is not None:
            estimator = self.importance.to_estimator_spec(mode="channel")
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
        if "importance" in payload:
            payload["importance"] = ImportanceConfig.from_dict(payload["importance"])
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
            raise ValueError("Only keep_strategy='prefix' is supported in v1.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "DepthLayerConfig":
        return cls(**payload)
