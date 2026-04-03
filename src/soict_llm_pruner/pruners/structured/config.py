from __future__ import annotations

from dataclasses import asdict, dataclass, field


def _validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError("{} must be in [0.0, 1.0).".format(name))


def _normalize_layers(value):
    if value is None:
        return None
    return tuple(sorted({int(layer_idx) for layer_idx in value}))


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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ImportanceConfig":
        return cls(**payload)


@dataclass(frozen=True)
class BlockWiseConfig:
    pruning_ratio: float
    attention_layers: tuple[int, ...] | None = None
    mlp_layers: tuple[int, ...] | None = None
    global_pruning: bool = False
    min_keep_per_layer: int = 1
    importance: ImportanceConfig = field(default_factory=lambda: ImportanceConfig(kind="l1"))
    clone_model: bool = True

    def __post_init__(self) -> None:
        _validate_ratio("pruning_ratio", self.pruning_ratio)
        object.__setattr__(self, "attention_layers", _normalize_layers(self.attention_layers))
        object.__setattr__(self, "mlp_layers", _normalize_layers(self.mlp_layers))
        if self.min_keep_per_layer < 0:
            raise ValueError("min_keep_per_layer must be non-negative.")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["importance"] = self.importance.to_dict()
        return data

    @classmethod
    def from_dict(cls, payload: dict) -> "BlockWiseConfig":
        payload = dict(payload)
        payload["importance"] = ImportanceConfig.from_dict(payload["importance"])
        return cls(**payload)


@dataclass(frozen=True)
class ChannelWiseConfig:
    pruning_ratio: float
    round_to: int | None = None
    importance: ImportanceConfig = field(default_factory=lambda: ImportanceConfig(kind="l1"))
    clone_model: bool = True

    def __post_init__(self) -> None:
        _validate_ratio("pruning_ratio", self.pruning_ratio)
        if self.round_to is not None and self.round_to <= 0:
            raise ValueError("round_to must be positive when provided.")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["importance"] = self.importance.to_dict()
        return data

    @classmethod
    def from_dict(cls, payload: dict) -> "ChannelWiseConfig":
        payload = dict(payload)
        payload["importance"] = ImportanceConfig.from_dict(payload["importance"])
        return cls(**payload)


@dataclass(frozen=True)
class LayerWiseConfig:
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
    def from_dict(cls, payload: dict) -> "LayerWiseConfig":
        return cls(**payload)
