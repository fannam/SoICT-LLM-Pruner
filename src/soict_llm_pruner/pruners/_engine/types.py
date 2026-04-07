from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SliceSpec:
    module_path: str
    param_name: str
    axis: int
    indices: tuple[int, ...]
    role: str

    def to_dict(self) -> dict:
        return {
            "module_path": self.module_path,
            "param_name": self.param_name,
            "axis": self.axis,
            "indices": list(self.indices),
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SliceSpec":
        return cls(
            module_path=payload["module_path"],
            param_name=payload["param_name"],
            axis=int(payload["axis"]),
            indices=tuple(int(index) for index in payload["indices"]),
            role=payload["role"],
        )


@dataclass(frozen=True)
class PruningGroup:
    group_id: str
    family: str
    layer_idx: int | None
    local_idx: int
    members: tuple[str, ...]
    dependent_slices: tuple[SliceSpec, ...]
    width: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "family": self.family,
            "layer_idx": self.layer_idx,
            "local_idx": self.local_idx,
            "members": list(self.members),
            "dependent_slices": [spec.to_dict() for spec in self.dependent_slices],
            "width": self.width,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PruningGroup":
        return cls(
            group_id=payload["group_id"],
            family=payload["family"],
            layer_idx=payload["layer_idx"],
            local_idx=int(payload["local_idx"]),
            members=tuple(payload["members"]),
            dependent_slices=tuple(
                SliceSpec.from_dict(spec) for spec in payload["dependent_slices"]
            ),
            width=int(payload.get("width", 1)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class LayerMetadata:
    layer_idx: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    hidden_size: int

    def to_dict(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "hidden_size": self.hidden_size,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LayerMetadata":
        return cls(**payload)


@dataclass(frozen=True)
class DiscoveryContext:
    mode: str
    family_key: str
    model_class_path: str
    config_class_path: str
    base_config: dict[str, Any]
    groups: tuple[PruningGroup, ...]
    layer_metadata: tuple[LayerMetadata, ...]
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def group_map(self) -> dict[str, PruningGroup]:
        return {group.group_id: group for group in self.groups}

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "family_key": self.family_key,
            "model_class_path": self.model_class_path,
            "config_class_path": self.config_class_path,
            "base_config": self.base_config,
            "groups": [group.to_dict() for group in self.groups],
            "layer_metadata": [metadata.to_dict() for metadata in self.layer_metadata],
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DiscoveryContext":
        return cls(
            mode=payload["mode"],
            family_key=payload["family_key"],
            model_class_path=payload["model_class_path"],
            config_class_path=payload["config_class_path"],
            base_config=dict(payload["base_config"]),
            groups=tuple(PruningGroup.from_dict(group) for group in payload["groups"]),
            layer_metadata=tuple(
                LayerMetadata.from_dict(metadata) for metadata in payload["layer_metadata"]
            ),
            hidden_size=int(payload["hidden_size"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            num_key_value_heads=int(payload["num_key_value_heads"]),
            head_dim=int(payload["head_dim"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class PruningPlan:
    mode: str
    selected_group_ids: tuple[str, ...]
    pruned_group_ids: tuple[str, ...]
    scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "selected_group_ids": list(self.selected_group_ids),
            "pruned_group_ids": list(self.pruned_group_ids),
            "scores": self.scores,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PruningPlan":
        return cls(
            mode=payload["mode"],
            selected_group_ids=tuple(payload["selected_group_ids"]),
            pruned_group_ids=tuple(payload["pruned_group_ids"]),
            scores={key: float(value) for key, value in payload["scores"].items()},
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class PruningResult:
    model: Any
    context: DiscoveryContext
    plan: PruningPlan
    manifest: dict[str, Any]
