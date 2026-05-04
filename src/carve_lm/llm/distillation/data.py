from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader


def _tokenizer_pad_id(tokenizer) -> int:
    if tokenizer is None:
        return 0
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    return 0


def _to_1d_long(value: Any) -> torch.Tensor:
    tensor = value.detach().clone() if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 1:
        raise ValueError("Token fields must be 1D per sample; received shape {}.".format(tuple(tensor.shape)))
    return tensor.long()


def _pad_1d(values: list[Any], *, pad_value: int) -> torch.Tensor:
    tensors = [_to_1d_long(value) for value in values]
    max_length = max((tensor.numel() for tensor in tensors), default=0)
    if max_length == 0:
        raise ValueError("Cannot collate an empty token sequence.")

    output = torch.full((len(tensors), max_length), int(pad_value), dtype=torch.long)
    for row_idx, tensor in enumerate(tensors):
        output[row_idx, : tensor.numel()] = tensor
    return output


def _stack_or_keep(values: list[Any]) -> Any:
    if all(torch.is_tensor(value) for value in values):
        try:
            return torch.stack(values)
        except RuntimeError:
            return values
    if all(isinstance(value, (int, float, bool)) for value in values):
        return torch.as_tensor(values)
    return values


class DistillationCollator:
    """Collate raw text or tokenized samples into distillation-ready LM batches."""

    def __init__(
        self,
        tokenizer=None,
        *,
        text_field: str = "text",
        max_length: int | None = None,
        padding: bool | str = True,
        truncation: bool = True,
        label_pad_token_id: int = -100,
        return_labels: bool = True,
    ):
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.label_pad_token_id = int(label_pad_token_id)
        self.return_labels = return_labels

    def __call__(self, samples: Sequence[Any]) -> dict[str, Any]:
        samples = list(samples)
        if not samples:
            raise ValueError("Cannot collate an empty batch.")

        first = samples[0]
        if isinstance(first, str) or (
            isinstance(first, Mapping) and self.text_field in first and "input_ids" not in first
        ):
            batch = self._collate_text(samples)
        elif isinstance(first, Mapping):
            batch = self._collate_mapping(samples)
        else:
            raise TypeError(
                "Distillation samples must be strings or mappings with either '{}' or 'input_ids'.".format(
                    self.text_field
                )
            )

        if self.return_labels and "labels" not in batch:
            labels = batch["input_ids"].clone()
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, self.label_pad_token_id)
            batch["labels"] = labels
        return batch

    def _collate_text(self, samples: list[Any]) -> dict[str, Any]:
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required to collate raw text samples.")

        texts = [
            sample if isinstance(sample, str) else sample[self.text_field]
            for sample in samples
        ]
        encoded = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return dict(encoded)

    def _collate_mapping(self, samples: list[Mapping[str, Any]]) -> dict[str, Any]:
        pad_id = _tokenizer_pad_id(self.tokenizer)
        keys = set().union(*(sample.keys() for sample in samples))
        batch: dict[str, Any] = {}

        if "input_ids" not in keys:
            raise ValueError("Tokenized distillation samples must contain `input_ids`.")

        batch["input_ids"] = _pad_1d(
            [sample["input_ids"] for sample in samples],
            pad_value=pad_id,
        )
        if "attention_mask" in keys:
            batch["attention_mask"] = _pad_1d(
                [sample.get("attention_mask", torch.ones_like(_to_1d_long(sample["input_ids"]))) for sample in samples],
                pad_value=0,
            )
        else:
            batch["attention_mask"] = batch["input_ids"].ne(pad_id).long()

        if "labels" in keys:
            batch["labels"] = _pad_1d(
                [sample.get("labels", sample["input_ids"]) for sample in samples],
                pad_value=self.label_pad_token_id,
            )

        for key in sorted(keys - {"input_ids", "attention_mask", "labels", self.text_field}):
            values = [sample[key] for sample in samples if key in sample]
            if len(values) == len(samples):
                batch[key] = _stack_or_keep(values)
        return batch


def create_distillation_dataloader(
    dataset,
    tokenizer=None,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    max_length: int | None = None,
    text_field: str = "text",
    padding: bool | str = True,
    truncation: bool = True,
    label_pad_token_id: int = -100,
    return_labels: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    **dataloader_kwargs,
) -> DataLoader:
    """Build a DataLoader that emits batches accepted by CarveLM distillers."""

    collate_fn = DistillationCollator(
        tokenizer=tokenizer,
        text_field=text_field,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        label_pad_token_id=label_pad_token_id,
        return_labels=return_labels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **dataloader_kwargs,
    )


__all__ = ["DistillationCollator", "create_distillation_dataloader"]
