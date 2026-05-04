from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch.utils.data import DataLoader

from carve_lm.llm.distillation.data import (
    DistillationCollator,
    _stack_or_keep,
)


class VLMDistillationCollator(DistillationCollator):
    """Collate token/text samples while preserving multimodal model inputs."""

    def __init__(
        self,
        tokenizer=None,
        *,
        processor=None,
        text_field: str = "text",
        image_field: str | None = None,
        processor_image_arg: str = "images",
        max_length: int | None = None,
        padding: bool | str = True,
        truncation: bool = True,
        label_pad_token_id: int = -100,
        return_labels: bool = True,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_field=text_field,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            label_pad_token_id=label_pad_token_id,
            return_labels=return_labels,
        )
        self.processor = processor
        self.image_field = image_field
        self.processor_image_arg = processor_image_arg

    def __call__(self, samples: Sequence[Any]) -> dict[str, Any]:
        samples = list(samples)
        if not samples:
            raise ValueError("Cannot collate an empty batch.")

        if self.processor is not None and self._can_use_processor(samples):
            batch = self._collate_with_processor(samples)
            if self.return_labels and "labels" not in batch and "input_ids" in batch:
                labels = batch["input_ids"].clone()
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    labels = labels.masked_fill(attention_mask == 0, self.label_pad_token_id)
                batch["labels"] = labels
            return batch

        return super().__call__(samples)

    def _can_use_processor(self, samples: list[Any]) -> bool:
        if not isinstance(samples[0], Mapping):
            return False
        if self.text_field not in samples[0]:
            return False
        image_field = self._resolve_image_field(samples)
        return image_field is None or image_field in samples[0]

    def _resolve_image_field(self, samples: list[Mapping[str, Any]]) -> str | None:
        if self.image_field is not None:
            return self.image_field
        for candidate in ("images", "image", "pixel_values"):
            if candidate in samples[0]:
                return candidate
        return None

    def _collate_with_processor(self, samples: list[Mapping[str, Any]]) -> dict[str, Any]:
        image_field = self._resolve_image_field(samples)
        processor_kwargs = {
            "text": [sample[self.text_field] for sample in samples],
            "padding": self.padding,
            "truncation": self.truncation,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }
        if image_field is not None:
            processor_kwargs[self.processor_image_arg] = [sample[image_field] for sample in samples]
        encoded = self.processor(**processor_kwargs)
        batch = dict(encoded)

        handled = {self.text_field}
        if image_field is not None:
            handled.add(image_field)
        handled.update(batch.keys())

        keys = set().union(*(sample.keys() for sample in samples))
        for key in sorted(keys - handled):
            values = [sample[key] for sample in samples if key in sample]
            if len(values) == len(samples):
                batch[key] = _stack_or_keep(values)
        return batch


def create_distillation_dataloader(
    dataset,
    tokenizer=None,
    *,
    processor=None,
    batch_size: int = 1,
    shuffle: bool = False,
    max_length: int | None = None,
    text_field: str = "text",
    image_field: str | None = None,
    processor_image_arg: str = "images",
    padding: bool | str = True,
    truncation: bool = True,
    label_pad_token_id: int = -100,
    return_labels: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    **dataloader_kwargs,
) -> DataLoader:
    """Build a DataLoader that emits batches accepted by VLM distillers."""

    collate_fn = VLMDistillationCollator(
        tokenizer=tokenizer,
        processor=processor,
        text_field=text_field,
        image_field=image_field,
        processor_image_arg=processor_image_arg,
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


create_vlm_distillation_dataloader = create_distillation_dataloader

__all__ = [
    "VLMDistillationCollator",
    "create_distillation_dataloader",
    "create_vlm_distillation_dataloader",
]
