from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _move_batch_to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class VLMMeasurer:
    """Measure latency and throughput for multimodal generation models."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        model=None,
        processor=None,
        device: str | None = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        if model is None:
            if model_name_or_path is None:
                raise ValueError("`model_name_or_path` is required when `model` is not provided.")
            try:
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            except ImportError as exc:
                raise ModuleNotFoundError(
                    "VLMMeasurer requires a Transformers build with AutoProcessor and "
                    "Qwen2_5_VLForConditionalGeneration support."
                ) from exc

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path)
            if processor is None:
                processor = AutoProcessor.from_pretrained(model_name_or_path)
        elif processor is None and model_name_or_path is not None:
            try:
                from transformers import AutoProcessor
            except ImportError as exc:
                raise ModuleNotFoundError("Loading a processor requires `transformers`.") from exc
            processor = AutoProcessor.from_pretrained(model_name_or_path)

        self.model = model.to(self.device)
        self.processor = processor

        if model_name_or_path is not None:
            print(f"Model '{model_name_or_path}' loaded successfully.")

    def prepare_batch(self, *args, **kwargs) -> dict[str, Any]:
        if self.processor is None:
            raise ValueError("`processor` is required to prepare a batch.")

        kwargs.setdefault("return_tensors", "pt")
        batch = self.processor(*args, **kwargs)
        if isinstance(batch, Mapping):
            return dict(batch)
        raise TypeError("Processor output must be a mapping of model inputs.")

    @torch.no_grad()
    def measure_latency(
        self,
        batch: Mapping[str, Any],
        max_new_tokens: int = 50,
        num_runs: int = 5,
        **generation_kwargs,
    ) -> tuple[float, int]:
        if "input_ids" not in batch:
            raise ValueError("Batch must contain `input_ids` to measure generated tokens.")
        if num_runs <= 1:
            print("Warning: For meaningful latency, num_runs should be > 1 for warm-up.")

        total_time_ms = 0.0
        generated_tokens = 0

        if num_runs > 1:
            warmup_batch = _move_batch_to_device(batch, self.device)
            _ = self.model.generate(
                **warmup_batch,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
            if self.device == "cuda":
                torch.cuda.synchronize()

        for _ in range(max(1, num_runs - 1)):
            moved_batch = _move_batch_to_device(batch, self.device)
            input_ids_length = moved_batch["input_ids"].shape[1]

            start_time = time.perf_counter()
            outputs = self.model.generate(
                **moved_batch,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            total_time_ms += (end_time - start_time) * 1000
            generated_tokens = outputs.shape[1] - input_ids_length

        avg_latency_ms = total_time_ms / max(1, num_runs - 1)
        return avg_latency_ms, generated_tokens

    @torch.no_grad()
    def measure_throughput(
        self,
        batches: Sequence[Mapping[str, Any]],
        max_new_tokens: int = 50,
        **generation_kwargs,
    ) -> tuple[float, float, int]:
        total_generated_tokens = 0
        total_time_sec = 0.0

        for batch in batches:
            if "input_ids" not in batch:
                raise ValueError("Each batch must contain `input_ids` to measure generated tokens.")
            moved_batch = _move_batch_to_device(batch, self.device)
            input_ids_length = moved_batch["input_ids"].shape[1]

            start_time = time.perf_counter()
            outputs = self.model.generate(
                **moved_batch,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            total_time_sec += end_time - start_time
            total_generated_tokens += outputs.shape[1] - input_ids_length

        if total_time_sec == 0:
            throughput_tokens_per_sec = float("inf") if total_generated_tokens > 0 else 0.0
        else:
            throughput_tokens_per_sec = total_generated_tokens / total_time_sec

        return throughput_tokens_per_sec, total_time_sec, total_generated_tokens
