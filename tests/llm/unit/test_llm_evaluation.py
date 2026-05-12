from __future__ import annotations

from types import SimpleNamespace

import torch

from carve_lm.llm.evaluation import LLMMeasurer


class BatchEncoding(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, device):
        return BatchEncoding(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in self.items()
            }
        )


class DummyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token = None
    pad_token_id = None
    padding_side = "right"

    def __call__(self, texts, return_tensors="pt", padding=False, truncation=False):
        del return_tensors, truncation
        if isinstance(texts, str):
            texts = [texts]
        rows = [torch.tensor([ord(char) % 11 + 3 for char in text], dtype=torch.long) for text in texts]
        width = max(row.numel() for row in rows)
        pad_id = self.pad_token_id if self.pad_token_id is not None else self.eos_token_id
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row_idx, row in enumerate(rows):
            if padding and self.padding_side == "left":
                input_ids[row_idx, -row.numel() :] = row
                attention_mask[row_idx, -row.numel() :] = 1
            else:
                input_ids[row_idx, : row.numel()] = row
                attention_mask[row_idx, : row.numel()] = 1
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})


class DummyGenerationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=None, eos_token_id=2)

    def generate(self, input_ids, max_new_tokens=1, **kwargs):
        del kwargs
        new_tokens = torch.full(
            (input_ids.shape[0], max_new_tokens),
            self.config.eos_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, new_tokens], dim=1)


def test_llm_measurer_accepts_injected_model_and_tokenizer():
    measurer = LLMMeasurer(
        model=DummyGenerationModel(),
        tokenizer=DummyTokenizer(),
        device="cpu",
    )

    latency_ms, generated_tokens = measurer.measure_latency("hello", max_new_tokens=2, num_runs=2)
    throughput, total_time_sec, total_generated_tokens = measurer.measure_throughput(
        ["hello", "world"],
        max_new_tokens=3,
        batch_size=2,
    )

    assert latency_ms >= 0.0
    assert generated_tokens == 2
    assert throughput >= 0.0
    assert total_time_sec >= 0.0
    assert total_generated_tokens == 6
