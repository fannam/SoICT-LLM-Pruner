from __future__ import annotations

import torch
from tests.vlm.fixtures.synthetic_vlm import make_synthetic_vlm

from carve_lm.vlm.evaluation import VLMMeasurer


def make_batch() -> dict:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "pixel_values": torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),
    }


def test_vlm_measurer_preserves_multimodal_batch_keys():
    model = make_synthetic_vlm(num_hidden_layers=1)
    measurer = VLMMeasurer(model=model, device="cpu")
    batch = make_batch()

    latency_ms, generated_tokens = measurer.measure_latency(batch, max_new_tokens=2, num_runs=2)
    throughput, total_time_sec, total_generated_tokens = measurer.measure_throughput([batch], max_new_tokens=3)

    assert latency_ms >= 0.0
    assert generated_tokens == 2
    assert throughput >= 0.0
    assert total_time_sec >= 0.0
    assert total_generated_tokens == 3
