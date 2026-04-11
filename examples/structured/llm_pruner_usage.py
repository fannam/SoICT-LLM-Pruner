from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from carve_lm.llm.pruners import (
    EstimatorSpec,
    WidthGroupConfig,
    WidthGroupPruner,
)


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    pruner = WidthGroupPruner(
        model,
        WidthGroupConfig(
            pruning_ratio=0.2,
            estimator=EstimatorSpec("magnitude.group", {"norm": "l1"}),
        ),
        device="cpu",
    )

    dummy_loader = DataLoader(
        [
            {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "labels": torch.tensor([1, 2, 3], dtype=torch.long),
            }
        ],
        batch_size=1,
    )
    pruner.discover(example_batch=None)
    scores = pruner.estimate(dummy_loader)
    result = pruner.apply(pruner.select(scores))
    print(result.model.config.num_hidden_layers)


if __name__ == "__main__":
    main()
