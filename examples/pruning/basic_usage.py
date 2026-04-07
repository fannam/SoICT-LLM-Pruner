from __future__ import annotations

from transformers import AutoModelForCausalLM

from soict_llm_pruner.estimators import create_estimator
from soict_llm_pruner.pruners import create_pruner


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    estimator = create_estimator("magnitude.element", model, device="cpu")
    pruner = create_pruner("width", model, device="cpu")

    head_scores = estimator.estimate_attention_heads(agg="l1")
    pruned_model = pruner.prune_attention_query(
        head_importance=head_scores,
        target_num_attention_heads=model.config.num_attention_heads // 2,
    )
    print(pruned_model.config.num_attention_heads)


if __name__ == "__main__":
    main()
