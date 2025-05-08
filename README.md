# SOICT-LLM-Pruner

A tri-level framework for structured pruning of Large Language Models (LLMs). Currently supports Llama and Qwen2 models.

## Installation

```bash
git clone https://github.com/fannam/SoICT-LLM-Pruner.git
```

Move to SoICT-LLM-Pruner folder

```bash
pip install -e .
```

## Features

- **Element-level Pruning**: Prune individual attention heads and MLP neurons
- **Layer-level Pruning**: Remove entire attention or MLP layers
- **Block-level Pruning**: Remove contiguous blocks of layers
- Support for Llama and Qwen2 models
- Multiple importance estimation methods

## Quick Start

### 1. Import Required Modules

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from estimator.element_estimator import ActivationElementImportanceEstimator
from estimator.layer_estimator import LayerImportanceEstimator
from estimator.block_estimator import SimilarityBlockImportanceEstimator
```

### 2. Load Model and Prepare Data

```python
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Prepare your calibration dataset
# The dataset should be a list of texts or a DataLoader
```

### 3. Estimate Component Importance

#### Element-level Importance (Attention Heads and MLP Neurons)

```python
# Initialize estimator
element_estimator = ActivationElementImportanceEstimator(model, dataloader)

# Estimate attention head importance
head_importance = element_estimator.estimate_attention_heads(agg="l2")

# Estimate MLP neuron importance
neuron_importance = element_estimator.estimate_mlp_neurons(agg="l2")

# Estimate embedding channel importance
embedding_importance = element_estimator.estimate_embedding_channels(agg="l2")
```

#### Layer-level Importance

```python
# Initialize estimator
layer_estimator = LayerImportanceEstimator(model)

# Estimate importance of attention and MLP layers
layer_importance = layer_estimator.estimate(dataloader)
# Returns: {'attention': [imp0, imp1, ...], 'mlp': [imp0, imp1, ...]}
```

#### Block-level Importance

```python
# Initialize estimator with desired block size
block_estimator = SimilarityBlockImportanceEstimator(model, block_size=1)

# Estimate importance of contiguous blocks
block_importance = block_estimator.estimate(dataloader)
# Returns: [imp0, imp1, ...] for each possible block start position
```

## Advanced Usage

### Custom Aggregation Methods

The element-level estimator supports different aggregation methods:
- "sum": Sum of activations
- "mean": Mean of activations
- "l2": L2 norm of activations
- "var": Variance of activations

```python
# Example with different aggregation
head_importance = element_estimator.estimate_attention_heads(agg="var")
```

### Pruning Based on Importance Scores

After obtaining importance scores, you can use them to prune the model:

```python
# Example: Prune least important attention heads
threshold = 0.5  # Keep top 50% of heads
for layer_idx, importance in head_importance.items():
    mask = importance > torch.quantile(importance, threshold)
    # Apply mask to prune heads
```

## Supported Models

- Llama-3 (all sizes)
- Qwen2 (all sizes)

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- tqdm >= 4.65.0
- datasets >= 2.14.0
- numpy >= 1.24.0
- accelerate >= 0.4.10

## Citation

