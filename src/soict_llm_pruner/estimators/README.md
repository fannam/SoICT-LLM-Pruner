# Estimators

This package contains the score-producing side of pruning. The taxonomy is method-first:

- `activation.element`
- `magnitude.element`
- `similarity.layer`
- `similarity.block`
- `perplexity.block`
- `random.group`
- `magnitude.group`
- `magnitude.channel`
- `taylor.group`

This document describes the formulas actually implemented in this directory and its shared helpers.

## Notation

- `L`: number of decoder layers.
- `H_q`: number of query heads.
- `H_kv`: number of key/value heads.
- `q = H_q / H_kv`: number of query heads attached to one KV group.
- `d_h`: per-head width (`head_dim`).
- `D = H_q d_h`: residual hidden size.
- `D_ff`: MLP intermediate size.
- `B, T`: batch size and sequence length.
- `x_{b,t}`: residual hidden state at token `(b, t)`.
- `W in R^{m x n}`: a linear-layer weight matrix with `m` output rows and `n` input columns.
- `Omega`: set of valid next-token prediction positions after shifting labels by one token.

## Activation Estimator

Implemented by `ActivationEstimator` through `_BaseActivationElementEstimator`.

### Streaming reducer

All activation-based scores are reduced by `_ActivationAccumulator`. For a stream of scalar observations
`{u_1, ..., u_N}` attached to one coordinate, the implemented reducers are:

- `sum`:
  `S = sum_{n=1}^N u_n`
- `mean`:
  `S = (1 / N) sum_{n=1}^N u_n`
- `l2`:
  `S = sqrt(sum_{n=1}^N u_n^2)`
- `var`:
  `S = (sum_{n=1}^N u_n^2 - (sum_{n=1}^N u_n)^2 / N) / (N - 1)`

The variance branch is the sample variance and is clamped from below by `0.0`.

### Attention-head activation scores

The code hooks the input of `o_proj`, reshapes it from `(B, T, D)` into `(B, T, H_q, d_h)`, and computes the
per-token head magnitude

`a_{l,b,t,h} = ||c_{l,b,t,h,:}||_2`

where `c` is the `o_proj` input at layer `l`.

The final score for head `h` in layer `l` is the reducer above applied over all observed `(b, t)` positions.

### Attention-group activation scores

For grouped-query attention, the same hooked tensor is reshaped into `(B, T, H_kv, q d_h)`. The per-token group
score is

`a_{l,b,t,g} = ||c_{l,b,t,g,:}||_2`

So one atomic attention group is one KV group plus all attached query heads.

### MLP-neuron activation scores

The code hooks the input of `down_proj`, which has shape `(B, T, D_ff)`. For neuron `i`, the streamed quantity is
the raw coordinate value

`u_{l,b,t,i} = h_{l,b,t,i}`

and the chosen reducer is applied coordinate-wise across all tokens. This means `sum` and `mean` are signed,
while `l2` and `var` are sign-agnostic.

### Embedding-channel activation scores

The code hooks:

- every `input_layernorm` output,
- every `post_attention_layernorm` output,
- the final normalization output.

For each hook output `z in R^{B x T x D}`, channel `c` receives the streamed sequence `{z_{b,t,c}}`, and the same
reducer is applied independently for each channel.

## Magnitude Estimators

Implemented by `MagnitudeEstimator`, `MagnitudeGroupEstimator`, and `MagnitudeChannelEstimator`.

The supported norms are:

- `l1`: `||A||_1 = sum_i |A_i|`
- `l2`: `||A||_2 = sqrt(sum_i A_i^2)`

### Element-level weight magnitude

`MagnitudeEstimator` scores structural units by summing norms of all weight slices that must survive together.

### Attention-query head score

For head `h`, let `Q_h` be the contiguous row block in `q_proj` and the matching input-column block in `o_proj`.
The implemented score is

`s_{l,h} = ||W_q^{(l)}[Q_h, :]||_p + ||W_o^{(l)}[:, Q_h]||_p`

where `p in {1, 2}`.

Only `q_proj` and `o_proj` are used because query-head pruning leaves `k_proj` and `v_proj` untouched.

### Attention-group score

For group `g`, let `Q_g` be the query rows for all query heads attached to group `g`, and let `K_g` be the
matching KV rows. The score is

`s_{l,g} = ||W_q^{(l)}[Q_g, :]||_p + ||W_k^{(l)}[K_g, :]||_p + ||W_v^{(l)}[K_g, :]||_p + ||W_o^{(l)}[:, Q_g]||_p`

### MLP-neuron score

For neuron `i`, the implemented score is

`s_{l,i} = ||W_gate^{(l)}[i, :]||_p + ||W_up^{(l)}[i, :]||_p + ||W_down^{(l)}[:, i]||_p`

### Embedding-channel score

`estimate_embedding_channels()` aggregates every weight slice coupled to one hidden channel `c`. In code this is:

- embedding column `E[:, c]`,
- LM-head column `W_lm[:, c]` if present and not tied to embeddings,
- final-norm scale `gamma_final[c]`,
- per-layer norm scales `gamma_in^{(l)}[c]` and `gamma_post^{(l)}[c]`,
- attention input/output channel slices,
- MLP input/output channel slices.

The total score is therefore

`s_c = s_embed(c) + s_lm(c) + s_norm(c) + sum_{l=1}^L (s_attn^{(l)}(c) + s_mlp^{(l)}(c))`

with each term being an `l1` or `l2` norm over the corresponding row or column slice. Shared embedding / LM-head
weights are deduplicated by storage pointer, so tied weights are counted once.

### Context-based group and channel magnitude

`MagnitudeGroupEstimator` and `MagnitudeChannelEstimator` work on a discovered `DiscoveryContext`. For a pruning
group `g` with dependent slices `S(g)`, the implemented score is

`s_g = sum_{sigma in S(g)} ||vec(W[sigma])||_p`

where `W[sigma]` is the exact tensor slice described by `SliceSpec`.

`MagnitudeChannelEstimator` adds one extra check: every group in the context must have `family == "channel"`.

## Similarity Estimators

Implemented by `LayerSimilarityEstimator`, `BlockSimilarityEstimator`, and the `SimilarityEstimator` facade.

### Core importance function

`calculate_importance(x, y)` in `core.scoring` computes

`Imp(x, y) = 1 - mean_n cos(x_n, y_n)`

after flattening to token rows and replacing any `NaN` cosine value by `1.0`.

### Layer similarity

For each layer `l`, the code captures:

- `x_attn^{(l)}`: input to the attention sublayer,
- `o_attn^{(l)}`: output of the attention module,
- `x_mlp^{(l)}`: input to the MLP sublayer,
- `o_mlp^{(l)}`: output of the MLP module.

The reported attention score is

`s_attn^{(l)} = Imp(x_attn^{(l)}, x_attn^{(l)} + o_attn^{(l)})`

and the MLP score is

`s_mlp^{(l)} = Imp(x_mlp^{(l)}, x_mlp^{(l)} + o_mlp^{(l)})`

The final value is the arithmetic mean of these batch-level scores over the dataloader.

This implementation does not mask padded tokens at the similarity stage.

### Block similarity

For a contiguous block starting at `i` with size `b`, the code compares

- block input: the attention-sublayer input of layer `i`,
- block output: the residual stream after the last MLP in layer `i + b - 1`.

If `r_{i+b-1}` is the last MLP input and `m_{i+b-1}` is the last MLP output, then

`y_block = r_{i+b-1} + m_{i+b-1}`

and the score is the mean cosine distance over valid tokens:

`s_i = (1 / |Omega|) sum_{(b,t) in Omega} (1 - cos(x_{i,b,t}, y_{block,b,t}))`

If an `attention_mask` exists, `Omega` excludes masked positions.

## Perplexity Estimator

Implemented by `PerplexityEstimator` through `_BaseBlockPerplexityEstimator`.

### Token-weighted perplexity

For each processed batch, the model is run with shifted causal-LM labels. Let `ell_batch` be `outputs.loss` from the
model and `N_batch = |Omega_batch|` the number of valid next-token targets after masking. The estimator accumulates

`NLL = sum_batch ell_batch N_batch`

`N = sum_batch N_batch`

`avg_loss = NLL / N`

`PPL = exp(avg_loss)`

If `avg_loss <= 0`, `NaN`, or infinite, the helper returns `inf`.

### Block importance by identity ablation

For each block start `i`, all layers in `[i, i + b - 1]` are temporarily replaced by adapter-provided identity
decoder layers and a new perplexity is measured.

With baseline perplexity `PPL_base` and ablated perplexity `PPL_i`, the estimator supports:

- `perplexity_increase`:
  `s_i = PPL_i - PPL_base`
- `perplexity_ratio`:
  `s_i = PPL_i / PPL_base`

So a larger score means that removing that block hurts language modeling more.

## Taylor Group Estimator

Implemented by `TaylorGroupEstimator`.

This estimator operates on discovered pruning groups and uses backpropagated causal-LM loss. Let `w` denote one
parameter element inside a slice of group `g`.

Across calibration batches, the code accumulates:

- first-order gradient sum:
  `G(w) = sum_t (dL_t / dw)`
- squared-gradient sum:
  `H_diag(w) = sum_t (dL_t / dw)^2`

`H_diag` is a diagonal second-order proxy built from squared gradients, not an exact Hessian.

For each dependent slice of group `g`, the implemented variants are:

- `param_first`:
  `s_sigma = sum_{w in sigma} |w G(w)|`
- `vectorize`:
  `s_sigma = |sum_{w in sigma} w G(w)|`
- `param_second`:
  `s_sigma = sum_{w in sigma} |w H_diag(w) w|`
- `param_mix`:
  `s_sigma = sum_{w in sigma} |w G(w) - 0.5 w H_diag(w) w|`

The final group score is

`s_g = sum_{sigma in S(g)} s_sigma`

The code sums raw gradients over calibration steps; it does not divide by the number of batches.

## Random Group Estimator

Implemented by `RandomGroupEstimator`.

For `|G|` groups, the code samples

`u_g ~ Uniform(0, 1)`

from a CPU `torch.Generator` seeded by `seed`, and returns those values as the score map.

## Mapping From Formulas To Classes

- `ActivationEstimator`: activation reducers over heads, groups, neurons, and embedding channels.
- `MagnitudeEstimator`: element-level structural magnitude scores.
- `MagnitudeGroupEstimator`: group-wise magnitude over discovered `dependent_slices`.
- `MagnitudeChannelEstimator`: same as above, restricted to channel contexts.
- `SimilarityEstimator`: facade exposing both layer and block similarity.
- `LayerSimilarityEstimator`: attention/MLP residual cosine-distance scores.
- `BlockSimilarityEstimator`: contiguous-block residual cosine-distance scores.
- `PerplexityEstimator`: block ablation scored by perplexity increase or ratio.
- `TaylorGroupEstimator`: first-order / second-order Taylor-style group salience.
- `RandomGroupEstimator`: seeded random baseline.
