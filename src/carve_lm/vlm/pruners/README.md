# Pruners

This package contains the structural transformation side of the library. The canonical public families are:

- `width`
- `width.group`
- `width.channel`
- `component`
- `depth.block`
- `depth.layer`

This document focuses on the selection rules and tensor rewrites that the current implementation applies.

## Notation

- `L`: number of decoder layers.
- `H_q`: number of query heads.
- `H_kv`: number of key/value heads.
- `q = H_q / H_kv`: query heads per KV group.
- `d_h`: head dimension.
- `D = H_q d_h`: hidden size.
- `D_ff`: MLP intermediate size.
- `TopK(s, k)`: indices of the `k` largest scores in `s`, then sorted in ascending index order before slicing.
- `r`: pruning ratio.
- `m`: `min_keep_per_layer`.

## Width Pruner (`width`)

Implemented by `WidthPruner` in `element.py`. This is the canonical home of the old `ElementPruner`.

### Generic keep rule

For any per-layer score vector `s_l`, the kept index set is

`K_l = TopK(s_l, k)`

The pruner always keeps the largest scores and physically slices weights to the kept indices.

### Query-head pruning

Target head count `H_q'` must satisfy

- `H_q' < H_q`
- `H_q' >= H_kv`
- `H_q' mod H_kv = 0`

For each kept head `h in K_l`, define its row block

`R(h) = {h d_h, ..., (h + 1) d_h - 1}`

Then the rewritten projections are

- `W_q' = W_q[R(K_l), :]`
- `b_q' = b_q[R(K_l)]` if a bias exists
- `W_o' = W_o[:, R(K_l)]`
- `b_o' = b_o`

Only `q_proj` and `o_proj` are sliced because query-head pruning preserves the original KV heads.

### Attention-group pruning

Target KV-group count `G'` must satisfy `0 < G' < H_kv`.

Let `K_l = TopK(s_l, G')` be the kept KV groups. Each kept group expands to:

- KV rows:
  `R_kv(g) = {g d_h, ..., (g + 1) d_h - 1}`
- Query-head rows:
  `R_q(g) = union_{h = g q}^{(g + 1) q - 1} {h d_h, ..., (h + 1) d_h - 1}`

The rewritten projections are

- `W_q' = W_q[R_q(K_l), :]`
- `W_k' = W_k[R_kv(K_l), :]`
- `W_v' = W_v[R_kv(K_l), :]`
- `W_o' = W_o[:, R_q(K_l)]`

If legacy `head_importance` is passed instead of `group_importance`, the code first averages head scores within
each KV group:

`s_group(g) = (1 / q) sum_{h in group g} s_head(h)`

### MLP-neuron pruning

Target width `D_ff'` must satisfy `0 < D_ff' < D_ff`.

With `K_l = TopK(s_l, D_ff')`, the rewritten matrices are

- `W_gate' = W_gate[K_l, :]`
- `W_up' = W_up[K_l, :]`
- `W_down' = W_down[:, K_l]`

Biases are sliced on output rows where they exist.

### Embedding-channel pruning

Target hidden size `D'` must satisfy `0 < D' < D`.

If the user passes a mapping of per-module channel scores, the code first collapses them into a single global
score vector with

`s_global(c) = sum_m s_m(c)`

Then `C = TopK(s_global, D')` is used to slice every tensor coupled to the residual channel:

- embedding columns,
- LM-head columns,
- norm weights and biases,
- attention input / output channels,
- MLP input / output channels.

This shrinks the model hidden size directly.

## Width-Group Pruner (`width.group`)

Implemented by `WidthGroupPruner` in `_engine/facade.py`.

This family works on discovered pruning groups instead of raw score tensors.

### Discovery: atomic groups

`discover_blockwise()` creates two kinds of groups.

### Attention group

For layer `l` and KV group `g`, one atomic attention group contains:

- all query rows attached to KV group `g`,
- the matching `k_proj` rows,
- the matching `v_proj` rows,
- the matching `o_proj` input columns.

Its exact row sets are

- `R_q(l, g) = union_{h = g q}^{(g + 1) q - 1} {h d_h, ..., (h + 1) d_h - 1}`
- `R_kv(l, g) = {g d_h, ..., (g + 1) d_h - 1}`

### MLP group

For layer `l` and neuron `i`, one atomic MLP group contains:

- row `i` of `gate_proj`,
- row `i` of `up_proj`,
- column `i` of `down_proj`.

### Selection with `WidthGroupConfig`

Let `C` be the candidate groups and `n_{f,l}` the number of groups in family `f` and layer `l`.

The nominal prune count is

`p_nominal = floor(r |C|)`

The hard upper bound imposed by `min_keep_per_layer = m` is

`p_max = sum_{f,l} max(0, n_{f,l} - m)`

So the actual target is

`p = min(p_nominal, p_max)`

The pruning plan is built greedily by iterating groups in ascending score order and pruning a group only if its
family-layer bucket still has more than `m` survivors left.

If `global_pruning = False`, this greedy rule is applied separately to:

- all attention groups in the requested `attention_layers`,
- all MLP groups in the requested `mlp_layers`.

If `global_pruning = True`, both families are merged into one candidate set before sorting, but the per-layer
minimum-keep constraint is still enforced independently for each `(family, layer)` bucket.

### Execution

For each layer:

- kept attention groups are converted back into kept query rows and kept KV rows,
- `q_proj`, `k_proj`, `v_proj`, and `o_proj` are sliced,
- attention metadata is patched to the new head counts,
- `gate_proj`, `up_proj`, and `down_proj` are sliced for kept MLP neurons.

If a layer keeps all groups of one family, that family is left unchanged.

## Width-Channel Pruner (`width.channel`)

Implemented by `WidthChannelPruner` in `_engine/facade.py`.

This family prunes a hidden channel bundle across the entire model, not per-layer.

### Discovery: per-head channel bundles

`discover_channelwise()` groups channels by the offset inside each attention head.

For bundle index `c in {0, ..., d_h - 1}`:

- residual indices:
  `R(c) = {h d_h + c | h = 0, ..., H_q - 1}`
- KV indices:
  `K(c) = {g d_h + c | g = 0, ..., H_kv - 1}`

One channel bundle touches:

- embedding / LM-head columns,
- final norm and layernorm coordinates,
- attention input and output channels,
- MLP input and output channels.

Its width is `|R(c)| = H_q`, so keeping one bundle preserves one scalar channel per query head.

### Selection with `WidthChannelConfig`

The code first computes the unrounded hidden size target

`D_raw = max(1, D - floor(D r))`

Then it rounds this down to a multiple of

`rho = round_to if round_to is not None else H_q`

with the implementation rule

`D_keep = D_raw - (D_raw mod rho)`

If that becomes non-positive, the code falls back to `D_keep = rho`.

Since one bundle contributes `H_q` residual channels, the number of kept bundles is

`G_keep = max(1, D_keep / H_q)`

Bundles are ranked by descending score, and the top `G_keep` are kept.

### Execution

Let `C_keep` be the kept bundle ids. The resulting indices are

- `R_keep = union_{c in C_keep} R(c)`
- `K_keep = union_{c in C_keep} K(c)`

The model is rewritten as follows:

- embeddings and LM head keep columns `R_keep`,
- norms keep coordinates `R_keep`,
- `q_proj` keeps output rows `R_keep` and input columns `R_keep`,
- `k_proj` and `v_proj` keep output rows `K_keep` and input columns `R_keep`,
- `o_proj` keeps output rows `R_keep` and input columns `R_keep`,
- `gate_proj` and `up_proj` keep input columns `R_keep`,
- `down_proj` keeps output rows `R_keep`.

Finally:

- `hidden_size` becomes `|R_keep|`,
- `head_dim` becomes the number of kept bundles `|C_keep|`,
- the number of query heads and KV heads stays unchanged.

So width-channel pruning reduces per-head width, not the number of heads.

## Component Pruner (`component`)

Implemented by `ComponentPruner` in `layer.py`. This is the canonical home of the old `LayerPruner`.

This pruner does not delete decoder layers. It replaces selected attention modules and selected MLP modules with
identity modules.

For attention scores `s_attn` and prune count `p_attn`, the kept set is

`K_attn = TopK(s_attn, L - p_attn)`

The pruned set is `P_attn = {0, ..., L - 1} \ K_attn`.

The same rule is applied independently to MLP scores and `p_mlp`.

For each `l in P_attn`, the attention submodule is replaced by adapter-made identity attention.
For each `l in P_mlp`, the MLP submodule is replaced by adapter-made identity MLP.

Because layers remain in place, this is component removal, not depth reduction.

## Depth-Block Pruner (`depth.block`)

Implemented by `DepthBlockPruner` in `block.py`.

This pruner removes whole decoder layers by deleting contiguous blocks.

### Input score layout

For block size `b`, the expected score array has length

`N_blocks = L - b + 1`

where entry `s_i` corresponds to interval

`I_i = [i, i + b - 1]`

### Non-overlapping block selection

The implementation sorts start indices by ascending score and greedily selects a start `i` if

`I_i cap I_j = empty`

for every previously selected block `j`.

This continues until `num_block_to_prune` blocks are chosen. Therefore the optimizer is:

- score-minimizing,
- greedy,
- interval-disjoint.

After selection, all layers in the chosen intervals are deleted from the layer list in reverse order.

## Depth-Layer Pruner (`depth.layer`)

Implemented by `DepthLayerPruner` in `_engine/facade.py`.

Current v1 behavior supports only

- `keep_strategy = "prefix"`

and simply keeps the first `L' = target_num_layers` decoder layers:

`K = {0, 1, ..., L' - 1}`

`P = {L', L' + 1, ..., L - 1}`

The layer list is replaced by the prefix `layers[:L']`, and `num_hidden_layers` is patched accordingly.

This is a true depth reduction because the decoder stack becomes shorter.

## Plans And Results

`width.group`, `width.channel`, and `depth.layer` return a `PruningResult` containing:

- the rewritten model,
- the `DiscoveryContext`,
- the selected `PruningPlan`,
- a manifest.

The mathematical part of the plan is always the same:

- discover the atomic units,
- assign one scalar score per unit,
- select the keep / prune set under family-specific constraints,
- slice tensors or remove modules accordingly.
