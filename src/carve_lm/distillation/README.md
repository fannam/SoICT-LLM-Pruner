# Distillation

This package contains recovery-oriented training utilities used after pruning or when transferring behavior from a
teacher model to a smaller student. The implemented objectives are:

- logits distillation,
- hybrid logits + hidden-state distillation,
- teacher correction by ordinary language-model fine-tuning.

This document describes the exact losses used in the code.

## Notation

- `z_s`, `z_t`: student and teacher logits.
- `y`: shifted next-token labels.
- `m`: boolean mask of valid next-token positions.
- `Omega = {n | m_n = 1}`: valid positions after shifting by one token.
- `tau`: distillation temperature.
- `alpha`: interpolation weight between hard-label CE and teacher KL.
- `gamma`: weight of hidden-state matching in hybrid distillation.
- `P_i`: projector for student hidden state at matched layer `i`.

## Shared Causal-LM Batch Preparation

Implemented by `prepare_causal_lm_batch()` in `_common.py`.

Given a batch with `input_ids`, optional `attention_mask`, and optional `labels`, the code builds:

- `input_ids`: unchanged,
- `attention_mask`: all ones if not provided,
- `y = labels[..., 1:]`,
- `m = attention_mask[..., 1:]`.

If labels already contain ignore positions `-100`, the mask is refined to

`m = m and (y != -100)`

So every distillation loss in this package is computed only over `Omega`.

## Logits Distillation

Implemented by `LogitsDistiller`.

### Teacher and student alignment

The code drops the last logit position so that logits align with shifted labels:

- `z_t = teacher_logits[..., :-1, :]`
- `z_s = student_logits[..., :-1, :]`

### Hard-label cross-entropy term

Over valid positions,

`L_CE = -(1 / |Omega|) sum_{n in Omega} log softmax(z_s[n])_{y_n}`

This is implemented by `F.cross_entropy(masked_student, masked_labels)`.

### Soft teacher KL term

The teacher and student distributions at temperature `tau` are

- `p_t^tau(n) = softmax(z_t[n] / tau)`
- `p_s^tau(n) = softmax(z_s[n] / tau)`

The code computes

`L_KD = tau^2 * KL(p_t^tau || p_s^tau)`

using

- `log_student = log_softmax(z_s / tau)`
- `soft_teacher = softmax(z_t / tau)`
- `F.kl_div(log_student, soft_teacher, reduction="batchmean")`

### Final logits loss

The exact loss returned by `masked_logits_loss()` is

`L_logits = alpha L_CE + (1 - alpha) L_KD`

### Epoch-level averaging

Training and validation histories are token-weighted:

`L_epoch = (sum_b |Omega_b| L_b) / (sum_b |Omega_b|)`

The backward pass uses `L_b / grad_accumulation_steps`, but metrics are recorded with the unscaled `L_b`.

## Hybrid Distillation

Implemented by `HybridDistiller`.

This objective adds hidden-state regression to the logits loss above.

### Teacher-layer alignment

If `block_layers_to_prune` is provided, or the student config contains `block_layers_to_prune`, the code defines

`K_teacher = {0, 1, ..., L_teacher - 1} \ P_pruned`

The student is required to have exactly `|K_teacher|` layers. Student layer `i` is aligned to teacher layer
`K_teacher[i]`.

### Hidden states used in the loss

Both teacher and student run with `output_hidden_states=True`.

For aligned layer pair `(i, K_teacher[i])`, the compared tensors are

- student: `h_s^{(i)} = hidden_states_student[i + 1][..., :-1, :]`
- teacher: `h_t^{(K_teacher[i])} = hidden_states_teacher[K_teacher[i] + 1][..., :-1, :]`

The `+1` skips the embedding output because Hugging Face hidden-state lists are

`[embedding_output, layer1_output, layer2_output, ...]`

and `[..., :-1, :]` keeps them aligned with shifted next-token targets.

### Projectors

`DistilModel` creates one projector per aligned layer:

- if `d_student = d_teacher`, then `P_i = Identity`,
- otherwise `P_i(x) = W_i x + b_i` with `W_i in R^{d_teacher x d_student}`.

### Feature loss

For each aligned pair, the code masks valid token rows and computes mean squared error:

`L_feat^{(i)} = (1 / |Omega|) sum_{n in Omega} ||P_i(h_s^{(i)}[n]) - h_t^{(K_teacher[i])}[n]||_2^2`

The package-level feature loss is the mean across aligned layers:

`L_feat = (1 / M) sum_{i=1}^M L_feat^{(i)}`

where `M = |K_teacher|`.

### Final hybrid loss

The exact training objective is

`L_hybrid = L_logits + gamma L_feat`

Validation metrics are again token-weighted:

`L_val = (sum_b |Omega_b| (L_logits,b + gamma L_feat,b)) / (sum_b |Omega_b|)`

## Hybrid OT Distillation

Implemented by `HybridOTDistiller`.

This variant keeps the same logits loss and feature MSE as `HybridDistiller`, then adds a hidden-state
optimal-transport term on the aligned layers:

`L_hybrid_ot = L_logits + gamma L_feat + lambda_ot L_ot`

For one aligned layer and one sequence, let

- `S = {P_i(h_s[t]) : t in Omega}`
- `T = {h_t[t] : t in Omega}`

after masking invalid tokens and optionally subsampling to at most `max_tokens_per_sequence` positions.

The ground cost is

`C_uv = 1 - cos(S_u, T_v) + beta_pos ((u - v) / max(n - 1, 1))^2`

with a hard locality band `|u - v| <= window_radius`.

The implementation solves an entropic OT problem with log-domain Sinkhorn iterations, then uses the debiased
Sinkhorn divergence

`L_ot = OT_epsilon(S, T) - 0.5 OT_epsilon(S, S) - 0.5 OT_epsilon(T, T)`

where `OT_epsilon` reports only the final transport cost `<pi, C>`, not the entropy term.

The package-level OT loss is the mean over aligned layers and valid sequences in the batch.

## Teacher Correction

Implemented by `TeacherCorrection`.

Despite the name, this module does not add a separate correction regularizer. It simply fine-tunes the provided
model with the model's own training loss.

### Training objective

For each batch, the code runs

`outputs = model(**batch)`

and uses

`L_train = outputs.loss`

For Hugging Face causal-LM models this is usually the internal next-token cross-entropy induced by the batch's
`labels` and any masking already prepared by the dataloader.

### Optimization details

The training loop uses:

- gradient accumulation through `accelerate`,
- gradient clipping with threshold `max_grad_norm`,
- `AdamW`,
- cosine learning-rate schedule with warmup.

These are optimization mechanics; they do not change the loss formula itself.

### Evaluation metric

Validation computes the mean of the gathered per-batch losses:

`L_eval = mean_b L_b`

and reports

`PPL = exp(L_eval)`

with `inf` on overflow.

So the mathematical behavior of `TeacherCorrection` is ordinary language-model fine-tuning plus perplexity
tracking.

## Mapping From Formulas To Classes

- `LogitsDistiller`: `L_logits = alpha L_CE + (1 - alpha) L_KD`
- `HybridDistiller`: `L_hybrid = L_logits + gamma L_feat`
- `HybridOTDistiller`: `L_hybrid_ot = L_logits + gamma L_feat + lambda_ot L_ot`
- `DistilModel`: per-layer projector bank `P_i`
- `TeacherCorrection`: plain `outputs.loss` training with perplexity reporting
