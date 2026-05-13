Qwen3VLForConditionalGeneration(
  (model): Qwen3VLModel(
    (visual): Qwen3VLVisionModel(
      (patch_embed): Qwen3VLVisionPatchEmbed(
        (proj): Conv3d(3, 1024, kernel_size=(2, 16, 16), stride=(2, 16, 16))
      )
      (pos_embed): Embedding(2304, 1024)
      (rotary_pos_emb): Qwen3VLVisionRotaryEmbedding()
      (blocks): ModuleList(
        (0-23): 24 x Qwen3VLVisionBlock(
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (attn): Qwen3VLVisionAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): Qwen3VLVisionMLP(
            (linear_fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (linear_fc2): Linear(in_features=4096, out_features=1024, bias=True)
            (act_fn): GELUTanh()
          )
        )
      )
      (merger): Qwen3VLVisionPatchMerger(
        (norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
        (linear_fc1): Linear(in_features=4096, out_features=4096, bias=True)
        (act_fn): GELU(approximate='none')
        (linear_fc2): Linear(in_features=4096, out_features=2048, bias=True)
      )
      (deepstack_merger_list): ModuleList(
        (0-2): 3 x Qwen3VLVisionPatchMerger(
          (norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
          (linear_fc1): Linear(in_features=4096, out_features=4096, bias=True)
          (act_fn): GELU(approximate='none')
          (linear_fc2): Linear(in_features=4096, out_features=2048, bias=True)
        )
      )
    )
    (language_model): Qwen3VLTextModel(
      (embed_tokens): Embedding(151936, 2048)
      (layers): ModuleList(
        (0-27): 28 x Qwen3VLTextDecoderLayer(
          (self_attn): Qwen3VLTextAttention(
            (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (k_proj): Linear(in_features=2048, out_features=1024, bias=False)
            (v_proj): Linear(in_features=2048, out_features=1024, bias=False)
            (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (q_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
            (k_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
          )
          (mlp): Qwen3VLTextMLP(
            (gate_proj): Linear(in_features=2048, out_features=6144, bias=False)
            (up_proj): Linear(in_features=2048, out_features=6144, bias=False)
            (down_proj): Linear(in_features=6144, out_features=2048, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): Qwen3VLTextRMSNorm((2048,), eps=1e-06)
          (post_attention_layernorm): Qwen3VLTextRMSNorm((2048,), eps=1e-06)
        )
      )
      (norm): Qwen3VLTextRMSNorm((2048,), eps=1e-06)
      (rotary_emb): Qwen3VLTextRotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)