# NANOCHAT/ CORE LIBRARY

## OVERVIEW
Core transformer library: GPT model, tokenizers, optimizers, inference engine, training utilities.

## WHERE TO LOOK
| Component | File | Key Details |
|-----------|------|-------------|
| Model architecture | gpt.py | GPTConfig, GPT, Block, CausalSelfAttention, MLP |
| Inference engine | engine.py | KVCache, Engine.generate(), calculator tool, streaming |
| Tokenizer | tokenizer.py | BPE, SPECIAL_TOKENS, RustBPETokenizer/HuggingFaceTokenizer |
| Optimizers | muon.py, adamw.py | Muon (matrix), DistAdamW (non-matrix) |
| Attention | flash_attention.py | FA3/SDPA auto-detect, sliding window support |
| Data pipeline | dataloader.py, dataset.py | Distributed tokenization, FineWeb-Edu shards |

## CONVENTIONS
- Meta device init in `GPT.__init__()` - shapes/dtypes only; real init in `init_weights()`
- No learnable params in `norm()` - purely functional RMS norm
- FA3 auto-detects on Hopper+ (sm90), falls back to SDPA elsewhere
- DDP wrappers: DistMuon, DistAdamW (ZeRO-2 style sharding)

## ANTI-PATTERNS
- Using meta device tensors for actual data - they're only shapes, call `init_weights()` first
- Expecting rotary embeddings to grow dynamically - pre-computed 10X seq_len (TODO)
- Using Muon on embeddings/lm_head - only for 2D matrix parameters

## UNIQUE STYLES
**Meta Device Initialization:**
- All model creation happens in `torch.device("meta")` context
- `init_weights()` called separately for real initialization (uniform, zeros, normal distributions)
- Vocab padded to multiple of 64 for DDP efficiency (outputs cropped in forward())

**Sliding Window Pattern:**
- Pattern string tiled across layers: `window_pattern="SSSL"` (L=full, S=half context)
- Final layer always gets full context regardless of pattern
- Per-layer window sizes: `(-1, 0)` for full, `(seq_len/2, 0)` for half

**Mixed Optimizer Strategy:**
- Muon for matrix parameters (linear layers) - Newton-Schulz orthogonalization
- AdamW for embeddings, lm_head, value_embeds, per-layer scalars
- LRs scaled by `√(model_dim/768)` for AdamW params
- `setup_optimizers()` returns `[adamw_optimizer, muon_optimizer]`

**Per-Layer Scalers (modded-nanogpt):**
- `resid_lambdas`: scales residual stream (init=1.0)
- `x0_lambdas`: blends initial embedding back in (init=0.0)

**Value Embeddings (ResFormer):**
- Alternating layers: `has_ve(layer_idx, n_layer) = layer_idx % 2 == (n_layer - 1) % 2`
- Gate: `v = v + sigmoid(gate(x)) * ve` per key/value head

**Logit Softcap:**
- `15.0 * tanh(logits / 15.0)` to squash logits before cross-entropy
