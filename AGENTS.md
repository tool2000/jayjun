# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-23 13:52:56
**Commit:** (not available)
**Branch:** (not available)

## OVERVIEW
nanochat is a full-stack ChatGPT clone designed to train on a single 8XH100 node for ~4 hours ($100 budget). Implements entire LLM pipeline: tokenization → pretraining → midtraining → SFT → RLHF → evaluation → inference → web serving.

**Stack:** Python 3.10+, PyTorch 2.9+, uv package manager, Flash Attention 3, Muon+AdamW optimizers

## STRUCTURE
```
./
├── nanochat/      # Core library (GPT model, tokenizer, engine, optimizers)
├── scripts/        # Training/eval/serving entry points
├── tasks/          # Evaluation tasks (ARC, GSM8K, MMLU, HumanEval, etc.)
├── runs/           # End-to-end training recipes (speedrun.sh, run1000.sh)
├── tests/          # Pytest tests (engine, attention fallback)
└── dev/            # Development utilities (data generation, logs)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Model architecture | nanochat/gpt.py | GPTConfig, GPT, Block, CausalSelfAttention |
| Inference engine | nanochat/engine.py | KVCache, sampling, calculator tool |
| Tokenizer | nanochat/tokenizer.py | BPE, SPECIAL_TOKENS (do not change) |
| Training scripts | scripts/base_train.py, scripts/mid_train.py, scripts/chat_sft.py | Base pretraining, midtraining, SFT |
| Web serving | scripts/chat_web.py | FastAPI + uvicorn |
| Evaluation | scripts/chat_eval.py, tasks/ | ARC, GSM8K, MMLU, HumanEval |
| Data utilities | nanochat/dataset.py | FineWeb-Edu parquet shards |
| Optimizers | nanochat/muon.py, nanochat/adamw.py | Muon for matrices, AdamW for embeddings |

## CONVENTIONS
- No formatter enforced (no Black/Ruff config)
- 4-space indents
- Imports: stdlib → third-party → local
- Classes: PascalCase, functions: snake_case
- Use `assert` for internal invariants, raise `ValueError` for API misuse
- CLI scripts use `print` for progress; library modules use `nanochat.common.logger`
- Meta device init in GPT.__init__ - actual init in `init_weights()`
- Last parquet shard = validation (`nanochat.dataset.parquets_iter_batched`)
- Empty `nanochat/__init__.py` - import directly from submodules
- Vocab changes break checkpoint compatibility

## ANTI-PATTERNS (THIS PROJECT)
- Changing `SPECIAL_TOKENS` in tokenizer.py - model checkpoints become incompatible
- Using `--window-pattern` without FA3 - GPU utilization is terrible (SDPA has no sliding window support)
- Modifying special tokens without retraining tokenizer from scratch
- Overriding device/dtype assumptions (code assumes cuda→bfloat16, else→float32)

## UNIQUE STYLES
**Mixed Optimizer Strategy:**
- Muon for matrix parameters (linear layers)
- AdamW for embeddings, lm_head, scalars
- Different learning rates per component, scaled by ∝1/√(model_dim/768)

**Meta Device Initialization:**
- GPT.__init__ runs in `torch.device("meta")` context
- All tensors are shapes/dtypes only, no actual data
- `init_weights()` called separately for real initialization

**Sliding Window Pattern:**
- Pattern string tiled across layers: L=full context, S=half context
- Example: "SSSL" = two short, one short, last layer full
- Config: `--window-pattern` flag

**Checkpoint Tagging:**
- Models identified by depth: d20, d26, d32, d34
- Directory: `base_checkpoints/d{depth}/`
- Each checkpoint saves model+optimizer state+dataloader state+loop state

## COMMANDS
```bash
# Environment setup
uv venv
uv sync --extra gpu  # or --extra cpu
source .venv/bin/activate

# Tests
python -m pytest
python -m pytest tests/test_engine.py -v -s

# Training
python -m scripts.base_train --depth=20 --target-param-data-ratio=4
python -m scripts.mid_train
python -m scripts.chat_sft

# Serving
python -m scripts.chat_web  # Visit http://IP:8000/
```

## NOTES
- **FA3 Detection:** Auto-detects on Hopper+ (sm90), falls back to SDPA elsewhere
- **DDP:** Use `torchrun --standalone --nproc_per_node=N -m scripts.base_train`
- **GPU Memory:** Tune via `--device-batch-size` (reduce to avoid OOM)
- **WandB:** Disabled by default (`--run=dummy`), enable with wandb run name
- **Batch Size Scaling:** LRs scaled by √(batch_size / 2^19) for non-standard batch sizes
- **Weight Decay Scaling:** Scaled by (12/depth)² for models other than d12
