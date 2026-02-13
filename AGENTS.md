# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-23 13:52:56
**Commit:** (not available)
**Branch:** (not available)

## OVERVIEW
nanochat is a full-stack ChatGPT clone implementing the entire LLM pipeline: tokenization → pretraining → midtraining → SFT → RLHF → evaluation → inference → web serving.

**JayJun variant:** Bilingual (Korean + English) 0.5B model with GQA 2:1, depth=20, trained with ratio=40 (~19.3B tokens) on 1x A100-96GB in ~3 days.

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
| Model architecture | nanochat/gpt.py | GPTConfig, GPT, Block, CausalSelfAttention, GQA support via n_kv_head |
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

<skills_system priority="1">

## Available Skills

<!-- SKILLS_TABLE_START -->
<usage>
When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively. Skills provide specialized capabilities and domain knowledge.

How to use skills:
- Invoke: `npx openskills read <skill-name>` (run in your shell)
  - For multiple: `npx openskills read skill-one,skill-two`
- The skill content will load with detailed instructions on how to complete the task
- Base directory provided in output for resolving bundled resources (references/, scripts/, assets/)

Usage notes:
- Only use skills listed in <available_skills> below
- Do not invoke a skill that is already loaded in your context
- Each skill invocation is stateless
</usage>

<available_skills>

<skill>
<name>composition-patterns</name>
<description>React composition patterns that scale. Use when refactoring components with</description>
<location>project</location>
</skill>

<skill>
<name>react-best-practices</name>
<description>React and Next.js performance optimization guidelines from Vercel Engineering. This skill should be used when writing, reviewing, or refactoring React/Next.js code to ensure optimal performance patterns. Triggers on tasks involving React components, Next.js pages, data fetching, bundle optimization, or performance improvements.</description>
<location>project</location>
</skill>

<skill>
<name>react-native-skills</name>
<description>React Native and Expo best practices for building performant mobile apps. Use</description>
<location>project</location>
</skill>

<skill>
<name>vercel-deploy-claimable</name>
<description>Deploy applications and websites to Vercel. Use this skill when the user requests deployment actions such as "Deploy my app", "Deploy this to production", "Create a preview deployment", "Deploy and give me the link", or "Push this live". No authentication required - returns preview URL and claimable deployment link.</description>
<location>project</location>
</skill>

<skill>
<name>web-design-guidelines</name>
<description>Review UI code for Web Interface Guidelines compliance. Use when asked to "review my UI", "check accessibility", "audit design", "review UX", or "check my site against best practices".</description>
<location>project</location>
</skill>

</available_skills>
<!-- SKILLS_TABLE_END -->

</skills_system>
