# PROJECT KNOWLEDGE BASE: scripts/

## OVERVIEW
Entry point scripts for training pipeline: pretraining → midtraining → SFT → RL → evaluation → serving.

## WHERE TO LOOK
| Script | Purpose | Key Args |
|--------|---------|----------|
| base_train.py | Pretrain from scratch | --target-param-data-ratio, --depth, --window-pattern |
| mid_train.py | Teach special tokens, tool use, multiple choice | --model-step, --num-iterations |
| chat_sft.py | SFT on SmolTalk conversations | --source, --num-epochs |
| chat_rl.py | RLHF (PPO) | --source, --num-iterations |
| base_eval.py | CORE benchmark | (no args - loads latest) |
| chat_eval.py | ARC, GSM8K, MMLU, HumanEval | --source, --task |
| chat_web.py | FastAPI web server | --num-gpus, --source |
| chat_cli.py | CLI interactive chat | --source, --temperature |

## CONVENTIONS
- **Execution**: All scripts run as modules: `python -m scripts.base_train`
- **CLI**: argparse only (no YAML/JSON configs) - use `--help` for options
- **Distributed**: `torchrun --standalone --nproc_per_node=N -m scripts.XXX`
- **Base train structure**: compute_init → wandb init → model init (meta device) → optimizer → dataloader → training loop
- **Training scripts**: Load checkpoint → torch.compile(model, dynamic=False) → train → save
- **Evaluation scripts**: Use original uncompiled model (inputs change shape)
- **Batch LR scaling**: √(batch_size / 2^19) for non-standard batch sizes
- **Weight decay scaling**: (12/depth)² for models other than d12

## ANTI-PATTERNS
- Using `--window-pattern` without FA3 (GPU utilization is terrible)
- Skipping `torch.compile` on training (slow training)
- Running compiled model for eval (fails when inputs change shape)
- Not using `--dry-run` for testing pipeline without checkpointing

## UNIQUE STYLES
**Checkpoint-based Resume**: All training scripts support `--resume-from-step` (or inferred from model metadata). Checkpoints saved via `nanochat.checkpoint_manager.save_checkpoint()` include model, optimizer state, dataloader state, and loop state.

**Horizon Specification**: base_train accepts three mutually exclusive stopping conditions (in precedence order): `--num-iterations` (explicit), `--target-flops` (FLOP budget), `--target-param-data-ratio` (Chinchilla-optimal data for given params).

**Module Pattern**: Scripts executed via `python -m scripts.XXX` (not `python scripts/XXX.py`). This ensures proper imports and relative imports work correctly across the project.
