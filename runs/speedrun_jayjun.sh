#!/bin/bash

# ============================================================================
# speedrun_jayjun.sh - Bilingual ChatGPT Clone "JayJun" (제이준)
# ============================================================================
# 
# This script trains a bilingual (Korean + English) ChatGPT clone named "JayJun"
# Based on the original speedrun.sh but with:
# - 70% English + 30% Korean pretraining data
# - JayJun identity (제이준, created by "준이 아빠")
# - Bilingual tokenizer trained on mixed data
#
# Target: ~$100 tier (~4 hours on 8XH100)
# Model: d20 (561M parameters)
#
# Usage:
#   bash runs/speedrun_jayjun.sh
#
# Or in a screen session:
#   screen -L -Logfile jayjun.log -S jayjun bash runs/speedrun_jayjun.sh
#
# ============================================================================

# Default intermediate artifacts directory
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Set language ratio for bilingual training (70% English, 30% Korean)
export NANOCHAT_LANG_RATIO=0.7

# -----------------------------------------------------------------------------
# Python venv setup with uv

# Install uv if not already installed
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# Create .venv if it doesn't exist
[ -d ".venv" ] || uv venv
# Install dependencies
uv sync --extra gpu
# Activate venv
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup (optional)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Reset report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Download datasets (English + Korean)

echo "=============================================="
echo "Step 1: Downloading datasets"
echo "=============================================="

# Download English dataset (same as speedrun.sh)
# For d20: 561M params * 20 = 11.2B tokens
# At 4.8 chars/token = 54B chars
# At 250M chars/shard = 216 shards (round up to 240)
# With 35% waste = 370 shards
python -m nanochat.dataset -n 8  # First 8 shards for tokenizer training

# Start downloading more English shards in background
python -m nanochat.dataset -n 370 &
ENGLISH_DOWNLOAD_PID=$!

# Download Korean dataset (for 30% of training)
# Need ~30% of English data = ~111 shards worth
# Korean dataset from HuggingFace (eliceai/korean-fineweb-edu-demo)
echo "Downloading Korean dataset..."
python -m nanochat.dataset --korean &
KOREAN_DOWNLOAD_PID=$!

# -----------------------------------------------------------------------------
# Train bilingual tokenizer

echo "=============================================="
echo "Step 2: Training bilingual tokenizer"
echo "=============================================="

# Wait for initial English data
wait $ENGLISH_DOWNLOAD_PID 2>/dev/null || true

# Train tokenizer on mixed English + Korean data (70:30 ratio)
python -m scripts.tok_train --max-chars=2000000000 --vocab-size=65536 --en-ratio=0.7

# Evaluate tokenizer
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Generate JayJun identity data

echo "=============================================="
echo "Step 3: Generating JayJun identity data"
echo "=============================================="

# Generate JayJun identity conversations (template-based, no API needed)
python -m dev.gen_jayjun_identity --num-conversations=1000 --use-templates

# Also download the original identity file as backup
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl 2>/dev/null || true

# -----------------------------------------------------------------------------
# Wait for all downloads to complete

echo "Waiting for dataset downloads to complete..."
wait $KOREAN_DOWNLOAD_PID 2>/dev/null || true
wait $ENGLISH_DOWNLOAD_PID 2>/dev/null || true

# -----------------------------------------------------------------------------
# Base model pretraining

echo "=============================================="
echo "Step 4: Pretraining d20 model (bilingual)"
echo "=============================================="

# Number of GPUs
NPROC_PER_NODE=8

# Pretrain the d20 model
# Note: The dataloader will use NANOCHAT_LANG_RATIO for 70:30 mixing
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
    --depth=20 \
    --target-param-data-ratio=20 \
    --run=$WANDB_RUN

# Evaluate base model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (teach JayJun identity, conversation format, tool use)

echo "=============================================="
echo "Step 5: Midtraining (JayJun identity)"
echo "=============================================="

# Midtraining will automatically use jayjun_identity_conversations.jsonl if available
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Fine-Tuning

echo "=============================================="
echo "Step 6: Supervised Fine-Tuning (SFT)"
echo "=============================================="

# SFT will automatically include Korean data if available
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Generate final report

echo "=============================================="
echo "Step 7: Generating report"
echo "=============================================="

python -m nanochat.report generate

# -----------------------------------------------------------------------------
# Done!

echo "=============================================="
echo "JayJun (제이준) training complete!"
echo "=============================================="
echo ""
echo "To chat with JayJun:"
echo "  python -m scripts.chat_web"
echo ""
echo "Or via CLI:"
echo "  python -m scripts.chat_cli -p '안녕하세요! 당신은 누구예요?'"
echo ""
echo "Report saved to: report.md"
