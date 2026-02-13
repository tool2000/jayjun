"""
Chain-of-Thought math dataset for reasoning training.
Uses nvidia/OpenMathInstruct-2 from HuggingFace.

Each example is a math problem with detailed step-by-step solution.
Converted to conversation format for mid-training and SFT.

Usage:
    from tasks.cot_math import CoTMath

    # Use CoT math dataset
    train_ds = CoTMath(split="train", stop=30000)
"""

import re
from tasks.common import Task

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class CoTMath(Task):
    """
    CoT math problems with step-by-step solutions.
    Uses nvidia/OpenMathInstruct-2 (train_1M split for manageable size).

    Falls back to GSM8K with full chain-of-thought answers if OpenMathInstruct
    is not available.
    """

    def __init__(self, split="train", dataset_name="nvidia/OpenMathInstruct-2", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "split must be train|test"

        if not HAS_DATASETS:
            raise ImportError("Please install datasets: pip install datasets")

        self._fallback = False

        try:
            # OpenMathInstruct-2 has a train_1M split that's a 1M subset
            ds_split = "train_1M" if split == "train" else "train_1M"
            self.ds = load_dataset(dataset_name, split=ds_split, streaming=False).shuffle(seed=42)
            self.length = len(self.ds)
        except Exception as e:
            print(f"Warning: Could not load {dataset_name}: {e}")
            print("Falling back to gsm8k for CoT data.")
            # Fallback: use GSM8K with full chain-of-thought answers
            self.ds = load_dataset("openai/gsm8k", "main", split=split).shuffle(seed=42)
            self.length = len(self.ds)
            self._fallback = True

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]

        if self._fallback:
            # GSM8K fallback: use full answer text as CoT
            question = row["question"]
            answer = row["answer"].replace("####", "The answer is")
            # Remove <<calc>> tags for cleaner CoT
            answer = re.sub(r'<<[^>]+>>', '', answer)
        else:
            # OpenMathInstruct-2 format
            question = row.get("problem", row.get("question", ""))
            # generated_solution contains step-by-step reasoning
            answer = row.get("generated_solution", row.get("solution", ""))

        messages = [
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": str(answer)},
        ]
        return {"messages": messages}
