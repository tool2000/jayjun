"""
Korean conversational datasets for bilingual training.

Supports multiple Korean chat datasets from HuggingFace:
1. maywell/ko_wikidata_QA - Korean Wikipedia QA pairs
2. heegyu/korquad-chat-v1 - Korean QA in chat format
3. Custom JSONL files for Korean conversations

Usage:
    from tasks.korean_chat import KoreanChat, KoreanQA

    # Use Korean QA dataset
    train_ds = KoreanQA(split="train", stop=10000)

    # Use custom Korean JSONL
    custom_ds = KoreanChat(filepath="path/to/korean_conversations.jsonl")
"""

import os
import json
from tasks.common import Task

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class KoreanQA(Task):
    """
    Korean QA dataset converted to conversation format.
    Uses maywell/ko_wikidata_QA from HuggingFace.

    Each example becomes a 2-turn conversation:
    - User: question
    - Assistant: answer
    """

    def __init__(self, split="train", dataset_name="maywell/ko_wikidata_QA", **kwargs):
        super().__init__(**kwargs)

        if not HAS_DATASETS:
            raise ImportError("Please install datasets: pip install datasets")

        assert split in ["train", "test"], "split must be train|test"

        # Load dataset
        try:
            self.ds = load_dataset(dataset_name, split=split).shuffle(seed=42)
        except Exception as e:
            print(f"Warning: Could not load {dataset_name}: {e}")
            print("Creating empty dataset.")
            self.ds = []

        self.length = len(self.ds) if self.ds else 0

    def num_examples(self):
        return self.length

    def get_example(self, index):
        if not self.ds:
            return {"messages": []}

        row = self.ds[index]

        # Handle different dataset formats
        if "question" in row and "answer" in row:
            question = row["question"]
            answer = row["answer"]
        elif "instruction" in row and "output" in row:
            question = row["instruction"]
            answer = row["output"]
        elif "input" in row and "output" in row:
            question = row["input"]
            answer = row["output"]
        else:
            # Fallback: try to find any suitable fields
            question = str(
                row.get("question", row.get("input", row.get("instruction", "")))
            )
            answer = str(row.get("answer", row.get("output", row.get("response", ""))))

        messages = [
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": str(answer)},
        ]

        return {"messages": messages}


class KoreanChat(Task):
    """
    Load Korean conversations from a JSONL file.
    Same format as CustomJSON but specifically for Korean data.

    Each line should be a JSON array of message objects:
    [{"role":"user","content":"안녕"},{"role":"assistant","content":"안녕하세요!"}]
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        if not os.path.exists(filepath):
            print(f"Warning: Korean chat file {filepath} does not exist")
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        messages = json.loads(line)
                        if isinstance(messages, list) and len(messages) >= 2:
                            self.conversations.append(messages)
                    except json.JSONDecodeError:
                        continue

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        return {"messages": messages}


class KoreanSmolTalk(Task):
    """
    Korean version of SmolTalk-style conversations.
    Uses heegyu/korquad-chat-v1 or similar datasets.

    Falls back gracefully if dataset is not available.
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)

        if not HAS_DATASETS:
            raise ImportError("Please install datasets: pip install datasets")

        self.conversations = []

        # Try multiple Korean conversation datasets
        dataset_candidates = [
            ("heegyu/korquad-chat-v1", "train"),
            ("nlpai-lab/kullm-v2", "train"),
        ]

        for dataset_name, ds_split in dataset_candidates:
            try:
                ds = load_dataset(dataset_name, split=ds_split).shuffle(seed=42)
                print(f"Loaded Korean dataset: {dataset_name} ({len(ds)} examples)")

                for row in ds:
                    messages = self._extract_messages(row)
                    if messages and len(messages) >= 2:
                        self.conversations.append(messages)

                if self.conversations:
                    break
            except Exception as e:
                print(f"Could not load {dataset_name}: {e}")
                continue

        self.length = len(self.conversations)
        if self.length == 0:
            print("Warning: No Korean conversation data loaded")

    def _extract_messages(self, row):
        """Extract messages from various dataset formats."""
        messages = []

        # Format 1: messages list
        if "messages" in row:
            return row["messages"]

        # Format 2: conversation list
        if "conversation" in row:
            return row["conversation"]

        # Format 3: instruction/output pairs
        if "instruction" in row and "output" in row:
            instruction = (
                row.get("input", "") + " " + row["instruction"]
                if row.get("input")
                else row["instruction"]
            )
            messages = [
                {"role": "user", "content": instruction.strip()},
                {"role": "assistant", "content": row["output"]},
            ]

        # Format 4: question/answer pairs
        elif "question" in row and "answer" in row:
            messages = [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ]

        return messages

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        return {"messages": messages}


class KoreanMMLU(Task):
    """
    Korean MMLU (KMMLU) for Korean reasoning and knowledge.
    Uses HAERAE-HUB/KMMLU from HuggingFace.

    Converts multiple-choice questions to conversation format
    with the correct answer.
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)

        if not HAS_DATASETS:
            raise ImportError("Please install datasets: pip install datasets")

        self.conversations = []

        try:
            # KMMLU has multiple subjects; load "all" config
            ds = load_dataset("HAERAE-HUB/KMMLU", "all", split=split).shuffle(seed=42)

            for row in ds:
                question = row.get("question", "")
                choices = []
                for key in ["A", "B", "C", "D"]:
                    if key in row:
                        choices.append(f"{key}. {row[key]}")

                answer_key = str(row.get("answer", ""))
                # Build conversation: question with choices -> answer
                user_content = question
                if choices:
                    user_content += "\n" + "\n".join(choices)

                # The answer is the correct choice letter and text
                answer_text = answer_key
                for key in ["A", "B", "C", "D"]:
                    if key == answer_key and key in row:
                        answer_text = f"{key}. {row[key]}"
                        break

                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer_text},
                ]
                self.conversations.append(messages)

            print(f"Loaded KMMLU: {len(self.conversations)} examples")
        except Exception as e:
            print(f"Could not load KMMLU: {e}")

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        return {"messages": messages}
