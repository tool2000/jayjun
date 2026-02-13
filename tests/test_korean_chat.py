"""Test Korean chat datasets. Run: python -m pytest tests/test_korean_chat.py -v"""

from tasks.korean_chat import KoreanMMLU


def test_korean_mmlu_loads():
    """KoreanMMLU loads and returns conversation-format examples."""
    ds = KoreanMMLU(split="train", stop=10)
    assert len(ds) > 0
    assert len(ds) <= 10


def test_korean_mmlu_conversation_format():
    """Each example has messages with user question and assistant answer."""
    ds = KoreanMMLU(split="train", stop=10)
    example = ds[0]
    assert "messages" in example
    messages = example["messages"]
    assert len(messages) >= 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
