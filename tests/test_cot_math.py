"""Test CoT math dataset. Run: python -m pytest tests/test_cot_math.py -v"""

from tasks.cot_math import CoTMath


def test_cot_math_loads():
    """CoTMath loads and returns conversation-format examples."""
    ds = CoTMath(split="train", stop=10)
    assert len(ds) > 0
    assert len(ds) <= 10


def test_cot_math_conversation_format():
    """Each example has messages with user question and assistant CoT answer."""
    ds = CoTMath(split="train", stop=10)
    example = ds[0]
    assert "messages" in example
    messages = example["messages"]
    assert len(messages) >= 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    # Assistant answer should be detailed, not just a number
    answer = messages[1]["content"]
    assert isinstance(answer, str)
    assert len(answer) > 20, "CoT answer should be detailed, not just a number"
