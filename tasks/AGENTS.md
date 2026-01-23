# TASKS EVALUATION SYSTEM

**Generated:** 2026-01-23 13:52:56
**Commit:** (not available)
**Branch:** (not available)

## OVERVIEW
Eight evaluation tasks for model benchmarking (ARC, GSM8K, MMLU, HumanEval, SmolTalk, SpellingBee, CustomJSON).

## STRUCTURE
```
tasks/
├── common.py           # Task base class, TaskMixture, TaskSequence
├── arc.py              # ARC-Easy, ARC-Challenge (categorical)
├── gsm8k.py            # Grade school math (generative, tool calls)
├── mmlu.py             # 57 subjects multiple choice (categorical)
├── humaneval.py        # Python coding benchmark (generative)
├── smoltalk.py         # HuggingFace conversations
├── spellingbee.py      # Letter counting task
└── customjson.py       # Load from arbitrary JSONL files
```

## WHERE TO LOOK
| Component | Location | Notes |
|-----------|----------|-------|
| Task base class | tasks/common.py | Slicing support, eval_type, get_example, evaluate |
| TaskMixture | tasks/common.py | Deterministic shuffle (seed=42), oversample by repeating |
| TaskSequence | tasks/common.py | Sequential curriculum training |
| Multiple choice rendering | tasks/common.py/render_mc() | Letter after choice, no whitespace before letter |
| GSM8K tool calls | tasks/gsm8k.py | <<expr>> syntax, python/python_output parts |
| MMLU subjects | tasks/mmlu.py | 57 subjects, letters=(A,B,C,D) |
| HumanEval extraction | tasks/humaneval.py | extract_program() handles markdown blocks |
| RL rewards | tasks/gsm8k.py/reward() | Optional, returns float |

## CONVENTIONS
- **Slicing**: All tasks support start/stop/step via Task.__init__
- **eval_type**: 'categorical' (multiple choice) or 'generative' (text/code)
- **Conversation format**: Dict with 'messages' list of {role, content} dicts
- **Content types**: String (simple) or list of parts (tool calls) for assistant messages
- **Tool calls**: [{"type": "python", "text": expr}, {"type": "python_output", "text": result}]
- **Evaluation**: evaluate(conversation, assistant_response) → bool/int
- **Dataset loading**: HuggingFace datasets, shuffle(seed=42) for reproducibility
- **Multiple choice**: render_mc() uses "- {choice}={letter}\n" format
- **Assistant response**: Letter only for categorical tasks ("A" not " A")

## ANTI-PATTERNS
- Mixing categorical/generative tasks in TaskMixture without handling eval_type differences
- Changing render_mc format (letter position affects token binding for small models)
- Whitespace before letters in assistant responses (tokenizer differentiates "A" vs " A")
- Using TaskSequence for training on mixed data (should use TaskMixture)
- Forgetting to shuffle datasets with fixed seed (reproducibility breaks)
- Not clamping predictions to valid choices for categorical tasks
- Ignoring tool call structure in GSM8K (<<expr>> format is critical)

## UNIQUE STYLES
**TaskMixture shuffle**: Creates flattened (task_idx, local_idx) pairs, shuffles with seed=42 for deterministic mixing regardless of task sizes.

**GSM8K tool parsing**: Splits on <<expr>> delimiters, creates python/python_output parts, extracts final answer after #### marker.

**HumanEval extraction**: Extracts imports from prompt, combines with completion and test, executes via nanochat.execution.execute_code().

**Multiple choice format**: Letter after choice ("- Choice 1=A") with no whitespace for better small model token binding.

**Categorical evaluation**: Asserts response is in valid letters set, compares to ground truth directly.

**Slicing math**: __len__ computes ceil((stop-start)/step), __getitem__ maps logical index to physical (start + index*step).

**RL rewards**: Optional reward() method mirrors evaluate() but returns float (0.0 or 1.0) for RLHF training.
