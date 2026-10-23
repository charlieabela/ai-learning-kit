# Language models in ai-learning-kit

Nothing in the kit requires a language model. Every check and every piece of
feedback is deterministic and comes from reference values computed from the
data. A model, when available, adds two things for students, `kit.hint(task)`
(one next step, never the answer) and `kit.ask(task, question)` (a concept
explained in the task's context), and gives session authors `llm.complete()` for
their own uses. The model never grades: the answer book records only the kit's
checks.

## Configuring

In the notebook:

```python
from ai_learning_kit import llm
llm.configure("ollama", model="llama3.1")                               # a local open-weight model
llm.configure("openai-compatible", model="Qwen2.5-7B-Instruct",
              base_url="http://gpu.server.local:8000/v1", api_key="...")  # vLLM / LM Studio / a department server
llm.configure("anthropic", model="claude-sonnet-4-5", api_key="...")
llm.configure("none")                                                   # switch help off
```

Or with environment variables, which Colab's Secrets panel also provides
(`AILK_LLM_PROVIDER`, `AILK_LLM_MODEL`, `AILK_LLM_BASE_URL`, `AILK_LLM_API_KEY`).

Without any of these, the kit autodetects: `ANTHROPIC_API_KEY`, then
`OPENAI_API_KEY`, then a running Ollama server (`OLLAMA_HOST` or
`http://localhost:11434`) with at least one installed model, else none.
`llm.describe()` says what was chosen.

## Open-weight models

Any server that speaks the OpenAI chat-completions API works, so an open-weight
model is a base URL, not new code: Ollama (`http://host:11434/v1`), vLLM,
LM Studio, llama.cpp's server, Groq, Together. In Colab a small model can run in
the runtime itself behind Ollama if the lecturer provides a setup cell; on a
department server, students only need the URL.

## Adding a provider

Subclass `ai_learning_kit.llm.providers.Provider`, implement `complete()`, and
add it to `PROVIDERS` in `ai_learning_kit/llm/__init__.py`. No other code changes.

## What the tutor prompt enforces

The system prompt tells the model it is a teaching assistant, must not give the
final answer or the completing code, gives one next step in at most 120 words,
refers to the task guidance it is given, and must not invent facts about the data.
`hint()` passes the task's guide, the student's last saved value and reasoning,
and the kit's verdict; `ask()` passes the guide and the problem summary. A
student can still coax a model into more than a hint; because nothing is graded
by the model and the deterministic feedback already reveals the answer after
three attempts, this costs the student learning, not marks.
