"""Optional language-model support.

Nothing in the kit requires a language model: every check and every piece of
feedback is deterministic. When a model is available, two student-facing helpers
use it, kit.hint() and kit.ask(), and session authors may use complete() for
their own purposes. The model never grades: the answer book records only the
kit's own checks.

Configuration, in order of precedence:
1. configure(provider=..., model=..., base_url=..., api_key=...) in the notebook;
2. environment variables (also read from Colab's Secrets panel):
      AILK_LLM_PROVIDER   "gemini" | "openai-compatible" | "openai" | "anthropic" | "ollama" | "none"
      AILK_LLM_MODEL      model name
      AILK_LLM_BASE_URL   for OpenAI-compatible servers (default https://api.openai.com/v1)
      AILK_LLM_API_KEY    key for that server, if it needs one
3. autodetection: GEMINI_API_KEY or GOOGLE_API_KEY → Gemini; ANTHROPIC_API_KEY →
   Anthropic; OPENAI_API_KEY with AILK_LLM_MODEL → OpenAI; an Ollama server at
   OLLAMA_HOST or http://localhost:11434 with at least one model → Ollama;
   otherwise none.

Inside Notebook Tutor, the tutor sets these variables so that the kit's requests go
through the tutor's own connection: the student's key never enters the notebook.

Open-weight models therefore need no code: run them behind an OpenAI-compatible
endpoint (Ollama, vLLM, LM Studio, a department server) and point AILK_LLM_BASE_URL
at it. See docs/LLM.md.
"""
from __future__ import annotations

import os

from .providers import Anthropic, Gemini, LLMError, Null, OpenAICompatible, Provider, ollama_models

# Defaults only; model names change, so front ends should offer list_models() instead.
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-haiku-5-5",
}
PROVIDERS = {"gemini": Gemini, "openai-compatible": OpenAICompatible, "openai": OpenAICompatible,
             "ollama": OpenAICompatible, "anthropic": Anthropic, "none": Null}

_provider: Provider | None = None


# --------------------------------------------------------------------------- settings
def _secret(name: str) -> str | None:
    """An environment variable, or a Colab secret of the same name."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(name) or None
    except Exception:
        return None


def configure(provider: str | None = None, model: str | None = None, base_url: str | None = None,
              api_key: str | None = None, **extra) -> Provider:
    """Choose the language model for this session, or pass provider="none" to use none.

    Examples
        llm.configure("gemini", api_key="...")                          # a key from Google AI Studio
        llm.configure("ollama", model="llama3.1")                       # local Ollama
        llm.configure("openai-compatible", model="Qwen2.5-7B-Instruct",
                      base_url="http://gpu.server.local:8000/v1")       # vLLM on a department server
        llm.configure("anthropic", model="claude-sonnet-4-5", api_key="...")
    """
    global _provider
    key = (provider or "none").lower()
    if key not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Choose one of: {', '.join(PROVIDERS)}")
    if key == "none":
        _provider = Null("language-model help was switched off with configure('none')")
    elif key == "gemini":
        api_key = api_key or _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY") or _secret("AILK_LLM_API_KEY")
        if not api_key:
            raise ValueError("Gemini needs an API key: create one at https://aistudio.google.com/apikey, then pass "
                             "api_key=... or set GEMINI_API_KEY.")
        _provider = Gemini(model or DEFAULT_MODELS["gemini"], api_key, **({"base_url": base_url} if base_url else {}))
    elif key == "anthropic":
        api_key = api_key or _secret("ANTHROPIC_API_KEY") or _secret("AILK_LLM_API_KEY")
        if not api_key:
            raise ValueError("Anthropic needs an API key: pass api_key=... or set ANTHROPIC_API_KEY.")
        _provider = Anthropic(model or DEFAULT_MODELS["anthropic"], api_key, **({"base_url": base_url} if base_url else {}))
    elif key == "ollama":
        host = (base_url or _secret("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        host = host[:-3] if host.endswith("/v1") else host
        if not model:
            models = ollama_models(host)
            if not models:
                raise ValueError(f"No model given and no Ollama models found at {host}. Run `ollama pull <model>`.")
            model = models[0]
        _provider = OpenAICompatible(model, base_url=f"{host}/v1", api_key=api_key or "ollama", label="ollama")
    else:  # openai-compatible, openai
        api_key = api_key or _secret("AILK_LLM_API_KEY") or _secret("OPENAI_API_KEY")
        base_url = base_url or _secret("AILK_LLM_BASE_URL") or "https://api.openai.com/v1"
        model = model or _secret("AILK_LLM_MODEL")
        if not model:
            raise ValueError("Give a model name: configure(..., model='...') or set AILK_LLM_MODEL. "
                             "list_models() shows what the server offers.")
        _provider = OpenAICompatible(model, base_url=base_url, api_key=api_key,
                                     label="openai" if "api.openai.com" in base_url else "openai-compatible")
    return _provider


def autodetect() -> Provider:
    """Pick a provider from the environment, without network calls except an Ollama ping."""
    choice = (_secret("AILK_LLM_PROVIDER") or "").lower()
    if choice:
        return configure(choice, model=_secret("AILK_LLM_MODEL"), base_url=_secret("AILK_LLM_BASE_URL"),
                         api_key=_secret("AILK_LLM_API_KEY"))
    if _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY"):
        return configure("gemini", model=_secret("AILK_LLM_MODEL"))
    if _secret("ANTHROPIC_API_KEY"):
        return configure("anthropic", model=_secret("AILK_LLM_MODEL"))
    if _secret("OPENAI_API_KEY") and _secret("AILK_LLM_MODEL"):
        return configure("openai", model=_secret("AILK_LLM_MODEL"))
    host = _secret("OLLAMA_HOST") or "http://localhost:11434"
    models = ollama_models(host)
    if models:
        return configure("ollama", model=_secret("AILK_LLM_MODEL") or models[0], base_url=host)
    return Null("no language model is configured (see ai_learning_kit.llm or docs/LLM.md)")


def get() -> Provider:
    global _provider
    if _provider is None:
        _provider = autodetect()
    return _provider


def reset() -> None:
    """Forget the current choice; the next use autodetects again."""
    global _provider
    _provider = None


def available() -> bool:
    return get().available


def describe() -> str:
    return get().describe()


def complete(prompt: str, system: str | None = None, max_tokens: int = 600, temperature: float = 0.2,
             model: str | None = None) -> str:
    """One completion from the configured model. Raises LLMError when there is none."""
    return get().complete(prompt, system=system, max_tokens=max_tokens, temperature=temperature, model=model)


def list_models() -> list:
    """The models the configured provider offers (empty when none is configured)."""
    return get().list_models()


def build(provider: str, model: str | None = None, base_url: str | None = None, api_key: str | None = None) -> Provider:
    """A provider object built from explicit settings, without changing the kit's current choice.

    Front ends such as Notebook Tutor use this to hold their own connection.
    """
    global _provider
    saved = _provider
    try:
        return configure(provider, model=model, base_url=base_url, api_key=api_key)
    finally:
        _provider = saved


# --------------------------------------------------------------------------- student-facing helpers
NOT_AVAILABLE = ("No language model is configured, so this help is not available. The kit works fully "
                 "without one: kit.task('{tid}') explains the task, and the feedback from kit.answer() says "
                 "what to look at. To enable hints, see ai_learning_kit.llm (for example "
                 "llm.configure('ollama') with a local model, or set an API key).")

TUTOR_SYSTEM = (
    "You are a patient teaching assistant for a university data-science course. You help a student who is "
    "learning Python and data science. Rules: never give the final answer or the exact value or code that "
    "completes the task; give one step the student can take next, or explain the concept asked about, in "
    "plain language, in at most 120 words; refer to the task guidance you are given; if the student's "
    "feedback shows a specific slip, point at it without correcting it for them; do not invent facts about "
    "the data beyond what the guidance states."
)


def _task_context(mod, task_id: str) -> str:
    from ..core.session import scope_text
    t = mod.TASKS[task_id]
    g = t.get("guide") or {}
    scope = scope_text(mod)
    return (f"Unit {mod.UNIT}, session {mod.SESSION}: {mod.TITLE}\nTask {task_id}: {t['title']}\n"
            f"Question: {t['question']}\nWhat to do: {g.get('what', '').strip()}\n"
            f"Why it matters: {g.get('why', '').strip()}\nWhat to record: {g.get('record', '').strip()}"
            + (f"\n{scope}" if scope else ""))


def hint(mod, book, task_id: str) -> None:
    if task_id not in mod.TASKS:
        raise KeyError(f"{task_id} is not a task in {mod.UNIT} {mod.SESSION}.")
    if not available():
        print(NOT_AVAILABLE.format(tid=task_id))
        return
    entry = book.data["answers"].get(task_id) or {}
    last = ""
    if entry:
        last = (f"\nThe student's last saved value: {str(entry.get('value'))[:400]}\n"
                f"Their reasoning: {str(entry.get('reasoning'))[:600]}\n"
                f"The kit's verdict: {entry.get('result', 'not checked')}; attempts: {entry.get('attempts', 1)}.")
    prompt = (f"{_task_context(mod, task_id)}{last}\n\nGive the student one hint for their next step. "
              "Do not state the answer.")
    try:
        text = complete(prompt, system=TUTOR_SYSTEM, max_tokens=300)
    except LLMError as e:
        print(f"The language model could not be reached ({e}). Use kit.task('{task_id}') instead.")
        return
    print(f"Hint ({describe()}):\n{text}")


def ask(mod, problem, task_id: str, question: str) -> None:
    if task_id not in mod.TASKS:
        raise KeyError(f"{task_id} is not a task in {mod.UNIT} {mod.SESSION}.")
    if not available():
        print(NOT_AVAILABLE.format(tid=task_id))
        return
    prompt = (f"{_task_context(mod, task_id)}\nProblem: {problem.title}. {problem.summary}\n\n"
              f"The student asks: {question}\n\nAnswer the question. Do not give the task's answer.")
    try:
        text = complete(prompt, system=TUTOR_SYSTEM, max_tokens=400)
    except LLMError as e:
        print(f"The language model could not be reached ({e}).")
        return
    print(f"Answer ({describe()}):\n{text}")
