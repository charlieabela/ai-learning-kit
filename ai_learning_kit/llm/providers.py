"""Language-model providers, all optional and all reached with the standard library only.

- Gemini           : Google's Gemini API (an API key from Google AI Studio).
- OpenAICompatible : any server that speaks the OpenAI chat-completions API: OpenAI
                     itself, Ollama (http://localhost:11434/v1), vLLM, LM Studio,
                     Groq, Together, a university-hosted open-weight model, and so on.
- Anthropic        : the Anthropic Messages API.
- Null             : no model configured; every call explains how to configure one.

Every provider can list the models its key gives access to (list_models), so a
front end can offer a choice instead of a hard-coded model name. Errors carry a
short code (not_configured, bad_key, forbidden, model_not_found, rate_limited,
offline, error) so a front end can say what to do.

Adding a provider: subclass Provider, implement complete() and list_models(), and
add it to PROVIDERS in __init__.py. Nothing else in the kit needs to change.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request


class LLMError(RuntimeError):
    def __init__(self, message: str, code: str = "error"):
        super().__init__(message)
        self.code = code


def _code_for_status(status: int, body: str = "") -> str:
    if status == 401:
        return "bad_key"
    if status == 403:
        return "forbidden"
    if status == 404:
        return "model_not_found"
    if status == 429:
        return "rate_limited"
    if status == 400 and ("API key" in body or "API_KEY" in body):
        return "bad_key"
    return "error"


def _request(url: str, payload: dict | None, headers: dict, timeout: float = 120) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json", **headers})
    opener = urllib.request.build_opener()       # honours HTTP(S)_PROXY like any client
    try:
        with opener.open(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:600]
        detail = body
        try:
            j = json.loads(body)
            detail = (j.get("error") or {}).get("message") or j.get("message") or body
        except Exception:
            pass
        raise LLMError(f"{e.code}: {detail}", _code_for_status(e.code, body)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"Could not reach {urllib.parse.urlsplit(url).netloc}: {e}", "offline") from None


_post_json = _request  # kept for older imports


# Current models "think" before answering (Claude 5.5, Gemini 2.5 and later, OpenAI's reasoning models), and the
# thinking counts against the output limit. A small limit then leaves no room for the answer: the reply is cut off,
# or is empty. Every request asks for at least this many output tokens; only the tokens used are billed.
MIN_OUTPUT_TOKENS = 4096
TRUNCATED = {"max_tokens", "length", "MAX_TOKENS"}


def _send_adaptive(provider, url: str, payload: dict, headers: dict) -> dict:
    """POST, dropping or renaming a parameter the model rejects, and remembering that for later calls.

    Newer models refuse some settings: Claude 5.5 models answer 400 "'temperature' is deprecated for this
    model", and OpenAI's reasoning models refuse temperature and want max_completion_tokens instead of
    max_tokens. The kit's settings are only preferences, so it retries without them.
    """
    memo = provider.__dict__.setdefault("_adapted", {})
    model = payload.get("model")
    for key in memo.get(model, ()):
        if key == "temperature":
            payload.pop("temperature", None)
        elif key == "output_config":
            payload.pop("output_config", None)
        elif key == "max_tokens" and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
    for _ in range(3):
        try:
            return _request(url, payload, headers)
        except LLMError as e:
            msg = str(e)
            if not msg.startswith("400"):
                raise
            if "temperature" in msg and "temperature" in payload:
                payload.pop("temperature")
                memo.setdefault(model, set()).add("temperature")
            elif ("output_config" in msg or "effort" in msg) and "output_config" in payload:
                payload.pop("output_config")                       # older models have no effort setting
                memo.setdefault(model, set()).add("output_config")
            elif "max_completion_tokens" in msg and "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
                memo.setdefault(model, set()).add("max_tokens")
            else:
                raise
    return _request(url, payload, headers)


class Provider:
    name = "base"

    def __init__(self, model: str | None = None, **_):
        self.model = model
        self.last_stop = None      # why the last reply ended, as the provider reported it

    @property
    def truncated(self) -> bool:
        """True when the last reply stopped because it reached the output limit."""
        return self.last_stop in TRUNCATED

    def _finish(self, text: str, stop) -> str:
        self.last_stop = stop
        if not text.strip() and stop in TRUNCATED:
            raise LLMError("The model used its whole output allowance (probably on thinking) and wrote no "
                           "answer. Ask again, or choose a faster model.", "truncated")
        return text.strip()

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 600,
                 temperature: float = 0.2, model: str | None = None) -> str:
        raise NotImplementedError

    def list_models(self) -> list:
        return [self.model] if self.model else []

    def describe(self) -> str:
        return f"{self.name} ({self.model})"

    @property
    def available(self) -> bool:
        return True


class Null(Provider):
    name = "none"

    def __init__(self, reason: str = "no language model is configured", **_):
        super().__init__(None)
        self.reason = reason

    @property
    def available(self) -> bool:
        return False

    def describe(self) -> str:
        return f"none: {self.reason}"

    def complete(self, prompt, system=None, max_tokens=600, temperature=0.2, model=None):
        raise LLMError(self.reason, "not_configured")

    def list_models(self):
        return []


class OpenAICompatible(Provider):
    """Chat completions at <base_url>/chat/completions (OpenAI, Ollama, vLLM, LM Studio, ...)."""
    name = "openai-compatible"

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: str | None = None,
                 label: str | None = None, **_):
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        if label:
            self.name = label

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def describe(self) -> str:
        return f"{self.name} ({self.model} at {self.base_url})"

    def complete(self, prompt, system=None, max_tokens=600, temperature=0.2, model=None):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        out = _send_adaptive(self, f"{self.base_url}/chat/completions",
                             {"model": model or self.model, "messages": messages,
                              "max_tokens": max(max_tokens, MIN_OUTPUT_TOKENS), "temperature": temperature},
                             self._headers())
        try:
            choice = out["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMError(f"Unexpected reply from {self.base_url}: {str(out)[:300]}") from None
        return self._finish(text, choice.get("finish_reason"))

    def list_models(self):
        out = _request(f"{self.base_url}/models", None, self._headers(), timeout=20)
        return sorted(m.get("id") for m in out.get("data", []) if m.get("id"))


class Anthropic(Provider):
    name = "anthropic"

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.anthropic.com", **_):
        super().__init__(model)
        self.api_key, self.base_url = api_key, base_url.rstrip("/")

    def _headers(self):
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

    # Claude 5.5 models think by default at high effort. For tutoring, low effort answers faster and costs less;
    # models without an effort setting reject it, and the request is then retried without it.
    effort = "low"

    def complete(self, prompt, system=None, max_tokens=600, temperature=0.2, model=None):
        payload = {"model": model or self.model, "max_tokens": max(max_tokens, MIN_OUTPUT_TOKENS),
                   "temperature": temperature, "messages": [{"role": "user", "content": prompt}]}
        if self.effort:
            payload["output_config"] = {"effort": self.effort}
        if system:
            payload["system"] = system
        out = _send_adaptive(self, f"{self.base_url}/v1/messages", payload, self._headers())
        try:
            text = "".join(part.get("text", "") for part in out["content"] if part.get("type", "text") == "text")
        except (KeyError, TypeError, AttributeError):
            raise LLMError(f"Unexpected reply from Anthropic: {str(out)[:300]}") from None
        return self._finish(text, out.get("stop_reason"))

    def list_models(self):
        out = _request(f"{self.base_url}/v1/models?limit=100", None, self._headers(), timeout=20)
        return [m["id"] for m in out.get("data", []) if m.get("id")]


class Gemini(Provider):
    """Google's Gemini API (generateContent), with a key from Google AI Studio."""
    name = "gemini"

    def __init__(self, model: str, api_key: str,
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta", **_):
        super().__init__(model)
        self.api_key, self.base_url = api_key, base_url.rstrip("/")

    def _headers(self):
        return {"x-goog-api-key": self.api_key}

    def complete(self, prompt, system=None, max_tokens=600, temperature=0.2, model=None):
        name = model or self.model
        name = name if name.startswith("models/") else f"models/{name}"
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                   "generationConfig": {"maxOutputTokens": max(max_tokens, MIN_OUTPUT_TOKENS),
                                        "temperature": temperature}}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        out = _request(f"{self.base_url}/{name}:generateContent", payload, self._headers())
        try:
            cand = out["candidates"][0]
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
        except (KeyError, IndexError, TypeError):
            block = (out.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(f"Gemini returned no text{f' (blocked: {block})' if block else ''}.",
                           "refused" if block else "error") from None
        if not text and cand.get("finishReason") not in (None, "STOP", "MAX_TOKENS"):
            raise LLMError(f"Gemini stopped without text ({cand.get('finishReason')}).", "refused")
        return self._finish(text, cand.get("finishReason"))

    def list_models(self):
        out = _request(f"{self.base_url}/models?pageSize=200", None, self._headers(), timeout=20)
        names = []
        for m in out.get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                names.append(m["name"].split("/", 1)[-1])
        return sorted(names)


def ollama_models(host: str, timeout: float = 1.5) -> list:
    """Names of the models an Ollama server has installed, or [] when it is not reachable."""
    try:
        with urllib.request.build_opener().open(f"{host.rstrip('/')}/api/tags", timeout=timeout) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []
