"""Language-model providers, all optional and all reached with the standard library only.

- OpenAICompatible : any server that speaks the OpenAI chat-completions API: OpenAI
                     itself, Ollama (http://localhost:11434/v1), vLLM, LM Studio,
                     Groq, Together, a university-hosted open-weight model, and so on.
- Anthropic        : the Anthropic Messages API.
- Null             : no model configured; every call explains how to configure one.

Adding a provider: subclass Provider, implement complete(), and add it to
PROVIDERS in __init__.py. Nothing else in the kit needs to change.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


class Provider:
    name = "base"

    def __init__(self, model: str | None = None, **_):
        self.model = model

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 600,
                 temperature: float = 0.2) -> str:
        raise NotImplementedError

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

    def complete(self, prompt, system=None, max_tokens=600, temperature=0.2):
        raise LLMError(self.reason)


def _post_json(url: str, payload: dict, headers: dict, timeout: float = 90) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    opener = urllib.request.build_opener()       # honours HTTP(S)_PROXY like any client
    try:
        with opener.open(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        raise LLMError(f"{url} answered {e.code}: {body}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"Could not reach {url}: {e}") from None


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

    def describe(self) -> str:
        return f"{self.name} ({self.model} at {self.base_url})"

    def complete(self, prompt, system=None, max_tokens=600, temperature=0.2):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        out = _post_json(f"{self.base_url}/chat/completions",
                         {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                          "temperature": temperature}, headers)
        try:
            return out["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Unexpected reply from {self.base_url}: {str(out)[:300]}") from None


class Anthropic(Provider):
    name = "anthropic"

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.anthropic.com", **_):
        super().__init__(model)
        self.api_key, self.base_url = api_key, base_url.rstrip("/")

    def complete(self, prompt, system=None, max_tokens=600, temperature=0.2):
        payload = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                   "messages": [{"role": "user", "content": prompt}]}
        if system:
            payload["system"] = system
        out = _post_json(f"{self.base_url}/v1/messages", payload,
                         {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        try:
            return "".join(part.get("text", "") for part in out["content"]).strip()
        except (KeyError, TypeError):
            raise LLMError(f"Unexpected reply from Anthropic: {str(out)[:300]}") from None


def ollama_models(host: str, timeout: float = 1.5) -> list:
    """Names of the models an Ollama server has installed, or [] when it is not reachable."""
    try:
        with urllib.request.build_opener().open(f"{host.rstrip('/')}/api/tags", timeout=timeout) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []
