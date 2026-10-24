"""The optional language-model layer: never required, pluggable, exercised against a local fake server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import ai_learning_kit as kit
from ai_learning_kit import llm
from ai_learning_kit.llm.providers import LLMError


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for var in ("AILK_LLM_PROVIDER", "AILK_LLM_MODEL", "AILK_LLM_BASE_URL", "AILK_LLM_API_KEY",
                "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "ollama_models", lambda host, timeout=1.5: [])
    from ai_learning_kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    llm.reset()
    yield
    llm.reset()


class _FakeOpenAI(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOpenAI.seen.append((self.path, self.headers.get("Authorization"), body))
        reply = {"choices": [{"message": {"role": "assistant",
                                          "content": f"fake reply to: {body['messages'][-1]['content'][:40]}"}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_without_configuration_nothing_is_required(capsys):
    assert not llm.available()
    assert llm.describe().startswith("none")
    with pytest.raises(LLMError):
        llm.complete("hello")
    kit.start("ARI1101", "S02", student="No LLM")
    kit.hint("T3.3")
    kit.ask("T3.3", "what does euro equivalent mean?")
    out = capsys.readouterr().out
    assert out.count("No language model is configured") == 2 and "kit.task('T3.3')" in out


def test_openai_compatible_provider_talks_to_any_such_server(fake_server, monkeypatch):
    monkeypatch.setattr(kit.llm, "ollama_models", lambda host, timeout=1.5: [])
    p = llm.configure("openai-compatible", model="my-open-weight-model", base_url=fake_server, api_key="k")
    assert llm.available() and "my-open-weight-model" in llm.describe()
    assert llm.complete("ping", system="sys") == "fake reply to: ping"
    path, auth, body = _FakeOpenAI.seen[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer k"
    assert body["messages"][0] == {"role": "system", "content": "sys"} and body["model"] == "my-open-weight-model"


def test_environment_variables_select_the_provider(fake_server, monkeypatch):
    monkeypatch.setenv("AILK_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("AILK_LLM_BASE_URL", fake_server)
    monkeypatch.setenv("AILK_LLM_MODEL", "env-model")
    assert llm.available() and "env-model" in llm.describe()


def test_ollama_autodetect_uses_the_first_installed_model(monkeypatch):
    monkeypatch.setattr(llm, "ollama_models", lambda host, timeout=1.5: ["llama3.1:8b", "qwen2.5"])
    p = llm.get()
    assert p.name == "ollama" and p.model == "llama3.1:8b" and p.base_url.endswith("/v1")


def test_hint_uses_the_task_guide_and_the_students_last_answer(fake_server, capsys):
    llm.configure("openai-compatible", model="m", base_url=fake_server)
    kit.start("ARI1101", "S02", student="Hint Test")
    kit.answer("T3.3", value=40)
    capsys.readouterr()
    kit.hint("T3.3")
    out = capsys.readouterr().out
    assert out.startswith("Hint (")
    _, _, body = _FakeOpenAI.seen[-1]
    prompt = body["messages"][-1]["content"]
    assert "Task T3.3" in prompt and "last saved value: 40" in prompt and "not yet" in prompt
    assert "never give the final answer" in body["messages"][0]["content"]


def test_unreachable_server_degrades_gracefully(capsys):
    llm.configure("openai-compatible", model="m", base_url="http://127.0.0.1:9/v1")
    kit.start("ARI1101", "S02", student="Down")
    kit.hint("T1.1")
    assert "could not be reached" in capsys.readouterr().out


def test_configure_none_switches_help_off():
    llm.configure("none")
    assert not llm.available() and "switched off" in llm.describe()


# --------------------------------------------------------------------------- Gemini, model lists, error codes
class _FakeGemini(BaseHTTPRequestHandler):
    seen = []

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        _FakeGemini.seen.append(("GET", self.path, self.headers.get("x-goog-api-key"), None))
        self._send(200, {"models": [
            {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent", "countTokens"]},
            {"name": "models/text-embedding-9", "supportedGenerationMethods": ["embedContent"]}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeGemini.seen.append(("POST", self.path, self.headers.get("x-goog-api-key"), body))
        if self.headers.get("x-goog-api-key") != "good":
            self._send(400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key."}})
            return
        if "missing-model" in self.path:
            self._send(404, {"error": {"code": 404, "message": "models/missing-model is not found"}})
            return
        self._send(200, {"candidates": [{"content": {"parts": [{"text": "gemini says hi"}]}, "finishReason": "STOP"}]})

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_gemini():
    srv = HTTPServer(("127.0.0.1", 0), _FakeGemini)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1beta"
    srv.shutdown()


def test_gemini_provider_completes_lists_models_and_reports_errors(fake_gemini):
    p = llm.configure("gemini", model="gemini-3.8-flash", api_key="good", base_url=fake_gemini)
    assert llm.complete("hello", system="be brief") == "gemini says hi"
    method, path, key, body = _FakeGemini.seen[-1]
    assert path == "/v1beta/models/gemini-3.8-flash:generateContent" and key == "good"
    assert body["systemInstruction"]["parts"][0]["text"] == "be brief"
    assert llm.list_models() == ["gemini-3.8-flash"]                 # only models that can generate text
    with pytest.raises(LLMError) as e:
        llm.complete("x", model="missing-model")
    assert e.value.code == "model_not_found"
    bad = llm.build("gemini", model="gemini-3.8-flash", api_key="wrong", base_url=fake_gemini)
    with pytest.raises(LLMError) as e:
        bad.complete("x")
    assert e.value.code == "bad_key"
    assert llm.get() is p                                             # build() leaves the kit's choice alone


def test_gemini_key_from_environment_is_autodetected(monkeypatch, fake_gemini):
    monkeypatch.setenv("GEMINI_API_KEY", "good")
    p = llm.get()
    assert p.name == "gemini" and p.model == llm.DEFAULT_MODELS["gemini"]


def test_unreachable_server_has_the_offline_code():
    p = llm.build("openai-compatible", model="m", base_url="http://127.0.0.1:9/v1")
    with pytest.raises(LLMError) as e:
        p.complete("x")
    assert e.value.code == "offline"


def test_openai_compatible_needs_a_model_name():
    with pytest.raises(ValueError, match="model"):
        llm.configure("openai-compatible", base_url="http://127.0.0.1:9/v1")


# --------------------------------------------------------------------------- models that refuse some settings
@pytest.fixture
def picky_server():
    """An Anthropic- and OpenAI-style server whose models refuse temperature, and max_tokens (OpenAI)."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, obj):
            body = _json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(dict(payload))
            if "temperature" in payload:
                return self._send(400, {"type": "error", "error": {"type": "invalid_request_error",
                                        "message": "`temperature` is deprecated for this model."}})
            if self.path.endswith("/chat/completions"):
                if "max_tokens" in payload:
                    return self._send(400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not "
                                            "supported with this model. Use 'max_completion_tokens' instead."}})
                return self._send(200, {"choices": [{"message": {"content": "openai ok"}}]})
            return self._send(200, {"content": [{"type": "text", "text": "claude ok"}]})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", seen
    httpd.shutdown()


def test_claude_models_that_refuse_temperature_still_answer(picky_server):
    url, seen = picky_server
    p = llm.build("anthropic", model="claude-sonnet-5-5", api_key="k", base_url=url)
    assert p.complete("hi") == "claude ok"
    assert "temperature" in seen[0] and "temperature" not in seen[1]
    n = len(seen)
    assert p.complete("again") == "claude ok"
    assert len(seen) == n + 1                      # remembered: no second refusal


def test_openai_reasoning_models_get_the_parameters_they_accept(picky_server):
    url, seen = picky_server
    p = llm.build("openai-compatible", model="o9", api_key="k", base_url=url)
    assert p.complete("hi") == "openai ok"
    assert "max_completion_tokens" in seen[-1] and "max_tokens" not in seen[-1] and "temperature" not in seen[-1]


# --------------------------------------------------------------------------- thinking models
@pytest.fixture
def thinking_server():
    """Claude-style server: new models think (using output tokens); old models reject output_config."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, obj):
            body = _json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            p = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(dict(p))
            if p["model"] == "claude-old" and "output_config" in p:
                return self._send(400, {"error": {"message": "output_config: Extra inputs are not permitted"}})
            if p["model"] == "claude-greedy":            # thinks so much that nothing is left for the answer
                return self._send(200, {"content": [{"type": "thinking", "thinking": "..."}],
                                        "stop_reason": "max_tokens"})
            thinking = 3000 if (p.get("output_config") or {}).get("effort") != "low" else 200
            if p["max_tokens"] < thinking + 100:
                return self._send(200, {"content": [{"type": "thinking", "thinking": "..."},
                                                    {"type": "text", "text": "cut"}], "stop_reason": "max_tokens"})
            return self._send(200, {"content": [{"type": "thinking", "thinking": "..."},
                                                {"type": "text", "text": "full answer"}], "stop_reason": "end_turn"})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", seen
    httpd.shutdown()


def test_thinking_models_get_room_for_the_answer(thinking_server):
    url, seen = thinking_server
    p = llm.build("anthropic", model="claude-sonnet-5-5", api_key="k", base_url=url)
    assert p.complete("hi", max_tokens=300) == "full answer"           # a small request still gets room
    assert seen[-1]["max_tokens"] >= 4096 and seen[-1]["output_config"] == {"effort": "low"}
    assert not p.truncated


def test_models_without_effort_retry_without_it(thinking_server):
    url, seen = thinking_server
    p = llm.build("anthropic", model="claude-old", api_key="k", base_url=url)
    assert p.complete("hi") == "full answer"
    assert "output_config" not in seen[-1]
    n = len(seen)
    p.complete("again")
    assert len(seen) == n + 1                                          # remembered


def test_an_answer_lost_to_thinking_is_reported(thinking_server):
    url, _ = thinking_server
    p = llm.build("anthropic", model="claude-greedy", api_key="k", base_url=url)
    with pytest.raises(llm.LLMError) as e:
        p.complete("hi")
    assert e.value.code == "truncated" and p.truncated
