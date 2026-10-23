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
