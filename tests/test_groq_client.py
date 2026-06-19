"""Tests for the Groq (GPT-OSS 120B) client.

No key, no network: a fake `groq` module is injected so we can assert the
request shape (system + user messages, model, reasoning params) and the
response parsing, and that passing an image to a text-only model fails loudly.
"""

import sys
import types

import pytest


def _install_fake_sdk(monkeypatch, captured):
    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content='{"summary": "ok", "objects": []}')
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(id="cmpl_test", choices=[choice])

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeGroq:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = FakeChat()

    fake = types.ModuleType("groq")
    fake.Groq = FakeGroq
    monkeypatch.setitem(sys.modules, "groq", fake)


def test_groq_builds_chat_request(monkeypatch):
    captured = {}
    _install_fake_sdk(monkeypatch, captured)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

    from cortex.llm.groq_client import GroqClient

    client = GroqClient()
    resp = client.complete(system="ROLE: PLANNER", prompt="plan it")

    assert captured["model"] == "openai/gpt-oss-120b"
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user"]
    assert captured["messages"][0]["content"] == "ROLE: PLANNER"
    assert captured["reasoning_effort"] == "low"
    assert "max_completion_tokens" in captured
    assert resp.text.startswith("{")


def test_groq_rejects_image(monkeypatch):
    _install_fake_sdk(monkeypatch, {})
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

    from cortex.llm.groq_client import GroqClient
    from cortex.llm.base import ImageInput

    client = GroqClient()
    with pytest.raises(RuntimeError, match="text-only"):
        client.complete(system="ROLE: PERCEPTION", prompt="x", image=ImageInput("image/png", "QUJD"))


def test_groq_requires_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from cortex.llm.groq_client import GroqClient

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqClient(api_key=None)
