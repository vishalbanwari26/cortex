"""Tests for the real Anthropic client.

These run with no API key and no network: they inject a fake `anthropic` module
so we can assert that the client builds the request correctly (image block +
text block, base64 passed through, system prompt forwarded) and parses the
response. This is what de-risks the live path before you ever spend a token.
"""

import sys
import types

import pytest


def _install_fake_sdk(monkeypatch, captured):
    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)

            text_block = types.SimpleNamespace(type="text", text='{"summary": "ok", "objects": []}')
            return types.SimpleNamespace(id="msg_test", content=[text_block])

    class FakeAnthropic:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.messages = FakeMessages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake)


def test_client_builds_image_and_text_blocks(monkeypatch):
    captured = {}
    _install_fake_sdk(monkeypatch, captured)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    from cortex.llm.anthropic_client import AnthropicClient
    from cortex.llm.base import ImageInput

    client = AnthropicClient(model="claude-sonnet-4-6")
    resp = client.complete(
        system="ROLE: PERCEPTION",
        prompt="describe the scene",
        image=ImageInput(media_type="image/png", data_b64="QUJD"),
    )

    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["data"] == "QUJD"
    assert content[1] == {"type": "text", "text": "describe the scene"}
    assert captured["system"] == "ROLE: PERCEPTION"
    assert captured["model"] == "claude-sonnet-4-6"
    assert resp.text.startswith("{")


def test_client_text_only_omits_image_block(monkeypatch):
    captured = {}
    _install_fake_sdk(monkeypatch, captured)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    from cortex.llm.anthropic_client import AnthropicClient

    client = AnthropicClient()
    client.complete(system="ROLE: PLANNER", prompt="plan it")

    content = captured["messages"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"


def test_client_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from cortex.llm.anthropic_client import AnthropicClient

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(api_key=None)
