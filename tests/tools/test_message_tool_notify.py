"""Tests for the deployment-only `notify` escape hatch on MessageTool.

notify=true routes to the NANOBOT_NOTIFY_TARGET destination regardless of the
current conversation (e.g. an api-channel triage turn reaching the Telegram
DM). The agent cannot choose the destination — only flip the flag.
"""

import pytest

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import OutboundMessage


@pytest.mark.asyncio
async def test_message_tool_notify_delivers_to_configured_target(monkeypatch, tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    monkeypatch.setenv("NANOBOT_NOTIFY_TARGET", "telegram:112851500")
    tool = MessageTool(send_callback=_send)
    f = tmp_path / "report.md"
    f.write_text("alert", encoding="utf-8")

    with request_context(RequestContext(channel="api", chat_id="default", metadata={})):
        result = await tool.execute(content="disk almost full", notify=True)

    assert result == "Message sent to telegram:112851500"
    assert len(sent) == 1
    assert sent[0].channel == "telegram"
    assert sent[0].chat_id == "112851500"
    assert sent[0].metadata == {}


@pytest.mark.asyncio
async def test_message_tool_notify_without_configured_target_errors(monkeypatch) -> None:
    async def _send(msg: OutboundMessage) -> None:
        raise AssertionError("must not be called")

    monkeypatch.delenv("NANOBOT_NOTIFY_TARGET", raising=False)
    tool = MessageTool(send_callback=_send)
    result = await tool.execute(content="hi", notify=True)
    assert result.startswith("Error: notify requested but NANOBOT_NOTIFY_TARGET")


@pytest.mark.asyncio
async def test_message_tool_notify_with_malformed_target_errors(monkeypatch) -> None:
    async def _send(msg: OutboundMessage) -> None:
        raise AssertionError("must not be called")

    monkeypatch.setenv("NANOBOT_NOTIFY_TARGET", "telegram-only-no-separator")
    tool = MessageTool(send_callback=_send)
    result = await tool.execute(content="hi", notify=True)
    assert "channel:chat_id" in result


@pytest.mark.asyncio
async def test_message_tool_notify_does_not_relax_explicit_cross_chat(monkeypatch) -> None:
    """notify=true unlocks exactly one destination; explicit channel/chat_id stays blocked."""
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    monkeypatch.setenv("NANOBOT_NOTIFY_TARGET", "telegram:112851500")
    tool = MessageTool(send_callback=_send)
    with request_context(RequestContext(channel="api", chat_id="default", metadata={})):
        result = await tool.execute(
            content="hi",
            channel="telegram",
            chat_id="999",
            notify=True,
        )
    assert result.startswith("Error: notify=true cannot be combined")
    assert sent == []
