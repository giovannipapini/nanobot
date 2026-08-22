"""Agent hook that automates the Engram memory protocol on every agent run.

Opt-in: the hook is only active when ``ENGRAM_MCP_URL`` is set (presence of
the URL is the flag; no config entry needed). When active it:

- ``before_run``: pulls the latest stored context for the session from Engram
  (``GET /session/context``) and, if the payload is small, logs it as a compact
  context block. The runner hands ``before_run`` a deepcopy of the messages and
  the model request is snapshotted before ``before_iteration``, so there is no
  reliable hook seam to push a message into this turn's model input — the
  context is therefore exposed via logs only (and kept available on the hook
  for future injection seams).
- ``after_run``: writes a structured session summary to Engram
  (``POST /session/handoff``): session key, final content (truncated),
  tools used, stop reason, error, usage, timestamp.

All network calls are fire-and-forget with a hard timeout: Engram being down
or slow must never fail the agent run (silent skip + log).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.hook import (
    AgentHook,
    AgentRunHookContext,
    AgentTurnHookContext,
)

ENGRAM_MCP_URL_ENV = "ENGRAM_MCP_URL"
ENGRAM_MCP_TOKEN_ENV = "MCP_ENGRAM_SERVER_TOKEN"

#: Payloads larger than this are skipped for before_run (log-only) injection.
MAX_CONTEXT_CHARS = 1500
#: final_content is truncated to this length in the written handoff.
MAX_SUMMARY_CHARS = 2000
#: Per-request budget; a hard asyncio cap (+0.5s) applies on top.
REQUEST_TIMEOUT_S = 3.0

_CONTEXT_MARKER = "engram-memory-context"


def _summarize_run(context: AgentRunHookContext) -> str:
    """One-line human-readable summary of the run for the handoff."""
    tools = ", ".join(context.tools_used) or "none"
    error = context.error or "none"
    usage = context.usage or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    return (
        f"nanobot run: stop={context.stop_reason or 'completed'}; "
        f"tools=[{tools}]; error={error}; "
        f"usage={{prompt_tokens:{prompt_tokens}, completion_tokens:{completion_tokens}}}"
    )


def _format_context_payload(payload: dict[str, Any]) -> str | None:
    """Format a stored handoff row into a compact context block."""
    summary = payload.get("summary")
    if not summary:
        return None
    lines = [
        f"— {_CONTEXT_MARKER}: session {payload.get('session_id', '')} —",
        str(summary),
    ]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        tools = metadata.get("tools_used") or []
        if tools:
            lines.append("Tools used: " + ", ".join(str(tool) for tool in tools))
        stop_reason = metadata.get("stop_reason")
        if stop_reason:
            lines.append(f"Stop reason: {stop_reason}")
        error = metadata.get("error")
        if error:
            lines.append(f"Error: {error}")
    return "\n".join(lines)


class EngramMemoryHook(AgentHook):
    """Pull context before a run and persist a handoff after it.

    ``base_url`` is the Engram HTTP bridge root (e.g. ``http://engram:8421``)
    and ``token`` an optional bearer token. ``transport`` is injectable for
    tests (``httpx.MockTransport``). Nothing here raises: every network op is
    wrapped so a broken Engram degrades to a log line.
    """

    def __init__(
        self,
        session_key: str | None,
        base_url: str,
        token: str | None = None,
        *,
        timeout_s: float = REQUEST_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__()
        self._session_key = session_key or "nanobot"
        self._base_url = base_url.rstrip("/")
        self._token = token or None
        self._timeout_s = timeout_s
        self._transport = transport
        #: Compact context block fetched in before_run (for logs and any
        #: future injection seam). None when Engram was unreachable, the
        #: payload was missing, or it exceeded MAX_CONTEXT_CHARS.
        self.context_payload: str | None = None

    def _client(self) -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=self._timeout_s,
            transport=self._transport,
        )

    async def before_run(self, context: AgentRunHookContext) -> None:
        """Fetch the latest Engram context for this session and expose it."""
        try:
            payload = await self._fetch_context()
        except Exception:
            logger.warning(
                "Engram memory: context fetch failed for session '{}'",
                self._session_key,
                exc_info=True,
            )
            return
        text = _format_context_payload(payload) if payload else None
        if text is None:
            logger.info(
                "Engram memory: no stored context for session '{}'",
                self._session_key,
            )
            return
        if len(text) > MAX_CONTEXT_CHARS:
            logger.info(
                "Engram memory: context for session '{}' too large ({} chars) — skipping",
                self._session_key,
                len(text),
            )
            return
        self.context_payload = text
        logger.info("Engram memory: context for session '{}':\n{}", self._session_key, text)

    async def after_run(self, context: AgentRunHookContext) -> None:
        """Write a structured session summary to Engram (fire-and-forget)."""
        try:
            await self._write_handoff(context)
        except Exception:
            logger.warning(
                "Engram memory: handoff write failed for session '{}'",
                self._session_key,
                exc_info=True,
            )

    async def _fetch_context(self) -> dict[str, Any] | None:
        async with self._client() as client:
            response = await asyncio.wait_for(
                client.get(
                    "/session/context",
                    params={"session_key": self._session_key},
                ),
                timeout=self._timeout_s + 0.5,
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or not payload.get("summary"):
            return None
        return payload

    async def _write_handoff(self, context: AgentRunHookContext) -> None:
        final_content = context.final_content or ""
        payload = {
            "session_key": self._session_key,
            "summary": _summarize_run(context),
            "final_content": final_content[:MAX_SUMMARY_CHARS],
            "tools_used": list(context.tools_used),
            "stop_reason": context.stop_reason,
            "error": context.error,
            "usage": dict(context.usage),
            "timestamp": time.time(),
        }
        async with self._client() as client:
            response = await asyncio.wait_for(
                client.post("/session/handoff", json=payload),
                timeout=self._timeout_s + 0.5,
            )
            response.raise_for_status()


def create_engram_memory_hook(context: AgentTurnHookContext) -> AgentHook | None:
    """Create the Engram memory hook for one agent turn.

    Opt-in via ``ENGRAM_MCP_URL``; returns ``None`` when unset (no behavior
    change for existing deployments). Ephemeral turns (dream, subagent tool
    calls) are skipped so only real user sessions touch the memory store.
    """
    base_url = os.environ.get(ENGRAM_MCP_URL_ENV, "").strip()
    if not base_url or context.ephemeral:
        return None
    return EngramMemoryHook(
        session_key=context.session_key,
        base_url=base_url,
        token=os.environ.get(ENGRAM_MCP_TOKEN_ENV, "").strip() or None,
    )
