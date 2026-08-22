"""Agent tools module."""

from nanobot.agent.tools.base import Schema, Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.schema import (
    ArraySchema,
    BooleanSchema,
    IntegerSchema,
    NumberSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)

__all__ = [
    "Schema",
    "ArraySchema",
    "BooleanSchema",
    "IntegerSchema",
    "NumberSchema",
    "ObjectSchema",
    "StringSchema",
    "Tool",
    "ToolContext",
    "ToolLoader",
    "ToolResult",
    "ToolRegistry",
    "tool_parameters",
    "tool_parameters_schema",
]

# Hardening overlay (nanobot-brain image): apply workspace-guard and write-cap
# wrappers. install() logs loudly on stderr if a nanobot-ai bump changed the
# wrapped shapes, instead of silently shipping without the hardening.
try:
    from nanobot.agent.tools.guard_hardening import install as _install_guard_hardening

    _install_guard_hardening()
except Exception as _hardening_error:  # pragma: no cover - startup diagnostics
    import sys

    print(f"[nanobot-hardening] install failed: {_hardening_error}", file=sys.stderr)
