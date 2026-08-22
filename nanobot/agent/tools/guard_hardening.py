"""Workspace-guard hardening overlay for the nanobot-brain image.

Overlay step (see images/nanobot/Dockerfile): copied into the wheel's
``nanobot/agent/tools/`` and activated from the package ``__init__``. Wraps,
without replacing internals, three classes so the patch survives upstream
refactors as long as the wrapped methods keep their shape:

- ``ExecTool._guard_command``: the stock guard statically extracts *literal*
  absolute paths, so environment-variable path construction (``$HOME/...``,
  ``${XDG_...}``) and env dumps (``env``, ``printenv``, bare ``export``,
  ``os.environ`` in python -c) slip through it. Under a restricted workspace
  these are blocked here; ``allow_patterns`` still wins (operator exemption).
- ``WriteFileTool.execute`` / ``ApplyPatchTool.execute``: cap per-call write
  size so a prompt-injected agent cannot fill the disk (read paths already cap).

Re-apply/verify on every nanobot-ai bump; install() logs loudly to stderr if a
wrapped shape changed instead of silently shipping without the hardening.
"""

from __future__ import annotations

import re

_WORKSPACE_BOUNDARY_NOTE = (
    "\n\nNote: this is a hard policy boundary, not a transient failure. "
    "Do NOT retry with shell tricks (symlinks, base64 piping, alternative "
    "tools, working_dir overrides). If the user genuinely needs this "
    "resource, tell them you cannot reach it under the current "
    "restrict_to_workspace policy and ask how to proceed."
)

# Letter/underscore `$VAR` / `${VAR}` references to variables whose runtime
# values the static path scan cannot see AND which can point at sensitive
# data: HOME/XDG_*/SSH_*/AWS_* paths and credential vars (*_TOKEN, *_KEY,
# *_SECRET, *_PASSWORD, ...). Local shell variables ($x, $p, ...) keep working
# — upstream legitimately supports `x="<abs>"; cat "$x"` inside the workspace.
# Mass dumps (`env`, `printenv`, bare `export`, os.environ) are blocked below.
_DANGEROUS_VAR_RE = re.compile(
    r"\$(?:\{)?(?:"
    r"home"
    r"|xdg_[a-z0-9_]+"
    r"|(?:ssh|aws)_[a-z0-9_]+"
    r"|[a-z_][a-z0-9_]*(?:_(?:token|key|secret|password|passwd|pass|credential))"
    r"|(?:token|key|secret|password|passwd|pass|credentials?)"
    r")(?:\})?"
)
# Standalone env-dump segments (after splitting on chaining operators).
_ENV_DUMP_SEGMENT_RE = re.compile(r"^(?:env|printenv|export)$")
# Bare interactive shells: a session started as `bash`/`sh`/`exec zsh` is a
# persistent shell whose stdin commands are never re-guarded.
_INTERACTIVE_SHELL_RE = re.compile(
    r"^(?:exec\s+)?(?:bash|sh|zsh|dash|fish|ksh)(?:\s+-i\b)?\s*$"
)
# Python one-liners can dump the environment without any `$` or `env` word.
_ENV_API_RE = re.compile(r"\bos\.(?:environ|getenv)\b")

_WRITE_CAP_CHARS = 64 * 1024 * 1024  # 64 Mi characters, matching read-side caps

_installed = False


def _log(message: str) -> None:
    import sys

    print(f"[nanobot-hardening] {message}", file=sys.stderr, flush=True)


def _wrap_guard(original):
    """Return a wrapper adding restricted-mode env-var exfil checks."""

    def guarded(
        self,
        command: str,
        cwd: str,
        *,
        restrict_to_workspace: bool | None = None,
        workspace_root: str | None = None,
    ):
        error = original(
            self,
            command,
            cwd,
            restrict_to_workspace=restrict_to_workspace,
            workspace_root=workspace_root,
        )
        if error is not None:
            return error
        should_restrict = (
            self.restrict_to_workspace
            if restrict_to_workspace is None
            else restrict_to_workspace
        )
        if not should_restrict:
            return None
        if self.allow_patterns:  # operator allowlist = explicit exemption
            return None

        from nanobot.agent.tools.base import ToolResult

        lowered = command.strip().lower()
        if _DANGEROUS_VAR_RE.search(lowered):
            return ToolResult.error(
                "Error: Command blocked by safety guard "
                "(sensitive environment-variable reference under restricted workspace)"
                + _WORKSPACE_BOUNDARY_NOTE
            )
        try:
            segments = self._split_shell_segments(lowered)
        except Exception:  # malformed quoting etc.: fail closed
            segments = [lowered]
        if any(_ENV_DUMP_SEGMENT_RE.fullmatch(segment.strip()) for segment in segments):
            return ToolResult.error(
                "Error: Command blocked by safety guard "
                "(environment dump under restricted workspace)"
                + _WORKSPACE_BOUNDARY_NOTE
            )
        if any(
            _INTERACTIVE_SHELL_RE.fullmatch(segment.strip()) for segment in segments
        ):
            return ToolResult.error(
                "Error: Command blocked by safety guard "
                "(bare interactive shell under restricted workspace)"
                + _WORKSPACE_BOUNDARY_NOTE
            )
        if _ENV_API_RE.search(lowered):
            return ToolResult.error(
                "Error: Command blocked by safety guard "
                "(environment read via python under restricted workspace)"
                + _WORKSPACE_BOUNDARY_NOTE
            )
        return None

    return guarded


def _wrap_write_file(original):
    async def wrapped(self, path=None, content=None, **kwargs):
        from nanobot.agent.tools.base import ToolResult

        if content is not None and len(content) > _WRITE_CAP_CHARS:
            return ToolResult.error(
                f"Error: content exceeds {_WRITE_CAP_CHARS} characters "
                "(nanobot hardening write cap)"
            )
        return await original(self, path=path, content=content, **kwargs)

    return wrapped


def _wrap_apply_patch(original):
    async def wrapped(self, edits=None, dry_run=False, **kwargs):
        from nanobot.agent.tools.base import ToolResult

        total = 0
        if isinstance(edits, list):
            for edit in edits:
                if not isinstance(edit, dict):
                    continue
                new_text = edit.get("new_text")
                if isinstance(new_text, str):
                    total += len(new_text)
                    if total > _WRITE_CAP_CHARS:
                        return ToolResult.error(
                            f"Error: patch writes exceed {_WRITE_CAP_CHARS} "
                            "characters (nanobot hardening write cap)"
                        )
        return await original(self, edits=edits, dry_run=dry_run, **kwargs)

    return wrapped


def install() -> bool:
    """Apply the wrappers. Idempotent; never raises (logs failures loudly)."""
    global _installed
    if _installed:
        return True

    from nanobot.agent.tools.apply_patch import ApplyPatchTool
    from nanobot.agent.tools.filesystem import WriteFileTool
    from nanobot.agent.tools.shell import ExecTool

    def apply(owner, name, wrapper):
        try:
            original = getattr(owner, name)
            setattr(owner, name, wrapper(original))
        except Exception as exc:  # shape changed on bump: log, stay weak but alive
            _log(f"could not wrap {owner.__name__}.{name}: {exc!r}")

    apply(ExecTool, "_guard_command", _wrap_guard)
    apply(WriteFileTool, "execute", _wrap_write_file)
    apply(ApplyPatchTool, "execute", _wrap_apply_patch)
    _installed = True
    return True
