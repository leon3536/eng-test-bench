"""What the assistant's tools are given, and what they hand back.

The engine and tools.py are written separately, so the boundary between them
is spelled out here. A ToolContext holds read-only snapshot callables and a
DraftWriter, and nothing that can start, queue, approve or cancel a run or
read a key: whatever the model asks for, a tool can only read bench data and
save a draft. The engine builds one per turn through the server's factory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..designs import DraftWriter
    from ..settings import Config

# Drafts the assistant may save in one turn. The engine hands each turn a
# fresh ToolContext.saves_left = [SAVES_PER_TURN]; save_draft_design counts it down.
SAVES_PER_TURN = 5


class LedgerReader(Protocol):
    """Read-only view of the trial ledger (coverage.ReadOnlyLedger)."""

    def rows(self) -> list[dict[str, Any]]: ...

    def fingerprint(self) -> str: ...

    def coverage(self, provider: str, model: str, baseline_family: str | None, group_by: str) -> dict[str, Any]: ...

    def summary(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ToolContext:
    cfg: Config
    readiness: Callable[[], list[Any]]
    key_flags: Callable[[], dict[str, bool]]        # env key name -> set; never a value
    runs_snapshot: Callable[[], list[dict[str, Any]]]
    active_run: Callable[[], dict[str, Any] | None]
    queue_snapshot: Callable[[], dict[str, Any]]
    ledger: LedgerReader
    drafts: DraftWriter
    conversation_id: str
    spend: Callable[[], dict[str, Any]]             # {turn_usd, today_usd, per_turn_cap_usd, daily_cap_usd}
    saves_left: list[int]                           # one-element counter, shared with the engine


@dataclass
class ToolOutcome:
    ok: bool
    content: str                # the tool_result text the model reads (JSON)
    summary: str                # one line for the person, e.g. "Checked coverage · GLM 5.3 · 3 gaps"
    draft: dict[str, Any] | None = None   # set by save_draft_design: {design_id, version, spec_hash, ...}
