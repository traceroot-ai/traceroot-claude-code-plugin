"""
Transcript reader and turn assembler for Claude Code JSONL transcripts.

Provides:
  - read_all_records(path: str) -> list[dict]  — reads the WHOLE file, parses each
    line defensively (skip bad JSON, never raise). Immune to read-boundary splits.
  - assemble_turns(records)        — groups JSONL rows into Turn objects, retaining all
    assistant frames (per-requestId grouping happens in tokens.llm_calls, which uses
    the earliest frame as start_ts and the latest as end_ts to produce correct
    LLM span durations). Attaches tool_result rows by tool_use_id.
    Skips isMeta phantom rows (slash-command/skill injections).
"""

import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class Turn:
    user_msg: dict
    assistant_msgs: list = field(default_factory=list)
    tool_results: dict = field(default_factory=dict)  # tool_use_id -> {"content":..., "timestamp":..., "is_error": bool}


def read_all_records(path: str) -> list[dict]:
    """
    Read and parse the entire transcript file.

    Parses each newline-delimited JSON line defensively (skips bad JSON, never raises).
    Returns all valid records. Immune to read-boundary splits — always sees the full file.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return []
    out = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def _role(m: dict) -> str | None:
    """Return 'user' or 'assistant' from a transcript row, or None if unknown."""
    t = m.get("type")
    if t in ("user", "assistant"):
        return t
    return (m.get("message") or {}).get("role")


def _content(m: dict):
    """Return the content field from a transcript row (handles nested message dict)."""
    msg = m.get("message")
    return (msg or {}).get("content") if isinstance(msg, dict) else m.get("content")


def _is_tool_result(m: dict) -> bool:
    """Return True if this user row carries tool_result content blocks."""
    c = _content(m)
    return _role(m) == "user" and isinstance(c, list) and any(
        isinstance(x, dict) and x.get("type") == "tool_result" for x in c
    )


def assemble_turns(records: list[dict]) -> list[Turn]:
    """
    Group flat transcript rows into Turn objects.

    Rules:
    - isMeta rows are skipped entirely (phantom rows from slash-command/skill injection).
    - tool_result rows (user rows whose content is a list of tool_result blocks) are
      attached to the current turn's tool_results dict, keyed by tool_use_id.
    - Each non-tool-result user row starts a new Turn.
    - Every assistant row is appended to the current turn's assistant_msgs in order.
      Claude Code writes one row per content block (thinking/text/tool_use), all
      sharing the same message.id and requestId but with different timestamps. Keeping
      all rows is required so that tokens.llm_calls can set start_ts from the FIRST
      frame and end_ts from the LAST — collapsing to a single row would make
      start_ts == end_ts, producing zero-duration LLM spans.
    - Turns with no assistant messages are dropped (incomplete).
    """
    turns: list[Turn] = []
    cur: Turn | None = None

    for m in records:
        # Skip isMeta phantom rows — they are slash-command/skill injections, not real turns.
        if m.get("isMeta"):
            continue

        if _is_tool_result(m):
            if cur is None:
                continue
            for x in _content(m):
                if isinstance(x, dict) and x.get("type") == "tool_result":
                    tid = str(x.get("tool_use_id", ""))
                    cur.tool_results[tid] = {
                        "content": x.get("content"),
                        "timestamp": m.get("timestamp"),
                        # Claude Code sets is_error=true on a tool_result when the tool
                        # failed (failed Bash, missing file, etc.). Surfaced as span status.
                        "is_error": bool(x.get("is_error")),
                    }
            continue

        role = _role(m)

        if role == "user":
            # Start a fresh turn; previous turn (if any) stays in the list.
            cur = Turn(user_msg=m)
            turns.append(cur)

        elif role == "assistant" and cur is not None:
            # Append every frame — llm_calls groups by requestId, so multiple
            # frames per response still produce exactly one LlmCall.
            cur.assistant_msgs.append(m)

    # Drop turns that have no assistant responses (incomplete / trailing user msg).
    return [t for t in turns if t.assistant_msgs]


def _user_text(turn: Turn) -> str:
    """Extract the plain text of a turn's user message (for is_task_notification check)."""
    content = (turn.user_msg.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            x.get("text", "")
            for x in content
            if isinstance(x, dict) and x.get("type") == "text"
        )
    return ""


def is_task_notification(turn: Turn) -> bool:
    """Return True if this turn is a task-notification continuation (not a human query)."""
    return _user_text(turn).strip().startswith("<task-notification>")


def _notif_task_id(turn: Turn) -> str | None:
    """Extract the task-id from a task-notification turn's user text, or None."""
    text = _user_text(turn)
    m = re.search(r"<task-id>([^<]+)</task-id>", text)
    return m.group(1).strip() if m else None


def _meta_tool_use_id(task_id: str, subagents_dir: str) -> str | None:
    """
    Read agent-<task_id>.meta.json from subagents_dir and return its toolUseId.
    Fail-open: returns None on any error (missing file, bad JSON, missing key).
    """
    meta_path = os.path.join(subagents_dir, f"agent-{task_id}.meta.json")
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
        return meta.get("toolUseId") or None
    except Exception:
        return None


def group_turns_into_asks(turns: list, subagents_dir: str | None = None) -> list:
    """
    Group turns into asks (a human query + its task-notification continuations).

    When subagents_dir is provided, each task-notification is id-matched to the ask
    that launched its sub-agent: the notification's <task-id> is resolved via
    agent-<task_id>.meta.json → toolUseId, then matched to the ask whose turns
    issued that toolUseId in a Task/Agent tool_use block.

    Falls back to positional attachment (append to the ask that was open when the
    notification was encountered) when subagents_dir is None, the meta.json is
    missing, or no matching ask is found.

    Returns a list of lists (each inner list is one ask). Notifications are appended
    in transcript order so each ask's turns stay chronological.
    Edge: a leading task-notification (no prior human turn) starts its own ask.
    """
    # --- Pass 1: identify human-turn asks and collect notifications with fallbacks ---
    asks: list[list] = []
    # For each notification, record (turn, fallback_ask) where fallback_ask is the
    # ask that was "current" at the time the notification was encountered.
    notif_queue: list[tuple] = []  # (turn, fallback_ask_or_None)

    current: list | None = None
    for t in turns:
        if is_task_notification(t):
            notif_queue.append((t, current))
        else:
            current = [t]
            asks.append(current)

    if not notif_queue:
        return asks

    # --- Build tool_use_id -> ask index (from Task/Agent tool_use blocks) ---
    tool_use_id_to_ask: dict[str, list] = {}
    for ask in asks:
        for turn in ask:
            for msg in turn.assistant_msgs:
                content = (msg.get("message") or {}).get("content") or []
                if isinstance(content, list):
                    for block in content:
                        if (isinstance(block, dict)
                                and block.get("type") == "tool_use"
                                and block.get("name") in ("Task", "Agent")):
                            tid = block.get("id")
                            if tid:
                                tool_use_id_to_ask[tid] = ask

    # --- Pass 2: attach each notification to the correct ask ---
    for notif_turn, fallback_ask in notif_queue:
        target = None

        if subagents_dir:
            task_id = _notif_task_id(notif_turn)
            if task_id:
                tool_use_id = _meta_tool_use_id(task_id, subagents_dir)
                if tool_use_id:
                    target = tool_use_id_to_ask.get(tool_use_id)

        if target is None:
            # Positional fallback: use the ask that was open at this position
            target = fallback_ask

        if target is None:
            # Leading notification with no prior ask: start its own ask
            target = []
            asks.append(target)

        target.append(notif_turn)

    return asks
